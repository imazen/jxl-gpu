// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Raw cudarc HtoD bandwidth microbench — bypasses cubecl entirely.
//!
//! Production cubecl 0.10 paths showed pathological per-byte cost at large
//! transfers (4 MB → 1.26 GB/s, 192 MB → 0.13 GB/s — 13–190× below the
//! PCIe 4.0 x16 practical ceiling of ~16-25 GB/s). This bench answers the
//! question: is the GPU/host/PCIe link itself slow, or is cubecl losing the
//! bandwidth somewhere above the driver?
//!
//! For each of {4 MB, 48 MB, 192 MB}:
//!
//!   - cudaMalloc-only timing (allocate, sync, free).
//!   - Path B: pageable `Vec<u8>` → device via `cuMemcpyHtoD_v2` (sync).
//!   - Path C: pinned host (`cuMemHostAlloc` WRITECOMBINED) → device via
//!     `cuMemcpyHtoDAsync_v2` + `cuStreamSynchronize`.
//!
//! 1 warmup + 5 timed iterations per cell; min/median/mean reported.
//!
//! Run:
//!   cargo run --release -p jxl-encoder-gpu --features 'cuda encoder' \
//!     --example perf_raw_cuda_upload

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use cudarc::driver::CudaContext;
    use cudarc::driver::result;

    // Sizes we actually care about in jxl-encoder-gpu.
    // Names match the cubecl-side numbers we're benchmarking against.
    const SIZES: &[(usize, &str)] = &[
        (4 * 1024 * 1024, "4 MB (1 MP f32 plane)"),
        (48 * 1024 * 1024, "48 MB (16 MP RGB u8 interleaved)"),
        (192 * 1024 * 1024, "192 MB (16 MP × 3 padded f32 planes)"),
    ];
    const RUNS: usize = 5;

    println!("perf_raw_cuda_upload: cudarc 0.19, RTX-class GPU");
    println!("warmup=1, runs={}, all sizes in bytes\n", RUNS);

    // Init CUDA + grab a stream.
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    println!(
        "device 0: {}  (cap {:?}, has_async_alloc={})",
        ctx.name().unwrap_or_else(|_| "<unknown>".into()),
        ctx.compute_capability().unwrap_or((0, 0)),
        ctx.has_async_alloc(),
    );
    let stream = ctx.default_stream();

    // ---- Allocation-only timing ----
    println!("\n=== cudaMalloc + cudaFree (no copy) ===");
    println!(
        "{:32}  {:>10}  {:>10}  {:>10}",
        "size", "min ms", "median ms", "mean ms"
    );
    for &(bytes, label) in SIZES {
        // Warmup.
        unsafe {
            let p = result::malloc_sync(bytes).expect("malloc warmup");
            ctx.synchronize().unwrap();
            result::free_sync(p).unwrap();
        }
        let mut times = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t0 = Instant::now();
            let p = unsafe { result::malloc_sync(bytes).expect("malloc") };
            // sync to make sure any deferred work completes (sync allocator
            // is normally truly synchronous, but be explicit).
            ctx.synchronize().unwrap();
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            unsafe { result::free_sync(p).unwrap() };
            times.push(elapsed);
        }
        report(label, &times, bytes);
    }

    // ---- Path B: pageable Vec<u8> → device (sync HtoD) ----
    println!("\n=== Path B: pageable Vec<u8> → device (cuMemcpyHtoD_v2 sync) ===");
    println!(
        "{:32}  {:>10}  {:>10}  {:>10}  {:>10}",
        "size", "min ms", "median ms", "mean ms", "GB/s (min)"
    );
    for &(bytes, label) in SIZES {
        // Pre-fill so the page-faulting cost happens BEFORE timing
        // (touching every page also primes the kernel-side bounce buffer
        // path the driver will use for pageable copies).
        let host: Vec<u8> = vec![0xA5u8; bytes];
        // Warmup malloc + copy.
        let dptr = unsafe { result::malloc_sync(bytes).expect("malloc warmup") };
        unsafe { result::memcpy_htod_sync(dptr, &host[..]).expect("warmup htod") };
        ctx.synchronize().unwrap();

        let mut times = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t0 = Instant::now();
            unsafe { result::memcpy_htod_sync(dptr, &host[..]).expect("htod") };
            // memcpy_htod_sync is the truly-synchronous CUDA driver call;
            // it returns only when the copy is complete. No extra sync.
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        unsafe { result::free_sync(dptr).unwrap() };
        report_bw(label, &times, bytes);
    }

    // ---- Path C: pinned host → device (async HtoD + stream sync) ----
    println!("\n=== Path C: pinned host → device (cuMemHostAlloc + cuMemcpyHtoDAsync) ===");
    println!(
        "{:32}  {:>10}  {:>10}  {:>10}  {:>10}",
        "size", "min ms", "median ms", "mean ms", "GB/s (min)"
    );
    for &(bytes, label) in SIZES {
        // Allocate pinned host buffer (WRITECOMBINED — fast device-side
        // reads, fast sequential CPU writes, slow CPU reads. We never
        // read from it after fill, so this is the right choice).
        let mut pinned = unsafe {
            ctx.alloc_pinned::<u8>(bytes)
                .expect("alloc_pinned (pinned host)")
        };
        // Fill the pinned buffer ONCE outside the timing loop.
        // Sequential writes to write-combining memory are fast.
        {
            let slice = pinned.as_mut_slice().expect("as_mut_slice");
            // Use copy_from_slice from a regular vec to fill — equivalent
            // perf to a fill loop, lets us not measure write-combining
            // write speed (which is not what we're benchmarking).
            let src = vec![0xA5u8; bytes];
            slice.copy_from_slice(&src);
        }
        let dptr = unsafe { result::malloc_sync(bytes).expect("malloc") };

        // Warmup async copy.
        unsafe {
            // We have a *const T pointer to pinned storage; build a slice.
            let host_ptr = pinned.as_ptr().expect("pinned as_ptr");
            let host_slice = std::slice::from_raw_parts(host_ptr, bytes);
            result::memcpy_htod_async(dptr, host_slice, stream.cu_stream()).expect("htod async");
        }
        stream.synchronize().expect("warmup sync");

        let mut times = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t0 = Instant::now();
            unsafe {
                let host_ptr = pinned.as_ptr().expect("pinned as_ptr");
                let host_slice = std::slice::from_raw_parts(host_ptr, bytes);
                result::memcpy_htod_async(dptr, host_slice, stream.cu_stream())
                    .expect("htod async");
            }
            stream.synchronize().expect("stream sync");
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        unsafe { result::free_sync(dptr).unwrap() };
        report_bw(label, &times, bytes);
        // pinned dropped here, frees host pinned memory.
    }

    // ---- Comparison footer ----
    println!("\n=== Cubecl-side reference (already measured, separate run) ===");
    println!(
        "  {:32}  {:>10}  {:>14}  {:>14}",
        "size", "median ms", "GB/s (effective)", "vs PCIe ~16 GB/s"
    );
    println!(
        "  {:32}  {:>10}  {:>14}  {:>14}",
        "4 MB f32 plane (cubecl)", "3.18", "1.26", "0.08×"
    );
    println!(
        "  {:32}  {:>10}  {:>14}  {:>14}",
        "48 MB u8 RGB (cubecl)", "298", "0.16", "0.01×"
    );
    println!(
        "  {:32}  {:>10}  {:>14}  {:>14}",
        "192 MB f32 padded (cubecl)", "1423", "0.13", "0.008×"
    );
    println!("\nDeltas above show how much bandwidth cubecl 0.10's create_from_slice");
    println!("is leaving on the table relative to raw cuMemcpyHtoD.");
}

#[cfg(feature = "cuda")]
fn report(label: &str, times: &[f64], _bytes: usize) {
    let mut s: Vec<f64> = times.iter().copied().collect();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let min = s[0];
    let med = s[s.len() / 2];
    let mean: f64 = s.iter().sum::<f64>() / s.len() as f64;
    println!(
        "  {:32}  {:>10.3}  {:>10.3}  {:>10.3}",
        label, min, med, mean
    );
}

#[cfg(feature = "cuda")]
fn report_bw(label: &str, times: &[f64], bytes: usize) {
    let mut s: Vec<f64> = times.iter().copied().collect();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let min = s[0];
    let med = s[s.len() / 2];
    let mean: f64 = s.iter().sum::<f64>() / s.len() as f64;
    // GB/s = bytes / (ms / 1000) / 1e9 = bytes / (ms * 1e6)
    let gbps_min = (bytes as f64) / (min * 1e6);
    println!(
        "  {:32}  {:>10.3}  {:>10.3}  {:>10.3}  {:>10.3}",
        label, min, med, mean, gbps_min
    );
}
