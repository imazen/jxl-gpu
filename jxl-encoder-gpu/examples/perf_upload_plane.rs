// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Microbench: per-call cost of `upload_plane` for a 1024×1024 f32 plane.
//!
//! Measures 50 sequential uploads after a warmup. Reports each iter
//! time so we can see whether the cubecl allocator pool warms up
//! (first slow, rest fast) or stays uniformly slow (per-call cudaMalloc
//! / cudaMemcpy overhead dominates).
//!
//! Used to inform the pad_upload optimization: if upload-cost is
//! ~uniform, pre-allocating the 3 input planes once doesn't help
//! because each upload still does its own cuMemcpyHtoD + sync.

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
}

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use std::time::Instant;
    use jxl_encoder_gpu::encoder::GpuEncoder;

    type B = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<B> = GpuEncoder::new();

    let w = 1024_u32;
    let h = 1024_u32;
    let n = (w as usize) * (h as usize);
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-6).collect();

    // Warmup: a few uploads to prime the cubecl allocator pool.
    for _ in 0..3 {
        let _ = enc.upload_plane(&data, w, h);
    }

    println!("perf_upload_plane: 50 sequential 4 MB uploads (1024² f32 plane)");
    println!("  iter   ms     MB/s");
    let mut times: Vec<f64> = Vec::with_capacity(50);
    for i in 0..50 {
        let t0 = Instant::now();
        let h_plane = enc.upload_plane(&data, w, h);
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        times.push(dt_ms);
        let mb_per_s = 4.0 / (dt_ms / 1000.0);
        println!("  {:4}  {:5.2}  {:7.1}", i, dt_ms, mb_per_s);
        // Drop the handle to let the pool reclaim. (May or may not
        // matter depending on cubecl pool policy.)
        drop(h_plane);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times[0];
    let median = times[25];
    let max = times[49];
    let mean = times.iter().sum::<f64>() / (times.len() as f64);
    let p25 = times[12];
    let p75 = times[37];

    println!();
    println!("Summary (50 iters, 4 MB each):");
    println!("  min     = {:5.2} ms ({:.1} MB/s)", min, 4.0 / (min / 1000.0));
    println!("  p25     = {:5.2} ms", p25);
    println!("  median  = {:5.2} ms ({:.1} MB/s)", median, 4.0 / (median / 1000.0));
    println!("  p75     = {:5.2} ms", p75);
    println!("  max     = {:5.2} ms", max);
    println!("  mean    = {:5.2} ms", mean);
}
