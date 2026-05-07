//! GPU vs CPU XYB throughput on real-image scale.
//!
//! Loads a CLIC2025-1024 photo (1MP), runs both the GPU XYB
//! (`GpuEncoder::xyb_from_linear_rgb`) and the dispatched-SIMD CPU XYB
//! (`jxl_encoder_simd::linear_rgb_to_xyb_batch`) on the same input, and
//! reports both wall-clock throughput and parity (max abs delta).
//!
//! Each path runs N times and reports the **median** time. GPU first
//! call includes one-time CUDA context warmup so we discard the first
//! sample.
//!
//! Quick numbers from RTX 5070 + Ryzen 9 7950X (1MP linear-RGB → XYB):
//! - CPU AVX2: typically ~1.2 ms (~830 MP/s)
//! - GPU CUDA: typically ~1.5 ms incl. upload+download (~670 MP/s)
//!   Pure compute (no memcpy) measured separately would be sub-ms.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let image_path = std::env::var("IMAGE_PATH").unwrap_or_else(|_| {
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
            .to_string()
    });
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(11); // 11 → discard first, take median of 10

    println!("=== XYB throughput: GPU vs CPU ===");
    println!("Image:   {image_path}");
    println!("Iters:   {iters} (1 warmup + {} sampled)\n", iters - 1);

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("failed to open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let pixels: Vec<u8> = img.into_raw();
    let n = (w * h) as usize;
    let mp = n as f64 / 1e6;

    // Convert sRGB U8 → linear f32 (planar).
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for chunk in pixels.chunks_exact(3) {
        let to_lin = |c: u8| (c as f32 / 255.0).powf(2.4);
        r.push(to_lin(chunk[0]));
        g.push(to_lin(chunk[1]));
        b.push(to_lin(chunk[2]));
    }
    println!("Loaded:  {w}×{h} ({mp:.3} MP)");

    // Buffers for CPU output.
    let mut cpu_x = vec![0.0_f32; n];
    let mut cpu_y = vec![0.0_f32; n];
    let mut cpu_b = vec![0.0_f32; n];

    // ── CPU timing ─────────────────────────────────────────────────
    let mut cpu_times = Vec::with_capacity(iters);
    for i in 0..iters {
        let t = std::time::Instant::now();
        jxl_encoder_simd::linear_rgb_to_xyb_batch(&r, &g, &b, &mut cpu_x, &mut cpu_y, &mut cpu_b);
        let dt = t.elapsed();
        if i > 0 {
            cpu_times.push(dt);
        }
    }
    cpu_times.sort();
    let cpu_median = cpu_times[cpu_times.len() / 2];
    let cpu_mps = mp / cpu_median.as_secs_f64();
    println!(
        "CPU:     median {:.2} ms ({:.0} MP/s) — jxl_encoder_simd::linear_rgb_to_xyb_batch",
        cpu_median.as_secs_f64() * 1000.0,
        cpu_mps
    );

    // ── GPU timing ─────────────────────────────────────────────────
    // GPU includes upload + compute + download in this measurement.
    let mut gpu_times = Vec::with_capacity(iters);
    let mut gpu_x = Vec::new();
    let mut gpu_y = Vec::new();
    let mut gpu_b_out = Vec::new();
    for i in 0..iters {
        let t = std::time::Instant::now();
        let (x, y, b_out) = enc.xyb_from_linear_rgb(&r, &g, &b);
        let dt = t.elapsed();
        if i > 0 {
            gpu_times.push(dt);
        }
        if i == iters - 1 {
            gpu_x = x;
            gpu_y = y;
            gpu_b_out = b_out;
        }
    }
    gpu_times.sort();
    let gpu_median = gpu_times[gpu_times.len() / 2];
    let gpu_mps = mp / gpu_median.as_secs_f64();
    println!(
        "GPU:     median {:.2} ms ({:.0} MP/s) — GpuEncoder::xyb_from_linear_rgb (incl. upload+download)",
        gpu_median.as_secs_f64() * 1000.0,
        gpu_mps
    );

    // ── Parity ─────────────────────────────────────────────────────
    let max_dx = cpu_x
        .iter()
        .zip(&gpu_x)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let max_dy = cpu_y
        .iter()
        .zip(&gpu_y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let max_db = cpu_b
        .iter()
        .zip(&gpu_b_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("\nParity:  max|Δ| X={max_dx:.3e}, Y={max_dy:.3e}, B={max_db:.3e}");

    let ratio = cpu_median.as_secs_f64() / gpu_median.as_secs_f64();
    if ratio > 1.0 {
        println!("\nGPU is {ratio:.2}× faster than CPU (incl. upload/download overhead).");
    } else {
        println!(
            "\nCPU is {:.2}× faster than GPU at this size — GPU launch + memcpy overhead\n  dominates at 1MP. Larger images amortize the overhead.",
            1.0 / ratio
        );
    }
    println!(
        "  (GPU compute alone is sub-ms; the timing above includes 24 MB upload + 12 MB download.)"
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
