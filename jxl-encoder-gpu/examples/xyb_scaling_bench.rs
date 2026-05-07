//! GPU vs CPU XYB scaling across sizes 64² → 4096².
//!
//! Sweeps a synthetic deterministic input across 7 sizes (64, 128,
//! 256, 512, 1024, 2048, 4096), measures both the dispatched-SIMD CPU
//! XYB and the GPU XYB (incl. upload+download), and prints the
//! crossover point where GPU starts winning.
//!
//! Use this to see where GPU XYB pays off versus where the launch +
//! memcpy overhead dominates. Numbers from RTX 5070 + Ryzen 9 7950X
//! were ~3× CPU win at 1MP; this bench shows where (if at all) that
//! flips.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let sizes: Vec<usize> = std::env::var("SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![64, 128, 256, 512, 1024, 2048, 4096]);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7); // 1 warmup + 6 sampled, take median

    println!("=== XYB scaling: GPU vs CPU ===");
    println!(
        "Iters per size: {iters} (1 warmup + {} sampled)\n",
        iters - 1
    );
    println!(
        "{:>6}  {:>9}  {:>10}  {:>10}  {:>10}  {:>8}  {:>10}",
        "side", "MP", "CPU ms", "CPU MP/s", "GPU ms", "GPU MP/s", "ratio"
    );
    println!("{}", "─".repeat(70));

    for &side in &sizes {
        let n = side * side;
        let mp = n as f64 / 1e6;

        // Synthetic deterministic input.
        let r: Vec<f32> = (0..n)
            .map(|i| 0.1 + 0.6 * ((i.wrapping_mul(2654435761) & 0xFFFF) as f32 / 65535.0))
            .collect();
        let g: Vec<f32> = (0..n)
            .map(|i| 0.2 + 0.5 * ((i.wrapping_mul(0x9E3779B9) & 0xFFFF) as f32 / 65535.0))
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| 0.3 + 0.4 * ((i.wrapping_mul(0xBF58476D) & 0xFFFF) as f32 / 65535.0))
            .collect();

        // CPU
        let mut cpu_x = vec![0.0_f32; n];
        let mut cpu_y = vec![0.0_f32; n];
        let mut cpu_b = vec![0.0_f32; n];
        let mut cpu_times = Vec::with_capacity(iters);
        for i in 0..iters {
            let t = std::time::Instant::now();
            jxl_encoder_simd::linear_rgb_to_xyb_batch(
                &r, &g, &b, &mut cpu_x, &mut cpu_y, &mut cpu_b,
            );
            let dt = t.elapsed();
            if i > 0 {
                cpu_times.push(dt);
            }
        }
        cpu_times.sort();
        let cpu_med = cpu_times[cpu_times.len() / 2];
        let cpu_ms = cpu_med.as_secs_f64() * 1000.0;
        let cpu_mps = mp / cpu_med.as_secs_f64();

        // GPU
        let mut gpu_times = Vec::with_capacity(iters);
        for i in 0..iters {
            let t = std::time::Instant::now();
            let _ = enc.xyb_from_linear_rgb(&r, &g, &b);
            let dt = t.elapsed();
            if i > 0 {
                gpu_times.push(dt);
            }
        }
        gpu_times.sort();
        let gpu_med = gpu_times[gpu_times.len() / 2];
        let gpu_ms = gpu_med.as_secs_f64() * 1000.0;
        let gpu_mps = mp / gpu_med.as_secs_f64();

        let ratio = cpu_med.as_secs_f64() / gpu_med.as_secs_f64();
        let marker = if ratio > 1.0 { "GPU>" } else { "<CPU" };
        println!(
            "{:>6}  {:>9.3}  {:>10.3}  {:>10.0}  {:>10.3}  {:>8.0}  {:>5.2}× {marker}",
            side, mp, cpu_ms, cpu_mps, gpu_ms, gpu_mps, ratio
        );
    }
    println!("\n  ratio = CPU_time / GPU_time. >1.0 = GPU faster. \"GPU>\" marker shows GPU wins.");
    println!("  GPU column includes upload+download per call (worst case for round-trip API).");
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
