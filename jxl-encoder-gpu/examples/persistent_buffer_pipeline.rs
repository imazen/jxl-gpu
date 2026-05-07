//! Persistent-buffer GPU pipeline benchmark.
//!
//! Compares 3 paths for a real encoder front-end pipeline at sizes
//! 256² → 4096²:
//!
//! 1. **CPU AVX2** — `jxl_encoder_simd` dispatched SIMD (one-shot
//!    sequential, single thread).
//! 2. **GPU round-trip** — the default `GpuEncoder` facade methods,
//!    each of which uploads inputs + downloads outputs every call.
//! 3. **GPU persistent (typed API)** — the new `crate::persistent`
//!    module: upload R/G/B once via `upload_plane`, chain pipeline
//!    stages through `*_persistent` methods (returning typed
//!    `GpuPlane<R>`s that stay on-GPU), download only the final
//!    mask via `download_plane`.
//!
//! Pipeline (mirrors the encoder front-end):
//!   linear-RGB → XYB (3 channels) → gaborish_5x5 (3 channels) →
//!   mask1x1 (Y only)
//!
//! Earlier hand-coded `launch::*`-chain version of (3) showed
//! 1.7-3.4× speedup over (2). The typed API should match that
//! speedup with much cleaner code.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::persistent::GaborishWeights;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let sizes: Vec<usize> = std::env::var("SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![256, 512, 1024, 2048, 4096]);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    // Gaborish weights for mul=1.0 (matches forks::gaborish::compute_weights).
    const K_GABORISH: [f64; 5] = [
        -0.094_958_15_67,
        -0.041_031_725,
        0.013_710_005,
        0.006_510_206,
        -0.001_478_906_3,
    ];
    let sum_w = 1.0
        + 4.0
            * (K_GABORISH[0] + K_GABORISH[1] + K_GABORISH[2] + K_GABORISH[4] + 2.0 * K_GABORISH[3]);
    let norm = 1.0 / sum_w;
    let weights_scalar = (
        norm as f32,
        (norm * K_GABORISH[0]) as f32,
        (norm * K_GABORISH[1]) as f32,
        (norm * K_GABORISH[2]) as f32,
        (norm * K_GABORISH[3]) as f32,
        (norm * K_GABORISH[4]) as f32,
    );
    let (wc, wr, wd, w_big_r, wl, w_big_d) = weights_scalar;
    let weights = GaborishWeights {
        wc,
        wr,
        wd,
        w_big_r,
        wl,
        w_big_d,
    };

    println!("=== persistent-buffer pipeline: XYB + gaborish + mask1x1 ===");
    println!(
        "Iters per size: {iters} (1 warmup + {} sampled)\n",
        iters - 1
    );
    println!(
        "{:>6}  {:>9}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
        "side", "MP", "CPU ms", "GPU rt ms", "GPU pers ms", "vs CPU", "vs rt"
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

        // ── CPU baseline ───────────────────────────────────────────
        let mut cx = vec![0.0_f32; n];
        let mut cy = vec![0.0_f32; n];
        let mut cb = vec![0.0_f32; n];
        let mut sx = vec![0.0_f32; n];
        let mut sy = vec![0.0_f32; n];
        let mut sb = vec![0.0_f32; n];
        let mut mask = vec![0.0_f32; n];
        let mut cpu_times = Vec::with_capacity(iters);
        for i in 0..iters {
            let t = std::time::Instant::now();
            jxl_encoder_simd::linear_rgb_to_xyb_batch(&r, &g, &b, &mut cx, &mut cy, &mut cb);
            jxl_encoder_simd::gaborish_5x5_channel(
                &mut cx, &mut sx, side, side, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            jxl_encoder_simd::gaborish_5x5_channel(
                &mut cy, &mut sy, side, side, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            jxl_encoder_simd::gaborish_5x5_channel(
                &mut cb, &mut sb, side, side, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            jxl_encoder_simd::compute_mask1x1(&cy, side, side, &mut mask);
            let dt = t.elapsed();
            if i > 0 {
                cpu_times.push(dt);
            }
        }
        cpu_times.sort();
        let cpu_med = cpu_times[cpu_times.len() / 2];

        // ── GPU round-trip (via GpuEncoder facade) ─────────────────
        let mut rt_times = Vec::with_capacity(iters);
        for i in 0..iters {
            let t = std::time::Instant::now();
            let (xyb_x, xyb_y, xyb_b) = enc.xyb_from_linear_rgb(&r, &g, &b);
            let _gx = enc.gaborish_5x5_channel(
                &xyb_x,
                side as u32,
                side as u32,
                wc,
                wr,
                wd,
                w_big_r,
                wl,
                w_big_d,
            );
            let gy = enc.gaborish_5x5_channel(
                &xyb_y,
                side as u32,
                side as u32,
                wc,
                wr,
                wd,
                w_big_r,
                wl,
                w_big_d,
            );
            let _gb = enc.gaborish_5x5_channel(
                &xyb_b,
                side as u32,
                side as u32,
                wc,
                wr,
                wd,
                w_big_r,
                wl,
                w_big_d,
            );
            let _ = enc.mask1x1_field(&gy, side as u32, side as u32);
            let dt = t.elapsed();
            if i > 0 {
                rt_times.push(dt);
            }
        }
        rt_times.sort();
        let rt_med = rt_times[rt_times.len() / 2];

        // ── GPU persistent (typed API; data stays on GPU) ──────────
        let mut pers_times = Vec::with_capacity(iters);
        for i in 0..iters {
            let t = std::time::Instant::now();
            // Upload R, G, B once.
            let g_r = enc.upload_plane(&r, side as u32, side as u32);
            let g_g = enc.upload_plane(&g, side as u32, side as u32);
            let g_b = enc.upload_plane(&b, side as u32, side as u32);
            // Pipeline — all intermediates stay on-GPU.
            let (xx, xy, xbo) = enc.xyb_from_linear_rgb_persistent(&g_r, &g_g, &g_b);
            let _xx_g = enc.gaborish_5x5_persistent(&xx, &weights);
            let xy_g = enc.gaborish_5x5_persistent(&xy, &weights);
            let _xb_g = enc.gaborish_5x5_persistent(&xbo, &weights);
            let mask = enc.mask1x1_persistent(&xy_g);
            // Download only the final mask.
            let _ = enc.download_plane(&mask);
            let dt = t.elapsed();
            if i > 0 {
                pers_times.push(dt);
            }
        }
        pers_times.sort();
        let pers_med = pers_times[pers_times.len() / 2];

        let cpu_ms = cpu_med.as_secs_f64() * 1000.0;
        let rt_ms = rt_med.as_secs_f64() * 1000.0;
        let pers_ms = pers_med.as_secs_f64() * 1000.0;
        let vs_cpu = cpu_med.as_secs_f64() / pers_med.as_secs_f64();
        let vs_rt = rt_med.as_secs_f64() / pers_med.as_secs_f64();
        println!(
            "{:>6}  {:>9.3}  {:>10.3}  {:>10.3}  {:>10.3}  {:>6.2}×  {:>6.2}×",
            side, mp, cpu_ms, rt_ms, pers_ms, vs_cpu, vs_rt
        );
    }
    println!(
        "\n  vs CPU: persistent-GPU vs CPU AVX2 (>1.0 = GPU wins)\n  vs rt:  persistent-GPU vs round-trip-GPU API (always >1)"
    );
    println!(
        "\n  Persistent path uses the typed API in `crate::persistent`:\n  upload_plane → xyb_from_linear_rgb_persistent →\n  gaborish_5x5_persistent (×3) → mask1x1_persistent → download_plane."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
