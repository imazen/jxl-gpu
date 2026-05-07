//! Pipeline throughput when encoding the SAME image at multiple
//! settings (e.g., a quality sweep). Demonstrates amortizing the
//! input upload + persistent intermediate buffers across iterations
//! — the realistic batch-workload use case.
//!
//! Two paths compared at 1024² × 5 distances:
//!
//! 1. **Per-iter upload**: each iter calls `upload_plane(R/G/B)`,
//!    runs full pipeline, downloads RGB. Mirrors what an encoder
//!    library would naively do.
//!
//! 2. **Pre-uploaded input**: input planes uploaded ONCE; all 5
//!    iterations re-use those handles. Pipeline still allocates +
//!    releases intermediate buffers per iter (cubecl pools recycle
//!    the GPU memory under the hood).
//!
//! Compares to single CPU encode time at the same settings × 5.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::persistent::GaborishWeights;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let side: usize = std::env::var("SIDE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let n_settings: usize = std::env::var("SETTINGS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    let n = side * side;
    let nb = (side / 8) * (side / 8);

    println!("=== repeated-input throughput @ {side}×{side} × {n_settings} settings ===");
    println!("Iters: {iters} (1 warmup + {} sampled)\n", iters - 1);

    let mut r_plane = Vec::with_capacity(n);
    let mut g_plane = Vec::with_capacity(n);
    let mut b_plane = Vec::with_capacity(n);
    for y in 0..side {
        for x in 0..side {
            r_plane.push(0.1 + 0.7 * (x as f32 / side as f32));
            g_plane.push(0.2 + 0.6 * (y as f32 / side as f32));
            b_plane.push(0.3 + 0.5 * (((x + y) % 17) as f32 / 17.0));
        }
    }

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
    let weights = GaborishWeights {
        wc: norm as f32,
        wr: (norm * K_GABORISH[0]) as f32,
        wd: (norm * K_GABORISH[1]) as f32,
        w_big_r: (norm * K_GABORISH[2]) as f32,
        wl: (norm * K_GABORISH[3]) as f32,
        w_big_d: (norm * K_GABORISH[4]) as f32,
    };
    let weights_g = enc.upload_blocks(&vec![1.0_f32; nb * 64], nb as u32, 64);
    // 5 different qac settings simulating a quality sweep (lower = higher
    // quality / less aggressive quant).
    let qac_settings: Vec<f32> = (0..n_settings).map(|i| 1.0 + (i as f32) * 0.75).collect();
    let xf = vec![0.0_f32; nb];
    let bf = vec![0.0_f32; nb];
    let thr = [0.56_f32, 0.62, 0.62, 0.62];

    let run_pipeline = |g_r: &jxl_encoder_gpu::persistent::GpuPlane<Backend>,
                        g_g: &jxl_encoder_gpu::persistent::GpuPlane<Backend>,
                        g_b: &jxl_encoder_gpu::persistent::GpuPlane<Backend>,
                        qac_qm: f32| {
        let qac_vec = vec![qac_qm; nb];
        let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(g_r, g_g, g_b);
        let xx_g = enc.gaborish_5x5_persistent(&xx, &weights);
        let xy_g = enc.gaborish_5x5_persistent(&xy, &weights);
        let xb_g = enc.gaborish_5x5_persistent(&xb, &weights);
        let bx_g = enc.gather_blocks_persistent(&xx_g, 8, 8);
        let by_g = enc.gather_blocks_persistent(&xy_g, 8, 8);
        let bb_g = enc.gather_blocks_persistent(&xb_g, 8, 8);
        let coeffs_x = enc.dct_8x8_wide_persistent(&bx_g);
        let coeffs_y = enc.dct_8x8_wide_persistent(&by_g);
        let coeffs_b = enc.dct_8x8_wide_persistent(&bb_g);
        let q_x = enc.quantize_dct8_persistent(&coeffs_x, &weights_g, &qac_vec, &thr);
        let q_y = enc.quantize_dct8_persistent(&coeffs_y, &weights_g, &qac_vec, &thr);
        let q_b = enc.quantize_dct8_persistent(&coeffs_b, &weights_g, &qac_vec, &thr);
        let (dq_x, dq_y, dq_b) = enc.dequant_dct8_persistent(
            &q_x, &q_y, &q_b, &weights_g, &weights_g, &weights_g, &qac_vec, &qac_vec, &qac_vec,
            &xf, &bf,
        );
        enc.restore_dc_persistent(&coeffs_x, &dq_x);
        enc.restore_dc_persistent(&coeffs_y, &dq_y);
        enc.restore_dc_persistent(&coeffs_b, &dq_b);
        let recon_x_b = enc.idct_8x8_wide_persistent(&dq_x);
        let recon_y_b = enc.idct_8x8_wide_persistent(&dq_y);
        let recon_b_b = enc.idct_8x8_wide_persistent(&dq_b);
        let recon_x_p = enc.scatter_blocks_persistent(&recon_x_b, side as u32, side as u32, 8, 8);
        let recon_y_p = enc.scatter_blocks_persistent(&recon_y_b, side as u32, side as u32, 8, 8);
        let recon_b_p = enc.scatter_blocks_persistent(&recon_b_b, side as u32, side as u32, 8, 8);
        let (rgb_r, rgb_g, rgb_b) =
            enc.xyb_to_linear_rgb_planar_persistent(&recon_x_p, &recon_y_p, &recon_b_p);
        let _ = enc.download_plane(&rgb_r);
        let _ = enc.download_plane(&rgb_g);
        let _ = enc.download_plane(&rgb_b);
    };

    // ── Per-iter upload (naive) ────────────────────────────────────
    let mut naive_times = Vec::with_capacity(iters);
    for it in 0..iters {
        let t = std::time::Instant::now();
        for &qac in &qac_settings {
            let g_r = enc.upload_plane(&r_plane, side as u32, side as u32);
            let g_g = enc.upload_plane(&g_plane, side as u32, side as u32);
            let g_b = enc.upload_plane(&b_plane, side as u32, side as u32);
            run_pipeline(&g_r, &g_g, &g_b, qac);
        }
        let dt = t.elapsed();
        if it > 0 {
            naive_times.push(dt);
        }
    }
    naive_times.sort();
    let naive_med = naive_times[naive_times.len() / 2];

    // ── Pre-uploaded input ─────────────────────────────────────────
    let mut reuse_times = Vec::with_capacity(iters);
    for it in 0..iters {
        let g_r = enc.upload_plane(&r_plane, side as u32, side as u32);
        let g_g = enc.upload_plane(&g_plane, side as u32, side as u32);
        let g_b = enc.upload_plane(&b_plane, side as u32, side as u32);
        let t = std::time::Instant::now();
        for &qac in &qac_settings {
            run_pipeline(&g_r, &g_g, &g_b, qac);
        }
        let dt = t.elapsed();
        if it > 0 {
            reuse_times.push(dt);
        }
    }
    reuse_times.sort();
    let reuse_med = reuse_times[reuse_times.len() / 2];

    let naive_ms = naive_med.as_secs_f64() * 1000.0;
    let reuse_ms = reuse_med.as_secs_f64() * 1000.0;
    let speedup = naive_med.as_secs_f64() / reuse_med.as_secs_f64();
    let mp_per_set = (n as f64 / 1e6) * (n_settings as f64);
    println!(
        "{:>30}  {:>12}  {:>12}",
        "scenario", "total ms", "MP/s aggregate"
    );
    println!("{}", "─".repeat(60));
    println!(
        "{:>30}  {:>12.2}  {:>12.0}",
        "per-iter upload (naive)",
        naive_ms,
        mp_per_set / naive_med.as_secs_f64()
    );
    println!(
        "{:>30}  {:>12.2}  {:>12.0}  ({:.2}× faster)",
        "pre-uploaded input",
        reuse_ms,
        mp_per_set / reuse_med.as_secs_f64(),
        speedup
    );
    println!(
        "\n  Pipeline run {n_settings}× per timed iter, processing\n  {:.2} aggregate MP per iter (5 quality settings on the same image).",
        mp_per_set
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
