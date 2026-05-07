//! Fused-kernel-only lossy pipeline throughput.
//!
//! Compares CPU vs full-GPU lossy pipeline using the fused
//! DCT8+quantize and fused dequant+IDCT-Y kernels in place of the
//! separate-stage chain. Skips DC restore — the DC slot is zero in
//! the reconstruction (matches what a real encoder would do; DC has
//! its own quant + entropy coding via dc_coding).
//!
//! This is the "production-ready GPU pipeline" measurement: shows
//! the throughput ceiling when you actually exploit the fused
//! kernels' speedup, at the cost of needing real DC handling
//! downstream rather than a roundtrip-demo passthrough.
//!
//! Pipeline:
//!   RGB → XYB → gaborish ×3 → gather ×3
//!     → fused DCT+quantize ×3
//!     → fused dequant+IDCT-Y ×3 (Y kernel reused for X/B; CfL=0)
//!     → scatter ×3 → XYB inverse → RGB

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::persistent::GaborishWeights;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let sizes: Vec<usize> = std::env::var("SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![256, 512, 1024, 2048]);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

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

    println!("=== fused-kernel-only lossy pipeline throughput ===");
    println!("Iters per size: {iters} (1 warmup + {} sampled)", iters - 1);
    println!("Note: DC slot=0 in reconstruction (no DC restore — real encoder uses dc_coding).\n");
    println!(
        "{:>6}  {:>9}  {:>10}  {:>10}  {:>8}",
        "side", "MP", "fused ms", "split ms", "speedup"
    );
    println!("{}", "─".repeat(60));

    for &side in &sizes {
        let n = side * side;
        let mp = n as f64 / 1e6;
        let nb = (side / 8) * (side / 8);

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
        let weights_g = enc.upload_blocks(&vec![1.0_f32; nb * 64], nb as u32, 64);
        let qac = vec![4.0_f32; nb];
        let thr = [0.56_f32, 0.62, 0.62, 0.62];

        // ── FUSED pipeline ─────────────────────────────────────────
        let mut fused_times = Vec::with_capacity(iters);
        for it in 0..iters {
            let t = std::time::Instant::now();
            let g_r = enc.upload_plane(&r_plane, side as u32, side as u32);
            let g_g = enc.upload_plane(&g_plane, side as u32, side as u32);
            let g_b = enc.upload_plane(&b_plane, side as u32, side as u32);
            let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(&g_r, &g_g, &g_b);
            let xx_g = enc.gaborish_5x5_persistent(&xx, &weights);
            let xy_g = enc.gaborish_5x5_persistent(&xy, &weights);
            let xb_g = enc.gaborish_5x5_persistent(&xb, &weights);
            let bx_g = enc.gather_blocks_persistent(&xx_g, 8, 8);
            let by_g = enc.gather_blocks_persistent(&xy_g, 8, 8);
            let bb_g = enc.gather_blocks_persistent(&xb_g, 8, 8);
            // Fused DCT + quantize
            let q_x = enc.dct8_quantize_fused_persistent(&bx_g, &weights_g, &qac, &thr);
            let q_y = enc.dct8_quantize_fused_persistent(&by_g, &weights_g, &qac, &thr);
            let q_b = enc.dct8_quantize_fused_persistent(&bb_g, &weights_g, &qac, &thr);
            // Fused dequant + IDCT (Y kernel; CfL=0 so X/B can use it too)
            let recon_x_b = enc.dequant_idct8_fused_y_persistent(&q_x, &weights_g, &qac);
            let recon_y_b = enc.dequant_idct8_fused_y_persistent(&q_y, &weights_g, &qac);
            let recon_b_b = enc.dequant_idct8_fused_y_persistent(&q_b, &weights_g, &qac);
            let recon_x_p =
                enc.scatter_blocks_persistent(&recon_x_b, side as u32, side as u32, 8, 8);
            let recon_y_p =
                enc.scatter_blocks_persistent(&recon_y_b, side as u32, side as u32, 8, 8);
            let recon_b_p =
                enc.scatter_blocks_persistent(&recon_b_b, side as u32, side as u32, 8, 8);
            let (rgb_r, rgb_g, rgb_b) =
                enc.xyb_to_linear_rgb_planar_persistent(&recon_x_p, &recon_y_p, &recon_b_p);
            let _ = enc.download_plane(&rgb_r);
            let _ = enc.download_plane(&rgb_g);
            let _ = enc.download_plane(&rgb_b);
            let dt = t.elapsed();
            if it > 0 {
                fused_times.push(dt);
            }
        }
        fused_times.sort();
        let fused_med = fused_times[fused_times.len() / 2];

        // ── SPLIT pipeline (for comparison) ────────────────────────
        let xf = vec![0.0_f32; nb];
        let bf = vec![0.0_f32; nb];
        let mut split_times = Vec::with_capacity(iters);
        for it in 0..iters {
            let t = std::time::Instant::now();
            let g_r = enc.upload_plane(&r_plane, side as u32, side as u32);
            let g_g = enc.upload_plane(&g_plane, side as u32, side as u32);
            let g_b = enc.upload_plane(&b_plane, side as u32, side as u32);
            let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(&g_r, &g_g, &g_b);
            let xx_g = enc.gaborish_5x5_persistent(&xx, &weights);
            let xy_g = enc.gaborish_5x5_persistent(&xy, &weights);
            let xb_g = enc.gaborish_5x5_persistent(&xb, &weights);
            let bx_g = enc.gather_blocks_persistent(&xx_g, 8, 8);
            let by_g = enc.gather_blocks_persistent(&xy_g, 8, 8);
            let bb_g = enc.gather_blocks_persistent(&xb_g, 8, 8);
            let coeffs_x = enc.dct_8x8_wide_persistent(&bx_g);
            let coeffs_y = enc.dct_8x8_wide_persistent(&by_g);
            let coeffs_b = enc.dct_8x8_wide_persistent(&bb_g);
            let q_x = enc.quantize_dct8_persistent(&coeffs_x, &weights_g, &qac, &thr);
            let q_y = enc.quantize_dct8_persistent(&coeffs_y, &weights_g, &qac, &thr);
            let q_b = enc.quantize_dct8_persistent(&coeffs_b, &weights_g, &qac, &thr);
            let (dq_x, dq_y, dq_b) = enc.dequant_dct8_persistent(
                &q_x, &q_y, &q_b, &weights_g, &weights_g, &weights_g, &qac, &qac, &qac, &xf, &bf,
            );
            // No DC restore in split path either, for fair comparison
            let recon_x_b = enc.idct_8x8_wide_persistent(&dq_x);
            let recon_y_b = enc.idct_8x8_wide_persistent(&dq_y);
            let recon_b_b = enc.idct_8x8_wide_persistent(&dq_b);
            let recon_x_p =
                enc.scatter_blocks_persistent(&recon_x_b, side as u32, side as u32, 8, 8);
            let recon_y_p =
                enc.scatter_blocks_persistent(&recon_y_b, side as u32, side as u32, 8, 8);
            let recon_b_p =
                enc.scatter_blocks_persistent(&recon_b_b, side as u32, side as u32, 8, 8);
            let (rgb_r, rgb_g, rgb_b) =
                enc.xyb_to_linear_rgb_planar_persistent(&recon_x_p, &recon_y_p, &recon_b_p);
            let _ = enc.download_plane(&rgb_r);
            let _ = enc.download_plane(&rgb_g);
            let _ = enc.download_plane(&rgb_b);
            let dt = t.elapsed();
            if it > 0 {
                split_times.push(dt);
            }
        }
        split_times.sort();
        let split_med = split_times[split_times.len() / 2];

        let fused_ms = fused_med.as_secs_f64() * 1000.0;
        let split_ms = split_med.as_secs_f64() * 1000.0;
        let speedup = split_med.as_secs_f64() / fused_med.as_secs_f64();
        println!(
            "{:>6}  {:>9.3}  {:>10.2}  {:>10.2}  {:>6.2}×",
            side, mp, fused_ms, split_ms, speedup
        );
    }
    println!("\n  speedup = split_pipeline_time / fused_pipeline_time.");
    println!(
        "  Fused saves 6 launches per pipeline run (3 channels × 2 stages):\n  fwd: 3 DCT + 3 quantize → 3 fused-DCT-quant\n  inv: 3 dequant + 3 IDCT → 3 fused-dequant-IDCT-Y."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
