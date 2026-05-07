//! Full lossy pipeline throughput: CPU vs full-GPU end-to-end.
//!
//! Pipeline:
//!   linear-RGB → XYB → gaborish (×3) → DCT8 (×3) → quantize (×3)
//!   → dequant → DC-restore → IDCT (×3) → XYB inverse → linear-RGB
//!
//! Compares:
//! - **CPU**: dispatched-SIMD via jxl_encoder_simd, sequential per-block
//!   DCT/quant/dequant/IDCT loop on host.
//! - **GPU**: persistent API end-to-end (3 uploads + ~14 GPU stages
//!   + 3 downloads). All intermediates GPU-resident.
//!
//! Sweeps sizes 256² → 4096². Includes a parity check (max abs RGB
//! delta) to confirm both paths produce equivalent output.

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

    // Gaborish weights for mul=1.0.
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
    let (wc, wr, wd, w_big_r, wl, w_big_d) = (
        norm as f32,
        (norm * K_GABORISH[0]) as f32,
        (norm * K_GABORISH[1]) as f32,
        (norm * K_GABORISH[2]) as f32,
        (norm * K_GABORISH[3]) as f32,
        (norm * K_GABORISH[4]) as f32,
    );
    let weights = GaborishWeights { wc, wr, wd, w_big_r, wl, w_big_d };

    println!("=== full lossy pipeline throughput: CPU vs full-GPU ===");
    println!("Iters per size: {iters} (1 warmup + {} sampled)\n", iters - 1);
    println!(
        "{:>6}  {:>9}  {:>10}  {:>10}  {:>10}  {:>8}  {:>10}",
        "side", "MP", "CPU ms", "GPU ms", "ratio", "max|Δ|R", "throughput"
    );
    println!("{}", "─".repeat(80));

    for &side in &sizes {
        let n = side * side;
        let mp = n as f64 / 1e6;
        let nb = (side / 8) * (side / 8);

        // Synthetic input.
        let mut r_plane = Vec::with_capacity(n);
        let mut g_plane = Vec::with_capacity(n);
        let mut b_plane = Vec::with_capacity(n);
        for y in 0..side {
            for x in 0..side {
                let r = 0.1 + 0.7 * (x as f32 / side as f32);
                let g = 0.2 + 0.6 * (y as f32 / side as f32);
                let b = 0.3 + 0.5 * (((x + y) % 17) as f32 / 17.0);
                r_plane.push(r);
                g_plane.push(g);
                b_plane.push(b);
            }
        }

        // ── CPU pipeline ────────────────────────────────────────────
        let mut cpu_times = Vec::with_capacity(iters);
        let mut cpu_r_out = vec![0.0_f32; n];
        let mut cpu_g_out = vec![0.0_f32; n];
        let mut cpu_b_out = vec![0.0_f32; n];

        let weights_unit = [1.0_f32; 64];
        let thr = [0.56_f32, 0.62, 0.62, 0.62];
        let qac_qm = 4.0_f32;

        for it in 0..iters {
            let t = std::time::Instant::now();

            // 1. XYB
            let mut xx = vec![0.0_f32; n];
            let mut xy = vec![0.0_f32; n];
            let mut xb = vec![0.0_f32; n];
            jxl_encoder_simd::linear_rgb_to_xyb_batch(
                &r_plane, &g_plane, &b_plane, &mut xx, &mut xy, &mut xb,
            );
            // 2. Gaborish (3 channels)
            let mut sx = vec![0.0_f32; n];
            let mut sy = vec![0.0_f32; n];
            let mut sb = vec![0.0_f32; n];
            jxl_encoder_simd::gaborish_5x5_channel(
                &mut xx, &mut sx, side, side, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            jxl_encoder_simd::gaborish_5x5_channel(
                &mut xy, &mut sy, side, side, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            jxl_encoder_simd::gaborish_5x5_channel(
                &mut xb, &mut sb, side, side, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            // 3. Per-block: DCT → quant → dequant → IDCT (per 8×8 tile)
            for by in 0..(side / 8) {
                for bx in 0..(side / 8) {
                    let mut bx_block = [0.0_f32; 64];
                    let mut by_block = [0.0_f32; 64];
                    let mut bb_block = [0.0_f32; 64];
                    for dy in 0..8 {
                        let src_off = (by * 8 + dy) * side + bx * 8;
                        let dst_off = dy * 8;
                        bx_block[dst_off..dst_off + 8]
                            .copy_from_slice(&xx[src_off..src_off + 8]);
                        by_block[dst_off..dst_off + 8]
                            .copy_from_slice(&xy[src_off..src_off + 8]);
                        bb_block[dst_off..dst_off + 8]
                            .copy_from_slice(&xb[src_off..src_off + 8]);
                    }
                    let mut cx = [0.0_f32; 64];
                    let mut cy = [0.0_f32; 64];
                    let mut cb = [0.0_f32; 64];
                    jxl_encoder_simd::dct_8x8_scalar(&bx_block, &mut cx);
                    jxl_encoder_simd::dct_8x8_scalar(&by_block, &mut cy);
                    jxl_encoder_simd::dct_8x8_scalar(&bb_block, &mut cb);

                    let mut qx = [0_i32; 64];
                    let mut qy = [0_i32; 64];
                    let mut qb = [0_i32; 64];
                    jxl_encoder_simd::quantize_block_dct8(&cx, &weights_unit, qac_qm, &thr, &mut qx);
                    jxl_encoder_simd::quantize_block_dct8(&cy, &weights_unit, qac_qm, &thr, &mut qy);
                    jxl_encoder_simd::quantize_block_dct8(&cb, &weights_unit, qac_qm, &thr, &mut qb);
                    let mut dx = [0.0_f32; 64];
                    let mut dy = [0.0_f32; 64];
                    let mut db = [0.0_f32; 64];
                    jxl_encoder_simd::dequant_block_dct8(
                        &qx, &qy, &qb, &weights_unit, &weights_unit, &weights_unit,
                        [qac_qm, qac_qm, qac_qm], 0.0, 0.0,
                        &mut dx, &mut dy, &mut db,
                    );
                    // DC restore (encoder dc_coding equivalent: passthrough).
                    dx[0] = cx[0];
                    dy[0] = cy[0];
                    db[0] = cb[0];
                    let mut rx = [0.0_f32; 64];
                    let mut ry = [0.0_f32; 64];
                    let mut rb = [0.0_f32; 64];
                    jxl_encoder_simd::idct_8x8_scalar(&dx, &mut rx);
                    jxl_encoder_simd::idct_8x8_scalar(&dy, &mut ry);
                    jxl_encoder_simd::idct_8x8_scalar(&db, &mut rb);
                    for dy_ in 0..8 {
                        let dst_off = (by * 8 + dy_) * side + bx * 8;
                        let src_off = dy_ * 8;
                        xx[dst_off..dst_off + 8].copy_from_slice(&rx[src_off..src_off + 8]);
                        xy[dst_off..dst_off + 8].copy_from_slice(&ry[src_off..src_off + 8]);
                        xb[dst_off..dst_off + 8].copy_from_slice(&rb[src_off..src_off + 8]);
                    }
                }
            }
            // 4. XYB → linear RGB
            jxl_encoder_simd::xyb_to_linear_rgb_planar(
                &xx, &xy, &xb, &mut cpu_r_out, &mut cpu_g_out, &mut cpu_b_out, n,
            );

            let dt = t.elapsed();
            if it > 0 {
                cpu_times.push(dt);
            }
        }
        cpu_times.sort();
        let cpu_med = cpu_times[cpu_times.len() / 2];

        // ── GPU pipeline (full persistent) ─────────────────────────
        let mut gpu_times = Vec::with_capacity(iters);
        let mut gpu_r_out = vec![0.0_f32; n];
        let mut gpu_g_out = vec![0.0_f32; n];
        let mut gpu_b_out = vec![0.0_f32; n];
        let weights_g = enc.upload_blocks(&vec![1.0_f32; nb * 64], nb as u32, 64);
        let qac_vec = vec![qac_qm; nb];
        let xf = vec![0.0_f32; nb];
        let bf = vec![0.0_f32; nb];

        for it in 0..iters {
            let t = std::time::Instant::now();
            // Upload
            let g_r = enc.upload_plane(&r_plane, side as u32, side as u32);
            let g_g = enc.upload_plane(&g_plane, side as u32, side as u32);
            let g_b = enc.upload_plane(&b_plane, side as u32, side as u32);
            // XYB
            let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(&g_r, &g_g, &g_b);
            // Gaborish ×3
            let xx_g = enc.gaborish_5x5_persistent(&xx, &weights);
            let xy_g = enc.gaborish_5x5_persistent(&xy, &weights);
            let xb_g = enc.gaborish_5x5_persistent(&xb, &weights);
            // Gather ×3
            let bx_g = enc.gather_blocks_persistent(&xx_g, 8, 8);
            let by_g = enc.gather_blocks_persistent(&xy_g, 8, 8);
            let bb_g = enc.gather_blocks_persistent(&xb_g, 8, 8);
            // DCT8 ×3
            let coeffs_x = enc.dct_8x8_wide_persistent(&bx_g);
            let coeffs_y = enc.dct_8x8_wide_persistent(&by_g);
            let coeffs_b = enc.dct_8x8_wide_persistent(&bb_g);
            // Quantize ×3
            let q_x =
                enc.quantize_dct8_persistent(&coeffs_x, &weights_g, &qac_vec, &thr);
            let q_y =
                enc.quantize_dct8_persistent(&coeffs_y, &weights_g, &qac_vec, &thr);
            let q_b =
                enc.quantize_dct8_persistent(&coeffs_b, &weights_g, &qac_vec, &thr);
            // Dequant
            let (dq_x, dq_y, dq_b) = enc.dequant_dct8_persistent(
                &q_x, &q_y, &q_b, &weights_g, &weights_g, &weights_g,
                &qac_vec, &qac_vec, &qac_vec, &xf, &bf,
            );
            // DC restore on-GPU
            enc.restore_dc_persistent(&coeffs_x, &dq_x);
            enc.restore_dc_persistent(&coeffs_y, &dq_y);
            enc.restore_dc_persistent(&coeffs_b, &dq_b);
            // IDCT ×3
            let recon_x_b = enc.idct_8x8_wide_persistent(&dq_x);
            let recon_y_b = enc.idct_8x8_wide_persistent(&dq_y);
            let recon_b_b = enc.idct_8x8_wide_persistent(&dq_b);
            // Scatter ×3
            let recon_x_p =
                enc.scatter_blocks_persistent(&recon_x_b, side as u32, side as u32, 8, 8);
            let recon_y_p =
                enc.scatter_blocks_persistent(&recon_y_b, side as u32, side as u32, 8, 8);
            let recon_b_p =
                enc.scatter_blocks_persistent(&recon_b_b, side as u32, side as u32, 8, 8);
            // XYB inverse
            let (rgb_r, rgb_g, rgb_b) =
                enc.xyb_to_linear_rgb_planar_persistent(&recon_x_p, &recon_y_p, &recon_b_p);
            // Download
            gpu_r_out = enc.download_plane(&rgb_r);
            gpu_g_out = enc.download_plane(&rgb_g);
            gpu_b_out = enc.download_plane(&rgb_b);
            let dt = t.elapsed();
            if it > 0 {
                gpu_times.push(dt);
            }
        }
        gpu_times.sort();
        let gpu_med = gpu_times[gpu_times.len() / 2];

        let cpu_ms = cpu_med.as_secs_f64() * 1000.0;
        let gpu_ms = gpu_med.as_secs_f64() * 1000.0;
        let ratio = cpu_med.as_secs_f64() / gpu_med.as_secs_f64();
        let max_dr = cpu_r_out
            .iter()
            .zip(&gpu_r_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let throughput_gpu = mp / gpu_med.as_secs_f64();
        println!(
            "{:>6}  {:>9.3}  {:>10.2}  {:>10.2}  {:>5.2}× {:>3}  {:>8.2e}  {:>7.0} MP/s",
            side,
            mp,
            cpu_ms,
            gpu_ms,
            ratio,
            if ratio > 1.0 { "GPU" } else { "CPU" },
            max_dr,
            throughput_gpu
        );
    }
    println!(
        "\n  ratio = CPU_time / GPU_time. \"GPU\" marker = GPU is faster.\n  max|Δ|R = max abs delta on R channel between CPU and GPU output.\n  Both paths run identical algorithm (XYB → gaborish → DCT8 → quant → dequant → DC-restore → IDCT → XYB inverse)."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
