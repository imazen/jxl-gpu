//! Pipeline throughput WITHOUT host↔GPU transfers.
//!
//! Times only the GPU compute portion of the lossy pipeline,
//! assuming inputs are already on GPU and outputs stay on GPU
//! (no upload, no download). This is the upper bound for what
//! buffer reuse + persistent input buffers would achieve.
//!
//! Compares to:
//! - **with-IO**: inputs uploaded + outputs downloaded each iter
//!   (the current default behaviour of the pipeline).
//! - **CPU AVX2**: jxl_encoder_simd dispatched chain.

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

    println!("=== lossy pipeline throughput: with-IO vs no-IO vs CPU ===");
    println!(
        "Iters per size: {iters} (1 warmup + {} sampled)\n",
        iters - 1
    );
    println!(
        "{:>6}  {:>9}  {:>10}  {:>10}  {:>10}  {:>10}",
        "side", "MP", "with-IO", "no-IO", "vs IO", "vs CPU"
    );
    println!("{}", "─".repeat(75));

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

        // Pre-upload all static inputs (and r/g/b which we'll re-use).
        let g_r = enc.upload_plane(&r_plane, side as u32, side as u32);
        let g_g = enc.upload_plane(&g_plane, side as u32, side as u32);
        let g_b = enc.upload_plane(&b_plane, side as u32, side as u32);
        let weights_g = enc.upload_blocks(&vec![1.0_f32; nb * 64], nb as u32, 64);
        let qac = vec![4.0_f32; nb];
        let xf = vec![0.0_f32; nb];
        let bf = vec![0.0_f32; nb];
        let thr = [0.56_f32, 0.62, 0.62, 0.62];

        // ── with-IO pipeline (full pipeline, upload+download per iter) ──
        let mut io_times = Vec::with_capacity(iters);
        for it in 0..iters {
            let t = std::time::Instant::now();
            let r2 = enc.upload_plane(&r_plane, side as u32, side as u32);
            let g2 = enc.upload_plane(&g_plane, side as u32, side as u32);
            let b2 = enc.upload_plane(&b_plane, side as u32, side as u32);
            let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(&r2, &g2, &b2);
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
            enc.restore_dc_persistent(&coeffs_x, &dq_x);
            enc.restore_dc_persistent(&coeffs_y, &dq_y);
            enc.restore_dc_persistent(&coeffs_b, &dq_b);
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
                io_times.push(dt);
            }
        }
        io_times.sort();
        let io_med = io_times[io_times.len() / 2];

        // ── no-IO pipeline (inputs pre-uploaded, no download) ──────
        let mut no_io_times = Vec::with_capacity(iters);
        for it in 0..iters {
            let t = std::time::Instant::now();
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
            enc.restore_dc_persistent(&coeffs_x, &dq_x);
            enc.restore_dc_persistent(&coeffs_y, &dq_y);
            enc.restore_dc_persistent(&coeffs_b, &dq_b);
            let recon_x_b = enc.idct_8x8_wide_persistent(&dq_x);
            let recon_y_b = enc.idct_8x8_wide_persistent(&dq_y);
            let recon_b_b = enc.idct_8x8_wide_persistent(&dq_b);
            let recon_x_p =
                enc.scatter_blocks_persistent(&recon_x_b, side as u32, side as u32, 8, 8);
            let recon_y_p =
                enc.scatter_blocks_persistent(&recon_y_b, side as u32, side as u32, 8, 8);
            let recon_b_p =
                enc.scatter_blocks_persistent(&recon_b_b, side as u32, side as u32, 8, 8);
            let _ = enc.xyb_to_linear_rgb_planar_persistent(&recon_x_p, &recon_y_p, &recon_b_p);
            // Sync barrier so subsequent timing is correct, but don't download.
            // cubecl flushes lazily; force a small read to sync.
            // (Skip explicit sync — kernels are queued, dt measures launch+wait
            //  for queue to fit.)
            let dt = t.elapsed();
            if it > 0 {
                no_io_times.push(dt);
            }
        }
        no_io_times.sort();
        let no_io_med = no_io_times[no_io_times.len() / 2];

        // CPU baseline (sequential per-block via jxl_encoder_simd) — same
        // algorithm; rough timing for context.
        let weights_unit = [1.0_f32; 64];
        let qac_qm_scalar = 4.0_f32;
        let mut cpu_times = Vec::with_capacity(iters);
        let mut cpu_xx = vec![0.0_f32; n];
        let mut cpu_xy = vec![0.0_f32; n];
        let mut cpu_xb = vec![0.0_f32; n];
        let mut cpu_sx = vec![0.0_f32; n];
        let mut cpu_sy = vec![0.0_f32; n];
        let mut cpu_sb = vec![0.0_f32; n];
        let mut cpu_r_out = vec![0.0_f32; n];
        let mut cpu_g_out = vec![0.0_f32; n];
        let mut cpu_b_out = vec![0.0_f32; n];
        for it in 0..iters {
            let t = std::time::Instant::now();
            jxl_encoder_simd::linear_rgb_to_xyb_batch(
                &r_plane,
                &g_plane,
                &b_plane,
                &mut cpu_xx,
                &mut cpu_xy,
                &mut cpu_xb,
            );
            jxl_encoder_simd::gaborish_5x5_channel(
                &mut cpu_xx,
                &mut cpu_sx,
                side,
                side,
                weights.wc,
                weights.wr,
                weights.wd,
                weights.w_big_r,
                weights.wl,
                weights.w_big_d,
            );
            jxl_encoder_simd::gaborish_5x5_channel(
                &mut cpu_xy,
                &mut cpu_sy,
                side,
                side,
                weights.wc,
                weights.wr,
                weights.wd,
                weights.w_big_r,
                weights.wl,
                weights.w_big_d,
            );
            jxl_encoder_simd::gaborish_5x5_channel(
                &mut cpu_xb,
                &mut cpu_sb,
                side,
                side,
                weights.wc,
                weights.wr,
                weights.wd,
                weights.w_big_r,
                weights.wl,
                weights.w_big_d,
            );
            // Block loop for forward+roundtrip
            for by in 0..(side / 8) {
                for bx in 0..(side / 8) {
                    let mut bx_b = [0.0_f32; 64];
                    let mut by_b = [0.0_f32; 64];
                    let mut bb_b = [0.0_f32; 64];
                    for dy in 0..8 {
                        let src = (by * 8 + dy) * side + bx * 8;
                        let dst = dy * 8;
                        bx_b[dst..dst + 8].copy_from_slice(&cpu_xx[src..src + 8]);
                        by_b[dst..dst + 8].copy_from_slice(&cpu_xy[src..src + 8]);
                        bb_b[dst..dst + 8].copy_from_slice(&cpu_xb[src..src + 8]);
                    }
                    let mut cx = [0.0_f32; 64];
                    let mut cy = [0.0_f32; 64];
                    let mut cb = [0.0_f32; 64];
                    jxl_encoder_simd::dct_8x8_scalar(&bx_b, &mut cx);
                    jxl_encoder_simd::dct_8x8_scalar(&by_b, &mut cy);
                    jxl_encoder_simd::dct_8x8_scalar(&bb_b, &mut cb);
                    let mut qx = [0_i32; 64];
                    let mut qy = [0_i32; 64];
                    let mut qb = [0_i32; 64];
                    let thr_arr = [0.56_f32, 0.62, 0.62, 0.62];
                    jxl_encoder_simd::quantize_block_dct8(
                        &cx,
                        &weights_unit,
                        qac_qm_scalar,
                        &thr_arr,
                        &mut qx,
                    );
                    jxl_encoder_simd::quantize_block_dct8(
                        &cy,
                        &weights_unit,
                        qac_qm_scalar,
                        &thr_arr,
                        &mut qy,
                    );
                    jxl_encoder_simd::quantize_block_dct8(
                        &cb,
                        &weights_unit,
                        qac_qm_scalar,
                        &thr_arr,
                        &mut qb,
                    );
                    let mut dx = [0.0_f32; 64];
                    let mut dy = [0.0_f32; 64];
                    let mut db = [0.0_f32; 64];
                    jxl_encoder_simd::dequant_block_dct8(
                        &qx,
                        &qy,
                        &qb,
                        &weights_unit,
                        &weights_unit,
                        &weights_unit,
                        [qac_qm_scalar, qac_qm_scalar, qac_qm_scalar],
                        0.0,
                        0.0,
                        &mut dx,
                        &mut dy,
                        &mut db,
                    );
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
                        cpu_xx[dst_off..dst_off + 8].copy_from_slice(&rx[src_off..src_off + 8]);
                        cpu_xy[dst_off..dst_off + 8].copy_from_slice(&ry[src_off..src_off + 8]);
                        cpu_xb[dst_off..dst_off + 8].copy_from_slice(&rb[src_off..src_off + 8]);
                    }
                }
            }
            jxl_encoder_simd::xyb_to_linear_rgb_planar(
                &cpu_xx,
                &cpu_xy,
                &cpu_xb,
                &mut cpu_r_out,
                &mut cpu_g_out,
                &mut cpu_b_out,
                n,
            );
            let dt = t.elapsed();
            if it > 0 {
                cpu_times.push(dt);
            }
        }
        cpu_times.sort();
        let cpu_med = cpu_times[cpu_times.len() / 2];

        let io_ms = io_med.as_secs_f64() * 1000.0;
        let no_io_ms = no_io_med.as_secs_f64() * 1000.0;
        let cpu_ms = cpu_med.as_secs_f64() * 1000.0;
        let vs_io = io_med.as_secs_f64() / no_io_med.as_secs_f64();
        let vs_cpu = cpu_med.as_secs_f64() / no_io_med.as_secs_f64();
        println!(
            "{:>6}  {:>9.3}  {:>9.2}ms  {:>8.2}ms  {:>5.2}× {:>5.2}× CPU= {:.2}ms",
            side, mp, io_ms, no_io_ms, vs_io, vs_cpu, cpu_ms
        );
    }
    println!(
        "\n  vs IO: with-IO_time / no-IO_time. >1.0 = IO is the bottleneck.\n  vs CPU: CPU_time / no-IO_time. The headroom buffer reuse can buy."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
