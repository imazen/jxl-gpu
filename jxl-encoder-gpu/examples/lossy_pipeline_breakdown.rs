//! Per-stage timing breakdown of the fused lossy pipeline at 2048².
//!
//! Diagnoses where the GPU pipeline's wall-clock time is going
//! between (1) host-side allocs / handle juggling, (2) GPU
//! upload+download, (3) actual kernel launches. This is the data
//! needed to decide what to optimize next.

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
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(11); // 1 warmup + 10 sampled, take median per stage

    let n = side * side;
    let nb = (side / 8) * (side / 8);
    println!("=== fused pipeline breakdown @ {side}×{side} ({n} px, {nb} blocks) ===");
    println!("Iters: {iters} (1 warmup + {} sampled, median per stage)\n", iters - 1);

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
    let qac = vec![4.0_f32; nb];
    let thr = [0.56_f32, 0.62, 0.62, 0.62];

    let stages = [
        "1. upload_plane × 3     (RGB → GPU)",
        "2. xyb_from_linear_rgb  (3 channels → XYB)",
        "3. gaborish × 3         (5×5 sharpen each ch)",
        "4. gather × 3           (planes → blocks)",
        "5. fused DCT+quant × 3",
        "6. fused dequant+IDCT × 3",
        "7. scatter × 3          (blocks → planes)",
        "8. xyb_to_linear_rgb    (XYB → RGB)",
        "9. download_plane × 3   (GPU → host)",
    ];
    let n_stages = stages.len();
    let mut all: Vec<Vec<u128>> = (0..n_stages).map(|_| Vec::with_capacity(iters)).collect();

    for it in 0..iters {
        let t = std::time::Instant::now();
        let g_r = enc.upload_plane(&r_plane, side as u32, side as u32);
        let g_g = enc.upload_plane(&g_plane, side as u32, side as u32);
        let g_b = enc.upload_plane(&b_plane, side as u32, side as u32);
        let t1 = std::time::Instant::now();
        let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(&g_r, &g_g, &g_b);
        let t2 = std::time::Instant::now();
        let xx_g = enc.gaborish_5x5_persistent(&xx, &weights);
        let xy_g = enc.gaborish_5x5_persistent(&xy, &weights);
        let xb_g = enc.gaborish_5x5_persistent(&xb, &weights);
        let t3 = std::time::Instant::now();
        let bx_g = enc.gather_blocks_persistent(&xx_g, 8, 8);
        let by_g = enc.gather_blocks_persistent(&xy_g, 8, 8);
        let bb_g = enc.gather_blocks_persistent(&xb_g, 8, 8);
        let t4 = std::time::Instant::now();
        let q_x = enc.dct8_quantize_fused_persistent(&bx_g, &weights_g, &qac, &thr);
        let q_y = enc.dct8_quantize_fused_persistent(&by_g, &weights_g, &qac, &thr);
        let q_b = enc.dct8_quantize_fused_persistent(&bb_g, &weights_g, &qac, &thr);
        let t5 = std::time::Instant::now();
        let recon_x_b = enc.dequant_idct8_fused_y_persistent(&q_x, &weights_g, &qac);
        let recon_y_b = enc.dequant_idct8_fused_y_persistent(&q_y, &weights_g, &qac);
        let recon_b_b = enc.dequant_idct8_fused_y_persistent(&q_b, &weights_g, &qac);
        let t6 = std::time::Instant::now();
        let recon_x_p =
            enc.scatter_blocks_persistent(&recon_x_b, side as u32, side as u32, 8, 8);
        let recon_y_p =
            enc.scatter_blocks_persistent(&recon_y_b, side as u32, side as u32, 8, 8);
        let recon_b_p =
            enc.scatter_blocks_persistent(&recon_b_b, side as u32, side as u32, 8, 8);
        let t7 = std::time::Instant::now();
        let (rgb_r, rgb_g, rgb_b) =
            enc.xyb_to_linear_rgb_planar_persistent(&recon_x_p, &recon_y_p, &recon_b_p);
        let t8 = std::time::Instant::now();
        let _ = enc.download_plane(&rgb_r);
        let _ = enc.download_plane(&rgb_g);
        let _ = enc.download_plane(&rgb_b);
        let t9 = std::time::Instant::now();

        if it > 0 {
            all[0].push((t1 - t).as_micros());
            all[1].push((t2 - t1).as_micros());
            all[2].push((t3 - t2).as_micros());
            all[3].push((t4 - t3).as_micros());
            all[4].push((t5 - t4).as_micros());
            all[5].push((t6 - t5).as_micros());
            all[6].push((t7 - t6).as_micros());
            all[7].push((t8 - t7).as_micros());
            all[8].push((t9 - t8).as_micros());
        }
    }

    println!("{:>50}  {:>10}  {:>8}", "stage", "median µs", "% total");
    println!("{}", "─".repeat(75));
    let mut total: u128 = 0;
    let mut medians = Vec::with_capacity(n_stages);
    for s in &mut all {
        s.sort();
        let med = s[s.len() / 2];
        medians.push(med);
        total += med;
    }
    for (i, name) in stages.iter().enumerate() {
        let pct = 100.0 * medians[i] as f64 / total as f64;
        println!("{:>50}  {:>10}  {:>7.1}%", name, medians[i], pct);
    }
    println!("{}", "─".repeat(75));
    println!("{:>50}  {:>10}  {:>7.1}%", "TOTAL", total, 100.0);
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
