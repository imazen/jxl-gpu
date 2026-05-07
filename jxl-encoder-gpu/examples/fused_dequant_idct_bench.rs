//! Fused dequant+IDCT8 (Y channel) bench — vs separate dequant + IDCT.
//!
//! Compares 3 paths producing Y-channel reconstructed pixels from
//! quantized i32 + per-coeff weights + per-block scale:
//!
//! 1. Split: dequant_dct8 (3-channel kernel, used for Y; X/B set to
//!    placeholder zero buffers + zero CfL factors) → idct_8x8_wide
//! 2. Fused: dequant_idct8_fused_y_wide_kernel (single channel Y,
//!    no CfL) → recon directly
//!
//! Reports throughput + parity (vs split path which is the reference).
//! Both paths zero DC; caller would restore DC via dc_coding in a
//! real encoder.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use cubecl::Runtime;
    use cubecl::prelude::*;
    use jxl_encoder_gpu::launch::dct8::idct_8x8_wide;
    use jxl_encoder_gpu::launch::dequant::dequant_dct8;
    use jxl_encoder_gpu::launch::fused_dct_quant::dequant_idct8_fused_y_wide;

    type Backend = cubecl::cuda::CudaRuntime;
    let device = <Backend as Runtime>::Device::default();
    let client = <Backend as Runtime>::client(&device);

    let sizes: Vec<usize> = std::env::var("SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![256, 512, 1024, 2048, 4096]);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    println!("=== fused dequant+IDCT8 (Y) vs separate-stage ===");
    println!("Iters per size: {iters} (1 warmup + {} sampled)\n", iters - 1);
    println!(
        "{:>6}  {:>9}  {:>10}  {:>10}  {:>8}  {:>10}",
        "side", "blocks", "split ms", "fused ms", "ratio", "max|Δ|"
    );
    println!("{}", "─".repeat(70));

    for &side in &sizes {
        let nb = (side / 8) * (side / 8);
        let n = nb * 64;

        // Synthetic quantized input. Mostly small ints with a few larger.
        let quant: Vec<i32> = (0..n).map(|i| ((i as i32 * 7) % 11) - 5).collect();
        let weights = vec![1.0_f32; n];
        let qac = vec![4.0_f32; nb];

        // ── Split (3-channel dequant_dct8 with zero X/B) → wide IDCT ──
        let mut split_times = Vec::with_capacity(iters);
        let mut split_out = vec![0.0_f32; n];
        let zeros_i32 = vec![0_i32; n];
        let cfl = vec![0.0_f32; nb];
        for it in 0..iters {
            let t = std::time::Instant::now();
            let h_qx = client.create_from_slice(i32::as_bytes(&zeros_i32));
            let h_qy = client.create_from_slice(i32::as_bytes(&quant));
            let h_qb = client.create_from_slice(i32::as_bytes(&zeros_i32));
            let h_w = client.create_from_slice(f32::as_bytes(&weights));
            let h_qm = client.create_from_slice(f32::as_bytes(&qac));
            let h_xf = client.create_from_slice(f32::as_bytes(&cfl));
            let h_bf = client.create_from_slice(f32::as_bytes(&cfl));
            let h_dx = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            let h_dy = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            let h_db = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            dequant_dct8::<Backend>(
                &client,
                h_qx,
                h_qy,
                h_qb,
                h_w.clone(),
                h_w.clone(),
                h_w.clone(),
                h_qm.clone(),
                h_qm.clone(),
                h_qm.clone(),
                h_xf,
                h_bf,
                h_dx,
                h_dy.clone(),
                h_db,
                nb as u32,
            );
            let h_ry = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            idct_8x8_wide::<Backend>(&client, h_dy, h_ry.clone(), nb as u32);
            let bytes = client.read_one(h_ry).expect("read");
            let result: &[f32] = f32::from_bytes(&bytes);
            let dt = t.elapsed();
            if it > 0 {
                split_times.push(dt);
            }
            if it == iters - 1 {
                split_out = result.to_vec();
            }
        }
        split_times.sort();
        let split_med = split_times[split_times.len() / 2];

        // ── Fused dequant+IDCT (Y only) ────────────────────────────
        let mut fused_times = Vec::with_capacity(iters);
        let mut fused_out = vec![0.0_f32; n];
        for it in 0..iters {
            let t = std::time::Instant::now();
            let h_qy = client.create_from_slice(i32::as_bytes(&quant));
            let h_w = client.create_from_slice(f32::as_bytes(&weights));
            let h_qm = client.create_from_slice(f32::as_bytes(&qac));
            let h_ry = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            dequant_idct8_fused_y_wide::<Backend>(
                &client, h_qy, h_w, h_qm, h_ry.clone(), nb as u32,
            );
            let bytes = client.read_one(h_ry).expect("read");
            let result: &[f32] = f32::from_bytes(&bytes);
            let dt = t.elapsed();
            if it > 0 {
                fused_times.push(dt);
            }
            if it == iters - 1 {
                fused_out = result.to_vec();
            }
        }
        fused_times.sort();
        let fused_med = fused_times[fused_times.len() / 2];

        let max_diff = split_out
            .iter()
            .zip(&fused_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        let split_ms = split_med.as_secs_f64() * 1000.0;
        let fused_ms = fused_med.as_secs_f64() * 1000.0;
        let ratio = split_med.as_secs_f64() / fused_med.as_secs_f64();
        println!(
            "{:>6}  {:>9}  {:>10.2}  {:>10.2}  {:>6.2}×  {:>10.2e}",
            side, nb, split_ms, fused_ms, ratio, max_diff
        );
    }
    println!(
        "\n  ratio = split_time / fused_time. >1.0 = fused faster.\n  max|Δ| = max abs f32 diff between split + fused output (Y channel)."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
