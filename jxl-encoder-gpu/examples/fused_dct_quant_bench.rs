//! Fused DCT8+quantize bench: vs separate-stage (DCT then quantize).
//!
//! Measures whether fusing the two stages into one kernel (avoiding
//! the intermediate `coeffs` global memory write+read) yields a
//! measurable speedup at production batch sizes.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use cubecl::Runtime;
    use cubecl::prelude::*;
    use jxl_encoder_gpu::launch::dct8::dct_8x8_wide;
    use jxl_encoder_gpu::launch::fused_dct_quant::dct8_quantize_fused_wide;
    use jxl_encoder_gpu::launch::quantize::quantize_dct8;

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

    println!("=== fused DCT+quantize vs separate-stage ===");
    println!(
        "Iters per size: {iters} (1 warmup + {} sampled)\n",
        iters - 1
    );
    println!(
        "{:>6}  {:>9}  {:>10}  {:>10}  {:>8}  {:>10}",
        "side", "blocks", "split ms", "fused ms", "ratio", "max|Δ|"
    );
    println!("{}", "─".repeat(70));

    for &side in &sizes {
        let nb = (side / 8) * (side / 8);
        let n = nb * 64;

        let pixels: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
        let weights = vec![1.0_f32; n];
        let qac = vec![4.0_f32; nb];
        let thr = [0.56_f32, 0.62, 0.62, 0.62];

        // ── Split (separate DCT + quantize) ─────────────────────────
        let mut split_times = Vec::with_capacity(iters);
        let mut split_out = vec![0_i32; n];
        for it in 0..iters {
            let t = std::time::Instant::now();
            let h_in = client.create_from_slice(f32::as_bytes(&pixels));
            let h_w = client.create_from_slice(f32::as_bytes(&weights));
            let h_qac = client.create_from_slice(f32::as_bytes(&qac));
            let h_thr = client.create_from_slice(f32::as_bytes(&thr));
            let h_coef = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            let h_out = client.create_from_slice(i32::as_bytes(&vec![0_i32; n]));
            dct_8x8_wide::<Backend>(&client, h_in, h_coef.clone(), nb as u32);
            quantize_dct8::<Backend>(&client, h_coef, h_w, h_qac, h_thr, h_out.clone(), nb as u32);
            let bytes = client.read_one(h_out).expect("read");
            let result: &[i32] = i32::from_bytes(&bytes);
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

        // ── Fused ──────────────────────────────────────────────────
        let mut fused_times = Vec::with_capacity(iters);
        let mut fused_out = vec![0_i32; n];
        for it in 0..iters {
            let t = std::time::Instant::now();
            let h_in = client.create_from_slice(f32::as_bytes(&pixels));
            let h_w = client.create_from_slice(f32::as_bytes(&weights));
            let h_qac = client.create_from_slice(f32::as_bytes(&qac));
            let h_thr = client.create_from_slice(f32::as_bytes(&thr));
            let h_out = client.create_from_slice(i32::as_bytes(&vec![0_i32; n]));
            dct8_quantize_fused_wide::<Backend>(
                &client,
                h_in,
                h_w,
                h_qac,
                h_thr,
                h_out.clone(),
                nb as u32,
            );
            let bytes = client.read_one(h_out).expect("read");
            let result: &[i32] = i32::from_bytes(&bytes);
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

        // Parity
        let max_diff = split_out
            .iter()
            .zip(&fused_out)
            .map(|(a, b)| (a - b).unsigned_abs())
            .max()
            .unwrap_or(0);

        let split_ms = split_med.as_secs_f64() * 1000.0;
        let fused_ms = fused_med.as_secs_f64() * 1000.0;
        let ratio = split_med.as_secs_f64() / fused_med.as_secs_f64();
        println!(
            "{:>6}  {:>9}  {:>10.2}  {:>10.2}  {:>6.2}×  {:>10}",
            side, nb, split_ms, fused_ms, ratio, max_diff
        );
    }
    println!(
        "\n  ratio = split_time / fused_time. >1.0 = fused faster.\n  max|Δ| = max abs i32 diff between split + fused output (0 = bit-exact)."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
