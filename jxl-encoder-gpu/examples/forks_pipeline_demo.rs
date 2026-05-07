//! Demonstrates the `forks::*` modules composing into a real pipeline.
//!
//! Chains four GPU pipeline stages on a synthetic 64×64 linear-RGB
//! image, mirroring the shape of a real VarDCT encoder:
//!
//! 1. `forks::xyb::convert_image_to_xyb_gpu` — RGB → XYB
//! 2. `forks::gaborish::gaborish_inverse_gpu` — 5×5 sharpen
//! 3. `forks::adaptive_quant::compute_mask1x1_gpu` — Y → mask field
//! 4. `forks::transform::apply_dct_batch_gpu` — Y → DCT8 coefficients
//!
//! Then verifies a forward+inverse roundtrip via
//! `forks::transform::apply_idct_batch_gpu` to confirm the DCT batch
//! dispatcher is consistent.
//!
//! This example is the closest thing the repo has today to "what the
//! GPU encoder pipeline looks like end-to-end" — it composes four
//! independent fork modules through GpuEncoder calls.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::adaptive_quant::compute_mask1x1_gpu;
    use jxl_encoder_gpu::forks::gaborish::gaborish_inverse_gpu;
    use jxl_encoder_gpu::forks::transform::{
        RAW_STRATEGY_DCT, apply_dct_batch_gpu, apply_idct_batch_gpu,
    };
    use jxl_encoder_gpu::forks::xyb::convert_image_to_xyb_gpu_alloc;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    const W: usize = 64;
    const H: usize = 64;

    // Build a synthetic linear-RGB image (interleaved RGB, length W*H*3).
    let mut linear_rgb = Vec::with_capacity(W * H * 3);
    for y in 0..H {
        for x in 0..W {
            let r = 0.1 + 0.6 * (x as f32 / W as f32);
            let g = 0.2 + 0.5 * (y as f32 / H as f32);
            let b = 0.3 + 0.4 * (((x + y) % 17) as f32 / 17.0);
            linear_rgb.push(r);
            linear_rgb.push(g);
            linear_rgb.push(b);
        }
    }

    println!("=== forks pipeline demo: 64×64 linear-RGB → DCT8 coefficients ===\n");

    // Stage 1: RGB → XYB (whole-image GPU launch).
    let (mut xyb_x, mut xyb_y, mut xyb_b) =
        convert_image_to_xyb_gpu_alloc(&enc, W, H, W, &linear_rgb, None);
    println!(
        "1. XYB:        Y[0]={:.4}, X[0]={:.4}, B[0]={:.4}",
        xyb_y[0], xyb_x[0], xyb_b[0]
    );
    assert!(xyb_x.iter().all(|v| v.is_finite()));
    assert!(xyb_y.iter().all(|v| v.is_finite()));
    assert!(xyb_b.iter().all(|v| v.is_finite()));

    // Stage 2: gaborish-inverse 5×5 sharpen on all 3 channels.
    gaborish_inverse_gpu(&enc, &mut xyb_x, &mut xyb_y, &mut xyb_b, W, H);
    println!(
        "2. gaborish:   Y[0]={:.4} (after sharpen)",
        xyb_y[0]
    );

    // Stage 3: mask1x1 field on the Y channel (post-gaborish).
    let mask = compute_mask1x1_gpu(&enc, &xyb_y, W, H);
    let mask_min = mask.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let mask_max = mask.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mask_mean: f32 = mask.iter().sum::<f32>() / mask.len() as f32;
    println!(
        "3. mask1x1:    {} values in [{:.4}, {:.4}], mean={:.4}",
        mask.len(),
        mask_min,
        mask_max,
        mask_mean
    );
    assert!(mask.iter().all(|v| v.is_finite() && *v > 0.0));

    // Stage 4: batched DCT8 over all 64 (= 8×8) blocks of the Y channel.
    let mut blocks = Vec::new();
    for by in 0..(H / 8) {
        for bx in 0..(W / 8) {
            blocks.push((bx, by));
        }
    }
    let coeffs = apply_dct_batch_gpu(&enc, &xyb_y, W, &blocks, RAW_STRATEGY_DCT);
    assert_eq!(coeffs.len(), 64 * 64);
    println!(
        "4. DCT8 batch: {} blocks → {} coefficients (block 0 DC = {:.4})",
        blocks.len(),
        coeffs.len(),
        coeffs[0]
    );

    // Stage 5: roundtrip via inverse to verify dispatcher consistency.
    let recon = apply_idct_batch_gpu(&enc, &coeffs, RAW_STRATEGY_DCT);
    let mut max_err = 0.0_f32;
    for (lin_idx, &(bx, by)) in blocks.iter().enumerate() {
        for dy in 0..8 {
            for dx in 0..8 {
                let orig_off = (by * 8 + dy) * W + bx * 8 + dx;
                let recon_off = lin_idx * 64 + dy * 8 + dx;
                max_err = max_err.max((xyb_y[orig_off] - recon[recon_off]).abs());
            }
        }
    }
    println!(
        "5. roundtrip:  DCT8 forward + inverse on Y, max|Δ|={max_err:.3e}\n"
    );
    assert!(
        max_err < 1e-4,
        "DCT8 batch roundtrip drift too large: {max_err:.3e}"
    );

    println!("✓ All 4 fork modules composed successfully through GpuEncoder.");
    println!(
        "✓ DCT8 batch round-trip consistent (single GPU launch of 64 blocks)."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
