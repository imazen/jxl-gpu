//! End-to-end demo composing 8 GpuEncoder methods on real image-shaped data.
//!
//! Pipeline:
//!   1. Linear-RGB input (synthetic 32×32)
//!   2. GPU XYB transform
//!   3. GPU mask1x1 field
//!   4. Pack Y plane into 8x8 blocks
//!   5. GPU DCT8
//!   6. GPU quantize_dct8
//!   7. GPU dequant (Y-only path, single channel)
//!   8. GPU IDCT8
//!   9. Unpack reconstructed Y plane
//!  10. GPU block_l2 (use Y for all 3 channels as a single-channel proxy)
//!
//! Reports per-block costs from the Y channel reconstruction.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    // 32×32 = 4×4 blocks of 8×8
    const W: u32 = 32;
    const H: u32 = 32;
    const N: usize = (W * H) as usize;
    const NB: usize = 16;

    // Synthetic linear-RGB
    let r: Vec<f32> = (0..N).map(|i| 0.1 + 0.5 * (i as f32 / N as f32)).collect();
    let g: Vec<f32> = (0..N).map(|i| 0.5 - 0.3 * (i as f32 / N as f32)).collect();
    let b: Vec<f32> = (0..N)
        .map(|i| 0.3 + 0.4 * ((i % 17) as f32 / 17.0))
        .collect();

    // Step 2: XYB
    let (_x, y_plane, _b_xyb) = enc.xyb_from_linear_rgb(&r, &g, &b);

    // Step 3: mask1x1
    let mask = enc.mask1x1_field(&y_plane, W, H);
    let mask_ok = mask.iter().all(|v| v.is_finite() && *v > 0.0);
    assert!(mask_ok);

    // Step 4: pack Y into block-major 8x8 layout (row-major within blocks)
    let mut y_blocks = vec![0.0_f32; N];
    for by in 0..4 {
        for bx in 0..4 {
            for ly in 0..8 {
                for lx in 0..8 {
                    let block_idx = by * 4 + bx;
                    y_blocks[block_idx * 64 + ly * 8 + lx] =
                        y_plane[(by * 8 + ly) * (W as usize) + (bx * 8 + lx)];
                }
            }
        }
    }

    // Step 5: DCT8
    let dct = enc.dct_8x8_blocks(&y_blocks);

    // Step 6: Quantize
    let weights: Vec<f32> = {
        let mut per = vec![1.0_f32; 64];
        for i in 0..64 {
            let r = (i / 8) as f32;
            let c = (i % 8) as f32;
            per[i] = 1.0 + 0.5 * (r + c);
        }
        let mut w = vec![0.0_f32; N];
        for b in 0..NB {
            w[b * 64..b * 64 + 64].copy_from_slice(&per);
        }
        w
    };
    let qac = vec![1.7_f32; NB];
    let thresholds = [0.62_f32; 4];
    let quant = enc.quantize_dct8_blocks(&dct, &weights, &qac, &thresholds);
    let nonzero_count = quant.iter().filter(|&&v| v != 0).count();
    println!(
        "Quantized {} coefficients ({} non-zero)",
        quant.len(),
        nonzero_count
    );

    // Step 7: Dequantize via the simple "quant * weight" path. dequant_dct8
    // expects 3 channels + CfL — for this single-channel demo, we run it
    // with all 3 channels = Y and zero CfL factors, then take Y output.
    let weights_per_block = vec![1.0_f32; NB];
    let zero_cfl = vec![0.0_f32; NB];
    let (_dq_x, dq_y, _dq_b) = enc.dequant_dct8_blocks(
        &quant,
        &quant,
        &quant,
        &weights,
        &weights,
        &weights,
        &weights_per_block, // qac_qm: per-block scalar
        &weights_per_block,
        &weights_per_block,
        &zero_cfl,
        &zero_cfl,
    );

    // Step 8: IDCT
    let recon_blocks = enc.idct_8x8_blocks(&dq_y);

    // Step 9: Unpack reconstructed Y
    let mut recon_y = vec![0.0_f32; N];
    for by in 0..4 {
        for bx in 0..4 {
            for ly in 0..8 {
                for lx in 0..8 {
                    let block_idx = by * 4 + bx;
                    recon_y[(by * 8 + ly) * (W as usize) + (bx * 8 + lx)] =
                        recon_blocks[block_idx * 64 + ly * 8 + lx];
                }
            }
        }
    }

    // Step 10: block_l2 with Y in all 3 channels
    let costs = enc.block_l2_errors(
        &y_plane, &y_plane, &y_plane, &recon_y, &recon_y, &recon_y, &mask, 4, 4, W,
    );

    println!("Per-block reconstruction L2 (16 blocks):");
    for (i, c) in costs.iter().enumerate() {
        print!(" {:.3}", c);
        if (i + 1) % 4 == 0 {
            println!();
        }
    }

    let max_cost = costs.iter().cloned().fold(0.0_f32, f32::max);
    let min_cost = costs.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("Cost range: min={min_cost:.3}, max={max_cost:.3}");

    assert!(costs.iter().all(|c| c.is_finite() && *c >= 0.0));
    println!("\n✓ End-to-end GpuEncoder pipeline (8 methods composed) works on 32x32 input.");
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
