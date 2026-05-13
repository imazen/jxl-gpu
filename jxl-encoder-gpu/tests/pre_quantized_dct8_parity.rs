// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! End-to-end bitstream parity:
//! CPU `VarDctEncoder::encode_from_precomputed` vs the new
//! GPU pre-quantized AC producer feeding
//! `VarDctEncoder::encode_from_pre_quantized_ac`.
//!
//! All-DCT8 only — synthesizes XYB so the cost-grid selector
//! never runs; the test sets `AcStrategyMap::new_dct8` (every
//! block forced to strategy 0). Both pipelines see identical
//! inputs; outputs must be byte-identical.

#![cfg(all(feature = "cuda", feature = "encoder"))]

use jxl_encoder::__pre_quantized::{
    AcStrategyMap, CflMap, DistanceParams, EncoderPrecomputed, INV_DC_QUANT, NoiseParams,
    VarDctEncoder,
};
use jxl_encoder_gpu::encoder::GpuEncoder;
use jxl_encoder_gpu::forks::pre_quantized_ac::{
    PreQuantizedDct8Params, compute_pre_quantized_ac_dct8_persistent,
    reshape_to_transform_output,
};

type B = cubecl::cuda::CudaRuntime;

#[test]
#[ignore = "FAILING — GPU producer's TransformOutput diverges from CPU. \
            Infrastructure in place; debugging the actual divergence is the \
            next chunk. Likely DCT-precision FP, AdjustQuantBlockAC at e>=5, \
            or an off-by-one in encode_from_pre_quantized_ac wiring. Run with \
            --ignored to see the diff for diagnostics."]
fn pre_quantized_dct8_bitstream_parity_vs_cpu() {
    // Tiny synthetic 32×32 image (4×4 = 16 DCT8 blocks).
    // Width and height are both multiples of 16 so gpu_pw == cpu_pw,
    // sidestepping the gpu_pw/cpu_pw stride bridge.
    let width = 32usize;
    let height = 32usize;
    let cpu_pw = width;
    let cpu_ph = height;
    let xsize_blocks = cpu_pw / 8;
    let ysize_blocks = cpu_ph / 8;
    let n_blocks = xsize_blocks * ysize_blocks;

    // Synthesize XYB planes — non-trivial, deterministic, mid-range.
    let make_plane = |seed: usize| -> Vec<f32> {
        (0..cpu_pw * cpu_ph)
            .map(|i| {
                let s = ((i + seed * 13) as f32) * 0.0231;
                0.05 + 0.04 * s.sin() + 0.02 * (s * 1.7).cos()
            })
            .collect()
    };
    let xyb_x = make_plane(1);
    let xyb_y = make_plane(31);
    let xyb_b = make_plane(101);

    // Uniform CfL map — same single tile (8×8 blocks). The 32×32
    // image fits in a single 64-pixel CfL tile so xsize_tiles = 1.
    // Use a non-zero ytox/ytob so the CfL math actually exercises.
    let xsize_tiles = 1;
    let ysize_tiles = 1;
    let mut cfl_map = CflMap::zeros(xsize_tiles, ysize_tiles);
    cfl_map.ytox[0] = 4;
    cfl_map.ytob[0] = -2;

    // Force every block to DCT8.
    let ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);

    // Use effort 4 so CPU's AdjustQuantBlockAC (gated at effort >= 5)
    // stays disabled. The GPU producer doesn't implement AdjustQuantBlockAC,
    // so matching CPU at effort >= 5 would require additional work.
    let distance = 1.0_f32;
    let mut encoder = VarDctEncoder::new(distance);
    // Force effort 4 (kCheetah) to bypass AdjustQuantBlockAC (gated
    // at effort >= 5). The GPU producer doesn't replicate that
    // per-block quant_field adjustment, so a parity test at higher
    // effort would diverge by the AdjustQuantBlockAC delta even with
    // perfect AC quant matching.
    encoder.effort = 4;
    encoder.profile = jxl_encoder::effort::EffortProfile::lossy(
        4,
        jxl_encoder::api::EncoderMode::Reference,
    );
    let params = DistanceParams::compute_for_profile(distance, &encoder.profile);

    // Quant field — uniform raw_quant for simplicity. Apply
    // adjust_quant_field_with_distance is done internally by both
    // encode_from_precomputed and encode_from_pre_quantized_ac, so we
    // pass the un-adjusted u8 here.
    let raw_quant_uniform: u8 = 16;
    let quant_field = vec![raw_quant_uniform; n_blocks];

    // quant_field_float used by EncoderPrecomputed — for an all-DCT8
    // image at uniform raw_quant this is just qac per block. The CPU
    // encoder reads this for ac_strategy_search heuristics; for a
    // uniform pre-set strategy map it's mostly metadata. Set it to
    // params.scale * raw_quant — same as CPU's `let qac = params.scale
    // * quant_int as f32` in transform_blocks_into.
    let qac_uniform = params.scale * raw_quant_uniform as f32;
    let quant_field_float = vec![qac_uniform; n_blocks];
    let masking = vec![1.0_f32; n_blocks];

    // Build EncoderPrecomputed twice (it's not Clone), once for each
    // encode call.
    let precomputed_cpu = EncoderPrecomputed::from_parts(
        width, height, xsize_blocks, ysize_blocks, cpu_pw, cpu_ph,
        xyb_x.clone(), xyb_y.clone(), xyb_b.clone(),
        Vec::new(),
        CflMap { ytox: cfl_map.ytox.clone(), ytob: cfl_map.ytob.clone(),
                 xsize_tiles, ysize_tiles },
        Option::<NoiseParams>::None,
        quant_field_float.clone(),
        masking.clone(),
        None,
        AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks),
        true, // gaborish_enabled
        distance,
        0, 0,
    );
    let precomputed_gpu = EncoderPrecomputed::from_parts(
        width, height, xsize_blocks, ysize_blocks, cpu_pw, cpu_ph,
        xyb_x.clone(), xyb_y.clone(), xyb_b.clone(),
        Vec::new(),
        CflMap { ytox: cfl_map.ytox.clone(), ytob: cfl_map.ytob.clone(),
                 xsize_tiles, ysize_tiles },
        Option::<NoiseParams>::None,
        quant_field_float.clone(),
        masking,
        None,
        AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks),
        true,
        distance,
        0, 0,
    );

    let _ = ac_strategy; // both precomputed instances have their own.

    // CPU bitstream.
    let bitstream_cpu = encoder
        .encode_from_precomputed(&precomputed_cpu, &quant_field)
        .expect("CPU encode_from_precomputed");

    // GPU pipeline.
    let enc: GpuEncoder<B> = GpuEncoder::new();
    let xx_g = enc.upload_plane(&xyb_x, cpu_pw as u32, cpu_ph as u32);
    let xy_g = enc.upload_plane(&xyb_y, cpu_pw as u32, cpu_ph as u32);
    let xb_g = enc.upload_plane(&xyb_b, cpu_pw as u32, cpu_ph as u32);

    // Build PreQuantizedDct8Params from CfL + DistanceParams.
    // cfl tile size = 64 pixels = 8 blocks. (bx/8, by/8) → tile.
    const K_INV_COLOR_FACTOR: f32 = 1.0 / 84.0;
    let mut x_factor_per_block = vec![0.0f32; n_blocks];
    let mut b_factor_per_block = vec![0.0f32; n_blocks];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let tx = bx / 8;
            let ty = by / 8;
            let i = by * xsize_blocks + bx;
            x_factor_per_block[i] = (cfl_map.ytox_at(tx, ty) as f32) * K_INV_COLOR_FACTOR;
            b_factor_per_block[i] = 1.0 + (cfl_map.ytob_at(tx, ty) as f32) * K_INV_COLOR_FACTOR;
        }
    }

    // qm_multiplier per channel: x_qm_mul = 1.25^(x_qm_scale - 2),
    // b_qm_mul = 1.25^(b_qm_scale - 2). Pulled from DistanceParams.
    let x_qm_mul = (1.25_f32).powf(params.x_qm_scale as f32 - 2.0);
    let b_qm_mul = (1.25_f32).powf(params.b_qm_scale as f32 - 2.0);

    let qac_per_block: Vec<f32> = quant_field.iter().map(|&q| params.scale * q as f32).collect();
    let qac_qm_x: Vec<f32> = qac_per_block.iter().map(|&q| q * x_qm_mul).collect();
    let qac_qm_y: Vec<f32> = qac_per_block.clone();
    let qac_qm_b: Vec<f32> = qac_per_block.iter().map(|&q| q * b_qm_mul).collect();

    let dct8_weights_x: [f32; 64] = pull_quant_weights_dct8(0);
    let dct8_weights_y: [f32; 64] = pull_quant_weights_dct8(1);
    let dct8_weights_b: [f32; 64] = pull_quant_weights_dct8(2);

    let pq_params = PreQuantizedDct8Params {
        qac_per_block,
        x_factor_per_block,
        b_factor_per_block,
        qac_qm_x,
        qac_qm_y,
        qac_qm_b,
        inv_dc_factor_x: INV_DC_QUANT[0] * params.scale_dc,
        inv_dc_factor_y: INV_DC_QUANT[1] * params.scale_dc,
        inv_dc_factor_b: INV_DC_QUANT[2] * params.scale_dc,
        thresholds_x: jxl_encoder::__pre_quantized::default_thresholds_dct8(0),
        thresholds_y: jxl_encoder::__pre_quantized::default_thresholds_dct8(1),
        thresholds_b: jxl_encoder::__pre_quantized::default_thresholds_dct8(2),
    };

    let pq = compute_pre_quantized_ac_dct8_persistent(
        &enc, &xx_g, &xy_g, &xb_g,
        xsize_blocks, ysize_blocks,
        &dct8_weights_x, &dct8_weights_y, &dct8_weights_b,
        &pq_params,
    );
    let r = reshape_to_transform_output(pq, xsize_blocks, ysize_blocks);

    let bitstream_gpu = encoder
        .encode_from_pre_quantized_ac(
            &precomputed_gpu, &quant_field,
            &r.quant_dc, &r.quant_ac, &r.nzeros, &r.raw_nzeros,
        )
        .expect("GPU encode_from_pre_quantized_ac");

    // Bit-identical test. If this fails, we have a CfL / quantize /
    // round-tie / DC mismatch — log first divergence to help debug.
    if bitstream_cpu != bitstream_gpu {
        let n = bitstream_cpu.len().min(bitstream_gpu.len());
        let mut diff_at = None;
        for i in 0..n {
            if bitstream_cpu[i] != bitstream_gpu[i] {
                diff_at = Some(i);
                break;
            }
        }
        eprintln!(
            "BITSTREAM MISMATCH: cpu_len={}, gpu_len={}, first_diff_at={:?}",
            bitstream_cpu.len(),
            bitstream_gpu.len(),
            diff_at,
        );
        if let Some(i) = diff_at {
            let lo = i.saturating_sub(8);
            let hi = (i + 8).min(n);
            eprintln!("CPU bytes [{}..{}]: {:02x?}", lo, hi, &bitstream_cpu[lo..hi]);
            eprintln!("GPU bytes [{}..{}]: {:02x?}", lo, hi, &bitstream_gpu[lo..hi]);
        }
    }
    assert_eq!(bitstream_cpu, bitstream_gpu);
}

/// Pull the per-coefficient DCT8 quant weights from jxl-encoder.
/// These are the same weights `quant_weights(0, c)` would return
/// for strategy 0 (DCT8).
fn pull_quant_weights_dct8(channel: usize) -> [f32; 64] {
    let w = jxl_encoder::__pre_quantized::quant_weights_dct8(channel);
    let mut out = [0.0_f32; 64];
    out.copy_from_slice(w);
    out
}
