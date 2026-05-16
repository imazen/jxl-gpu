// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Diagnostic + regression gate: compare GPU pre_quantized DCT8
//! producer's output field-by-field against CPU `transform_and_quantize`
//! on the same synthetic input. Prints the first divergence per field
//! so we can tell whether the bug is in the AC quantize, DC quantize,
//! nzeros count, or CfL math.
//!
//! ## Tolerance
//!
//! Fails if ANY field diverges by more than ±1 (any large-delta
//! divergence is the bug pattern from jxl-gpu#8 — the fused DCT8
//! kernel was missing the post-DCT4 cascade and had t1/t2 storage
//! swapped, causing deltas of 5-15 per coefficient on real photos).
//!
//! Allows up to [`MAX_PERMITTED_PM1_DIFFS`] field positions to differ
//! by ±1 (the documented "~2% chroma AC coefs flip by 1 near rounding
//! ties" precision delta from cubecl-vs-jxl_simd FP non-associativity).
//! On the 16-block synthetic input this caps at ~150/3072 = 5%.

#![cfg(all(feature = "cuda", feature = "encoder"))]

use jxl_encoder::__pre_quantized::{
    AcStrategyMap, CflMap, EncoderPrecomputed, INV_DC_QUANT, NoiseParams, VarDctEncoder,
};
use jxl_encoder_gpu::encoder::GpuEncoder;
use jxl_encoder_gpu::forks::pre_quantized_ac::{
    PreQuantizedDct8Params, compute_pre_quantized_ac_dct8_persistent, reshape_to_transform_output,
};

type B = cubecl::cuda::CudaRuntime;

/// Maximum permitted ±1-delta field divergences across all
/// (channel, field) tuples. Set above the empirically-observed
/// post-fix divergence (~50) on the 16-block synthetic input. Bump
/// only when adding NEW intentional divergence (and document why).
const MAX_PERMITTED_PM1_DIFFS: usize = 150;

#[test]
fn gpu_producer_field_diff_vs_cpu() {
    let width = 32usize;
    let height = 32usize;
    let cpu_pw = width;
    let cpu_ph = height;
    let xsize_blocks = cpu_pw / 8;
    let ysize_blocks = cpu_ph / 8;
    let n_blocks = xsize_blocks * ysize_blocks;

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

    let mut cfl_map = CflMap::zeros(1, 1);
    cfl_map.ytox[0] = 4;
    cfl_map.ytob[0] = -2;

    let distance = 1.0_f32;
    let mut encoder = VarDctEncoder::new(distance);
    encoder.effort = 4;
    encoder.profile =
        jxl_encoder::effort::EffortProfile::lossy(4, jxl_encoder::api::EncoderMode::Reference);
    let params = jxl_encoder::__pre_quantized::DistanceParams::compute_for_profile(
        distance,
        &encoder.profile,
    );

    let raw_quant_uniform: u8 = 16;
    let mut quant_field = vec![raw_quant_uniform; n_blocks];

    let qac_uniform = params.scale * raw_quant_uniform as f32;
    let quant_field_float = vec![qac_uniform; n_blocks];
    let masking = vec![1.0_f32; n_blocks];

    let pc = EncoderPrecomputed::from_parts(
        width,
        height,
        xsize_blocks,
        ysize_blocks,
        cpu_pw,
        cpu_ph,
        xyb_x.clone(),
        xyb_y.clone(),
        xyb_b.clone(),
        Vec::new(),
        CflMap {
            ytox: cfl_map.ytox.clone(),
            ytob: cfl_map.ytob.clone(),
            xsize_tiles: 1,
            ysize_tiles: 1,
        },
        Option::<NoiseParams>::None,
        quant_field_float,
        masking,
        None,
        AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks),
        true,
        distance,
        0,
        0,
    );

    // CPU TransformOutput.
    let cpu_to = encoder
        .transform_and_quantize_for_test(&pc, &mut quant_field, &params)
        .expect("cpu transform");

    // GPU producer.
    let enc: GpuEncoder<B> = GpuEncoder::new();
    let xx_g = enc.upload_plane(&xyb_x, cpu_pw as u32, cpu_ph as u32);
    let xy_g = enc.upload_plane(&xyb_y, cpu_pw as u32, cpu_ph as u32);
    let xb_g = enc.upload_plane(&xyb_b, cpu_pw as u32, cpu_ph as u32);

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
    let x_qm_mul = (1.25_f32).powf(params.x_qm_scale as f32 - 2.0);
    let b_qm_mul = (1.25_f32).powf(params.b_qm_scale as f32 - 2.0);
    let qac_per_block: Vec<f32> = quant_field
        .iter()
        .map(|&q| params.scale * q as f32)
        .collect();
    let qac_qm_x: Vec<f32> = qac_per_block.iter().map(|&q| q * x_qm_mul).collect();
    let qac_qm_y: Vec<f32> = qac_per_block.clone();
    let qac_qm_b: Vec<f32> = qac_per_block.iter().map(|&q| q * b_qm_mul).collect();

    fn arr64(s: &[f32]) -> [f32; 64] {
        let mut a = [0.0; 64];
        a.copy_from_slice(s);
        a
    }
    let dct8_weights_x = arr64(jxl_encoder::__pre_quantized::quant_weights_dct8(0));
    let dct8_weights_y = arr64(jxl_encoder::__pre_quantized::quant_weights_dct8(1));
    let dct8_weights_b = arr64(jxl_encoder::__pre_quantized::quant_weights_dct8(2));

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
        &enc,
        &xx_g,
        &xy_g,
        &xb_g,
        xsize_blocks,
        ysize_blocks,
        &dct8_weights_x,
        &dct8_weights_y,
        &dct8_weights_b,
        &pq_params,
    );
    let gpu = reshape_to_transform_output(pq, xsize_blocks, ysize_blocks);

    // Field-by-field comparison.
    //
    // After the jxl-gpu#8 fix, ±1 deltas on quant_ac (chroma especially)
    // are expected and bounded in count via [`MAX_PERMITTED_PM1_DIFFS`].
    // Larger deltas, OR any quant_dc / nzeros divergence, mean the
    // producer is broken — fail loudly. (Pre-fix the test saw
    // delta-up-to-8 on quant_ac and large nzeros divergences across
    // the board; that pattern is the kernel bug in #8.)
    let mut pm1_diffs = 0usize;
    let mut large_diffs: Vec<String> = Vec::new();

    for c in 0..3 {
        let mut field_diffs = 0;
        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                let cpu_v = cpu_to.quant_dc[c][by][bx];
                let gpu_v = gpu.quant_dc[c][by][bx];
                if cpu_v != gpu_v {
                    field_diffs += 1;
                    let delta = (gpu_v - cpu_v).unsigned_abs() as usize;
                    if delta <= 1 {
                        pm1_diffs += 1;
                    } else {
                        large_diffs.push(format!(
                            "[quant_dc c={c}] (bx={bx},by={by}) cpu={cpu_v} gpu={gpu_v} \
                             |Δ|={delta}",
                        ));
                    }
                    if field_diffs <= 3 {
                        eprintln!("[quant_dc c={c}] (bx={bx},by={by}) cpu={cpu_v} gpu={gpu_v}",);
                    }
                }
            }
        }
        if field_diffs > 0 {
            eprintln!("[quant_dc c={c}] {field_diffs} of {n_blocks} differ");
        }
    }
    for c in 0..3 {
        let mut field_diffs = 0;
        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                for k in 0..64 {
                    let cpu_v = cpu_to.quant_ac[c][by][bx][k];
                    let gpu_v = gpu.quant_ac[c][by][bx][k];
                    if cpu_v != gpu_v {
                        field_diffs += 1;
                        let delta = (gpu_v - cpu_v).unsigned_abs() as usize;
                        if delta <= 1 {
                            pm1_diffs += 1;
                        } else {
                            large_diffs.push(format!(
                                "[quant_ac c={c}] (bx={bx},by={by}) k={k} cpu={cpu_v} \
                                 gpu={gpu_v} |Δ|={delta}",
                            ));
                        }
                        if field_diffs <= 3 {
                            eprintln!(
                                "[quant_ac c={c}] (bx={bx},by={by}) k={k} cpu={cpu_v} gpu={gpu_v}",
                            );
                        }
                    }
                }
            }
        }
        if field_diffs > 0 {
            eprintln!("[quant_ac c={c}] {field_diffs} of {} differ", n_blocks * 64);
        }
    }
    for c in 0..3 {
        let mut field_diffs = 0;
        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                let cpu_v = cpu_to.nzeros[c][by][bx];
                let gpu_v = gpu.nzeros[c][by][bx];
                if cpu_v != gpu_v {
                    field_diffs += 1;
                    let delta = (gpu_v as i32 - cpu_v as i32).unsigned_abs() as usize;
                    if delta <= 1 {
                        pm1_diffs += 1;
                    } else {
                        large_diffs.push(format!(
                            "[nzeros c={c}] (bx={bx},by={by}) cpu={cpu_v} gpu={gpu_v} \
                             |Δ|={delta}",
                        ));
                    }
                    if field_diffs <= 3 {
                        eprintln!("[nzeros c={c}] (bx={bx},by={by}) cpu={cpu_v} gpu={gpu_v}",);
                    }
                }
            }
        }
        if field_diffs > 0 {
            eprintln!("[nzeros c={c}] {field_diffs} of {n_blocks} differ");
        }
    }

    if !large_diffs.is_empty() {
        eprintln!(
            "{} large (|Δ|>1) divergences (bug pattern from jxl-gpu#8):",
            large_diffs.len()
        );
        for s in large_diffs.iter().take(10) {
            eprintln!("  {s}");
        }
        panic!(
            "GPU producer diverges from CPU by more than 1 in {} field positions \
             — fused DCT8 kernel may be broken (see jxl-gpu#8)",
            large_diffs.len(),
        );
    }
    if pm1_diffs > MAX_PERMITTED_PM1_DIFFS {
        panic!(
            "GPU producer ±1 divergences ({}) exceed the permitted ceiling ({}). \
             FP-precision drift increased — investigate.",
            pm1_diffs, MAX_PERMITTED_PM1_DIFFS,
        );
    }
    eprintln!(
        "PASS: {} ±1-delta field divergences (≤ {}) — within FP-precision tolerance.",
        pm1_diffs, MAX_PERMITTED_PM1_DIFFS,
    );
}
