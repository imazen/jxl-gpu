// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Diagnostic: feed CPU `transform_and_quantize` output through
//! `encode_from_pre_quantized_ac` and compare against
//! `encode_from_precomputed`. They should produce IDENTICAL bitstreams
//! since the only difference is "skip transform_and_quantize internally"
//! vs "do it then forward the same data".
//!
//! If this test FAILS, the bug is in the new entry point itself —
//! likely a parameter handling difference (DistanceParams, quant_field
//! adjustment, etc.) between `encode_from_precomputed` and
//! `encode_from_pre_quantized_ac`.
//!
//! If this test PASSES, the entry point is correct and any failure of
//! `pre_quantized_dct8_parity` is in the GPU producer (DCT precision,
//! CfL formula, etc.).

#![cfg(feature = "encoder")]

use jxl_encoder::__pre_quantized::{
    AcStrategyMap, CflMap, EncoderPrecomputed, NoiseParams, VarDctEncoder,
};

#[test]
fn encode_from_pre_quantized_ac_matches_encode_from_precomputed() {
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

    let raw_quant_uniform: u8 = 16;
    let mut quant_field_a = vec![raw_quant_uniform; n_blocks];

    let qac_uniform = jxl_encoder::__pre_quantized::DistanceParams::compute_for_profile(
        distance,
        &encoder.profile,
    )
    .scale
        * raw_quant_uniform as f32;
    let quant_field_float = vec![qac_uniform; n_blocks];
    let masking = vec![1.0_f32; n_blocks];

    let pc_a = EncoderPrecomputed::from_parts(
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
        quant_field_float.clone(),
        masking.clone(),
        None,
        AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks),
        true,
        distance,
        0,
        0,
    );

    // Path A: encode_from_precomputed (runs transform_and_quantize internally)
    let bitstream_a = encoder
        .encode_from_precomputed(&pc_a, &quant_field_a)
        .expect("path A");

    // Path B: manually run transform_and_quantize, feed result to encode_from_pre_quantized_ac.
    let mut quant_field_b = vec![raw_quant_uniform; n_blocks];
    let pc_b = EncoderPrecomputed::from_parts(
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
        quant_field_float.clone(),
        masking,
        None,
        AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks),
        true,
        distance,
        0,
        0,
    );

    let params = jxl_encoder::__pre_quantized::DistanceParams::compute_for_profile(
        distance,
        &encoder.profile,
    );
    // This needs a tiny bit of adjustment: encode_from_precomputed
    // calls adjust_quant_field_with_distance internally on a clone
    // of quant_field. transform_and_quantize_for_test takes a &mut
    // and applies its own adjustments. To keep both paths in sync,
    // feed the SAME unadjusted quant_field to both paths and let
    // each one adjust internally.
    let to = encoder
        .transform_and_quantize_for_test(&pc_b, &mut quant_field_b, &params)
        .expect("transform_and_quantize");

    let bitstream_b = encoder
        .encode_from_pre_quantized_ac(
            &pc_b,
            &quant_field_a, // un-adjusted; encode_from_pre_quantized_ac adjusts internally
            &to.quant_dc,
            &to.quant_ac,
            &to.nzeros,
            &to.raw_nzeros,
        )
        .expect("path B");

    if bitstream_a != bitstream_b {
        let n = bitstream_a.len().min(bitstream_b.len());
        let mut diff_at = None;
        for i in 0..n {
            if bitstream_a[i] != bitstream_b[i] {
                diff_at = Some(i);
                break;
            }
        }
        eprintln!(
            "ENTRY POINT BUG: cpu_len={}, gpu_len={}, first_diff_at={:?}",
            bitstream_a.len(),
            bitstream_b.len(),
            diff_at,
        );
        if let Some(i) = diff_at {
            let lo = i.saturating_sub(8);
            let hi = (i + 8).min(n);
            eprintln!(
                "encode_from_precomputed bytes [{lo}..{hi}]: {:02x?}",
                &bitstream_a[lo..hi]
            );
            eprintln!(
                "encode_from_pre_quantized_ac bytes [{lo}..{hi}]: {:02x?}",
                &bitstream_b[lo..hi]
            );
        }
    }
    assert_eq!(bitstream_a, bitstream_b);
}
