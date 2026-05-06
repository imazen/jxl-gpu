// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-block DCT8 AC dequantization with CfL restore.
//!
//! Mirrors `jxl_encoder_simd::dequant::dequant_dct8_scalar`.
//!
//! For each AC coefficient (i ≥ 1):
//!   `biased[c] = adjust_quant_bias(quant_ac[c][i], c)`
//!   `dequant[c][i] = biased[c] * weights[c][i] / qac_qm[c]`
//!   plus CfL restore: `X += x_factor * Y`, `B += b_factor * Y`
//!
//! DC (index 0) is forced to 0 in the output (caller restores LLF).

use cubecl::prelude::*;

const BIAS_X: f32 = 0.945_349_93;
const BIAS_Y: f32 = 0.929_945_5;
const BIAS_B: f32 = 0.950_064_9;
const BIAS_RECIP: f32 = 0.145;

#[cube]
fn adjust_quant_bias(q_int: i32, channel_bias: f32) -> f32 {
    if q_int == 0i32 {
        f32::new(0.0)
    } else {
        let q = q_int as f32;
        if f32::abs(q) < 1.125f32 {
            // sign(q) * channel_bias
            if q > 0.0f32 {
                channel_bias
            } else {
                -channel_bias
            }
        } else {
            q - BIAS_RECIP / q
        }
    }
}

/// One cube per block.
///
/// Layout (each array is `num_blocks * 64`):
/// - `quant_x/y/b`: i32 quantized AC coefficients per channel
/// - `weights_x/y/b`: f32 dequant weights per channel
///
/// Per-block (length `num_blocks`):
/// - `qac_qm_x`, `qac_qm_y`, `qac_qm_b`: per-block `qac * qm_mul`
/// - `x_factor`, `b_factor`: per-block CfL factors (host replicates from tile)
///
/// Output (each `num_blocks * 64`): `out_x/y/b` f32 dequantized coefficients.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn dequant_dct8_kernel(
    quant_x: &Array<i32>,
    quant_y: &Array<i32>,
    quant_b: &Array<i32>,
    weights_x: &Array<f32>,
    weights_y: &Array<f32>,
    weights_b: &Array<f32>,
    qac_qm_x: &Array<f32>,
    qac_qm_y: &Array<f32>,
    qac_qm_b: &Array<f32>,
    x_factor: &Array<f32>,
    b_factor: &Array<f32>,
    out_x: &mut Array<f32>,
    out_y: &mut Array<f32>,
    out_b: &mut Array<f32>,
) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = qac_qm_y.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;
    let inv_qx = 1.0f32 / qac_qm_x[block_idx];
    let inv_qy = 1.0f32 / qac_qm_y[block_idx];
    let inv_qb = 1.0f32 / qac_qm_b[block_idx];
    let xf = x_factor[block_idx];
    let bf = b_factor[block_idx];

    out_x[off] = f32::new(0.0);
    out_y[off] = f32::new(0.0);
    out_b[off] = f32::new(0.0);

    let mut i: u32 = 1u32;
    while i < 64u32 {
        let iu = i as usize;
        let bx = adjust_quant_bias(quant_x[off + iu], BIAS_X);
        let by = adjust_quant_bias(quant_y[off + iu], BIAS_Y);
        let bb = adjust_quant_bias(quant_b[off + iu], BIAS_B);

        let dq_y = by * weights_y[off + iu] * inv_qy;
        out_y[off + iu] = dq_y;
        out_x[off + iu] = bx * weights_x[off + iu] * inv_qx + xf * dq_y;
        out_b[off + iu] = bb * weights_b[off + iu] * inv_qb + bf * dq_y;
        i += 1u32;
    }
}
