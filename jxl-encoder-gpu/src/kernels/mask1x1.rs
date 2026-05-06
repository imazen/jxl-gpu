// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-pixel masking field from XYB Y channel.
//!
//! Mirrors `jxl_encoder_simd::mask1x1::compute_mask1x1_scalar`.
//!
//! For each pixel: 4-neighbor average → ratio_of_derivatives →
//! |gammac * (pixel - base)| → ln(1 + diff) → K_MUL / (diff + K_OFFSET).

use cubecl::prelude::*;

const MATCH_GAMMA_OFFSET: f32 = 0.019;
const K_MUL: f32 = 1.0;
const K_OFFSET: f32 = 0.01;

const SG_MUL: f32 = 226.772_16;
const SG_MUL2: f32 = 1.0 / 73.377_13;
const K_INV_LOG2E: f32 = core::f32::consts::LN_2;
const SG_RET_MUL: f32 = SG_MUL2 * 18.658_092 * K_INV_LOG2E;
const SG_V_OFFSET: f32 = 7.782_599;

const EPSILON: f32 = 1e-2;
const K_NUM_MUL: f32 = SG_RET_MUL * 3.0 * SG_MUL;
const K_V_OFFSET: f32 = SG_V_OFFSET * K_INV_LOG2E + EPSILON;
const K_DEN_MUL: f32 = K_INV_LOG2E * SG_MUL;

const LN2: f32 = K_INV_LOG2E;

const LOG2_P0: f32 = -1.850_383_3e-6;
const LOG2_P1: f32 = 1.428_716;
const LOG2_P2: f32 = 0.742_458_7;
const LOG2_Q0: f32 = 0.990_328_14;
const LOG2_Q1: f32 = 1.009_671_9;
const LOG2_Q2: f32 = 0.174_093_43;

/// Magic constant matching the CPU's exponent-bias adjustment.
const FAST_LOG2_BIAS: u32 = 0x3f2a_aaab;

/// Fast log2 approximation. Input must be > 0. Max relative error ~3e-7.
///
/// Uses `i32` raw subtract (wrapping by default in 2's complement) for
/// the bit-hack steps because cubecl 0.10 does not register
/// `u32::wrapping_sub` (gotcha catalogue, addendum to G1.4).
#[cube]
fn fast_log2f(x: f32) -> f32 {
    let x_bits = u32::reinterpret(x) as i32;
    let exp_bits = x_bits - (FAST_LOG2_BIAS as i32);
    let exp_shifted = exp_bits >> 23i32;
    let mantissa_bits = x_bits - (exp_shifted << 23i32);
    let mantissa = f32::reinterpret(mantissa_bits as u32);
    let exp_val = exp_shifted as f32;
    let frac = mantissa - 1.0f32;
    let num = LOG2_P0 + frac * (LOG2_P1 + frac * LOG2_P2);
    let den = LOG2_Q0 + frac * (LOG2_Q1 + frac * LOG2_Q2);
    num / den + exp_val
}

#[cube]
fn ratio_of_derivatives(v_in: f32) -> f32 {
    let v = f32::max(v_in, 0.0f32);
    let v2 = v * v;
    let num = K_NUM_MUL * v2 + EPSILON;
    let den = K_DEN_MUL * v * v2 + K_V_OFFSET;
    den / num
}

/// Compute per-pixel masking field. One thread per pixel.
#[cube(launch_unchecked)]
pub fn mask1x1_kernel(xyb_y: &Array<f32>, output: &mut Array<f32>, width: u32, height: u32) {
    let idx = ABSOLUTE_POS;
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if idx >= n {
        terminate!();
    }
    let y = idx / w;
    let x = idx - y * w;

    let y1 = usize::saturating_sub(y, 1usize);
    let y2 = usize::min(y + 1usize, h - 1usize);
    let x1 = usize::saturating_sub(x, 1usize);
    let x2 = usize::min(x + 1usize, w - 1usize);

    let base =
        0.25f32 * (xyb_y[y1 * w + x] + xyb_y[y2 * w + x] + xyb_y[y * w + x1] + xyb_y[y * w + x2]);

    let pixel_val = xyb_y[idx];
    let gammac = ratio_of_derivatives(pixel_val + MATCH_GAMMA_OFFSET);

    let diff_val = f32::abs(gammac * (pixel_val - base));
    let diff = fast_log2f(1.0f32 + diff_val) * LN2;

    output[idx] = K_MUL / (diff + K_OFFSET);
}
