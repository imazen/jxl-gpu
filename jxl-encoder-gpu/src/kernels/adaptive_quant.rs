// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-block adaptive-quantization modulations.
//!
//! Mirrors `jxl_encoder_simd::adaptive_quant::per_block_modulations_scalar`.
//!
//! Per 8x8 block: ComputeMask → GammaModulation → min(HfModulation,
//! BlueModulation) → exp2(out * log2(e)) * mul + add.

#![allow(clippy::assign_op_pattern)]
// CPU-reference constants must be bit-matched even when they encode more
// precision than f32 holds (the literal rounding is identical on both sides).
#![allow(clippy::excessive_precision)]

use cubecl::prelude::*;

// ---- Constants (must match jxl_encoder_simd::adaptive_quant) ----

#[allow(clippy::excessive_precision)]
const SG_MUL: f32 = 226.772_16;
#[allow(clippy::excessive_precision)]
const SG_MUL2: f32 = 1.0 / 73.377_13;
const K_INV_LOG2E: f32 = core::f32::consts::LN_2;
#[allow(clippy::excessive_precision)]
const SG_RET_MUL: f32 = SG_MUL2 * 18.658_092 * K_INV_LOG2E;
#[allow(clippy::excessive_precision)]
const SG_V_OFFSET: f32 = 7.782_599;
const EPSILON: f32 = 1e-2;
const K_NUM_MUL: f32 = SG_RET_MUL * 3.0 * SG_MUL;
const K_V_OFFSET: f32 = SG_V_OFFSET * K_INV_LOG2E + EPSILON;
const K_DEN_MUL: f32 = K_INV_LOG2E * SG_MUL;

#[allow(clippy::excessive_precision)]
const CM_K_BASE: f32 = -0.764_7;
#[allow(clippy::excessive_precision)]
const CM_K_MUL4: f32 = 9.470_874;
#[allow(clippy::excessive_precision)]
const CM_K_MUL2: f32 = 17.350_366;
#[allow(clippy::excessive_precision)]
const CM_K_OFFSET2: f32 = 302.595_88;
#[allow(clippy::excessive_precision)]
const CM_K_MUL3: f32 = 6.794_325;
#[allow(clippy::excessive_precision)]
const CM_K_OFFSET3: f32 = 3.717_963_6;
const CM_K_OFFSET4: f32 = 0.25 * CM_K_OFFSET3;
#[allow(clippy::excessive_precision)]
const CM_K_MUL0: f32 = 0.800_617_6;

const GM_K_BIAS: f32 = 0.16;
#[allow(clippy::excessive_precision)]
const GM_K_GAMMA: f32 = 0.100_561_336;

const HF_VALMIN_Y: f32 = 0.0206;
const HF_K_MUL_Y: f32 = -0.38;
const HF_K_OFFSET: f32 = 0.42;

#[allow(clippy::excessive_precision)]
const BM_K_LIMIT: f32 = 0.010_474_085;
#[allow(clippy::excessive_precision)]
const BM_K_OFFSET: f32 = 0.003_199_476_8;
#[allow(clippy::excessive_precision)]
const BM_K_MAX_LIMIT: f32 = 15.463_398;
#[allow(clippy::excessive_precision)]
const BM_K_MUL: f32 = 0.905_908_05;

const LOG2_P0: f32 = -1.850_383_3e-6;
const LOG2_P1: f32 = 1.428_716;
const LOG2_P2: f32 = 0.742_458_7;
const LOG2_Q0: f32 = 0.990_328_14;
const LOG2_Q1: f32 = 1.009_671_9;
const LOG2_Q2: f32 = 0.174_093_43;
const FAST_LOG2_BIAS: u32 = 0x3f2a_aaab;

// compute_pre_erosion constants
const MATCH_GAMMA_OFFSET: f32 = 0.019;
const LIMIT: f32 = 0.2;
#[allow(clippy::excessive_precision)]
const MASKING_K_LOG_OFFSET: f32 = 27.505_837;
#[allow(clippy::excessive_precision)]
const MASKING_K_MUL: f32 = 211.665_68;

// fast_pow2f polynomial coefficients
const POW2_P0: f32 = 9.855_065_91e+01;
const POW2_P1: f32 = 4.886_877_98e+01;
const POW2_P2: f32 = 1.017_490_63e+01;
const POW2_Q0: f32 = 9.855_066_33e+01;
#[allow(clippy::excessive_precision)]
const POW2_Q1: f32 = -1.944_149_9e+01;
#[allow(clippy::excessive_precision)]
const POW2_Q2: f32 = -2.223_288_56e-02;
#[allow(clippy::excessive_precision)]
const POW2_Q3: f32 = 2.102_429_58e-01;

// ---- Helper functions ----

#[cube]
fn fast_log2f(x: f32) -> f32 {
    let x_bits = u32::reinterpret(x) as i32;
    let exp_bits = x_bits - (FAST_LOG2_BIAS as i32);
    let exp_shifted = exp_bits >> 23i32;
    let mantissa_bits = x_bits - (exp_shifted << 23i32);
    let mantissa = f32::reinterpret(mantissa_bits as u32);
    let frac = mantissa - 1.0f32;
    let num = LOG2_P0 + frac * (LOG2_P1 + frac * LOG2_P2);
    let den = LOG2_Q0 + frac * (LOG2_Q1 + frac * LOG2_Q2);
    num / den + (exp_shifted as f32)
}

#[cube]
fn fast_pow2f(x: f32) -> f32 {
    let floorx = f32::floor(x);
    let exp_bits = (((floorx as i32) + 127i32) << 23i32) as u32;
    let exp = f32::reinterpret(exp_bits);
    let frac = x - floorx;
    let num = ((frac + POW2_P2) * frac + POW2_P1) * frac + POW2_P0;
    let num = num * exp;
    let den = ((frac * POW2_Q3 + POW2_Q2) * frac + POW2_Q1) * frac + POW2_Q0;
    num / den
}

#[cube]
fn ratio_of_deriv_inverted(v_in: f32) -> f32 {
    let v = f32::max(v_in, 0.0f32);
    let v2 = v * v;
    let num = K_NUM_MUL * v2 + EPSILON;
    let den = K_DEN_MUL * v * v2 + K_V_OFFSET;
    num / den
}

/// `ratio_of_deriv_normal` — returns `den/num` (opposite of inverted).
#[cube]
fn ratio_of_deriv_normal(v_in: f32) -> f32 {
    let v = f32::max(v_in, 0.0f32);
    let v2 = v * v;
    let num = K_NUM_MUL * v2 + EPSILON;
    let den = K_DEN_MUL * v * v2 + K_V_OFFSET;
    den / num
}

#[cube]
fn masking_sqrt(v: f32) -> f32 {
    let mul_v = MASKING_K_MUL * 1e8f32;
    0.25f32 * f32::sqrt(v * f32::sqrt(mul_v) + MASKING_K_LOG_OFFSET)
}

#[cube]
fn compute_mask(out_val: f32) -> f32 {
    let v1 = f32::max(out_val * CM_K_MUL0, 1e-3f32);
    let v2 = 1.0f32 / (v1 + CM_K_OFFSET2);
    let v3 = 1.0f32 / (v1 * v1 + CM_K_OFFSET3);
    let v4 = 1.0f32 / (v1 * v1 + CM_K_OFFSET4);
    CM_K_BASE + CM_K_MUL4 * v4 + CM_K_MUL2 * v2 + CM_K_MUL3 * v3
}

/// Compute pre-erosion map. One thread per OUTPUT pixel.
///
/// Mirrors `jxl_encoder_simd::adaptive_quant::compute_pre_erosion_scalar`,
/// but with the variable output size handled by caller pre-allocation.
/// Caller computes `pre_erosion_w` and `pre_erosion_h` from tile bounds:
///   `x0 = saturating_sub(tile_x0, 4); x1 = if tile_x1 < width { tile_x1 + 4 } else { tile_x1 };`
///   `y_start = saturating_sub(tile_y0, 4); y_end = if tile_y1 < height { tile_y1 + 4 } else { tile_y1 };`
///   `pre_erosion_w = (x1 - x0) / 4; pre_erosion_h = (y_end - y_start) / 4;`
///
/// Each output pixel sums 16 source diffs (4x4 area) and multiplies by 0.25
/// (matches CPU: 4 column-sums of 4 row-diffs each, then *0.25).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn compute_pre_erosion_kernel(
    xyb_y: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    x0: u32,
    y_start: u32,
    pre_erosion_w: u32,
) {
    let oidx = ABSOLUTE_POS;
    let pew = pre_erosion_w as usize;
    let total = output.len();
    if oidx >= total {
        terminate!();
    }
    let out_y = oidx / pew;
    let out_x = oidx - out_y * pew;
    let w = width as usize;
    let h = height as usize;
    let max_x = w - 1usize;
    let max_y = h - 1usize;

    let base_x = (x0 as usize) + out_x * 4usize;
    let base_y = (y_start as usize) + out_y * 4usize;

    let mut sum = f32::new(0.0);
    let mut dy: u32 = 0u32;
    while dy < 4u32 {
        let dyu = dy as usize;
        let y = base_y + dyu;
        let yc = usize::min(y, max_y);
        let y2 = usize::min(y + 1usize, max_y);
        let y1 = usize::min(usize::saturating_sub(y, 1usize), max_y);

        let mut dx: u32 = 0u32;
        while dx < 4u32 {
            let dxu = dx as usize;
            let x = base_x + dxu;
            let xc = usize::min(x, max_x);
            let x2 = usize::min(x + 1usize, max_x);
            let x1 = usize::min(usize::saturating_sub(x, 1usize), max_x);

            let base = 0.25f32
                * (xyb_y[y2 * w + xc]
                    + xyb_y[y1 * w + xc]
                    + xyb_y[yc * w + x1]
                    + xyb_y[yc * w + x2]);

            let gammac = ratio_of_deriv_normal(xyb_y[yc * w + xc] + MATCH_GAMMA_OFFSET);
            let mut diff = gammac * (xyb_y[yc * w + xc] - base);
            diff = diff * diff;
            if diff >= LIMIT {
                diff = LIMIT;
            }
            diff = masking_sqrt(diff);
            sum = sum + diff;
            dx += 1u32;
        }
        dy += 1u32;
    }

    output[oidx] = sum * 0.25f32;
}

/// Per-block modulations kernel. One cube per block in the rect.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn per_block_modulations_kernel(
    xyb_x: &Array<f32>,
    xyb_y: &Array<f32>,
    xyb_b: &Array<f32>,
    aq_map: &mut Array<f32>,
    stride: u32,
    aq_map_stride: u32,
    rect_x0_blocks: u32,
    rect_y0_blocks: u32,
    rect_w_blocks: u32,
    butteraugli_target: f32,
    scale: f32,
) {
    let block_idx = ABSOLUTE_POS;
    let rw = rect_w_blocks as usize;
    let total = rw * (aq_map.len() / (aq_map_stride as usize));
    // total isn't quite right when rect_h_blocks * aq_map_stride != aq_map.len()
    // but we pass rect_h_blocks via a trailing terminate guard via total threads.
    let _ = total;
    let n_threads = aq_map.len();
    if block_idx >= n_threads {
        terminate!();
    }
    let iy = block_idx / rw;
    let ix = block_idx - iy * rw;
    // Skip threads beyond the actual rect (caller passes ceil(n) threads).
    let aq_stride = aq_map_stride as usize;
    if iy * aq_stride + ix >= aq_map.len() {
        terminate!();
    }

    // Compute base_level/dampen/mul/add (could be scalars but cubecl
    // constants don't easily survive into kernel — recompute per thread).
    let base_level = 0.48f32 * scale;
    let k_dampen_ramp_start = 2.0f32;
    let k_dampen_ramp_end = 14.0f32;
    let dampen = if butteraugli_target >= k_dampen_ramp_start {
        let raw = 1.0f32
            - ((butteraugli_target - k_dampen_ramp_start)
                / (k_dampen_ramp_end - k_dampen_ramp_start));
        f32::max(raw, 0.0f32)
    } else {
        f32::new(1.0)
    };
    let mul = scale * dampen;
    let add = (1.0f32 - dampen) * base_level;

    // Block coordinates in xyb planes (8x8 block at top-left px, py).
    let block_iy = (rect_y0_blocks as usize) + iy;
    let block_ix = (rect_x0_blocks as usize) + ix;
    let py = block_iy * 8usize;
    let px = block_ix * 8usize;
    let s = stride as usize;

    let aq_idx = iy * aq_stride + ix;
    let mut out_val = aq_map[aq_idx];

    // ---- ComputeMask ----
    out_val = compute_mask(out_val);

    // ---- GammaModulation ----
    {
        let mut overall_ratio = f32::new(0.0);
        let mut dy: u32 = 0u32;
        while dy < 8u32 {
            let dyu = dy as usize;
            let mut dx: u32 = 0u32;
            while dx < 8u32 {
                let dxu = dx as usize;
                let idx = (py + dyu) * s + (px + dxu);
                let iny = xyb_y[idx] + GM_K_BIAS;
                let inx = xyb_x[idx];
                overall_ratio = overall_ratio
                    + ratio_of_deriv_inverted(iny - inx)
                    + ratio_of_deriv_inverted(iny + inx);
                dx += 1u32;
            }
            dy += 1u32;
        }
        overall_ratio = overall_ratio * (0.5f32 / 64.0f32);
        out_val = out_val + GM_K_GAMMA * fast_log2f(overall_ratio);
    }

    let mask_val = out_val;

    // ---- HfModulation ----
    let after_hf = {
        let mut sum_y = f32::new(0.0);
        let mut dy: u32 = 0u32;
        while dy < 8u32 {
            let dyu = dy as usize;
            let py_next = if dy == 7u32 {
                py + dyu
            } else {
                py + dyu + 1usize
            };
            let mut dx: u32 = 0u32;
            while dx < 8u32 {
                let dxu = dx as usize;
                let p_y = xyb_y[(py + dyu) * s + (px + dxu)];
                if dx < 7u32 {
                    let nx = xyb_y[(py + dyu) * s + (px + dxu + 1usize)];
                    sum_y = sum_y + f32::min(f32::abs(p_y - nx), HF_VALMIN_Y);
                }
                let ny = xyb_y[py_next * s + (px + dxu)];
                sum_y = sum_y + f32::min(f32::abs(p_y - ny), HF_VALMIN_Y);
                dx += 1u32;
            }
            dy += 1u32;
        }
        mask_val + sum_y * HF_K_MUL_Y + HF_K_OFFSET
    };

    // ---- BlueModulation ----
    let after_blue = {
        let mut sum = f32::new(0.0);
        let mut dy: u32 = 0u32;
        while dy < 8u32 {
            let dyu = dy as usize;
            let mut dx: u32 = 0u32;
            while dx < 8u32 {
                let dxu = dx as usize;
                let idx = (py + dyu) * s + (px + dxu);
                let p_x = xyb_x[idx];
                let p_b = xyb_b[idx];
                let p_y_eff = xyb_y[idx] + BM_K_OFFSET + f32::abs(p_x);
                if p_b > p_y_eff {
                    sum = sum + f32::min(p_b - p_y_eff, BM_K_LIMIT);
                }
                dx += 1u32;
            }
            dy += 1u32;
        }
        if sum >= 32.0f32 * BM_K_LIMIT {
            sum = 64.0f32 * BM_K_LIMIT - sum;
        }
        if sum >= BM_K_MAX_LIMIT * BM_K_LIMIT {
            sum = BM_K_MAX_LIMIT * BM_K_LIMIT;
        }
        sum = sum * BM_K_MUL;
        mask_val + sum
    };

    out_val = f32::min(after_hf, after_blue);
    aq_map[aq_idx] = fast_pow2f(out_val * core::f32::consts::LOG2_E) * mul + add;
}
