// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! XYB ↔ Linear RGB color conversion kernels.
//!
//! Mirrors `jxl_encoder_simd::xyb::{forward_xyb_scalar, inverse_xyb_planar_scalar}`.
//!
//! Forward (linear RGB → XYB): chained-FMA matrix multiply + bias →
//! clamp non-negative → cube root via Newton-Raphson → mix to XYB.
//!
//! Cube root precision note: the CPU reference uses Newton-Raphson in f64
//! with a bit-manipulation initial guess. We use the same algorithm but in
//! f32 (cubecl 0.10 does not register `f64` reliably across all backends);
//! 2 Newton iterations in f32 from a 5%-error initial guess converge to
//! within ~1 ulp of the f64 reference, well inside the `< 1e-6` parity
//! target for unit-range data.

use cubecl::prelude::*;

// --- Constants (must match jxl_encoder_simd::xyb) ---

const OPSIN_M00: f32 = 0.30;
const OPSIN_M01: f32 = 0.622;
const OPSIN_M02: f32 = 0.078;
const OPSIN_M10: f32 = 0.23;
const OPSIN_M11: f32 = 0.692;
const OPSIN_M12: f32 = 0.078;
const OPSIN_M20: f32 = 0.243_422_69;
const OPSIN_M21: f32 = 0.204_767_45;
const OPSIN_M22: f32 = 0.551_809_87;

// `excessive_precision` is intentional — these constants must bit-match
// the CPU reference (`jxl_encoder_simd::xyb`) for the parity-test target.
#[allow(clippy::excessive_precision)]
const INV_OPSIN_00: f32 = 11.031_566_9;
const INV_OPSIN_01: f32 = -9.866_944;
const INV_OPSIN_02: f32 = -0.164_623;
#[allow(clippy::excessive_precision)]
const INV_OPSIN_10: f32 = -3.254_147_4;
const INV_OPSIN_11: f32 = 4.418_770_4;
const INV_OPSIN_12: f32 = -0.164_623;
#[allow(clippy::excessive_precision)]
const INV_OPSIN_20: f32 = -3.658_851_3;
const INV_OPSIN_21: f32 = 2.712_923;
const INV_OPSIN_22: f32 = 1.945_928_2;

#[allow(clippy::excessive_precision)]
const OPSIN_BIAS: f32 = 0.003_793_073_4;
const NEG_CBRT_BIAS: f32 = -0.155_954_2;

// Magic constant for the initial cube-root guess via bit manipulation
// (Sun Microsystems / fdlibm `cbrtf`). Yields ~5% relative error; refined
// with two Newton iterations.
const CBRT_INITIAL_BIAS_U32: u32 = 709_958_130;

/// Newton-Raphson cube root for non-negative `x`.
///
/// Caller is responsible for `x >= 0` (e.g. via `f32::max(_, 0.0)`).
/// Single-expression form (no early return — cubecl 0.10 doesn't support
/// `return` inside `#[cube]` bodies).
#[cube]
fn cbrt_fast_nonneg(x: f32) -> f32 {
    // Initial guess: divide exponent by 3 (approximately) via bit hack.
    // Assumes x > 0 so the sign bit is 0. For x == 0, the divide produces
    // CBRT_INITIAL_BIAS_U32 — a tiny positive number; the Newton iteration
    // then converges to 0 because the numerator (2x + t³) is dominated by
    // t³ which collapses toward 0 as x = 0.
    //
    // To avoid 0/0 issues we substitute a tiny epsilon when x is exactly 0.
    let safe_x = if x > 0.0f32 { x } else { f32::new(1e-30) };
    let ui = u32::reinterpret(safe_x);
    let approx = ui / 3u32 + CBRT_INITIAL_BIAS_U32;
    let mut t = f32::reinterpret(approx);
    let r = t * t * t;
    t = t * (safe_x + safe_x + r) / (safe_x + r + r);
    let r = t * t * t;
    t = t * (safe_x + safe_x + r) / (safe_x + r + r);
    // Force exact 0 result for exact 0 input (matches CPU's early-return path).
    if x > 0.0f32 { t } else { f32::new(0.0) }
}

/// Forward: planar linear RGB → planar XYB. One thread per pixel.
///
/// Mirrors `jxl_encoder_simd::xyb::forward_xyb_scalar`.
///
/// Launch with `cube_count = ceil(n / 256)`, `cube_dim = (256, 1, 1)`.
#[cube(launch_unchecked)]
pub fn xyb_forward_kernel(
    r: &Array<f32>,
    g: &Array<f32>,
    b: &Array<f32>,
    x_out: &mut Array<f32>,
    y_out: &mut Array<f32>,
    b_out: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= x_out.len() {
        terminate!();
    }

    let ri = r[idx];
    let gi = g[idx];
    let bi = b[idx];

    // Chained-FMA matrix multiply + bias (matches scalar reference's
    // `mul_add` ordering for single-rounding parity).
    let mixed0 = OPSIN_M00 * ri + (OPSIN_M01 * gi + (OPSIN_M02 * bi + OPSIN_BIAS));
    let mixed1 = OPSIN_M10 * ri + (OPSIN_M11 * gi + (OPSIN_M12 * bi + OPSIN_BIAS));
    let mixed2 = OPSIN_M20 * ri + (OPSIN_M21 * gi + (OPSIN_M22 * bi + OPSIN_BIAS));

    let l = cbrt_fast_nonneg(f32::max(mixed0, 0.0f32)) + NEG_CBRT_BIAS;
    let m = cbrt_fast_nonneg(f32::max(mixed1, 0.0f32)) + NEG_CBRT_BIAS;
    let s = cbrt_fast_nonneg(f32::max(mixed2, 0.0f32)) + NEG_CBRT_BIAS;

    x_out[idx] = 0.5f32 * (l - m);
    y_out[idx] = 0.5f32 * (l + m);
    b_out[idx] = s;
}

/// Inverse: planar XYB → planar linear RGB. One thread per pixel.
///
/// Mirrors `jxl_encoder_simd::xyb::inverse_xyb_planar_scalar`.
#[cube(launch_unchecked)]
pub fn xyb_inverse_kernel(
    xyb_x: &Array<f32>,
    xyb_y: &Array<f32>,
    xyb_b: &Array<f32>,
    out_r: &mut Array<f32>,
    out_g: &mut Array<f32>,
    out_b: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= out_r.len() {
        terminate!();
    }

    let x = xyb_x[idx];
    let y = xyb_y[idx];
    let b = xyb_b[idx];

    let gamma_r = y + x - NEG_CBRT_BIAS;
    let gamma_g = y - x - NEG_CBRT_BIAS;
    let gamma_b = b - NEG_CBRT_BIAS;

    let mixed_r = gamma_r * gamma_r * gamma_r - OPSIN_BIAS;
    let mixed_g = gamma_g * gamma_g * gamma_g - OPSIN_BIAS;
    let mixed_b = gamma_b * gamma_b * gamma_b - OPSIN_BIAS;

    out_r[idx] = INV_OPSIN_00 * mixed_r + (INV_OPSIN_01 * mixed_g + INV_OPSIN_02 * mixed_b);
    out_g[idx] = INV_OPSIN_10 * mixed_r + (INV_OPSIN_11 * mixed_g + INV_OPSIN_12 * mixed_b);
    out_b[idx] = INV_OPSIN_20 * mixed_r + (INV_OPSIN_21 * mixed_g + INV_OPSIN_22 * mixed_b);
}
