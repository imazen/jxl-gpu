// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chroma-from-luma (CfL) per-tile multiplier search.
//!
//! Mirrors `jxl_encoder_simd::cfl::find_best_multiplier_scalar` —
//! regularized least-squares fit per tile, returning an integer multiplier
//! shifted towards zero with bias 2.6.
//!
//! Output is i32 (caller can clamp/cast to i8). Each tile has the same
//! `num` count and `distance_mul`, broadcast as scalars; per-tile `base`
//! comes through an array.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

const K_INV_COLOR_FACTOR: f32 = 1.0 / 84.0;
const TOWARDS_ZERO: f32 = 2.6;
const NEWTON_CLAMP: f32 = 20.0;
const NEWTON_COEFF: f32 = 1.0 / 3.0;
const NEWTON_THRES: f32 = 100.0;
const NEWTON_STABILIZER: f32 = 0.85;
const NEWTON_CONVERGENCE: f32 = 3e-3;

/// Bias towards zero and round to nearest integer, then clamp to [-128, 127].
#[cube]
fn bias_and_quantize(x: f32) -> i32 {
    let biased = if x >= TOWARDS_ZERO {
        x - TOWARDS_ZERO
    } else if x <= -TOWARDS_ZERO {
        x + TOWARDS_ZERO
    } else {
        f32::new(0.0)
    };
    // round half-away-from-zero (matches Rust f32::round, libjxl uses same here)
    let rounded = if biased >= 0.0f32 {
        (biased + 0.5f32) as i32
    } else {
        -((-biased + 0.5f32) as i32)
    };
    i32::clamp(rounded, -128i32, 127i32)
}

/// Newton's method variant with smoothed-L1 objective.
///
/// Mirrors `find_best_multiplier_newton_scalar`. Warm-starts from the LS
/// solution, refines via Newton iteration. Falls back to LS if Newton
/// doesn't converge within `max_iters`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_multiplier_newton_kernel(
    values_m: &Array<f32>,
    values_s: &Array<f32>,
    bases: &Array<f32>,
    output: &mut Array<i32>,
    num_per_tile: u32,
    distance_mul: f32,
    eps: f32,
    max_iters: u32,
) {
    let tile_idx = ABSOLUTE_POS;
    let n_tiles = bases.len();
    if tile_idx >= n_tiles {
        terminate!();
    }
    let n = num_per_tile as usize;
    if n == 0usize {
        output[tile_idx] = i32::new(0);
    } else {
        let off = tile_idx * n;
        let base = bases[tile_idx];
        let n_f = num_per_tile as f32;

        // LS warm start
        let mut sum_aa = f32::new(0.0);
        let mut sum_ab = f32::new(0.0);
        let mut i: u32 = 0u32;
        while (i as usize) < n {
            let iu = i as usize;
            let m = values_m[off + iu];
            let s = values_s[off + iu];
            let a = K_INV_COLOR_FACTOR * m;
            let b = base * m - s;
            sum_aa = sum_aa + a * a;
            sum_ab = sum_ab + a * b;
            i += 1u32;
        }
        let ls_x = -sum_ab / (sum_aa + n_f * distance_mul * 0.5f32);

        let coeffx2 = NEWTON_COEFF * 2.0f32;
        let mut x = ls_x;
        let mut converged = false;
        let mut iter: u32 = 0u32;
        while iter < max_iters && !converged {
            let mut fd = 2.0f32 * distance_mul * n_f * x;
            let mut fd_pe = 2.0f32 * distance_mul * n_f * (x + eps);
            let mut fd_me = 2.0f32 * distance_mul * n_f * (x - eps);

            let mut i: u32 = 0u32;
            while (i as usize) < n {
                let iu = i as usize;
                let m = values_m[off + iu];
                let s = values_s[off + iu];
                let a = K_INV_COLOR_FACTOR * m;
                let b = base * m - s;

                let v = a * x + b;
                let vpe = a * (x + eps) + b;
                let vme = a * (x - eps) + b;

                let av = f32::abs(v);
                let avpe = f32::abs(vpe);
                let avme = f32::abs(vme);

                let acoeffx2 = coeffx2 * a;

                let mut d = acoeffx2 * (av + 1.0f32);
                let mut dpe = acoeffx2 * (avpe + 1.0f32);
                let mut dme = acoeffx2 * (avme + 1.0f32);

                if v < 0.0f32 {
                    d = -d;
                }
                if vpe < 0.0f32 {
                    dpe = -dpe;
                }
                if vme < 0.0f32 {
                    dme = -dme;
                }

                if av < NEWTON_THRES {
                    fd = fd + d;
                }
                if avpe < NEWTON_THRES {
                    fd_pe = fd_pe + dpe;
                }
                if avme < NEWTON_THRES {
                    fd_me = fd_me + dme;
                }
                i += 1u32;
            }

            let ddf = (fd_pe - fd_me) / (2.0f32 * eps);
            let step = fd / (ddf + NEWTON_STABILIZER);
            let step_clamped = f32::clamp(step, -NEWTON_CLAMP, NEWTON_CLAMP);
            x = x - step_clamped;
            if f32::abs(step) < NEWTON_CONVERGENCE {
                converged = true;
            }
            iter += 1u32;
        }

        let final_x = if converged { x } else { ls_x };
        output[tile_idx] = bias_and_quantize(final_x);
    }
}

/// One cube per tile. Inputs:
/// - `values_m`: `num_tiles * num_per_tile` m values
/// - `values_s`: same shape, s values
/// - `bases`: per-tile base value (num_tiles)
/// - `output`: per-tile result (num_tiles, i32 — cast to i8 on host)
#[cube(launch_unchecked)]
pub fn find_best_multiplier_kernel(
    values_m: &Array<f32>,
    values_s: &Array<f32>,
    bases: &Array<f32>,
    output: &mut Array<i32>,
    num_per_tile: u32,
    distance_mul: f32,
) {
    let tile_idx = ABSOLUTE_POS;
    let n_tiles = bases.len();
    if tile_idx >= n_tiles {
        terminate!();
    }
    let n = num_per_tile as usize;
    if n == 0usize {
        output[tile_idx] = i32::new(0);
    } else {
        let off = tile_idx * n;
        let base = bases[tile_idx];
        let mut sum_aa = f32::new(0.0);
        let mut sum_ab = f32::new(0.0);
        let mut i: u32 = 0u32;
        while (i as usize) < n {
            let iu = i as usize;
            let m = values_m[off + iu];
            let s = values_s[off + iu];
            let a = K_INV_COLOR_FACTOR * m;
            let b = base * m - s;
            sum_aa = sum_aa + a * a;
            sum_ab = sum_ab + a * b;
            i += 1u32;
        }
        let n_f = num_per_tile as f32;
        let x = -sum_ab / (sum_aa + n_f * distance_mul * 0.5f32);
        output[tile_idx] = bias_and_quantize(x);
    }
}
