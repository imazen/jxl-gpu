// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Fused 3-channel pixel-domain entropy kernel.
//!
//! Mirrors `entropy_coeffs_pixel_kernel_broadcast_w` × 3 fused into
//! one launch. Y / X / B channels share `block_y` (the luma coefs)
//! but each has its own block_c, weights, inv_weights, cmap_factor,
//! quant and accumulators. Saves 2 launches per cost-grid call.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_pixel_3ch_kernel(
    block_x: &Array<f32>,
    block_y: &Array<f32>,
    block_b: &Array<f32>,
    weights_x: &Array<f32>,
    weights_y: &Array<f32>,
    weights_b: &Array<f32>,
    inv_weights_x: &Array<f32>,
    inv_weights_y: &Array<f32>,
    inv_weights_b: &Array<f32>,
    error_x: &mut Array<f32>,
    error_y: &mut Array<f32>,
    error_b: &mut Array<f32>,
    out_x: &mut Array<f32>,
    out_y: &mut Array<f32>,
    out_b: &mut Array<f32>,
    n: u32,
    cmap_factor_x: f32,
    cmap_factor_b: f32,
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    k_cost_delta: f32,
) {
    let block_idx = ABSOLUTE_POS;
    let n_per = n as usize;
    let n_blocks = block_y.len() / n_per;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * n_per;
    let oo = block_idx * 4usize;

    let mut e_x = f32::new(0.0);
    let mut nz_x = f32::new(0.0);
    let mut e_y = f32::new(0.0);
    let mut nz_y = f32::new(0.0);
    let mut e_b = f32::new(0.0);
    let mut nz_b = f32::new(0.0);
    let mut i: u32 = 0u32;
    while (i as usize) < n_per {
        let iu = i as usize;
        let yc = block_y[off + iu];
        // Y channel (cmap_factor = 0 always for Y).
        let val_y = yc * inv_weights_y[iu] * quant_y;
        let r_y = f32::round(val_y);
        let d_y = val_y - r_y;
        error_y[off + iu] = weights_y[iu] * d_y;
        let q_y = f32::abs(r_y);
        e_y = e_y + f32::sqrt(q_y) * k_cost_delta;
        if q_y != 0.0f32 {
            nz_y = nz_y + 1.0f32;
        }
        // X channel.
        let xc = block_x[off + iu];
        let val_x = (xc - yc * cmap_factor_x) * inv_weights_x[iu] * quant_x;
        let r_x = f32::round(val_x);
        let d_x = val_x - r_x;
        error_x[off + iu] = weights_x[iu] * d_x;
        let q_x = f32::abs(r_x);
        e_x = e_x + f32::sqrt(q_x) * k_cost_delta;
        if q_x != 0.0f32 {
            nz_x = nz_x + 1.0f32;
        }
        // B channel.
        let bc = block_b[off + iu];
        let val_b = (bc - yc * cmap_factor_b) * inv_weights_b[iu] * quant_b;
        let r_b = f32::round(val_b);
        let d_b = val_b - r_b;
        error_b[off + iu] = weights_b[iu] * d_b;
        let q_b = f32::abs(r_b);
        e_b = e_b + f32::sqrt(q_b) * k_cost_delta;
        if q_b != 0.0f32 {
            nz_b = nz_b + 1.0f32;
        }
        i += 1u32;
    }

    out_x[oo] = e_x;
    out_x[oo + 1usize] = nz_x;
    out_x[oo + 2usize] = f32::new(0.0);
    out_x[oo + 3usize] = f32::new(0.0);
    out_y[oo] = e_y;
    out_y[oo + 1usize] = nz_y;
    out_y[oo + 2usize] = f32::new(0.0);
    out_y[oo + 3usize] = f32::new(0.0);
    out_b[oo] = e_b;
    out_b[oo + 1usize] = nz_b;
    out_b[oo + 2usize] = f32::new(0.0);
    out_b[oo + 3usize] = f32::new(0.0);
}
