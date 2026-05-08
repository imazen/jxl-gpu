// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Entropy coefficient estimation per block.
//!
//! Mirrors `jxl_encoder_simd::entropy::entropy_coeffs_scalar`. Per
//! coefficient: `val = (block_c - block_y * cmap_factor) * inv_weights *
//! quant`; round and accumulate entropy via `sqrt(|round|) * k_cost_delta`,
//! count non-zeros, and optionally compute info_loss / info_loss2 in
//! coefficient-domain mode.
//!
//! Two kernels (pixel_domain bool can't be a comptime generic per
//! cubecl 0.10 G1.7): `entropy_coeffs_pixel_kernel` and
//! `entropy_coeffs_coeff_kernel`. Output per block is 4 f32:
//! [entropy_sum, nzeros_sum, info_loss_sum, info_loss2_sum].

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

/// Pixel-domain mode: writes `error_coeffs[i] = weights[i] * diff`,
/// skips info_loss accumulation. `info_loss_sum` / `info_loss2_sum`
/// are 0.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_pixel_kernel(
    block_c: &Array<f32>,
    block_y: &Array<f32>,
    weights: &Array<f32>,
    inv_weights: &Array<f32>,
    error_coeffs: &mut Array<f32>,
    output: &mut Array<f32>, // num_blocks * 4
    n: u32,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
) {
    let block_idx = ABSOLUTE_POS;
    let n_per = n as usize;
    let n_blocks = block_c.len() / n_per;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * n_per;
    let oo = block_idx * 4usize;

    let mut entropy_sum = f32::new(0.0);
    let mut nzeros_sum = f32::new(0.0);
    let mut i: u32 = 0u32;
    while (i as usize) < n_per {
        let iu = i as usize;
        let val_in = block_c[off + iu];
        let val_y = block_y[off + iu] * cmap_factor;
        let val = (val_in - val_y) * inv_weights[off + iu] * quant;
        let rval = f32::round(val);
        let diff = val - rval;
        error_coeffs[off + iu] = weights[off + iu] * diff;
        let q = f32::abs(rval);
        entropy_sum = entropy_sum + f32::sqrt(q) * k_cost_delta;
        if q != 0.0f32 {
            nzeros_sum = nzeros_sum + 1.0f32;
        }
        i += 1u32;
    }

    output[oo] = entropy_sum;
    output[oo + 1usize] = nzeros_sum;
    output[oo + 2usize] = f32::new(0.0);
    output[oo + 3usize] = f32::new(0.0);
}

/// Broadcast-weights variant of [`entropy_coeffs_pixel_kernel`].
/// `weights` and `inv_weights` are each exactly `n` f32 (one quant
/// matrix and its inverse), broadcast across all blocks. Saves
/// `2 * (num_blocks - 1) * n * 4` bytes of upload traffic — twice
/// the savings of the dequant variants because this kernel reads
/// both forward AND inverse weights.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_pixel_kernel_broadcast_w(
    block_c: &Array<f32>,
    block_y: &Array<f32>,
    weights: &Array<f32>,
    inv_weights: &Array<f32>,
    error_coeffs: &mut Array<f32>,
    output: &mut Array<f32>,
    n: u32,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
) {
    let block_idx = ABSOLUTE_POS;
    let n_per = n as usize;
    let n_blocks = block_c.len() / n_per;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * n_per;
    let oo = block_idx * 4usize;

    let mut entropy_sum = f32::new(0.0);
    let mut nzeros_sum = f32::new(0.0);
    let mut i: u32 = 0u32;
    while (i as usize) < n_per {
        let iu = i as usize;
        let val_in = block_c[off + iu];
        let val_y = block_y[off + iu] * cmap_factor;
        // Broadcast: weights[iu] / inv_weights[iu] not [off + iu].
        let val = (val_in - val_y) * inv_weights[iu] * quant;
        let rval = f32::round(val);
        let diff = val - rval;
        error_coeffs[off + iu] = weights[iu] * diff;
        let q = f32::abs(rval);
        entropy_sum = entropy_sum + f32::sqrt(q) * k_cost_delta;
        if q != 0.0f32 {
            nzeros_sum = nzeros_sum + 1.0f32;
        }
        i += 1u32;
    }

    output[oo] = entropy_sum;
    output[oo + 1usize] = nzeros_sum;
    output[oo + 2usize] = f32::new(0.0);
    output[oo + 3usize] = f32::new(0.0);
}

/// Coefficient-domain mode: skips error_coeffs writes, computes info_loss
/// and info_loss2, adds k_cost2 for q >= 1.5.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_coeff_kernel(
    block_c: &Array<f32>,
    block_y: &Array<f32>,
    inv_weights: &Array<f32>,
    output: &mut Array<f32>, // num_blocks * 4
    n: u32,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
    k_cost2: f32,
) {
    let block_idx = ABSOLUTE_POS;
    let n_per = n as usize;
    let n_blocks = block_c.len() / n_per;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * n_per;
    let oo = block_idx * 4usize;

    let mut entropy_sum = f32::new(0.0);
    let mut nzeros_sum = f32::new(0.0);
    let mut info_loss_sum = f32::new(0.0);
    let mut info_loss2_sum = f32::new(0.0);
    let mut i: u32 = 0u32;
    while (i as usize) < n_per {
        let iu = i as usize;
        let val_in = block_c[off + iu];
        let val_y = block_y[off + iu] * cmap_factor;
        let val = (val_in - val_y) * inv_weights[off + iu] * quant;
        let rval = f32::round(val);
        let diff = val - rval;
        let q = f32::abs(rval);
        entropy_sum = entropy_sum + f32::sqrt(q) * k_cost_delta;
        if q != 0.0f32 {
            nzeros_sum = nzeros_sum + 1.0f32;
        }
        let diff_abs = f32::abs(diff);
        info_loss_sum = info_loss_sum + diff_abs;
        info_loss2_sum = info_loss2_sum + diff_abs * diff_abs;
        if q >= 1.5f32 {
            entropy_sum = entropy_sum + k_cost2;
        }
        i += 1u32;
    }

    output[oo] = entropy_sum;
    output[oo + 1usize] = nzeros_sum;
    output[oo + 2usize] = info_loss_sum;
    output[oo + 3usize] = info_loss2_sum;
}
