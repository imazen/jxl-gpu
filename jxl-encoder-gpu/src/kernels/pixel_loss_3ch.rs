// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Fused 3-channel pixel-domain loss (8th-power-norm of masked error).
//!
//! Mirrors `pixel_loss_kernel` × 3 fused into one launch — same mask
//! plane, same mask_row_base, same block dims, just 3 different
//! `mask_offset`s and 3 different per-channel error inputs / loss
//! outputs. Saves 2 launches per strategy in cost-grid evaluation.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn pixel_loss_3ch_kernel(
    pixel_error_x: &Array<f32>,
    pixel_error_y: &Array<f32>,
    pixel_error_b: &Array<f32>,
    mask: &Array<f32>,
    mask_row_base: &Array<u32>,
    output_x: &mut Array<f64>,
    output_y: &mut Array<f64>,
    output_b: &mut Array<f64>,
    mask_stride: u32,
    mask_offset_x: f32,
    mask_offset_y: f32,
    mask_offset_b: f32,
    block_width: u32,
    block_height: u32,
) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = output_y.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let bw = block_width as usize;
    let bh = block_height as usize;
    let ms = mask_stride as usize;
    let mrb = mask_row_base[block_idx] as usize;
    let err_base = block_idx * bw * bh;

    let mut acc_x = f64::new(0.0);
    let mut acc_y = f64::new(0.0);
    let mut acc_b = f64::new(0.0);
    let mut py: u32 = 0u32;
    while py < block_height {
        let pyu = py as usize;
        let mask_row_start = mrb + pyu * ms;
        let error_row_start = err_base + pyu * bw;
        let mut px: u32 = 0u32;
        while px < block_width {
            let pxu = px as usize;
            let mask_val = mask[mask_row_start + pxu];
            let ex = pixel_error_x[error_row_start + pxu];
            let ey = pixel_error_y[error_row_start + pxu];
            let eb = pixel_error_b[error_row_start + pxu];
            let mx = (mask_val + mask_offset_x) * ex;
            let my = (mask_val + mask_offset_y) * ey;
            let mb = (mask_val + mask_offset_b) * eb;
            let mx2 = (mx * mx) as f64;
            let my2 = (my * my) as f64;
            let mb2 = (mb * mb) as f64;
            let mx4 = mx2 * mx2;
            let my4 = my2 * my2;
            let mb4 = mb2 * mb2;
            acc_x = acc_x + mx4 * mx4;
            acc_y = acc_y + my4 * my4;
            acc_b = acc_b + mb4 * mb4;
            px += 1u32;
        }
        py += 1u32;
    }
    output_x[block_idx] = acc_x;
    output_y[block_idx] = acc_y;
    output_b[block_idx] = acc_b;
}
