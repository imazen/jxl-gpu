// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-block masked weighted L2 error.
//!
//! Mirrors `jxl_encoder_simd::block_l2::compute_block_l2_errors_scalar`.
//!
//! For each 8x8 block, computes:
//!   `error = Σ_pixels mask[px]^2 · Σ_c weight[c] · (orig[c][px] - recon[c][px])^2`
//!
//! One cube per block (cube_dim = 1, cube_count = num_blocks). Inner loop
//! reduces 64 pixels × 3 channels into a single f32 per block.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

const W_X: f32 = 12.339_445;
const W_Y: f32 = 1.0;
const W_B: f32 = 0.2;

/// One cube per block; output is `xsize_blocks * ysize_blocks` floats.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn block_l2_kernel(
    orig_x: &Array<f32>,
    orig_y: &Array<f32>,
    orig_b: &Array<f32>,
    recon_x: &Array<f32>,
    recon_y: &Array<f32>,
    recon_b: &Array<f32>,
    mask1x1: &Array<f32>,
    output: &mut Array<f32>,
    xsize_blocks: u32,
    padded_width: u32,
) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = output.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let xb = xsize_blocks as usize;
    let pw = padded_width as usize;
    let bx = block_idx - (block_idx / xb) * xb;
    let by = block_idx / xb;

    let mut total = 0.0f32;
    let mut py: u32 = 0u32;
    while py < 8u32 {
        let mut px: u32 = 0u32;
        while px < 8u32 {
            let pyu = py as usize;
            let pxu = px as usize;
            let y = by * 8usize + pyu;
            let x = bx * 8usize + pxu;
            let pidx = y * pw + x;
            let mask = mask1x1[pidx];
            let ms = mask * mask;
            let dx = orig_x[pidx] - recon_x[pidx];
            let dy = orig_y[pidx] - recon_y[pidx];
            let db = orig_b[pidx] - recon_b[pidx];
            total = total + W_X * ms * dx * dx;
            total = total + W_Y * ms * dy * dy;
            total = total + W_B * ms * db * db;
            px += 1u32;
        }
        py += 1u32;
    }
    output[block_idx] = total;
}
