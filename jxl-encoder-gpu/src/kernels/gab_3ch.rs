// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! 3-channel fused gab_smooth.
//!
//! Same 3×3 weighted stencil as `gab_smooth_kernel` but processes
//! 3 planes (X / Y / B) per launch. The 9 reads per channel
//! contribute the same arithmetic; the per-thread work tripled but
//! the launch overhead drops by 3×. Used in the e8/e9 inner loop's
//! postpass where the same stencil runs on the 3 recon planes back-
//! to-back.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn gab_smooth_3ch_kernel(
    input_x: &Array<f32>,
    input_y: &Array<f32>,
    input_b: &Array<f32>,
    output_x: &mut Array<f32>,
    output_y: &mut Array<f32>,
    output_b: &mut Array<f32>,
    width: u32,
    height: u32,
    w_center: f32,
    w1: f32,
    w2: f32,
) {
    let idx = ABSOLUTE_POS;
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if idx >= n {
        terminate!();
    }
    let y = idx / w;
    let x = idx - y * w;

    let xm = usize::saturating_sub(x, 1usize);
    let xp = usize::min(x + 1usize, w - 1usize);
    let ym = usize::saturating_sub(y, 1usize);
    let yp = usize::min(y + 1usize, h - 1usize);

    let row_c = y * w;
    let row_t = ym * w;
    let row_b = yp * w;
    let i_cc = row_c + x;
    let i_tc = row_t + x;
    let i_bc = row_b + x;
    let i_cl = row_c + xm;
    let i_cr = row_c + xp;
    let i_tl = row_t + xm;
    let i_tr = row_t + xp;
    let i_bl = row_b + xm;
    let i_br = row_b + xp;

    // X channel.
    {
        let center = input_x[i_cc];
        let top = input_x[i_tc];
        let bottom = input_x[i_bc];
        let left = input_x[i_cl];
        let right = input_x[i_cr];
        let tl = input_x[i_tl];
        let tr = input_x[i_tr];
        let bl = input_x[i_bl];
        let br = input_x[i_br];
        output_x[idx] =
            w_center * center + w1 * (top + bottom + left + right) + w2 * (tl + tr + bl + br);
    }
    // Y channel.
    {
        let center = input_y[i_cc];
        let top = input_y[i_tc];
        let bottom = input_y[i_bc];
        let left = input_y[i_cl];
        let right = input_y[i_cr];
        let tl = input_y[i_tl];
        let tr = input_y[i_tr];
        let bl = input_y[i_bl];
        let br = input_y[i_br];
        output_y[idx] =
            w_center * center + w1 * (top + bottom + left + right) + w2 * (tl + tr + bl + br);
    }
    // B channel.
    {
        let center = input_b[i_cc];
        let top = input_b[i_tc];
        let bottom = input_b[i_bc];
        let left = input_b[i_cl];
        let right = input_b[i_cr];
        let tl = input_b[i_tl];
        let tr = input_b[i_tr];
        let bl = input_b[i_bl];
        let br = input_b[i_br];
        output_b[idx] =
            w_center * center + w1 * (top + bottom + left + right) + w2 * (tl + tr + bl + br);
    }
}
