// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Gaborish smooth: 3x3 weighted stencil.
//!
//! Mirrors `jxl_encoder_simd::gab::gab_smooth_scalar`.

use cubecl::prelude::*;

/// Apply 3x3 gab stencil: out = w_center*center + w1*4-cardinals + w2*4-diagonals.
///
/// Clamp-to-edge boundary. `input` and `output` MUST be distinct buffers.
#[cube(launch_unchecked)]
pub fn gab_smooth_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
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

    let center = input[row_c + x];
    let top = input[row_t + x];
    let bottom = input[row_b + x];
    let left = input[row_c + xm];
    let right = input[row_c + xp];
    let tl = input[row_t + xm];
    let tr = input[row_t + xp];
    let bl = input[row_b + xm];
    let br = input[row_b + xp];

    output[idx] = w_center * center + w1 * (top + bottom + left + right) + w2 * (tl + tr + bl + br);
}
