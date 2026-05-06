// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Gaborish inverse: 5x5 symmetric sharpening stencil.
//!
//! Mirrors `jxl_encoder_simd::gaborish5x5::gaborish_5x5_scalar`.
//! 6 weight classes (c, r, d, R, L, D) with clamp-to-edge boundary.

use cubecl::prelude::*;

/// Apply 5x5 gaborish stencil: out = stencil(in).
///
/// `input` and `output` MUST be distinct buffers.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn gaborish_5x5_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    wc: f32,
    wr: f32,
    wd: f32,
    w_big_r: f32,
    wl: f32,
    w_big_d: f32,
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

    // Per-axis clamps for ±1 and ±2 offsets (clamp-to-edge).
    let xm1 = usize::saturating_sub(x, 1usize);
    let xp1 = usize::min(x + 1usize, w - 1usize);
    let xm2 = usize::saturating_sub(x, 2usize);
    let xp2 = usize::min(x + 2usize, w - 1usize);
    let ym1 = usize::saturating_sub(y, 1usize);
    let yp1 = usize::min(y + 1usize, h - 1usize);
    let ym2 = usize::saturating_sub(y, 2usize);
    let yp2 = usize::min(y + 2usize, h - 1usize);

    let row_c = y * w;
    let row_tm1 = ym1 * w;
    let row_tp1 = yp1 * w;
    let row_tm2 = ym2 * w;
    let row_tp2 = yp2 * w;

    let mut val = wc * input[row_c + x];

    // r: 4 orthogonal at distance 1
    val += wr * (input[row_c + xm1] + input[row_c + xp1] + input[row_tm1 + x] + input[row_tp1 + x]);

    // d: 4 diagonals at distance sqrt(2)
    val += wd
        * (input[row_tm1 + xm1]
            + input[row_tm1 + xp1]
            + input[row_tp1 + xm1]
            + input[row_tp1 + xp1]);

    // R: 4 orthogonal at distance 2
    val += w_big_r
        * (input[row_c + xm2] + input[row_c + xp2] + input[row_tm2 + x] + input[row_tp2 + x]);

    // L: 8 knight's moves
    val += wl
        * (input[row_tm1 + xm2]
            + input[row_tp1 + xm2]
            + input[row_tm1 + xp2]
            + input[row_tp1 + xp2]
            + input[row_tm2 + xm1]
            + input[row_tm2 + xp1]
            + input[row_tp2 + xm1]
            + input[row_tp2 + xp1]);

    // D: 4 corners at distance 2*sqrt(2)
    val += w_big_d
        * (input[row_tm2 + xm2]
            + input[row_tm2 + xp2]
            + input[row_tp2 + xm2]
            + input[row_tp2 + xp2]);

    output[idx] = val;
}
