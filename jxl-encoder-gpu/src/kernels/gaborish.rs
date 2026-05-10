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

/// 3-channel fused variant of [`fn@gaborish_5x5_kernel`]. Same math
/// applied independently to each input plane; produces three output
/// planes in one launch instead of three.
///
/// Saves 2 of 3 kernel launches in `prepare_strategy_search_plan`'s
/// `xyb_gab` stage. Per-cube work is 3× more (25 × 3 = 75 reads, 3
/// writes), but launch / SM-scheduling overhead is paid once. At
/// 16 MP that overhead appears to dominate (~170 ms for 4 launches
/// vs ~8 ms HBM-bandwidth predicted), so collapsing 3 → 1 of those
/// launches should give a real speedup.
///
/// All three input planes share the same `(width, height)` and the
/// same gaborish weight set (per `compute_weights(mul=1.0)`).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn gaborish_5x5_3ch_kernel(
    in_x: &Array<f32>,
    in_y: &Array<f32>,
    in_b: &Array<f32>,
    out_x: &mut Array<f32>,
    out_y: &mut Array<f32>,
    out_b: &mut Array<f32>,
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

    // Same stencil applied 3 times (once per channel). Index
    // computations + clamps amortized. cubecl 0.10 doesn't support
    // runtime array literals inside #[cube], so the offsets are
    // computed inline (matches the single-channel kernel above
    // line-by-line).
    let mut vx = wc * in_x[row_c + x];
    let mut vy = wc * in_y[row_c + x];
    let mut vb = wc * in_b[row_c + x];

    // r: orthogonal distance 1
    let r0 = row_c + xm1;
    let r1 = row_c + xp1;
    let r2 = row_tm1 + x;
    let r3 = row_tp1 + x;
    vx += wr * (in_x[r0] + in_x[r1] + in_x[r2] + in_x[r3]);
    vy += wr * (in_y[r0] + in_y[r1] + in_y[r2] + in_y[r3]);
    vb += wr * (in_b[r0] + in_b[r1] + in_b[r2] + in_b[r3]);

    // d: diagonal distance sqrt(2)
    let d0 = row_tm1 + xm1;
    let d1 = row_tm1 + xp1;
    let d2 = row_tp1 + xm1;
    let d3 = row_tp1 + xp1;
    vx += wd * (in_x[d0] + in_x[d1] + in_x[d2] + in_x[d3]);
    vy += wd * (in_y[d0] + in_y[d1] + in_y[d2] + in_y[d3]);
    vb += wd * (in_b[d0] + in_b[d1] + in_b[d2] + in_b[d3]);

    // R: orthogonal distance 2
    let br0 = row_c + xm2;
    let br1 = row_c + xp2;
    let br2 = row_tm2 + x;
    let br3 = row_tp2 + x;
    vx += w_big_r * (in_x[br0] + in_x[br1] + in_x[br2] + in_x[br3]);
    vy += w_big_r * (in_y[br0] + in_y[br1] + in_y[br2] + in_y[br3]);
    vb += w_big_r * (in_b[br0] + in_b[br1] + in_b[br2] + in_b[br3]);

    // L: 8 knight's moves
    let l0 = row_tm1 + xm2;
    let l1 = row_tp1 + xm2;
    let l2 = row_tm1 + xp2;
    let l3 = row_tp1 + xp2;
    let l4 = row_tm2 + xm1;
    let l5 = row_tm2 + xp1;
    let l6 = row_tp2 + xm1;
    let l7 = row_tp2 + xp1;
    vx += wl
        * (in_x[l0] + in_x[l1] + in_x[l2] + in_x[l3]
            + in_x[l4] + in_x[l5] + in_x[l6] + in_x[l7]);
    vy += wl
        * (in_y[l0] + in_y[l1] + in_y[l2] + in_y[l3]
            + in_y[l4] + in_y[l5] + in_y[l6] + in_y[l7]);
    vb += wl
        * (in_b[l0] + in_b[l1] + in_b[l2] + in_b[l3]
            + in_b[l4] + in_b[l5] + in_b[l6] + in_b[l7]);

    // D: corners distance 2*sqrt(2)
    let bd0 = row_tm2 + xm2;
    let bd1 = row_tm2 + xp2;
    let bd2 = row_tp2 + xm2;
    let bd3 = row_tp2 + xp2;
    vx += w_big_d * (in_x[bd0] + in_x[bd1] + in_x[bd2] + in_x[bd3]);
    vy += w_big_d * (in_y[bd0] + in_y[bd1] + in_y[bd2] + in_y[bd3]);
    vb += w_big_d * (in_b[bd0] + in_b[bd1] + in_b[bd2] + in_b[bd3]);

    out_x[idx] = vx;
    out_y[idx] = vy;
    out_b[idx] = vb;
}
