// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! FuzzyErosion: per-pixel min-of-4-from-9 weighted sum + 2x downsample.
//!
//! Mirrors `jxl_encoder::vardct::adaptive_quant::fuzzy_erosion`. For each
//! input pixel: read 9 neighbors (3×3 with edge clamping), find the 4
//! smallest, weighted-sum them with `k_mul[0..4]`. Then accumulate 4
//! adjacent input contributions into one output pixel (2× downsample).
//!
//! GPU strategy: one thread per OUTPUT pixel, gathering its 4 input
//! contributions directly. Avoids the unsynchronized `+=` write race
//! that the CPU avoids by sequential iteration.
//!
//! `k_mul` is precomputed by the caller from `butteraugli_target` and
//! passed as 4 scalars (no GPU-side derivation needed — saves a state
//! buffer).

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

#[cube]
fn store_min4(v: f32, m0: &mut f32, m1: &mut f32, m2: &mut f32, m3: &mut f32) {
    if v < *m3 {
        if v < *m0 {
            *m3 = *m2;
            *m2 = *m1;
            *m1 = *m0;
            *m0 = v;
        } else if v < *m1 {
            *m3 = *m2;
            *m2 = *m1;
            *m1 = v;
        } else if v < *m2 {
            *m3 = *m2;
            *m2 = v;
        } else {
            *m3 = v;
        }
    }
}

/// Sum the smallest 4 of the 3×3 neighborhood around (`x`, `y`),
/// weighted by `k_mul[0..4]`. Edge-clamped reads — a thread on the
/// boundary reuses the center value where neighbors would be OOB.
#[cube]
fn weighted_min4_3x3(
    src: &Array<f32>,
    src_w: u32,
    src_h: u32,
    x: u32,
    y: u32,
    k0: f32,
    k1: f32,
    k2: f32,
    k3: f32,
) -> f32 {
    let xu = x as usize;
    let yu = y as usize;
    let wu = src_w as usize;
    let hu = src_h as usize;
    let xm1 = if xu >= 1usize { xu - 1usize } else { xu };
    let xp1 = if xu + 1usize < wu { xu + 1usize } else { xu };
    let ym1 = if yu >= 1usize { yu - 1usize } else { yu };
    let yp1 = if yu + 1usize < hu { yu + 1usize } else { yu };
    let _ = src_h; // unused after clamp computation

    let center = src[yu * wu + xu];
    let left = src[yu * wu + xm1];
    let right = src[yu * wu + xp1];
    let tl = src[ym1 * wu + xm1];
    let top = src[ym1 * wu + xu];
    let tr = src[ym1 * wu + xp1];
    let bl = src[yp1 * wu + xm1];
    let bot = src[yp1 * wu + xu];
    let br = src[yp1 * wu + xp1];

    // Sort first 4 (center, left, right, top-left) ascending into m0..m3.
    let mut m0 = center;
    let mut m1 = left;
    let mut m2 = right;
    let mut m3 = tl;
    if m0 > m1 {
        let t = m0;
        m0 = m1;
        m1 = t;
    }
    if m0 > m2 {
        let t = m0;
        m0 = m2;
        m2 = t;
    }
    if m0 > m3 {
        let t = m0;
        m0 = m3;
        m3 = t;
    }
    if m1 > m2 {
        let t = m1;
        m1 = m2;
        m2 = t;
    }
    if m1 > m3 {
        let t = m1;
        m1 = m3;
        m3 = t;
    }
    if m2 > m3 {
        let t = m2;
        m2 = m3;
        m3 = t;
    }

    // Insert remaining 5 values.
    store_min4(top, &mut m0, &mut m1, &mut m2, &mut m3);
    store_min4(tr, &mut m0, &mut m1, &mut m2, &mut m3);
    store_min4(bl, &mut m0, &mut m1, &mut m2, &mut m3);
    store_min4(bot, &mut m0, &mut m1, &mut m2, &mut m3);
    store_min4(br, &mut m0, &mut m1, &mut m2, &mut m3);

    k0 * m0 + k1 * m1 + k2 * m2 + k3 * m3
}

/// Fuzzy-erosion + 2× downsample. One thread per output pixel; gathers
/// 4 input contributions (input region offset by `from_x0`, `from_y0`).
///
/// Caller pre-computes `k0..k3` via the same formula libjxl uses:
/// `K_MUL_BASE[i] + mul * K_MUL_ADD[i]` then normalized to `K_TOTAL`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn fuzzy_erosion_kernel(
    src: &Array<f32>,
    output: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    from_x0: u32,
    from_y0: u32,
    out_w: u32,
    out_h: u32,
    k0: f32,
    k1: f32,
    k2: f32,
    k3: f32,
) {
    let idx = ABSOLUTE_POS;
    let n_out = (out_w as usize) * (out_h as usize);
    if idx >= n_out {
        terminate!();
    }
    let oy = idx / (out_w as usize);
    let ox = idx - oy * (out_w as usize);

    // Each output pixel collects from 4 input pixels:
    //   fy ∈ {2*oy, 2*oy+1}, fx ∈ {2*ox, 2*ox+1}
    let mut sum = 0.0f32;
    let mut dy: u32 = 0u32;
    while dy < 2u32 {
        let mut dx: u32 = 0u32;
        while dx < 2u32 {
            let fy = (oy as u32) * 2u32 + dy;
            let fx = (ox as u32) * 2u32 + dx;
            let y = fy + from_y0;
            let x = fx + from_x0;
            sum = sum + weighted_min4_3x3(src, src_w, src_h, x, y, k0, k1, k2, k3);
            dx += 1u32;
        }
        dy += 1u32;
    }
    output[idx] = sum;
}
