// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Raw 4×4 and 4×8 inverse DCTs — the *primitive* inverse transforms
//! (16 → 16 and 32 → 32 floats per block) that the AFV0-3 corner family
//! consumes on the decoder side.
//!
//! Mirror of `crate::kernels::dct4_raw`; matches
//! `jxl_encoder::vardct::dct::inverse::{idct_4x4, idct_4x8}` exactly.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

const INV_SQRT2: f32 = 0.707_106_77;
const INV_WC4_0: f32 = 1.0 / 0.541_196_1;
const INV_WC4_1: f32 = 1.0 / 1.306_563;
const INV_WC8_0: f32 = 1.0 / 0.509_795_6;
const INV_WC8_1: f32 = 1.0 / 0.601_344_9;
const INV_WC8_2: f32 = 1.0 / 0.899_976_2;
const INV_WC8_3: f32 = 1.0 / 2.562_915_5;
const FOUR: f32 = 4.0;
const EIGHT: f32 = 8.0;

/// 1D 4-point inverse DCT writing back to `mem[base..base+4]`.
/// Reads (even0, even1, odd0, odd1) = (mem[b], mem[b+1], mem[b+2], mem[b+3]).
/// Mirrors upstream `idct1d_4_val(a, b, c, d)` returning 4 values.
#[cube]
fn idct1d_4(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    let a = mem[b];
    let bv = mem[b + 1usize];
    let c = mem[b + 2usize];
    let d = mem[b + 3usize];
    // Reverse B transform on odd half
    let odd0 = (c - d) * INV_SQRT2;
    // Reverse idct1d_2 on odd half
    let o0_pre = (odd0 + d) * 0.5_f32;
    let o1_pre = (odd0 - d) * 0.5_f32;
    let o0 = o0_pre * INV_WC4_0;
    let o1 = o1_pre * INV_WC4_1;
    // Reverse idct1d_2 on even half
    let e0 = (a + bv) * 0.5_f32;
    let e1 = (a - bv) * 0.5_f32;
    mem[b] = (e0 + o0) * 0.5_f32;
    mem[b + 1usize] = (e1 + o1) * 0.5_f32;
    mem[b + 2usize] = (e1 - o1) * 0.5_f32;
    mem[b + 3usize] = (e0 - o0) * 0.5_f32;
}

/// 1D 8-point inverse DCT (core, no N-scaling). Reads/writes
/// `mem[base..base+8]`. Mirrors upstream `idct1d_8_core_val(m)`.
#[cube]
fn idct1d_8_core(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    // De-interleave: even = m[0,2,4,6], odd = m[1,3,5,7].
    let e0 = mem[b];
    let e1 = mem[b + 2usize];
    let e2 = mem[b + 4usize];
    let e3 = mem[b + 6usize];
    let mut o0 = mem[b + 1usize];
    let mut o1 = mem[b + 3usize];
    let mut o2 = mem[b + 5usize];
    let o3 = mem[b + 7usize];
    // Reverse B transform
    o2 = o2 - o3;
    o1 = o1 - o2;
    o0 = (o0 - o1) * INV_SQRT2;
    // Reverse idct1d_4 on odd half: idct1d_4_val(o0, o2, o1, o3) → 4 values
    // Inline (avoid two-helper interaction):
    let odd_a = o0;
    let odd_bv = o2;
    let odd_c = o1;
    let odd_d = o3;
    let odd_odd0 = (odd_c - odd_d) * INV_SQRT2;
    let odd_o0_pre = (odd_odd0 + odd_d) * 0.5_f32;
    let odd_o1_pre = (odd_odd0 - odd_d) * 0.5_f32;
    let odd_oe0 = odd_o0_pre * INV_WC4_0;
    let odd_oe1 = odd_o1_pre * INV_WC4_1;
    let odd_e0 = (odd_a + odd_bv) * 0.5_f32;
    let odd_e1 = (odd_a - odd_bv) * 0.5_f32;
    let odd_r0 = (odd_e0 + odd_oe0) * 0.5_f32;
    let odd_r1 = (odd_e1 + odd_oe1) * 0.5_f32;
    let odd_r2 = (odd_e1 - odd_oe1) * 0.5_f32;
    let odd_r3 = (odd_e0 - odd_oe0) * 0.5_f32;
    // Multiply by reciprocal WC8.
    let oa = odd_r0 * INV_WC8_0;
    let ob = odd_r1 * INV_WC8_1;
    let oc = odd_r2 * INV_WC8_2;
    let od = odd_r3 * INV_WC8_3;
    // Reverse idct1d_4 on even half: idct1d_4_val(e0, e2, e1, e3)
    let ev_a = e0;
    let ev_bv = e2;
    let ev_c = e1;
    let ev_d = e3;
    let ev_odd0 = (ev_c - ev_d) * INV_SQRT2;
    let ev_o0_pre = (ev_odd0 + ev_d) * 0.5_f32;
    let ev_o1_pre = (ev_odd0 - ev_d) * 0.5_f32;
    let ev_oe0 = ev_o0_pre * INV_WC4_0;
    let ev_oe1 = ev_o1_pre * INV_WC4_1;
    let ev_e0_in = (ev_a + ev_bv) * 0.5_f32;
    let ev_e1_in = (ev_a - ev_bv) * 0.5_f32;
    let er0 = (ev_e0_in + ev_oe0) * 0.5_f32;
    let er1 = (ev_e1_in + ev_oe1) * 0.5_f32;
    let er2 = (ev_e1_in - ev_oe1) * 0.5_f32;
    let er3 = (ev_e0_in - ev_oe0) * 0.5_f32;
    // Combine even/odd.
    mem[b] = (er0 + oa) * 0.5_f32;
    mem[b + 1usize] = (er1 + ob) * 0.5_f32;
    mem[b + 2usize] = (er2 + oc) * 0.5_f32;
    mem[b + 3usize] = (er3 + od) * 0.5_f32;
    mem[b + 4usize] = (er3 - od) * 0.5_f32;
    mem[b + 5usize] = (er2 - oc) * 0.5_f32;
    mem[b + 6usize] = (er1 - ob) * 0.5_f32;
    mem[b + 7usize] = (er0 - oa) * 0.5_f32;
}

/// Inverse raw 4×4 DCT. Input/output: `num_blocks * 16` floats.
// Cube macro expands the unrolled inner loops with literal `0 * COLS + row`
// offsets that the lints flag as no-effect / erasing-op.
#[allow(clippy::erasing_op, clippy::no_effect_underscore_binding, clippy::identity_op)]
#[cube(launch_unchecked)]
pub fn idct_4x4_raw_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 16u32 as usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 16usize;

    let mut buf = SharedMemory::<f32>::new(32usize);
    // tile region: [0..16]; temp region: [16..32].

    // Pass 1: per-row 4pt IDCT (scaled by 4), store transposed (col, row).
    let mut row: u32 = 0u32;
    while row < 4u32 {
        let rs = (row * 4u32) as usize;
        // Load de-interleaved into tile[0..4]: (input[s]*4, input[s+2]*4,
        // input[s+1]*4, input[s+3]*4).
        buf[0] = input[off + rs] * FOUR;
        buf[1] = input[off + rs + 2usize] * FOUR;
        buf[2] = input[off + rs + 1usize] * FOUR;
        buf[3] = input[off + rs + 3usize] * FOUR;
        idct1d_4(&mut buf, 0u32);
        // Store transposed into temp.
        buf[16usize + (0u32 * 4u32 + row) as usize] = buf[0];
        buf[16usize + (1u32 * 4u32 + row) as usize] = buf[1];
        buf[16usize + (2u32 * 4u32 + row) as usize] = buf[2];
        buf[16usize + (3u32 * 4u32 + row) as usize] = buf[3];
        row += 1u32;
    }

    // Pass 2: per-row 4pt IDCT on temp (scaled by 4), copy to output.
    let mut row2: u32 = 0u32;
    while row2 < 4u32 {
        let s = 16usize + (row2 * 4u32) as usize;
        // De-interleave: tile[0..4] = (temp[s]*4, temp[s+2]*4, temp[s+1]*4, temp[s+3]*4).
        buf[0] = buf[s] * FOUR;
        buf[1] = buf[s + 2usize] * FOUR;
        buf[2] = buf[s + 1usize] * FOUR;
        buf[3] = buf[s + 3usize] * FOUR;
        idct1d_4(&mut buf, 0u32);
        let os = (row2 * 4u32) as usize;
        output[off + os] = buf[0];
        output[off + os + 1usize] = buf[1];
        output[off + os + 2usize] = buf[2];
        output[off + os + 3usize] = buf[3];
        row2 += 1u32;
    }
}

/// Inverse raw 4×8 DCT. Input layout: 4 cols × 8 rows (transposed,
/// matching the forward kernel's output). Output: 4 rows × 8 cols
/// row-major (`num_blocks * 32` floats).
// Cube macro expands the unrolled inner loops with literal `0 * COLS + row`
// offsets — see kernel-level note on idct_4x4_raw_kernel above.
#[allow(clippy::erasing_op, clippy::no_effect_underscore_binding, clippy::identity_op)]
#[cube(launch_unchecked)]
pub fn idct_4x8_raw_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 32u32 as usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 32usize;

    // 32-elem temp + 4-elem scratch for inner idct1d_4 = 36; round to 64.
    let mut buf = SharedMemory::<f32>::new(64usize);
    // Layout: temp = [0..32]; scratch = [32..36].

    // Pass 1: For each col, gather 4-element column from input,
    // run 4pt IDCT on (a, c, b, d) with scale 4, store transposed.
    let mut col: u32 = 0u32;
    while col < 8u32 {
        let cu = col as usize;
        let a = input[off + cu] * FOUR;
        let bv = input[off + 8usize + cu] * FOUR;
        let c = input[off + 16usize + cu] * FOUR;
        let d = input[off + 24usize + cu] * FOUR;
        // De-interleave: idct1d_4_val(a, c, b, d) = (even0=a, even1=c, odd0=b, odd1=d)
        buf[32usize] = a;
        buf[33usize] = c;
        buf[34usize] = bv;
        buf[35usize] = d;
        idct1d_4(&mut buf, 32u32);
        let r0 = buf[32usize];
        let r1 = buf[33usize];
        let r2 = buf[34usize];
        let r3 = buf[35usize];
        // Store transposed: temp[row*8 + col] = r[row].
        buf[(0u32 * 8u32 + col) as usize] = r0;
        buf[(1u32 * 8u32 + col) as usize] = r1;
        buf[(2u32 * 8u32 + col) as usize] = r2;
        buf[(3u32 * 8u32 + col) as usize] = r3;
        col += 1u32;
    }

    // Pass 2: 8pt IDCT (scaled by 8) on each of 4 rows of temp.
    let mut row: u32 = 0u32;
    while row < 4u32 {
        let s = (row * 8u32) as usize;
        // Scale by 8 in-place.
        let mut k: u32 = 0u32;
        while k < 8u32 {
            buf[s + k as usize] = buf[s + k as usize] * EIGHT;
            k += 1u32;
        }
        idct1d_8_core(&mut buf, row * 8u32);
        // Copy to output.
        let mut kk: u32 = 0u32;
        while kk < 8u32 {
            output[off + s + kk as usize] = buf[s + kk as usize];
            kk += 1u32;
        }
        row += 1u32;
    }
}
