// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Raw 4×4 and 4×8 forward DCTs — the *primitive* transforms (16 →
//! 16 and 32 → 32 floats per block) that the AFV0-3 corner family
//! consumes.
//!
//! These are distinct from the 8×8-layout `dct_4x4_full` / `dct_4x8_full`
//! kernels in `crate::kernels::dct4`: those operate on 64-coeff 8×8
//! blocks with internal sub-block + DC-merge structure. AFV needs the
//! plain 16-coeff 4×4 and 32-coeff 4×8 forms, separately.
//!
//! Mirrors `jxl_encoder::vardct::dct::forward::{dct_4x4, dct_4x8}`.
//! One thread per sub-block; per-block scratch in
//! `SharedMemory<f32>::new(...)`.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

const SQRT2: f32 = core::f32::consts::SQRT_2;
const WC4_0: f32 = 0.541_196_1;
const WC4_1: f32 = 1.306_563;
const WC8_0: f32 = 0.509_795_6;
const WC8_1: f32 = 0.601_344_9;
const WC8_2: f32 = 0.899_976_2;
const WC8_3: f32 = 2.562_915_5;
const ONE_OVER_4: f32 = 0.25;
const ONE_OVER_8: f32 = 0.125;

/// 1D 4-point DCT, in-place on `mem[base..base+4]`.
/// Mirrors upstream `dct1d_4_val(a, b, c, d)` returning `[u0, b0, u1, w1]`.
#[cube]
fn dct1d_4(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    let a = mem[b];
    let bv = mem[b + 1usize];
    let c = mem[b + 2usize];
    let d = mem[b + 3usize];
    let t0 = a + d;
    let t1 = bv + c;
    let t2 = a - d;
    let t3 = bv - c;
    let u0 = t0 + t1;
    let u1 = t0 - t1;
    let v0 = t2 * WC4_0;
    let v1 = t3 * WC4_1;
    let w0 = v0 + v1;
    let w1 = v0 - v1;
    let b0 = SQRT2 * w0 + w1;
    mem[b] = u0;
    mem[b + 1usize] = b0;
    mem[b + 2usize] = u1;
    mem[b + 3usize] = w1;
}

/// 1D 8-point DCT, in-place on `mem[base..base+8]`. Inlines both
/// internal 4-pt DCT calls to avoid any cube-macro scoping ambiguity
/// around the in-place dct1d_4 helper sharing the same SharedMemory.
#[cube]
fn dct1d_8(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    let m0 = mem[b];
    let m1 = mem[b + 1usize];
    let m2 = mem[b + 2usize];
    let m3 = mem[b + 3usize];
    let m4 = mem[b + 4usize];
    let m5 = mem[b + 5usize];
    let m6 = mem[b + 6usize];
    let m7 = mem[b + 7usize];
    let t0 = m0 + m7;
    let t1 = m1 + m6;
    let t2 = m2 + m5;
    let t3 = m3 + m4;
    let t4 = m0 - m7;
    let t5 = m1 - m6;
    let t6 = m2 - m5;
    let t7 = m3 - m4;
    // ── First-half: dct1d_4 inlined on (t0, t1, t2, t3) ──
    let s0 = t0 + t3;
    let s1 = t1 + t2;
    let s2 = t0 - t3;
    let s3 = t1 - t2;
    let r0_0 = s0 + s1;             // u0
    let r0_2 = s0 - s1;             // u1
    let v0 = s2 * WC4_0;
    let v1 = s3 * WC4_1;
    let r0_3 = v0 - v1;             // w1
    let r0_1 = SQRT2 * (v0 + v1) + r0_3; // b0 = SQRT2*w0 + w1
    // ── Second-half: WC8 multiply + dct1d_4 inlined ──
    let w4 = t4 * WC8_0;
    let w5 = t5 * WC8_1;
    let w6 = t6 * WC8_2;
    let w7 = t7 * WC8_3;
    let q0 = w4 + w7;
    let q1 = w5 + w6;
    let q2 = w4 - w7;
    let q3 = w5 - w6;
    let r1_0 = q0 + q1;             // u0
    let r1_2 = q0 - q1;             // u1
    let v0b = q2 * WC4_0;
    let v1b = q3 * WC4_1;
    let r1_3 = v0b - v1b;           // w1
    let r1_1 = SQRT2 * (v0b + v1b) + r1_3; // b0
    // ── Final B-transform + interleave to libjxl factorization order ──
    let b0 = SQRT2 * r1_0 + r1_1;
    let b1 = r1_1 + r1_2;
    let b2 = r1_2 + r1_3;
    let b3 = r1_3;
    mem[b] = r0_0;
    mem[b + 1usize] = b0;
    mem[b + 2usize] = r0_2;
    mem[b + 3usize] = b2;
    mem[b + 4usize] = r0_1;
    mem[b + 5usize] = b1;
    mem[b + 6usize] = r0_3;
    mem[b + 7usize] = b3;
}

/// Forward raw 4×4 DCT. Input/output: `num_blocks * 16` floats.
#[cube(launch_unchecked)]
pub fn dct_4x4_raw_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 16u32 as usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 16usize;

    // tile holds intermediate row-DCT outputs (one row at a time).
    let mut tile = SharedMemory::<f32>::new(16usize);
    let mut temp = SharedMemory::<f32>::new(16usize);

    // Pass 1: row DCTs (4-point) on input rows. Store transposed
    // (col, row) into temp with 1/4 scale.
    let mut row: u32 = 0u32;
    while row < 4u32 {
        let rs = (row * 4u32) as usize;
        tile[rs] = input[off + rs];
        tile[rs + 1usize] = input[off + rs + 1usize];
        tile[rs + 2usize] = input[off + rs + 2usize];
        tile[rs + 3usize] = input[off + rs + 3usize];
        dct1d_4(&mut tile, row * 4u32);
        let r0 = tile[rs];
        let r1 = tile[rs + 1usize];
        let r2 = tile[rs + 2usize];
        let r3 = tile[rs + 3usize];
        temp[(0u32 * 4u32 + row) as usize] = r0 * ONE_OVER_4;
        temp[(1u32 * 4u32 + row) as usize] = r1 * ONE_OVER_4;
        temp[(2u32 * 4u32 + row) as usize] = r2 * ONE_OVER_4;
        temp[(3u32 * 4u32 + row) as usize] = r3 * ONE_OVER_4;
        row += 1u32;
    }

    // Pass 2: column DCTs (operating on temp's rows). Output is
    // direct (no further transpose for the square case).
    let mut row2: u32 = 0u32;
    while row2 < 4u32 {
        dct1d_4(&mut temp, row2 * 4u32);
        let s = (row2 * 4u32) as usize;
        output[off + s] = temp[s] * ONE_OVER_4;
        output[off + s + 1usize] = temp[s + 1usize] * ONE_OVER_4;
        output[off + s + 2usize] = temp[s + 2usize] * ONE_OVER_4;
        output[off + s + 3usize] = temp[s + 3usize] * ONE_OVER_4;
        row2 += 1u32;
    }
}

/// Forward raw 4×8 DCT (4 rows × 8 cols). Input/output: `num_blocks * 32` floats.
/// Output layout: 4 cols × 8 rows (transposed) per upstream's ROWS<COLS convention.
///
/// **KNOWN BUG**: this kernel currently has a parity divergence vs
/// upstream `jxl_encoder::vardct::dct::dct_4x8` (max|Δ| ≈ 7.3e-2 at
/// some output positions). Tried: helper-based + inlined dct1d_4
/// inside dct1d_8, two 32-elem SharedMemories vs one 64. None
/// fixed it. dct_4x4_raw using the same dct1d_4 helper is bit-perfect,
/// so the issue is somewhere in the dct_4x8 composition or dct1d_8.
/// See `examples/dct4_raw_parity.rs` for the reproducer.
///
/// Uses ONE 64-element SharedMemory split into a tile region [0..32]
/// + temp region [32..64].
#[cube(launch_unchecked)]
pub fn dct_4x8_raw_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 32u32 as usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 32usize;

    let mut buf = SharedMemory::<f32>::new(64usize);
    // tile region: [0..32]; temp region: [32..64].

    // Pass 1: row DCTs (8-point) on 4 rows × 8 cols. Store transposed
    // (col, row) into temp region with 1/8 scale.
    let mut row: u32 = 0u32;
    while row < 4u32 {
        let rs = (row * 8u32) as usize;
        let mut k: u32 = 0u32;
        while k < 8u32 {
            buf[rs + k as usize] = input[off + rs + k as usize];
            k += 1u32;
        }
        dct1d_8(&mut buf, row * 8u32);
        let mut col: u32 = 0u32;
        while col < 8u32 {
            buf[32usize + (col * 4u32 + row) as usize] = buf[rs + col as usize] * ONE_OVER_8;
            col += 1u32;
        }
        row += 1u32;
    }

    // Pass 2: column DCTs (4-point) on temp region's 8 rows × 4 cols.
    // Final transpose: write output as 4 cols × 8 rows (col-major in output).
    let mut row2: u32 = 0u32;
    while row2 < 8u32 {
        dct1d_4(&mut buf, 32u32 + row2 * 4u32);
        let s = 32usize + (row2 * 4u32) as usize;
        output[off + (0u32 * 8u32 + row2) as usize] = buf[s] * ONE_OVER_4;
        output[off + (1u32 * 8u32 + row2) as usize] = buf[s + 1usize] * ONE_OVER_4;
        output[off + (2u32 * 8u32 + row2) as usize] = buf[s + 2usize] * ONE_OVER_4;
        output[off + (3u32 * 8u32 + row2) as usize] = buf[s + 3usize] * ONE_OVER_4;
        row2 += 1u32;
    }
}
