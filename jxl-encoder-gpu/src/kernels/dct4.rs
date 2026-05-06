// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! DCT4-based full transforms on 8x8 blocks (4x4, 4x8, 8x4) and inverses.
//!
//! Mirrors `jxl_encoder_simd::dct4::*_full_scalar`. Each transform operates
//! on an 8x8 pixel block partitioned into sub-blocks:
//!   - `dct_4x4_full`: 2x2 grid of 4x4 sub-blocks, with 2x2 Hadamard on DCs
//!   - `dct_4x8_full`: 2 vertically-stacked 4x8 sub-blocks, DC averaging
//!   - `dct_8x4_full`: 2 horizontally-adjacent 8x4 sub-blocks, DC averaging
//!
//! Strategy: one cube per 8x8 block (cube_dim=1). All scratch in
//! `SharedMemory<f32>`.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

const SQRT2: f32 = core::f32::consts::SQRT_2;
const INV_SQRT2: f32 = 0.707_106_77;
const WC4_0: f32 = 0.541_196_1;
const WC4_1: f32 = 1.306_563;
const INV_WC4_0: f32 = 1.0 / WC4_0;
const INV_WC4_1: f32 = 1.0 / WC4_1;
const WC8_0: f32 = 0.509_795_6;
const WC8_1: f32 = 0.601_344_9;
const WC8_2: f32 = 0.899_976_2;
const WC8_3: f32 = 2.562_915_5;
const INV_WC8_0: f32 = 1.0 / WC8_0;
const INV_WC8_1: f32 = 1.0 / WC8_1;
const INV_WC8_2: f32 = 1.0 / WC8_2;
const INV_WC8_3: f32 = 1.0 / WC8_3;

// =============================================================================
// Butterfly helpers (in-place on SharedMemory)
// =============================================================================

/// Forward 4-pt DCT, reads/writes mem[base..base+4]. Matches CPU
/// `dct1d_4_val(a,b,c,d)` returning `[u0, b0, u1, w1]`.
#[cube]
fn fwd_dct4(mem: &mut SharedMemory<f32>, base: u32) {
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

/// Forward 8-pt DCT, reads/writes mem[base..base+8]. Matches CPU
/// `dct1d_8_val(m)` from dct4.rs (interleaved output).
#[cube]
fn fwd_dct8(mem: &mut SharedMemory<f32>, base: u32) {
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

    // dct1d_4(t0..t3) → r0
    let a0 = t0 + t3;
    let a1 = t1 + t2;
    let a2 = t0 - t3;
    let a3 = t1 - t2;
    let p0 = a0 + a1;
    let p1 = a0 - a1;
    let q0 = a2 * WC4_0;
    let q1 = a3 * WC4_1;
    let r0 = q0 + q1;
    let r1 = q0 - q1;
    let r0p = SQRT2 * r0 + r1;
    let e_0 = p0;
    let e_1 = r0p;
    let e_2 = p1;
    let e_3 = r1;

    // Multiply t4..t7 by WC8
    let w4 = t4 * WC8_0;
    let w5 = t5 * WC8_1;
    let w6 = t6 * WC8_2;
    let w7 = t7 * WC8_3;

    // dct1d_4(w4..w7) → r1
    let a0 = w4 + w7;
    let a1 = w5 + w6;
    let a2 = w4 - w7;
    let a3 = w5 - w6;
    let p0 = a0 + a1;
    let p1 = a0 - a1;
    let q0 = a2 * WC4_0;
    let q1 = a3 * WC4_1;
    let r0 = q0 + q1;
    let r1v = q0 - q1;
    let r0p2 = SQRT2 * r0 + r1v;
    let f_0 = p0;
    let f_1 = r0p2;
    let f_2 = p1;
    let f_3 = r1v;

    // Post: b0 = SQRT2*r1[0] + r1[1]; b1 = r1[1]+r1[2]; b2 = r1[2]+r1[3]; b3 = r1[3]
    let b0 = SQRT2 * f_0 + f_1;
    let b1 = f_1 + f_2;
    let b2 = f_2 + f_3;
    let b3 = f_3;

    // Result: [r0[0], b0, r0[1], b1, r0[2], b2, r0[3], b3]
    mem[b] = e_0;
    mem[b + 1usize] = b0;
    mem[b + 2usize] = e_1;
    mem[b + 3usize] = b1;
    mem[b + 4usize] = e_2;
    mem[b + 5usize] = b2;
    mem[b + 6usize] = e_3;
    mem[b + 7usize] = b3;
}

/// Inverse 4-pt IDCT taking input in (even0, even1, odd0, odd1) order
/// (caller pre-permutes). Reads mem[base..base+4], writes mem[base..base+4]
/// in natural order. Matches CPU `idct1d_4_val(a,b,c,d)`.
#[cube]
fn inv_idct4(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    let a = mem[b];
    let bv = mem[b + 1usize];
    let c = mem[b + 2usize];
    let d = mem[b + 3usize];
    let odd0 = (c - d) * INV_SQRT2;
    let o0_pre = (odd0 + d) * 0.5f32;
    let o1_pre = (odd0 - d) * 0.5f32;
    let o0 = o0_pre * INV_WC4_0;
    let o1 = o1_pre * INV_WC4_1;
    let e0 = (a + bv) * 0.5f32;
    let e1 = (a - bv) * 0.5f32;
    mem[b] = (e0 + o0) * 0.5f32;
    mem[b + 1usize] = (e1 + o1) * 0.5f32;
    mem[b + 2usize] = (e1 - o1) * 0.5f32;
    mem[b + 3usize] = (e0 - o0) * 0.5f32;
}

/// Inverse 8-pt IDCT core (no N scaling). Matches CPU `idct1d_8_core_val(m)`.
#[cube]
fn inv_idct8_core(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    let e0 = mem[b];
    let e1 = mem[b + 2usize];
    let e2 = mem[b + 4usize];
    let e3 = mem[b + 6usize];
    let mut o0 = mem[b + 1usize];
    let mut o1 = mem[b + 3usize];
    let mut o2 = mem[b + 5usize];
    let o3 = mem[b + 7usize];

    o2 = o2 - o3;
    o1 = o1 - o2;
    o0 = (o0 - o1) * INV_SQRT2;

    // odd = idct1d_4(o0, o2, o1, o3) — input order (even0, even1, odd0, odd1)
    let odd_a = o0;
    let odd_b = o2;
    let odd_c = o1;
    let odd_d = o3;
    let odd0_pre = (odd_c - odd_d) * INV_SQRT2;
    let oo0_pre = (odd0_pre + odd_d) * 0.5f32;
    let oo1_pre = (odd0_pre - odd_d) * 0.5f32;
    let oo0 = oo0_pre * INV_WC4_0;
    let oo1 = oo1_pre * INV_WC4_1;
    let oe0 = (odd_a + odd_b) * 0.5f32;
    let oe1 = (odd_a - odd_b) * 0.5f32;
    let odd_0 = (oe0 + oo0) * 0.5f32;
    let odd_1 = (oe1 + oo1) * 0.5f32;
    let odd_2 = (oe1 - oo1) * 0.5f32;
    let odd_3 = (oe0 - oo0) * 0.5f32;

    let oo_0 = odd_0 * INV_WC8_0;
    let oo_1 = odd_1 * INV_WC8_1;
    let oo_2 = odd_2 * INV_WC8_2;
    let oo_3 = odd_3 * INV_WC8_3;

    // even = idct1d_4(e0, e2, e1, e3)
    let ev_a = e0;
    let ev_b = e2;
    let ev_c = e1;
    let ev_d = e3;
    let odd0_pre = (ev_c - ev_d) * INV_SQRT2;
    let eo0_pre = (odd0_pre + ev_d) * 0.5f32;
    let eo1_pre = (odd0_pre - ev_d) * 0.5f32;
    let eo0 = eo0_pre * INV_WC4_0;
    let eo1 = eo1_pre * INV_WC4_1;
    let ee0 = (ev_a + ev_b) * 0.5f32;
    let ee1 = (ev_a - ev_b) * 0.5f32;
    let even_0 = (ee0 + eo0) * 0.5f32;
    let even_1 = (ee1 + eo1) * 0.5f32;
    let even_2 = (ee1 - eo1) * 0.5f32;
    let even_3 = (ee0 - eo0) * 0.5f32;

    // Output: [(e[i]+o[i])/2, ..., (e[3-i]-o[3-i])/2 reverse]
    mem[b] = (even_0 + oo_0) * 0.5f32;
    mem[b + 1usize] = (even_1 + oo_1) * 0.5f32;
    mem[b + 2usize] = (even_2 + oo_2) * 0.5f32;
    mem[b + 3usize] = (even_3 + oo_3) * 0.5f32;
    mem[b + 4usize] = (even_3 - oo_3) * 0.5f32;
    mem[b + 5usize] = (even_2 - oo_2) * 0.5f32;
    mem[b + 6usize] = (even_1 - oo_1) * 0.5f32;
    mem[b + 7usize] = (even_0 - oo_0) * 0.5f32;
}

// =============================================================================
// dct_4x4_full
// =============================================================================

/// Forward DCT4x4 full: 8x8 input → 8x8 output partitioned into 4 sub-blocks
/// with 2x2 Hadamard DC combination.
#[cube(launch_unchecked)]
pub fn dct_4x4_full_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;

    let mut tile = SharedMemory::<f32>::new(64usize);
    let mut out = SharedMemory::<f32>::new(64usize);
    let mut temp = SharedMemory::<f32>::new(16usize);
    let mut row = SharedMemory::<f32>::new(4usize);

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        tile[iu] = input[off + iu];
        i += 1u32;
    }

    let mut y: u32 = 0u32;
    while y < 2u32 {
        let mut x: u32 = 0u32;
        while x < 2u32 {
            // Row pass: per row of sub-block, 4-pt DCT, scale 0.25, store transposed
            let mut iy: u32 = 0u32;
            while iy < 4u32 {
                let yu = y as usize;
                let xu = x as usize;
                let iyu = iy as usize;
                let base = (yu * 4usize + iyu) * 8usize + xu * 4usize;
                row[0usize] = tile[base];
                row[1usize] = tile[base + 1usize];
                row[2usize] = tile[base + 2usize];
                row[3usize] = tile[base + 3usize];
                fwd_dct4(&mut row, 0u32);
                // temp[col*4 + iy] = row[col] * 0.25
                temp[iyu] = row[0usize] * 0.25f32;
                temp[4usize + iyu] = row[1usize] * 0.25f32;
                temp[8usize + iyu] = row[2usize] * 0.25f32;
                temp[12usize + iyu] = row[3usize] * 0.25f32;
                iy += 1u32;
            }
            // Col pass: per col, 4-pt DCT, scale 0.25, store to output with sub-block layout
            let mut col: u32 = 0u32;
            while col < 4u32 {
                let yu = y as usize;
                let xu = x as usize;
                let cu = col as usize;
                let s = cu * 4usize;
                row[0usize] = temp[s];
                row[1usize] = temp[s + 1usize];
                row[2usize] = temp[s + 2usize];
                row[3usize] = temp[s + 3usize];
                fwd_dct4(&mut row, 0u32);
                // output[(y + col*2)*8 + x + ix*2] = row[ix] * 0.25
                let mut ix: u32 = 0u32;
                while ix < 4u32 {
                    let ixu = ix as usize;
                    out[(yu + cu * 2usize) * 8usize + xu + ixu * 2usize] =
                        row[ixu] * 0.25f32;
                    ix += 1u32;
                }
                col += 1u32;
            }
            x += 1u32;
        }
        y += 1u32;
    }

    // 2x2 Hadamard on the 4 sub-block DCs (positions 0, 1, 8, 9)
    let block00 = out[0usize];
    let block01 = out[1usize];
    let block10 = out[8usize];
    let block11 = out[9usize];
    out[0usize] = (block00 + block01 + block10 + block11) * 0.25f32;
    out[1usize] = (block00 + block01 - block10 - block11) * 0.25f32;
    out[8usize] = (block00 - block01 + block10 - block11) * 0.25f32;
    out[9usize] = (block00 - block01 - block10 + block11) * 0.25f32;

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        output[off + iu] = out[iu];
        i += 1u32;
    }
}

/// Inverse DCT4x4 full.
#[cube(launch_unchecked)]
pub fn idct_4x4_full_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;

    let mut coeffs = SharedMemory::<f32>::new(64usize);
    let mut block = SharedMemory::<f32>::new(16usize);
    let mut temp = SharedMemory::<f32>::new(16usize);
    let mut row = SharedMemory::<f32>::new(4usize);

    // Load coefficients
    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        coeffs[iu] = input[off + iu];
        i += 1u32;
    }

    // Reverse 2x2 DC combination
    let a = coeffs[0usize];
    let bv = coeffs[1usize];
    let c = coeffs[8usize];
    let d = coeffs[9usize];
    coeffs[0usize] = a + bv + c + d;
    coeffs[1usize] = a + bv - c - d;
    coeffs[8usize] = a - bv + c - d;
    coeffs[9usize] = a - bv - c + d;

    let mut y: u32 = 0u32;
    while y < 2u32 {
        let mut x: u32 = 0u32;
        while x < 2u32 {
            // Gather sub-block coefficients with stride-2 layout
            let mut iy: u32 = 0u32;
            while iy < 4u32 {
                let mut ix: u32 = 0u32;
                while ix < 4u32 {
                    let yu = y as usize;
                    let xu = x as usize;
                    let iyu = iy as usize;
                    let ixu = ix as usize;
                    block[iyu * 4usize + ixu] =
                        coeffs[(yu + iyu * 2usize) * 8usize + (xu + ixu * 2usize)];
                    ix += 1u32;
                }
                iy += 1u32;
            }

            // Row pass: per row, multiply by 4, IDCT4 (de-interleave: even0,
            // even1, odd0, odd1 = block[0], block[2], block[1], block[3]).
            // Store transposed: temp[col*4 + row] = result[col]
            let mut r: u32 = 0u32;
            while r < 4u32 {
                let ru = r as usize;
                let s = ru * 4usize;
                row[0usize] = block[s] * 4.0f32;
                row[1usize] = block[s + 2usize] * 4.0f32;
                row[2usize] = block[s + 1usize] * 4.0f32;
                row[3usize] = block[s + 3usize] * 4.0f32;
                inv_idct4(&mut row, 0u32);
                temp[ru] = row[0usize];
                temp[4usize + ru] = row[1usize];
                temp[8usize + ru] = row[2usize];
                temp[12usize + ru] = row[3usize];
                r += 1u32;
            }
            // Col pass: per col, *4, IDCT4 (same de-interleave), store to output
            let mut r: u32 = 0u32;
            while r < 4u32 {
                let yu = y as usize;
                let xu = x as usize;
                let ru = r as usize;
                let s = ru * 4usize;
                row[0usize] = temp[s] * 4.0f32;
                row[1usize] = temp[s + 2usize] * 4.0f32;
                row[2usize] = temp[s + 1usize] * 4.0f32;
                row[3usize] = temp[s + 3usize] * 4.0f32;
                inv_idct4(&mut row, 0u32);
                let mut ix: u32 = 0u32;
                while ix < 4u32 {
                    let ixu = ix as usize;
                    output[off + (yu * 4usize + ru) * 8usize + (xu * 4usize + ixu)] =
                        row[ixu];
                    ix += 1u32;
                }
                r += 1u32;
            }
            x += 1u32;
        }
        y += 1u32;
    }
}

// =============================================================================
// dct_4x8_full / idct_4x8_full
// =============================================================================

#[cube(launch_unchecked)]
pub fn dct_4x8_full_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;

    let mut tile = SharedMemory::<f32>::new(64usize);
    let mut out = SharedMemory::<f32>::new(64usize);
    let mut temp = SharedMemory::<f32>::new(32usize);
    let mut row = SharedMemory::<f32>::new(8usize);

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        tile[iu] = input[off + iu];
        i += 1u32;
    }

    let mut y: u32 = 0u32;
    while y < 2u32 {
        // Row pass: per row of 8 pixels, 8-pt DCT, scale 0.125, store transposed
        let mut iy: u32 = 0u32;
        while iy < 4u32 {
            let yu = y as usize;
            let iyu = iy as usize;
            let base = (yu * 4usize + iyu) * 8usize;
            let mut k: u32 = 0u32;
            while k < 8u32 {
                let ku = k as usize;
                row[ku] = tile[base + ku];
                k += 1u32;
            }
            fwd_dct8(&mut row, 0u32);
            // temp[col*4 + iy] = row[col] * 0.125
            let mut col: u32 = 0u32;
            while col < 8u32 {
                let cu = col as usize;
                temp[cu * 4usize + iyu] = row[cu] * 0.125f32;
                col += 1u32;
            }
            iy += 1u32;
        }
        // Col pass: per col of 4 elements, 4-pt DCT, scale 0.25
        let mut col: u32 = 0u32;
        while col < 8u32 {
            let yu = y as usize;
            let cu = col as usize;
            let s = cu * 4usize;
            let mut row4 = SharedMemory::<f32>::new(4usize);
            row4[0usize] = temp[s];
            row4[1usize] = temp[s + 1usize];
            row4[2usize] = temp[s + 2usize];
            row4[3usize] = temp[s + 3usize];
            fwd_dct4(&mut row4, 0u32);
            // output[(y + iy*2)*8 + col] = row4[iy] * 0.25
            let mut iy: u32 = 0u32;
            while iy < 4u32 {
                let iyu = iy as usize;
                out[(yu + iyu * 2usize) * 8usize + cu] = row4[iyu] * 0.25f32;
                iy += 1u32;
            }
            col += 1u32;
        }
        y += 1u32;
    }

    // DC combine: (out[0]+out[8])/2, (out[0]-out[8])/2
    let dc0 = out[0usize];
    let dc1 = out[8usize];
    out[0usize] = (dc0 + dc1) * 0.5f32;
    out[8usize] = (dc0 - dc1) * 0.5f32;

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        output[off + iu] = out[iu];
        i += 1u32;
    }
}

#[cube(launch_unchecked)]
pub fn idct_4x8_full_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;

    let mut coeffs = SharedMemory::<f32>::new(64usize);
    let mut block = SharedMemory::<f32>::new(32usize);
    let mut temp = SharedMemory::<f32>::new(32usize);

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        coeffs[iu] = input[off + iu];
        i += 1u32;
    }

    // Reverse DC averaging
    let combined_dc = coeffs[0usize];
    let combined_ac = coeffs[8usize];
    coeffs[0usize] = combined_dc + combined_ac;
    coeffs[8usize] = combined_dc - combined_ac;

    let mut y: u32 = 0u32;
    while y < 2u32 {
        // Gather: block[iy*8 + ix] = coeffs[(y + iy*2)*8 + ix]
        let mut iy: u32 = 0u32;
        while iy < 4u32 {
            let mut ix: u32 = 0u32;
            while ix < 8u32 {
                let yu = y as usize;
                let iyu = iy as usize;
                let ixu = ix as usize;
                block[iyu * 8usize + ixu] = coeffs[(yu + iyu * 2usize) * 8usize + ixu];
                ix += 1u32;
            }
            iy += 1u32;
        }

        // Col pass first: for col in 0..8, IDCT4 with input (a, c, b, d) where
        // a=block[col]*4, b=block[8+col]*4, c=block[16+col]*4, d=block[24+col]*4.
        // Store transposed: temp[row*8 + col] = result[row]
        let mut col: u32 = 0u32;
        while col < 8u32 {
            let cu = col as usize;
            let mut row4 = SharedMemory::<f32>::new(4usize);
            row4[0usize] = block[cu] * 4.0f32;
            row4[1usize] = block[16usize + cu] * 4.0f32;
            row4[2usize] = block[8usize + cu] * 4.0f32;
            row4[3usize] = block[24usize + cu] * 4.0f32;
            inv_idct4(&mut row4, 0u32);
            temp[cu] = row4[0usize];
            temp[8usize + cu] = row4[1usize];
            temp[16usize + cu] = row4[2usize];
            temp[24usize + cu] = row4[3usize];
            col += 1u32;
        }

        // Row pass: per row of 8, multiply by 8, IDCT8-core, store to output
        let mut r: u32 = 0u32;
        while r < 4u32 {
            let yu = y as usize;
            let ru = r as usize;
            let s = ru * 8usize;
            let mut row8 = SharedMemory::<f32>::new(8usize);
            let mut k: u32 = 0u32;
            while k < 8u32 {
                let ku = k as usize;
                row8[ku] = temp[s + ku] * 8.0f32;
                k += 1u32;
            }
            inv_idct8_core(&mut row8, 0u32);
            let mut ix: u32 = 0u32;
            while ix < 8u32 {
                let ixu = ix as usize;
                output[off + (yu * 4usize + ru) * 8usize + ixu] = row8[ixu];
                ix += 1u32;
            }
            r += 1u32;
        }
        y += 1u32;
    }
}

// =============================================================================
// dct_8x4_full / idct_8x4_full
// =============================================================================

#[cube(launch_unchecked)]
pub fn dct_8x4_full_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;

    let mut tile = SharedMemory::<f32>::new(64usize);
    let mut out = SharedMemory::<f32>::new(64usize);
    let mut temp = SharedMemory::<f32>::new(32usize);
    let mut row = SharedMemory::<f32>::new(4usize);

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        tile[iu] = input[off + iu];
        i += 1u32;
    }

    let mut x: u32 = 0u32;
    while x < 2u32 {
        // Row pass: for iy in 0..8, 4-pt DCT on tile[iy*8 + x*4 .. +4],
        // scale 0.25, store transposed: temp[col*8 + iy] = result[col]*0.25
        let mut iy: u32 = 0u32;
        while iy < 8u32 {
            let xu = x as usize;
            let iyu = iy as usize;
            let base = iyu * 8usize + xu * 4usize;
            row[0usize] = tile[base];
            row[1usize] = tile[base + 1usize];
            row[2usize] = tile[base + 2usize];
            row[3usize] = tile[base + 3usize];
            fwd_dct4(&mut row, 0u32);
            temp[iyu] = row[0usize] * 0.25f32;
            temp[8usize + iyu] = row[1usize] * 0.25f32;
            temp[16usize + iyu] = row[2usize] * 0.25f32;
            temp[24usize + iyu] = row[3usize] * 0.25f32;
            iy += 1u32;
        }
        // Col pass: for col in 0..4, 8-pt DCT on temp[col*8..+8],
        // scale 0.125, store: output[(x + col*2)*8 + ix] = result[ix]*0.125
        let mut col: u32 = 0u32;
        while col < 4u32 {
            let xu = x as usize;
            let cu = col as usize;
            let s = cu * 8usize;
            let mut row8 = SharedMemory::<f32>::new(8usize);
            let mut k: u32 = 0u32;
            while k < 8u32 {
                let ku = k as usize;
                row8[ku] = temp[s + ku];
                k += 1u32;
            }
            fwd_dct8(&mut row8, 0u32);
            let mut ix: u32 = 0u32;
            while ix < 8u32 {
                let ixu = ix as usize;
                out[(xu + cu * 2usize) * 8usize + ixu] = row8[ixu] * 0.125f32;
                ix += 1u32;
            }
            col += 1u32;
        }
        x += 1u32;
    }

    // DC combine: (out[0]+out[8])/2, (out[0]-out[8])/2
    let dc0 = out[0usize];
    let dc1 = out[8usize];
    out[0usize] = (dc0 + dc1) * 0.5f32;
    out[8usize] = (dc0 - dc1) * 0.5f32;

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        output[off + iu] = out[iu];
        i += 1u32;
    }
}

#[cube(launch_unchecked)]
pub fn idct_8x4_full_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;

    let mut coeffs = SharedMemory::<f32>::new(64usize);
    let mut block = SharedMemory::<f32>::new(32usize);
    let mut temp = SharedMemory::<f32>::new(32usize);

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        coeffs[iu] = input[off + iu];
        i += 1u32;
    }

    // Reverse DC averaging
    let combined_dc = coeffs[0usize];
    let combined_ac = coeffs[8usize];
    coeffs[0usize] = combined_dc + combined_ac;
    coeffs[8usize] = combined_dc - combined_ac;

    let mut x: u32 = 0u32;
    while x < 2u32 {
        // Gather: block[iy*8 + ix] = coeffs[(x + iy*2)*8 + ix]
        let mut iy: u32 = 0u32;
        while iy < 4u32 {
            let mut ix: u32 = 0u32;
            while ix < 8u32 {
                let xu = x as usize;
                let iyu = iy as usize;
                let ixu = ix as usize;
                block[iyu * 8usize + ixu] = coeffs[(xu + iyu * 2usize) * 8usize + ixu];
                ix += 1u32;
            }
            iy += 1u32;
        }

        // Row pass: for r in 0..4, multiply by 8, IDCT8-core on block[r*8..+8],
        // store transposed: temp[col*4 + r] = result[col]
        let mut r: u32 = 0u32;
        while r < 4u32 {
            let ru = r as usize;
            let s = ru * 8usize;
            let mut row8 = SharedMemory::<f32>::new(8usize);
            let mut k: u32 = 0u32;
            while k < 8u32 {
                let ku = k as usize;
                row8[ku] = block[s + ku] * 8.0f32;
                k += 1u32;
            }
            inv_idct8_core(&mut row8, 0u32);
            let mut col: u32 = 0u32;
            while col < 8u32 {
                let cu = col as usize;
                temp[cu * 4usize + ru] = row8[cu];
                col += 1u32;
            }
            r += 1u32;
        }

        // Col pass: matches CPU `for row in 0..8` (the outer loop iterates
        // 8 rows of temp[]). My loop variable `r2` corresponds to CPU's `row`.
        // CPU output index is `row * 8 + (x * 4 + ix)` — we mirror that exactly.
        let mut r2: u32 = 0u32;
        while r2 < 8u32 {
            let xu = x as usize;
            let r2u = r2 as usize;
            let s = r2u * 4usize;
            let mut row4 = SharedMemory::<f32>::new(4usize);
            row4[0usize] = temp[s] * 4.0f32;
            row4[1usize] = temp[s + 2usize] * 4.0f32;
            row4[2usize] = temp[s + 1usize] * 4.0f32;
            row4[3usize] = temp[s + 3usize] * 4.0f32;
            inv_idct4(&mut row4, 0u32);
            let mut ix: u32 = 0u32;
            while ix < 4u32 {
                let ixu = ix as usize;
                output[off + r2u * 8usize + (xu * 4usize + ixu)] = row4[ixu];
                ix += 1u32;
            }
            r2 += 1u32;
        }
        x += 1u32;
    }
}
