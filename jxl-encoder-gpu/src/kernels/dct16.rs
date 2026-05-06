// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! 16x16 forward and inverse DCT.
//!
//! Mirrors `jxl_encoder_simd::dct16::dct_16x16_scalar` and
//! `jxl_encoder_simd::idct16::idct_16x16_scalar`.
//!
//! Strategy: one cube per 16x16 block (cube_dim = 1). Per-block scratch in
//! `SharedMemory<f32>::new(256)` (single-threaded → register/local memory
//! after codegen). Recursive butterflies inlined: dct1d_16 → dct1d_8 →
//! dct1d_4 → dct1d_2 (and idct mirror).

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

const SQRT2: f32 = core::f32::consts::SQRT_2;
const ONE_OVER_SQRT2: f32 = 0.707_106_77;
const ONE_OVER_16: f32 = 0.062_5;
const HALF: f32 = 0.5;
const SIXTEEN: f32 = 16.0;

const WC4_0: f32 = 0.541_196_1;
const WC4_1: f32 = 1.306_563;
const WC8_0: f32 = 0.509_795_6;
const WC8_1: f32 = 0.601_344_9;
const WC8_2: f32 = 0.899_976_2;
const WC8_3: f32 = 2.562_915_5;
#[allow(clippy::excessive_precision)]
const WC16_0: f32 = 0.502_419_3;
const WC16_1: f32 = 0.522_498_6;
#[allow(clippy::excessive_precision)]
const WC16_2: f32 = 0.566_944_06;
const WC16_3: f32 = 0.646_821_8;
#[allow(clippy::excessive_precision)]
const WC16_4: f32 = 0.788_154_65;
const WC16_5: f32 = 1.060_677_7;
const WC16_6: f32 = 1.722_447_1;
const WC16_7: f32 = 5.101_148_6;
const INV_WC4_0: f32 = 1.0 / WC4_0;
const INV_WC4_1: f32 = 1.0 / WC4_1;
const INV_WC8_0: f32 = 1.0 / WC8_0;
const INV_WC8_1: f32 = 1.0 / WC8_1;
const INV_WC8_2: f32 = 1.0 / WC8_2;
const INV_WC8_3: f32 = 1.0 / WC8_3;
const INV_WC16_0: f32 = 1.0 / WC16_0;
const INV_WC16_1: f32 = 1.0 / WC16_1;
const INV_WC16_2: f32 = 1.0 / WC16_2;
const INV_WC16_3: f32 = 1.0 / WC16_3;
const INV_WC16_4: f32 = 1.0 / WC16_4;
const INV_WC16_5: f32 = 1.0 / WC16_5;
const INV_WC16_6: f32 = 1.0 / WC16_6;
const INV_WC16_7: f32 = 1.0 / WC16_7;

// =============================================================================
// Forward 16x16 DCT
// =============================================================================

/// Forward 1D 4-point DCT, in-place at offset `base`.
#[cube]
fn fwd_dct1d_4(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    let m0 = mem[b];
    let m1 = mem[b + 1usize];
    let m2 = mem[b + 2usize];
    let m3 = mem[b + 3usize];

    // AddReverse / SubReverse
    let t0 = m0 + m3;
    let t1 = m1 + m2;
    let mut t2 = m0 - m3;
    let mut t3 = m1 - m2;

    // DCT-2 on first half: [t0, t1] -> [t0+t1, t0-t1]
    let a0 = t0 + t1;
    let a1 = t0 - t1;

    // Multiply second half by WC4
    t2 = t2 * WC4_0;
    t3 = t3 * WC4_1;

    // DCT-2 on second half: [t2, t3] -> [t2+t3, t2-t3]
    let b0 = t2 + t3;
    let b1 = t2 - t3;

    // B transform on second half: b0 = SQRT2*b0 + b1
    let b0p = SQRT2 * b0 + b1;

    // InverseEvenOdd interleave
    mem[b] = a0;
    mem[b + 2usize] = a1;
    mem[b + 1usize] = b0p;
    mem[b + 3usize] = b1;
}

/// Forward 1D 8-point DCT, in-place at offset `base`.
#[cube]
fn fwd_dct1d_8(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    let m0 = mem[b];
    let m1 = mem[b + 1usize];
    let m2 = mem[b + 2usize];
    let m3 = mem[b + 3usize];
    let m4 = mem[b + 4usize];
    let m5 = mem[b + 5usize];
    let m6 = mem[b + 6usize];
    let m7 = mem[b + 7usize];

    // AddReverse / SubReverse
    let mut t0 = m0 + m7;
    let mut t1 = m1 + m6;
    let mut t2 = m2 + m5;
    let mut t3 = m3 + m4;
    let mut t4 = m0 - m7;
    let mut t5 = m1 - m6;
    let mut t6 = m2 - m5;
    let mut t7 = m3 - m4;

    // dct1d_4 on [t0..t4]
    let a0 = t0 + t3;
    let a1 = t1 + t2;
    let a2 = t0 - t3;
    let a3 = t1 - t2;
    let p0 = a0 + a1;
    let p1 = a0 - a1;
    let a2s = a2 * WC4_0;
    let a3s = a3 * WC4_1;
    let q0 = a2s + a3s;
    let q1 = a2s - a3s;
    let q0p = SQRT2 * q0 + q1;
    t0 = p0;
    t2 = p1;
    t1 = q0p;
    t3 = q1;

    // Multiply second half by WC8
    t4 = t4 * WC8_0;
    t5 = t5 * WC8_1;
    t6 = t6 * WC8_2;
    t7 = t7 * WC8_3;

    // dct1d_4 on [t4..t8]
    let a0 = t4 + t7;
    let a1 = t5 + t6;
    let a2 = t4 - t7;
    let a3 = t5 - t6;
    let p0 = a0 + a1;
    let p1 = a0 - a1;
    let a2s = a2 * WC4_0;
    let a3s = a3 * WC4_1;
    let q0 = a2s + a3s;
    let q1 = a2s - a3s;
    let q0p = SQRT2 * q0 + q1;
    t4 = p0;
    t6 = p1;
    t5 = q0p;
    t7 = q1;

    // B transform on second half
    t4 = SQRT2 * t4 + t5;
    t5 = t5 + t6;
    t6 = t6 + t7;

    // InverseEvenOdd
    mem[b] = t0;
    mem[b + 1usize] = t4;
    mem[b + 2usize] = t1;
    mem[b + 3usize] = t5;
    mem[b + 4usize] = t2;
    mem[b + 5usize] = t6;
    mem[b + 6usize] = t3;
    mem[b + 7usize] = t7;
}

/// Forward 1D 16-point DCT, in-place at offset `base` (no scaling — caller
/// applies 1/16).
#[cube]
fn fwd_dct1d_16(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    let m0 = mem[b];
    let m1 = mem[b + 1usize];
    let m2 = mem[b + 2usize];
    let m3 = mem[b + 3usize];
    let m4 = mem[b + 4usize];
    let m5 = mem[b + 5usize];
    let m6 = mem[b + 6usize];
    let m7 = mem[b + 7usize];
    let m8 = mem[b + 8usize];
    let m9 = mem[b + 9usize];
    let ma = mem[b + 10usize];
    let mb = mem[b + 11usize];
    let mc = mem[b + 12usize];
    let md = mem[b + 13usize];
    let me = mem[b + 14usize];
    let mf = mem[b + 15usize];

    // AddReverse for first half (8 elements)
    let mut h0 = m0 + mf;
    let mut h1 = m1 + me;
    let mut h2 = m2 + md;
    let mut h3 = m3 + mc;
    let mut h4 = m4 + mb;
    let mut h5 = m5 + ma;
    let mut h6 = m6 + m9;
    let mut h7 = m7 + m8;
    // SubReverse for second half
    let mut k0 = m0 - mf;
    let mut k1 = m1 - me;
    let mut k2 = m2 - md;
    let mut k3 = m3 - mc;
    let mut k4 = m4 - mb;
    let mut k5 = m5 - ma;
    let mut k6 = m6 - m9;
    let mut k7 = m7 - m8;

    // dct1d_8 on first half [h0..h8] — inline
    {
        let mut t0 = h0 + h7;
        let mut t1 = h1 + h6;
        let mut t2 = h2 + h5;
        let mut t3 = h3 + h4;
        let mut t4 = h0 - h7;
        let mut t5 = h1 - h6;
        let mut t6 = h2 - h5;
        let mut t7 = h3 - h4;

        let a0 = t0 + t3;
        let a1 = t1 + t2;
        let a2 = t0 - t3;
        let a3 = t1 - t2;
        let p0 = a0 + a1;
        let p1 = a0 - a1;
        let a2s = a2 * WC4_0;
        let a3s = a3 * WC4_1;
        let q0 = a2s + a3s;
        let q1 = a2s - a3s;
        let q0p = SQRT2 * q0 + q1;
        t0 = p0;
        t2 = p1;
        t1 = q0p;
        t3 = q1;

        t4 = t4 * WC8_0;
        t5 = t5 * WC8_1;
        t6 = t6 * WC8_2;
        t7 = t7 * WC8_3;

        let a0 = t4 + t7;
        let a1 = t5 + t6;
        let a2 = t4 - t7;
        let a3 = t5 - t6;
        let p0 = a0 + a1;
        let p1 = a0 - a1;
        let a2s = a2 * WC4_0;
        let a3s = a3 * WC4_1;
        let q0 = a2s + a3s;
        let q1 = a2s - a3s;
        let q0p = SQRT2 * q0 + q1;
        t4 = p0;
        t6 = p1;
        t5 = q0p;
        t7 = q1;

        t4 = SQRT2 * t4 + t5;
        t5 = t5 + t6;
        t6 = t6 + t7;

        h0 = t0;
        h1 = t4;
        h2 = t1;
        h3 = t5;
        h4 = t2;
        h5 = t6;
        h6 = t3;
        h7 = t7;
    }

    // Multiply second half by WC16
    k0 = k0 * WC16_0;
    k1 = k1 * WC16_1;
    k2 = k2 * WC16_2;
    k3 = k3 * WC16_3;
    k4 = k4 * WC16_4;
    k5 = k5 * WC16_5;
    k6 = k6 * WC16_6;
    k7 = k7 * WC16_7;

    // dct1d_8 on second half [k0..k8] — inline
    {
        let mut t0 = k0 + k7;
        let mut t1 = k1 + k6;
        let mut t2 = k2 + k5;
        let mut t3 = k3 + k4;
        let mut t4 = k0 - k7;
        let mut t5 = k1 - k6;
        let mut t6 = k2 - k5;
        let mut t7 = k3 - k4;

        let a0 = t0 + t3;
        let a1 = t1 + t2;
        let a2 = t0 - t3;
        let a3 = t1 - t2;
        let p0 = a0 + a1;
        let p1 = a0 - a1;
        let a2s = a2 * WC4_0;
        let a3s = a3 * WC4_1;
        let q0 = a2s + a3s;
        let q1 = a2s - a3s;
        let q0p = SQRT2 * q0 + q1;
        t0 = p0;
        t2 = p1;
        t1 = q0p;
        t3 = q1;

        t4 = t4 * WC8_0;
        t5 = t5 * WC8_1;
        t6 = t6 * WC8_2;
        t7 = t7 * WC8_3;

        let a0 = t4 + t7;
        let a1 = t5 + t6;
        let a2 = t4 - t7;
        let a3 = t5 - t6;
        let p0 = a0 + a1;
        let p1 = a0 - a1;
        let a2s = a2 * WC4_0;
        let a3s = a3 * WC4_1;
        let q0 = a2s + a3s;
        let q1 = a2s - a3s;
        let q0p = SQRT2 * q0 + q1;
        t4 = p0;
        t6 = p1;
        t5 = q0p;
        t7 = q1;

        t4 = SQRT2 * t4 + t5;
        t5 = t5 + t6;
        t6 = t6 + t7;

        k0 = t0;
        k1 = t4;
        k2 = t1;
        k3 = t5;
        k4 = t2;
        k5 = t6;
        k6 = t3;
        k7 = t7;
    }

    // B transform on second half: k0 = SQRT2*k0 + k1; k_i += k_{i+1} for i=1..7
    k0 = SQRT2 * k0 + k1;
    k1 = k1 + k2;
    k2 = k2 + k3;
    k3 = k3 + k4;
    k4 = k4 + k5;
    k5 = k5 + k6;
    k6 = k6 + k7;

    // InverseEvenOdd: mem[2i] = h[i], mem[2i+1] = k[i]
    mem[b] = h0;
    mem[b + 1usize] = k0;
    mem[b + 2usize] = h1;
    mem[b + 3usize] = k1;
    mem[b + 4usize] = h2;
    mem[b + 5usize] = k2;
    mem[b + 6usize] = h3;
    mem[b + 7usize] = k3;
    mem[b + 8usize] = h4;
    mem[b + 9usize] = k4;
    mem[b + 10usize] = h5;
    mem[b + 11usize] = k5;
    mem[b + 12usize] = h6;
    mem[b + 13usize] = k6;
    mem[b + 14usize] = h7;
    mem[b + 15usize] = k7;
}

/// Forward 16x16 DCT. One cube per block. `input`/`output` each
/// `num_blocks * 256` floats.
#[cube(launch_unchecked)]
pub fn dct_16x16_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 256usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 256usize;

    let mut scratch = SharedMemory::<f32>::new(256usize);
    let mut transposed = SharedMemory::<f32>::new(256usize);

    // Load + per-row DCT + scale 1/16
    let mut r: u32 = 0u32;
    while r < 16u32 {
        let row_off = r * 16u32;
        let row_off_us = row_off as usize;
        let mut c: u32 = 0u32;
        while c < 16u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = input[off + row_off_us + cu];
            c += 1u32;
        }
        fwd_dct1d_16(&mut scratch, row_off);
        let mut c: u32 = 0u32;
        while c < 16u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = scratch[row_off_us + cu] * ONE_OVER_16;
            c += 1u32;
        }
        r += 1u32;
    }

    // Transpose
    let mut r: u32 = 0u32;
    while r < 16u32 {
        let mut c: u32 = 0u32;
        while c < 16u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 16usize + ru] = scratch[ru * 16usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // Per-row DCT on transposed + scale 1/16
    let mut r: u32 = 0u32;
    while r < 16u32 {
        let row_off = r * 16u32;
        let row_off_us = row_off as usize;
        fwd_dct1d_16(&mut transposed, row_off);
        let mut c: u32 = 0u32;
        while c < 16u32 {
            let cu = c as usize;
            transposed[row_off_us + cu] = transposed[row_off_us + cu] * ONE_OVER_16;
            c += 1u32;
        }
        r += 1u32;
    }

    // Write out (no final transpose for square blocks)
    let mut i: u32 = 0u32;
    while i < 256u32 {
        let iu = i as usize;
        output[off + iu] = transposed[iu];
        i += 1u32;
    }
}

// =============================================================================
// Inverse 16x16 DCT
// =============================================================================

/// Inverse 1D 4-point DCT, in-place at offset `base`.
#[cube]
fn inv_idct1d_4(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    // De-interleave: even -> first half, odd -> second half
    let t0 = mem[b];
    let t1 = mem[b + 2usize];
    let mut t2 = mem[b + 1usize];
    let t3 = mem[b + 3usize];

    // Reverse B transform on second half: t2 = (t2 - t3) / SQRT2
    t2 = (t2 - t3) * ONE_OVER_SQRT2;

    // IDCT-2 on second half [t2, t3]: outputs (t2+t3)/2, (t2-t3)/2
    let s2 = (t2 + t3) * HALF;
    let s3 = (t2 - t3) * HALF;

    // Divide by WC4
    let s2 = s2 * INV_WC4_0;
    let s3 = s3 * INV_WC4_1;

    // IDCT-2 on first half [t0, t1]
    let s0 = (t0 + t1) * HALF;
    let s1 = (t0 - t1) * HALF;

    // Combine
    mem[b] = (s0 + s2) * HALF;
    mem[b + 3usize] = (s0 - s2) * HALF;
    mem[b + 1usize] = (s1 + s3) * HALF;
    mem[b + 2usize] = (s1 - s3) * HALF;
}

/// IDCT-8 core (no N scaling), in-place at offset `base`.
#[cube]
fn inv_idct1d_8_core(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    // De-interleave
    let mut t0 = mem[b];
    let mut t1 = mem[b + 2usize];
    let mut t2 = mem[b + 4usize];
    let mut t3 = mem[b + 6usize];
    let mut t4 = mem[b + 1usize];
    let mut t5 = mem[b + 3usize];
    let mut t6 = mem[b + 5usize];
    let mut t7 = mem[b + 7usize];

    // Reverse B transform: t6 -= t7; t5 -= t6; t4 = (t4 - t5) / SQRT2
    t6 = t6 - t7;
    t5 = t5 - t6;
    t4 = (t4 - t5) * ONE_OVER_SQRT2;

    // IDCT-4 on second half [t4, t5, t6, t7]
    {
        let f0 = t4;
        let f1 = t6;
        let mut f2 = t5;
        let f3 = t7;

        f2 = (f2 - f3) * ONE_OVER_SQRT2;
        let g2 = (f2 + f3) * HALF;
        let g3 = (f2 - f3) * HALF;
        let g2 = g2 * INV_WC4_0;
        let g3 = g3 * INV_WC4_1;
        let g0 = (f0 + f1) * HALF;
        let g1 = (f0 - f1) * HALF;

        t4 = (g0 + g2) * HALF;
        t7 = (g0 - g2) * HALF;
        t5 = (g1 + g3) * HALF;
        t6 = (g1 - g3) * HALF;
    }

    // Divide by WC8
    t4 = t4 * INV_WC8_0;
    t5 = t5 * INV_WC8_1;
    t6 = t6 * INV_WC8_2;
    t7 = t7 * INV_WC8_3;

    // IDCT-4 on first half [t0, t1, t2, t3]
    {
        let f0 = t0;
        let f1 = t2;
        let mut f2 = t1;
        let f3 = t3;

        f2 = (f2 - f3) * ONE_OVER_SQRT2;
        let g2 = (f2 + f3) * HALF;
        let g3 = (f2 - f3) * HALF;
        let g2 = g2 * INV_WC4_0;
        let g3 = g3 * INV_WC4_1;
        let g0 = (f0 + f1) * HALF;
        let g1 = (f0 - f1) * HALF;

        t0 = (g0 + g2) * HALF;
        t3 = (g0 - g2) * HALF;
        t1 = (g1 + g3) * HALF;
        t2 = (g1 - g3) * HALF;
    }

    // Combine: mem[i] = (t[i] + t[4+i]) / 2; mem[7-i] = (t[i] - t[4+i]) / 2
    mem[b] = (t0 + t4) * HALF;
    mem[b + 7usize] = (t0 - t4) * HALF;
    mem[b + 1usize] = (t1 + t5) * HALF;
    mem[b + 6usize] = (t1 - t5) * HALF;
    mem[b + 2usize] = (t2 + t6) * HALF;
    mem[b + 5usize] = (t2 - t6) * HALF;
    mem[b + 3usize] = (t3 + t7) * HALF;
    mem[b + 4usize] = (t3 - t7) * HALF;
}

/// Inverse 1D 16-point IDCT, in-place at offset `base`. Includes *=16
/// scaling (matches CPU `idct1d_16_scalar`).
///
/// Implementation strategy: copy the 16 elements through small
/// `SharedMemory` scratches so the recursive `inv_idct1d_8_core` calls
/// can operate on them. Single-threaded cube → these become register/local
/// memory after codegen.
#[cube]
fn inv_idct1d_16(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;

    // Multiply by 16
    let mut i: u32 = 0u32;
    while i < 16u32 {
        let iu = i as usize;
        mem[b + iu] = mem[b + iu] * SIXTEEN;
        i += 1u32;
    }

    // De-interleave: even -> first[0..8], odd -> second[0..8]
    let mut first = SharedMemory::<f32>::new(8usize);
    let mut second = SharedMemory::<f32>::new(8usize);
    let mut i: u32 = 0u32;
    while i < 8u32 {
        let iu = i as usize;
        first[iu] = mem[b + 2usize * iu];
        second[iu] = mem[b + 2usize * iu + 1usize];
        i += 1u32;
    }

    // Reverse B transform on second half: for i in (1..7).rev(): s[i] -= s[i+1]
    // then s[0] = (s[0] - s[1]) / SQRT2
    second[6usize] = second[6usize] - second[7usize];
    second[5usize] = second[5usize] - second[6usize];
    second[4usize] = second[4usize] - second[5usize];
    second[3usize] = second[3usize] - second[4usize];
    second[2usize] = second[2usize] - second[3usize];
    second[1usize] = second[1usize] - second[2usize];
    second[0usize] = (second[0usize] - second[1usize]) * ONE_OVER_SQRT2;

    // IDCT-8 core on second half (in scratch)
    inv_idct1d_8_core(&mut second, 0u32);

    // Divide by WC16
    second[0usize] = second[0usize] * INV_WC16_0;
    second[1usize] = second[1usize] * INV_WC16_1;
    second[2usize] = second[2usize] * INV_WC16_2;
    second[3usize] = second[3usize] * INV_WC16_3;
    second[4usize] = second[4usize] * INV_WC16_4;
    second[5usize] = second[5usize] * INV_WC16_5;
    second[6usize] = second[6usize] * INV_WC16_6;
    second[7usize] = second[7usize] * INV_WC16_7;

    // IDCT-8 core on first half
    inv_idct1d_8_core(&mut first, 0u32);

    // Combine: mem[i] = (first[i] + second[i]) / 2;
    //          mem[15-i] = (first[i] - second[i]) / 2
    let mut i: u32 = 0u32;
    while i < 8u32 {
        let iu = i as usize;
        let f = first[iu];
        let s = second[iu];
        mem[b + iu] = (f + s) * HALF;
        mem[b + 15usize - iu] = (f - s) * HALF;
        i += 1u32;
    }
}

/// Inverse 16x16 DCT. One cube per block.
#[cube(launch_unchecked)]
pub fn idct_16x16_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 256usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 256usize;

    let mut scratch = SharedMemory::<f32>::new(256usize);
    let mut transposed = SharedMemory::<f32>::new(256usize);

    let mut i: u32 = 0u32;
    while i < 256u32 {
        let iu = i as usize;
        scratch[iu] = input[off + iu];
        i += 1u32;
    }

    // Per-row IDCT
    let mut r: u32 = 0u32;
    while r < 16u32 {
        inv_idct1d_16(&mut scratch, r * 16u32);
        r += 1u32;
    }

    // Transpose
    let mut r: u32 = 0u32;
    while r < 16u32 {
        let mut c: u32 = 0u32;
        while c < 16u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 16usize + ru] = scratch[ru * 16usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // Per-row IDCT on transposed
    let mut r: u32 = 0u32;
    while r < 16u32 {
        inv_idct1d_16(&mut transposed, r * 16u32);
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 256u32 {
        let iu = i as usize;
        output[off + iu] = transposed[iu];
        i += 1u32;
    }
}

// Suppress fwd_dct1d_4/8 unused-helper warnings (kept in source as
// reference for the inlined butterfly logic above; future cooperative
// kernels will call them directly).
#[allow(dead_code)]
#[cube]
fn _unused_keepalive(mem: &mut SharedMemory<f32>) {
    fwd_dct1d_4(mem, 0u32);
    fwd_dct1d_8(mem, 0u32);
}
