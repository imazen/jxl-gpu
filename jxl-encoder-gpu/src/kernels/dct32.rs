// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! 32x32 / 32x16 / 16x32 forward and inverse DCT.
//!
//! Mirrors `jxl_encoder_simd::dct32::*_scalar` and `idct32::*_scalar`.
//!
//! Strategy: one cube per block (cube_dim=1). Per-block scratch in
//! `SharedMemory<f32>` (1024 floats for 32x32, 512 for rectangular).
//! Reuses `fwd_dct1d_16` / `fwd_dct1d_8` / `inv_idct1d_16_core` /
//! `inv_idct1d_8_core` helpers from `kernels::dct16`.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

use crate::kernels::dct16::{fwd_dct1d_16, inv_idct1d_16_core};

const SQRT2: f32 = core::f32::consts::SQRT_2;
const ONE_OVER_SQRT2: f32 = 0.707_106_77;
const ONE_OVER_16: f32 = 0.062_5;
const ONE_OVER_32: f32 = 0.031_25;
const HALF: f32 = 0.5;

#[allow(clippy::excessive_precision)]
const WC32_0: f32 = 0.500_602_98;
#[allow(clippy::excessive_precision)]
const WC32_1: f32 = 0.505_471;
#[allow(clippy::excessive_precision)]
const WC32_2: f32 = 0.515_447_3;
#[allow(clippy::excessive_precision)]
const WC32_3: f32 = 0.531_042_6;
#[allow(clippy::excessive_precision)]
const WC32_4: f32 = 0.553_103_9;
#[allow(clippy::excessive_precision)]
const WC32_5: f32 = 0.582_935;
#[allow(clippy::excessive_precision)]
const WC32_6: f32 = 0.622_504_1;
#[allow(clippy::excessive_precision)]
const WC32_7: f32 = 0.674_808_36;
#[allow(clippy::excessive_precision)]
const WC32_8: f32 = 0.744_536_27;
#[allow(clippy::excessive_precision)]
const WC32_9: f32 = 0.839_349_64;
#[allow(clippy::excessive_precision)]
const WC32_10: f32 = 0.972_568_25;
#[allow(clippy::excessive_precision)]
const WC32_11: f32 = 1.169_439_9;
#[allow(clippy::excessive_precision)]
const WC32_12: f32 = 1.484_164_6;
#[allow(clippy::excessive_precision)]
const WC32_13: f32 = 2.057_781;
#[allow(clippy::excessive_precision)]
const WC32_14: f32 = 3.407_608_4;
#[allow(clippy::excessive_precision)]
const WC32_15: f32 = 10.190_008;

const INV_WC32_0: f32 = 1.0 / WC32_0;
const INV_WC32_1: f32 = 1.0 / WC32_1;
const INV_WC32_2: f32 = 1.0 / WC32_2;
const INV_WC32_3: f32 = 1.0 / WC32_3;
const INV_WC32_4: f32 = 1.0 / WC32_4;
const INV_WC32_5: f32 = 1.0 / WC32_5;
const INV_WC32_6: f32 = 1.0 / WC32_6;
const INV_WC32_7: f32 = 1.0 / WC32_7;
const INV_WC32_8: f32 = 1.0 / WC32_8;
const INV_WC32_9: f32 = 1.0 / WC32_9;
const INV_WC32_10: f32 = 1.0 / WC32_10;
const INV_WC32_11: f32 = 1.0 / WC32_11;
const INV_WC32_12: f32 = 1.0 / WC32_12;
const INV_WC32_13: f32 = 1.0 / WC32_13;
const INV_WC32_14: f32 = 1.0 / WC32_14;
const INV_WC32_15: f32 = 1.0 / WC32_15;

// =============================================================================
// 32-pt forward DCT helper
// =============================================================================

/// Forward 1D 32-point DCT, in-place at offset `base`. No scaling.
#[cube]
pub(crate) fn fwd_dct1d_32(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;

    // AddReverse first half + SubReverse second half, into 16+16 scratches.
    let mut first = SharedMemory::<f32>::new(16usize);
    let mut second = SharedMemory::<f32>::new(16usize);
    let mut i: u32 = 0u32;
    while i < 16u32 {
        let iu = i as usize;
        let a = mem[b + iu];
        let bv = mem[b + 31usize - iu];
        first[iu] = a + bv;
        second[iu] = a - bv;
        i += 1u32;
    }

    // dct1d_16 on first half
    fwd_dct1d_16(&mut first, 0u32);

    // Multiply second half by WC32
    second[0usize] = second[0usize] * WC32_0;
    second[1usize] = second[1usize] * WC32_1;
    second[2usize] = second[2usize] * WC32_2;
    second[3usize] = second[3usize] * WC32_3;
    second[4usize] = second[4usize] * WC32_4;
    second[5usize] = second[5usize] * WC32_5;
    second[6usize] = second[6usize] * WC32_6;
    second[7usize] = second[7usize] * WC32_7;
    second[8usize] = second[8usize] * WC32_8;
    second[9usize] = second[9usize] * WC32_9;
    second[10usize] = second[10usize] * WC32_10;
    second[11usize] = second[11usize] * WC32_11;
    second[12usize] = second[12usize] * WC32_12;
    second[13usize] = second[13usize] * WC32_13;
    second[14usize] = second[14usize] * WC32_14;
    second[15usize] = second[15usize] * WC32_15;

    // dct1d_16 on second half
    fwd_dct1d_16(&mut second, 0u32);

    // B transform on second half: s[0] = SQRT2*s[0] + s[1]; for i in 1..15: s[i] += s[i+1]
    second[0usize] = SQRT2 * second[0usize] + second[1usize];
    second[1usize] = second[1usize] + second[2usize];
    second[2usize] = second[2usize] + second[3usize];
    second[3usize] = second[3usize] + second[4usize];
    second[4usize] = second[4usize] + second[5usize];
    second[5usize] = second[5usize] + second[6usize];
    second[6usize] = second[6usize] + second[7usize];
    second[7usize] = second[7usize] + second[8usize];
    second[8usize] = second[8usize] + second[9usize];
    second[9usize] = second[9usize] + second[10usize];
    second[10usize] = second[10usize] + second[11usize];
    second[11usize] = second[11usize] + second[12usize];
    second[12usize] = second[12usize] + second[13usize];
    second[13usize] = second[13usize] + second[14usize];
    second[14usize] = second[14usize] + second[15usize];

    // InverseEvenOdd interleave: mem[2i] = first[i], mem[2i+1] = second[i]
    let mut i: u32 = 0u32;
    while i < 16u32 {
        let iu = i as usize;
        mem[b + 2usize * iu] = first[iu];
        mem[b + 2usize * iu + 1usize] = second[iu];
        i += 1u32;
    }
}

/// Inverse 1D 32-point IDCT core (no scaling).
#[cube]
pub(crate) fn inv_idct1d_32_core(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;

    // De-interleave: even -> first[0..16], odd -> second[0..16]
    let mut first = SharedMemory::<f32>::new(16usize);
    let mut second = SharedMemory::<f32>::new(16usize);
    let mut i: u32 = 0u32;
    while i < 16u32 {
        let iu = i as usize;
        first[iu] = mem[b + 2usize * iu];
        second[iu] = mem[b + 2usize * iu + 1usize];
        i += 1u32;
    }

    // Reverse B transform on second half: for i in (1..15).rev(): s[i] -= s[i+1]
    // then s[0] = (s[0] - s[1]) / SQRT2
    second[14usize] = second[14usize] - second[15usize];
    second[13usize] = second[13usize] - second[14usize];
    second[12usize] = second[12usize] - second[13usize];
    second[11usize] = second[11usize] - second[12usize];
    second[10usize] = second[10usize] - second[11usize];
    second[9usize] = second[9usize] - second[10usize];
    second[8usize] = second[8usize] - second[9usize];
    second[7usize] = second[7usize] - second[8usize];
    second[6usize] = second[6usize] - second[7usize];
    second[5usize] = second[5usize] - second[6usize];
    second[4usize] = second[4usize] - second[5usize];
    second[3usize] = second[3usize] - second[4usize];
    second[2usize] = second[2usize] - second[3usize];
    second[1usize] = second[1usize] - second[2usize];
    second[0usize] = (second[0usize] - second[1usize]) * ONE_OVER_SQRT2;

    // IDCT-16 core on second half
    inv_idct1d_16_core(&mut second, 0u32);

    // Divide by WC32
    second[0usize] = second[0usize] * INV_WC32_0;
    second[1usize] = second[1usize] * INV_WC32_1;
    second[2usize] = second[2usize] * INV_WC32_2;
    second[3usize] = second[3usize] * INV_WC32_3;
    second[4usize] = second[4usize] * INV_WC32_4;
    second[5usize] = second[5usize] * INV_WC32_5;
    second[6usize] = second[6usize] * INV_WC32_6;
    second[7usize] = second[7usize] * INV_WC32_7;
    second[8usize] = second[8usize] * INV_WC32_8;
    second[9usize] = second[9usize] * INV_WC32_9;
    second[10usize] = second[10usize] * INV_WC32_10;
    second[11usize] = second[11usize] * INV_WC32_11;
    second[12usize] = second[12usize] * INV_WC32_12;
    second[13usize] = second[13usize] * INV_WC32_13;
    second[14usize] = second[14usize] * INV_WC32_14;
    second[15usize] = second[15usize] * INV_WC32_15;

    // IDCT-16 core on first half
    inv_idct1d_16_core(&mut first, 0u32);

    // Combine: mem[i] = (first[i] + second[i]) / 2; mem[31-i] = (first[i] - second[i]) / 2
    let mut i: u32 = 0u32;
    while i < 16u32 {
        let iu = i as usize;
        let f = first[iu];
        let s = second[iu];
        mem[b + iu] = (f + s) * HALF;
        mem[b + 31usize - iu] = (f - s) * HALF;
        i += 1u32;
    }
}

/// Inverse 1D 32-point IDCT with *=32 scaling.
#[cube]
fn inv_idct1d_32(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    let mut i: u32 = 0u32;
    while i < 32u32 {
        let iu = i as usize;
        mem[b + iu] = mem[b + iu] * 32.0f32;
        i += 1u32;
    }
    inv_idct1d_32_core(mem, base);
}

// =============================================================================
// 32x32 forward + inverse
// =============================================================================

#[cube(launch_unchecked)]
pub fn dct_32x32_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 1024usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 1024usize;

    let mut scratch = SharedMemory::<f32>::new(1024usize);
    let mut transposed = SharedMemory::<f32>::new(1024usize);

    let mut r: u32 = 0u32;
    while r < 32u32 {
        let row_off = r * 32u32;
        let row_off_us = row_off as usize;
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = input[off + row_off_us + cu];
            c += 1u32;
        }
        fwd_dct1d_32(&mut scratch, row_off);
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = scratch[row_off_us + cu] * ONE_OVER_32;
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 32u32 {
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 32usize + ru] = scratch[ru * 32usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 32u32 {
        let row_off = r * 32u32;
        let row_off_us = row_off as usize;
        fwd_dct1d_32(&mut transposed, row_off);
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let cu = c as usize;
            transposed[row_off_us + cu] = transposed[row_off_us + cu] * ONE_OVER_32;
            c += 1u32;
        }
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 1024u32 {
        let iu = i as usize;
        output[off + iu] = transposed[iu];
        i += 1u32;
    }
}

#[cube(launch_unchecked)]
pub fn idct_32x32_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 1024usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 1024usize;

    let mut scratch = SharedMemory::<f32>::new(1024usize);
    let mut transposed = SharedMemory::<f32>::new(1024usize);

    let mut i: u32 = 0u32;
    while i < 1024u32 {
        let iu = i as usize;
        scratch[iu] = input[off + iu];
        i += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 32u32 {
        inv_idct1d_32(&mut scratch, r * 32u32);
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 32u32 {
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 32usize + ru] = scratch[ru * 32usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 32u32 {
        inv_idct1d_32(&mut transposed, r * 32u32);
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 1024u32 {
        let iu = i as usize;
        output[off + iu] = transposed[iu];
        i += 1u32;
    }
}

// =============================================================================
// 32x16 forward + inverse
// =============================================================================

/// Forward 32x16 DCT (32 rows × 16 cols). Per-row 16-pt DCT (1/16),
/// transpose to 16x32, per-row 32-pt DCT (1/32). No final transpose.
#[cube(launch_unchecked)]
pub fn dct_32x16_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 512usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 512usize;

    let mut scratch = SharedMemory::<f32>::new(512usize);
    let mut transposed = SharedMemory::<f32>::new(512usize);

    let mut r: u32 = 0u32;
    while r < 32u32 {
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

    // Transpose 32x16 → 16x32: transposed[c*32 + r] = scratch[r*16 + c]
    let mut r: u32 = 0u32;
    while r < 32u32 {
        let mut c: u32 = 0u32;
        while c < 16u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 32usize + ru] = scratch[ru * 16usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // Per-row 32-pt DCT (16 rows × 32 cols) + scale 1/32
    let mut r: u32 = 0u32;
    while r < 16u32 {
        let row_off = r * 32u32;
        let row_off_us = row_off as usize;
        fwd_dct1d_32(&mut transposed, row_off);
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let cu = c as usize;
            transposed[row_off_us + cu] = transposed[row_off_us + cu] * ONE_OVER_32;
            c += 1u32;
        }
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 512u32 {
        let iu = i as usize;
        output[off + iu] = transposed[iu];
        i += 1u32;
    }
}

/// Inverse 32x16 IDCT: input in 16x32 layout, output in 32x16 layout.
#[cube(launch_unchecked)]
pub fn idct_32x16_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 512usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 512usize;

    let mut scratch = SharedMemory::<f32>::new(512usize);

    // IDCT-32 on each of 16 rows (stride 32)
    let mut r: u32 = 0u32;
    while r < 16u32 {
        let row_off = r * 32u32;
        let row_off_us = row_off as usize;
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = input[off + row_off_us + cu];
            c += 1u32;
        }
        inv_idct1d_32(&mut scratch, row_off);
        r += 1u32;
    }

    // Transpose 16x32 → 32x16
    let mut transposed = SharedMemory::<f32>::new(512usize);
    let mut r: u32 = 0u32;
    while r < 16u32 {
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 16usize + ru] = scratch[ru * 32usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // IDCT-16 on each of 32 rows (stride 16) — output goes to global directly
    let mut r: u32 = 0u32;
    while r < 32u32 {
        let row_off = r * 16u32;
        let row_off_us = row_off as usize;
        // Use inv_idct1d_16_core then *16 scale
        // Easier: just use a small scratch and call inv_idct1d_16-equivalent
        let mut row_buf = SharedMemory::<f32>::new(16usize);
        let mut c: u32 = 0u32;
        while c < 16u32 {
            let cu = c as usize;
            row_buf[cu] = transposed[row_off_us + cu] * 16.0f32;
            c += 1u32;
        }
        inv_idct1d_16_core(&mut row_buf, 0u32);
        let mut c: u32 = 0u32;
        while c < 16u32 {
            let cu = c as usize;
            output[off + row_off_us + cu] = row_buf[cu];
            c += 1u32;
        }
        r += 1u32;
    }
}

// =============================================================================
// 16x32 forward + inverse
// =============================================================================

/// Forward 16x32 DCT (16 rows × 32 cols). Per-row 32-pt DCT (1/32),
/// transpose, per-row 16-pt DCT (1/16), FINAL transpose (ROWS < COLS).
#[cube(launch_unchecked)]
pub fn dct_16x32_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 512usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 512usize;

    let mut scratch = SharedMemory::<f32>::new(512usize);
    let mut transposed = SharedMemory::<f32>::new(512usize);

    // Per-row 32-pt DCT + scale 1/32 (16 rows × 32 cols)
    let mut r: u32 = 0u32;
    while r < 16u32 {
        let row_off = r * 32u32;
        let row_off_us = row_off as usize;
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = input[off + row_off_us + cu];
            c += 1u32;
        }
        fwd_dct1d_32(&mut scratch, row_off);
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = scratch[row_off_us + cu] * ONE_OVER_32;
            c += 1u32;
        }
        r += 1u32;
    }

    // Transpose 16x32 → 32x16: transposed[c*16 + r] = scratch[r*32 + c]
    let mut r: u32 = 0u32;
    while r < 16u32 {
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 16usize + ru] = scratch[ru * 32usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // Per-row 16-pt DCT + scale 1/16 (32 rows × 16 cols)
    let mut r: u32 = 0u32;
    while r < 32u32 {
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

    // FINAL transpose 32x16 → 16x32 (ROWS < COLS branch)
    let mut r: u32 = 0u32;
    while r < 32u32 {
        let mut c: u32 = 0u32;
        while c < 16u32 {
            let ru = r as usize;
            let cu = c as usize;
            output[off + cu * 32usize + ru] = transposed[ru * 16usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }
}

/// Inverse 16x32 IDCT: input in 16x32 layout, output in 16x32 layout.
/// CPU: un-transpose 16x32 → 32x16, IDCT-16 on 32 rows, transpose
/// 32x16 → 16x32, IDCT-32 on 16 rows.
#[cube(launch_unchecked)]
pub fn idct_16x32_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 512usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 512usize;

    let mut transposed = SharedMemory::<f32>::new(512usize);
    let mut tmp = SharedMemory::<f32>::new(512usize);

    // Un-transpose 16x32 input → 32x16 in transposed
    let mut r: u32 = 0u32;
    while r < 16u32 {
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 16usize + ru] = input[off + ru * 32usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // IDCT-16 on each of 32 rows of transposed (stride 16)
    let mut r: u32 = 0u32;
    while r < 32u32 {
        let row_off = r * 16u32;
        let row_off_us = row_off as usize;
        let mut row_buf = SharedMemory::<f32>::new(16usize);
        let mut c: u32 = 0u32;
        while c < 16u32 {
            let cu = c as usize;
            row_buf[cu] = transposed[row_off_us + cu] * 16.0f32;
            c += 1u32;
        }
        inv_idct1d_16_core(&mut row_buf, 0u32);
        let mut c: u32 = 0u32;
        while c < 16u32 {
            let cu = c as usize;
            tmp[row_off_us + cu] = row_buf[cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // Transpose 32x16 → 16x32 into a fresh buffer
    let mut transposed2 = SharedMemory::<f32>::new(512usize);
    let mut r: u32 = 0u32;
    while r < 32u32 {
        let mut c: u32 = 0u32;
        while c < 16u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed2[cu * 32usize + ru] = tmp[ru * 16usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // IDCT-32 on each of 16 rows
    let mut r: u32 = 0u32;
    while r < 16u32 {
        inv_idct1d_32(&mut transposed2, r * 32u32);
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 512u32 {
        let iu = i as usize;
        output[off + iu] = transposed2[iu];
        i += 1u32;
    }
}
