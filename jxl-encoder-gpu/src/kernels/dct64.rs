// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! 64x64 / 64x32 / 32x64 forward and inverse DCT.
//!
//! Same recursive pattern as DCT32. Reuses `fwd_dct1d_32`/`inv_idct1d_32_core`
//! from `kernels::dct32` and `fwd_dct1d_16`/`inv_idct1d_16_core` from
//! `kernels::dct16` indirectly.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

use crate::kernels::dct32::{fwd_dct1d_32, inv_idct1d_32_core};

const SQRT2: f32 = core::f32::consts::SQRT_2;
const ONE_OVER_SQRT2: f32 = 0.707_106_77;
const ONE_OVER_32: f32 = 0.031_25;
const ONE_OVER_64: f32 = 0.015_625;
const HALF: f32 = 0.5;

// 32 multipliers for the 64-pt second half. All marked excessive_precision
// — must bit-match CPU reference.
macro_rules! wc64 {
    ($name:ident, $val:expr) => {
        #[allow(clippy::excessive_precision)]
        const $name: f32 = $val;
    };
}

wc64!(WC64_0, 0.500_150_64);
wc64!(WC64_1, 0.501_358_45);
wc64!(WC64_2, 0.503_788_73);
wc64!(WC64_3, 0.507_471_2);
wc64!(WC64_4, 0.512_451_5);
wc64!(WC64_5, 0.518_792_7);
wc64!(WC64_6, 0.526_577_3);
wc64!(WC64_7, 0.535_909_8);
wc64!(WC64_8, 0.546_920_44);
wc64!(WC64_9, 0.559_769_8);
wc64!(WC64_10, 0.574_655_2);
wc64!(WC64_11, 0.591_818_55);
wc64!(WC64_12, 0.611_557_4);
wc64!(WC64_13, 0.634_238_94);
wc64!(WC64_14, 0.660_319_8);
wc64!(WC64_15, 0.690_372_15);
wc64!(WC64_16, 0.725_120_53);
wc64!(WC64_17, 0.765_494_2);
wc64!(WC64_18, 0.812_702_1);
wc64!(WC64_19, 0.868_344_7);
wc64!(WC64_20, 0.934_583_6);
wc64!(WC64_21, 1.014_408_3);
wc64!(WC64_22, 1.112_071_6);
wc64!(WC64_23, 1.233_832_7);
wc64!(WC64_24, 1.389_294);
wc64!(WC64_25, 1.593_972_3);
wc64!(WC64_26, 1.874_676);
wc64!(WC64_27, 2.282_050);
wc64!(WC64_28, 2.924_628_4);
wc64!(WC64_29, 4.084_611);
wc64!(WC64_30, 6.796_751);
wc64!(WC64_31, 20.373_878);

const INV_WC64_0: f32 = 1.0 / WC64_0;
const INV_WC64_1: f32 = 1.0 / WC64_1;
const INV_WC64_2: f32 = 1.0 / WC64_2;
const INV_WC64_3: f32 = 1.0 / WC64_3;
const INV_WC64_4: f32 = 1.0 / WC64_4;
const INV_WC64_5: f32 = 1.0 / WC64_5;
const INV_WC64_6: f32 = 1.0 / WC64_6;
const INV_WC64_7: f32 = 1.0 / WC64_7;
const INV_WC64_8: f32 = 1.0 / WC64_8;
const INV_WC64_9: f32 = 1.0 / WC64_9;
const INV_WC64_10: f32 = 1.0 / WC64_10;
const INV_WC64_11: f32 = 1.0 / WC64_11;
const INV_WC64_12: f32 = 1.0 / WC64_12;
const INV_WC64_13: f32 = 1.0 / WC64_13;
const INV_WC64_14: f32 = 1.0 / WC64_14;
const INV_WC64_15: f32 = 1.0 / WC64_15;
const INV_WC64_16: f32 = 1.0 / WC64_16;
const INV_WC64_17: f32 = 1.0 / WC64_17;
const INV_WC64_18: f32 = 1.0 / WC64_18;
const INV_WC64_19: f32 = 1.0 / WC64_19;
const INV_WC64_20: f32 = 1.0 / WC64_20;
const INV_WC64_21: f32 = 1.0 / WC64_21;
const INV_WC64_22: f32 = 1.0 / WC64_22;
const INV_WC64_23: f32 = 1.0 / WC64_23;
const INV_WC64_24: f32 = 1.0 / WC64_24;
const INV_WC64_25: f32 = 1.0 / WC64_25;
const INV_WC64_26: f32 = 1.0 / WC64_26;
const INV_WC64_27: f32 = 1.0 / WC64_27;
const INV_WC64_28: f32 = 1.0 / WC64_28;
const INV_WC64_29: f32 = 1.0 / WC64_29;
const INV_WC64_30: f32 = 1.0 / WC64_30;
const INV_WC64_31: f32 = 1.0 / WC64_31;

// =============================================================================
// 64-pt forward DCT helper
// =============================================================================

/// Forward 1D 64-point DCT, in-place at offset `base`. No scaling.
#[cube]
fn fwd_dct1d_64(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;

    let mut first = SharedMemory::<f32>::new(32usize);
    let mut second = SharedMemory::<f32>::new(32usize);
    let mut i: u32 = 0u32;
    while i < 32u32 {
        let iu = i as usize;
        let a = mem[b + iu];
        let bv = mem[b + 63usize - iu];
        first[iu] = a + bv;
        second[iu] = a - bv;
        i += 1u32;
    }

    fwd_dct1d_32(&mut first, 0u32);

    second[0usize] = second[0usize] * WC64_0;
    second[1usize] = second[1usize] * WC64_1;
    second[2usize] = second[2usize] * WC64_2;
    second[3usize] = second[3usize] * WC64_3;
    second[4usize] = second[4usize] * WC64_4;
    second[5usize] = second[5usize] * WC64_5;
    second[6usize] = second[6usize] * WC64_6;
    second[7usize] = second[7usize] * WC64_7;
    second[8usize] = second[8usize] * WC64_8;
    second[9usize] = second[9usize] * WC64_9;
    second[10usize] = second[10usize] * WC64_10;
    second[11usize] = second[11usize] * WC64_11;
    second[12usize] = second[12usize] * WC64_12;
    second[13usize] = second[13usize] * WC64_13;
    second[14usize] = second[14usize] * WC64_14;
    second[15usize] = second[15usize] * WC64_15;
    second[16usize] = second[16usize] * WC64_16;
    second[17usize] = second[17usize] * WC64_17;
    second[18usize] = second[18usize] * WC64_18;
    second[19usize] = second[19usize] * WC64_19;
    second[20usize] = second[20usize] * WC64_20;
    second[21usize] = second[21usize] * WC64_21;
    second[22usize] = second[22usize] * WC64_22;
    second[23usize] = second[23usize] * WC64_23;
    second[24usize] = second[24usize] * WC64_24;
    second[25usize] = second[25usize] * WC64_25;
    second[26usize] = second[26usize] * WC64_26;
    second[27usize] = second[27usize] * WC64_27;
    second[28usize] = second[28usize] * WC64_28;
    second[29usize] = second[29usize] * WC64_29;
    second[30usize] = second[30usize] * WC64_30;
    second[31usize] = second[31usize] * WC64_31;

    fwd_dct1d_32(&mut second, 0u32);

    // B transform on second half: s[0] = SQRT2*s[0] + s[1]; for i in 1..31: s[i] += s[i+1]
    second[0usize] = SQRT2 * second[0usize] + second[1usize];
    let mut i: u32 = 1u32;
    while i < 31u32 {
        let iu = i as usize;
        second[iu] = second[iu] + second[iu + 1usize];
        i += 1u32;
    }

    // InverseEvenOdd interleave
    let mut i: u32 = 0u32;
    while i < 32u32 {
        let iu = i as usize;
        mem[b + 2usize * iu] = first[iu];
        mem[b + 2usize * iu + 1usize] = second[iu];
        i += 1u32;
    }
}

/// Inverse 1D 64-point IDCT core (no scaling).
#[cube]
fn inv_idct1d_64_core(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;

    let mut first = SharedMemory::<f32>::new(32usize);
    let mut second = SharedMemory::<f32>::new(32usize);
    let mut i: u32 = 0u32;
    while i < 32u32 {
        let iu = i as usize;
        first[iu] = mem[b + 2usize * iu];
        second[iu] = mem[b + 2usize * iu + 1usize];
        i += 1u32;
    }

    // Reverse B transform on second half: for i in (1..31).rev(): s[i] -= s[i+1]
    let mut i: u32 = 30u32;
    loop {
        let iu = i as usize;
        second[iu] = second[iu] - second[iu + 1usize];
        if i == 1u32 {
            break;
        }
        i -= 1u32;
    }
    second[0usize] = (second[0usize] - second[1usize]) * ONE_OVER_SQRT2;

    inv_idct1d_32_core(&mut second, 0u32);

    second[0usize] = second[0usize] * INV_WC64_0;
    second[1usize] = second[1usize] * INV_WC64_1;
    second[2usize] = second[2usize] * INV_WC64_2;
    second[3usize] = second[3usize] * INV_WC64_3;
    second[4usize] = second[4usize] * INV_WC64_4;
    second[5usize] = second[5usize] * INV_WC64_5;
    second[6usize] = second[6usize] * INV_WC64_6;
    second[7usize] = second[7usize] * INV_WC64_7;
    second[8usize] = second[8usize] * INV_WC64_8;
    second[9usize] = second[9usize] * INV_WC64_9;
    second[10usize] = second[10usize] * INV_WC64_10;
    second[11usize] = second[11usize] * INV_WC64_11;
    second[12usize] = second[12usize] * INV_WC64_12;
    second[13usize] = second[13usize] * INV_WC64_13;
    second[14usize] = second[14usize] * INV_WC64_14;
    second[15usize] = second[15usize] * INV_WC64_15;
    second[16usize] = second[16usize] * INV_WC64_16;
    second[17usize] = second[17usize] * INV_WC64_17;
    second[18usize] = second[18usize] * INV_WC64_18;
    second[19usize] = second[19usize] * INV_WC64_19;
    second[20usize] = second[20usize] * INV_WC64_20;
    second[21usize] = second[21usize] * INV_WC64_21;
    second[22usize] = second[22usize] * INV_WC64_22;
    second[23usize] = second[23usize] * INV_WC64_23;
    second[24usize] = second[24usize] * INV_WC64_24;
    second[25usize] = second[25usize] * INV_WC64_25;
    second[26usize] = second[26usize] * INV_WC64_26;
    second[27usize] = second[27usize] * INV_WC64_27;
    second[28usize] = second[28usize] * INV_WC64_28;
    second[29usize] = second[29usize] * INV_WC64_29;
    second[30usize] = second[30usize] * INV_WC64_30;
    second[31usize] = second[31usize] * INV_WC64_31;

    inv_idct1d_32_core(&mut first, 0u32);

    let mut i: u32 = 0u32;
    while i < 32u32 {
        let iu = i as usize;
        let f = first[iu];
        let s = second[iu];
        mem[b + iu] = (f + s) * HALF;
        mem[b + 63usize - iu] = (f - s) * HALF;
        i += 1u32;
    }
}

#[cube]
fn inv_idct1d_64(mem: &mut SharedMemory<f32>, base: u32) {
    let b = base as usize;
    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        mem[b + iu] = mem[b + iu] * 64.0f32;
        i += 1u32;
    }
    inv_idct1d_64_core(mem, base);
}

// =============================================================================
// 64x64 forward + inverse
// =============================================================================

#[cube(launch_unchecked)]
pub fn dct_64x64_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 4096usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 4096usize;

    let mut scratch = SharedMemory::<f32>::new(4096usize);
    let mut transposed = SharedMemory::<f32>::new(4096usize);

    let mut r: u32 = 0u32;
    while r < 64u32 {
        let row_off = r * 64u32;
        let row_off_us = row_off as usize;
        let mut c: u32 = 0u32;
        while c < 64u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = input[off + row_off_us + cu];
            c += 1u32;
        }
        fwd_dct1d_64(&mut scratch, row_off);
        let mut c: u32 = 0u32;
        while c < 64u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = scratch[row_off_us + cu] * ONE_OVER_64;
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 64u32 {
        let mut c: u32 = 0u32;
        while c < 64u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 64usize + ru] = scratch[ru * 64usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 64u32 {
        let row_off = r * 64u32;
        let row_off_us = row_off as usize;
        fwd_dct1d_64(&mut transposed, row_off);
        let mut c: u32 = 0u32;
        while c < 64u32 {
            let cu = c as usize;
            transposed[row_off_us + cu] = transposed[row_off_us + cu] * ONE_OVER_64;
            c += 1u32;
        }
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 4096u32 {
        let iu = i as usize;
        output[off + iu] = transposed[iu];
        i += 1u32;
    }
}

#[cube(launch_unchecked)]
pub fn idct_64x64_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 4096usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 4096usize;

    let mut scratch = SharedMemory::<f32>::new(4096usize);
    let mut transposed = SharedMemory::<f32>::new(4096usize);

    let mut i: u32 = 0u32;
    while i < 4096u32 {
        let iu = i as usize;
        scratch[iu] = input[off + iu];
        i += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 64u32 {
        inv_idct1d_64(&mut scratch, r * 64u32);
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 64u32 {
        let mut c: u32 = 0u32;
        while c < 64u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 64usize + ru] = scratch[ru * 64usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 64u32 {
        inv_idct1d_64(&mut transposed, r * 64u32);
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 4096u32 {
        let iu = i as usize;
        output[off + iu] = transposed[iu];
        i += 1u32;
    }
}

// =============================================================================
// 64x32 forward + inverse
// =============================================================================

/// Forward 64x32 DCT (64 rows × 32 cols). Per-row 32-pt DCT (1/32),
/// transpose, per-row 64-pt DCT (1/64). No final transpose.
#[cube(launch_unchecked)]
pub fn dct_64x32_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 2048usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 2048usize;

    let mut scratch = SharedMemory::<f32>::new(2048usize);
    let mut transposed = SharedMemory::<f32>::new(2048usize);

    let mut r: u32 = 0u32;
    while r < 64u32 {
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

    // Transpose 64x32 → 32x64
    let mut r: u32 = 0u32;
    while r < 64u32 {
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 64usize + ru] = scratch[ru * 32usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 32u32 {
        let row_off = r * 64u32;
        let row_off_us = row_off as usize;
        fwd_dct1d_64(&mut transposed, row_off);
        let mut c: u32 = 0u32;
        while c < 64u32 {
            let cu = c as usize;
            transposed[row_off_us + cu] = transposed[row_off_us + cu] * ONE_OVER_64;
            c += 1u32;
        }
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 2048u32 {
        let iu = i as usize;
        output[off + iu] = transposed[iu];
        i += 1u32;
    }
}

#[cube(launch_unchecked)]
pub fn idct_64x32_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 2048usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 2048usize;

    let mut scratch = SharedMemory::<f32>::new(2048usize);

    // IDCT-64 on 32 rows
    let mut r: u32 = 0u32;
    while r < 32u32 {
        let row_off = r * 64u32;
        let row_off_us = row_off as usize;
        let mut c: u32 = 0u32;
        while c < 64u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = input[off + row_off_us + cu];
            c += 1u32;
        }
        inv_idct1d_64(&mut scratch, row_off);
        r += 1u32;
    }

    // Transpose 32x64 → 64x32
    let mut transposed = SharedMemory::<f32>::new(2048usize);
    let mut r: u32 = 0u32;
    while r < 32u32 {
        let mut c: u32 = 0u32;
        while c < 64u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 32usize + ru] = scratch[ru * 64usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // IDCT-32 on 64 rows
    let mut r: u32 = 0u32;
    while r < 64u32 {
        let row_off = r * 32u32;
        let mut row_buf = SharedMemory::<f32>::new(32usize);
        let row_off_us = row_off as usize;
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let cu = c as usize;
            row_buf[cu] = transposed[row_off_us + cu] * 32.0f32;
            c += 1u32;
        }
        inv_idct1d_32_core(&mut row_buf, 0u32);
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let cu = c as usize;
            output[off + row_off_us + cu] = row_buf[cu];
            c += 1u32;
        }
        r += 1u32;
    }
}

// =============================================================================
// 32x64 forward + inverse
// =============================================================================

/// Forward 32x64 DCT (32 rows × 64 cols). Per-row 64-pt DCT (1/64),
/// transpose, per-row 32-pt DCT (1/32), FINAL transpose (ROWS < COLS).
#[cube(launch_unchecked)]
pub fn dct_32x64_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 2048usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 2048usize;

    let mut scratch = SharedMemory::<f32>::new(2048usize);
    let mut transposed = SharedMemory::<f32>::new(2048usize);

    let mut r: u32 = 0u32;
    while r < 32u32 {
        let row_off = r * 64u32;
        let row_off_us = row_off as usize;
        let mut c: u32 = 0u32;
        while c < 64u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = input[off + row_off_us + cu];
            c += 1u32;
        }
        fwd_dct1d_64(&mut scratch, row_off);
        let mut c: u32 = 0u32;
        while c < 64u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = scratch[row_off_us + cu] * ONE_OVER_64;
            c += 1u32;
        }
        r += 1u32;
    }

    // Transpose 32x64 → 64x32
    let mut r: u32 = 0u32;
    while r < 32u32 {
        let mut c: u32 = 0u32;
        while c < 64u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 32usize + ru] = scratch[ru * 64usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 64u32 {
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

    // FINAL transpose 64x32 → 32x64
    let mut r: u32 = 0u32;
    while r < 64u32 {
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let ru = r as usize;
            let cu = c as usize;
            output[off + cu * 64usize + ru] = transposed[ru * 32usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }
}

#[cube(launch_unchecked)]
pub fn idct_32x64_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 2048usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 2048usize;

    let mut transposed = SharedMemory::<f32>::new(2048usize);
    let mut tmp = SharedMemory::<f32>::new(2048usize);

    // Un-transpose 32x64 input → 64x32
    let mut r: u32 = 0u32;
    while r < 32u32 {
        let mut c: u32 = 0u32;
        while c < 64u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 32usize + ru] = input[off + ru * 64usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // IDCT-32 on each of 64 rows of transposed
    let mut r: u32 = 0u32;
    while r < 64u32 {
        let row_off = r * 32u32;
        let row_off_us = row_off as usize;
        let mut row_buf = SharedMemory::<f32>::new(32usize);
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let cu = c as usize;
            row_buf[cu] = transposed[row_off_us + cu] * 32.0f32;
            c += 1u32;
        }
        inv_idct1d_32_core(&mut row_buf, 0u32);
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let cu = c as usize;
            tmp[row_off_us + cu] = row_buf[cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // Transpose 64x32 → 32x64
    let mut transposed2 = SharedMemory::<f32>::new(2048usize);
    let mut r: u32 = 0u32;
    while r < 64u32 {
        let mut c: u32 = 0u32;
        while c < 32u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed2[cu * 64usize + ru] = tmp[ru * 32usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // IDCT-64 on each of 32 rows
    let mut r: u32 = 0u32;
    while r < 32u32 {
        inv_idct1d_64(&mut transposed2, r * 64u32);
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 2048u32 {
        let iu = i as usize;
        output[off + iu] = transposed2[iu];
        i += 1u32;
    }
}
