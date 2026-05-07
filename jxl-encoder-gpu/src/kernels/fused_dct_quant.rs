// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Fused DCT8 + quantize kernel.
//!
//! Same pipeline as separate `dct_8x8_wide_kernel` + `quantize_dct8_kernel`,
//! but never materializes the intermediate `coeffs[num_blocks * 64]` f32
//! buffer in global memory. Coefficients live in shared memory only;
//! quantization writes i32 output directly.
//!
//! Saves 1 global memory write + 1 global memory read (256 KB total at
//! 1024² for one channel) compared to the separate-stage chain. On
//! memory-bound GPUs this is a meaningful fraction of total runtime.
//!
//! Reuses `dct1d_8` from `crate::kernels::dct8` to keep the DCT
//! butterfly bit-identical with the standalone DCT kernels.

use cubecl::prelude::*;

const ONE_OVER_8: f32 = 0.125;
const WIDE_CUBE_DIM: u32 = 64;

// Re-implement dct1d_8 here as `pub(crate)` from dct8.rs is currently
// crate-internal only — duplicating the constants is bit-exact and
// avoids changing the dct8 module visibility.
const SQRT2: f32 = core::f32::consts::SQRT_2;
const WC_M4_0: f32 = 0.541_196_1;
const WC_M4_1: f32 = 1.306_563;
const WC_M8_0: f32 = 0.509_795_6;
const WC_M8_1: f32 = 0.601_344_9;
const WC_M8_2: f32 = 0.899_976_2;
const WC_M8_3: f32 = 2.562_915_5;

#[cube]
fn dct1d_8_local(mem: &mut SharedMemory<f32>, base: u32) {
    let b0 = base as usize;
    let m0 = mem[b0];
    let m1 = mem[b0 + 1usize];
    let m2 = mem[b0 + 2usize];
    let m3 = mem[b0 + 3usize];
    let m4 = mem[b0 + 4usize];
    let m5 = mem[b0 + 5usize];
    let m6 = mem[b0 + 6usize];
    let m7 = mem[b0 + 7usize];

    let mut t0 = m0 + m7;
    let mut t1 = m1 + m6;
    let mut t2 = m2 + m5;
    let mut t3 = m3 + m4;
    let mut t4 = m0 - m7;
    let mut t5 = m1 - m6;
    let mut t6 = m2 - m5;
    let mut t7 = m3 - m4;

    let a0 = t0 + t3;
    let a1 = t1 + t2;
    let a2 = t0 - t3;
    let a3 = t1 - t2;
    let b0v = a0 + a1;
    let b1v = a0 - a1;
    let a2s = a2 * WC_M4_0;
    let a3s = a3 * WC_M4_1;
    let c0 = a2s + a3s;
    let c1 = a2s - a3s;
    let c0_post = SQRT2 * c0 + c1;
    t0 = b0v;
    t2 = b1v;
    t1 = c0_post;
    t3 = c1;

    t4 = t4 * WC_M8_0;
    t5 = t5 * WC_M8_1;
    t6 = t6 * WC_M8_2;
    t7 = t7 * WC_M8_3;

    let a0 = t4 + t7;
    let a1 = t5 + t6;
    let a2 = t4 - t7;
    let a3 = t5 - t6;
    let b0v = a0 + a1;
    let b1v = a0 - a1;
    let a2s = a2 * WC_M4_0;
    let a3s = a3 * WC_M4_1;
    let c0 = a2s + a3s;
    let c1 = a2s - a3s;
    let c0_post = SQRT2 * c0 + c1;
    t4 = b0v;
    t6 = b1v;
    t5 = c0_post;
    t7 = c1;

    t4 = SQRT2 * t4 + t5;
    t5 = t5 + t6;
    t6 = t6 + t7;

    mem[b0] = t0;
    mem[b0 + 1usize] = t4;
    mem[b0 + 2usize] = t1;
    mem[b0 + 3usize] = t5;
    mem[b0 + 4usize] = t2;
    mem[b0 + 5usize] = t6;
    mem[b0 + 6usize] = t3;
    mem[b0 + 7usize] = t7;
}

#[cube]
fn round_ties_even_to_i32(x: f32) -> i32 {
    let trunc_i = x as i32;
    let trunc_f = trunc_i as f32;
    let frac = x - trunc_f;
    let abs_frac = f32::abs(frac);
    let trunc_is_odd = (trunc_i & 1i32) != 0i32;
    let bump_unconditional = abs_frac > 0.5f32;
    let bump_tie = (abs_frac == 0.5f32) && trunc_is_odd;
    let do_bump = bump_unconditional || bump_tie;
    if do_bump {
        if frac > 0.0f32 {
            trunc_i + 1i32
        } else {
            trunc_i - 1i32
        }
    } else {
        trunc_i
    }
}

/// Fused DCT8 + quantize-DCT8. cube_dim=64, one block per thread.
///
/// Inputs:
/// - `pixels`: `num_blocks * 64` f32 — pixel-domain blocks
/// - `weights`: `num_blocks * 64` f32 — per-coefficient inverse quant
///   matrix entries
/// - `qac_qm`: `num_blocks` f32 — per-block scale
/// - `thresholds`: 4 f32 — per-quadrant dead-zone thresholds
///
/// Output:
/// - `output`: `num_blocks * 64` i32 — quantized AC coefficients (DC=0)
#[cube(launch_unchecked)]
pub fn dct8_quantize_fused_wide_kernel(
    pixels: &Array<f32>,
    weights: &Array<f32>,
    qac_qm: &Array<f32>,
    thresholds: &Array<f32>,
    output: &mut Array<i32>,
) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = qac_qm.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;
    let unit = UNIT_POS;
    let private_base = unit * 64u32;
    let private_base_us = private_base as usize;

    let mut scratch = SharedMemory::<f32>::new((WIDE_CUBE_DIM * 64u32) as usize);
    let mut transposed = SharedMemory::<f32>::new((WIDE_CUBE_DIM * 64u32) as usize);

    // ── Forward DCT8 (mirrors dct_8x8_wide_kernel) ─────────────────
    let mut r: u32 = 0u32;
    while r < 8u32 {
        let row_off = r * 8u32;
        let row_off_us = row_off as usize;
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let cu = c as usize;
            scratch[private_base_us + row_off_us + cu] = pixels[off + row_off_us + cu];
            c += 1u32;
        }
        dct1d_8_local(&mut scratch, private_base + row_off);
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let cu = c as usize;
            scratch[private_base_us + row_off_us + cu] =
                scratch[private_base_us + row_off_us + cu] * ONE_OVER_8;
            c += 1u32;
        }
        r += 1u32;
    }
    let mut r: u32 = 0u32;
    while r < 8u32 {
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[private_base_us + cu * 8usize + ru] =
                scratch[private_base_us + ru * 8usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }
    let mut r: u32 = 0u32;
    while r < 8u32 {
        let row_off = r * 8u32;
        let row_off_us = row_off as usize;
        dct1d_8_local(&mut transposed, private_base + row_off);
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let cu = c as usize;
            transposed[private_base_us + row_off_us + cu] =
                transposed[private_base_us + row_off_us + cu] * ONE_OVER_8;
            c += 1u32;
        }
        r += 1u32;
    }

    // ── Quantize directly from shared memory ───────────────────────
    let qac = qac_qm[block_idx];
    let t0 = thresholds[0usize];
    let t1 = thresholds[1usize];
    let t2 = thresholds[2usize];
    let t3 = thresholds[3usize];

    output[off] = 0i32; // DC

    let mut idx: u32 = 1u32;
    while idx < 64u32 {
        let iu = idx as usize;
        let y = idx / 8u32;
        let x = idx - y * 8u32;
        let row_hi = y >= 4u32;
        let col_hi = x >= 4u32;
        let thr = if row_hi {
            if col_hi { t3 } else { t2 }
        } else if col_hi {
            t1
        } else {
            t0
        };
        let coef = transposed[private_base_us + iu];
        let val = coef * (1.0f32 / weights[off + iu]) * qac;
        let absv = f32::abs(val);
        output[off + iu] = if absv < thr {
            i32::new(0)
        } else {
            round_ties_even_to_i32(val)
        };
        idx += 1u32;
    }
}
