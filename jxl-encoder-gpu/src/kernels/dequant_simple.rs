// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Generic per-coefficient dequant: `output[i] = quant[i] * weights[i]`.
//!
//! Mirrors the inverse of `quantize_large` for any `block_size` (DCT8
//! → 64, DCT16x16 → 256, DCT32x32 → 1024, etc.). One thread per
//! block. Useful as a building block for cost grids and for the
//! larger-strategy decoder path that doesn't need DCT8's CfL +
//! adjust_quant_bias adjustments.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn dequant_simple_kernel(
    quant: &Array<i32>,
    weights: &Array<f32>,
    output: &mut Array<f32>,
    block_size: u32,
) {
    let block_idx = ABSOLUTE_POS;
    let bs = block_size as usize;
    let n_blocks = quant.len() / bs;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * bs;
    let mut i: u32 = 0u32;
    while (i as usize) < bs {
        let iu = i as usize;
        output[off + iu] = (quant[off + iu] as f32) * weights[off + iu];
        i += 1u32;
    }
}

/// Broadcast-weights variant of [`dequant_simple_kernel`]. `weights`
/// is exactly `block_size` f32 (one quant matrix), broadcast across
/// all blocks.
///
/// Saves `(num_blocks - 1) * block_size * 4` bytes of GPU memory +
/// upload traffic when callers were previously replicating the same
/// matrix per-block — the common case for the AFV cost grid (which
/// runs the same DCT8 weight table over all candidate blocks). Same
/// algorithmic semantics as the per-block variant when called with
/// replicated weights.
#[cube(launch_unchecked)]
pub fn dequant_simple_kernel_broadcast_w(
    quant: &Array<i32>,
    weights: &Array<f32>,
    output: &mut Array<f32>,
    block_size: u32,
) {
    let block_idx = ABSOLUTE_POS;
    let bs = block_size as usize;
    let n_blocks = quant.len() / bs;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * bs;
    let mut i: u32 = 0u32;
    while (i as usize) < bs {
        let iu = i as usize;
        // Broadcast: weights[iu] not weights[off + iu].
        output[off + iu] = (quant[off + iu] as f32) * weights[iu];
        i += 1u32;
    }
}

/// Strategy-aware dequant: `output[i] = quant[i] * weight[i] / qac[block]`.
/// Broadcast weights, per-block qac. No bias correction.
///
/// Use for non-DCT8 strategies (DCT16x8, DCT8x16, DCT16x16, etc.) where
/// the upstream dequant is purely `q * weight / qac` without
/// `adjust_quant_bias` (which only applies to DCT8 in libjxl).
///
/// LLF positions are NOT zeroed here — caller is expected to overwrite
/// them with restored LLF values via `dispatch_restore_llf` (or its
/// GPU variant). This matches the existing host-side dequant loop in
/// `forks/reconstruct.rs::encode_and_reconstruct_mixed_strategy_single_channel`.
#[cube(launch_unchecked)]
pub fn dequant_strategy_kernel_broadcast_w(
    quant: &Array<i32>,
    weights: &Array<f32>, // block_size f32
    qac: &Array<f32>,     // num_blocks f32
    output: &mut Array<f32>,
    block_size: u32,
) {
    let block_idx = ABSOLUTE_POS;
    let bs = block_size as usize;
    let n_blocks = qac.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * bs;
    let inv_qac = 1.0f32 / qac[block_idx];
    let mut i: u32 = 0u32;
    while (i as usize) < bs {
        let iu = i as usize;
        output[off + iu] = (quant[off + iu] as f32) * weights[iu] * inv_qac;
        i += 1u32;
    }
}

/// DCT8 variant of [`dequant_strategy_kernel_broadcast_w`] with
/// `adjust_quant_bias` applied per channel. `channel_bias` is one of
/// `BIAS_X / BIAS_Y / BIAS_B` (caller supplies the constant for the
/// channel being processed).
///
/// `output[i] = adjust_quant_bias(quant[i], channel_bias) * weight[i] / qac[block]`
///
/// Constants kept in sync with `kernels::dequant`:
///   X: 0.945_349_93, Y: 0.929_945_5, B: 0.950_064_9.
///   |q| < 1.125 → sign(q) * channel_bias; else q - 0.145 / q.
#[cube(launch_unchecked)]
pub fn dequant_strategy_kernel_broadcast_w_dct8(
    quant: &Array<i32>,
    weights: &Array<f32>,
    qac: &Array<f32>,
    output: &mut Array<f32>,
    block_size: u32,
    channel_bias: f32,
) {
    const BIAS_RECIP: f32 = 0.145;
    let block_idx = ABSOLUTE_POS;
    let bs = block_size as usize;
    let n_blocks = qac.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * bs;
    let inv_qac = 1.0f32 / qac[block_idx];
    let mut i: u32 = 0u32;
    while (i as usize) < bs {
        let iu = i as usize;
        let q_int = quant[off + iu];
        let biased = if q_int == 0i32 {
            f32::new(0.0)
        } else {
            let q = q_int as f32;
            if f32::abs(q) < 1.125f32 {
                if q > 0.0f32 { channel_bias } else { -channel_bias }
            } else {
                q - BIAS_RECIP / q
            }
        };
        output[off + iu] = biased * weights[iu] * inv_qac;
        i += 1u32;
    }
}
