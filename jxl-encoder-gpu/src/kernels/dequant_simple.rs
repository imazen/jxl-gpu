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
