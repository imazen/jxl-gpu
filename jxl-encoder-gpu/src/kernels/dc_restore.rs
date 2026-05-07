// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Per-block DC restoration kernel.
//!
//! Copies DC values (slot 0 of each block) from one per-block buffer
//! to another, leaving all AC coefficients untouched. Used to bridge
//! the GPU `quantize_dct8` kernel's unconditional DC-zeroing with the
//! reconstructed dequant output: encoder-side dc_coding handles DC
//! quantization separately, but for a roundtrip demo we just carry
//! the original DC through bit-exact.
//!
//! One thread per block.

use cubecl::prelude::*;

/// For each of `num_blocks` blocks, copy `src[b * coeffs_per_block]`
/// into `dst[b * coeffs_per_block]`. AC coefficients untouched.
#[cube(launch_unchecked)]
pub fn restore_dc_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    coeffs_per_block: u32,
    num_blocks: u32,
) {
    let block_idx = ABSOLUTE_POS;
    let nb = num_blocks as usize;
    if block_idx >= nb {
        terminate!();
    }
    let cpb = coeffs_per_block as usize;
    let off = block_idx * cpb;
    dst[off] = src[off];
}
