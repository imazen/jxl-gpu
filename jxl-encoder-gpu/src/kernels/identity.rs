// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! IDENTITY 8x8 forward + inverse transform.
//!
//! Mirrors `jxl_encoder::vardct::dct::special::{identity_transform,
//! inverse_identity_transform}` exactly. The 8x8 block is processed
//! as 4 4x4 sub-blocks (2x2 grid). For each sub-block:
//!   - block_dc = mean of 16 pixels (×1/16)
//!   - ref_pixel = pixel(1, 1) within sub-block
//!   - AC coefficients: pixel - ref_pixel, stored at interleaved
//!     positions (y + iy*2, x + ix*2)
//!   - Corner pixel saved at (y+2)*8 + (x+2)
//!   - DC stored at (y, x)
//!
//! Final step: 2x2 Hadamard merge of the 4 DC values at positions
//! [0], [1], [8], [9] (×0.25).
//!
//! One thread per 8x8 block.

// Match the rest of the kernel set — the cube macro generates code that
// triggers `assign_op_pattern` on the `x = x + y` style we use for
// parity with the CPU `_scalar` reference.
#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

const ONE_OVER_16: f32 = 1.0 / 16.0;
const QUARTER: f32 = 0.25;

/// Forward IDENTITY transform on a contiguous batch of 8x8 blocks.
/// Input/output: `num_blocks * 64` floats, row-major within each block.
#[cube(launch_unchecked)]
pub fn identity_forward_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64u32 as usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;

    // Load all 64 pixels into private scratch.
    let mut p = SharedMemory::<f32>::new(64usize);
    let mut c = SharedMemory::<f32>::new(64usize);
    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        p[iu] = input[off + iu];
        c[iu] = 0.0f32;
        i += 1u32;
    }

    // Process the 2x2 grid of 4x4 sub-blocks.
    for y in 0..2u32 {
        for x in 0..2u32 {
            let yu = y as usize;
            let xu = x as usize;

            // Sum 16 pixels for block_dc.
            let mut sum = 0.0f32;
            for iy in 0..4u32 {
                for ix in 0..4u32 {
                    let iyu = iy as usize;
                    let ixu = ix as usize;
                    sum = sum + p[(yu * 4 + iyu) * 8 + xu * 4 + ixu];
                }
            }
            let block_dc = sum * ONE_OVER_16;
            let ref_pixel = p[(yu * 4 + 1usize) * 8 + xu * 4 + 1usize];

            // Store residual coefficients at interleaved positions.
            for iy in 0..4u32 {
                for ix in 0..4u32 {
                    let iyu = iy as usize;
                    let ixu = ix as usize;
                    if !(iyu == 1usize && ixu == 1usize) {
                        c[(yu + iyu * 2) * 8 + xu + ixu * 2] =
                            p[(yu * 4 + iyu) * 8 + xu * 4 + ixu] - ref_pixel;
                    }
                }
            }

            // Save the existing value at (y, x) into the corner slot
            // before we overwrite (y, x) with the DC.
            c[(yu + 2usize) * 8 + xu + 2usize] = c[yu * 8 + xu];
            c[yu * 8 + xu] = block_dc;
        }
    }

    // 2x2 Hadamard merge of the 4 DC values at positions [0], [1], [8], [9].
    let block00 = c[0usize];
    let block01 = c[1usize];
    let block10 = c[8usize];
    let block11 = c[9usize];
    c[0usize] = (block00 + block01 + block10 + block11) * QUARTER;
    c[1usize] = (block00 + block01 - block10 - block11) * QUARTER;
    c[8usize] = (block00 - block01 + block10 - block11) * QUARTER;
    c[9usize] = (block00 - block01 - block10 + block11) * QUARTER;

    // Write back.
    let mut j: u32 = 0u32;
    while j < 64u32 {
        let ju = j as usize;
        output[off + ju] = c[ju];
        j += 1u32;
    }
}

/// Inverse IDENTITY transform. Per-sub-block:
///   - inverse Hadamard on DC positions [0], [1], [8], [9] (no scaling)
///   - residual_sum = sum of all coefficients in the sub-block except
///     the (0, 0) (DC) position
///   - ref_pixel = block_dc - residual_sum / 16
///   - pixel(iy, ix) = coef + ref_pixel (for non-(1,1) positions)
///   - corner pixel (y*4, x*4) = coef[(y+2)*8 + x+2] + ref_pixel
#[cube(launch_unchecked)]
pub fn identity_inverse_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64u32 as usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;

    let mut c = SharedMemory::<f32>::new(64usize);
    let mut p = SharedMemory::<f32>::new(64usize);
    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        c[iu] = input[off + iu];
        p[iu] = 0.0f32;
        i += 1u32;
    }

    // Inverse Hadamard on DC positions (no x0.25).
    let block00 = c[0usize];
    let block01 = c[1usize];
    let block10 = c[8usize];
    let block11 = c[9usize];
    let dc00 = block00 + block01 + block10 + block11;
    let dc01 = block00 + block01 - block10 - block11;
    let dc10 = block00 - block01 + block10 - block11;
    let dc11 = block00 - block01 - block10 + block11;

    for y in 0..2u32 {
        for x in 0..2u32 {
            let yu = y as usize;
            let xu = x as usize;

            // Pick this sub-block's reconstructed DC.
            let block_dc = if yu == 0usize && xu == 0usize {
                dc00
            } else if yu == 0usize && xu == 1usize {
                dc01
            } else if yu == 1usize && xu == 0usize {
                dc10
            } else {
                dc11
            };

            // Sum all residual coefficients (skip [0][0] = DC).
            let mut residual_sum = 0.0f32;
            for iy in 0..4u32 {
                for ix in 0..4u32 {
                    let iyu = iy as usize;
                    let ixu = ix as usize;
                    if !(iyu == 0usize && ixu == 0usize) {
                        residual_sum = residual_sum + c[(yu + iyu * 2) * 8 + xu + ixu * 2];
                    }
                }
            }
            let ref_pixel = block_dc - residual_sum * ONE_OVER_16;
            p[(yu * 4 + 1usize) * 8 + xu * 4 + 1usize] = ref_pixel;

            // Reconstruct AC pixels: coef + ref_pixel.
            for iy in 0..4u32 {
                for ix in 0..4u32 {
                    let iyu = iy as usize;
                    let ixu = ix as usize;
                    if !(iyu == 1usize && ixu == 1usize) {
                        p[(yu * 4 + iyu) * 8 + xu * 4 + ixu] =
                            c[(yu + iyu * 2) * 8 + xu + ixu * 2] + ref_pixel;
                    }
                }
            }
            // Corner pixel reconstruction from saved slot.
            p[yu * 4 * 8 + xu * 4] = c[(yu + 2usize) * 8 + xu + 2usize] + ref_pixel;
        }
    }

    let mut j: u32 = 0u32;
    while j < 64u32 {
        let ju = j as usize;
        output[off + ju] = p[ju];
        j += 1u32;
    }
}
