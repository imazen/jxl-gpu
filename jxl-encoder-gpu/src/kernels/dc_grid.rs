// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-(8×8) block DC grid: one f32 per 8×8 block = mean of the 64
//! pixels covered.
//!
//! Mirrors `crate::forks::reconstruct::compute_dc_grid_per_8x8_block`
//! (host scalar). Plane dimensions must be multiples of 8.
//!
//! Output layout: `n_blocks` floats, raster order over the 8×8-block grid
//! (`block_y * (width / 8) + block_x`).
//!
//! Thread strategy: one thread per output block. Each thread sums its
//! 64 covered pixels and divides by 64. n_blocks is typically large
//! (16384 at 1024², 65536 at 2048²) so dispatch is comfortable.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn dc_grid_8x8_kernel(
    plane: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    blocks_per_row: u32,
    n_blocks: u32,
) {
    let idx = ABSOLUTE_POS;
    let n = n_blocks as usize;
    if idx >= n {
        terminate!();
    }
    let bpr = blocks_per_row as usize;
    let w = width as usize;
    let by = idx / bpr;
    let bx = idx - by * bpr;
    let y0 = by * 8;
    let x0 = bx * 8;
    let mut sum = 0f32;
    let mut dy: u32 = 0u32;
    while dy < 8u32 {
        let row_off = (y0 + dy as usize) * w + x0;
        let mut dx: u32 = 0u32;
        while dx < 8u32 {
            sum += plane[row_off + dx as usize];
            dx += 1u32;
        }
        dy += 1u32;
    }
    output[idx] = sum * (1.0f32 / 64.0f32);
}
