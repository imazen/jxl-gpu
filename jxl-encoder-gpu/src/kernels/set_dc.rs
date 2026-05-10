// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Indexed DC-from-grid set: writes one DC value per block into
//! position 0 of each output coefficient block.
//!
//! Equivalent to the host loop in
//! `forks::reconstruct::dispatch_restore_llf` for the 1×1 LLF case
//! (DCT8 / DCT4x4 / DCT4x8 / DCT8x4 / IDENTITY / DCT2x2 / AFV0-3 — every
//! strategy whose `llf_dim_x = llf_dim_y = 1`).
//!
//! Layout:
//! - **dc_grid**: per-(8×8) block scalar plane
//!   (`xsize_blocks_8 × ysize_blocks_8` floats, raster order).
//! - **coords**: `n_blocks * 2` u32 values, packed as
//!   `[bx_0, by_0, bx_1, by_1, …]` — same shape as
//!   [`crate::kernels::indexed_gather`].
//! - **dst**: caller-supplied per-block coefficient buffer
//!   (`n_blocks * coeffs_per_block` floats). Only position 0 of each
//!   block is mutated; AC coefficients untouched.
//!
//! Thread strategy: one thread per block in the strategy's batch.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn set_dc_from_grid_indexed_kernel(
    dc_grid: &Array<f32>,
    coords: &Array<u32>,
    dst: &mut Array<f32>,
    dc_stride: u32,
    coeffs_per_block: u32,
    n_blocks: u32,
) {
    let i = ABSOLUTE_POS;
    let n = n_blocks as usize;
    if i >= n {
        terminate!();
    }
    let bx = coords[i * 2usize] as usize;
    let by = coords[i * 2usize + 1usize] as usize;
    let stride = dc_stride as usize;
    let cpb = coeffs_per_block as usize;
    dst[i * cpb] = dc_grid[by * stride + bx];
}
