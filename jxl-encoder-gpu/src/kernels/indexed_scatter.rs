// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Indexed per-block buffer → spatial-plane scatter: inverse of
//! [`crate::kernels::indexed_gather`]. Writes a sparse subset of
//! `(bx, by)` 8×8-block-grid positions back into a destination plane.
//!
//! Layout:
//! - **blocks**: `n_blocks * (tile_w * tile_h)` floats, row-major
//!   within each tile (matches indexed_gather's output / IDCT output
//!   layout).
//! - **coords**: `n_blocks * 2` u32 values, packed as
//!   `[bx_0, by_0, bx_1, by_1, …]` — same shape as
//!   [`crate::kernels::indexed_gather`].
//! - **plane**: `[width × height]` (row-major). Mutated in place at
//!   each `(bx, by)`'s tile rectangle. Pixels outside any covered
//!   tile are untouched.
//!
//! Block `b` writes pixels
//! `[bx*8, bx*8 + tile_w) × [by*8, by*8 + tile_h)` of the destination.
//!
//! Thread strategy: one thread per block-pixel.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn indexed_scatter_kernel(
    blocks: &Array<f32>,
    coords: &Array<u32>,
    plane: &mut Array<f32>,
    width: u32,
    tile_w: u32,
    tile_h: u32,
    total_threads: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = total_threads as usize;
    if idx >= total {
        terminate!();
    }
    let tw = tile_w as usize;
    let th = tile_h as usize;
    let tile_pixels = tw * th;
    let block_idx = idx / tile_pixels;
    let pix_in_block = idx - block_idx * tile_pixels;
    let dy = pix_in_block / tw;
    let dx = pix_in_block - dy * tw;

    let bx = coords[block_idx * 2usize] as usize;
    let by = coords[block_idx * 2usize + 1usize] as usize;

    let dst_y = by * 8usize + dy;
    let dst_x = bx * 8usize + dx;
    let w = width as usize;
    plane[dst_y * w + dst_x] = blocks[idx];
}
