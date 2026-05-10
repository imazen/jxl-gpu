// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Indexed plane-to-blocks gather: pulls a sparse subset of `(bx, by)`
//! 8×8-block-grid positions (each spanning `tile_w × tile_h` pixels)
//! from a spatial plane into contiguous per-block buffers.
//!
//! Differs from [`crate::kernels::gather`] (which gathers the FULL
//! raster grid of tiles) — this one walks an arbitrary list of block
//! coordinates, so caller can batch only the blocks that a particular
//! AC strategy was assigned to.
//!
//! Layout:
//! - **Input**: spatial plane `[width × height]` (row-major).
//! - **Coords**: `n_blocks * 2` `u32` values, packed as `[bx_0, by_0,
//!   bx_1, by_1, ...]`. Each `(bx, by)` is in 8×8-block-grid units —
//!   the kernel multiplies by 8 to compute the pixel-space top-left
//!   of the strategy's tile.
//! - **Output**: `n_blocks * (tile_w * tile_h)` floats, row-major
//!   within each tile.
//!
//! Block `b` covers pixels
//! `[bx*8, bx*8 + tile_w) × [by*8, by*8 + tile_h)` in the source plane.
//!
//! Thread strategy: one thread per output pixel.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn indexed_gather_kernel(
    plane: &Array<f32>,
    coords: &Array<u32>,
    output: &mut Array<f32>,
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

    // coords[block_idx * 2 + 0] = bx, coords[block_idx * 2 + 1] = by.
    let bx = coords[block_idx * 2usize] as usize;
    let by = coords[block_idx * 2usize + 1usize] as usize;

    let src_y = by * 8usize + dy;
    let src_x = bx * 8usize + dx;
    let w = width as usize;
    output[idx] = plane[src_y * w + src_x];
}
