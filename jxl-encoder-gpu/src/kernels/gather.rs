// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Spatial-plane → per-block buffer gather kernel.
//!
//! Closes the API gap between [`crate::persistent::GpuPlane`]
//! (spatially laid out) and [`crate::persistent::GpuBlocks`]
//! (contiguous per-block layout) — without this kernel, callers have
//! to download the plane to host, gather into per-block buffers
//! host-side, then re-upload (expensive PCIe round-trip mid-pipeline).
//!
//! Layout:
//! - **Input**: spatial plane `[width × height]` (row-major).
//! - **Output**: `num_blocks * tile_pixels` floats, where
//!   `tile_pixels = tile_w * tile_h`. Block `b` occupies the slice
//!   `[b * tile_pixels, (b+1) * tile_pixels)` in row-major order
//!   within the tile.
//!
//! Block coordinates are implicit raster grid: block `b` lives at
//! `(bx, by) = (b % blocks_per_row, b / blocks_per_row)` and spans
//! pixels `[bx * tile_w, (bx+1) * tile_w) × [by * tile_h, (by+1) * tile_h)`
//! in the source plane.
//!
//! Generic over tile dimensions; one kernel handles all DCT strategy
//! tile sizes (8×8 DCT8 through 64×64 DCT64).
//!
//! Thread strategy: one thread per output pixel.
//! `total_threads = num_blocks * tile_pixels`.

use cubecl::prelude::*;

/// Gather raster-grid blocks from a spatial plane into a contiguous
/// per-block buffer.
///
/// One thread per output pixel. Cast scalars to usize at function
/// entry so all indexing arithmetic flows through usize (mirrors
/// the mask1x1.rs pattern).
#[cube(launch_unchecked)]
pub fn gather_blocks_kernel(
    plane: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    blocks_per_row: u32,
    tile_w: u32,
    tile_h: u32,
    total_threads: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = total_threads as usize;
    if idx >= total {
        terminate!();
    }
    let w_u = width as usize;
    let bpr = blocks_per_row as usize;
    let tw = tile_w as usize;
    let th = tile_h as usize;
    let tile_pixels = tw * th;
    let block_idx = idx / tile_pixels;
    let pix_in_block = idx - block_idx * tile_pixels;
    let dy = pix_in_block / tw;
    let dx = pix_in_block - dy * tw;
    let by = block_idx / bpr;
    let bx = block_idx - by * bpr;

    let src_y = by * th + dy;
    let src_x = bx * tw + dx;
    let src_off = src_y * w_u + src_x;
    output[idx] = plane[src_off];
}

/// Inverse of [`fn@gather_blocks_kernel`]: scatter per-block buffer back
/// into a spatial plane.
#[cube(launch_unchecked)]
pub fn scatter_blocks_kernel(
    blocks: &Array<f32>,
    plane: &mut Array<f32>,
    width: u32,
    blocks_per_row: u32,
    tile_w: u32,
    tile_h: u32,
    total_threads: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = total_threads as usize;
    if idx >= total {
        terminate!();
    }
    let w_u = width as usize;
    let bpr = blocks_per_row as usize;
    let tw = tile_w as usize;
    let th = tile_h as usize;
    let tile_pixels = tw * th;
    let block_idx = idx / tile_pixels;
    let pix_in_block = idx - block_idx * tile_pixels;
    let dy = pix_in_block / tw;
    let dx = pix_in_block - dy * tw;
    let by = block_idx / bpr;
    let bx = block_idx - by * bpr;

    let dst_y = by * th + dy;
    let dst_x = bx * tw + dx;
    let dst_off = dst_y * w_u + dst_x;
    plane[dst_off] = blocks[idx];
}
