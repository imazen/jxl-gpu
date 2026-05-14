// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Tile-major scatter + scale of DCT8 blocks for CfL fitting.
//!
//! Mirrors the inner loop of `compute_cfl_map` (jxl_encoder
//! chroma_from_luma.rs lines 322-355): zero DC, multiply by inv_qm,
//! and store into per-tile buffers (m, s) consumed by
//! `find_best_multiplier_newton_kernel`.
//!
//! Layout: one cube per (block_in_image), 64 threads/cube each
//! handling one coefficient.
//!
//! - `dct_y`, `dct_x`, `dct_b`: per-block DCT8 outputs (block-major,
//!   `block_idx * 64 + coef_idx`). All have `num_blocks` blocks of 64
//!   coefficients each. Block index = `by * xsize_blocks + bx`.
//! - `inv_qm_x`, `inv_qm_b`: 64 floats each, broadcast across blocks.
//! - Outputs `m_yx`, `s_x`, `m_yb`, `s_b`: tile-major scaled values.
//!   Layout: `tile_idx * (TILE_BLOCKS * 64) + block_in_tile_idx * 64 +
//!   coef_idx`. Padding slots (block_in_tile_idx beyond actual tile
//!   coverage) are written as 0.0.
//!
//! `xsize_blocks`/`ysize_blocks` are the CPU-aligned block grid (may
//! be < gpu-padded). Blocks past these are skipped (their slots in
//! the output are written as 0.0 by separate clear logic — caller
//! must zero buffers first).

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

const TILE_DIM_IN_BLOCKS: u32 = 8;
const COEFFS_PER_BLOCK: u32 = 64;
const VALUES_PER_TILE: u32 = TILE_DIM_IN_BLOCKS * TILE_DIM_IN_BLOCKS * COEFFS_PER_BLOCK; // 4096

/// One cube per tile, 64 threads per cube — each thread covers one
/// block slot within the tile (8x8 grid). This guarantees every slot
/// in the per-tile buffers is written (active blocks → DCT data,
/// padding slots → zero), so no separate zero-init upload is needed.
///
/// Layout: launch with `CubeCount(num_tiles)`, `CubeDim(64)`.
/// `CUBE_POS` = tile_idx, `UNIT_POS` = block_in_tile_idx (0..63).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cfl_collect_kernel(
    dct_y: &Array<f32>,
    dct_x: &Array<f32>,
    dct_b: &Array<f32>,
    inv_qm_x: &Array<f32>,
    inv_qm_b: &Array<f32>,
    m_yx: &mut Array<f32>,
    s_x: &mut Array<f32>,
    m_yb: &mut Array<f32>,
    s_b: &mut Array<f32>,
    xsize_blocks: u32,
    ysize_blocks: u32,
    gpu_xsize_blocks: u32,
    xsize_tiles: u32,
) {
    let tile_idx_u = CUBE_POS as u32;
    let block_in_tile_idx_u = UNIT_POS as u32;
    if block_in_tile_idx_u >= TILE_DIM_IN_BLOCKS * TILE_DIM_IN_BLOCKS {
        terminate!();
    }
    let tx = tile_idx_u % xsize_tiles;
    let ty = tile_idx_u / xsize_tiles;
    let bx_in_tile = block_in_tile_idx_u % TILE_DIM_IN_BLOCKS;
    let by_in_tile = block_in_tile_idx_u / TILE_DIM_IN_BLOCKS;
    let bx = tx * TILE_DIM_IN_BLOCKS + bx_in_tile;
    let by = ty * TILE_DIM_IN_BLOCKS + by_in_tile;
    let tile_base = tile_idx_u * VALUES_PER_TILE
        + block_in_tile_idx_u * COEFFS_PER_BLOCK;
    let active = bx < xsize_blocks && by < ysize_blocks;
    if active {
        let block_idx_u = by * gpu_xsize_blocks + bx;
        let in_off = (block_idx_u * COEFFS_PER_BLOCK) as usize;
        // DC = 0 (AC-only fitting).
        let dc_off = tile_base as usize;
        m_yx[dc_off] = 0.0f32;
        s_x[dc_off] = 0.0f32;
        m_yb[dc_off] = 0.0f32;
        s_b[dc_off] = 0.0f32;
        let mut i: u32 = 1u32;
        while i < COEFFS_PER_BLOCK {
            let iu = i as usize;
            let dy = dct_y[in_off + iu];
            let dx = dct_x[in_off + iu];
            let db = dct_b[in_off + iu];
            let iqx = inv_qm_x[iu];
            let iqb = inv_qm_b[iu];
            let out_off = (tile_base + i) as usize;
            m_yx[out_off] = dy * iqx;
            s_x[out_off] = dx * iqx;
            m_yb[out_off] = dy * iqb;
            s_b[out_off] = db * iqb;
            i += 1u32;
        }
    } else {
        // Padding slot — write zeros so Newton sums skip these.
        let mut i: u32 = 0u32;
        while i < COEFFS_PER_BLOCK {
            let out_off = (tile_base + i) as usize;
            m_yx[out_off] = 0.0f32;
            s_x[out_off] = 0.0f32;
            m_yb[out_off] = 0.0f32;
            s_b[out_off] = 0.0f32;
            i += 1u32;
        }
    }
}
