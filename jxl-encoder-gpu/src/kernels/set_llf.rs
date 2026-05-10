// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-strategy LLF restore GPU kernels — write the inverse-Hadamard
//! LLF coefficients into per-block coefficient buffers from the per-
//! (8×8) block DC grid.
//!
//! Each kernel handles ONE LLF rectangle shape; dispatch on the host
//! based on `raw_strategy`. All kernels follow the same shape:
//!   - one thread per block in the strategy's batch
//!   - read the strategy's DC subgrid from `dc_grid` using `coords[i]`
//!     and the strategy's `dc_step_x` / `dc_step_y` (in DC-grid
//!     elements)
//!   - compute the inverse Hadamard via the strategy's restore math
//!   - write LLF positions into `dst[i * coeffs_per_block + ...]`
//!
//! For 1×1 LLF strategies (DCT8 / DCT4-family / IDENTITY / DCT2X2 /
//! AFV) use [`crate::kernels::set_dc::set_dc_from_grid_indexed_kernel`]
//! instead — it's a strict simplification.

use cubecl::prelude::*;

/// LLF restore for DCT16×8 / DCT8×16 (1×2 or 2×1 DC-grid, two LLF
/// positions per block at coeffs[0] and coeffs[1]).
///
/// Math (mirrors `forks::reconstruct::restore_llf_dct16x8_or_8x16`):
/// ```text
///   dc0 = dc_grid[by * stride + bx]
///   dc1 = dc_grid[by * stride + bx + dc_step]   // dc_step = stride for
///                                                // DCT16x8 (vertical pair),
///                                                // 1 for DCT8x16 (horizontal pair)
///   llf0 = (dc0 + dc1) / (2 * s0)              // s0 = 1.0
///   llf1 = (dc0 - dc1) / (2 * s1)              // s1 ≈ 0.9017642
///   dst[i * coeffs_per_block + 0] = llf0
///   dst[i * coeffs_per_block + 1] = llf1
/// ```
///
/// `dc_step` differentiates DCT16x8 (vertical pair, dc_step = stride)
/// from DCT8x16 (horizontal pair, dc_step = 1). All other parameters
/// are the same; one kernel covers both strategies.
#[cube(launch_unchecked)]
pub fn set_llf_dct16x8_or_8x16_indexed_kernel(
    dc_grid: &Array<f32>,
    coords: &Array<u32>,
    dst: &mut Array<f32>,
    dc_stride: u32,
    dc_step: u32,
    coeffs_per_block: u32,
    n_blocks: u32,
) {
    // s0 = DCT_RESAMPLE_SCALE_16_TO_2[0] = 1.0,
    // s1 = DCT_RESAMPLE_SCALE_16_TO_2[1] ≈ 0.9017642.
    // The Hadamard normalization is folded in via (1 / (2 * s_k)):
    //   1 / (2 * s0) = 0.5
    //   1 / (2 * s1) ≈ 0.554489
    // Inlined into the write expressions because cubecl 0.10 rejects
    // typed f32 let-bindings (NativeExpand<f32> → ConstantValue).
    let i = ABSOLUTE_POS;
    let n = n_blocks as usize;
    if i >= n {
        terminate!();
    }
    let bx = coords[i * 2usize] as usize;
    let by = coords[i * 2usize + 1usize] as usize;
    let stride = dc_stride as usize;
    let step = dc_step as usize;
    let cpb = coeffs_per_block as usize;
    let dc0_idx = by * stride + bx;
    let dc1_idx = dc0_idx + step;
    let dc0 = dc_grid[dc0_idx];
    let dc1 = dc_grid[dc1_idx];
    dst[i * cpb] = (dc0 + dc1) * 0.5f32;
    dst[i * cpb + 1usize] = (dc0 - dc1) * (0.5f32 / 0.9017642f32);
}
