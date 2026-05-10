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

/// LLF restore for DCT16×16 (2×2 DC-grid → 4 LLF positions per block
/// at coeffs[0], [1], [16], [17]).
///
/// Math (mirrors `forks::reconstruct::restore_llf_dct16x16`):
/// ```text
///   dc00 = dc_grid[(by + 0) * stride + (bx + 0)]
///   dc01 = dc_grid[(by + 0) * stride + (bx + 1)]
///   dc10 = dc_grid[(by + 1) * stride + (bx + 0)]
///   dc11 = dc_grid[(by + 1) * stride + (bx + 1)]
///   h00 = dc00 + dc01 + dc10 + dc11    // 2D Hadamard
///   h01 = dc00 + dc01 - dc10 - dc11
///   h10 = dc00 - dc01 + dc10 - dc11
///   h11 = dc00 - dc01 - dc10 + dc11
///   llf00 = h00 / (4 * s0 * s0)        // s0 = 1, so 0.25
///   llf01 = h01 / (4 * s0 * s1)        // ≈ 0.27725
///   llf10 = h10 / (4 * s1 * s0)        // ≈ 0.27725
///   llf11 = h11 / (4 * s1 * s1)        // ≈ 0.30746
///   dst[i * cpb + 0]  = llf00
///   dst[i * cpb + 1]  = llf01
///   dst[i * cpb + 16] = llf10
///   dst[i * cpb + 17] = llf11
/// ```
///
/// `coeffs_per_block` is typically 256 for DCT16x16. The "16" stride
/// inside the coeff block (for positions [16] and [17]) is hardcoded —
/// DCT16x16 always has a 16-wide coefficient layout.
#[cube(launch_unchecked)]
pub fn set_llf_dct16x16_indexed_kernel(
    dc_grid: &Array<f32>,
    coords: &Array<u32>,
    dst: &mut Array<f32>,
    dc_stride: u32,
    coeffs_per_block: u32,
    n_blocks: u32,
) {
    // Inverse normalization factors. s0 = 1, s1 ≈ 0.9017642.
    //   1 / (4 * 1 * 1)             = 0.25
    //   1 / (4 * 1 * 0.9017642)     ≈ 0.27725
    //   1 / (4 * 0.9017642 * 1)     ≈ 0.27725 (same as above)
    //   1 / (4 * 0.9017642^2)       ≈ 0.30746
    // Inlined into the writes (cubecl 0.10 NativeExpand<f32> gotcha,
    // see set_llf_dct16x8_or_8x16_indexed_kernel).
    let i = ABSOLUTE_POS;
    let n = n_blocks as usize;
    if i >= n {
        terminate!();
    }
    let bx = coords[i * 2usize] as usize;
    let by = coords[i * 2usize + 1usize] as usize;
    let stride = dc_stride as usize;
    let cpb = coeffs_per_block as usize;

    let row0 = by * stride + bx;
    let row1 = (by + 1usize) * stride + bx;
    let dc00 = dc_grid[row0];
    let dc01 = dc_grid[row0 + 1usize];
    let dc10 = dc_grid[row1];
    let dc11 = dc_grid[row1 + 1usize];

    let h00 = dc00 + dc01 + dc10 + dc11;
    let h01 = dc00 + dc01 - dc10 - dc11;
    let h10 = dc00 - dc01 + dc10 - dc11;
    let h11 = dc00 - dc01 - dc10 + dc11;

    let off = i * cpb;
    dst[off] = h00 * 0.25f32;
    dst[off + 1usize] = h01 * (1.0f32 / (4.0f32 * 0.9017642f32));
    dst[off + 16usize] = h10 * (1.0f32 / (4.0f32 * 0.9017642f32));
    dst[off + 17usize] = h11 * (1.0f32 / (4.0f32 * 0.9017642f32 * 0.9017642f32));
}
