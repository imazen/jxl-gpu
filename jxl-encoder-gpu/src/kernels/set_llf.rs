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

/// Bit-for-bit port of `forks::reconstruct::dct1d_4`. Returns the four
/// DCT-1d-4 outputs in the order `(u0, b0, u1, w1)`.
///
/// Cubecl gotcha: returns a tuple — works in cubecl 0.10 because tuple
/// returns from `#[cube]` functions are supported.
#[cube]
fn dct1d_4(a: f32, b: f32, c: f32, d: f32) -> (f32, f32, f32, f32) {
    let t0 = a + d;
    let t1 = b + c;
    let t2 = a - d;
    let t3 = b - c;
    let u0 = t0 + t1;
    let u1 = t0 - t1;
    // WC4[0] = 0.5411961, WC4[1] = 1.3065630
    let v0 = t2 * 0.5411961f32;
    let v1 = t3 * 1.3065630f32;
    let w0 = v0 + v1;
    let w1 = v0 - v1;
    // SQRT2 = 1.4142135
    let b0 = 1.4142135f32 * w0 + w1;
    (u0, b0, u1, w1)
}

/// LLF restore for DCT32×16 (4×2 DC-grid → 8 LLF positions per block
/// at coeffs[iy * 32 + ix] for iy in 0..2, ix in 0..4).
///
/// DC layout: 4 rows × 2 cols, indexed `dc_grid[(by + iy) * stride +
/// (bx + ix)]` for iy in 0..4, ix in 0..2.
///
/// Math (mirrors `forks::reconstruct::restore_llf_dct32x16`):
/// 1. dct1d_2 on each of 4 rows (in-place `[a, b] -> [a+b, a-b]`).
/// 2. Transpose 4×2 → 2×4.
/// 3. dct1d_4 on each of 2 rows.
/// 4. Per-position scale by `1 / (8 * SCALE_16_TO_2[iy] * SCALE_32_TO_4[ix])`.
/// 5. Write to `dst[i * cpb + iy * 32 + ix]` for iy in 0..2, ix in 0..4.
///
/// `coeffs_per_block` must be 512 (DCT32x16 standard size).
#[cube(launch_unchecked)]
pub fn set_llf_dct32x16_indexed_kernel(
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

    // Read 4×2 DC values: dc[iy*2 + ix] for iy in 0..4, ix in 0..2.
    let r0 = by * stride + bx;
    let r1 = (by + 1usize) * stride + bx;
    let r2 = (by + 2usize) * stride + bx;
    let r3 = (by + 3usize) * stride + bx;
    let m00 = dc_grid[r0];
    let m01 = dc_grid[r0 + 1usize];
    let m10 = dc_grid[r1];
    let m11 = dc_grid[r1 + 1usize];
    let m20 = dc_grid[r2];
    let m21 = dc_grid[r2 + 1usize];
    let m30 = dc_grid[r3];
    let m31 = dc_grid[r3 + 1usize];

    // dct1d_2 on each row: [a, b] -> [a+b, a-b].
    let p00 = m00 + m01;
    let p01 = m00 - m01;
    let p10 = m10 + m11;
    let p11 = m10 - m11;
    let p20 = m20 + m21;
    let p21 = m20 - m21;
    let p30 = m30 + m31;
    let p31 = m30 - m31;

    // Transpose 4×2 → 2×4: t[ix*4+iy] = p[iy*2+ix].
    // Row 0 of transposed = (p00, p10, p20, p30); row 1 = (p01, p11, p21, p31).
    // dct1d_4 on each row of transposed.
    let (b00, b01, b02, b03) = dct1d_4(p00, p10, p20, p30);
    let (b10, b11, b12, b13) = dct1d_4(p01, p11, p21, p31);

    // Per-position scale by 1 / (8 * s16[iy] * s32[ix]).
    let s16_0 = 1.0f32;
    let s16_1 = 0.9017642f32;
    let s32_0 = 1.0f32;
    let s32_1 = 0.9748868f32;
    let s32_2 = 0.9017642f32;
    let s32_3 = 0.7870549f32;

    let off = i * cpb;
    // Row 0 (iy=0, s16_0 = 1.0)
    dst[off] = b00 * (1.0f32 / (8.0f32 * s16_0 * s32_0));
    dst[off + 1usize] = b01 * (1.0f32 / (8.0f32 * s16_0 * s32_1));
    dst[off + 2usize] = b02 * (1.0f32 / (8.0f32 * s16_0 * s32_2));
    dst[off + 3usize] = b03 * (1.0f32 / (8.0f32 * s16_0 * s32_3));
    // Row 1 (iy=1, stride 32)
    dst[off + 32usize] = b10 * (1.0f32 / (8.0f32 * s16_1 * s32_0));
    dst[off + 33usize] = b11 * (1.0f32 / (8.0f32 * s16_1 * s32_1));
    dst[off + 34usize] = b12 * (1.0f32 / (8.0f32 * s16_1 * s32_2));
    dst[off + 35usize] = b13 * (1.0f32 / (8.0f32 * s16_1 * s32_3));
}

/// LLF restore for DCT16×32 (2×4 DC-grid → 8 LLF positions per block
/// at coeffs[iy * 32 + ix] for iy in 0..2, ix in 0..4).
///
/// DC layout: 2 rows × 4 cols, indexed `dc_grid[(by + iy) * stride +
/// (bx + ix)]` for iy in 0..2, ix in 0..4.
///
/// Math (mirrors `forks::reconstruct::restore_llf_dct16x32`):
/// 1. dct1d_4 on each of 2 rows.
/// 2. Transpose 2×4 → 4×2.
/// 3. dct1d_2 on each of 4 rows (in-place).
/// 4. Transpose back 4×2 → 2×4.
/// 5. Per-position scale by `1 / (8 * SCALE_16_TO_2[iy] * SCALE_32_TO_4[ix])`.
/// 6. Write to `dst[i * cpb + iy * 32 + ix]` for iy in 0..2, ix in 0..4.
///
/// `coeffs_per_block` must be 512 (DCT16x32 standard size).
#[cube(launch_unchecked)]
pub fn set_llf_dct16x32_indexed_kernel(
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

    // Read 2×4 DC values: dc[iy*4 + ix] for iy in 0..2, ix in 0..4.
    let r0 = by * stride + bx;
    let r1 = (by + 1usize) * stride + bx;
    let m00 = dc_grid[r0];
    let m01 = dc_grid[r0 + 1usize];
    let m02 = dc_grid[r0 + 2usize];
    let m03 = dc_grid[r0 + 3usize];
    let m10 = dc_grid[r1];
    let m11 = dc_grid[r1 + 1usize];
    let m12 = dc_grid[r1 + 2usize];
    let m13 = dc_grid[r1 + 3usize];

    // dct1d_4 on each of 2 rows.
    let (a00, a01, a02, a03) = dct1d_4(m00, m01, m02, m03);
    let (a10, a11, a12, a13) = dct1d_4(m10, m11, m12, m13);

    // Transpose 2×4 → 4×2: t[ix*2 + iy] = a[iy*4 + ix].
    // After transpose, rows are: (a00, a10), (a01, a11), (a02, a12), (a03, a13).
    // dct1d_2 on each row: [a, b] -> [a+b, a-b].
    let q00 = a00 + a10;
    let q01 = a00 - a10;
    let q10 = a01 + a11;
    let q11 = a01 - a11;
    let q20 = a02 + a12;
    let q21 = a02 - a12;
    let q30 = a03 + a13;
    let q31 = a03 - a13;
    // After dct1d_2, transposed-and-dct'd buffer is:
    //   t[0] = q00, t[1] = q01, t[2] = q10, t[3] = q11,
    //   t[4] = q20, t[5] = q21, t[6] = q30, t[7] = q31
    // (laid out as ix-rows of 2-elements: row ix = [q_{ix,0}, q_{ix,1}]).

    // Transpose back 4×2 → 2×4: result[iy*4 + ix] = t[ix*2 + iy].
    // result[0] = t[0] = q00, result[1] = t[2] = q10, result[2] = t[4] = q20, result[3] = t[6] = q30
    // result[4] = t[1] = q01, result[5] = t[3] = q11, result[6] = t[5] = q21, result[7] = t[7] = q31
    let r00 = q00;
    let r01 = q10;
    let r02 = q20;
    let r03 = q30;
    let r10 = q01;
    let r11 = q11;
    let r12 = q21;
    let r13 = q31;

    // Per-position scale by 1 / (8 * s16[iy] * s32[ix]).
    let s16_0 = 1.0f32;
    let s16_1 = 0.9017642f32;
    let s32_0 = 1.0f32;
    let s32_1 = 0.9748868f32;
    let s32_2 = 0.9017642f32;
    let s32_3 = 0.7870549f32;

    let off = i * cpb;
    dst[off] = r00 * (1.0f32 / (8.0f32 * s16_0 * s32_0));
    dst[off + 1usize] = r01 * (1.0f32 / (8.0f32 * s16_0 * s32_1));
    dst[off + 2usize] = r02 * (1.0f32 / (8.0f32 * s16_0 * s32_2));
    dst[off + 3usize] = r03 * (1.0f32 / (8.0f32 * s16_0 * s32_3));
    dst[off + 32usize] = r10 * (1.0f32 / (8.0f32 * s16_1 * s32_0));
    dst[off + 33usize] = r11 * (1.0f32 / (8.0f32 * s16_1 * s32_1));
    dst[off + 34usize] = r12 * (1.0f32 / (8.0f32 * s16_1 * s32_2));
    dst[off + 35usize] = r13 * (1.0f32 / (8.0f32 * s16_1 * s32_3));
}

/// LLF restore for DCT32×32 (4×4 DC-grid → 16 LLF positions per block
/// at coeffs[iy * 32 + ix] for iy, ix in 0..4).
///
/// Math (mirrors `forks::reconstruct::restore_llf_dct32x32`):
/// 1. Read 16 DC values: `dc[iy*4+ix] = dc_grid[(by+iy)*stride + (bx+ix)]`.
/// 2. dct1d_4 on each of 4 rows.
/// 3. Transpose 4×4.
/// 4. dct1d_4 on each of 4 rows again.
/// 5. Per-position scale by `1 / (16 * SCALE_32_TO_4[iy] * SCALE_32_TO_4[ix])`.
/// 6. Write to `dst[i * cpb + iy * 32 + ix]` for iy, ix in 0..4.
///
/// `coeffs_per_block` must be 1024 (DCT32x32 standard size). AC
/// coefficients (the 1008 positions outside the 4×4 LLF region within
/// the 32×32 layout) are untouched.
///
/// Per-thread work: 8 dct1d_4 calls + 16 scale-multiply-and-store. One
/// thread per block — typical 1024² encode hits this kernel with at
/// most ~1024 DCT32x32 blocks per channel.
#[cube(launch_unchecked)]
pub fn set_llf_dct32x32_indexed_kernel(
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

    // Step 1: read 4×4 DC values.
    let r0 = by * stride + bx;
    let r1 = (by + 1usize) * stride + bx;
    let r2 = (by + 2usize) * stride + bx;
    let r3 = (by + 3usize) * stride + bx;
    let m00 = dc_grid[r0];
    let m01 = dc_grid[r0 + 1usize];
    let m02 = dc_grid[r0 + 2usize];
    let m03 = dc_grid[r0 + 3usize];
    let m10 = dc_grid[r1];
    let m11 = dc_grid[r1 + 1usize];
    let m12 = dc_grid[r1 + 2usize];
    let m13 = dc_grid[r1 + 3usize];
    let m20 = dc_grid[r2];
    let m21 = dc_grid[r2 + 1usize];
    let m22 = dc_grid[r2 + 2usize];
    let m23 = dc_grid[r2 + 3usize];
    let m30 = dc_grid[r3];
    let m31 = dc_grid[r3 + 1usize];
    let m32 = dc_grid[r3 + 2usize];
    let m33 = dc_grid[r3 + 3usize];

    // Step 2: dct1d_4 on each row.
    let (a00, a01, a02, a03) = dct1d_4(m00, m01, m02, m03);
    let (a10, a11, a12, a13) = dct1d_4(m10, m11, m12, m13);
    let (a20, a21, a22, a23) = dct1d_4(m20, m21, m22, m23);
    let (a30, a31, a32, a33) = dct1d_4(m30, m31, m32, m33);

    // Step 3+4: transpose then dct1d_4 on rows of transposed.
    // Transposed rows are: col0 = (a00, a10, a20, a30), col1 = (a01, a11, a21, a31), etc.
    let (b00, b01, b02, b03) = dct1d_4(a00, a10, a20, a30);
    let (b10, b11, b12, b13) = dct1d_4(a01, a11, a21, a31);
    let (b20, b21, b22, b23) = dct1d_4(a02, a12, a22, a32);
    let (b30, b31, b32, b33) = dct1d_4(a03, a13, a23, a33);
    // After transpose+dct1d_4, b[iy*4+ix] is at position (iy, ix) of the
    // transposed-and-dct'd matrix. The host code's transpose writes
    // transposed[ix * 4 + iy] = block[iy * 4 + ix], then dct1d_4 on
    // transposed[0..4], etc — so b[iy*4+ix] above corresponds to
    // host transposed[iy*4 + ix] after the second dct1d_4.

    // Step 5: per-position scale by 1 / (16 * s[iy] * s[ix]).
    // SCALE_32_TO_4 = [1.0, 0.9748868, 0.9017642, 0.7870549]
    // Pre-multiplied by 1/16:
    //   inv16_s[i] = 1 / (16 * s[i])
    //   inv16_s[0] = 1/16            = 0.0625
    //   inv16_s[1] = 1/(16*0.9748868) ≈ 0.064108
    //   inv16_s[2] = 1/(16*0.9017642) ≈ 0.069306
    //   inv16_s[3] = 1/(16*0.7870549) ≈ 0.079402
    // Final factor f[iy][ix] = inv16_s[iy] * 16 * inv16_s[ix] / 16 — no,
    // simpler: f[iy][ix] = (1/16) / (s[iy] * s[ix]) = (1/(16 * s[iy])) / s[ix].
    // Inline each 1/(s[i]) via literal divisions; cubecl folds these.
    let s0 = 1.0f32;
    let s1 = 0.9748868f32;
    let s2 = 0.9017642f32;
    let s3 = 0.7870549f32;

    let off = i * cpb;
    // Row 0
    dst[off] = b00 * (1.0f32 / (16.0f32 * s0 * s0));
    dst[off + 1usize] = b01 * (1.0f32 / (16.0f32 * s0 * s1));
    dst[off + 2usize] = b02 * (1.0f32 / (16.0f32 * s0 * s2));
    dst[off + 3usize] = b03 * (1.0f32 / (16.0f32 * s0 * s3));
    // Row 1 (stride 32)
    dst[off + 32usize] = b10 * (1.0f32 / (16.0f32 * s1 * s0));
    dst[off + 33usize] = b11 * (1.0f32 / (16.0f32 * s1 * s1));
    dst[off + 34usize] = b12 * (1.0f32 / (16.0f32 * s1 * s2));
    dst[off + 35usize] = b13 * (1.0f32 / (16.0f32 * s1 * s3));
    // Row 2 (stride 32 × 2)
    dst[off + 64usize] = b20 * (1.0f32 / (16.0f32 * s2 * s0));
    dst[off + 65usize] = b21 * (1.0f32 / (16.0f32 * s2 * s1));
    dst[off + 66usize] = b22 * (1.0f32 / (16.0f32 * s2 * s2));
    dst[off + 67usize] = b23 * (1.0f32 / (16.0f32 * s2 * s3));
    // Row 3 (stride 32 × 3)
    dst[off + 96usize] = b30 * (1.0f32 / (16.0f32 * s3 * s0));
    dst[off + 97usize] = b31 * (1.0f32 / (16.0f32 * s3 * s1));
    dst[off + 98usize] = b32 * (1.0f32 / (16.0f32 * s3 * s2));
    dst[off + 99usize] = b33 * (1.0f32 / (16.0f32 * s3 * s3));
}
