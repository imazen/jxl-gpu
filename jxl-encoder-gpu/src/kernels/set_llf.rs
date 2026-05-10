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

/// Bit-for-bit port of `forks::reconstruct::dct1d_8`. Returns the
/// eight DCT-1d-8 outputs in interleaved order matching the host
/// `mem[0..8]` writes after the InverseEvenOdd step.
#[cube]
#[allow(clippy::too_many_arguments)]
fn dct1d_8(
    a0: f32, a1: f32, a2: f32, a3: f32, a4: f32, a5: f32, a6: f32, a7: f32,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    let t0 = a0 + a7;
    let t1 = a1 + a6;
    let t2 = a2 + a5;
    let t3 = a3 + a4;
    let t4 = a0 - a7;
    let t5 = a1 - a6;
    let t6 = a2 - a5;
    let t7 = a3 - a4;
    // dct4(t0..t3) — host returns [v0, SQRT2*s0+s1, v1, s1] from
    // (a, b, c, d) inputs:
    //   u0 = a + d = t0 + t3
    //   u1 = b + c = t1 + t2
    //   u2 = a - d = t0 - t3
    //   u3 = b - c = t1 - t2
    let r0_0 = (t0 + t3) + (t1 + t2);
    let r0_2 = (t0 + t3) - (t1 + t2);
    let r0_w0 = (t0 - t3) * 0.5411961f32;
    let r0_w1 = (t1 - t2) * 1.3065630f32;
    let r0_s1 = r0_w0 - r0_w1;
    let r0_1 = 1.4142135f32 * (r0_w0 + r0_w1) + r0_s1;
    let r0_3 = r0_s1;
    // Wc multiply on second half.
    // WC8 = [0.5097956, 0.6013449, 0.8999762, 2.5629154]
    let w4 = t4 * 0.5097956f32;
    let w5 = t5 * 0.6013449f32;
    let w6 = t6 * 0.8999762f32;
    let w7 = t7 * 2.5629154f32;
    // dct4(w4..w7) — same math, with (a, b, c, d) = (w4, w5, w6, w7):
    //   u0 = w4 + w7,  u1 = w5 + w6,  u2 = w4 - w7,  u3 = w5 - w6
    let r1_0 = (w4 + w7) + (w5 + w6);
    let r1_2 = (w4 + w7) - (w5 + w6);
    let r1_w0 = (w4 - w7) * 0.5411961f32;
    let r1_w1 = (w5 - w6) * 1.3065630f32;
    let r1_s1 = r1_w0 - r1_w1;
    let r1_1 = 1.4142135f32 * (r1_w0 + r1_w1) + r1_s1;
    let r1_3 = r1_s1;
    // B transform.
    let b0 = 1.4142135f32 * r1_0 + r1_1;
    let b1 = r1_1 + r1_2;
    let b2 = r1_2 + r1_3;
    let b3 = r1_3;
    // InverseEvenOdd interleave: mem[0] = r0_0, mem[1] = b0, mem[2] = r0_1,
    // mem[3] = b1, mem[4] = r0_2, mem[5] = b2, mem[6] = r0_3, mem[7] = b3.
    (r0_0, b0, r0_1, b1, r0_2, b2, r0_3, b3)
}

/// LLF restore for DCT64×64 (8×8 DC-grid → 64 LLF positions per block
/// at coeffs[iy * 64 + ix] for iy, ix in 0..8).
///
/// Math (mirrors `forks::reconstruct::restore_llf_dct64x64`):
/// 1. Read 64 DC values from an 8×8 region of dc_grid.
/// 2. dct1d_8 + 1/8 scale on each of 8 rows.
/// 3. Transpose 8×8.
/// 4. dct1d_8 + 1/8 scale on each of 8 rows again.
/// 5. Per-position scale by `1 / (SCALE_64_TO_8[iy] * SCALE_64_TO_8[ix])`.
///    (Note: NO transpose-back for DCT64×64 — square-block convention.)
/// 6. Write to `dst[i * cpb + iy * 64 + ix]` for iy, ix in 0..8.
///
/// `coeffs_per_block` must be 4096 (DCT64x64 standard size). AC
/// coefficients (the 4032 positions outside the 8×8 LLF region within
/// the 64×64 layout) are untouched.
#[cube(launch_unchecked)]
pub fn set_llf_dct64x64_indexed_kernel(
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

    // Step 1: read 8×8 DC values (m[iy * 8 + ix]).
    let r0 = by * stride + bx;
    let r1 = (by + 1usize) * stride + bx;
    let r2 = (by + 2usize) * stride + bx;
    let r3 = (by + 3usize) * stride + bx;
    let r4 = (by + 4usize) * stride + bx;
    let r5 = (by + 5usize) * stride + bx;
    let r6 = (by + 6usize) * stride + bx;
    let r7 = (by + 7usize) * stride + bx;
    let m00 = dc_grid[r0]; let m01 = dc_grid[r0 + 1usize]; let m02 = dc_grid[r0 + 2usize]; let m03 = dc_grid[r0 + 3usize];
    let m04 = dc_grid[r0 + 4usize]; let m05 = dc_grid[r0 + 5usize]; let m06 = dc_grid[r0 + 6usize]; let m07 = dc_grid[r0 + 7usize];
    let m10 = dc_grid[r1]; let m11 = dc_grid[r1 + 1usize]; let m12 = dc_grid[r1 + 2usize]; let m13 = dc_grid[r1 + 3usize];
    let m14 = dc_grid[r1 + 4usize]; let m15 = dc_grid[r1 + 5usize]; let m16 = dc_grid[r1 + 6usize]; let m17 = dc_grid[r1 + 7usize];
    let m20 = dc_grid[r2]; let m21 = dc_grid[r2 + 1usize]; let m22 = dc_grid[r2 + 2usize]; let m23 = dc_grid[r2 + 3usize];
    let m24 = dc_grid[r2 + 4usize]; let m25 = dc_grid[r2 + 5usize]; let m26 = dc_grid[r2 + 6usize]; let m27 = dc_grid[r2 + 7usize];
    let m30 = dc_grid[r3]; let m31 = dc_grid[r3 + 1usize]; let m32 = dc_grid[r3 + 2usize]; let m33 = dc_grid[r3 + 3usize];
    let m34 = dc_grid[r3 + 4usize]; let m35 = dc_grid[r3 + 5usize]; let m36 = dc_grid[r3 + 6usize]; let m37 = dc_grid[r3 + 7usize];
    let m40 = dc_grid[r4]; let m41 = dc_grid[r4 + 1usize]; let m42 = dc_grid[r4 + 2usize]; let m43 = dc_grid[r4 + 3usize];
    let m44 = dc_grid[r4 + 4usize]; let m45 = dc_grid[r4 + 5usize]; let m46 = dc_grid[r4 + 6usize]; let m47 = dc_grid[r4 + 7usize];
    let m50 = dc_grid[r5]; let m51 = dc_grid[r5 + 1usize]; let m52 = dc_grid[r5 + 2usize]; let m53 = dc_grid[r5 + 3usize];
    let m54 = dc_grid[r5 + 4usize]; let m55 = dc_grid[r5 + 5usize]; let m56 = dc_grid[r5 + 6usize]; let m57 = dc_grid[r5 + 7usize];
    let m60 = dc_grid[r6]; let m61 = dc_grid[r6 + 1usize]; let m62 = dc_grid[r6 + 2usize]; let m63 = dc_grid[r6 + 3usize];
    let m64 = dc_grid[r6 + 4usize]; let m65 = dc_grid[r6 + 5usize]; let m66 = dc_grid[r6 + 6usize]; let m67 = dc_grid[r6 + 7usize];
    let m70 = dc_grid[r7]; let m71 = dc_grid[r7 + 1usize]; let m72 = dc_grid[r7 + 2usize]; let m73 = dc_grid[r7 + 3usize];
    let m74 = dc_grid[r7 + 4usize]; let m75 = dc_grid[r7 + 5usize]; let m76 = dc_grid[r7 + 6usize]; let m77 = dc_grid[r7 + 7usize];

    // Step 2: dct1d_8 on each row, then × 1/8.
    let inv8 = 0.125f32;
    let (a00, a01, a02, a03, a04, a05, a06, a07) = dct1d_8(m00, m01, m02, m03, m04, m05, m06, m07);
    let (a10, a11, a12, a13, a14, a15, a16, a17) = dct1d_8(m10, m11, m12, m13, m14, m15, m16, m17);
    let (a20, a21, a22, a23, a24, a25, a26, a27) = dct1d_8(m20, m21, m22, m23, m24, m25, m26, m27);
    let (a30, a31, a32, a33, a34, a35, a36, a37) = dct1d_8(m30, m31, m32, m33, m34, m35, m36, m37);
    let (a40, a41, a42, a43, a44, a45, a46, a47) = dct1d_8(m40, m41, m42, m43, m44, m45, m46, m47);
    let (a50, a51, a52, a53, a54, a55, a56, a57) = dct1d_8(m50, m51, m52, m53, m54, m55, m56, m57);
    let (a60, a61, a62, a63, a64, a65, a66, a67) = dct1d_8(m60, m61, m62, m63, m64, m65, m66, m67);
    let (a70, a71, a72, a73, a74, a75, a76, a77) = dct1d_8(m70, m71, m72, m73, m74, m75, m76, m77);

    // Step 3+4: transpose then dct1d_8 on rows. Transposed row k = column k of a:
    //   col0 = (a00, a10, a20, a30, a40, a50, a60, a70), etc.
    // We pre-multiply each input by inv8 (deferred from step 2) to absorb
    // the first 1/8 scale; then × inv8 again on output.
    let (b00, b01, b02, b03, b04, b05, b06, b07) = dct1d_8(
        a00 * inv8, a10 * inv8, a20 * inv8, a30 * inv8, a40 * inv8, a50 * inv8, a60 * inv8, a70 * inv8,
    );
    let (b10, b11, b12, b13, b14, b15, b16, b17) = dct1d_8(
        a01 * inv8, a11 * inv8, a21 * inv8, a31 * inv8, a41 * inv8, a51 * inv8, a61 * inv8, a71 * inv8,
    );
    let (b20, b21, b22, b23, b24, b25, b26, b27) = dct1d_8(
        a02 * inv8, a12 * inv8, a22 * inv8, a32 * inv8, a42 * inv8, a52 * inv8, a62 * inv8, a72 * inv8,
    );
    let (b30, b31, b32, b33, b34, b35, b36, b37) = dct1d_8(
        a03 * inv8, a13 * inv8, a23 * inv8, a33 * inv8, a43 * inv8, a53 * inv8, a63 * inv8, a73 * inv8,
    );
    let (b40, b41, b42, b43, b44, b45, b46, b47) = dct1d_8(
        a04 * inv8, a14 * inv8, a24 * inv8, a34 * inv8, a44 * inv8, a54 * inv8, a64 * inv8, a74 * inv8,
    );
    let (b50, b51, b52, b53, b54, b55, b56, b57) = dct1d_8(
        a05 * inv8, a15 * inv8, a25 * inv8, a35 * inv8, a45 * inv8, a55 * inv8, a65 * inv8, a75 * inv8,
    );
    let (b60, b61, b62, b63, b64, b65, b66, b67) = dct1d_8(
        a06 * inv8, a16 * inv8, a26 * inv8, a36 * inv8, a46 * inv8, a56 * inv8, a66 * inv8, a76 * inv8,
    );
    let (b70, b71, b72, b73, b74, b75, b76, b77) = dct1d_8(
        a07 * inv8, a17 * inv8, a27 * inv8, a37 * inv8, a47 * inv8, a57 * inv8, a67 * inv8, a77 * inv8,
    );

    // Step 5: per-position scale by (inv8 / (s64[iy] * s64[ix])).
    // SCALE_64_TO_8 = [1.0, 0.9936866, 0.9748868, 0.9440181, 0.9017642, 0.8490575, 0.7870549, 0.7171081]
    let s0 = 1.0f32;
    let s1 = 0.9936866f32;
    let s2 = 0.9748868f32;
    let s3 = 0.9440181f32;
    let s4 = 0.9017642f32;
    let s5 = 0.8490575f32;
    let s6 = 0.7870549f32;
    let s7 = 0.7171081f32;

    let off = i * cpb;
    // Macro-style: row iy stride is 64.
    // Row 0 (iy=0)
    dst[off] = b00 * (inv8 / (s0 * s0));
    dst[off + 1usize] = b01 * (inv8 / (s0 * s1));
    dst[off + 2usize] = b02 * (inv8 / (s0 * s2));
    dst[off + 3usize] = b03 * (inv8 / (s0 * s3));
    dst[off + 4usize] = b04 * (inv8 / (s0 * s4));
    dst[off + 5usize] = b05 * (inv8 / (s0 * s5));
    dst[off + 6usize] = b06 * (inv8 / (s0 * s6));
    dst[off + 7usize] = b07 * (inv8 / (s0 * s7));
    // Row 1 (iy=1, +64)
    dst[off + 64usize] = b10 * (inv8 / (s1 * s0));
    dst[off + 65usize] = b11 * (inv8 / (s1 * s1));
    dst[off + 66usize] = b12 * (inv8 / (s1 * s2));
    dst[off + 67usize] = b13 * (inv8 / (s1 * s3));
    dst[off + 68usize] = b14 * (inv8 / (s1 * s4));
    dst[off + 69usize] = b15 * (inv8 / (s1 * s5));
    dst[off + 70usize] = b16 * (inv8 / (s1 * s6));
    dst[off + 71usize] = b17 * (inv8 / (s1 * s7));
    // Row 2 (iy=2, +128)
    dst[off + 128usize] = b20 * (inv8 / (s2 * s0));
    dst[off + 129usize] = b21 * (inv8 / (s2 * s1));
    dst[off + 130usize] = b22 * (inv8 / (s2 * s2));
    dst[off + 131usize] = b23 * (inv8 / (s2 * s3));
    dst[off + 132usize] = b24 * (inv8 / (s2 * s4));
    dst[off + 133usize] = b25 * (inv8 / (s2 * s5));
    dst[off + 134usize] = b26 * (inv8 / (s2 * s6));
    dst[off + 135usize] = b27 * (inv8 / (s2 * s7));
    // Row 3 (iy=3, +192)
    dst[off + 192usize] = b30 * (inv8 / (s3 * s0));
    dst[off + 193usize] = b31 * (inv8 / (s3 * s1));
    dst[off + 194usize] = b32 * (inv8 / (s3 * s2));
    dst[off + 195usize] = b33 * (inv8 / (s3 * s3));
    dst[off + 196usize] = b34 * (inv8 / (s3 * s4));
    dst[off + 197usize] = b35 * (inv8 / (s3 * s5));
    dst[off + 198usize] = b36 * (inv8 / (s3 * s6));
    dst[off + 199usize] = b37 * (inv8 / (s3 * s7));
    // Row 4 (iy=4, +256)
    dst[off + 256usize] = b40 * (inv8 / (s4 * s0));
    dst[off + 257usize] = b41 * (inv8 / (s4 * s1));
    dst[off + 258usize] = b42 * (inv8 / (s4 * s2));
    dst[off + 259usize] = b43 * (inv8 / (s4 * s3));
    dst[off + 260usize] = b44 * (inv8 / (s4 * s4));
    dst[off + 261usize] = b45 * (inv8 / (s4 * s5));
    dst[off + 262usize] = b46 * (inv8 / (s4 * s6));
    dst[off + 263usize] = b47 * (inv8 / (s4 * s7));
    // Row 5 (iy=5, +320)
    dst[off + 320usize] = b50 * (inv8 / (s5 * s0));
    dst[off + 321usize] = b51 * (inv8 / (s5 * s1));
    dst[off + 322usize] = b52 * (inv8 / (s5 * s2));
    dst[off + 323usize] = b53 * (inv8 / (s5 * s3));
    dst[off + 324usize] = b54 * (inv8 / (s5 * s4));
    dst[off + 325usize] = b55 * (inv8 / (s5 * s5));
    dst[off + 326usize] = b56 * (inv8 / (s5 * s6));
    dst[off + 327usize] = b57 * (inv8 / (s5 * s7));
    // Row 6 (iy=6, +384)
    dst[off + 384usize] = b60 * (inv8 / (s6 * s0));
    dst[off + 385usize] = b61 * (inv8 / (s6 * s1));
    dst[off + 386usize] = b62 * (inv8 / (s6 * s2));
    dst[off + 387usize] = b63 * (inv8 / (s6 * s3));
    dst[off + 388usize] = b64 * (inv8 / (s6 * s4));
    dst[off + 389usize] = b65 * (inv8 / (s6 * s5));
    dst[off + 390usize] = b66 * (inv8 / (s6 * s6));
    dst[off + 391usize] = b67 * (inv8 / (s6 * s7));
    // Row 7 (iy=7, +448)
    dst[off + 448usize] = b70 * (inv8 / (s7 * s0));
    dst[off + 449usize] = b71 * (inv8 / (s7 * s1));
    dst[off + 450usize] = b72 * (inv8 / (s7 * s2));
    dst[off + 451usize] = b73 * (inv8 / (s7 * s3));
    dst[off + 452usize] = b74 * (inv8 / (s7 * s4));
    dst[off + 453usize] = b75 * (inv8 / (s7 * s5));
    dst[off + 454usize] = b76 * (inv8 / (s7 * s6));
    dst[off + 455usize] = b77 * (inv8 / (s7 * s7));
}

/// LLF restore for DCT64×32 (8×4 DC-grid → 32 LLF positions per block
/// at coeffs[iy * 64 + ix] for iy in 0..4, ix in 0..8).
///
/// Math (mirrors `forks::reconstruct::restore_llf_dct64x32`):
/// 1. Read 8×4 DC values: `dc[iy*4 + ix]` for iy in 0..8, ix in 0..4.
/// 2. dct1d_4 on each of 8 rows.
/// 3. Transpose 8×4 → 4×8.
/// 4. dct1d_8 + 1/8 scale on each of 4 rows.
/// 5. Per-position scale by `1 / (4 * SCALE_32_TO_4[iy] * SCALE_64_TO_8[ix])`.
/// 6. Write to `dst[i * cpb + iy * 64 + ix]` for iy in 0..4, ix in 0..8.
///
/// `coeffs_per_block` must be 2048 (DCT64x32 standard size).
#[cube(launch_unchecked)]
pub fn set_llf_dct64x32_indexed_kernel(
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

    // Read 8×4 DC values (m[iy*4 + ix] for iy in 0..8, ix in 0..4).
    let r0 = by * stride + bx;
    let r1 = (by + 1usize) * stride + bx;
    let r2 = (by + 2usize) * stride + bx;
    let r3 = (by + 3usize) * stride + bx;
    let r4 = (by + 4usize) * stride + bx;
    let r5 = (by + 5usize) * stride + bx;
    let r6 = (by + 6usize) * stride + bx;
    let r7 = (by + 7usize) * stride + bx;
    let m00 = dc_grid[r0]; let m01 = dc_grid[r0 + 1usize]; let m02 = dc_grid[r0 + 2usize]; let m03 = dc_grid[r0 + 3usize];
    let m10 = dc_grid[r1]; let m11 = dc_grid[r1 + 1usize]; let m12 = dc_grid[r1 + 2usize]; let m13 = dc_grid[r1 + 3usize];
    let m20 = dc_grid[r2]; let m21 = dc_grid[r2 + 1usize]; let m22 = dc_grid[r2 + 2usize]; let m23 = dc_grid[r2 + 3usize];
    let m30 = dc_grid[r3]; let m31 = dc_grid[r3 + 1usize]; let m32 = dc_grid[r3 + 2usize]; let m33 = dc_grid[r3 + 3usize];
    let m40 = dc_grid[r4]; let m41 = dc_grid[r4 + 1usize]; let m42 = dc_grid[r4 + 2usize]; let m43 = dc_grid[r4 + 3usize];
    let m50 = dc_grid[r5]; let m51 = dc_grid[r5 + 1usize]; let m52 = dc_grid[r5 + 2usize]; let m53 = dc_grid[r5 + 3usize];
    let m60 = dc_grid[r6]; let m61 = dc_grid[r6 + 1usize]; let m62 = dc_grid[r6 + 2usize]; let m63 = dc_grid[r6 + 3usize];
    let m70 = dc_grid[r7]; let m71 = dc_grid[r7 + 1usize]; let m72 = dc_grid[r7 + 2usize]; let m73 = dc_grid[r7 + 3usize];

    // Step 2: dct1d_4 on each of 8 rows.
    let (a00, a01, a02, a03) = dct1d_4(m00, m01, m02, m03);
    let (a10, a11, a12, a13) = dct1d_4(m10, m11, m12, m13);
    let (a20, a21, a22, a23) = dct1d_4(m20, m21, m22, m23);
    let (a30, a31, a32, a33) = dct1d_4(m30, m31, m32, m33);
    let (a40, a41, a42, a43) = dct1d_4(m40, m41, m42, m43);
    let (a50, a51, a52, a53) = dct1d_4(m50, m51, m52, m53);
    let (a60, a61, a62, a63) = dct1d_4(m60, m61, m62, m63);
    let (a70, a71, a72, a73) = dct1d_4(m70, m71, m72, m73);

    // Step 3+4: transpose then dct1d_8 + 1/8 on rows. Transposed row k = column k.
    // We multiply by inv8 inside the dct1d_8 inputs to absorb the 1/8 scale.
    let inv8 = 0.125f32;
    let (b00, b01, b02, b03, b04, b05, b06, b07) = dct1d_8(
        a00 * inv8, a10 * inv8, a20 * inv8, a30 * inv8, a40 * inv8, a50 * inv8, a60 * inv8, a70 * inv8,
    );
    let (b10, b11, b12, b13, b14, b15, b16, b17) = dct1d_8(
        a01 * inv8, a11 * inv8, a21 * inv8, a31 * inv8, a41 * inv8, a51 * inv8, a61 * inv8, a71 * inv8,
    );
    let (b20, b21, b22, b23, b24, b25, b26, b27) = dct1d_8(
        a02 * inv8, a12 * inv8, a22 * inv8, a32 * inv8, a42 * inv8, a52 * inv8, a62 * inv8, a72 * inv8,
    );
    let (b30, b31, b32, b33, b34, b35, b36, b37) = dct1d_8(
        a03 * inv8, a13 * inv8, a23 * inv8, a33 * inv8, a43 * inv8, a53 * inv8, a63 * inv8, a73 * inv8,
    );

    // Step 5: per-position scale by 1 / (4 * SCALE_32_TO_4[iy] * SCALE_64_TO_8[ix]).
    let s32_0 = 1.0f32;
    let s32_1 = 0.9748868f32;
    let s32_2 = 0.9017642f32;
    let s32_3 = 0.7870549f32;
    let s64_0 = 1.0f32;
    let s64_1 = 0.9936866f32;
    let s64_2 = 0.9748868f32;
    let s64_3 = 0.9440181f32;
    let s64_4 = 0.9017642f32;
    let s64_5 = 0.8490575f32;
    let s64_6 = 0.7870549f32;
    let s64_7 = 0.7171081f32;

    let off = i * cpb;
    // Row 0
    dst[off]            = b00 * (1.0f32 / (4.0f32 * s32_0 * s64_0));
    dst[off + 1usize]   = b01 * (1.0f32 / (4.0f32 * s32_0 * s64_1));
    dst[off + 2usize]   = b02 * (1.0f32 / (4.0f32 * s32_0 * s64_2));
    dst[off + 3usize]   = b03 * (1.0f32 / (4.0f32 * s32_0 * s64_3));
    dst[off + 4usize]   = b04 * (1.0f32 / (4.0f32 * s32_0 * s64_4));
    dst[off + 5usize]   = b05 * (1.0f32 / (4.0f32 * s32_0 * s64_5));
    dst[off + 6usize]   = b06 * (1.0f32 / (4.0f32 * s32_0 * s64_6));
    dst[off + 7usize]   = b07 * (1.0f32 / (4.0f32 * s32_0 * s64_7));
    // Row 1 (+64)
    dst[off + 64usize]  = b10 * (1.0f32 / (4.0f32 * s32_1 * s64_0));
    dst[off + 65usize]  = b11 * (1.0f32 / (4.0f32 * s32_1 * s64_1));
    dst[off + 66usize]  = b12 * (1.0f32 / (4.0f32 * s32_1 * s64_2));
    dst[off + 67usize]  = b13 * (1.0f32 / (4.0f32 * s32_1 * s64_3));
    dst[off + 68usize]  = b14 * (1.0f32 / (4.0f32 * s32_1 * s64_4));
    dst[off + 69usize]  = b15 * (1.0f32 / (4.0f32 * s32_1 * s64_5));
    dst[off + 70usize]  = b16 * (1.0f32 / (4.0f32 * s32_1 * s64_6));
    dst[off + 71usize]  = b17 * (1.0f32 / (4.0f32 * s32_1 * s64_7));
    // Row 2 (+128)
    dst[off + 128usize] = b20 * (1.0f32 / (4.0f32 * s32_2 * s64_0));
    dst[off + 129usize] = b21 * (1.0f32 / (4.0f32 * s32_2 * s64_1));
    dst[off + 130usize] = b22 * (1.0f32 / (4.0f32 * s32_2 * s64_2));
    dst[off + 131usize] = b23 * (1.0f32 / (4.0f32 * s32_2 * s64_3));
    dst[off + 132usize] = b24 * (1.0f32 / (4.0f32 * s32_2 * s64_4));
    dst[off + 133usize] = b25 * (1.0f32 / (4.0f32 * s32_2 * s64_5));
    dst[off + 134usize] = b26 * (1.0f32 / (4.0f32 * s32_2 * s64_6));
    dst[off + 135usize] = b27 * (1.0f32 / (4.0f32 * s32_2 * s64_7));
    // Row 3 (+192)
    dst[off + 192usize] = b30 * (1.0f32 / (4.0f32 * s32_3 * s64_0));
    dst[off + 193usize] = b31 * (1.0f32 / (4.0f32 * s32_3 * s64_1));
    dst[off + 194usize] = b32 * (1.0f32 / (4.0f32 * s32_3 * s64_2));
    dst[off + 195usize] = b33 * (1.0f32 / (4.0f32 * s32_3 * s64_3));
    dst[off + 196usize] = b34 * (1.0f32 / (4.0f32 * s32_3 * s64_4));
    dst[off + 197usize] = b35 * (1.0f32 / (4.0f32 * s32_3 * s64_5));
    dst[off + 198usize] = b36 * (1.0f32 / (4.0f32 * s32_3 * s64_6));
    dst[off + 199usize] = b37 * (1.0f32 / (4.0f32 * s32_3 * s64_7));
}

/// LLF restore for DCT32×64 (4×8 DC-grid → 32 LLF positions per block
/// at coeffs[iy * 64 + ix] for iy in 0..4, ix in 0..8).
///
/// Math (mirrors `forks::reconstruct::restore_llf_dct32x64`):
/// 1. Read 4×8 DC values: `dc[iy*8 + ix]` for iy in 0..4, ix in 0..8.
/// 2. dct1d_8 + 1/8 scale on each of 4 rows.
/// 3. Transpose 4×8 → 8×4.
/// 4. dct1d_4 on each of 8 rows.
/// 5. Transpose back 8×4 → 4×8.
/// 6. Per-position scale by `1 / (4 * SCALE_32_TO_4[iy] * SCALE_64_TO_8[ix])`.
/// 7. Write to `dst[i * cpb + iy * 64 + ix]` for iy in 0..4, ix in 0..8.
#[cube(launch_unchecked)]
pub fn set_llf_dct32x64_indexed_kernel(
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

    // Read 4×8 DC values.
    let r0 = by * stride + bx;
    let r1 = (by + 1usize) * stride + bx;
    let r2 = (by + 2usize) * stride + bx;
    let r3 = (by + 3usize) * stride + bx;
    let m00 = dc_grid[r0]; let m01 = dc_grid[r0 + 1usize]; let m02 = dc_grid[r0 + 2usize]; let m03 = dc_grid[r0 + 3usize];
    let m04 = dc_grid[r0 + 4usize]; let m05 = dc_grid[r0 + 5usize]; let m06 = dc_grid[r0 + 6usize]; let m07 = dc_grid[r0 + 7usize];
    let m10 = dc_grid[r1]; let m11 = dc_grid[r1 + 1usize]; let m12 = dc_grid[r1 + 2usize]; let m13 = dc_grid[r1 + 3usize];
    let m14 = dc_grid[r1 + 4usize]; let m15 = dc_grid[r1 + 5usize]; let m16 = dc_grid[r1 + 6usize]; let m17 = dc_grid[r1 + 7usize];
    let m20 = dc_grid[r2]; let m21 = dc_grid[r2 + 1usize]; let m22 = dc_grid[r2 + 2usize]; let m23 = dc_grid[r2 + 3usize];
    let m24 = dc_grid[r2 + 4usize]; let m25 = dc_grid[r2 + 5usize]; let m26 = dc_grid[r2 + 6usize]; let m27 = dc_grid[r2 + 7usize];
    let m30 = dc_grid[r3]; let m31 = dc_grid[r3 + 1usize]; let m32 = dc_grid[r3 + 2usize]; let m33 = dc_grid[r3 + 3usize];
    let m34 = dc_grid[r3 + 4usize]; let m35 = dc_grid[r3 + 5usize]; let m36 = dc_grid[r3 + 6usize]; let m37 = dc_grid[r3 + 7usize];

    // Step 2: dct1d_8 + 1/8 on each of 4 rows.
    let inv8 = 0.125f32;
    let (a00, a01, a02, a03, a04, a05, a06, a07) = dct1d_8(m00, m01, m02, m03, m04, m05, m06, m07);
    let (a10, a11, a12, a13, a14, a15, a16, a17) = dct1d_8(m10, m11, m12, m13, m14, m15, m16, m17);
    let (a20, a21, a22, a23, a24, a25, a26, a27) = dct1d_8(m20, m21, m22, m23, m24, m25, m26, m27);
    let (a30, a31, a32, a33, a34, a35, a36, a37) = dct1d_8(m30, m31, m32, m33, m34, m35, m36, m37);

    // Step 3+4: transpose then dct1d_4 on rows. Transposed row k = column k.
    // Multiply each input by inv8 to absorb the row 1/8 scale.
    // Transposed has 8 rows of 4: row ix = (a0_ix, a1_ix, a2_ix, a3_ix).
    let (c00, c01, c02, c03) = dct1d_4(a00 * inv8, a10 * inv8, a20 * inv8, a30 * inv8);
    let (c10, c11, c12, c13) = dct1d_4(a01 * inv8, a11 * inv8, a21 * inv8, a31 * inv8);
    let (c20, c21, c22, c23) = dct1d_4(a02 * inv8, a12 * inv8, a22 * inv8, a32 * inv8);
    let (c30, c31, c32, c33) = dct1d_4(a03 * inv8, a13 * inv8, a23 * inv8, a33 * inv8);
    let (c40, c41, c42, c43) = dct1d_4(a04 * inv8, a14 * inv8, a24 * inv8, a34 * inv8);
    let (c50, c51, c52, c53) = dct1d_4(a05 * inv8, a15 * inv8, a25 * inv8, a35 * inv8);
    let (c60, c61, c62, c63) = dct1d_4(a06 * inv8, a16 * inv8, a26 * inv8, a36 * inv8);
    let (c70, c71, c72, c73) = dct1d_4(a07 * inv8, a17 * inv8, a27 * inv8, a37 * inv8);

    // Step 5: transpose back 8×4 → 4×8. result[iy*8 + ix] = c[ix*4 + iy].
    // For iy in 0..4, ix in 0..8: result[iy*8 + ix] = c{ix, iy}.
    // We index c by (ix, iy):  c00..c03 = ix=0; c10..c13 = ix=1; etc.
    // result row iy = (c0_iy, c1_iy, c2_iy, c3_iy, c4_iy, c5_iy, c6_iy, c7_iy)
    let r00 = c00; let r01 = c10; let r02 = c20; let r03 = c30; let r04 = c40; let r05 = c50; let r06 = c60; let r07 = c70;
    let r10 = c01; let r11 = c11; let r12 = c21; let r13 = c31; let r14 = c41; let r15 = c51; let r16 = c61; let r17 = c71;
    let r20 = c02; let r21 = c12; let r22 = c22; let r23 = c32; let r24 = c42; let r25 = c52; let r26 = c62; let r27 = c72;
    let r30 = c03; let r31 = c13; let r32 = c23; let r33 = c33; let r34 = c43; let r35 = c53; let r36 = c63; let r37 = c73;

    // Step 6: per-position scale by 1 / (4 * SCALE_32_TO_4[iy] * SCALE_64_TO_8[ix]).
    let s32_0 = 1.0f32;
    let s32_1 = 0.9748868f32;
    let s32_2 = 0.9017642f32;
    let s32_3 = 0.7870549f32;
    let s64_0 = 1.0f32;
    let s64_1 = 0.9936866f32;
    let s64_2 = 0.9748868f32;
    let s64_3 = 0.9440181f32;
    let s64_4 = 0.9017642f32;
    let s64_5 = 0.8490575f32;
    let s64_6 = 0.7870549f32;
    let s64_7 = 0.7171081f32;

    let off = i * cpb;
    // Row 0
    dst[off]            = r00 * (1.0f32 / (4.0f32 * s32_0 * s64_0));
    dst[off + 1usize]   = r01 * (1.0f32 / (4.0f32 * s32_0 * s64_1));
    dst[off + 2usize]   = r02 * (1.0f32 / (4.0f32 * s32_0 * s64_2));
    dst[off + 3usize]   = r03 * (1.0f32 / (4.0f32 * s32_0 * s64_3));
    dst[off + 4usize]   = r04 * (1.0f32 / (4.0f32 * s32_0 * s64_4));
    dst[off + 5usize]   = r05 * (1.0f32 / (4.0f32 * s32_0 * s64_5));
    dst[off + 6usize]   = r06 * (1.0f32 / (4.0f32 * s32_0 * s64_6));
    dst[off + 7usize]   = r07 * (1.0f32 / (4.0f32 * s32_0 * s64_7));
    // Row 1 (+64)
    dst[off + 64usize]  = r10 * (1.0f32 / (4.0f32 * s32_1 * s64_0));
    dst[off + 65usize]  = r11 * (1.0f32 / (4.0f32 * s32_1 * s64_1));
    dst[off + 66usize]  = r12 * (1.0f32 / (4.0f32 * s32_1 * s64_2));
    dst[off + 67usize]  = r13 * (1.0f32 / (4.0f32 * s32_1 * s64_3));
    dst[off + 68usize]  = r14 * (1.0f32 / (4.0f32 * s32_1 * s64_4));
    dst[off + 69usize]  = r15 * (1.0f32 / (4.0f32 * s32_1 * s64_5));
    dst[off + 70usize]  = r16 * (1.0f32 / (4.0f32 * s32_1 * s64_6));
    dst[off + 71usize]  = r17 * (1.0f32 / (4.0f32 * s32_1 * s64_7));
    // Row 2 (+128)
    dst[off + 128usize] = r20 * (1.0f32 / (4.0f32 * s32_2 * s64_0));
    dst[off + 129usize] = r21 * (1.0f32 / (4.0f32 * s32_2 * s64_1));
    dst[off + 130usize] = r22 * (1.0f32 / (4.0f32 * s32_2 * s64_2));
    dst[off + 131usize] = r23 * (1.0f32 / (4.0f32 * s32_2 * s64_3));
    dst[off + 132usize] = r24 * (1.0f32 / (4.0f32 * s32_2 * s64_4));
    dst[off + 133usize] = r25 * (1.0f32 / (4.0f32 * s32_2 * s64_5));
    dst[off + 134usize] = r26 * (1.0f32 / (4.0f32 * s32_2 * s64_6));
    dst[off + 135usize] = r27 * (1.0f32 / (4.0f32 * s32_2 * s64_7));
    // Row 3 (+192)
    dst[off + 192usize] = r30 * (1.0f32 / (4.0f32 * s32_3 * s64_0));
    dst[off + 193usize] = r31 * (1.0f32 / (4.0f32 * s32_3 * s64_1));
    dst[off + 194usize] = r32 * (1.0f32 / (4.0f32 * s32_3 * s64_2));
    dst[off + 195usize] = r33 * (1.0f32 / (4.0f32 * s32_3 * s64_3));
    dst[off + 196usize] = r34 * (1.0f32 / (4.0f32 * s32_3 * s64_4));
    dst[off + 197usize] = r35 * (1.0f32 / (4.0f32 * s32_3 * s64_5));
    dst[off + 198usize] = r36 * (1.0f32 / (4.0f32 * s32_3 * s64_6));
    dst[off + 199usize] = r37 * (1.0f32 / (4.0f32 * s32_3 * s64_7));
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
