// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/reconstruct.rs (BSD-3-Clause via libjxl
// + AGPL/commercial), with the SIMD calls substituted for GPU launches.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted final-stage reconstruction utilities.
//!
//! Currently covers the cleanly substitutable, kernel-bound pieces of
//! `jxl_encoder::vardct::reconstruct`:
//! - `gab_smooth_gpu` — 3-channel decoder gab smoothing
//! - `xyb_to_linear_rgb_planar_gpu` — XYB → planar linear RGB
//! - `xyb_to_linear_rgb_gpu` — XYB → interleaved linear RGB
//!   (re-interleaves on host after GPU planar inverse)
//!
//! Not yet covered (algorithm-heavy, not pure SIMD):
//! - `restore_llf_from_dc` (Hadamard inverse for large transforms)
//! - `idct_for_strategy` (per-strategy IDCT dispatch)
//! - `reconstruct_xyb_impl` (full pipeline orchestration)
//!
//! These can be progressively forked once we have GPU kernels for the
//! per-strategy IDCT dispatch logic (we have all the IDCT math; we just
//! need a host-side strategy selector that picks the right kernel).
//!
//! Reshape vs upstream:
//! - `gab_smooth`: original CPU code reuses one scratch buffer across
//!   all 3 channels. GPU kernel manages its own buffer; we just call it
//!   3 times sequentially. Future fusion: a 3-channel GPU kernel that
//!   does X+Y+B in one launch.
//! - `xyb_to_linear_rgb` (interleaved): GPU kernel returns planar
//!   buffers; we re-interleave on host. The alternative (interleaved
//!   GPU output) would force stride-3 stores on CUDA — slower than
//!   planar + host re-interleave at the sizes we care about.

use alloc::vec;
use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

/// `INV_DC_QUANT[c]` constants — channel-specific inverse DC quantizers
/// from upstream `jxl_encoder::vardct::quant::INV_DC_QUANT`. Used by
/// the DC override step of `reconstruct_xyb_impl`.
pub const INV_DC_QUANT: [f32; 3] = [4096.0, 512.0, 256.0];

/// `DCT_RESAMPLE_SCALE_16_TO_2[i]` — scale factors for the 2-point
/// resample used by the DC-from-DCT16 forward operation. Bit-for-bit
/// from upstream `jxl_encoder::vardct::dct::constants::DCT_RESAMPLE_SCALE_16_TO_2`.
pub const DCT_RESAMPLE_SCALE_16_TO_2: [f32; 2] = [1.000_000_000_0, 0.901_764_2];

/// Dequantize a single channel's DC value with the channel-specific
/// CfL contribution from Y. Mirrors the inline `dequant_dc` closure in
/// upstream's `restore_llf_from_dc`.
///
/// - `channel == 0` (X) or `channel == 1` (Y): no CfL → returns
///   `quant_dc / inv_factor`.
/// - `channel == 2` (B): adds `quant_dc_y * 0.5 / inv_factor` (the
///   fixed B-channel DC-level CfL contribution).
///
/// `inv_factor = INV_DC_QUANT[channel] * scale_dc`.
#[inline]
pub fn dequant_dc_channel(quant_dc: f32, quant_dc_y: f32, channel: usize, scale_dc: f32) -> f32 {
    let dc_cfl_factor: f32 = if channel == 2 { 0.5 } else { 0.0 };
    let inv_factor = INV_DC_QUANT[channel] * scale_dc;
    (quant_dc + quant_dc_y * dc_cfl_factor) / inv_factor
}

/// `DCT_RESAMPLE_SCALE_32_TO_4[i]` — scale factors for the 4-point
/// resample used by the DC-from-DCT32 forward operation.
/// Bit-for-bit from upstream.
pub const DCT_RESAMPLE_SCALE_32_TO_4: [f32; 4] =
    [1.0, 0.974_886_8, 0.901_764_2, 0.787_054_9];

/// In-place 4-point DCT (libjxl `dct1d_4`). Pure scalar, used by the
/// DCT32 LLF restoration.
fn dct1d_4(mem: &mut [f32]) {
    const SQRT2: f32 = 1.414_213_5;
    const WC4: [f32; 2] = [0.541_196_1, 1.306_563_0];
    let (a, b, c, d) = (mem[0], mem[1], mem[2], mem[3]);
    let t0 = a + d;
    let t1 = b + c;
    let t2 = a - d;
    let t3 = b - c;
    let u0 = t0 + t1;
    let u1 = t0 - t1;
    let v0 = t2 * WC4[0];
    let v1 = t3 * WC4[1];
    let w0 = v0 + v1;
    let w1 = v0 - v1;
    let b0 = SQRT2 * w0 + w1;
    mem[0] = u0;
    mem[1] = b0;
    mem[2] = u1;
    mem[3] = w1;
}

/// In-place 2-point DCT (libjxl `dct1d_2`). Pure scalar:
/// `[a, b] -> [a + b, a - b]`. Used by the rectangular DCT32×16 /
/// DCT16×32 LLF restoration.
fn dct1d_2(mem: &mut [f32]) {
    let a = mem[0];
    let b = mem[1];
    mem[0] = a + b;
    mem[1] = a - b;
}

/// Restore the 2×4 LLF coefficients of a DCT32×16 block from the 4×2
/// stored DC grid. Mirrors upstream `restore_llf_from_dc` for
/// `RAW_STRATEGY_DCT32X16` (reconstruct.rs lines 643-684).
///
/// `dc_grid[iy * 2 + ix]` is the dequantized DC value at sub-block
/// `(iy, ix)` within the 4×2 region (4 rows × 2 cols, post-swap layout).
///
/// Returns `[f32; 8]` ordered as `out[iy * 4 + ix]` for `iy in 0..2,
/// ix in 0..4` — to be written at coefficient positions
/// `coeffs[iy * 32 + ix]` in the 32×16 coefficient block.
///
/// Math: forward 2-pt DCT on each of 4 rows → transpose 4×2 → 2×4 →
/// forward 4-pt DCT on each of 2 rows → divide by `(scale * 8)`.
/// The 8 = `dct1d_2(2) * dct1d_4(4)` forward gain.
pub fn restore_llf_dct32x16(dc_grid: [f32; 8]) -> [f32; 8] {
    let mut block = dc_grid;
    // Forward 2-pt DCT on rows (4 rows of 2).
    for iy in 0..4 {
        dct1d_2(&mut block[iy * 2..(iy + 1) * 2]);
    }
    // Transpose 4×2 → 2×4.
    let mut t = [0.0_f32; 8];
    for iy in 0..4 {
        for ix in 0..2 {
            t[ix * 4 + iy] = block[iy * 2 + ix];
        }
    }
    // Forward 4-pt DCT on rows (2 rows of 4).
    dct1d_4(&mut t[0..4]);
    dct1d_4(&mut t[4..8]);
    // Apply per-position scale + 1/8 normalization.
    let mut out = [0.0_f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            let scale = DCT_RESAMPLE_SCALE_16_TO_2[iy] * DCT_RESAMPLE_SCALE_32_TO_4[ix];
            out[iy * 4 + ix] = t[iy * 4 + ix] / (scale * 8.0);
        }
    }
    out
}

/// Restore the 2×4 LLF coefficients of a DCT16×32 block from the 2×4
/// stored DC grid. Mirrors upstream `restore_llf_from_dc` for
/// `RAW_STRATEGY_DCT16X32` (reconstruct.rs lines 686-734).
///
/// `dc_grid[iy * 4 + ix]` is the dequantized DC value at sub-block
/// `(iy, ix)` within the 2×4 region (2 rows × 4 cols, post-swap layout).
///
/// Returns `[f32; 8]` ordered as `out[iy * 4 + ix]` for `iy in 0..2,
/// ix in 0..4` — to be written at coefficient positions
/// `coeffs[iy * 32 + ix]` in the 16×32 coefficient block.
///
/// Math: forward 4-pt DCT on each of 2 rows → transpose 2×4 → 4×2 →
/// forward 2-pt DCT on each of 4 rows → transpose 4×2 → 2×4 →
/// divide by `(scale * 8)`.
pub fn restore_llf_dct16x32(dc_grid: [f32; 8]) -> [f32; 8] {
    let mut block = dc_grid;
    // Forward 4-pt DCT on rows (2 rows of 4).
    dct1d_4(&mut block[0..4]);
    dct1d_4(&mut block[4..8]);
    // Transpose 2×4 → 4×2.
    let mut t = [0.0_f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            t[ix * 2 + iy] = block[iy * 4 + ix];
        }
    }
    // Forward 2-pt DCT on rows (4 rows of 2).
    for iy in 0..4 {
        dct1d_2(&mut t[iy * 2..(iy + 1) * 2]);
    }
    // Transpose back 4×2 → 2×4.
    let mut result = [0.0_f32; 8];
    for iy in 0..4 {
        for ix in 0..2 {
            result[ix * 4 + iy] = t[iy * 2 + ix];
        }
    }
    // Apply per-position scale + 1/8 normalization.
    let mut out = [0.0_f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            let scale = DCT_RESAMPLE_SCALE_16_TO_2[iy] * DCT_RESAMPLE_SCALE_32_TO_4[ix];
            out[iy * 4 + ix] = result[iy * 4 + ix] / (scale * 8.0);
        }
    }
    out
}

/// Restore the 4×4 LLF coefficients of a DCT32×32 block from the 4×4
/// stored DC grid. Mirrors upstream `restore_llf_from_dc` for
/// `RAW_STRATEGY_DCT32X32` (reconstruct.rs lines 600-641).
///
/// `dc_grid[iy * 4 + ix]` is the dequantized DC value at sub-block
/// `(iy, ix)` within the 4×4 region the DCT32×32 covers (already
/// produced by [`dequant_dc_channel`] for each of `(by..by+4, bx..bx+4)`).
///
/// Output layout: returns `[f32; 16]` ordered to be written at
/// coefficient positions `coeffs[iy * 32 + ix]` for `iy, ix in 0..4`.
/// The caller is responsible for placing the 16 values at the right
/// positions in the larger 32×32 coefficient buffer.
///
/// Math (inverse of `dc_from_dct_32x32`):
/// ```text
///   Forward: scale + 4×4 IDCT (idct1d_4 on rows, transpose,
///            idct1d_4 on rows). The 4×4 IDCT is the inverse of
///            our 4-point DCT divided by 4 (libjxl IDCT
///            normalization).
///   Inverse: forward 4-point DCT on rows, transpose, forward
///            4-point DCT on rows, then divide by (scale * 16).
/// ```
/// where `scale = DCT_RESAMPLE_SCALE_32_TO_4[iy] *
///                DCT_RESAMPLE_SCALE_32_TO_4[ix]`.
pub fn restore_llf_dct32x32(dc_grid: [f32; 16]) -> [f32; 16] {
    let mut block = dc_grid;
    // Forward 4pt DCT on rows.
    dct1d_4(&mut block[0..4]);
    dct1d_4(&mut block[4..8]);
    dct1d_4(&mut block[8..12]);
    dct1d_4(&mut block[12..16]);
    // Transpose 4×4.
    let mut transposed = [0.0_f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            transposed[ix * 4 + iy] = block[iy * 4 + ix];
        }
    }
    // Forward 4pt DCT on rows.
    dct1d_4(&mut transposed[0..4]);
    dct1d_4(&mut transposed[4..8]);
    dct1d_4(&mut transposed[8..12]);
    dct1d_4(&mut transposed[12..16]);
    // Apply per-position scale + 1/16 normalization.
    let mut out = [0.0_f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            let scale = DCT_RESAMPLE_SCALE_32_TO_4[iy] * DCT_RESAMPLE_SCALE_32_TO_4[ix];
            out[iy * 4 + ix] = transposed[iy * 4 + ix] / (scale * 16.0);
        }
    }
    out
}

/// Restore the 2 LLF coefficients of a DCT16×8 or DCT8×16 block from
/// the 2 stored DC values. Mirrors upstream
/// `restore_llf_from_dc` for `RAW_STRATEGY_DCT16X8` /
/// `RAW_STRATEGY_DCT8X16` (reconstruct.rs lines 546-570).
///
/// Inputs:
/// - `dc0`, `dc1`: the two stored DC values (already dequantized via
///   [`dequant_dc_channel`]). For DCT16×8 these come from the
///   vertically-adjacent pair `(by, by+1)`; for DCT8×16 the
///   horizontally-adjacent pair `(bx, bx+1)`.
///
/// Returns `[llf0, llf1]` as a `[f32; 2]` ready to be written into
/// `coeffs[0]` and `coeffs[1]` of the rectangular coefficient block.
///
/// Math (inverse of `dc_from_dct_16x8` / `dc_from_dct_8x16`):
/// ```text
///   Forward: dc0 = llf0 * s0 + llf1 * s1
///            dc1 = llf0 * s0 - llf1 * s1
///   Inverse: llf0 = (dc0 + dc1) / (2 * s0)
///            llf1 = (dc0 - dc1) / (2 * s1)
/// ```
/// where `s0 = DCT_RESAMPLE_SCALE_16_TO_2[0] = 1.0` and
/// `s1 = DCT_RESAMPLE_SCALE_16_TO_2[1] ≈ 0.9018`. The factor 2 comes
/// from the 2-point Hadamard's `H * H = 2 * I` self-product.
#[inline]
pub fn restore_llf_dct16x8_or_8x16(dc0: f32, dc1: f32) -> [f32; 2] {
    let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
    let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];
    [(dc0 + dc1) / (2.0 * s0), (dc0 - dc1) / (2.0 * s1)]
}

/// Restore the 2×2 LLF coefficients of a DCT16×16 block from the
/// 2×2 stored DC grid. Mirrors upstream `restore_llf_from_dc` for
/// `RAW_STRATEGY_DCT16X16` (reconstruct.rs lines 572-598).
///
/// `dc_grid[iy * 2 + ix]` is the dequantized DC value at sub-block
/// `(iy, ix)` within the 2×2 region the DCT16×16 covers (already
/// produced by [`dequant_dc_channel`] for each of `(by..by+2, bx..bx+2)`).
///
/// Returns `[llf00, llf01, llf10, llf11]` to be written at coefficient
/// positions `[0, 1, 16, 17]` of the 16×16 coefficient block.
///
/// Math (inverse of `dc_from_dct_16x16` — 2-point row+column DCT
/// followed by SCALE_16_TO_2 scaling, where the 2-point DCT is
/// Hadamard with `H * H = 4 * I` for the 2×2):
/// ```text
///   h00 = dc00 + dc01 + dc10 + dc11
///   h01 = dc00 + dc01 - dc10 - dc11
///   h10 = dc00 - dc01 + dc10 - dc11
///   h11 = dc00 - dc01 - dc10 + dc11
///   llf00 = h00 / (4 * s0 * s0)
///   llf01 = h01 / (4 * s0 * s1)
///   llf10 = h10 / (4 * s1 * s0)
///   llf11 = h11 / (4 * s1 * s1)
/// ```
#[inline]
pub fn restore_llf_dct16x16(dc_grid: [f32; 4]) -> [f32; 4] {
    let h00 = dc_grid[0] + dc_grid[1] + dc_grid[2] + dc_grid[3];
    let h01 = dc_grid[0] + dc_grid[1] - dc_grid[2] - dc_grid[3];
    let h10 = dc_grid[0] - dc_grid[1] + dc_grid[2] - dc_grid[3];
    let h11 = dc_grid[0] - dc_grid[1] - dc_grid[2] + dc_grid[3];
    let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
    let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];
    [
        h00 / (4.0 * s0 * s0),
        h01 / (4.0 * s0 * s1),
        h10 / (4.0 * s1 * s0),
        h11 / (4.0 * s1 * s1),
    ]
}

/// DC restoration for the DCT8 fast path of upstream's
/// `reconstruct_xyb`. Pure scalar — bit-for-bit copy of upstream
/// (reconstruct.rs lines 297-317).
///
/// Inputs:
/// - `dq_x`/`dq_y`/`dq_b`: 64-element dequantized coefficient arrays
///   for the block (output of dequant_dct8 — positions 1..64 are AC).
/// - `quant_dc_x`/`quant_dc_y`/`quant_dc_b`: stored DC values
///   (typically `i16`, cast to `f32` here).
/// - `scale_dc`: from upstream `params.scale_dc`.
///
/// Behavior (matches upstream):
/// 1. Compute per-channel `inv_factor[c] = INV_DC_QUANT[c] * scale_dc`.
/// 2. Override the DC slot:
///    - `dq_y[0] = quant_dc_y / inv_factor[1]`
///    - `dq_x[0] = quant_dc_x / inv_factor[0]`
///    - `dq_b[0] = (quant_dc_b + quant_dc_y * dc_cfl_factor_b) / inv_factor[2]`
///      where `dc_cfl_factor_b = 0.5` (B-channel DC-level CfL).
///
/// Note: the AC-level CfL (per-tile `ytox_ratio` / `ytob_ratio`) is
/// already applied during dequant. This function applies *only* the
/// DC-level CfL — a separate fixed 0.5× contribution from Y to B at
/// position 0.
pub fn restore_dct8_dc_override(
    dq_x: &mut [f32; 64],
    dq_y: &mut [f32; 64],
    dq_b: &mut [f32; 64],
    quant_dc_x: f32,
    quant_dc_y: f32,
    quant_dc_b: f32,
    scale_dc: f32,
) {
    let inv_factor = [
        INV_DC_QUANT[0] * scale_dc,
        INV_DC_QUANT[1] * scale_dc,
        INV_DC_QUANT[2] * scale_dc,
    ];
    const DC_CFL_FACTOR_B: f32 = 0.5;
    dq_y[0] = quant_dc_y / inv_factor[1];
    dq_x[0] = quant_dc_x / inv_factor[0];
    dq_b[0] = (quant_dc_b + quant_dc_y * DC_CFL_FACTOR_B) / inv_factor[2];
}

/// Batched form of [`restore_dct8_dc_override`] for `n_blocks` 64-coef
/// DCT8 blocks. Operates on flat slices in block-major layout —
/// matches what `dequant_dct8_blocks_gpu` returns.
///
/// `quant_dc_*` are per-block DC values (length `n_blocks`, typically
/// `i16`-stored, passed as `f32` via `as f32` cast). `dq_*` are
/// dequantized coefficient blocks (length `n_blocks * 64`); only the
/// `[b * 64]` slot of each block is mutated.
///
/// Bit-for-bit equivalent to running [`restore_dct8_dc_override`]
/// in a per-block loop. Pure scalar — kept on host because the
/// per-block work is just three scalar divides and an FMA, dwarfed
/// by GPU-launch overhead at typical batch sizes.
pub fn restore_dct8_dc_override_batched(
    dq_x: &mut [f32],
    dq_y: &mut [f32],
    dq_b: &mut [f32],
    quant_dc_x: &[f32],
    quant_dc_y: &[f32],
    quant_dc_b: &[f32],
    scale_dc: f32,
) {
    let n_blocks = quant_dc_y.len();
    debug_assert_eq!(quant_dc_x.len(), n_blocks);
    debug_assert_eq!(quant_dc_b.len(), n_blocks);
    debug_assert_eq!(dq_x.len(), n_blocks * 64);
    debug_assert_eq!(dq_y.len(), n_blocks * 64);
    debug_assert_eq!(dq_b.len(), n_blocks * 64);
    let inv_factor = [
        INV_DC_QUANT[0] * scale_dc,
        INV_DC_QUANT[1] * scale_dc,
        INV_DC_QUANT[2] * scale_dc,
    ];
    const DC_CFL_FACTOR_B: f32 = 0.5;
    for b in 0..n_blocks {
        let dy = quant_dc_y[b];
        dq_y[b * 64] = dy / inv_factor[1];
        dq_x[b * 64] = quant_dc_x[b] / inv_factor[0];
        dq_b[b * 64] = (quant_dc_b[b] + dy * DC_CFL_FACTOR_B) / inv_factor[2];
    }
}

/// Reconstruct XYB pixel planes from quantized DC + AC coefficients,
/// for an image where every block is DCT8. Mirrors the DCT8 fast path
/// of upstream `reconstruct_xyb_impl` (reconstruct.rs lines 268-344)
/// composed end-to-end on GPU.
///
/// Pipeline (host-orchestrated, 4 GPU launches per image):
/// 1. `dequant_dct8_blocks_gpu` — batched 3-channel dequant + AC-level
///    CfL fold (one launch).
/// 2. host: `restore_dct8_dc_override_batched` — DC override with the
///    fixed 0.5× Y→B DC-level CfL.
/// 3. `apply_idct_batch_gpu(DCT8)` per channel — three launches.
/// 4. host: scatter the per-block 8×8 outputs into padded
///    `(xsize_blocks * 8) × (ysize_blocks * 8)` planes.
///
/// Inputs are all flat block-major slices (`n_blocks * 64` for AC,
/// `n_blocks` for per-block scalars). Per-block CfL factors must
/// already be resolved from the per-tile CfL map by the caller.
///
/// Returns `[plane_x, plane_y, plane_b]` of length
/// `xsize_blocks * ysize_blocks * 64` (= padded width × padded height).
///
/// **Note**: this is the all-blocks-are-DCT8 path. Real images use a
/// mix of strategies via the AC strategy map; supporting that
/// requires the per-strategy IDCT dispatch + scatter for non-DCT8
/// blocks, which is a separate piece of `reconstruct_xyb_impl`.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_xyb_dct8_only_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    quant_dc_x: &[f32],
    quant_dc_y: &[f32],
    quant_dc_b: &[f32],
    quant_ac_x: &[i32],
    quant_ac_y: &[i32],
    quant_ac_b: &[i32],
    weights_x_per_block: &[f32; 64],
    weights_y_per_block: &[f32; 64],
    weights_b_per_block: &[f32; 64],
    qac_qm_x: &[f32],
    qac_qm_y: &[f32],
    qac_qm_b: &[f32],
    x_factor: &[f32],
    b_factor: &[f32],
    scale_dc: f32,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> [Vec<f32>; 3] {
    use crate::forks::dequant::dequant_dct8_blocks_gpu;
    use crate::forks::transform::{apply_idct_batch_gpu, RAW_STRATEGY_DCT};

    let n_blocks = xsize_blocks * ysize_blocks;
    debug_assert_eq!(quant_dc_x.len(), n_blocks);
    debug_assert_eq!(quant_dc_y.len(), n_blocks);
    debug_assert_eq!(quant_dc_b.len(), n_blocks);
    debug_assert_eq!(quant_ac_x.len(), n_blocks * 64);
    debug_assert_eq!(quant_ac_y.len(), n_blocks * 64);
    debug_assert_eq!(quant_ac_b.len(), n_blocks * 64);
    debug_assert_eq!(qac_qm_x.len(), n_blocks);
    debug_assert_eq!(qac_qm_y.len(), n_blocks);
    debug_assert_eq!(qac_qm_b.len(), n_blocks);
    debug_assert_eq!(x_factor.len(), n_blocks);
    debug_assert_eq!(b_factor.len(), n_blocks);

    // Replicate per-block weight tables for every block (the GPU dequant
    // kernel takes per-coefficient weights matching the quantized layout).
    let mut weights_x = Vec::with_capacity(n_blocks * 64);
    let mut weights_y = Vec::with_capacity(n_blocks * 64);
    let mut weights_b = Vec::with_capacity(n_blocks * 64);
    for _ in 0..n_blocks {
        weights_x.extend_from_slice(weights_x_per_block);
        weights_y.extend_from_slice(weights_y_per_block);
        weights_b.extend_from_slice(weights_b_per_block);
    }

    // Step 1: GPU dequant (one launch, three channels).
    let (mut dq_x, mut dq_y, mut dq_b) = dequant_dct8_blocks_gpu(
        enc, quant_ac_x, quant_ac_y, quant_ac_b, &weights_x, &weights_y, &weights_b,
        qac_qm_x, qac_qm_y, qac_qm_b, x_factor, b_factor,
    );

    // Step 2: host DC override (overwrites position [b * 64] of each plane).
    restore_dct8_dc_override_batched(
        &mut dq_x,
        &mut dq_y,
        &mut dq_b,
        quant_dc_x,
        quant_dc_y,
        quant_dc_b,
        scale_dc,
    );

    // Step 3: per-channel IDCT 8x8 (three launches).
    let pix_x = apply_idct_batch_gpu(enc, &dq_x, RAW_STRATEGY_DCT);
    let pix_y = apply_idct_batch_gpu(enc, &dq_y, RAW_STRATEGY_DCT);
    let pix_b = apply_idct_batch_gpu(enc, &dq_b, RAW_STRATEGY_DCT);

    // Step 4: scatter block-major pixels into padded planes.
    let padded_w = xsize_blocks * 8;
    let padded_h = ysize_blocks * 8;
    let n_pix = padded_w * padded_h;
    let mut plane_x = vec![0.0_f32; n_pix];
    let mut plane_y = vec![0.0_f32; n_pix];
    let mut plane_b = vec![0.0_f32; n_pix];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let b = by * xsize_blocks + bx;
            let src = b * 64;
            let dst_y0 = by * 8;
            let dst_x0 = bx * 8;
            for row in 0..8 {
                let s = src + row * 8;
                let d = (dst_y0 + row) * padded_w + dst_x0;
                plane_x[d..d + 8].copy_from_slice(&pix_x[s..s + 8]);
                plane_y[d..d + 8].copy_from_slice(&pix_y[s..s + 8]);
                plane_b[d..d + 8].copy_from_slice(&pix_b[s..s + 8]);
            }
        }
    }

    [plane_x, plane_y, plane_b]
}

/// Decoder-side gab smoothing weights from libjxl epf.cc / loop_filter.h.
/// Duplicated bit-for-bit from upstream `gab_smooth`.
fn gab_weights() -> (f32, f32, f32) {
    let w1_base = 0.104_699_57_f32 * 1.1;
    let w2_base = 0.055_680_54_f32 * 1.1;
    let div = 1.0 + 4.0 * (w1_base + w2_base);
    let w_center = 1.0 / div;
    let w1 = w1_base / div;
    let w2 = w2_base / div;
    (w_center, w1, w2)
}

/// GPU `gab_smooth`. Mirrors upstream
/// `jxl_encoder::vardct::reconstruct::gab_smooth`.
///
/// Three sequential GPU launches over the X/Y/B planes (planes order
/// matches upstream: `planes[0]=X`, `planes[1]=Y`, `planes[2]=B`). Each
/// channel is mutated in place via copy-from-Vec on the GPU return.
pub fn gab_smooth_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    planes: &mut [Vec<f32>; 3],
    width: usize,
    height: usize,
) {
    let (w_center, w1, w2) = gab_weights();
    for plane in planes.iter_mut() {
        assert_eq!(plane.len(), width * height);
        let out = enc.gab_smooth_channel(plane, width as u32, height as u32, w_center, w1, w2);
        plane.copy_from_slice(&out);
    }
}

/// GPU `xyb_to_linear_rgb_planar`. Mirrors upstream signature exactly.
#[allow(clippy::too_many_arguments)]
pub fn xyb_to_linear_rgb_planar_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    out_r: &mut [f32],
    out_g: &mut [f32],
    out_b: &mut [f32],
    num_pixels: usize,
) {
    assert_eq!(xyb_x.len(), num_pixels);
    assert_eq!(xyb_y.len(), num_pixels);
    assert_eq!(xyb_b.len(), num_pixels);
    assert_eq!(out_r.len(), num_pixels);
    assert_eq!(out_g.len(), num_pixels);
    assert_eq!(out_b.len(), num_pixels);
    let (r, g, b) = enc.xyb_to_linear_rgb_planar(xyb_x, xyb_y, xyb_b);
    out_r.copy_from_slice(&r);
    out_g.copy_from_slice(&g);
    out_b.copy_from_slice(&b);
}

/// GPU `xyb_to_linear_rgb` (interleaved). Mirrors upstream return shape:
/// a `Vec<f32>` of length `num_pixels * 3` with `[R, G, B, R, G, B, ...]`.
pub fn xyb_to_linear_rgb_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let num_pixels = width * height;
    assert_eq!(xyb_x.len(), num_pixels);
    let (r, g, b) = enc.xyb_to_linear_rgb_planar(xyb_x, xyb_y, xyb_b);
    let mut interleaved = vec![0.0_f32; num_pixels * 3];
    for i in 0..num_pixels {
        interleaved[i * 3] = r[i];
        interleaved[i * 3 + 1] = g[i];
        interleaved[i * 3 + 2] = b[i];
    }
    interleaved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_dc_channel_x_y_no_cfl() {
        // X / Y channels: no CfL contribution. Output = quant_dc / inv_factor.
        // X: inv_factor = 4096 * 0.5 = 2048; 100 / 2048 = 0.04883
        let v = dequant_dc_channel(100.0, 999.0, 0, 0.5);
        assert!((v - (100.0 / 2048.0)).abs() < 1e-6);
        // Y: inv_factor = 512 * 1.0 = 512; 50 / 512 = 0.09766
        let v = dequant_dc_channel(50.0, 999.0, 1, 1.0);
        assert!((v - (50.0 / 512.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dequant_dc_channel_b_includes_y_cfl() {
        // B: dc_cfl_factor = 0.5. inv_factor = 256.
        // (10 + 100 * 0.5) / 256 = 60 / 256 = 0.234375
        let v = dequant_dc_channel(10.0, 100.0, 2, 1.0);
        assert!((v - 0.234_375).abs() < 1e-6);
    }

    #[test]
    fn test_restore_llf_dct16x8_roundtrip() {
        // Forward dc_from_dct_16x8 followed by inverse should be identity.
        let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
        let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];
        // Pick arbitrary llf0/llf1 values, project forward to dc0/dc1, then invert.
        let llf0 = 1.7_f32;
        let llf1 = -0.4_f32;
        let dc0 = llf0 * s0 + llf1 * s1;
        let dc1 = llf0 * s0 - llf1 * s1;
        let [r0, r1] = restore_llf_dct16x8_or_8x16(dc0, dc1);
        assert!((r0 - llf0).abs() < 1e-5, "got {r0} expected {llf0}");
        assert!((r1 - llf1).abs() < 1e-5, "got {r1} expected {llf1}");
    }

    #[test]
    fn test_restore_llf_dct32x16_zero_in() {
        let r = restore_llf_dct32x16([0.0; 8]);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_restore_llf_dct32x16_constant_dc() {
        // Constant DC across 4×2: only LLF[0] should be non-zero.
        // 2-pt DCT [c,c]→[2c,0]; per-row 4 times → block = [[2c,0],[2c,0],...].
        // Transpose 4×2 → 2×4: t = [[2c,2c,2c,2c],[0,0,0,0]].
        // 4-pt DCT row 0: [2c,2c,2c,2c] → [8c, 0, 0, 0]; row 1 → [0,0,0,0].
        // Divide row 0 col 0: 8c / (1 * 1 * 8) = c. Other positions 0.
        let c = 0.5_f32;
        let r = restore_llf_dct32x16([c; 8]);
        assert!((r[0] - c).abs() < 1e-5);
        for i in 1..8 {
            assert!(r[i].abs() < 1e-5, "pos {i}: got {} expected 0", r[i]);
        }
    }

    #[test]
    fn test_restore_llf_dct16x32_zero_in() {
        let r = restore_llf_dct16x32([0.0; 8]);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_restore_llf_dct16x32_constant_dc() {
        let c = 0.5_f32;
        let r = restore_llf_dct16x32([c; 8]);
        assert!((r[0] - c).abs() < 1e-5);
        for i in 1..8 {
            assert!(r[i].abs() < 1e-5, "pos {i}: got {} expected 0", r[i]);
        }
    }

    #[test]
    fn test_restore_llf_dct32x32_zero_in_zero_out() {
        let r = restore_llf_dct32x32([0.0; 16]);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_restore_llf_dct32x32_constant_dc() {
        // dc_grid = constant c. Forward 4-pt DCT of [c,c,c,c] is
        // [4c, 0, 0, 0] (DC tap = sum, others zero by symmetry).
        // Per-row → block = [[4c,0,0,0],[4c,0,0,0],...]. Transpose →
        // [[4c,4c,4c,4c],[0,...],[0,...],[0,...]]. Forward 4-pt DCT
        // of row 0 [4c,4c,4c,4c] → [16c,0,0,0]; rows 1..3 stay zero.
        // After / (scale * 16): out[0] = 16c / (1*1*16) = c, others 0
        // (or scaled by 0).
        let c = 0.5_f32;
        let r = restore_llf_dct32x32([c; 16]);
        assert!((r[0] - c).abs() < 1e-5);
        for i in 1..16 {
            assert!(r[i].abs() < 1e-5, "pos {i}: got {} expected 0", r[i]);
        }
    }

    #[test]
    fn test_restore_llf_dct16x16_zero_dc_yields_zero_llf() {
        // All zeros in → all zeros out.
        let r = restore_llf_dct16x16([0.0; 4]);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_restore_llf_dct16x16_constant_dc_yields_dc_only() {
        // dc_grid = [c, c, c, c] → h00 = 4c, h01=h10=h11=0.
        // llf00 = 4c / (4 * s0^2) = c (since s0 = 1.0)
        let c = 0.5_f32;
        let r = restore_llf_dct16x16([c, c, c, c]);
        assert!((r[0] - c).abs() < 1e-6);
        assert!(r[1].abs() < 1e-6);
        assert!(r[2].abs() < 1e-6);
        assert!(r[3].abs() < 1e-6);
    }

    #[test]
    fn test_restore_dct8_dc_override_y() {
        // Y channel: dq_y[0] = quant_dc_y / inv_factor[1]
        // inv_factor[1] = 512 * scale_dc
        let mut dq_x = [0.0_f32; 64];
        let mut dq_y = [0.0_f32; 64];
        let mut dq_b = [0.0_f32; 64];
        let quant_dc_y = 100.0_f32;
        let scale_dc = 0.5_f32;
        restore_dct8_dc_override(
            &mut dq_x,
            &mut dq_y,
            &mut dq_b,
            0.0,
            quant_dc_y,
            0.0,
            scale_dc,
        );
        // dq_y[0] = 100 / (512 * 0.5) = 100 / 256 = 0.390625
        assert!((dq_y[0] - 0.390_625).abs() < 1e-6);
    }

    #[test]
    fn test_restore_dct8_dc_override_b_includes_y_cfl() {
        // B channel includes 0.5 * Y DC contribution.
        let mut dq_x = [0.0_f32; 64];
        let mut dq_y = [0.0_f32; 64];
        let mut dq_b = [0.0_f32; 64];
        // quant_dc_b=0, quant_dc_y=10, scale_dc=1.0
        // dq_b[0] = (0 + 10 * 0.5) / (256 * 1.0) = 5 / 256 = 0.01953125
        restore_dct8_dc_override(&mut dq_x, &mut dq_y, &mut dq_b, 0.0, 10.0, 0.0, 1.0);
        assert!((dq_b[0] - 0.019_531_25).abs() < 1e-6);
        // dq_x[0] = 0 / 4096 = 0
        assert_eq!(dq_x[0], 0.0);
        // dq_y[0] = 10 / 512 = 0.01953125
        assert!((dq_y[0] - 0.019_531_25).abs() < 1e-6);
    }

    #[test]
    fn test_restore_dct8_dc_override_batched_matches_per_block() {
        // Run both forms on the same inputs; outputs must agree exactly.
        const N: usize = 5;
        let mut dq_x_batch = vec![0.0_f32; N * 64];
        let mut dq_y_batch = vec![0.0_f32; N * 64];
        let mut dq_b_batch = vec![0.0_f32; N * 64];
        // Seed AC slots to ensure we don't touch them.
        for i in 0..N * 64 {
            if !i.is_multiple_of(64) {
                dq_x_batch[i] = (i as f32) * 0.001;
                dq_y_batch[i] = (i as f32) * 0.002;
                dq_b_batch[i] = (i as f32) * 0.003;
            }
        }
        let qx: Vec<f32> = (0..N).map(|b| 1.0 + b as f32 * 2.0).collect();
        let qy: Vec<f32> = (0..N).map(|b| 5.0 + b as f32 * 3.0).collect();
        let qb: Vec<f32> = (0..N).map(|b| -3.0 + b as f32).collect();
        let scale_dc = 0.7_f32;

        // Per-block reference.
        let mut dq_x_ref = dq_x_batch.clone();
        let mut dq_y_ref = dq_y_batch.clone();
        let mut dq_b_ref = dq_b_batch.clone();
        for b in 0..N {
            let block_x: &mut [f32; 64] = (&mut dq_x_ref[b * 64..b * 64 + 64])
                .try_into()
                .unwrap();
            let block_y: &mut [f32; 64] = (&mut dq_y_ref[b * 64..b * 64 + 64])
                .try_into()
                .unwrap();
            let block_b: &mut [f32; 64] = (&mut dq_b_ref[b * 64..b * 64 + 64])
                .try_into()
                .unwrap();
            restore_dct8_dc_override(
                block_x, block_y, block_b, qx[b], qy[b], qb[b], scale_dc,
            );
        }
        // Batched.
        restore_dct8_dc_override_batched(
            &mut dq_x_batch,
            &mut dq_y_batch,
            &mut dq_b_batch,
            &qx,
            &qy,
            &qb,
            scale_dc,
        );
        assert_eq!(dq_x_batch, dq_x_ref);
        assert_eq!(dq_y_batch, dq_y_ref);
        assert_eq!(dq_b_batch, dq_b_ref);
    }

    #[test]
    fn test_restore_dct8_dc_override_does_not_touch_ac() {
        // AC slots [1..64] must stay unchanged.
        let mut dq_x = [0.5_f32; 64];
        let mut dq_y = [0.7_f32; 64];
        let mut dq_b = [0.3_f32; 64];
        restore_dct8_dc_override(&mut dq_x, &mut dq_y, &mut dq_b, 1.0, 2.0, 3.0, 1.0);
        for i in 1..64 {
            assert_eq!(dq_x[i], 0.5);
            assert_eq!(dq_y[i], 0.7);
            assert_eq!(dq_b[i], 0.3);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_reconstruct_xyb_dct8_only_gpu_zero_input() {
        // All-zero quant + zero CfL → output is all zeros (DC=0, AC=0).
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let xb = 2_usize;
        let yb = 2_usize;
        let nb = xb * yb;
        let zeros_dc = vec![0.0_f32; nb];
        let zeros_ac_i = vec![0_i32; nb * 64];
        let weights_one = [1.0_f32; 64];
        let qac_qm = vec![1.0_f32; nb];
        let zero_factor = vec![0.0_f32; nb];

        let planes = reconstruct_xyb_dct8_only_gpu(
            &enc,
            &zeros_dc, &zeros_dc, &zeros_dc,
            &zeros_ac_i, &zeros_ac_i, &zeros_ac_i,
            &weights_one, &weights_one, &weights_one,
            &qac_qm, &qac_qm, &qac_qm,
            &zero_factor, &zero_factor,
            1.0, xb, yb,
        );
        for p in &planes {
            assert_eq!(p.len(), xb * 8 * yb * 8);
            for &v in p {
                assert!(v.abs() < 1e-6, "expected ~0, got {v}");
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_reconstruct_xyb_dct8_only_gpu_constant_dc() {
        // Constant DC across all blocks → constant output per channel.
        // Set quant_dc_y = 100, scale_dc = 1.0 → DC = 100/512 = 0.1953
        // After IDCT (which scales by 1/8 per dim → 1/8 from the DC tap),
        // each pixel = 0.1953 / 8 = 0.02441 (libjxl IDCT normalization).
        // We just check that the output is constant per plane and finite.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let xb = 1_usize;
        let yb = 1_usize;
        let nb = xb * yb;
        let dc_y = vec![100.0_f32; nb];
        let dc_zero = vec![0.0_f32; nb];
        let zeros_ac_i = vec![0_i32; nb * 64];
        let weights_one = [1.0_f32; 64];
        let qac_qm = vec![1.0_f32; nb];
        let zero_factor = vec![0.0_f32; nb];

        let planes = reconstruct_xyb_dct8_only_gpu(
            &enc,
            &dc_zero, &dc_y, &dc_zero,
            &zeros_ac_i, &zeros_ac_i, &zeros_ac_i,
            &weights_one, &weights_one, &weights_one,
            &qac_qm, &qac_qm, &qac_qm,
            &zero_factor, &zero_factor,
            1.0, xb, yb,
        );
        // Y plane should be constant non-zero; X plane zero; B plane non-zero
        // due to DC-CfL: dc_b = (0 + 100*0.5)/256 = 0.1953
        let v0_y = planes[1][0];
        for &v in &planes[1] {
            assert!((v - v0_y).abs() < 1e-5, "Y not constant: {v} vs {v0_y}");
        }
        assert!(v0_y.abs() > 0.0);
        for &v in &planes[0] {
            assert!(v.abs() < 1e-6, "X should be 0, got {v}");
        }
        // B is non-zero (Y CfL contribution).
        let v0_b = planes[2][0];
        assert!(v0_b.abs() > 0.0);
        for &v in &planes[2] {
            assert!((v - v0_b).abs() < 1e-5);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gab_smooth_uniform_gpu() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 16;
        let h = 16;
        // Uniform image stays uniform under symmetric blur.
        let mut planes = [
            vec![0.5_f32; w * h],
            vec![0.3_f32; w * h],
            vec![0.7_f32; w * h],
        ];
        gab_smooth_gpu(&enc, &mut planes, w, h);
        for &v in &planes[0] {
            assert!((v - 0.5).abs() < 1e-4);
        }
        for &v in &planes[1] {
            assert!((v - 0.3).abs() < 1e-4);
        }
        for &v in &planes[2] {
            assert!((v - 0.7).abs() < 1e-4);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_xyb_to_linear_rgb_planar_gpu_finite() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n = 64;
        let xyb_x: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.001).collect();
        let xyb_y: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.005).collect();
        let xyb_b: Vec<f32> = (0..n).map(|i| 0.05 + (i as f32) * 0.003).collect();
        let mut r = vec![0.0_f32; n];
        let mut g = vec![0.0_f32; n];
        let mut b = vec![0.0_f32; n];
        xyb_to_linear_rgb_planar_gpu(&enc, &xyb_x, &xyb_y, &xyb_b, &mut r, &mut g, &mut b, n);
        for i in 0..n {
            assert!(r[i].is_finite(), "r[{i}] not finite");
            assert!(g[i].is_finite(), "g[{i}] not finite");
            assert!(b[i].is_finite(), "b[{i}] not finite");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_xyb_roundtrip_via_gpu() {
        // Forward XYB then inverse XYB on GPU should round-trip linear RGB.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n = 256;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n)
            .map(|i| 0.2 + 0.5 * ((i + 7) as f32 / n as f32))
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| 0.3 + 0.4 * ((i + 13) as f32 / n as f32))
            .collect();
        let (xx, xy, xb) = enc.xyb_from_linear_rgb(&r, &g, &b);
        let (r2, g2, b2) = enc.xyb_to_linear_rgb_planar(&xx, &xy, &xb);
        let mut max_err = 0.0_f32;
        for i in 0..n {
            max_err = max_err.max((r[i] - r2[i]).abs());
            max_err = max_err.max((g[i] - g2[i]).abs());
            max_err = max_err.max((b[i] - b2[i]).abs());
        }
        // XYB roundtrip is not bit-exact (cube-root → cube can drift) but
        // should be well below 1e-3 absolute on normal RGB inputs.
        assert!(
            max_err < 5e-4,
            "XYB roundtrip drift too large: {max_err:.3e}"
        );
    }
}
