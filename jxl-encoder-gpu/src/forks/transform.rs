// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/transform.rs (BSD-3-Clause via libjxl
// + AGPL/commercial), reshaped from per-block dispatch to batched
// per-strategy GPU launches.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted, batched DCT dispatch.
//!
//! ## Reshape vs upstream `Transform::apply_dct`
//!
//! Upstream `apply_dct` operates on ONE block at a time, dispatched per
//! block by `raw_strategy`. CPUs love this — branch prediction is good,
//! ~64 floats per block fits in L1, and the per-block overhead is
//! sub-microsecond.
//!
//! GPUs hate it. Each block is a 64-thread or 256-thread kernel launch
//! (microseconds of overhead), and tiny launches starve the SMs. The
//! GPU shape is the *opposite*: gather all blocks of the same strategy
//! into one contiguous buffer, then run ONE big kernel launch covering
//! all of them.
//!
//! This module provides `apply_dct_batch_gpu`:
//! - Input: a channel plane + a list of block coordinates `(bx, by)`
//!   that all use the same `raw_strategy`.
//! - Output: a contiguous coefficient buffer with `coeff_count_per_strategy`
//!   floats per block, in the same order as the input list.
//!
//! Caller is responsible for the strategy-grouping pre-pass (scan
//! `ac_strategy`, bucket block coords by strategy). That part is cheap
//! and stays on CPU.
//!
//! ## Currently supported strategies
//!
//! All 15 standard JXL strategies the rest of the GPU port uses:
//! DCT8, DCT16x8, DCT8x16, DCT16x16, DCT32x32, DCT4x8, DCT8x4, DCT4x4,
//! DCT32x16, DCT16x32, DCT64x64, DCT64x32, DCT32x64, IDENTITY, DCT2X2.
//! Both forward (`apply_dct_batch_gpu`) and inverse (`apply_idct_batch_gpu`)
//! work for every strategy in this list.
//!
//! AFV0-3 corner DCTs are NOT routed through this dispatcher because
//! their composition (DCT4 + DCT4x4 + AFV4x4 + DC merge) is per-block,
//! not a single uniform GPU launch. Use `forks::afv::afv_transform_batch_gpu`
//! / `inverse_afv_transform_batch_gpu` for those — they batch all four
//! sub-transforms across N blocks of the same `afv_kind` into 3 launches
//! per direction.

use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

/// JXL `BLOCK_DIM` constant (8). Each "block coordinate" `(bx, by)`
/// addresses an 8×8 region; larger transforms cover N×M of those.
const BLOCK_DIM: usize = 8;

/// raw_strategy=0 → DCT 8×8. Mirrors libjxl `kDCT`.
pub const RAW_STRATEGY_DCT: u8 = 0;

/// raw_strategy=1,2 → DCT 16×8, 8×16. Mirrors libjxl `kDCT16X8`, `kDCT8X16`.
pub const RAW_STRATEGY_DCT16X8: u8 = 1;
pub const RAW_STRATEGY_DCT8X16: u8 = 2;
/// raw_strategy=3 → DCT 16×16. Mirrors libjxl `kDCT16X16`.
pub const RAW_STRATEGY_DCT16X16: u8 = 3;
/// raw_strategy=4 → DCT 32×32. Mirrors libjxl `kDCT32X32`.
pub const RAW_STRATEGY_DCT32X32: u8 = 4;
/// raw_strategy=5 → DCT 4×8. Mirrors libjxl `kDCT4X8`.
pub const RAW_STRATEGY_DCT4X8: u8 = 5;
/// raw_strategy=6 → DCT 8×4. Mirrors libjxl `kDCT8X4`.
pub const RAW_STRATEGY_DCT8X4: u8 = 6;
/// raw_strategy=7 → DCT 4×4. Mirrors libjxl `kDCT4X4`.
pub const RAW_STRATEGY_DCT4X4: u8 = 7;
/// raw_strategy=10 → DCT 32×16. Mirrors libjxl `kDCT32X16`.
pub const RAW_STRATEGY_DCT32X16: u8 = 10;
/// raw_strategy=11 → DCT 16×32. Mirrors libjxl `kDCT16X32`.
pub const RAW_STRATEGY_DCT16X32: u8 = 11;
/// raw_strategy=12,13,14 → DCT 64×64, 64×32, 32×64.
pub const RAW_STRATEGY_DCT64X64: u8 = 12;
pub const RAW_STRATEGY_DCT64X32: u8 = 13;
pub const RAW_STRATEGY_DCT32X64: u8 = 14;
/// raw_strategy=15 → IDENTITY (per-sub-block DC + residual on 8×8).
/// Mirrors libjxl `kIdentity` (whose wire code is 8 — we use a local
/// dispatcher code to fit alongside the existing rectangular DCTs).
pub const RAW_STRATEGY_IDENTITY: u8 = 15;
/// raw_strategy=16 → DCT2X2 (hierarchical 2×2 Hadamard at S=8/4/2).
/// Mirrors libjxl `kDCT2X2`.
pub const RAW_STRATEGY_DCT2X2: u8 = 16;

/// Number of coefficient floats produced per block by each strategy.
///
/// ```
/// use jxl_encoder_gpu::forks::transform::*;
///
/// // 64 coeffs (8×8 input, possibly subdivided):
/// // DCT8 + DCT4 family + IDENTITY + DCT2X2.
/// assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT), 64);
/// assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT4X4), 64);
/// assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_IDENTITY), 64);
/// assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT2X2), 64);
/// // 128: DCT16x8 / DCT8x16 (rectangular 16×8).
/// assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT16X8), 128);
/// assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT8X16), 128);
/// // 256: DCT16x16 square.
/// assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT16X16), 256);
/// // 512 / 1024 / 2048 / 4096: DCT32+/DCT64+ family.
/// assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT32X32), 1024);
/// assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT64X64), 4096);
/// ```
pub fn coeff_count_per_strategy(raw_strategy: u8) -> usize {
    match raw_strategy {
        RAW_STRATEGY_DCT
        | RAW_STRATEGY_DCT4X8
        | RAW_STRATEGY_DCT8X4
        | RAW_STRATEGY_DCT4X4
        | RAW_STRATEGY_IDENTITY
        | RAW_STRATEGY_DCT2X2 => 64,
        RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => 128,
        RAW_STRATEGY_DCT16X16 => 256,
        RAW_STRATEGY_DCT32X16 | RAW_STRATEGY_DCT16X32 => 512,
        RAW_STRATEGY_DCT32X32 => 1024,
        RAW_STRATEGY_DCT64X32 | RAW_STRATEGY_DCT32X64 => 2048,
        RAW_STRATEGY_DCT64X64 => 4096,
        _ => panic!(
            "unsupported strategy {raw_strategy} \
             (use forks::afv::afv_transform_batch_gpu for AFV0-3)"
        ),
    }
}

/// Tile dimensions (cols, rows) in PIXELS for each strategy.
///
/// IDENTITY/DCT2X2 extract from an 8×8 region (their internal layout
/// differs but the gather is the same). AFV0-3 also extract from 8×8
/// but use a per-block composition kernel — see `forks::afv` instead.
///
/// Public for use by reconstruct/scatter logic that needs the
/// per-strategy pixel block size.
pub fn tile_dims_pixels(raw_strategy: u8) -> (usize, usize) {
    tile_dims(raw_strategy)
}

fn tile_dims(raw_strategy: u8) -> (usize, usize) {
    match raw_strategy {
        // 8×8 extraction; transform sub-divides internally
        RAW_STRATEGY_DCT
        | RAW_STRATEGY_DCT4X8
        | RAW_STRATEGY_DCT8X4
        | RAW_STRATEGY_DCT4X4
        | RAW_STRATEGY_IDENTITY
        | RAW_STRATEGY_DCT2X2 => (8, 8),
        RAW_STRATEGY_DCT16X8 => (8, 16), // 8 wide × 16 tall
        RAW_STRATEGY_DCT8X16 => (16, 8), // 16 wide × 8 tall
        RAW_STRATEGY_DCT16X16 => (16, 16),
        RAW_STRATEGY_DCT32X16 => (16, 32), // 16 wide × 32 tall
        RAW_STRATEGY_DCT16X32 => (32, 16), // 32 wide × 16 tall
        RAW_STRATEGY_DCT32X32 => (32, 32),
        RAW_STRATEGY_DCT64X32 => (32, 64), // 32 wide × 64 tall
        RAW_STRATEGY_DCT32X64 => (64, 32), // 64 wide × 32 tall
        RAW_STRATEGY_DCT64X64 => (64, 64),
        _ => panic!(
            "unsupported strategy {raw_strategy} \
             (use forks::afv::afv_transform_batch_gpu for AFV0-3)"
        ),
    }
}

/// Batched DCT dispatch on the GPU. All `block_coords` MUST use the
/// same `raw_strategy`.
///
/// Returns a contiguous `Vec<f32>` of length
/// `block_coords.len() * coeff_count_per_strategy(raw_strategy)`,
/// with one block's coefficients packed after the next in the same
/// order as `block_coords`.
pub fn apply_dct_batch_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    channel_data: &[f32],
    stride: usize,
    block_coords: &[(usize, usize)],
    raw_strategy: u8,
) -> Vec<f32> {
    if block_coords.is_empty() {
        return Vec::new();
    }
    let (tile_w, tile_h) = tile_dims(raw_strategy);
    let tile_pixels = tile_w * tile_h;

    // Gather: extract every block's tile into one contiguous buffer.
    let mut batch = Vec::with_capacity(block_coords.len() * tile_pixels);
    for &(bx, by) in block_coords {
        let x0 = bx * BLOCK_DIM;
        let y0 = by * BLOCK_DIM;
        for dy in 0..tile_h {
            let src_off = (y0 + dy) * stride + x0;
            batch.extend_from_slice(&channel_data[src_off..src_off + tile_w]);
        }
    }

    // Dispatch: one GPU launch covers all blocks of this strategy.
    match raw_strategy {
        RAW_STRATEGY_DCT => enc.dct_8x8_blocks(&batch),
        RAW_STRATEGY_DCT4X8 => enc.dct_4x8_full_blocks(&batch),
        RAW_STRATEGY_DCT8X4 => enc.dct_8x4_full_blocks(&batch),
        RAW_STRATEGY_DCT4X4 => enc.dct_4x4_full_blocks(&batch),
        RAW_STRATEGY_DCT16X8 => enc.dct_16x8_blocks(&batch),
        RAW_STRATEGY_DCT8X16 => enc.dct_8x16_blocks(&batch),
        RAW_STRATEGY_DCT16X16 => enc.dct_16x16_blocks(&batch),
        RAW_STRATEGY_DCT32X16 => enc.dct_32x16_blocks(&batch),
        RAW_STRATEGY_DCT16X32 => enc.dct_16x32_blocks(&batch),
        RAW_STRATEGY_DCT32X32 => enc.dct_32x32_blocks(&batch),
        RAW_STRATEGY_DCT64X32 => enc.dct_64x32_blocks(&batch),
        RAW_STRATEGY_DCT32X64 => enc.dct_32x64_blocks(&batch),
        RAW_STRATEGY_DCT64X64 => enc.dct_64x64_blocks(&batch),
        RAW_STRATEGY_IDENTITY => enc.identity_blocks(&batch),
        RAW_STRATEGY_DCT2X2 => enc.dct2x2_blocks(&batch),
        _ => unreachable!(),
    }
}

/// Batched IDCT dispatch on the GPU. All `coeff_blocks` MUST come from
/// the same `raw_strategy` (one contiguous buffer of
/// `coeff_count_per_strategy(raw_strategy)` floats per block).
///
/// Returns a contiguous `Vec<f32>` of length `block_count * tile_pixels`
/// (with `tile_pixels` per block, in row-major order).
///
/// Mirror of `apply_dct_batch_gpu` for the inverse direction. Useful
/// for reconstruction (decoder-style IDCT) and pixel-domain loss
/// estimation.
pub fn apply_idct_batch_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    coeff_blocks: &[f32],
    raw_strategy: u8,
) -> Vec<f32> {
    let coeffs_per_block = coeff_count_per_strategy(raw_strategy);
    if coeff_blocks.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(
        coeff_blocks.len() % coeffs_per_block,
        0,
        "coeff_blocks length {} not a multiple of coeffs_per_block {} for strategy {}",
        coeff_blocks.len(),
        coeffs_per_block,
        raw_strategy
    );
    match raw_strategy {
        RAW_STRATEGY_DCT => enc.idct_8x8_blocks(coeff_blocks),
        RAW_STRATEGY_DCT4X8 => enc.idct_4x8_full_blocks(coeff_blocks),
        RAW_STRATEGY_DCT8X4 => enc.idct_8x4_full_blocks(coeff_blocks),
        RAW_STRATEGY_DCT4X4 => enc.idct_4x4_full_blocks(coeff_blocks),
        RAW_STRATEGY_DCT16X8 => enc.idct_16x8_blocks(coeff_blocks),
        RAW_STRATEGY_DCT8X16 => enc.idct_8x16_blocks(coeff_blocks),
        RAW_STRATEGY_DCT16X16 => enc.idct_16x16_blocks(coeff_blocks),
        RAW_STRATEGY_DCT32X16 => enc.idct_32x16_blocks(coeff_blocks),
        RAW_STRATEGY_DCT16X32 => enc.idct_16x32_blocks(coeff_blocks),
        RAW_STRATEGY_DCT32X32 => enc.idct_32x32_blocks(coeff_blocks),
        RAW_STRATEGY_DCT64X32 => enc.idct_64x32_blocks(coeff_blocks),
        RAW_STRATEGY_DCT32X64 => enc.idct_32x64_blocks(coeff_blocks),
        RAW_STRATEGY_DCT64X64 => enc.idct_64x64_blocks(coeff_blocks),
        RAW_STRATEGY_IDENTITY => enc.inverse_identity_blocks(coeff_blocks),
        RAW_STRATEGY_DCT2X2 => enc.inverse_dct2x2_blocks(coeff_blocks),
        _ => panic!(
            "unsupported strategy {raw_strategy} \
             (use forks::afv::inverse_afv_transform_batch_gpu for AFV0-3)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_apply_dct_batch_dct8_matches_per_block() {
        // Build a synthetic channel plane (64x64 = 8x8 blocks of size 8x8).
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let stride = 64;
        let height = 64;
        let n_pixels = stride * height;
        let plane: Vec<f32> = (0..n_pixels).map(|i| (i as f32 * 0.013).sin()).collect();

        // All 64 blocks use DCT8.
        let mut block_coords = Vec::new();
        for by in 0..8 {
            for bx in 0..8 {
                block_coords.push((bx, by));
            }
        }

        let batched = apply_dct_batch_gpu(&enc, &plane, stride, &block_coords, RAW_STRATEGY_DCT);
        assert_eq!(batched.len(), 64 * 64); // 64 blocks × 64 coeffs

        // Spot-check: extract one block manually, run DCT8 standalone, compare.
        let mut single_block = vec![0.0_f32; 64];
        let (bx, by) = (3, 5);
        for dy in 0..8 {
            let src_off = (by * 8 + dy) * stride + bx * 8;
            single_block[dy * 8..dy * 8 + 8].copy_from_slice(&plane[src_off..src_off + 8]);
        }
        let single_dct = enc.dct_8x8_blocks(&single_block);
        // Find the corresponding block in the batched output.
        let lin_idx = by * 8 + bx;
        let batched_block = &batched[lin_idx * 64..(lin_idx + 1) * 64];
        let mut max_err = 0.0_f32;
        for i in 0..64 {
            max_err = max_err.max((single_dct[i] - batched_block[i]).abs());
        }
        assert!(
            max_err < 1e-5,
            "batched DCT8 differs from single-block DCT8: max|Δ|={max_err:.3e}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_apply_dct_batch_dct16x16() {
        // 32x32 plane = 4 blocks of 16x16 (each spanning 2x2 = 4 BLOCK_DIM units).
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let stride = 32;
        let height = 32;
        let plane: Vec<f32> = (0..stride * height)
            .map(|i| 0.5 + 0.3 * ((i as f32 * 0.07).cos()))
            .collect();

        // (0,0), (2,0), (0,2), (2,2) — each 16x16 starts at a 2-block offset.
        let block_coords = vec![(0, 0), (2, 0), (0, 2), (2, 2)];
        let batched =
            apply_dct_batch_gpu(&enc, &plane, stride, &block_coords, RAW_STRATEGY_DCT16X16);
        assert_eq!(batched.len(), 4 * 256);
        assert!(batched.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_coeff_count_per_strategy() {
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT), 64);
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT4X8), 64);
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT8X4), 64);
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT4X4), 64);
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT16X8), 128);
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT8X16), 128);
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT16X16), 256);
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT32X16), 512);
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT16X32), 512);
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT32X32), 1024);
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT64X32), 2048);
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT32X64), 2048);
        assert_eq!(coeff_count_per_strategy(RAW_STRATEGY_DCT64X64), 4096);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_apply_dct_batch_dct32x32_finite() {
        // 64x64 plane = 4 blocks of 32x32 (each spanning 4×4 = 16 BLOCK_DIM units).
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let stride = 64;
        let height = 64;
        let plane: Vec<f32> = (0..stride * height)
            .map(|i| 0.4 + 0.4 * ((i as f32 * 0.011).cos()))
            .collect();
        let block_coords = vec![(0, 0), (4, 0), (0, 4), (4, 4)];
        let batched =
            apply_dct_batch_gpu(&enc, &plane, stride, &block_coords, RAW_STRATEGY_DCT32X32);
        assert_eq!(batched.len(), 4 * 1024);
        assert!(batched.iter().all(|v| v.is_finite()));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_dct_idct_batch_roundtrip_dct8() {
        // Round-trip a batch of DCT8 blocks through forward + inverse,
        // verify reconstruction matches input within float precision.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let stride = 64;
        let height = 64;
        let plane: Vec<f32> = (0..stride * height)
            .map(|i| (i as f32 * 0.017).sin())
            .collect();
        let mut block_coords = Vec::new();
        for by in 0..8 {
            for bx in 0..8 {
                block_coords.push((bx, by));
            }
        }
        let coeffs = apply_dct_batch_gpu(&enc, &plane, stride, &block_coords, RAW_STRATEGY_DCT);
        let recon = apply_idct_batch_gpu(&enc, &coeffs, RAW_STRATEGY_DCT);
        assert_eq!(recon.len(), 64 * 64);
        // Compare block-by-block: extract the original block's 64 floats,
        // compare to recon[lin_idx*64..]
        let mut max_err = 0.0_f32;
        for (lin_idx, &(bx, by)) in block_coords.iter().enumerate() {
            let mut orig = [0.0_f32; 64];
            for dy in 0..8 {
                let src_off = (by * 8 + dy) * stride + bx * 8;
                orig[dy * 8..dy * 8 + 8].copy_from_slice(&plane[src_off..src_off + 8]);
            }
            for i in 0..64 {
                max_err = max_err.max((orig[i] - recon[lin_idx * 64 + i]).abs());
            }
        }
        // DCT/IDCT roundtrip with our scale convention should be exact to
        // ~1e-5 absolute on inputs in [-1, 1].
        assert!(
            max_err < 5e-5,
            "DCT8 batch roundtrip drift too large: {max_err:.3e}"
        );
    }

    /// Verifies that the dispatcher routes IDENTITY through both
    /// forward + inverse paths cleanly. Same shape as
    /// `test_dct_idct_batch_roundtrip_dct8`.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_dct_idct_batch_roundtrip_identity() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let stride = 64;
        let plane: Vec<f32> = (0..stride * 64)
            .map(|i| 0.3 + 0.4 * (i as f32 * 0.011).sin())
            .collect();
        let mut block_coords = Vec::new();
        for by in 0..8 {
            for bx in 0..8 {
                block_coords.push((bx, by));
            }
        }
        let coeffs =
            apply_dct_batch_gpu(&enc, &plane, stride, &block_coords, RAW_STRATEGY_IDENTITY);
        let recon = apply_idct_batch_gpu(&enc, &coeffs, RAW_STRATEGY_IDENTITY);
        assert_eq!(recon.len(), 64 * 64);
        let mut max_err = 0.0_f32;
        for (lin_idx, &(bx, by)) in block_coords.iter().enumerate() {
            for dy in 0..8 {
                let src_off = (by * 8 + dy) * stride + bx * 8;
                for dx in 0..8 {
                    let orig = plane[src_off + dx];
                    let r = recon[lin_idx * 64 + dy * 8 + dx];
                    max_err = max_err.max((orig - r).abs());
                }
            }
        }
        assert!(max_err < 1e-5, "IDENTITY batch roundtrip drift: {max_err:.3e}");
    }

    /// DCT2X2 batch dispatcher round-trip.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_dct_idct_batch_roundtrip_dct2x2() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let stride = 64;
        let plane: Vec<f32> = (0..stride * 64)
            .map(|i| 0.3 + 0.4 * (i as f32 * 0.013).cos())
            .collect();
        let mut block_coords = Vec::new();
        for by in 0..8 {
            for bx in 0..8 {
                block_coords.push((bx, by));
            }
        }
        let coeffs =
            apply_dct_batch_gpu(&enc, &plane, stride, &block_coords, RAW_STRATEGY_DCT2X2);
        let recon = apply_idct_batch_gpu(&enc, &coeffs, RAW_STRATEGY_DCT2X2);
        assert_eq!(recon.len(), 64 * 64);
        let mut max_err = 0.0_f32;
        for (lin_idx, &(bx, by)) in block_coords.iter().enumerate() {
            for dy in 0..8 {
                let src_off = (by * 8 + dy) * stride + bx * 8;
                for dx in 0..8 {
                    let orig = plane[src_off + dx];
                    let r = recon[lin_idx * 64 + dy * 8 + dx];
                    max_err = max_err.max((orig - r).abs());
                }
            }
        }
        assert!(max_err < 1e-5, "DCT2X2 batch roundtrip drift: {max_err:.3e}");
    }
}
