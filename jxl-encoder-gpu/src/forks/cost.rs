// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/ac_strategy.rs (BSD-3-Clause via libjxl
// + AGPL/commercial), with the cost-evaluation primitives substituted
// for batched GPU launches.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted cost-evaluation primitives for AC strategy search.
//!
//! Upstream's `estimate_entropy_full` is the inner-loop function that
//! gives a "cost" for one (block-coordinate, strategy) pair. It does:
//!
//! 1. Forward DCT (per strategy)
//! 2. Quantize coefficients
//! 3. Estimate entropy of the quantized coefficients (info_loss term +
//!    nzeros term + per-coefficient cost delta term)
//! 4. Optionally: IDCT the quantized coefficients to reconstruct
//!    pixel-domain values, take a masked 8th-power norm of the
//!    pixel-error against the original pixels (the "pixel-domain loss"
//!    term)
//! 5. Apply per-strategy and per-channel multipliers + offsets
//!
//! Steps 1, 2, 4 already have batched GPU primitives via
//! [`forks::transform`], [`forks::quantize`], and the GPU's pixel-loss
//! kernel. Step 3 (entropy estimation) is the one with no batched
//! upstream API — `estimate_entropy_full` does it inline per-block.
//!
//! This module exposes the leaf cost primitives that the AC strategy
//! search would compose:
//!
//! - [`entropy_coeffs_pixel_blocks_gpu`] — batched per-block entropy
//!   estimation in the pixel-domain. Returns
//!   `(per_block_4_stats, per_coefficient_error)`.
//! - [`block_l2_errors_gpu`] — per-8×8-block masked weighted L2 error
//!   between original and reconstructed XYB. Useful as a quick proxy
//!   cost (cheaper than full pixel-loss).
//! - [`pixel_loss_blocks_gpu`] — per-block 8th-power norm of masked
//!   pixel errors, in f64 for numerical stability. The "real"
//!   pixel-domain loss libjxl uses for its cost model.
//!
//! ## Reshape vs upstream
//!
//! Upstream calls these inside `estimate_entropy_full` per (block,
//! strategy) — N×K calls for N blocks × K strategies. On GPU, batch
//! one strategy at a time: gather all N blocks, run one launch each
//! for forward-DCT + quantize + entropy + pixel-loss. Total GPU
//! launches drops from N×K×4 to K×4 — a ~num_blocks reduction in
//! kernel-launch overhead.
//!
//! ## Status
//!
//! The composing AC-strategy-search loop on GPU is `pipeline.rs` in
//! this crate (`compute_cost_grid_dct8` etc.). This `forks::cost`
//! module exposes the leaf primitives in a stable shape so callers
//! can build their own cost models on top of them.

use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

/// Per-block entropy estimation in the pixel-domain — wraps
/// [`GpuEncoder::entropy_coeffs_pixel_blocks`].
///
/// - `block_c`, `block_y`: per-block coefficient arrays for the
///   target chroma channel (X or B) and the luma channel respectively,
///   each of length `num_blocks * n_per_block`.
/// - `weights`, `inv_weights`: per-coefficient quant matrix entries.
/// - `cmap_factor`: CfL multiplier for the channel.
/// - `quant`: per-block scale (qac × qm_mul, single value broadcast
///   across all blocks).
/// - `k_cost_delta`: distance-dependent per-coefficient cost weight.
///
/// Returns `(per_block_4_stats, per_coefficient_error)`:
/// - `per_block_4_stats`: `num_blocks * 4` floats — for each block,
///   `[entropy_sum, nzeros_sum, info_loss_sum=0, info_loss2_sum=0]`.
/// - `per_coefficient_error`: `num_blocks * n_per_block` floats —
///   `weights[i] * (val - quantized_val)` per coefficient.
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_pixel_blocks_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    block_c: &[f32],
    block_y: &[f32],
    weights: &[f32],
    inv_weights: &[f32],
    n_per_block: u32,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
) -> (Vec<f32>, Vec<f32>) {
    enc.entropy_coeffs_pixel_blocks(
        block_c,
        block_y,
        weights,
        inv_weights,
        n_per_block,
        cmap_factor,
        quant,
        k_cost_delta,
    )
}

/// Per-8×8-block masked weighted L2 error for 3-channel original vs
/// reconstructed XYB. Wraps [`GpuEncoder::block_l2_errors`].
///
/// Useful as a quick proxy cost — cheaper than full pixel-loss but
/// still mask-aware.
#[allow(clippy::too_many_arguments)]
pub fn block_l2_errors_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    orig_x: &[f32],
    orig_y: &[f32],
    orig_b: &[f32],
    recon_x: &[f32],
    recon_y: &[f32],
    recon_b: &[f32],
    mask: &[f32],
    xsize_blocks: u32,
    ysize_blocks: u32,
    padded_width: u32,
) -> Vec<f32> {
    enc.block_l2_errors(
        orig_x,
        orig_y,
        orig_b,
        recon_x,
        recon_y,
        recon_b,
        mask,
        xsize_blocks,
        ysize_blocks,
        padded_width,
    )
}

/// Per-block 8th-power norm of masked pixel errors — wraps
/// [`GpuEncoder::pixel_loss_blocks`]. Returns f64 for numerical
/// stability.
///
/// The 8th-power norm is libjxl's "pixel-domain loss" used in
/// `estimate_entropy_full` when `mask1x1` is provided. f64 accumulation
/// avoids precision loss summing 64+ values each at the 8th power
/// (typical pixel errors are O(1e-2), so 8th-power values are O(1e-16)
/// — close to f32 precision floor).
#[allow(clippy::too_many_arguments)]
pub fn pixel_loss_blocks_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    pixel_error: &[f32],
    mask: &[f32],
    mask_row_base: &[u32],
    mask_stride: u32,
    mask_offset: f32,
    block_width: u32,
    block_height: u32,
) -> Vec<f64> {
    enc.pixel_loss_blocks(
        pixel_error,
        mask,
        mask_row_base,
        mask_stride,
        mask_offset,
        block_width,
        block_height,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_block_l2_zero_error_gpu() {
        // orig == recon → zero L2 error per block.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let xb = 2_u32;
        let yb = 2_u32;
        let pw = (xb * 8) as usize;
        let ph = (yb * 8) as usize;
        let n = pw * ph;
        let plane: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
        let mask = vec![1.0_f32; (xb * yb) as usize];
        let l2 = block_l2_errors_gpu(
            &enc, &plane, &plane, &plane, &plane, &plane, &plane, &mask, xb, yb, pw as u32,
        );
        assert_eq!(l2.len(), (xb * yb) as usize);
        for (i, &v) in l2.iter().enumerate() {
            assert_eq!(v, 0.0, "block {i} L2 error should be 0, got {v}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_block_l2_nonzero_error_gpu() {
        // recon = orig + small offset → positive per-block error.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let xb = 1_u32;
        let yb = 1_u32;
        let pw = 8;
        let ph = 8;
        let n = pw * ph;
        let orig = vec![0.5_f32; n];
        let recon = vec![0.6_f32; n]; // offset by 0.1
        let mask = vec![1.0_f32; (xb * yb) as usize];
        let l2 = block_l2_errors_gpu(
            &enc, &orig, &orig, &orig, &recon, &recon, &recon, &mask, xb, yb, pw as u32,
        );
        assert_eq!(l2.len(), 1);
        assert!(l2[0] > 0.0, "L2 should be positive on nonzero error");
        assert!(l2[0].is_finite());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_entropy_coeffs_pixel_zero_input_gpu() {
        // All-zero input → entropy stats should be deterministic.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 4;
        let n = 64;
        let block_c = vec![0.0_f32; nb * n];
        let block_y = vec![0.0_f32; nb * n];
        let weights = vec![1.0_f32; nb * n];
        let inv_weights = vec![1.0_f32; nb * n];
        let (stats, err) = entropy_coeffs_pixel_blocks_gpu(
            &enc,
            &block_c,
            &block_y,
            &weights,
            &inv_weights,
            n as u32,
            0.0,
            1.0,
            1.0,
        );
        assert_eq!(stats.len(), nb * 4);
        assert_eq!(err.len(), nb * n);
        for &v in &stats {
            assert!(v.is_finite());
        }
        for &v in &err {
            assert!(v.is_finite());
        }
    }
}
