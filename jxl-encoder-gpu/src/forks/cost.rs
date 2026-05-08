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
//! [`crate::forks::transform`], [`crate::forks::quantize`], and the GPU's pixel-loss
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

/// Extract per-block entropy values from the
/// `entropy_coeffs_pixel_blocks_gpu` 4-stat output.
///
/// The kernel returns `num_blocks * 4` floats per channel, where
/// `[block * 4 + 0]` is the per-block entropy sum (the other three
/// slots are nzeros, info_loss, info_loss2 — used by other code
/// paths that consume them but not always needed for the cost model).
///
/// Returns one `f32` per block — a thin reshape of the kernel
/// output for the cost-model orchestrator.
pub fn extract_per_block_entropy(stats_4x: &[f32], n_blocks: usize) -> Vec<f32> {
    debug_assert_eq!(stats_4x.len(), n_blocks * 4);
    (0..n_blocks).map(|b| stats_4x[b * 4]).collect()
}

/// Sum per-block entropy across 3 channels. Mirrors the
/// X+Y+B accumulation in upstream's `estimate_entropy_full`'s
/// process_channel inner loop.
///
/// All three input slices must be `n_blocks` floats long
/// (typically extracted via [`extract_per_block_entropy`] from the
/// 3 per-channel stat arrays).
pub fn sum_per_block_entropy_3channel(
    entropy_x: &[f32],
    entropy_y: &[f32],
    entropy_b: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(entropy_x.len(), entropy_y.len());
    debug_assert_eq!(entropy_x.len(), entropy_b.len());
    entropy_x
        .iter()
        .zip(entropy_y.iter())
        .zip(entropy_b.iter())
        .map(|((&x, &y), &b)| x + y + b)
        .collect()
}

/// Per-block total cost combiner — the final per-block scalar that
/// upstream's `estimate_entropy_full` returns. Mirrors the formula
/// `entropy_mul * total_entropy + total_pixel_loss` per block.
///
/// `entropy_total` is the sum of per-channel entropy values across
/// X/Y/B (output of [`sum_per_block_entropy_3channel`]).
/// `pixel_loss_total` is the sum of per-channel pixel-domain losses
/// scaled by [`CHANNEL_MUL`] (output of
/// [`combine_pixel_loss_3channel`]). `entropy_mul` is the per-strategy
/// multiplier from [`entropy_mul_for_strategy`].
///
/// Returns one `f32` cost per block — the final per-block cost the
/// AC strategy search would compare across candidates.
pub fn per_block_total_cost(
    entropy_total: &[f32],
    pixel_loss_total: &[f64],
    entropy_mul: f32,
) -> Vec<f32> {
    debug_assert_eq!(entropy_total.len(), pixel_loss_total.len());
    entropy_total
        .iter()
        .zip(pixel_loss_total.iter())
        .map(|(&e, &p)| entropy_mul * e + p as f32)
        .collect()
}

/// Combine per-block per-channel pixel-domain losses (output of
/// [`pixel_loss_blocks_gpu`] called once per channel) into a single
/// per-block total via the [`CHANNEL_MUL`] weights.
///
/// `losses_x[i] * CHANNEL_MUL[0] + losses_y[i] * CHANNEL_MUL[1] +
/// losses_b[i] * CHANNEL_MUL[2]` per block. Matches upstream's
/// process_channel inner loop in `estimate_entropy_full`'s DCT8 fast
/// path:
/// ```text
/// channel_loss *= CHANNEL_MUL[c];
/// total_pixel_loss += channel_loss;
/// ```
///
/// All three input slices must have the same length.
pub fn combine_pixel_loss_3channel(
    losses_x: &[f64],
    losses_y: &[f64],
    losses_b: &[f64],
) -> Vec<f64> {
    debug_assert_eq!(losses_x.len(), losses_y.len());
    debug_assert_eq!(losses_x.len(), losses_b.len());
    losses_x
        .iter()
        .zip(losses_y.iter())
        .zip(losses_b.iter())
        .map(|((&x, &y), &b)| x * CHANNEL_MUL[0] + y * CHANNEL_MUL[1] + b * CHANNEL_MUL[2])
        .collect()
}

/// Per-strategy entropy multipliers consumed by
/// `estimate_entropy_full`. Bit-for-bit port of upstream
/// `jxl_encoder::effort::EntropyMulTable` (which itself mirrors the
/// libjxl `enc_ac_strategy.cc:584` reference table).
///
/// The 8x8-class transforms (DCT8 / DCT4x4 / DCT4x8 / DCT8x4 /
/// IDENTITY / DCT2x2 / AFV0-3) are normalized by `dct8` in
/// upstream's FindBest8x8Transform — so DCT8 effectively contributes
/// 1.0. Larger transforms (DCT16+, DCT32+, DCT64+) use raw values
/// per upstream's TryMergeAcs.
///
/// Construct via [`EntropyMulTable::reference`] for the libjxl
/// defaults or [`EntropyMulTable::experimental`] for libjxl PR #4506
/// (Jon Sneyers' VarDCT cost tuning that lowers dct4x4 / identity /
/// afv).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EntropyMulTable {
    pub dct8: f32,
    pub dct4x4: f32,
    pub dct4x8: f32,
    pub identity: f32,
    pub dct2x2: f32,
    pub afv: f32,
    pub dct16x8: f32,
    pub dct16x16: f32,
    pub dct16x32: f32,
    pub dct32x32: f32,
    pub dct64x32: f32,
    pub dct64x64: f32,
}

impl EntropyMulTable {
    /// libjxl reference values (`enc_ac_strategy.cc:584`).
    pub fn reference() -> Self {
        Self {
            dct8: 0.8,
            dct4x4: 1.08,
            dct4x8: 0.859_316_37,
            identity: 1.0428,
            dct2x2: 0.95,
            afv: 0.817_794_9,
            dct16x8: 1.21,
            dct16x16: 1.34,
            dct16x32: 1.49,
            dct32x32: 1.48,
            dct64x32: 2.25,
            dct64x64: 2.25,
        }
    }

    /// libjxl PR #4506 (Jon Sneyers) experimental tuning. Lowers
    /// dct4x4 / identity / afv to favor those strategies for
    /// detail / flat / edge blocks respectively.
    pub fn experimental() -> Self {
        Self {
            dct4x4: 0.88,
            identity: 0.88,
            afv: 0.75,
            ..Self::reference()
        }
    }
}

/// Per-strategy entropy multiplier lookup. Mirrors upstream
/// `jxl_encoder::vardct::ac_strategy::entropy_mul_for_strategy` —
/// 8x8-class transforms get `table.X / table.dct8` (so DCT8 = 1.0
/// after the FindBest8x8Transform normalization), larger transforms
/// use raw values.
///
/// AFV0-3 strategies all share the same `table.afv` slot.
///
/// Returns 1.0 for any unknown / out-of-range strategy code (matches
/// upstream's defensive `_ => 1.0` fallback).
pub fn entropy_mul_for_strategy(raw_strategy: u8, table: &EntropyMulTable) -> f32 {
    use crate::forks::transform::{
        RAW_STRATEGY_DCT, RAW_STRATEGY_DCT16X16, RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT16X8,
        RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64,
        RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8, RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64,
        RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT8X4, RAW_STRATEGY_IDENTITY,
    };
    match raw_strategy {
        RAW_STRATEGY_DCT => 1.0,
        RAW_STRATEGY_DCT4X8 | RAW_STRATEGY_DCT8X4 => table.dct4x8 / table.dct8,
        RAW_STRATEGY_DCT4X4 => table.dct4x4 / table.dct8,
        RAW_STRATEGY_IDENTITY => table.identity / table.dct8,
        RAW_STRATEGY_DCT2X2 => table.dct2x2 / table.dct8,
        // AFV0-3 not in our dispatcher's strategy code map; callers
        // route AFV through forks::afv. Provide the value anyway for
        // completeness — accessible via forks::cost::afv_entropy_mul.
        RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => table.dct16x8,
        RAW_STRATEGY_DCT16X16 => table.dct16x16,
        RAW_STRATEGY_DCT32X16 | RAW_STRATEGY_DCT16X32 => table.dct16x32,
        RAW_STRATEGY_DCT32X32 => table.dct32x32,
        RAW_STRATEGY_DCT64X32 | RAW_STRATEGY_DCT32X64 => table.dct64x32,
        RAW_STRATEGY_DCT64X64 => table.dct64x64,
        _ => 1.0,
    }
}

/// AFV0-3 entropy multiplier — exposed separately because AFV
/// strategies are routed through `forks::afv` rather than the main
/// strategy dispatcher. Matches upstream's
/// `entropy_mul_for_strategy(AFV0..3, table)` arm:
/// `table.afv / table.dct8`.
#[inline]
pub fn afv_entropy_mul(table: &EntropyMulTable) -> f32 {
    table.afv / table.dct8
}

/// Per-channel offsets for pixel-domain loss masking. Bit-for-bit
/// from upstream `jxl_encoder::vardct::ac_strategy::MASK_CHANNEL_OFFSET`
/// (= libjxl `enc_ac_strategy.cc:446`).
///
/// Indexed by channel: X=0, Y=1, B=2. The Y channel has no offset
/// (luma is already perceptually well-scaled); X and B add a constant
/// before the 8th-power norm to dampen ultra-low-magnitude chroma
/// errors.
pub const MASK_CHANNEL_OFFSET: [f32; 3] = [12.0, 0.0, 4.0];

/// Per-channel multipliers for the pixel-domain 8th-power loss.
/// Bit-for-bit from upstream
/// `jxl_encoder::vardct::ac_strategy::CHANNEL_MUL` (= libjxl
/// `enc_ac_strategy.cc:479`). These are pre-computed `base^8` values
/// (X corresponds to base ≈ 8.222, Y to 1.0, B to 1.03 — the
/// upstream "8.2^8" comment is approximate).
///
/// Indexed by channel (X=0, Y=1, B=2). `f64` to keep precision
/// during the 8th-power accumulation that consumes them.
pub const CHANNEL_MUL: [f64; 3] = [
    20_882_706.465_593_6, // X — upstream comment "8.2^8" is approximate
    1.0,                  // Y = 1.0^8
    1.266_770_080_64,     // B = 1.03^8
];

/// Constants for coefficient-domain entropy estimation (libjxl-tiny
/// style, NOT distance-scaled). Bit-for-bit from upstream
/// `jxl_encoder::vardct::ac_strategy::COEFF_DOMAIN_CONSTANTS`.
///
/// Tuple order matches [`compute_scaled_constants`] output:
/// `(info_loss_mul, cost_delta, zeros_mul)`.
///
/// Use these as the `scaled_constants` argument to entropy
/// estimation when `mask1x1 = None` (i.e., the cheaper coefficient-
/// domain path that doesn't need an IDCT-back-to-pixels round-trip).
pub const COEFF_DOMAIN_CONSTANTS: (f32, f32, f32) = (138.0, 5.335_918_5, 7.565_053_4);

// Pixel-domain constants — ratio shape and exponents per upstream
// `jxl_encoder::vardct::ac_strategy::compute_scaled_constants`.
const K_BIAS: f32 = 0.137_317_43;
const K_POW_INFO_LOSS: f32 = 0.336_778_07;
const K_POW_ZEROS_MUL: f32 = 0.509_909_3;
const K_POW_COST_DELTA: f32 = 0.367_029_4;

/// Distance-scaled constants for the pixel-domain entropy / cost
/// model. Bit-for-bit port of upstream
/// `jxl_encoder::vardct::ac_strategy::compute_scaled_constants`.
///
/// `bases = (info_loss_base, zeros_base, cost_delta_base)` — the
/// per-encoder `EffortProfile` base values. **Argument tuple order
/// differs from output tuple order**, matching upstream:
/// - Input: `(info_loss, zeros, cost_delta)`
/// - Output: `(info_loss, cost_delta, zeros)` — same as
///   [`COEFF_DOMAIN_CONSTANTS`] layout.
///
/// At distance == 1.0 returns the bases unchanged (with the position
/// shuffle); for higher distances scales them up via
/// `((distance + 0.137) / 1.137).powf(K_POW_*)`.
///
/// Call this ONCE per AC-strategy search (not per
/// (block, strategy) pair) — the result is constant within a search.
/// For coefficient-domain mode (no pixel-loss IDCT round-trip), use
/// the precomputed [`COEFF_DOMAIN_CONSTANTS`] directly without
/// calling this function.
///
/// ```
/// use jxl_encoder_gpu::forks::cost::compute_scaled_constants;
///
/// // At d=1.0 the ratio is exactly 1.0 → no scaling.
/// // Bases are in (info_loss, zeros, cost_delta) order;
/// // output is (info_loss, cost_delta, zeros).
/// let (info, cost, zeros) =
///     compute_scaled_constants(1.0, (1.0, 2.0, 3.0));
/// assert!((info - 1.0).abs() < 1e-6);
/// assert!((cost - 3.0).abs() < 1e-6);
/// assert!((zeros - 2.0).abs() < 1e-6);
/// ```
pub fn compute_scaled_constants(
    distance: f32,
    bases: (f32, f32, f32),
) -> (f32, f32, f32) {
    let (info_loss_base, zeros_base, cost_delta_base) = bases;
    let ratio = (distance + K_BIAS) / (1.0 + K_BIAS);
    let info_loss_mul = info_loss_base * ratio.powf(K_POW_INFO_LOSS);
    let zeros_mul = zeros_base * ratio.powf(K_POW_ZEROS_MUL);
    let cost_delta = cost_delta_base * ratio.powf(K_POW_COST_DELTA);
    (info_loss_mul, cost_delta, zeros_mul)
}

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

    #[test]
    fn test_extract_per_block_entropy_takes_column_zero() {
        // 3 blocks × 4 stats = 12 floats. Column 0 is the entropy.
        let stats = vec![
            10.0_f32, 1.0, 2.0, 3.0, // block 0: entropy=10
            20.0_f32, 4.0, 5.0, 6.0, // block 1: entropy=20
            30.0_f32, 7.0, 8.0, 9.0, // block 2: entropy=30
        ];
        let e = extract_per_block_entropy(&stats, 3);
        assert_eq!(e, vec![10.0_f32, 20.0, 30.0]);
    }

    #[test]
    fn test_sum_per_block_entropy_3channel_adds() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let y = vec![10.0_f32, 20.0, 30.0];
        let b = vec![100.0_f32, 200.0, 300.0];
        let total = sum_per_block_entropy_3channel(&x, &y, &b);
        assert_eq!(total, vec![111.0_f32, 222.0, 333.0]);
    }

    #[test]
    fn test_per_block_total_cost_formula() {
        // total_cost[b] = entropy_mul * entropy_total[b] + pixel_loss_total[b]
        let entropy = vec![1.0_f32, 2.0, 3.0];
        let pixel_loss = vec![10.0_f64, 20.0, 30.0];
        let cost = per_block_total_cost(&entropy, &pixel_loss, 0.5);
        // Block 0: 0.5 * 1.0 + 10.0 = 10.5
        assert!((cost[0] - 10.5).abs() < 1e-6);
        // Block 1: 0.5 * 2.0 + 20.0 = 21.0
        assert!((cost[1] - 21.0).abs() < 1e-6);
        // Block 2: 0.5 * 3.0 + 30.0 = 31.5
        assert!((cost[2] - 31.5).abs() < 1e-6);
    }

    #[test]
    fn test_per_block_total_cost_zero_loss_only_entropy_term() {
        let entropy = vec![1.0_f32, 2.0, 3.0];
        let zero_loss = vec![0.0_f64; 3];
        let cost = per_block_total_cost(&entropy, &zero_loss, 1.5);
        for (i, &c) in cost.iter().enumerate() {
            assert!((c - 1.5 * entropy[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_combine_pixel_loss_3channel_zero_in() {
        let n = 5_usize;
        let zeros = alloc::vec![0.0_f64; n];
        let out = combine_pixel_loss_3channel(&zeros, &zeros, &zeros);
        assert_eq!(out.len(), n);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_combine_pixel_loss_3channel_per_channel_weighting() {
        // X-only input → out = X * CHANNEL_MUL[0]
        let x = alloc::vec![1.0_f64, 2.0, 3.0];
        let zero = alloc::vec![0.0_f64; 3];
        let out = combine_pixel_loss_3channel(&x, &zero, &zero);
        for i in 0..3 {
            assert!((out[i] - x[i] * CHANNEL_MUL[0]).abs() < 1e-3);
        }

        // Y-only input → out = Y * 1.0 (= Y)
        let y = alloc::vec![1.5_f64, 2.5, 3.5];
        let out = combine_pixel_loss_3channel(&zero, &y, &zero);
        for i in 0..3 {
            assert!((out[i] - y[i]).abs() < 1e-9);
        }

        // B-only → out = B * CHANNEL_MUL[2] (≈ 1.267)
        let b = alloc::vec![10.0_f64, 20.0, 30.0];
        let out = combine_pixel_loss_3channel(&zero, &zero, &b);
        for i in 0..3 {
            assert!((out[i] - b[i] * CHANNEL_MUL[2]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_entropy_mul_table_reference_matches_upstream() {
        // G5.1 parity: every field of our EntropyMulTable::reference()
        // must match jxl_encoder::effort::EntropyMulTable::reference()
        // exactly. Catches any drift if upstream tunes the table.
        let mine = EntropyMulTable::reference();
        let theirs = jxl_encoder::effort::EntropyMulTable::reference();
        assert_eq!(mine.dct8, theirs.dct8);
        assert_eq!(mine.dct4x4, theirs.dct4x4);
        assert_eq!(mine.dct4x8, theirs.dct4x8);
        assert_eq!(mine.identity, theirs.identity);
        assert_eq!(mine.dct2x2, theirs.dct2x2);
        assert_eq!(mine.afv, theirs.afv);
        assert_eq!(mine.dct16x8, theirs.dct16x8);
        assert_eq!(mine.dct16x16, theirs.dct16x16);
        assert_eq!(mine.dct16x32, theirs.dct16x32);
        assert_eq!(mine.dct32x32, theirs.dct32x32);
        assert_eq!(mine.dct64x32, theirs.dct64x32);
        assert_eq!(mine.dct64x64, theirs.dct64x64);
    }

    #[test]
    fn test_entropy_mul_table_experimental_matches_upstream() {
        let mine = EntropyMulTable::experimental();
        let theirs = jxl_encoder::effort::EntropyMulTable::experimental();
        assert_eq!(mine.dct8, theirs.dct8);
        assert_eq!(mine.dct4x4, theirs.dct4x4);
        assert_eq!(mine.dct4x8, theirs.dct4x8);
        assert_eq!(mine.identity, theirs.identity);
        assert_eq!(mine.dct2x2, theirs.dct2x2);
        assert_eq!(mine.afv, theirs.afv);
        assert_eq!(mine.dct16x8, theirs.dct16x8);
        assert_eq!(mine.dct16x16, theirs.dct16x16);
        assert_eq!(mine.dct16x32, theirs.dct16x32);
        assert_eq!(mine.dct32x32, theirs.dct32x32);
        assert_eq!(mine.dct64x32, theirs.dct64x32);
        assert_eq!(mine.dct64x64, theirs.dct64x64);
    }

    #[test]
    fn test_entropy_mul_table_reference() {
        let t = EntropyMulTable::reference();
        assert_eq!(t.dct8, 0.8);
        assert_eq!(t.dct4x4, 1.08);
        assert_eq!(t.identity, 1.0428);
        assert_eq!(t.dct2x2, 0.95);
        assert_eq!(t.dct16x8, 1.21);
        assert_eq!(t.dct16x16, 1.34);
        assert_eq!(t.dct32x32, 1.48);
        assert_eq!(t.dct64x64, 2.25);
    }

    #[test]
    fn test_entropy_mul_table_experimental_overrides() {
        let r = EntropyMulTable::reference();
        let e = EntropyMulTable::experimental();
        // Three fields differ.
        assert_eq!(e.dct4x4, 0.88);
        assert_eq!(e.identity, 0.88);
        assert_eq!(e.afv, 0.75);
        // Everything else inherits from reference.
        assert_eq!(e.dct8, r.dct8);
        assert_eq!(e.dct4x8, r.dct4x8);
        assert_eq!(e.dct2x2, r.dct2x2);
        assert_eq!(e.dct16x16, r.dct16x16);
    }

    #[test]
    fn test_entropy_mul_for_strategy_dct8_is_one() {
        use crate::forks::transform::RAW_STRATEGY_DCT;
        let t = EntropyMulTable::reference();
        assert_eq!(entropy_mul_for_strategy(RAW_STRATEGY_DCT, &t), 1.0);
    }

    #[test]
    fn test_entropy_mul_for_strategy_normalized_8x8_class() {
        use crate::forks::transform::{
            RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8,
            RAW_STRATEGY_DCT8X4, RAW_STRATEGY_IDENTITY,
        };
        let t = EntropyMulTable::reference();
        // 8x8-class transforms get table.X / table.dct8 (= 0.8 in reference).
        assert!((entropy_mul_for_strategy(RAW_STRATEGY_DCT4X4, &t) - 1.08 / 0.8).abs() < 1e-6);
        assert!(
            (entropy_mul_for_strategy(RAW_STRATEGY_DCT4X8, &t) - 0.859_316_37 / 0.8).abs() < 1e-6
        );
        assert!(
            (entropy_mul_for_strategy(RAW_STRATEGY_DCT8X4, &t) - 0.859_316_37 / 0.8).abs() < 1e-6
        );
        assert!((entropy_mul_for_strategy(RAW_STRATEGY_IDENTITY, &t) - 1.0428 / 0.8).abs() < 1e-6);
        assert!((entropy_mul_for_strategy(RAW_STRATEGY_DCT2X2, &t) - 0.95 / 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_entropy_mul_for_strategy_raw_for_large() {
        use crate::forks::transform::{
            RAW_STRATEGY_DCT16X16, RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT16X8,
            RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64,
            RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64, RAW_STRATEGY_DCT8X16,
        };
        let t = EntropyMulTable::reference();
        // Larger transforms use raw values per upstream TryMergeAcs.
        assert_eq!(entropy_mul_for_strategy(RAW_STRATEGY_DCT16X8, &t), 1.21);
        assert_eq!(entropy_mul_for_strategy(RAW_STRATEGY_DCT8X16, &t), 1.21);
        assert_eq!(entropy_mul_for_strategy(RAW_STRATEGY_DCT16X16, &t), 1.34);
        assert_eq!(entropy_mul_for_strategy(RAW_STRATEGY_DCT32X16, &t), 1.49);
        assert_eq!(entropy_mul_for_strategy(RAW_STRATEGY_DCT16X32, &t), 1.49);
        assert_eq!(entropy_mul_for_strategy(RAW_STRATEGY_DCT32X32, &t), 1.48);
        assert_eq!(entropy_mul_for_strategy(RAW_STRATEGY_DCT64X32, &t), 2.25);
        assert_eq!(entropy_mul_for_strategy(RAW_STRATEGY_DCT32X64, &t), 2.25);
        assert_eq!(entropy_mul_for_strategy(RAW_STRATEGY_DCT64X64, &t), 2.25);
    }

    #[test]
    fn test_entropy_mul_for_strategy_unknown_returns_one() {
        let t = EntropyMulTable::reference();
        // 17, 18 etc. fall through the match's default arm.
        assert_eq!(entropy_mul_for_strategy(99, &t), 1.0);
    }

    #[test]
    fn test_afv_entropy_mul() {
        let t = EntropyMulTable::reference();
        assert!((afv_entropy_mul(&t) - 0.817_794_9 / 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_channel_constants_match_upstream() {
        // Spot-check constants against libjxl enc_ac_strategy.cc reference.
        // (The MUL values are the literal upstream constants — the X-channel
        // comment "8.2^8" in libjxl is misleading; the actual value
        // corresponds to a base ≈ 8.222. We preserve the literal.)
        assert_eq!(MASK_CHANNEL_OFFSET, [12.0_f32, 0.0, 4.0]);
        assert_eq!(CHANNEL_MUL[0], 20_882_706.465_593_6);
        assert_eq!(CHANNEL_MUL[1], 1.0);
        assert_eq!(CHANNEL_MUL[2], 1.266_770_080_64);
    }

    #[test]
    fn test_compute_scaled_constants_d1_no_scale() {
        // ratio = 1.0 at distance == 1.0 → bases echo back.
        let (info, cost, zeros) =
            compute_scaled_constants(1.0, (1.0, 2.0, 3.0));
        assert!((info - 1.0).abs() < 1e-6);
        assert!((cost - 3.0).abs() < 1e-6);
        assert!((zeros - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_scaled_constants_d3_scales_up() {
        // distance 3.0 → ratio = (3.0 + 0.137) / 1.137 ≈ 2.760.
        // Each base scaled by ratio^K_POW_*; all > base.
        let bases = (10.0, 20.0, 30.0);
        let (info, cost, zeros) = compute_scaled_constants(3.0, bases);
        let ratio = (3.0_f32 + K_BIAS) / (1.0 + K_BIAS);
        let expected_info = 10.0 * ratio.powf(K_POW_INFO_LOSS);
        let expected_zeros = 20.0 * ratio.powf(K_POW_ZEROS_MUL);
        let expected_cost = 30.0 * ratio.powf(K_POW_COST_DELTA);
        assert!((info - expected_info).abs() < 1e-3);
        assert!((cost - expected_cost).abs() < 1e-3);
        assert!((zeros - expected_zeros).abs() < 1e-3);
        // All scale up at d=3.0.
        assert!(info > 10.0);
        assert!(zeros > 20.0);
        assert!(cost > 30.0);
    }

    #[test]
    fn test_coeff_domain_constants_match_upstream() {
        // Spot-check the const matches upstream values.
        assert_eq!(COEFF_DOMAIN_CONSTANTS.0, 138.0);
        assert!((COEFF_DOMAIN_CONSTANTS.1 - 5.335_918_5).abs() < 1e-6);
        assert!((COEFF_DOMAIN_CONSTANTS.2 - 7.565_053_4).abs() < 1e-6);
    }

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
