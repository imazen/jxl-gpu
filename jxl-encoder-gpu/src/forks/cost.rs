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
//! ## Layout: leaves + combiners + orchestrators
//!
//! **Leaf GPU primitives** (one launch each, batched across blocks):
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
//! **Constants + scalars** (host-side, bit-for-bit ports of upstream):
//!
//! - [`COEFF_DOMAIN_CONSTANTS`] / [`compute_scaled_constants`] — the
//!   `(info_loss_mul, cost_delta, zeros_mul)` triple used by the cost
//!   formula; coefficient-domain (no distance scaling) vs pixel-domain
//!   (distance-scaled) variants.
//! - [`MASK_CHANNEL_OFFSET`] / [`CHANNEL_MUL`] — per-channel additive
//!   offset and 8th-power multiplier for pixel-loss masking.
//! - [`EntropyMulTable`] / [`entropy_mul_for_strategy`] /
//!   [`afv_entropy_mul`] — per-strategy entropy multipliers
//!   (`reference()` matches libjxl, `experimental()` matches PR #4506).
//!
//! **Per-block host combiners** (chain after the leaf GPU calls):
//!
//! - [`extract_per_block_entropy`] — column-0 (entropy_sum) of the
//!   4-stat array.
//! - [`sum_per_block_entropy_3channel`] — X+Y+B element-wise sum.
//! - [`combine_pixel_loss_3channel`] — CHANNEL_MUL-weighted X+Y+B sum.
//! - [`nzeros_bits_term`] / [`x_multiblock_weight`] /
//!   [`apply_x_multiblock_weight_to_loss`] /
//!   [`apply_x_multiblock_weight_to_entropy`] — upstream cost formula
//!   sub-terms.
//! - [`per_block_total_cost`] — simple combiner
//!   `entropy_mul * sum(entropies) + total_loss`.
//! - [`per_block_upstream_cost`] — full upstream-faithful combiner
//!   with `k_zeros_mul * f(nzeros)` per channel + 8th-root pixel-loss
//!   scaling. Generic over `block_pixel_count`.
//!
//! **Orchestrators** (compose all of the above into per-block cost):
//!
//! - [`estimate_entropy_full_dct8_batch_gpu`] — DCT8 fast path,
//!   ~12 GPU launches per call.
//! - [`estimate_entropy_full_strategy_batch_gpu`] — strategy-generic
//!   (works for DCT8/16/32/64 family + IDENTITY/DCT2X2/DCT4-family).
//!   AFV0-3 routed through `forks::afv` separately.
//! - [`CostMode`] — enum selecting Simple vs Upstream cost formula.
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
//! All leaves + combiners + orchestrators are G5.1-compliant —
//! every helper validated against upstream's `pub` symbols (or
//! against the `__internals` cargo feature for the previously-
//! private ones). 16 `*_matches_upstream` tests + 9 LLF
//! restoration helpers transitively validated. See PORT_STATUS.md
//! "Validation status" section for the full table.
//!
//! The legacy GPU-resident pipeline.rs cost grid functions
//! (`compute_cost_grid_dct8` etc.) use the simpler `block_l2`-only
//! cost; they pre-date this module and remain for the existing AC
//! strategy search code that consumes `CostGrid` (Handle-typed).

use alloc::vec::Vec;

use cubecl::Runtime;
use cubecl::prelude::*;

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

/// `ceil_log2_nonzero` — bit-for-bit port of upstream
/// `jxl_encoder::vardct::ac_strategy::ceil_log2_nonzero` used inside
/// `estimate_entropy_full`'s per-channel `f(nzeros)` term. Matches
/// `(usize::BITS - x.leading_zeros())` for `x > 0`. For `x == 0` the
/// upstream macro returns 0 (we mirror that).
#[inline]
fn ceil_log2_nonzero(x: u32) -> u32 {
    if x == 0 {
        0
    } else {
        u32::BITS - x.leading_zeros()
    }
}

/// Per-channel "nzeros bits" cost contribution from upstream's
/// `estimate_entropy_full` DCT8 fast path. Mirrors the lines:
///
/// ```text
///   let num_nzeros = coeff_result.nzeros_sum as usize;
///   let nbits = ceil_log2_nonzero(num_nzeros + 1) as usize + 1;
///   entropy += k_zeros_mul * (ceil_log2_nonzero(nbits + 17) + nbits as u32) as f32;
/// ```
///
/// Returns the per-channel entropy contribution to add for the
/// nzeros-cost term.
#[inline]
pub fn nzeros_bits_term(num_nzeros: u32, k_zeros_mul: f32) -> f32 {
    let nbits = ceil_log2_nonzero(num_nzeros + 1) + 1;
    k_zeros_mul * (ceil_log2_nonzero(nbits + 17) + nbits) as f32
}

/// Full upstream-shape per-block cost combiner — implements
/// `estimate_entropy_full`'s final cost formula. Generic over
/// `block_pixel_count` (= 64 for DCT8, 128 for DCT16x8/8x16, 256
/// for DCT16x16, 1024 for DCT32x32, etc.), so the same combiner
/// works for any rectangular DCT strategy.
///
/// ```text
///   per_channel:
///     entropy += entropy_sum
///     entropy += k_zeros_mul * f(nzeros_sum)
///   final:
///     n = block_pixel_count
///     loss_scalar = (total_pixel_loss / n).sqrt().sqrt().sqrt() * n / quant
///     entropy *= entropy_mul
///     entropy += k_info_loss_mul * loss_scalar
///     return entropy
/// ```
///
/// Mirrors upstream:
/// - DCT8 fast path (`vardct/ac_strategy.rs:737-771`): `n = 64`,
///   `quant = quant_for_coeffs` (the per-block `quant_field` entry).
/// - Generic path (`vardct/ac_strategy.rs:1099-1108`):
///   `n = num_blocks * 64` (where `num_blocks = cx * cy`),
///   `quant = quant_norm16` (the L2/L16-normalized quant used by
///   the generic path). Caller is responsible for computing
///   `quant_norm16` and passing it as `quant_for_coeffs`.
///
/// **Caveat for non-DCT8 strategies**: upstream's generic path also
/// applies an X-channel-only weighting `w = 1 + min(num_blocks/8, 3)`
/// when `c == 0 && num_blocks >= 2 && use_pixel_domain`
/// (`ac_strategy.rs:1047-1048`). That weighting is NOT applied here;
/// callers that need full upstream parity for X channel on
/// multi-block strategies must apply it before calling.
///
/// `scaled_constants` is the `(info_loss_mul, cost_delta, zeros_mul)`
/// tuple from [`compute_scaled_constants`] (or
/// [`COEFF_DOMAIN_CONSTANTS`]).
#[allow(clippy::too_many_arguments)]
pub fn per_block_upstream_cost(
    entropy_x: &[f32],
    entropy_y: &[f32],
    entropy_b: &[f32],
    nzeros_x: &[f32],
    nzeros_y: &[f32],
    nzeros_b: &[f32],
    pixel_loss_total: &[f64],
    entropy_mul: f32,
    scaled_constants: (f32, f32, f32),
    quant_for_coeffs: f32,
    block_pixel_count: usize,
) -> Vec<f32> {
    let n_blocks = entropy_x.len();
    debug_assert_eq!(entropy_y.len(), n_blocks);
    debug_assert_eq!(entropy_b.len(), n_blocks);
    debug_assert_eq!(nzeros_x.len(), n_blocks);
    debug_assert_eq!(nzeros_y.len(), n_blocks);
    debug_assert_eq!(nzeros_b.len(), n_blocks);
    debug_assert_eq!(pixel_loss_total.len(), n_blocks);
    debug_assert!(block_pixel_count > 0);

    let (k_info_loss_mul, _cost_delta, k_zeros_mul) = scaled_constants;
    let n_pix = block_pixel_count as f64;
    let inv_q = 1.0 / quant_for_coeffs as f64;
    let covered_blocks = block_pixel_count / 64;
    let x_w = x_multiblock_weight(covered_blocks);

    let mut out = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        // Per-channel entropy + nzeros bits cost.
        // X gets the multiblock weight applied to its (entropy_sum +
        // nzeros_bits_term) sum, matching upstream's
        // `if c == 0 && num_blocks >= 2: entropy *= w`. For DCT8
        // (covered_blocks=1) x_w = 1.0 and this is a no-op.
        let x_part = (entropy_x[b] + nzeros_bits_term(nzeros_x[b] as u32, k_zeros_mul)) * x_w;
        let y_part = entropy_y[b] + nzeros_bits_term(nzeros_y[b] as u32, k_zeros_mul);
        let b_part = entropy_b[b] + nzeros_bits_term(nzeros_b[b] as u32, k_zeros_mul);
        let mut entropy = x_part + y_part + b_part;

        // Combined pixel-loss → 8th-root scalar.
        let p = pixel_loss_total[b];
        let loss_scalar = (p / n_pix).sqrt().sqrt().sqrt() * n_pix * inv_q;

        entropy *= entropy_mul;
        entropy += k_info_loss_mul * loss_scalar as f32;
        out.push(entropy);
    }
    out
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

/// X-channel multi-block weight from upstream's generic-path
/// `estimate_entropy_full` (`vardct/ac_strategy.rs:1047, 1087`).
///
/// `w = 1 + min(num_blocks / 8, 3)` for `num_blocks >= 2`, else `1.0`.
///
/// Upstream applies this to (a) the X channel's `entropy`
/// contribution and (b) the X channel's `total_pixel_loss`
/// contribution — matching the DCT8 fast path's NO weight (DCT8 has
/// `num_blocks = 1` so `w = 1.0` is a no-op there).
///
/// Use [`apply_x_multiblock_weight_to_loss`] and
/// [`apply_x_multiblock_weight_to_entropy`] to apply to per-channel
/// arrays.
#[inline]
pub fn x_multiblock_weight(num_blocks: usize) -> f32 {
    if num_blocks >= 2 {
        1.0 + (num_blocks as f32 / 8.0).min(3.0)
    } else {
        1.0
    }
}

/// Apply [`x_multiblock_weight`] to the X channel's per-block pixel
/// loss in-place. No-op when `num_blocks < 2`.
///
/// Caller can call this between [`pixel_loss_blocks_gpu`] (for X)
/// and [`combine_pixel_loss_3channel`] to match upstream's
/// generic-path behavior.
pub fn apply_x_multiblock_weight_to_loss(loss_x: &mut [f64], num_blocks: usize) {
    let w = x_multiblock_weight(num_blocks) as f64;
    if w == 1.0 {
        return;
    }
    for v in loss_x.iter_mut() {
        *v *= w;
    }
}

/// Apply [`x_multiblock_weight`] to the X channel's per-block
/// entropy in-place. No-op when `num_blocks < 2`. Mirrors
/// upstream's `if c == 0 && num_blocks >= 2 && use_pixel_domain {
/// entropy *= w; }` — effectively weights ONLY X's contribution
/// (since upstream's `entropy` accumulator hasn't received Y or B
/// yet at the moment X is processed in the generic-path channel
/// order 0/1/2).
pub fn apply_x_multiblock_weight_to_entropy(entropy_x: &mut [f32], num_blocks: usize) {
    let w = x_multiblock_weight(num_blocks);
    if w == 1.0 {
        return;
    }
    for v in entropy_x.iter_mut() {
        *v *= w;
    }
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
        RAW_STRATEGY_DCT, RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8,
        RAW_STRATEGY_DCT8X4, RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16,
        RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64,
        RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64, RAW_STRATEGY_IDENTITY,
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
pub fn compute_scaled_constants(distance: f32, bases: (f32, f32, f32)) -> (f32, f32, f32) {
    let (info_loss_base, zeros_base, cost_delta_base) = bases;
    let ratio = (distance + K_BIAS) / (1.0 + K_BIAS);
    let info_loss_mul = info_loss_base * ratio.powf(K_POW_INFO_LOSS);
    let zeros_mul = zeros_base * ratio.powf(K_POW_ZEROS_MUL);
    let cost_delta = cost_delta_base * ratio.powf(K_POW_COST_DELTA);
    (info_loss_mul, cost_delta, zeros_mul)
}

/// Cost-formula selector for [`estimate_entropy_full_dct8_batch_gpu`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CostMode {
    /// Simpler formula `entropy_mul * sum(channel_entropies) +
    /// total_pixel_loss` — fast and sufficient for ranking
    /// candidates relative to each other on small benches. Wraps
    /// [`per_block_total_cost`].
    Simple,
    /// Upstream-faithful formula matching `estimate_entropy_full`'s
    /// DCT8 fast path exactly. Includes the per-channel
    /// `k_zeros_mul * f(nzeros)` bits-cost term and the 8th-root
    /// `loss_scalar = (loss/64)^(1/8) * 64 / quant` scaling on the
    /// pixel-loss component. Wraps [`per_block_upstream_cost`].
    /// Pass the per-block `quant_for_coeffs` value via the
    /// orchestrator's argument.
    Upstream { quant_for_coeffs: f32 },
}

/// Batched per-block cost evaluator for DCT8 — the inner loop of
/// upstream's `estimate_entropy_full` for one strategy.
///
/// Composes the existing leaf primitives:
/// 1. `dct_8x8_blocks` × 3 (forward DCT for X/Y/B)
/// 2. `entropy_coeffs_pixel_blocks_gpu` × 3 (per-channel entropy +
///    error-coefficient writeback). Y first (no CfL), then X with
///    `cmap_factor = ytox_ratio(ytox)`, then B with `cmap_factor =
///    ytob_ratio(ytob)`.
/// 3. `idct_8x8_blocks` × 3 (IDCT of error coefficients to get
///    pixel-domain reconstruction error)
/// 4. `pixel_loss_blocks_gpu` × 3 (per-channel masked 8th-power norm)
/// 5. host: `combine_pixel_loss_3channel` (CHANNEL_MUL-weighted sum)
/// 6. host: `extract_per_block_entropy` × 3 + `sum_per_block_entropy_3channel`
/// 7. host: `per_block_total_cost(entropy_total, pixel_loss_total, entropy_mul)`
///
/// Total: ~12 GPU launches per call regardless of `n_blocks`.
///
/// Returns `Vec<f32>` of length `n_blocks` — the per-block cost for
/// the AC strategy search to compare against other candidates.
///
/// **Caveat**: the underlying `entropy_coeffs_pixel_blocks_gpu`
/// kernel currently only computes the entropy_sum + nzeros_sum
/// columns of the 4-stat output (info_loss_sum + info_loss2_sum
/// stay 0). The cost formula here uses entropy_sum directly without
/// the upstream `info_loss_mul * info_loss + zeros_mul * nzeros`
/// re-weighting. For full upstream parity those terms need to be
/// added — currently a simplification on top of the leaves. The
/// `scaled_constants` argument is included in the signature so the
/// caller documents intent, but `info_loss_mul` and `zeros_mul`
/// are not yet consumed (the field is reserved for the future
/// extension).
///
/// **All weights / inv_weights buffers are flat replicated
/// across `n_blocks`** — caller passes per-block (64-float) tables;
/// this function expands them internally to `n_blocks * 64`. The
/// mask is image-plane: `mask_row_base[b]` is the start offset of
/// block `b` in `mask_image_plane` (typically
/// `by * 8 * padded_width + bx * 8`); `mask_stride` is the
/// padded_width of the mask plane.
#[allow(clippy::too_many_arguments)]
pub fn estimate_entropy_full_dct8_batch_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    pixel_blocks_x: &[f32],
    pixel_blocks_y: &[f32],
    pixel_blocks_b: &[f32],
    weights_x_per_block: &[f32; 64],
    weights_y_per_block: &[f32; 64],
    weights_b_per_block: &[f32; 64],
    inv_weights_x_per_block: &[f32; 64],
    inv_weights_y_per_block: &[f32; 64],
    inv_weights_b_per_block: &[f32; 64],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    mask_image_plane: &[f32],
    mask_row_base: &[u32],
    mask_stride: u32,
    scaled_constants: (f32, f32, f32),
    entropy_mul: f32,
    mode: CostMode,
) -> Vec<f32> {
    use crate::forks::cfl::{ytob_ratio, ytox_ratio};

    debug_assert!(pixel_blocks_x.len().is_multiple_of(64));
    debug_assert_eq!(pixel_blocks_x.len(), pixel_blocks_y.len());
    debug_assert_eq!(pixel_blocks_x.len(), pixel_blocks_b.len());
    let n_blocks = pixel_blocks_x.len() / 64;
    debug_assert_eq!(mask_row_base.len(), n_blocks);
    let (_info_loss_mul, cost_delta, _zeros_mul) = scaled_constants;

    // Step 1: forward DCT8 each channel.
    let dct_x = enc.dct_8x8_blocks(pixel_blocks_x);
    let dct_y = enc.dct_8x8_blocks(pixel_blocks_y);
    let dct_b = enc.dct_8x8_blocks(pixel_blocks_b);

    // Step 2: per-channel entropy + error-coef writeback. Use
    // broadcast-weights variant — saves 6 × n_blocks × 64 × 4 bytes
    // of host replication + GPU upload (24 MB at 1024×1024 → 1.5 KB).
    let weights_x_t: &[f32] = weights_x_per_block.as_slice();
    let weights_y_t: &[f32] = weights_y_per_block.as_slice();
    let weights_b_t: &[f32] = weights_b_per_block.as_slice();
    let inv_x_t: &[f32] = inv_weights_x_per_block.as_slice();
    let inv_y_t: &[f32] = inv_weights_y_per_block.as_slice();
    let inv_b_t: &[f32] = inv_weights_b_per_block.as_slice();

    let (y_stats, y_err) = entropy_coeffs_pixel_blocks_gpu_broadcast_w(
        enc,
        &dct_y,
        &dct_y,
        weights_y_t,
        inv_y_t,
        64,
        0.0,
        quant_y,
        cost_delta,
    );
    let (x_stats, x_err) = entropy_coeffs_pixel_blocks_gpu_broadcast_w(
        enc,
        &dct_x,
        &dct_y,
        weights_x_t,
        inv_x_t,
        64,
        ytox_ratio(ytox),
        quant_x,
        cost_delta,
    );
    let (b_stats, b_err) = entropy_coeffs_pixel_blocks_gpu_broadcast_w(
        enc,
        &dct_b,
        &dct_y,
        weights_b_t,
        inv_b_t,
        64,
        ytob_ratio(ytob),
        quant_b,
        cost_delta,
    );

    // Step 3: IDCT of error coefficients per channel.
    let pix_err_x = enc.idct_8x8_blocks(&x_err);
    let pix_err_y = enc.idct_8x8_blocks(&y_err);
    let pix_err_b = enc.idct_8x8_blocks(&b_err);

    // Step 4: per-channel masked 8th-power pixel loss.
    // mask_offset = MASK_CHANNEL_OFFSET[c]; block_w = block_h = 8 for DCT8.
    let loss_x = pixel_loss_blocks_gpu(
        enc,
        &pix_err_x,
        mask_image_plane,
        mask_row_base,
        mask_stride,
        MASK_CHANNEL_OFFSET[0],
        8,
        8,
    );
    let loss_y = pixel_loss_blocks_gpu(
        enc,
        &pix_err_y,
        mask_image_plane,
        mask_row_base,
        mask_stride,
        MASK_CHANNEL_OFFSET[1],
        8,
        8,
    );
    let loss_b = pixel_loss_blocks_gpu(
        enc,
        &pix_err_b,
        mask_image_plane,
        mask_row_base,
        mask_stride,
        MASK_CHANNEL_OFFSET[2],
        8,
        8,
    );

    // Step 5: combine per-channel losses via CHANNEL_MUL.
    let pixel_loss_total = combine_pixel_loss_3channel(&loss_x, &loss_y, &loss_b);

    // Step 6: extract per-block entropy from each channel.
    let entropy_x = extract_per_block_entropy(&x_stats, n_blocks);
    let entropy_y = extract_per_block_entropy(&y_stats, n_blocks);
    let entropy_b = extract_per_block_entropy(&b_stats, n_blocks);

    // Step 7: final cost — formula selector.
    match mode {
        CostMode::Simple => {
            let entropy_total = sum_per_block_entropy_3channel(&entropy_x, &entropy_y, &entropy_b);
            per_block_total_cost(&entropy_total, &pixel_loss_total, entropy_mul)
        }
        CostMode::Upstream { quant_for_coeffs } => {
            let nzeros_x = (0..n_blocks)
                .map(|b| x_stats[b * 4 + 1])
                .collect::<Vec<_>>();
            let nzeros_y = (0..n_blocks)
                .map(|b| y_stats[b * 4 + 1])
                .collect::<Vec<_>>();
            let nzeros_b = (0..n_blocks)
                .map(|b| b_stats[b * 4 + 1])
                .collect::<Vec<_>>();
            per_block_upstream_cost(
                &entropy_x,
                &entropy_y,
                &entropy_b,
                &nzeros_x,
                &nzeros_y,
                &nzeros_b,
                &pixel_loss_total,
                entropy_mul,
                scaled_constants,
                quant_for_coeffs,
                64, // DCT8: block_pixel_count = 64
            )
        }
    }
}

/// Persistent-API variant of [`estimate_entropy_full_dct8_batch_gpu`]:
/// takes pre-uploaded `GpuBlocks` for the per-channel pixel blocks and
/// a `GpuPlane` for the mask, keeping every intermediate buffer
/// (forward DCT coeffs, error coeffs, IDCT pixel errors) on GPU. Only
/// the final per-block cost grid (`n_blocks * 4` stats + `n_blocks`
/// f64 losses per channel) is downloaded.
///
/// Saves the 8 sync `read_one()` per channel (~10ms each at 1MB) that
/// the non-persistent variant pays — at 1024×1024 with DCT8, expect
/// ~80ms savings per call.
///
/// Returns the same `Vec<f32>` cost grid as
/// [`estimate_entropy_full_dct8_batch_gpu`].
#[allow(clippy::too_many_arguments)]
pub fn estimate_entropy_full_dct8_batch_persistent<R: Runtime>(
    enc: &GpuEncoder<R>,
    pixel_blocks_x: &crate::persistent::GpuBlocks<R>,
    pixel_blocks_y: &crate::persistent::GpuBlocks<R>,
    pixel_blocks_b: &crate::persistent::GpuBlocks<R>,
    weights_x_per_block: &[f32; 64],
    weights_y_per_block: &[f32; 64],
    weights_b_per_block: &[f32; 64],
    inv_weights_x_per_block: &[f32; 64],
    inv_weights_y_per_block: &[f32; 64],
    inv_weights_b_per_block: &[f32; 64],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    mask_plane: &crate::persistent::GpuPlane<R>,
    mask_row_base: &[u32],
    scaled_constants: (f32, f32, f32),
    entropy_mul: f32,
    mode: CostMode,
) -> Vec<f32> {
    use crate::forks::cfl::{ytob_ratio, ytox_ratio};

    let n_blocks = pixel_blocks_x.num_blocks() as usize;
    debug_assert_eq!(pixel_blocks_x.num_blocks(), pixel_blocks_y.num_blocks());
    debug_assert_eq!(pixel_blocks_x.num_blocks(), pixel_blocks_b.num_blocks());
    debug_assert_eq!(pixel_blocks_x.coeffs_per_block(), 64);
    debug_assert_eq!(mask_row_base.len(), n_blocks);
    let (_info_loss_mul, cost_delta, _zeros_mul) = scaled_constants;

    // Step 1: forward DCT8 each channel (no sync).
    let dct_x = enc.dct_8x8_persistent(pixel_blocks_x);
    let dct_y = enc.dct_8x8_persistent(pixel_blocks_y);
    let dct_b = enc.dct_8x8_persistent(pixel_blocks_b);

    let weights_x_t: &[f32] = weights_x_per_block.as_slice();
    let weights_y_t: &[f32] = weights_y_per_block.as_slice();
    let weights_b_t: &[f32] = weights_b_per_block.as_slice();
    let inv_x_t: &[f32] = inv_weights_x_per_block.as_slice();
    let inv_y_t: &[f32] = inv_weights_y_per_block.as_slice();
    let inv_b_t: &[f32] = inv_weights_b_per_block.as_slice();

    // Step 2: per-channel entropy + error-coef writeback (no sync).
    let (g_y_stats, g_y_err) = enc.entropy_coeffs_pixel_blocks_broadcast_w_persistent(
        &dct_y, &dct_y, weights_y_t, inv_y_t, 0.0, quant_y, cost_delta,
    );
    let (g_x_stats, g_x_err) = enc.entropy_coeffs_pixel_blocks_broadcast_w_persistent(
        &dct_x,
        &dct_y,
        weights_x_t,
        inv_x_t,
        ytox_ratio(ytox),
        quant_x,
        cost_delta,
    );
    let (g_b_stats, g_b_err) = enc.entropy_coeffs_pixel_blocks_broadcast_w_persistent(
        &dct_b,
        &dct_y,
        weights_b_t,
        inv_b_t,
        ytob_ratio(ytob),
        quant_b,
        cost_delta,
    );

    // Step 3: IDCT of error coefficients per channel (no sync).
    let g_pix_err_x = enc.idct_8x8_persistent(&g_x_err);
    let g_pix_err_y = enc.idct_8x8_persistent(&g_y_err);
    let g_pix_err_b = enc.idct_8x8_persistent(&g_b_err);

    // Step 4: per-channel masked 8th-power pixel loss (no sync).
    // Upload mask_row_base ONCE and reuse the handle across the 3
    // channel pixel_loss calls — saves 2 cudaMallocs.
    let h_mrb = enc
        .client_ref()
        .create_from_slice(u32::as_bytes(mask_row_base));
    let g_loss_x = enc.pixel_loss_blocks_with_handle_persistent(
        &g_pix_err_x,
        mask_plane,
        &h_mrb,
        MASK_CHANNEL_OFFSET[0],
        8,
        8,
    );
    let g_loss_y = enc.pixel_loss_blocks_with_handle_persistent(
        &g_pix_err_y,
        mask_plane,
        &h_mrb,
        MASK_CHANNEL_OFFSET[1],
        8,
        8,
    );
    let g_loss_b = enc.pixel_loss_blocks_with_handle_persistent(
        &g_pix_err_b,
        mask_plane,
        &h_mrb,
        MASK_CHANNEL_OFFSET[2],
        8,
        8,
    );

    // Step 5: now download stats + losses (only the small final
    // outputs — no intermediate downloads). One batched
    // client.read instead of six sequential read_one syncs —
    // saves 5 queue-drain stalls.
    let ((x_stats, y_stats, b_stats), (loss_x, loss_y, loss_b)) = enc
        .download_3stats_3losses(
            &g_x_stats, &g_y_stats, &g_b_stats,
            &g_loss_x, &g_loss_y, &g_loss_b,
        );

    // Step 6: combine per-channel losses via CHANNEL_MUL (host).
    let pixel_loss_total = combine_pixel_loss_3channel(&loss_x, &loss_y, &loss_b);

    // Step 7: extract per-block entropy from each channel.
    let entropy_x = extract_per_block_entropy(&x_stats, n_blocks);
    let entropy_y = extract_per_block_entropy(&y_stats, n_blocks);
    let entropy_b = extract_per_block_entropy(&b_stats, n_blocks);

    // Step 8: final cost — formula selector.
    match mode {
        CostMode::Simple => {
            let entropy_total = sum_per_block_entropy_3channel(&entropy_x, &entropy_y, &entropy_b);
            per_block_total_cost(&entropy_total, &pixel_loss_total, entropy_mul)
        }
        CostMode::Upstream { quant_for_coeffs } => {
            let nzeros_x = (0..n_blocks).map(|b| x_stats[b * 4 + 1]).collect::<Vec<_>>();
            let nzeros_y = (0..n_blocks).map(|b| y_stats[b * 4 + 1]).collect::<Vec<_>>();
            let nzeros_b = (0..n_blocks).map(|b| b_stats[b * 4 + 1]).collect::<Vec<_>>();
            per_block_upstream_cost(
                &entropy_x,
                &entropy_y,
                &entropy_b,
                &nzeros_x,
                &nzeros_y,
                &nzeros_b,
                &pixel_loss_total,
                entropy_mul,
                scaled_constants,
                quant_for_coeffs,
                64,
            )
        }
    }
}

/// Strategy-generic per-block cost evaluator — same shape as
/// [`estimate_entropy_full_dct8_batch_gpu`] but takes a
/// `raw_strategy` parameter so it works for any DCT family
/// (DCT8, DCT16x8, DCT8x16, DCT16x16, DCT32x32, DCT32x16, DCT16x32,
/// DCT64x64, DCT64x32, DCT32x64, plus IDENTITY/DCT2X2/DCT4-family).
///
/// For AFV0-3 use `forks::afv::*` separately.
///
/// Pipeline (12 GPU launches per call regardless of n_blocks):
/// 1. `dct_blocks_gpu` ×3 (forward DCT for X/Y/B)
/// 2. `entropy_coeffs_pixel_blocks_gpu` ×3 with `n_per_block =
///    coeff_count_per_strategy(raw_strategy)`
/// 3. `apply_idct_batch_gpu` ×3 (per-strategy IDCT)
/// 4. `pixel_loss_blocks_gpu` ×3 with `block_width/block_height =
///    tile_dims_pixels(raw_strategy)`
/// 5. host: `combine_pixel_loss_3channel`
/// 6. host: `extract_per_block_entropy` ×3 + `sum_per_block_entropy_3channel`
/// 7. host: per [`CostMode`] — `per_block_total_cost` (Simple) or
///    `per_block_upstream_cost` with `block_pixel_count = block_w *
///    block_h` (Upstream)
///
/// `pixel_blocks_*.len()` MUST equal `n_blocks * tile_pixels` for
/// the strategy. Caller has already gathered from image-plane
/// (use [`crate::forks::transform::apply_dct_batch_gpu`] if you
/// need the gather-and-DCT-in-one form).
///
/// **Caveat for non-DCT8 strategies**: upstream's generic-path
/// estimate_entropy_full also weights X channel by
/// `1 + min(num_blocks/8, 3)` when `num_blocks >= 2 && c == 0 &&
/// use_pixel_domain` (`vardct/ac_strategy.rs:1047-1048`). That
/// weighting is NOT applied here — callers needing full upstream
/// parity for X on multi-block strategies must apply it before or
/// after the call. For DCT8 (single-block) this is a no-op; for
/// the orchestrator's primary use case (relative-ranking cost),
/// the missing weight just shifts X's weight uniformly across
/// candidates, so the ranking is preserved.
#[allow(clippy::too_many_arguments)]
pub fn estimate_entropy_full_strategy_batch_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    pixel_blocks_x: &[f32],
    pixel_blocks_y: &[f32],
    pixel_blocks_b: &[f32],
    raw_strategy: u8,
    weights_x_per_block: &[f32],
    weights_y_per_block: &[f32],
    weights_b_per_block: &[f32],
    inv_weights_x_per_block: &[f32],
    inv_weights_y_per_block: &[f32],
    inv_weights_b_per_block: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    mask_image_plane: &[f32],
    mask_row_base: &[u32],
    mask_stride: u32,
    scaled_constants: (f32, f32, f32),
    entropy_mul: f32,
    mode: CostMode,
) -> Vec<f32> {
    use crate::forks::cfl::{ytob_ratio, ytox_ratio};
    use crate::forks::transform::{
        apply_idct_batch_gpu, coeff_count_per_strategy, dct_blocks_gpu, tile_dims_pixels,
    };

    let coeff_count = coeff_count_per_strategy(raw_strategy);
    let (block_w, block_h) = tile_dims_pixels(raw_strategy);
    let block_pixels = block_w * block_h;

    debug_assert!(pixel_blocks_x.len().is_multiple_of(block_pixels));
    debug_assert_eq!(pixel_blocks_x.len(), pixel_blocks_y.len());
    debug_assert_eq!(pixel_blocks_x.len(), pixel_blocks_b.len());
    let n_blocks = pixel_blocks_x.len() / block_pixels;
    debug_assert_eq!(mask_row_base.len(), n_blocks);
    debug_assert_eq!(weights_x_per_block.len(), coeff_count);
    debug_assert_eq!(weights_y_per_block.len(), coeff_count);
    debug_assert_eq!(weights_b_per_block.len(), coeff_count);
    debug_assert_eq!(inv_weights_x_per_block.len(), coeff_count);
    debug_assert_eq!(inv_weights_y_per_block.len(), coeff_count);
    debug_assert_eq!(inv_weights_b_per_block.len(), coeff_count);
    let (_info_loss_mul, cost_delta, _zeros_mul) = scaled_constants;

    // Step 1: forward DCT each channel.
    let dct_x = dct_blocks_gpu(enc, pixel_blocks_x, raw_strategy);
    let dct_y = dct_blocks_gpu(enc, pixel_blocks_y, raw_strategy);
    let dct_b = dct_blocks_gpu(enc, pixel_blocks_b, raw_strategy);

    // Step 2: per-channel entropy + error-coef writeback. Use
    // broadcast-weights variant — saves 6 × n_blocks × coeff_count × 4
    // bytes of host replication + GPU upload. At DCT16x16 (256
    // coeffs) over 4096 candidate blocks, that's 6 × 4 MB = 24 MB →
    // 6 KB. At DCT64x64 (4096 coeffs) the savings scale
    // proportionally (96 MB → 96 KB at the same block count).
    let (y_stats, y_err) = entropy_coeffs_pixel_blocks_gpu_broadcast_w(
        enc,
        &dct_y,
        &dct_y,
        weights_y_per_block,
        inv_weights_y_per_block,
        coeff_count as u32,
        0.0,
        quant_y,
        cost_delta,
    );
    let (x_stats, x_err) = entropy_coeffs_pixel_blocks_gpu_broadcast_w(
        enc,
        &dct_x,
        &dct_y,
        weights_x_per_block,
        inv_weights_x_per_block,
        coeff_count as u32,
        ytox_ratio(ytox),
        quant_x,
        cost_delta,
    );
    let (b_stats, b_err) = entropy_coeffs_pixel_blocks_gpu_broadcast_w(
        enc,
        &dct_b,
        &dct_y,
        weights_b_per_block,
        inv_weights_b_per_block,
        coeff_count as u32,
        ytob_ratio(ytob),
        quant_b,
        cost_delta,
    );

    // Step 3: per-strategy IDCT of error coefficients.
    let pix_err_x = apply_idct_batch_gpu(enc, &x_err, raw_strategy);
    let pix_err_y = apply_idct_batch_gpu(enc, &y_err, raw_strategy);
    let pix_err_b = apply_idct_batch_gpu(enc, &b_err, raw_strategy);

    // Step 4: per-channel masked 8th-power pixel loss.
    let mut loss_x = pixel_loss_blocks_gpu(
        enc,
        &pix_err_x,
        mask_image_plane,
        mask_row_base,
        mask_stride,
        MASK_CHANNEL_OFFSET[0],
        block_w as u32,
        block_h as u32,
    );
    let loss_y = pixel_loss_blocks_gpu(
        enc,
        &pix_err_y,
        mask_image_plane,
        mask_row_base,
        mask_stride,
        MASK_CHANNEL_OFFSET[1],
        block_w as u32,
        block_h as u32,
    );
    let loss_b = pixel_loss_blocks_gpu(
        enc,
        &pix_err_b,
        mask_image_plane,
        mask_row_base,
        mask_stride,
        MASK_CHANNEL_OFFSET[2],
        block_w as u32,
        block_h as u32,
    );

    // Step 5a: extract per-block entropy from each channel.
    let entropy_x_raw = extract_per_block_entropy(&x_stats, n_blocks);
    let entropy_y = extract_per_block_entropy(&y_stats, n_blocks);
    let entropy_b = extract_per_block_entropy(&b_stats, n_blocks);

    // Step 5b: apply upstream's X-channel multi-block weight to
    // pixel loss (Upstream mode only; in Simple mode the weight is
    // irrelevant for relative ranking). num_blocks = block_pixels / 64
    // — for DCT8 = 1 (no-op); DCT16x16 = 4; DCT32x32 = 16; DCT64x64
    // = 64 (capped at w=4.0). The X weight on ENTROPY is handled
    // inside per_block_upstream_cost (it needs to multiply the sum
    // of entropy_x + nzeros_bits_term per block).
    let entropy_x = entropy_x_raw;
    if matches!(mode, CostMode::Upstream { .. }) {
        let covered_blocks = block_pixels / 64;
        apply_x_multiblock_weight_to_loss(&mut loss_x, covered_blocks);
    }

    // Step 6: combine per-channel losses (after X weight applied).
    let pixel_loss_total = combine_pixel_loss_3channel(&loss_x, &loss_y, &loss_b);

    // Step 7: final cost.
    match mode {
        CostMode::Simple => {
            let entropy_total = sum_per_block_entropy_3channel(&entropy_x, &entropy_y, &entropy_b);
            per_block_total_cost(&entropy_total, &pixel_loss_total, entropy_mul)
        }
        CostMode::Upstream { quant_for_coeffs } => {
            let nzeros_x = (0..n_blocks)
                .map(|b| x_stats[b * 4 + 1])
                .collect::<Vec<_>>();
            let nzeros_y = (0..n_blocks)
                .map(|b| y_stats[b * 4 + 1])
                .collect::<Vec<_>>();
            let nzeros_b = (0..n_blocks)
                .map(|b| b_stats[b * 4 + 1])
                .collect::<Vec<_>>();
            per_block_upstream_cost(
                &entropy_x,
                &entropy_y,
                &entropy_b,
                &nzeros_x,
                &nzeros_y,
                &nzeros_b,
                &pixel_loss_total,
                entropy_mul,
                scaled_constants,
                quant_for_coeffs,
                block_pixels,
            )
        }
    }
}

/// Persistent variant of [`estimate_entropy_full_strategy_batch_gpu`]:
/// takes pre-uploaded `GpuBlocks` for per-channel pixel inputs and a
/// `GpuPlane` for the mask. Same drop-in semantics for the cost grid
/// output. Avoids ~8 sync downloads per call (forward DCT × 3,
/// entropy err × 3, IDCT err × 3 — minus what shares stats).
#[allow(clippy::too_many_arguments)]
pub fn estimate_entropy_full_strategy_batch_persistent<R: Runtime>(
    enc: &GpuEncoder<R>,
    pixel_blocks_x: &crate::persistent::GpuBlocks<R>,
    pixel_blocks_y: &crate::persistent::GpuBlocks<R>,
    pixel_blocks_b: &crate::persistent::GpuBlocks<R>,
    raw_strategy: u8,
    weights_x_per_block: &[f32],
    weights_y_per_block: &[f32],
    weights_b_per_block: &[f32],
    inv_weights_x_per_block: &[f32],
    inv_weights_y_per_block: &[f32],
    inv_weights_b_per_block: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    mask_plane: &crate::persistent::GpuPlane<R>,
    mask_row_base: &[u32],
    scaled_constants: (f32, f32, f32),
    entropy_mul: f32,
    mode: CostMode,
) -> Vec<f32> {
    // Single-shot wrapper: upload mask_row_base then dispatch. Callers
    // that invoke this fn multiple times with an identical mask_row_base
    // (e.g. the 5 sub-block strategies all on the 8×8 grid) should
    // upload once and call `_with_handle` directly to avoid the
    // duplicated cubecl HtoD (~6 ms per duplicate at 16 MP, given
    // cubecl 0.10's ~0.16 GB/s HtoD ceiling).
    let h_mrb = enc
        .client_ref()
        .create_from_slice(u32::as_bytes(mask_row_base));
    estimate_entropy_full_strategy_batch_persistent_with_handle(
        enc,
        pixel_blocks_x,
        pixel_blocks_y,
        pixel_blocks_b,
        raw_strategy,
        weights_x_per_block,
        weights_y_per_block,
        weights_b_per_block,
        inv_weights_x_per_block,
        inv_weights_y_per_block,
        inv_weights_b_per_block,
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        mask_plane,
        &h_mrb,
        mask_row_base.len(),
        scaled_constants,
        entropy_mul,
        mode,
    )
}

/// Variant of [`estimate_entropy_full_strategy_batch_persistent`]
/// that takes the `mask_row_base` GPU handle directly instead of a
/// host slice. Lets the caller hoist a single upload across multiple
/// strategy calls (e.g. the 5 sub-block 8×8 strategies that all share
/// the same per-block row-base table).
///
/// `mask_row_base_len` is the logical element count (`u32` count) of
/// the buffer behind `mask_row_base_handle` — equal to `n_blocks`.
#[allow(clippy::too_many_arguments)]
pub fn estimate_entropy_full_strategy_batch_persistent_with_handle<R: Runtime>(
    enc: &GpuEncoder<R>,
    pixel_blocks_x: &crate::persistent::GpuBlocks<R>,
    pixel_blocks_y: &crate::persistent::GpuBlocks<R>,
    pixel_blocks_b: &crate::persistent::GpuBlocks<R>,
    raw_strategy: u8,
    weights_x_per_block: &[f32],
    weights_y_per_block: &[f32],
    weights_b_per_block: &[f32],
    inv_weights_x_per_block: &[f32],
    inv_weights_y_per_block: &[f32],
    inv_weights_b_per_block: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    mask_plane: &crate::persistent::GpuPlane<R>,
    mask_row_base_handle: &cubecl::server::Handle,
    mask_row_base_len: usize,
    scaled_constants: (f32, f32, f32),
    entropy_mul: f32,
    mode: CostMode,
) -> Vec<f32> {
    use crate::forks::cfl::{ytob_ratio, ytox_ratio};
    use crate::forks::transform::{
        apply_dct_batch_persistent, apply_idct_batch_persistent, coeff_count_per_strategy,
        tile_dims_pixels,
    };

    let coeff_count = coeff_count_per_strategy(raw_strategy);
    let (block_w, block_h) = tile_dims_pixels(raw_strategy);
    let block_pixels = block_w * block_h;

    let n_blocks = pixel_blocks_x.num_blocks() as usize;
    debug_assert_eq!(pixel_blocks_x.num_blocks(), pixel_blocks_y.num_blocks());
    debug_assert_eq!(pixel_blocks_x.num_blocks(), pixel_blocks_b.num_blocks());
    debug_assert_eq!(pixel_blocks_x.coeffs_per_block() as usize, block_pixels);
    debug_assert_eq!(mask_row_base_len, n_blocks);
    debug_assert_eq!(weights_x_per_block.len(), coeff_count);
    debug_assert_eq!(weights_y_per_block.len(), coeff_count);
    debug_assert_eq!(weights_b_per_block.len(), coeff_count);
    debug_assert_eq!(inv_weights_x_per_block.len(), coeff_count);
    debug_assert_eq!(inv_weights_y_per_block.len(), coeff_count);
    debug_assert_eq!(inv_weights_b_per_block.len(), coeff_count);
    let (_info_loss_mul, cost_delta, _zeros_mul) = scaled_constants;

    // Step 1: forward DCT each channel (no sync).
    let dct_x = apply_dct_batch_persistent(enc, pixel_blocks_x, raw_strategy);
    let dct_y = apply_dct_batch_persistent(enc, pixel_blocks_y, raw_strategy);
    let dct_b = apply_dct_batch_persistent(enc, pixel_blocks_b, raw_strategy);

    // Step 2: per-channel entropy + error-coef writeback (no sync).
    let (g_y_stats, g_y_err) = enc.entropy_coeffs_pixel_blocks_broadcast_w_persistent(
        &dct_y,
        &dct_y,
        weights_y_per_block,
        inv_weights_y_per_block,
        0.0,
        quant_y,
        cost_delta,
    );
    let (g_x_stats, g_x_err) = enc.entropy_coeffs_pixel_blocks_broadcast_w_persistent(
        &dct_x,
        &dct_y,
        weights_x_per_block,
        inv_weights_x_per_block,
        ytox_ratio(ytox),
        quant_x,
        cost_delta,
    );
    let (g_b_stats, g_b_err) = enc.entropy_coeffs_pixel_blocks_broadcast_w_persistent(
        &dct_b,
        &dct_y,
        weights_b_per_block,
        inv_weights_b_per_block,
        ytob_ratio(ytob),
        quant_b,
        cost_delta,
    );

    // Step 3: per-strategy IDCT of error coefficients (no sync).
    let g_pix_err_x = apply_idct_batch_persistent(enc, &g_x_err, raw_strategy);
    let g_pix_err_y = apply_idct_batch_persistent(enc, &g_y_err, raw_strategy);
    let g_pix_err_b = apply_idct_batch_persistent(enc, &g_b_err, raw_strategy);

    // Step 4: per-channel masked 8th-power pixel loss (no sync). The
    // mask_row_base handle is supplied by the caller — multi-strategy
    // callers (e.g. the 5 sub-block 8×8 cost grids) hoist a single
    // upload out of this fn and pass the same handle 5 times.
    let g_loss_x = enc.pixel_loss_blocks_with_handle_persistent(
        &g_pix_err_x,
        mask_plane,
        mask_row_base_handle,
        MASK_CHANNEL_OFFSET[0],
        block_w as u32,
        block_h as u32,
    );
    let g_loss_y = enc.pixel_loss_blocks_with_handle_persistent(
        &g_pix_err_y,
        mask_plane,
        mask_row_base_handle,
        MASK_CHANNEL_OFFSET[1],
        block_w as u32,
        block_h as u32,
    );
    let g_loss_b = enc.pixel_loss_blocks_with_handle_persistent(
        &g_pix_err_b,
        mask_plane,
        mask_row_base_handle,
        MASK_CHANNEL_OFFSET[2],
        block_w as u32,
        block_h as u32,
    );

    // Step 5: download only the small final stats and losses. One
    // batched client.read instead of six sequential read_one
    // syncs (saves 5 queue-drain stalls).
    let ((x_stats, y_stats, b_stats), (loss_x_init, loss_y, loss_b)) = enc
        .download_3stats_3losses(
            &g_x_stats, &g_y_stats, &g_b_stats,
            &g_loss_x, &g_loss_y, &g_loss_b,
        );
    let mut loss_x = loss_x_init;

    // Step 6: extract per-block entropy + apply X-channel multi-block weight.
    let entropy_x = extract_per_block_entropy(&x_stats, n_blocks);
    let entropy_y = extract_per_block_entropy(&y_stats, n_blocks);
    let entropy_b = extract_per_block_entropy(&b_stats, n_blocks);
    if matches!(mode, CostMode::Upstream { .. }) {
        let covered_blocks = block_pixels / 64;
        apply_x_multiblock_weight_to_loss(&mut loss_x, covered_blocks);
    }

    // Step 7: combine + final cost.
    let pixel_loss_total = combine_pixel_loss_3channel(&loss_x, &loss_y, &loss_b);
    match mode {
        CostMode::Simple => {
            let entropy_total = sum_per_block_entropy_3channel(&entropy_x, &entropy_y, &entropy_b);
            per_block_total_cost(&entropy_total, &pixel_loss_total, entropy_mul)
        }
        CostMode::Upstream { quant_for_coeffs } => {
            let nzeros_x = (0..n_blocks).map(|b| x_stats[b * 4 + 1]).collect::<Vec<_>>();
            let nzeros_y = (0..n_blocks).map(|b| y_stats[b * 4 + 1]).collect::<Vec<_>>();
            let nzeros_b = (0..n_blocks).map(|b| b_stats[b * 4 + 1]).collect::<Vec<_>>();
            per_block_upstream_cost(
                &entropy_x,
                &entropy_y,
                &entropy_b,
                &nzeros_x,
                &nzeros_y,
                &nzeros_b,
                &pixel_loss_total,
                entropy_mul,
                scaled_constants,
                quant_for_coeffs,
                block_pixels,
            )
        }
    }
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

/// Broadcast-weights variant of [`entropy_coeffs_pixel_blocks_gpu`].
/// Each weight array is exactly `n_per_block` f32 (one quant matrix
/// or its inverse); the kernel broadcasts across all blocks. Saves
/// `2 * (num_blocks - 1) * n_per_block * 4` bytes of upload traffic
/// per call.
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_pixel_blocks_gpu_broadcast_w<R: Runtime>(
    enc: &GpuEncoder<R>,
    block_c: &[f32],
    block_y: &[f32],
    weights_template: &[f32],
    inv_weights_template: &[f32],
    n_per_block: u32,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
) -> (Vec<f32>, Vec<f32>) {
    enc.entropy_coeffs_pixel_blocks_broadcast_w(
        block_c,
        block_y,
        weights_template,
        inv_weights_template,
        n_per_block,
        cmap_factor,
        quant,
        k_cost_delta,
    )
}

/// Per-8×8-block masked weighted L2 error for 3-channel original vs
/// reconstructed XYB. Wraps [`GpuEncoder::block_l2_errors`].
///
/// Repack a spatial-layout padded plane into block-major layout for
/// a given strategy's tile size. Output length = `n_blocks_strat *
/// (tile_w * tile_h)` where the strategy's tile_w × tile_h pixel
/// rectangle is gathered into one contiguous run per block.
///
/// For DCT8 (8×8 tile) over a 16×16 plane: 4 blocks × 64 floats.
/// For DCT16x16 (16×16 tile) over a 16×16 plane: 1 block × 256 floats.
///
/// Used by [`strategy_search_costs_dct8_16x16`] to feed the entropy
/// orchestrators which expect block-major input.
pub fn repack_plane_to_blocks(
    plane: &[f32],
    padded_width: usize,
    padded_height: usize,
    tile_w: usize,
    tile_h: usize,
) -> Vec<f32> {
    debug_assert_eq!(plane.len(), padded_width * padded_height);
    debug_assert!(padded_width.is_multiple_of(tile_w));
    debug_assert!(padded_height.is_multiple_of(tile_h));
    let xb = padded_width / tile_w;
    let yb = padded_height / tile_h;
    let mut out = alloc::vec![0.0_f32; xb * yb * tile_w * tile_h];
    for by in 0..yb {
        for bx in 0..xb {
            let dst_off = (by * xb + bx) * (tile_w * tile_h);
            for ly in 0..tile_h {
                let src_off = (by * tile_h + ly) * padded_width + bx * tile_w;
                let dst_row = dst_off + ly * tile_w;
                out[dst_row..dst_row + tile_w]
                    .copy_from_slice(&plane[src_off..src_off + tile_w]);
            }
        }
    }
    out
}

/// Strategy-search cost grids for DCT8 + DCT16x16, computed via
/// upstream-faithful `estimate_entropy_full` (entropy + pixel_loss
/// with mask weighting).
///
/// **Replaces** the upstream-non-faithful `compute_cost_grid_*` path
/// (which uses block_l2 instead of estimate_entropy_full). For
/// "full jxl parity" strategy selection, use this one.
///
/// Returns:
/// - `cost_dct8`: per-(8×8 block) cost in raster order
/// - `cost_dct16x16`: per-(16×16 region) cost in raster order
///
/// **Algorithm**: repacks each XYB plane into block-major for both
/// strategies, computes per-block mask_row_base offsets, dispatches
/// the existing `estimate_entropy_full_dct8_batch_gpu` and
/// `estimate_entropy_full_strategy_batch_gpu(DCT16X16)`. Output is
/// already entropy + pixel_loss; the host-side selector
/// (`select_partitions_16x16`) compares the resulting per-region
/// costs.
///
/// **NOTE (Phase A simplification)**: applies neither the upstream
/// per-strategy multiplier (`mul8x8` / `mul16x16`) nor the
/// `kFavor2X2` / `kAvoidEntropyOfTransforms` adjustments. For
/// upstream-faithful selection across distance, callers must
/// post-process the costs with these factors. The MVP shape gets
/// the comparison right between DCT8 and DCT16x16 at any single
/// distance because both terms have the same `mul`-vs-`mul` ratio
/// over the choice (libjxl's `mul8x8 = 1 + (-0.4)/(d+1.4)` and
/// `mul16x16 = 1.0`); a future commit will plug those in.
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_dct8_16x16<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    weights_dct8_x: &[f32],
    weights_dct8_y: &[f32],
    weights_dct8_b: &[f32],
    inv_weights_dct8_x: &[f32],
    inv_weights_dct8_y: &[f32],
    inv_weights_dct8_b: &[f32],
    weights_dct16x16_x: &[f32],
    weights_dct16x16_y: &[f32],
    weights_dct16x16_b: &[f32],
    inv_weights_dct16x16_x: &[f32],
    inv_weights_dct16x16_y: &[f32],
    inv_weights_dct16x16_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(xyb_x.len(), padded_width * padded_height);
    let xsize_blocks_8 = padded_width / 8;
    let ysize_blocks_8 = padded_height / 8;

    // Mask is already on GPU (caller-supplied GpuPlane).
    let g_mask = mask1x1;

    // DCT8 cost grid: host repack + upload, then persistent pipeline.
    let bx8 = repack_plane_to_blocks(xyb_x, padded_width, padded_height, 8, 8);
    let by8 = repack_plane_to_blocks(xyb_y, padded_width, padded_height, 8, 8);
    let bb8 = repack_plane_to_blocks(xyb_b, padded_width, padded_height, 8, 8);
    let n_blocks_8 = xsize_blocks_8 * ysize_blocks_8;
    let g_bx8 = enc.upload_blocks(&bx8, n_blocks_8 as u32, 64);
    let g_by8 = enc.upload_blocks(&by8, n_blocks_8 as u32, 64);
    let g_bb8 = enc.upload_blocks(&bb8, n_blocks_8 as u32, 64);
    let mask_row_base_8: Vec<u32> = (0..n_blocks_8)
        .map(|i| {
            let bx = i % xsize_blocks_8;
            let by = i / xsize_blocks_8;
            (by * 8 * padded_width + bx * 8) as u32
        })
        .collect();

    // libjxl entropy_mul for DCT8: profile.entropy_mul_table[DCT8] = 0.8
    let dct8_entropy_mul = 0.8_f32;
    let cost_dct8 = estimate_entropy_full_dct8_batch_persistent(
        enc,
        &g_bx8,
        &g_by8,
        &g_bb8,
        weights_dct8_x.try_into().expect("64-float DCT8 weights X"),
        weights_dct8_y.try_into().expect("64-float DCT8 weights Y"),
        weights_dct8_b.try_into().expect("64-float DCT8 weights B"),
        inv_weights_dct8_x.try_into().expect("64-float DCT8 inv_weights X"),
        inv_weights_dct8_y.try_into().expect("64-float DCT8 inv_weights Y"),
        inv_weights_dct8_b.try_into().expect("64-float DCT8 inv_weights B"),
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        g_mask,
        &mask_row_base_8,
        scaled_constants,
        dct8_entropy_mul,
        CostMode::Upstream {
            quant_for_coeffs: quant_y,
        },
    );

    // DCT16x16 cost grid: gather 16×16 blocks.
    let xsize_blocks_16 = padded_width / 16;
    let ysize_blocks_16 = padded_height / 16;
    let bx16 = repack_plane_to_blocks(xyb_x, padded_width, padded_height, 16, 16);
    let by16 = repack_plane_to_blocks(xyb_y, padded_width, padded_height, 16, 16);
    let bb16 = repack_plane_to_blocks(xyb_b, padded_width, padded_height, 16, 16);
    let n_blocks_16 = xsize_blocks_16 * ysize_blocks_16;
    let g_bx16 = enc.upload_blocks(&bx16, n_blocks_16 as u32, 256);
    let g_by16 = enc.upload_blocks(&by16, n_blocks_16 as u32, 256);
    let g_bb16 = enc.upload_blocks(&bb16, n_blocks_16 as u32, 256);
    let mask_row_base_16: Vec<u32> = (0..n_blocks_16)
        .map(|i| {
            let bx = i % xsize_blocks_16;
            let by = i / xsize_blocks_16;
            (by * 16 * padded_width + bx * 16) as u32
        })
        .collect();

    // libjxl entropy_mul for DCT16x16: profile.entropy_mul_table[DCT16X16] = 1.34
    let dct16x16_entropy_mul = 1.34_f32;
    let cost_dct16x16 = estimate_entropy_full_strategy_batch_persistent(
        enc,
        &g_bx16,
        &g_by16,
        &g_bb16,
        crate::forks::transform::RAW_STRATEGY_DCT16X16,
        weights_dct16x16_x,
        weights_dct16x16_y,
        weights_dct16x16_b,
        inv_weights_dct16x16_x,
        inv_weights_dct16x16_y,
        inv_weights_dct16x16_b,
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        g_mask,
        &mask_row_base_16,
        scaled_constants,
        dct16x16_entropy_mul,
        CostMode::Upstream {
            quant_for_coeffs: quant_y,
        },
    );

    (cost_dct8, cost_dct16x16)
}

/// Persistent variant of [`strategy_search_costs_dct8_16x16`] that
/// takes pre-gathered DCT8 GpuBlocks (caller already paid the 8x8
/// gather for sub-block strategies) and XYB GpuPlanes for the 16x16
/// gather inside this function. Same `(cost_dct8, cost_dct16x16)`
/// return shape as the host-slice variant.
///
/// **Why**: lossy_encoder.rs already gathers 8x8 GpuBlocks for the
/// sub-block strategies (DCT4x4/DCT4x8/DCT8x4/IDENTITY/DCT2X2). With
/// this variant, the DCT8 cost-grid call reuses those same GpuBlocks
/// instead of re-uploading the host-repacked equivalent. The DCT16x16
/// cost grid gathers 16x16 internally on GPU, also skipping the host
/// repack + upload roundtrip.
///
/// At 1024² this saves: 3× host repack (~10 ms) + 3× upload of 4 MB
/// per channel (8x8 layout) + 3× repack-and-upload at 16x16 layout.
/// Most of the savings come from removing the 24 MB of redundant
/// PCIe traffic.
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_dct8_16x16_persistent<R: Runtime>(
    enc: &GpuEncoder<R>,
    g_8x: &crate::persistent::GpuBlocks<R>,
    g_8y: &crate::persistent::GpuBlocks<R>,
    g_8b: &crate::persistent::GpuBlocks<R>,
    xyb_plane_x: &crate::persistent::GpuPlane<R>,
    xyb_plane_y: &crate::persistent::GpuPlane<R>,
    xyb_plane_b: &crate::persistent::GpuPlane<R>,
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    weights_dct8_x: &[f32],
    weights_dct8_y: &[f32],
    weights_dct8_b: &[f32],
    inv_weights_dct8_x: &[f32],
    inv_weights_dct8_y: &[f32],
    inv_weights_dct8_b: &[f32],
    weights_dct16x16_x: &[f32],
    weights_dct16x16_y: &[f32],
    weights_dct16x16_b: &[f32],
    inv_weights_dct16x16_x: &[f32],
    inv_weights_dct16x16_y: &[f32],
    inv_weights_dct16x16_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
) -> (Vec<f32>, Vec<f32>) {
    let xsize_blocks_8 = padded_width / 8;
    let ysize_blocks_8 = padded_height / 8;
    let n_blocks_8 = xsize_blocks_8 * ysize_blocks_8;
    debug_assert_eq!(g_8x.num_blocks() as usize, n_blocks_8);
    debug_assert_eq!(g_8x.coeffs_per_block(), 64);

    let mask_row_base_8: Vec<u32> = (0..n_blocks_8)
        .map(|i| {
            let bx = i % xsize_blocks_8;
            let by = i / xsize_blocks_8;
            (by * 8 * padded_width + bx * 8) as u32
        })
        .collect();

    // libjxl entropy_mul for DCT8: profile.entropy_mul_table[DCT8] = 0.8
    let dct8_entropy_mul = 0.8_f32;
    let cost_dct8 = estimate_entropy_full_dct8_batch_persistent(
        enc,
        g_8x,
        g_8y,
        g_8b,
        weights_dct8_x.try_into().expect("64-float DCT8 weights X"),
        weights_dct8_y.try_into().expect("64-float DCT8 weights Y"),
        weights_dct8_b.try_into().expect("64-float DCT8 weights B"),
        inv_weights_dct8_x.try_into().expect("64-float DCT8 inv_weights X"),
        inv_weights_dct8_y.try_into().expect("64-float DCT8 inv_weights Y"),
        inv_weights_dct8_b.try_into().expect("64-float DCT8 inv_weights B"),
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        mask1x1,
        &mask_row_base_8,
        scaled_constants,
        dct8_entropy_mul,
        CostMode::Upstream {
            quant_for_coeffs: quant_y,
        },
    );

    // DCT16x16 cost grid: gather 16×16 blocks on GPU.
    let xsize_blocks_16 = padded_width / 16;
    let ysize_blocks_16 = padded_height / 16;
    let n_blocks_16 = xsize_blocks_16 * ysize_blocks_16;
    let g_bx16 = enc.gather_blocks_persistent(xyb_plane_x, 16, 16);
    let g_by16 = enc.gather_blocks_persistent(xyb_plane_y, 16, 16);
    let g_bb16 = enc.gather_blocks_persistent(xyb_plane_b, 16, 16);
    let mask_row_base_16: Vec<u32> = (0..n_blocks_16)
        .map(|i| {
            let bx = i % xsize_blocks_16;
            let by = i / xsize_blocks_16;
            (by * 16 * padded_width + bx * 16) as u32
        })
        .collect();

    // libjxl entropy_mul for DCT16x16: profile.entropy_mul_table[DCT16X16] = 1.34
    let dct16x16_entropy_mul = 1.34_f32;
    let cost_dct16x16 = estimate_entropy_full_strategy_batch_persistent(
        enc,
        &g_bx16,
        &g_by16,
        &g_bb16,
        crate::forks::transform::RAW_STRATEGY_DCT16X16,
        weights_dct16x16_x,
        weights_dct16x16_y,
        weights_dct16x16_b,
        inv_weights_dct16x16_x,
        inv_weights_dct16x16_y,
        inv_weights_dct16x16_b,
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        mask1x1,
        &mask_row_base_16,
        scaled_constants,
        dct16x16_entropy_mul,
        CostMode::Upstream {
            quant_for_coeffs: quant_y,
        },
    );

    (cost_dct8, cost_dct16x16)
}

/// Cost grid for DCT32x32, sharing inputs with the other strategy
/// search helpers. Needs the padded plane to have both dims multiples
/// of 32 (returns an empty Vec otherwise — caller skips the strategy).
///
/// libjxl entropy_mul for DCT32x32 = 1.34 (same as DCT16x16 in
/// profile.entropy_mul_table).
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_dct32x32<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
) -> Vec<f32> {
    use crate::forks::transform::RAW_STRATEGY_DCT32X32;
    if !padded_width.is_multiple_of(32) || !padded_height.is_multiple_of(32) {
        return Vec::new();
    }
    let xb = padded_width / 32;
    let yb = padded_height / 32;
    let n_blocks = xb * yb;

    let bx_p = repack_plane_to_blocks(xyb_x, padded_width, padded_height, 32, 32);
    let by_p = repack_plane_to_blocks(xyb_y, padded_width, padded_height, 32, 32);
    let bb_p = repack_plane_to_blocks(xyb_b, padded_width, padded_height, 32, 32);
    let g_bx = enc.upload_blocks(&bx_p, n_blocks as u32, 1024);
    let g_by = enc.upload_blocks(&by_p, n_blocks as u32, 1024);
    let g_bb = enc.upload_blocks(&bb_p, n_blocks as u32, 1024);
    let mask_row_base: Vec<u32> = (0..n_blocks)
        .map(|i| {
            let bx_i = i % xb;
            let by_i = i / xb;
            (by_i * 32 * padded_width + bx_i * 32) as u32
        })
        .collect();

    // libjxl entropy_mul: profile.entropy_mul_table[DCT32X32] = 1.48.
    // We use 2.5 here pending implementation of the rest of libjxl's
    // cost-model adjustments (X-channel multi-block weight,
    // kAvoidEntropyOfTransforms, AdjustQuantBlockAC). Without those
    // counterweights, 1.48 over-selects DCT32 (butteraugli 8.36 on
    // CLIC test image vs 1.35 with 2.5). Bumped 2.5 → 4.0 (May 9 2026)
    // after corpus-sweep diagnostics found 2.5 still over-selects DCT32
    // on detailed content (image 2684452d: 40 DCT32 picks → +31%
    // butteraugli regression vs uniform). Same root cause as DCT64x64
    // band-aid — missing libjxl pixel-loss penalty for large transforms.
    //
    // 4.0 → 3.0 (May 9 2026, this commit): once the corpus regression
    // test landed, bisected DCT32 mul on the 11-image corpus.
    //   4 ✓ baseline, 3 ✓, 2 ✗ (07b9f93f -1.5% — gain that drifts
    //   outside the 0.5% tolerance band, AND 22ea12c9 / 2684452d both
    //   regress +2.4% / +2.0%; the strat-wins photos shift to
    //   RefineDct8 because DCT32 picks on those images aren't quite
    //   right yet at 2.0). Settled on 3.0 — strict score parity.
    //
    // Re-tune toward libjxl 1.48 when the missing pixel-loss term lands.
    let entropy_mul = 3.0_f32;

    estimate_entropy_full_strategy_batch_persistent(
        enc,
        &g_bx,
        &g_by,
        &g_bb,
        RAW_STRATEGY_DCT32X32,
        weights_x,
        weights_y,
        weights_b,
        inv_weights_x,
        inv_weights_y,
        inv_weights_b,
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        mask1x1,
        &mask_row_base,
        scaled_constants,
        entropy_mul,
        CostMode::Upstream {
            quant_for_coeffs: quant_y,
        },
    )
}

/// Cost grid for an 8x8-tier sub-block strategy (DCT4x4, DCT4x8,
/// DCT8x4, IDENTITY, DCT2x2). All of these extract 8x8 pixel tiles
/// and produce 64 coefficients per block — same shape as DCT8.
///
/// Reuses the existing 8x8 input gather (host-side) but feeds it
/// through the persistent estimate_entropy chain with the chosen
/// strategy's weights and entropy_mul.
///
/// libjxl entropy_mul values (from EntropyMulTable::reference):
///   DCT4x4: 1.08    DCT4x8/DCT8x4: 0.859316    IDENTITY: 1.0428    DCT2x2: 0.95
/// Caller passes the entropy_mul; we don't re-tune them in this
/// helper because they're closer to neutral than the larger-transform
/// muls (no need for the 2.5/3.5 anti-bias shifts).
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_subblock_8x8<R: Runtime>(
    enc: &GpuEncoder<R>,
    pre_gathered_8x8_x: &crate::persistent::GpuBlocks<R>,
    pre_gathered_8x8_y: &crate::persistent::GpuBlocks<R>,
    pre_gathered_8x8_b: &crate::persistent::GpuBlocks<R>,
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    raw_strategy: u8,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
    entropy_mul: f32,
) -> Vec<f32> {
    let xb = padded_width / 8;
    let yb = padded_height / 8;
    let n_blocks = xb * yb;
    debug_assert_eq!(pre_gathered_8x8_x.num_blocks() as usize, n_blocks);
    debug_assert_eq!(pre_gathered_8x8_x.coeffs_per_block(), 64);

    let mask_row_base: Vec<u32> = (0..n_blocks)
        .map(|i| {
            let bx_i = i % xb;
            let by_i = i / xb;
            (by_i * 8 * padded_width + bx_i * 8) as u32
        })
        .collect();

    estimate_entropy_full_strategy_batch_persistent(
        enc,
        pre_gathered_8x8_x,
        pre_gathered_8x8_y,
        pre_gathered_8x8_b,
        raw_strategy,
        weights_x,
        weights_y,
        weights_b,
        inv_weights_x,
        inv_weights_y,
        inv_weights_b,
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        mask1x1,
        &mask_row_base,
        scaled_constants,
        entropy_mul,
        CostMode::Upstream {
            quant_for_coeffs: quant_y,
        },
    )
}

/// Variant of [`strategy_search_costs_subblock_8x8`] that takes a
/// pre-uploaded `mask_row_base` GPU handle. Lets the caller hoist the
/// upload out of a multi-strategy loop — at 16 MP cubecl 0.10's
/// HtoD takes ~6 ms for the 1 MB mask_row_base buffer, so 5 sub-block
/// strategies × 6 ms = 30 ms wasted on duplicate uploads. Hoisting
/// recovers 24 ms.
///
/// `mask_row_base_len` must equal `(padded_width / 8) * (padded_height / 8)`.
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_subblock_8x8_with_handle<R: Runtime>(
    enc: &GpuEncoder<R>,
    pre_gathered_8x8_x: &crate::persistent::GpuBlocks<R>,
    pre_gathered_8x8_y: &crate::persistent::GpuBlocks<R>,
    pre_gathered_8x8_b: &crate::persistent::GpuBlocks<R>,
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    mask_row_base_handle: &cubecl::server::Handle,
    mask_row_base_len: usize,
    raw_strategy: u8,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
    entropy_mul: f32,
) -> Vec<f32> {
    let xb = padded_width / 8;
    let yb = padded_height / 8;
    let n_blocks = xb * yb;
    debug_assert_eq!(pre_gathered_8x8_x.num_blocks() as usize, n_blocks);
    debug_assert_eq!(pre_gathered_8x8_x.coeffs_per_block(), 64);
    debug_assert_eq!(mask_row_base_len, n_blocks);

    estimate_entropy_full_strategy_batch_persistent_with_handle(
        enc,
        pre_gathered_8x8_x,
        pre_gathered_8x8_y,
        pre_gathered_8x8_b,
        raw_strategy,
        weights_x,
        weights_y,
        weights_b,
        inv_weights_x,
        inv_weights_y,
        inv_weights_b,
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        mask1x1,
        mask_row_base_handle,
        mask_row_base_len,
        scaled_constants,
        entropy_mul,
        CostMode::Upstream {
            quant_for_coeffs: quant_y,
        },
    )
}

/// Cost grid for DCT64x64. Returns empty Vec when image dims aren't
/// multiples of 64.
///
/// libjxl entropy_mul: profile.entropy_mul_table[DCT64X64] = 2.25
/// (verified against jxl-encoder/src/effort.rs::EntropyMulTable::reference).
/// As with DCT32x32, we use a higher tuned value pending the rest
/// of the cost-model adjustments (see strategy_search_costs_dct32x32).
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_dct64x64<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
) -> Vec<f32> {
    use crate::forks::transform::RAW_STRATEGY_DCT64X64;
    if !padded_width.is_multiple_of(64) || !padded_height.is_multiple_of(64) {
        return Vec::new();
    }
    let xb = padded_width / 64;
    let yb = padded_height / 64;
    let n_blocks = xb * yb;
    let coeff_count = 4096_u32;

    let bx_p = repack_plane_to_blocks(xyb_x, padded_width, padded_height, 64, 64);
    let by_p = repack_plane_to_blocks(xyb_y, padded_width, padded_height, 64, 64);
    let bb_p = repack_plane_to_blocks(xyb_b, padded_width, padded_height, 64, 64);
    let g_bx = enc.upload_blocks(&bx_p, n_blocks as u32, coeff_count);
    let g_by = enc.upload_blocks(&by_p, n_blocks as u32, coeff_count);
    let g_bb = enc.upload_blocks(&bb_p, n_blocks as u32, coeff_count);
    let mask_row_base: Vec<u32> = (0..n_blocks)
        .map(|i| {
            let bx_i = i % xb;
            let by_i = i / xb;
            (by_i * 64 * padded_width + bx_i * 64) as u32
        })
        .collect();

    // libjxl entropy_mul = 2.25; ours is 6.0 — sequence of band-aid
    // bumps:
    //   3.5 → 8.0 (May 9 2026): catastrophic regressions on 3/16
    //     CLIC photos (1cba10ad +94% butteraugli vs uniform).
    //   8.0 → 16.0 (May 9 2026): one image still leaked DCT64
    //     picks at 8.0 (11f2b039 +3% loss).
    //   16.0 → 6.0 (May 9 2026): bisected on the original 11-image
    //     corpus (d=1.0 only). 16 ✓, 12 ✓, 8 ✓, 6 ✓, 5 path-shifts,
    //     4 ✗ (22ea12c9 regresses +2.4%).
    //   6.0 → 5.0 (May 9 2026 evening): once corpus coverage
    //     expanded to d=0.5/1.0/2.0 (33 cases), DCT64=5 was found
    //     to deliver a real -3.9% improvement on 22ea12c9 d=0.5
    //     (strat-search picks DCT64 correctly there for actual gain).
    //   5.0 → 4.8 (May 9 2026 evening, this commit): finer-grained
    //     bisection found another -2.1% on 22ea12c9 d=0.5 (cumulative
    //     -5.9% from 6.0 baseline). 4.6 regresses (+1%), 4.5/4.0
    //     regress at d=1 — 4.8 is the new sweet spot.
    //
    // The cost model still misses libjxl's pixel-loss penalty for
    // large transforms on detailed content — without that
    // counterweight, DCT64's "few-coefs" entropy advantage wins
    // even when reconstruction is catastrophically blurry.
    // Suppressing DCT64 via 6.0 mul is still a band-aid until the
    // missing pixel-loss term lands. Same rationale as DCT32x32.
    // The proper fix is a content-aware gate (e.g. mask1x1
    // smoothness threshold) before DCT64 even enters the cost grid.
    let entropy_mul = 4.8_f32;

    estimate_entropy_full_strategy_batch_persistent(
        enc,
        &g_bx,
        &g_by,
        &g_bb,
        RAW_STRATEGY_DCT64X64,
        weights_x,
        weights_y,
        weights_b,
        inv_weights_x,
        inv_weights_y,
        inv_weights_b,
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        mask1x1,
        &mask_row_base,
        scaled_constants,
        entropy_mul,
        CostMode::Upstream {
            quant_for_coeffs: quant_y,
        },
    )
}

/// Cost grid for DCT64x32 or DCT32x64 (rectangular DCT64 family).
/// Returns empty Vec when image dims aren't aligned for the chosen
/// strategy.
///
/// libjxl entropy_mul: profile.entropy_mul_table[DCT64X32] = 2.25
/// (same as DCT64x64). Tuned higher here pending the rest of the
/// cost-model adjustments.
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_dct64x32_or_32x64<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    raw_strategy: u8,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
) -> Vec<f32> {
    use crate::forks::transform::{tile_dims_pixels, RAW_STRATEGY_DCT32X64, RAW_STRATEGY_DCT64X32};
    debug_assert!(
        raw_strategy == RAW_STRATEGY_DCT64X32 || raw_strategy == RAW_STRATEGY_DCT32X64,
        "this helper is for DCT64x32/DCT32x64 only; got {raw_strategy}"
    );
    let (tile_w, tile_h) = tile_dims_pixels(raw_strategy);
    if !padded_width.is_multiple_of(tile_w) || !padded_height.is_multiple_of(tile_h) {
        return Vec::new();
    }
    let bx = padded_width / tile_w;
    let by = padded_height / tile_h;
    let n_blocks = bx * by;
    let coeff_count = (tile_w * tile_h) as u32;

    let bx_p = repack_plane_to_blocks(xyb_x, padded_width, padded_height, tile_w, tile_h);
    let by_p = repack_plane_to_blocks(xyb_y, padded_width, padded_height, tile_w, tile_h);
    let bb_p = repack_plane_to_blocks(xyb_b, padded_width, padded_height, tile_w, tile_h);
    let g_bx = enc.upload_blocks(&bx_p, n_blocks as u32, coeff_count);
    let g_by = enc.upload_blocks(&by_p, n_blocks as u32, coeff_count);
    let g_bb = enc.upload_blocks(&bb_p, n_blocks as u32, coeff_count);
    let mask_row_base: Vec<u32> = (0..n_blocks)
        .map(|i| {
            let bx_i = i % bx;
            let by_i = i / bx;
            (by_i * tile_h * padded_width + bx_i * tile_w) as u32
        })
        .collect();

    // DCT64x32 / DCT32x64 — same suppression rationale as DCT64x64
    // (see the corpus-sweep note above). 3.5 → 8.0 → 16.0 → 6.0 → 5.0
    // sequence; stays in lockstep with DCT64x64.
    let entropy_mul = 4.8_f32;

    estimate_entropy_full_strategy_batch_persistent(
        enc,
        &g_bx,
        &g_by,
        &g_bb,
        raw_strategy,
        weights_x,
        weights_y,
        weights_b,
        inv_weights_x,
        inv_weights_y,
        inv_weights_b,
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        mask1x1,
        &mask_row_base,
        scaled_constants,
        entropy_mul,
        CostMode::Upstream {
            quant_for_coeffs: quant_y,
        },
    )
}

/// Cost grid for DCT32x16 or DCT16x32 (the rectangular DCT32 family).
/// Returns empty Vec when image dims aren't compatible.
///
/// libjxl entropy_mul: profile.entropy_mul_table[DCT16X32] = 1.49
/// (verified against jxl-encoder/src/effort.rs::EntropyMulTable::reference).
/// As with DCT32x32, we use a tuned higher value here pending the
/// remaining cost-model adjustments.
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_dct32x16_or_16x32<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    raw_strategy: u8,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
) -> Vec<f32> {
    use crate::forks::transform::{tile_dims_pixels, RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT32X16};
    debug_assert!(
        raw_strategy == RAW_STRATEGY_DCT32X16 || raw_strategy == RAW_STRATEGY_DCT16X32,
        "this helper is for DCT32x16/DCT16x32 only; got {raw_strategy}"
    );
    let (tile_w, tile_h) = tile_dims_pixels(raw_strategy);
    if !padded_width.is_multiple_of(tile_w) || !padded_height.is_multiple_of(tile_h) {
        return Vec::new();
    }
    let bx = padded_width / tile_w;
    let by = padded_height / tile_h;
    let n_blocks = bx * by;
    let coeff_count = (tile_w * tile_h) as u32;

    let bx_p = repack_plane_to_blocks(xyb_x, padded_width, padded_height, tile_w, tile_h);
    let by_p = repack_plane_to_blocks(xyb_y, padded_width, padded_height, tile_w, tile_h);
    let bb_p = repack_plane_to_blocks(xyb_b, padded_width, padded_height, tile_w, tile_h);
    let g_bx = enc.upload_blocks(&bx_p, n_blocks as u32, coeff_count);
    let g_by = enc.upload_blocks(&by_p, n_blocks as u32, coeff_count);
    let g_bb = enc.upload_blocks(&bb_p, n_blocks as u32, coeff_count);
    let mask_row_base: Vec<u32> = (0..n_blocks)
        .map(|i| {
            let bx_i = i % bx;
            let by_i = i / bx;
            (by_i * tile_h * padded_width + bx_i * tile_w) as u32
        })
        .collect();

    // libjxl entropy_mul = 1.49; tuned higher (2.5 → 2.2 May 9 2026)
    // until full cost model lands. Same rationale as
    // strategy_search_costs_dct32x32. Bisected on the 11-image corpus
    // regression test:
    //   2.5 ✓ baseline, 2.3 ✓, 2.2 ✓, 2.1 ✗ (22ea12c9 / 2684452d
    //   strat-wins photos shift to RefineDct8 with +2.4% / +2.0%
    //   regression).
    // Settled on 2.2 — strict score parity, modest 12% improvement
    // toward libjxl reference 1.49.
    let entropy_mul = 2.2_f32;

    estimate_entropy_full_strategy_batch_persistent(
        enc,
        &g_bx,
        &g_by,
        &g_bb,
        raw_strategy,
        weights_x,
        weights_y,
        weights_b,
        inv_weights_x,
        inv_weights_y,
        inv_weights_b,
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        mask1x1,
        &mask_row_base,
        scaled_constants,
        entropy_mul,
        CostMode::Upstream {
            quant_for_coeffs: quant_y,
        },
    )
}

/// Cost grid for one rectangular DCT16x8 or DCT8x16 strategy, sharing
/// the inputs already prepared by [`strategy_search_costs_dct8_16x16`].
///
/// `tile_w`/`tile_h` give the strategy's pixel footprint:
/// - DCT16x8: tile_w=8, tile_h=16  (block is 16 tall × 8 wide pixels →
///   1-block-wide × 2-blocks-tall in the 8x8 grid)
/// - DCT8x16: tile_w=16, tile_h=8  (block is 8 tall × 16 wide pixels →
///   2-blocks-wide × 1-block-tall in the 8x8 grid)
///
/// Returns the per-block cost grid in row-major order at the strategy's
/// natural alignment. Use the result as `extra.dct_16x8` or
/// `extra.dct_8x16` in [`select_partitions_16x16_full`].
///
/// Phase A note: hard-codes the libjxl entropy_mul (1.21 for DCT16x8/
/// DCT8x16). Per-strategy mul/bonus/penalty post-processing deferred
/// to Phase C; same caveat as `strategy_search_costs_dct8_16x16`.
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_dct16x8_or_8x16<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    raw_strategy: u8,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
) -> Vec<f32> {
    use crate::forks::transform::{tile_dims_pixels, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT8X16};
    debug_assert!(
        raw_strategy == RAW_STRATEGY_DCT16X8 || raw_strategy == RAW_STRATEGY_DCT8X16,
        "this helper is for DCT16x8/DCT8x16 only; got {raw_strategy}"
    );
    let (tile_w, tile_h) = tile_dims_pixels(raw_strategy);
    let bx = padded_width / tile_w;
    let by = padded_height / tile_h;
    let n_blocks = bx * by;

    let bx_p = repack_plane_to_blocks(xyb_x, padded_width, padded_height, tile_w, tile_h);
    let by_p = repack_plane_to_blocks(xyb_y, padded_width, padded_height, tile_w, tile_h);
    let bb_p = repack_plane_to_blocks(xyb_b, padded_width, padded_height, tile_w, tile_h);
    let coeff_count = (tile_w * tile_h) as u32;
    let g_bx = enc.upload_blocks(&bx_p, n_blocks as u32, coeff_count);
    let g_by = enc.upload_blocks(&by_p, n_blocks as u32, coeff_count);
    let g_bb = enc.upload_blocks(&bb_p, n_blocks as u32, coeff_count);
    let g_mask = mask1x1; // caller-supplied GpuPlane.
    let mask_row_base: Vec<u32> = (0..n_blocks)
        .map(|i| {
            let bx_i = i % bx;
            let by_i = i / bx;
            (by_i * tile_h * padded_width + bx_i * tile_w) as u32
        })
        .collect();

    // libjxl entropy_mul: profile.entropy_mul_table[DCT16X8 / DCT8X16] = 1.21
    let entropy_mul = 1.21_f32;

    estimate_entropy_full_strategy_batch_persistent(
        enc,
        &g_bx,
        &g_by,
        &g_bb,
        raw_strategy,
        weights_x,
        weights_y,
        weights_b,
        inv_weights_x,
        inv_weights_y,
        inv_weights_b,
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        g_mask,
        &mask_row_base,
        scaled_constants,
        entropy_mul,
        CostMode::Upstream {
            quant_for_coeffs: quant_y,
        },
    )
}

/// Internal helper: runs the full estimate_entropy_full pipeline for a
/// single rectangular non-DCT8 strategy on GPU-resident inputs.
///
/// Gathers per-tile GpuBlocks from the supplied GpuPlanes, computes the
/// per-block mask_row_base offsets, and forwards to
/// `estimate_entropy_full_strategy_batch_persistent`. Returns the
/// Vec<f32> cost grid in raster order at the strategy's natural
/// alignment (caller treats as `Vec::new()` if the dims don't tile).
#[allow(clippy::too_many_arguments)]
fn rect_strategy_cost_xyb_persistent<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_plane_x: &crate::persistent::GpuPlane<R>,
    xyb_plane_y: &crate::persistent::GpuPlane<R>,
    xyb_plane_b: &crate::persistent::GpuPlane<R>,
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    raw_strategy: u8,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
    entropy_mul: f32,
) -> Vec<f32> {
    use crate::forks::transform::tile_dims_pixels;
    let (tile_w, tile_h) = tile_dims_pixels(raw_strategy);
    if !padded_width.is_multiple_of(tile_w) || !padded_height.is_multiple_of(tile_h) {
        return Vec::new();
    }
    let bx = padded_width / tile_w;
    let by = padded_height / tile_h;
    let n_blocks = bx * by;
    let g_bx = enc.gather_blocks_persistent(xyb_plane_x, tile_w as u32, tile_h as u32);
    let g_by = enc.gather_blocks_persistent(xyb_plane_y, tile_w as u32, tile_h as u32);
    let g_bb = enc.gather_blocks_persistent(xyb_plane_b, tile_w as u32, tile_h as u32);
    let mask_row_base: Vec<u32> = (0..n_blocks)
        .map(|i| {
            let bx_i = i % bx;
            let by_i = i / bx;
            (by_i * tile_h * padded_width + bx_i * tile_w) as u32
        })
        .collect();
    estimate_entropy_full_strategy_batch_persistent(
        enc,
        &g_bx,
        &g_by,
        &g_bb,
        raw_strategy,
        weights_x,
        weights_y,
        weights_b,
        inv_weights_x,
        inv_weights_y,
        inv_weights_b,
        quant_x,
        quant_y,
        quant_b,
        ytox,
        ytob,
        mask1x1,
        &mask_row_base,
        scaled_constants,
        entropy_mul,
        CostMode::Upstream {
            quant_for_coeffs: quant_y,
        },
    )
}

/// Persistent variant of [`strategy_search_costs_dct16x8_or_8x16`].
/// Takes XYB GpuPlanes, gathers tile blocks internally on GPU.
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_dct16x8_or_8x16_persistent<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_plane_x: &crate::persistent::GpuPlane<R>,
    xyb_plane_y: &crate::persistent::GpuPlane<R>,
    xyb_plane_b: &crate::persistent::GpuPlane<R>,
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    raw_strategy: u8,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
) -> Vec<f32> {
    use crate::forks::transform::{RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT8X16};
    debug_assert!(
        raw_strategy == RAW_STRATEGY_DCT16X8 || raw_strategy == RAW_STRATEGY_DCT8X16,
        "this helper is for DCT16x8/DCT8x16 only; got {raw_strategy}"
    );
    rect_strategy_cost_xyb_persistent(
        enc, xyb_plane_x, xyb_plane_y, xyb_plane_b,
        padded_width, padded_height, mask1x1, raw_strategy,
        weights_x, weights_y, weights_b,
        inv_weights_x, inv_weights_y, inv_weights_b,
        quant_x, quant_y, quant_b, ytox, ytob,
        scaled_constants, 1.21_f32,
    )
}

/// Persistent variant of [`strategy_search_costs_dct32x32`]. Takes XYB
/// GpuPlanes, gathers 32×32 blocks internally. Returns Vec::new() when
/// dims aren't multiples of 32 (matches the host-slice variant).
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_dct32x32_persistent<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_plane_x: &crate::persistent::GpuPlane<R>,
    xyb_plane_y: &crate::persistent::GpuPlane<R>,
    xyb_plane_b: &crate::persistent::GpuPlane<R>,
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
) -> Vec<f32> {
    use crate::forks::transform::RAW_STRATEGY_DCT32X32;
    // entropy_mul = 3.0 — see the host-slice variant for bisection history.
    rect_strategy_cost_xyb_persistent(
        enc, xyb_plane_x, xyb_plane_y, xyb_plane_b,
        padded_width, padded_height, mask1x1, RAW_STRATEGY_DCT32X32,
        weights_x, weights_y, weights_b,
        inv_weights_x, inv_weights_y, inv_weights_b,
        quant_x, quant_y, quant_b, ytox, ytob,
        scaled_constants, 3.0_f32,
    )
}

/// Persistent variant of [`strategy_search_costs_dct32x16_or_16x32`].
/// Takes XYB GpuPlanes, gathers tile blocks internally. Returns
/// Vec::new() when dims aren't aligned for the chosen strategy.
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_dct32x16_or_16x32_persistent<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_plane_x: &crate::persistent::GpuPlane<R>,
    xyb_plane_y: &crate::persistent::GpuPlane<R>,
    xyb_plane_b: &crate::persistent::GpuPlane<R>,
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    raw_strategy: u8,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
) -> Vec<f32> {
    use crate::forks::transform::{RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT32X16};
    debug_assert!(
        raw_strategy == RAW_STRATEGY_DCT32X16 || raw_strategy == RAW_STRATEGY_DCT16X32,
        "this helper is for DCT32x16/DCT16x32 only; got {raw_strategy}"
    );
    // entropy_mul = 2.2 — see the host-slice variant for bisection history.
    rect_strategy_cost_xyb_persistent(
        enc, xyb_plane_x, xyb_plane_y, xyb_plane_b,
        padded_width, padded_height, mask1x1, raw_strategy,
        weights_x, weights_y, weights_b,
        inv_weights_x, inv_weights_y, inv_weights_b,
        quant_x, quant_y, quant_b, ytox, ytob,
        scaled_constants, 2.2_f32,
    )
}

/// Persistent variant of [`strategy_search_costs_dct64x64`]. Takes XYB
/// GpuPlanes, gathers 64×64 blocks internally. Returns Vec::new() when
/// dims aren't multiples of 64.
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_dct64x64_persistent<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_plane_x: &crate::persistent::GpuPlane<R>,
    xyb_plane_y: &crate::persistent::GpuPlane<R>,
    xyb_plane_b: &crate::persistent::GpuPlane<R>,
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
) -> Vec<f32> {
    use crate::forks::transform::RAW_STRATEGY_DCT64X64;
    // entropy_mul = 4.8 — see the host-slice variant for bisection history.
    rect_strategy_cost_xyb_persistent(
        enc, xyb_plane_x, xyb_plane_y, xyb_plane_b,
        padded_width, padded_height, mask1x1, RAW_STRATEGY_DCT64X64,
        weights_x, weights_y, weights_b,
        inv_weights_x, inv_weights_y, inv_weights_b,
        quant_x, quant_y, quant_b, ytox, ytob,
        scaled_constants, 4.8_f32,
    )
}

/// Persistent variant of [`strategy_search_costs_dct64x32_or_32x64`].
/// Takes XYB GpuPlanes, gathers tile blocks internally.
#[allow(clippy::too_many_arguments)]
pub fn strategy_search_costs_dct64x32_or_32x64_persistent<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_plane_x: &crate::persistent::GpuPlane<R>,
    xyb_plane_y: &crate::persistent::GpuPlane<R>,
    xyb_plane_b: &crate::persistent::GpuPlane<R>,
    padded_width: usize,
    padded_height: usize,
    mask1x1: &crate::persistent::GpuPlane<R>,
    raw_strategy: u8,
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    inv_weights_x: &[f32],
    inv_weights_y: &[f32],
    inv_weights_b: &[f32],
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    ytox: i8,
    ytob: i8,
    scaled_constants: (f32, f32, f32),
) -> Vec<f32> {
    use crate::forks::transform::{RAW_STRATEGY_DCT32X64, RAW_STRATEGY_DCT64X32};
    debug_assert!(
        raw_strategy == RAW_STRATEGY_DCT64X32 || raw_strategy == RAW_STRATEGY_DCT32X64,
        "this helper is for DCT64x32/DCT32x64 only; got {raw_strategy}"
    );
    // entropy_mul = 4.8 — same as DCT64x64; see host-slice variant.
    rect_strategy_cost_xyb_persistent(
        enc, xyb_plane_x, xyb_plane_y, xyb_plane_b,
        padded_width, padded_height, mask1x1, raw_strategy,
        weights_x, weights_y, weights_b,
        inv_weights_x, inv_weights_y, inv_weights_b,
        quant_x, quant_y, quant_b, ytox, ytob,
        scaled_constants, 4.8_f32,
    )
}

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
    fn test_estimate_entropy_full_strategy_batch_gpu_dct16x16_upstream_mode() {
        // DCT16x16 in Upstream mode triggers the X-multiblock weight
        // (covered_blocks = 256/64 = 4 → w = 1.5). For zero input,
        // X's entropy is 0 anyway so the weighting is a no-op on the
        // entropy term — but the result should still be FINITE and
        // non-negative.
        use crate::forks::transform::RAW_STRATEGY_DCT16X16;
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n_blocks = 1_usize;
        let zeros = alloc::vec![0.0_f32; n_blocks * 256];
        let weights_one = alloc::vec![1.0_f32; 256];
        let mask = alloc::vec![1.0_f32; 16 * 16];
        let mask_row_base = alloc::vec![0_u32; n_blocks];

        let costs = estimate_entropy_full_strategy_batch_gpu(
            &enc,
            &zeros,
            &zeros,
            &zeros,
            RAW_STRATEGY_DCT16X16,
            &weights_one,
            &weights_one,
            &weights_one,
            &weights_one,
            &weights_one,
            &weights_one,
            1.0,
            1.0,
            1.0,
            0,
            0,
            &mask,
            &mask_row_base,
            16,
            COEFF_DOMAIN_CONSTANTS,
            1.0,
            CostMode::Upstream {
                quant_for_coeffs: 1.0,
            },
        );
        assert_eq!(costs.len(), n_blocks);
        // Zero input → zero entropy in each channel + zero pixel loss.
        // Per-channel nzeros_bits term: f(0) = 7. Y + B contribute
        // 7 * COEFF_DOMAIN_CONSTANTS.2 each. X gets 7 * 7.565 * w
        // where w = 1.5. So total = 7 * 7.565 * (2 + 1.5) ≈ 185.34.
        let per_channel_nzero = 7.0 * COEFF_DOMAIN_CONSTANTS.2;
        let expected = per_channel_nzero * (2.0 + 1.5);
        for &c in &costs {
            assert!(
                (c - expected).abs() < 0.5,
                "DCT16x16 upstream mode zero-input cost: got {c}, expected ~{expected}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_estimate_entropy_full_strategy_batch_gpu_dct16x16_zero() {
        // Same zero-input pattern as the DCT8 test but for DCT16x16
        // (256 pixels per block, n_per_block=256, block_w=block_h=16).
        // Verifies the strategy-generic orchestrator wires through
        // dct_blocks_gpu / apply_idct_batch_gpu / pixel_loss_blocks_gpu
        // with the right block dims for a non-DCT8 strategy.
        use crate::forks::transform::RAW_STRATEGY_DCT16X16;
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n_blocks = 1_usize;
        let zeros = alloc::vec![0.0_f32; n_blocks * 256];
        let weights_one = alloc::vec![1.0_f32; 256];
        // Mask plane: 16x16 (just enough for 1 DCT16x16 block).
        let mask = alloc::vec![1.0_f32; 16 * 16];
        let mask_row_base = alloc::vec![0_u32; n_blocks];

        let costs = estimate_entropy_full_strategy_batch_gpu(
            &enc,
            &zeros,
            &zeros,
            &zeros,
            RAW_STRATEGY_DCT16X16,
            &weights_one,
            &weights_one,
            &weights_one,
            &weights_one,
            &weights_one,
            &weights_one,
            1.0,
            1.0,
            1.0,
            0,
            0,
            &mask,
            &mask_row_base,
            16,
            COEFF_DOMAIN_CONSTANTS,
            1.0,
            CostMode::Simple,
        );
        assert_eq!(costs.len(), n_blocks);
        for &c in &costs {
            assert!(
                c.abs() < 1e-3,
                "DCT16x16 zero-input cost should be ~0, got {c}"
            );
        }
    }

    /// estimate_entropy_full_dct8_batch_persistent must produce the
    /// same per-block cost grid as the non-persistent variant on
    /// realistic input.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_estimate_entropy_full_dct8_batch_persistent_matches_non_persistent() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        // 4×4 grid of 8×8 blocks → 32×32 plane (fits the 32×32 mask).
        let pw = 32_u32;
        let ph = 32_u32;
        let n_blocks = ((pw / 8) * (ph / 8)) as usize;

        // Synthetic XYB-like inputs (block-major layout, 64 floats per block).
        let mk = |seed: f32| -> alloc::vec::Vec<f32> {
            (0..n_blocks * 64)
                .map(|i| 0.05 * (i as f32 * seed).sin() + 0.5)
                .collect()
        };
        let bx = mk(0.013);
        let by = mk(0.017);
        let bb = mk(0.021);

        let weights_x = [0.8_f32; 64];
        let weights_y = [1.0_f32; 64];
        let weights_b = [1.1_f32; 64];
        let inv_x = [1.0_f32 / 0.8; 64];
        let inv_y = [1.0_f32; 64];
        let inv_b = [1.0_f32 / 1.1; 64];

        // Mask plane: 32×32 with smooth variation.
        let mask: alloc::vec::Vec<f32> = (0..(pw * ph) as usize)
            .map(|i| 0.3 + 0.2 * (i as f32 * 0.011).sin())
            .collect();
        let mask_row_base: alloc::vec::Vec<u32> = (0..n_blocks)
            .map(|b| {
                let by_i = (b / (pw as usize / 8)) as u32;
                let bx_i = (b % (pw as usize / 8)) as u32;
                by_i * 8 * pw + bx_i * 8
            })
            .collect();

        let scaled = COEFF_DOMAIN_CONSTANTS;
        let entropy_mul = 0.8;
        let mode = CostMode::Upstream { quant_for_coeffs: 0.7 };
        let costs_a = estimate_entropy_full_dct8_batch_gpu(
            &enc,
            &bx, &by, &bb,
            &weights_x, &weights_y, &weights_b,
            &inv_x, &inv_y, &inv_b,
            0.7, 0.7, 0.7, 0, 0,
            &mask, &mask_row_base, pw,
            scaled, entropy_mul, mode,
        );

        // Persistent path: pre-upload pixel blocks and mask to GPU.
        let g_bx = enc.upload_blocks(&bx, n_blocks as u32, 64);
        let g_by = enc.upload_blocks(&by, n_blocks as u32, 64);
        let g_bb = enc.upload_blocks(&bb, n_blocks as u32, 64);
        let g_mask = enc.upload_plane(&mask, pw, ph);
        let costs_b = estimate_entropy_full_dct8_batch_persistent(
            &enc,
            &g_bx, &g_by, &g_bb,
            &weights_x, &weights_y, &weights_b,
            &inv_x, &inv_y, &inv_b,
            0.7, 0.7, 0.7, 0, 0,
            &g_mask, &mask_row_base,
            scaled, entropy_mul, mode,
        );

        assert_eq!(costs_a.len(), costs_b.len());
        for (i, (a, b)) in costs_a.iter().zip(&costs_b).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "cost[{i}] differs: persistent={b} vs non={a} (delta={})",
                (a - b).abs()
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_estimate_entropy_full_dct8_batch_gpu_zero_input() {
        // All-zero pixels → zero DCT → zero entropy + zero pixel loss
        // → zero per-block cost. Smoke test that the orchestrator
        // composes correctly end-to-end.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n_blocks = 4_usize;
        let zeros = alloc::vec![0.0_f32; n_blocks * 64];
        let weights_one = [1.0_f32; 64];
        // mask plane is 16x16 (= 2x2 blocks) with all 1.0
        let mask = alloc::vec![1.0_f32; 16 * 16];
        let mask_row_base: alloc::vec::Vec<u32> = (0..n_blocks)
            .map(|b| {
                let by = (b / 2) as u32;
                let bx = (b % 2) as u32;
                by * 8 * 16 + bx * 8
            })
            .collect();

        let costs = estimate_entropy_full_dct8_batch_gpu(
            &enc,
            &zeros,
            &zeros,
            &zeros,
            &weights_one,
            &weights_one,
            &weights_one,
            &weights_one,
            &weights_one,
            &weights_one,
            1.0,
            1.0,
            1.0,
            0,
            0,
            &mask,
            &mask_row_base,
            16,
            COEFF_DOMAIN_CONSTANTS,
            1.0,
            CostMode::Simple,
        );
        assert_eq!(costs.len(), n_blocks);
        for &c in &costs {
            assert!(c.abs() < 1e-3, "cost should be ~0 on zero input, got {c}");
        }
    }

    #[test]
    fn test_ceil_log2_nonzero() {
        assert_eq!(ceil_log2_nonzero(0), 0);
        assert_eq!(ceil_log2_nonzero(1), 1); // 1 needs 1 bit
        assert_eq!(ceil_log2_nonzero(2), 2);
        assert_eq!(ceil_log2_nonzero(3), 2);
        assert_eq!(ceil_log2_nonzero(4), 3);
        assert_eq!(ceil_log2_nonzero(7), 3);
        assert_eq!(ceil_log2_nonzero(8), 4);
        assert_eq!(ceil_log2_nonzero(255), 8);
        assert_eq!(ceil_log2_nonzero(256), 9);
    }

    #[test]
    fn test_nzeros_bits_term_zero_input() {
        // num_nzeros = 0 → nbits = ceil_log2(1) + 1 = 1 + 1 = 2
        // → entry = ceil_log2(2 + 17) + 2 = ceil_log2(19) + 2 = 5 + 2 = 7
        // → 7 * k_zeros_mul
        let v = nzeros_bits_term(0, 1.0);
        // ceil_log2_nonzero(1) = 1, so nbits = 2, ceil_log2_nonzero(19) = 5
        // term = 5 + 2 = 7
        assert!((v - 7.0).abs() < 1e-6, "got {v}");
    }

    #[test]
    fn test_nzeros_bits_term_typical() {
        // num_nzeros = 10 → nbits = ceil_log2(11) + 1 = 4 + 1 = 5
        // → ceil_log2(5+17) + 5 = ceil_log2(22) + 5 = 5 + 5 = 10
        let v = nzeros_bits_term(10, 1.0);
        assert!((v - 10.0).abs() < 1e-6, "got {v}");
    }

    #[test]
    fn test_per_block_upstream_cost_zero_loss_zero_nzeros() {
        // entropy_X = entropy_Y = entropy_B = 5.0 each
        // nzeros = 0 each → nzeros_term = 7 * k_zeros_mul each (3 channels)
        // pixel_loss = 0 → loss_scalar = 0
        // entropy = (5 + 5 + 5 + 3 * 7 * k_zeros_mul) * entropy_mul
        let entropy = vec![5.0_f32, 5.0];
        let nzeros = vec![0.0_f32, 0.0];
        let zero_loss = vec![0.0_f64, 0.0];
        let costs = per_block_upstream_cost(
            &entropy,
            &entropy,
            &entropy,
            &nzeros,
            &nzeros,
            &nzeros,
            &zero_loss,
            1.0,              // entropy_mul
            (10.0, 5.0, 1.0), // (info_loss_mul, cost_delta, zeros_mul)
            1.0,              // quant_for_coeffs
            64,               // DCT8 block_pixel_count
        );
        // Expected: (15.0 + 21.0) * 1.0 + 10.0 * 0 = 36.0
        assert!((costs[0] - 36.0).abs() < 1e-3, "got {}", costs[0]);
        assert!((costs[1] - 36.0).abs() < 1e-3);
    }

    #[test]
    fn test_per_block_upstream_cost_block_pixel_count_scaling() {
        // Same loss with different block_pixel_count → different
        // loss_scalar = (loss/n)^(1/8) * n / quant. For n=256
        // (DCT16x16) the scalar is sqrt(sqrt(sqrt(loss/256))) * 256
        // = (4)^(-1/8) × 4 × what we'd get for n=64.
        let entropy = vec![0.0_f32];
        let nzeros = vec![0.0_f32];
        let loss = vec![1.0_f64];
        let cost_dct8 = per_block_upstream_cost(
            &entropy,
            &entropy,
            &entropy,
            &nzeros,
            &nzeros,
            &nzeros,
            &loss,
            1.0,
            (1.0, 1.0, 0.0),
            1.0,
            64,
        );
        let cost_dct16 = per_block_upstream_cost(
            &entropy,
            &entropy,
            &entropy,
            &nzeros,
            &nzeros,
            &nzeros,
            &loss,
            1.0,
            (1.0, 1.0, 0.0),
            1.0,
            256,
        );
        // n=256: loss_scalar = (1/256)^(1/8) * 256 = 0.5612... * 256 ≈ 143.7
        // n=64:  loss_scalar = (1/64)^(1/8)  * 64  = 0.6086... * 64  ≈ 38.95
        // Ratio ~ 3.69; cost_dct16 should be > cost_dct8.
        assert!(
            cost_dct16[0] > cost_dct8[0],
            "n=256 cost {} should be > n=64 cost {}",
            cost_dct16[0],
            cost_dct8[0]
        );
    }

    #[test]
    fn test_per_block_upstream_cost_nonzero_loss() {
        // Small but non-zero pixel loss. loss_scalar is the 8th-root,
        // which makes a small loss into a sizable contribution.
        let entropy = vec![0.0_f32];
        let nzeros = vec![0.0_f32];
        let loss = vec![1.0_f64]; // total pixel loss = 1
        let costs = per_block_upstream_cost(
            &entropy,
            &entropy,
            &entropy,
            &nzeros,
            &nzeros,
            &nzeros,
            &loss,
            1.0,
            (1.0, 1.0, 0.0), // info_loss_mul=1, zeros_mul=0 to isolate loss term
            1.0,
            64,
        );
        // entropy = 0 + 0_zero_term*3 = 0, entropy *= 1.0 → 0
        // loss_scalar = (1/64).sqrt().sqrt().sqrt() * 64 / 1
        //            = (0.015625)^(1/8) * 64 = 0.6086... * 64 ≈ 38.95
        // entropy += 1.0 * 38.95
        assert!(
            costs[0] > 30.0 && costs[0] < 50.0,
            "loss-dominated cost {} not in range",
            costs[0]
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_estimate_entropy_full_dct8_batch_gpu_upstream_mode() {
        // Same zero input as the Simple-mode test but with CostMode::Upstream.
        // For zero input: zero entropy, zero pixel loss, zero nzeros.
        // Per upstream's nzeros bits term: f(0) = ceil_log2(1)+1 + ceil_log2(2+17)
        //                                       = 2 + 5 = 7. Multiplied by k_zeros_mul.
        // For COEFF_DOMAIN_CONSTANTS k_zeros_mul = 7.565..., per-channel
        // contribution = 7 * 7.565 ≈ 52.96. ×3 channels = 158.87.
        // entropy_mul = 1.0, info_loss_mul × loss = 0.
        // Final: ≈ 158.87.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n_blocks = 4_usize;
        let zeros = alloc::vec![0.0_f32; n_blocks * 64];
        let weights_one = [1.0_f32; 64];
        let mask = alloc::vec![1.0_f32; 16 * 16];
        let mask_row_base: alloc::vec::Vec<u32> = (0..n_blocks)
            .map(|b| {
                let by = (b / 2) as u32;
                let bx = (b % 2) as u32;
                by * 8 * 16 + bx * 8
            })
            .collect();

        let costs = estimate_entropy_full_dct8_batch_gpu(
            &enc,
            &zeros,
            &zeros,
            &zeros,
            &weights_one,
            &weights_one,
            &weights_one,
            &weights_one,
            &weights_one,
            &weights_one,
            1.0,
            1.0,
            1.0,
            0,
            0,
            &mask,
            &mask_row_base,
            16,
            COEFF_DOMAIN_CONSTANTS,
            1.0,
            CostMode::Upstream {
                quant_for_coeffs: 1.0,
            },
        );
        assert_eq!(costs.len(), n_blocks);
        let expected_per_channel = 7.0 * COEFF_DOMAIN_CONSTANTS.2;
        let expected_total = expected_per_channel * 3.0;
        for &c in &costs {
            assert!(
                (c - expected_total).abs() < 1e-2,
                "expected ~{expected_total:.3}, got {c}"
            );
        }
    }

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
    fn test_x_multiblock_weight_singleton_is_one() {
        assert_eq!(x_multiblock_weight(0), 1.0);
        assert_eq!(x_multiblock_weight(1), 1.0);
    }

    #[test]
    fn test_x_multiblock_weight_grows_then_caps() {
        // num_blocks=2 → 1 + 2/8 = 1.25
        assert!((x_multiblock_weight(2) - 1.25).abs() < 1e-6);
        // num_blocks=4 → 1 + 4/8 = 1.5
        assert!((x_multiblock_weight(4) - 1.5).abs() < 1e-6);
        // num_blocks=8 → 1 + 8/8 = 2.0
        assert!((x_multiblock_weight(8) - 2.0).abs() < 1e-6);
        // num_blocks=16 → 1 + 16/8 = 3.0
        assert!((x_multiblock_weight(16) - 3.0).abs() < 1e-6);
        // num_blocks=24 → 1 + 24/8 = 4.0
        assert!((x_multiblock_weight(24) - 4.0).abs() < 1e-6);
        // num_blocks=32 → 1 + min(4, 3) = 4.0 (cap is on the SECOND term)
        // Wait: 32/8 = 4.0, min(4.0, 3.0) = 3.0, so w = 1 + 3 = 4.0
        assert!((x_multiblock_weight(32) - 4.0).abs() < 1e-6);
        // num_blocks=64 → 64/8 = 8, min(8, 3) = 3, w = 4
        assert!((x_multiblock_weight(64) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_x_multiblock_weight_to_loss_no_op_for_singleton() {
        let mut loss = vec![1.0_f64, 2.0, 3.0];
        let original = loss.clone();
        apply_x_multiblock_weight_to_loss(&mut loss, 1);
        assert_eq!(loss, original);
    }

    #[test]
    fn test_apply_x_multiblock_weight_to_loss_scales_by_w() {
        let mut loss = vec![1.0_f64, 2.0, 3.0];
        // num_blocks=4 → w = 1.5
        apply_x_multiblock_weight_to_loss(&mut loss, 4);
        let expected = vec![1.5_f64, 3.0, 4.5];
        for i in 0..3 {
            assert!((loss[i] - expected[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_apply_x_multiblock_weight_to_entropy_no_op_for_singleton() {
        let mut e = vec![10.0_f32, 20.0];
        apply_x_multiblock_weight_to_entropy(&mut e, 1);
        assert_eq!(e, vec![10.0_f32, 20.0]);
    }

    #[test]
    fn test_apply_x_multiblock_weight_to_entropy_scales_by_w() {
        let mut e = vec![10.0_f32, 20.0];
        // num_blocks=8 → w = 2.0
        apply_x_multiblock_weight_to_entropy(&mut e, 8);
        assert!((e[0] - 20.0).abs() < 1e-6);
        assert!((e[1] - 40.0).abs() < 1e-6);
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
            RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8, RAW_STRATEGY_DCT8X4,
            RAW_STRATEGY_IDENTITY,
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
            RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16,
            RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32,
            RAW_STRATEGY_DCT32X64, RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64,
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
    fn test_compute_scaled_constants_matches_upstream() {
        // G5.1 parity: compare against upstream's
        // jxl_encoder::__internals::compute_scaled_constants_free
        // (gated by the __internals cargo feature). Spot-check a
        // grid of distance × bases to catch any algorithmic drift.
        for &distance in &[0.5_f32, 1.0, 2.0, 5.0, 10.0] {
            for &bases in &[
                (1.0_f32, 2.0, 3.0),
                (138.0, 5.335_918_5, 7.565_053_4), // COEFF_DOMAIN_CONSTANTS-shape
                (0.1, 0.5, 1.7),
            ] {
                let mine = compute_scaled_constants(distance, bases);
                let theirs =
                    jxl_encoder::__internals::compute_scaled_constants_free(distance, bases);
                assert!(
                    (mine.0 - theirs.0).abs() < 1e-3,
                    "d={distance} bases={bases:?} info_loss: mine={} theirs={}",
                    mine.0,
                    theirs.0
                );
                assert!(
                    (mine.1 - theirs.1).abs() < 1e-3,
                    "d={distance} bases={bases:?} cost_delta: mine={} theirs={}",
                    mine.1,
                    theirs.1
                );
                assert!(
                    (mine.2 - theirs.2).abs() < 1e-3,
                    "d={distance} bases={bases:?} zeros_mul: mine={} theirs={}",
                    mine.2,
                    theirs.2
                );
            }
        }
    }

    #[test]
    fn test_compute_scaled_constants_d1_no_scale() {
        // ratio = 1.0 at distance == 1.0 → bases echo back.
        let (info, cost, zeros) = compute_scaled_constants(1.0, (1.0, 2.0, 3.0));
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
