// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/quantize.rs (BSD-3-Clause via libjxl
// + AGPL/commercial), reshaped from per-block CPU loop to per-channel
// batched GPU launch.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted DCT8 quantization with dead-zone thresholding.
//!
//! ## Reshape vs upstream `VarDctEncoder::quantize_ac_block`
//!
//! Upstream takes ONE block at a time, dispatches to a SIMD kernel for
//! the DCT8 fast path (`quantize_block_dct8`), and writes into a
//! `Vec<[i32; 64]>` slot for that block.
//!
//! Our GPU shape: feed ALL DCT8 blocks for one channel as a contiguous
//! `Vec<f32>` of length `num_blocks * 64`. One launch quantizes
//! everything. The caller pre-groups blocks by channel + strategy
//! (DCT8 only here; non-DCT8 strategies stay on CPU until we add
//! their kernels).
//!
//! ## What this fork covers
//!
//! - `default_thresholds` — pure scalar copy. Computes the 4-quadrant
//!   dead-zone threshold table for a channel + coverage shape.
//! - `quantize_dct8_blocks_gpu` — batched DCT8 quantize. One launch
//!   per channel.
//! - `quantize_dct8_xyb_gpu` — convenience helper that runs all 3
//!   channels (X, Y, B) sequentially. Three GPU launches; future
//!   fusion = a 3-channel kernel that does X+Y+B in one launch.
//!
//! Larger-strategy quantize (DCT16+ family) is now wired via
//! [`quantize_blocks_gpu`] which dispatches to either the DCT8 fast
//! path or the generic `quantize_large` kernel based on the
//! `(grid_width, grid_height, llf_x, llf_y)` strategy descriptor.
//!
//! Not yet covered (no GPU kernel for these):
//! - `adjust_quant_block_ac` heuristics (sparse-block boost, HF corner
//!   increase, flatness detection, etc.) — stay on CPU
//! - Error diffusion in zigzag order — stay on CPU (libjxl never
//!   uses ED in QuantizeBlockAC anyway, despite accepting the param)
//! - AFV/IDENTITY/DCT2X2 use the 64-coeff DCT8 quant path (their
//!   coefficient layout matches), so they go through `quantize_dct8`
//!   not `quantize_large`.

use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

/// Default dead-zone thresholds for one channel + coverage shape.
///
/// Pure scalar copy of upstream
/// `VarDctEncoder::default_thresholds`. Returns the 4-quadrant
/// threshold array (TL, TR, BL, BR).
///
/// Y (`c=1`): `{0.56, 0.62, 0.62, 0.62}`
/// X/B (`c=0/2`): `{0.58, 0.62, 0.62, 0.62}`, with multi-block
/// reduction `-0.00744 * covered_x*covered_y` (floored at 0.5)
/// when `covered_x * covered_y >= 4`.
///
/// ```
/// use jxl_encoder_gpu::forks::quantize::default_thresholds;
///
/// // Y channel always {0.56, 0.62, 0.62, 0.62} regardless of coverage.
/// assert_eq!(default_thresholds(1, 1, 1), [0.56, 0.62, 0.62, 0.62]);
/// assert_eq!(default_thresholds(1, 8, 8), [0.56, 0.62, 0.62, 0.62]);
///
/// // X/B single-block: {0.58, 0.62, 0.62, 0.62}.
/// assert_eq!(default_thresholds(0, 1, 1), [0.58, 0.62, 0.62, 0.62]);
/// assert_eq!(default_thresholds(2, 1, 1), [0.58, 0.62, 0.62, 0.62]);
///
/// // X/B multi-block (coverage>=4): -0.00744 per coverage unit.
/// // 2x2 = 4 blocks → adjustment = 0.00744 * 4 = 0.02976.
/// let t = default_thresholds(0, 2, 2);
/// assert!((t[0] - (0.58 - 0.02976)).abs() < 1e-5);
/// assert!((t[1] - (0.62 - 0.02976)).abs() < 1e-5);
///
/// // Very large coverage clamps each threshold at 0.5.
/// let t = default_thresholds(0, 16, 16);
/// for &v in &t {
///     assert!(v >= 0.5);
/// }
/// ```
pub fn default_thresholds(c: usize, covered_x: usize, covered_y: usize) -> [f32; 4] {
    let mut thres = if c == 1 {
        [0.56_f32, 0.62, 0.62, 0.62]
    } else {
        [0.58_f32, 0.62, 0.62, 0.62]
    };
    if c != 1 && covered_x * covered_y >= 4 {
        let adj = 0.00744 * (covered_x * covered_y) as f32;
        for t in thres.iter_mut() {
            *t -= adj;
            if *t < 0.5 {
                *t = 0.5;
            }
        }
    }
    thres
}

/// Batched DCT8 quantize on GPU for one channel.
///
/// - `coeffs`: `num_blocks * 64` floats — DCT8 output coefficients in
///   8×8 row-major order per block.
/// - `weights`: `num_blocks * 64` floats — per-coefficient inverse
///   quant matrix entries. Same layout as `coeffs`.
/// - `qac_qm`: `num_blocks` floats — per-block scale (qac × qm_mul).
/// - `thresholds`: 4-quadrant dead-zone thresholds (see
///   [`default_thresholds`]).
///
/// Returns `num_blocks * 64` quantized i32 values.
pub fn quantize_dct8_blocks_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    coeffs: &[f32],
    weights: &[f32],
    qac_qm: &[f32],
    thresholds: &[f32; 4],
) -> Vec<i32> {
    enc.quantize_dct8_blocks(coeffs, weights, qac_qm, thresholds)
}

/// 3-channel batched DCT8 quantize. Runs X, Y, B sequentially with
/// channel-specific thresholds.
///
/// `coeffs_*`, `weights_*`, `qac_qm_*` follow the same shape as
/// [`quantize_dct8_blocks_gpu`]. `covered_x` / `covered_y` describe
/// the block coverage shape (1, 1) for plain DCT8 — used to compute
/// the channel-specific dead-zone thresholds.
#[allow(clippy::too_many_arguments)]
pub fn quantize_dct8_xyb_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    coeffs_x: &[f32],
    coeffs_y: &[f32],
    coeffs_b: &[f32],
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    qac_qm_x: &[f32],
    qac_qm_y: &[f32],
    qac_qm_b: &[f32],
    covered_x: usize,
    covered_y: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let thr_x = default_thresholds(0, covered_x, covered_y);
    let thr_y = default_thresholds(1, covered_x, covered_y);
    let thr_b = default_thresholds(2, covered_x, covered_y);
    let qx = enc.quantize_dct8_blocks(coeffs_x, weights_x, qac_qm_x, &thr_x);
    let qy = enc.quantize_dct8_blocks(coeffs_y, weights_y, qac_qm_y, &thr_y);
    let qb = enc.quantize_dct8_blocks(coeffs_b, weights_b, qac_qm_b, &thr_b);
    (qx, qy, qb)
}

/// Strategy-aware quantize. Dispatches to `quantize_dct8` or
/// `quantize_large` per the (`grid_width`, `grid_height`, `llf_x`,
/// `llf_y`) tuple — DCT8 (8/8/1/1) takes the fast path, everything
/// else takes the generic large-block path.
///
/// Mirrors upstream's `quantize_large_scalar` shape; useful for
/// quantizing the coefficient outputs of DCT16/16x8/8x16/32/32x16/
/// 16x32/64/64x32/32x64 transforms after the forward DCT.
#[allow(clippy::too_many_arguments)]
pub fn quantize_blocks_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    coeffs: &[f32],
    weights: &[f32],
    qac_qm: &[f32],
    thresholds: &[f32; 4],
    grid_width: u32,
    grid_height: u32,
    llf_x: u32,
    llf_y: u32,
) -> Vec<i32> {
    if grid_width == 8 && grid_height == 8 && llf_x == 1 && llf_y == 1 {
        // DCT8 fast path.
        enc.quantize_dct8_blocks(coeffs, weights, qac_qm, thresholds)
    } else {
        enc.quantize_large_blocks(
            coeffs, weights, qac_qm, thresholds, grid_width, grid_height, llf_x, llf_y,
        )
    }
}

/// Per-block coefficient statistics consumed by the AdjustQuantBlockAC
/// heuristics. Mirrors the locals computed in upstream's pre-scan loop
/// at `jxl_encoder::vardct::quantize.rs:178-220`.
///
/// All four-element arrays are indexed by `hfix = 2 * (y >= h/2) +
/// (x >= w/2)` — the high-frequency quadrant index.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdjustQuantBlockStats {
    pub sum_of_highest_freq: f32,
    pub sum_of_error: f32,
    pub sum_of_vals: f32,
    pub hf_nonzeros: [f32; 4],
    pub hf_max_error: [f32; 4],
}

/// Pre-scan over non-LLF coefficients of a single transform block,
/// computing the five statistics consumed by AdjustQuantBlockAC's
/// heuristics B / C / D / E / F (heuristic A only depends on
/// `xsize * ysize`, no coefficient pre-scan needed).
///
/// Pure-CPU port — bit-for-bit equivalent to upstream's pre-scan. The
/// `quant` and `qm_multiplier` are the per-block scalars; `qac =
/// scale * quant` and `qm_multiplier = x_qm_mul / 1.0 / b_qm_mul` per
/// channel as in upstream.
///
/// `block_coeffs` and `weights` MUST be `block_width * block_height`
/// floats each, in the same row-major layout the rest of the encoder
/// uses. `xsize`/`ysize` are the LLF coverage in 8×8 blocks (`cx`/`cy`
/// at the call site, e.g. 1×1 for DCT8, 2×2 for DCT16×16, 8×8 for
/// DCT64×64).
///
/// Returns `None` for the libjxl-defined "partial block kinds" — the
/// strategies whose AdjustQuantBlockAC body returns `(0, 0.0, 0.0, 0)`
/// without ever pre-scanning. Caller can match upstream by treating
/// `None` exactly the same way.
#[allow(clippy::too_many_arguments)]
pub fn adjust_quant_prescan(
    block_coeffs: &[f32],
    weights: &[f32],
    qac: f32,
    qm_multiplier: f32,
    c: usize,
    raw_strategy: u8,
    block_width: usize,
    block_height: usize,
    xsize: usize,
    ysize: usize,
    thresholds: &[f32; 4],
) -> Option<AdjustQuantBlockStats> {
    use crate::forks::transform::{
        RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8,
        RAW_STRATEGY_DCT8X4, RAW_STRATEGY_IDENTITY,
    };
    // Partial block kinds: pre-scan is skipped (matches upstream
    // `kPartialBlockKinds` skip → returns 0 stats). AFV variants are
    // also skipped upstream; the GPU port routes AFV through
    // `forks::afv` separately and never reaches this path with an
    // AFV-coded raw_strategy.
    match raw_strategy {
        RAW_STRATEGY_IDENTITY
        | RAW_STRATEGY_DCT2X2
        | RAW_STRATEGY_DCT4X4
        | RAW_STRATEGY_DCT4X8
        | RAW_STRATEGY_DCT8X4 => return None,
        _ => {}
    }

    debug_assert_eq!(block_coeffs.len(), block_width * block_height);
    debug_assert_eq!(weights.len(), block_width * block_height);

    let mut stats = AdjustQuantBlockStats::default();
    for y in 0..block_height {
        for x in 0..block_width {
            let pos = y * block_width + x;
            // Skip LLF positions.
            if x < xsize && y < ysize {
                continue;
            }
            let hfix = (if y >= block_height / 2 { 2 } else { 0 })
                + (if x >= block_width / 2 { 1 } else { 0 });

            // val = (1/weight) * qac * qm_mul * coeff — matches
            // quantize_coeff_ac formula upstream uses.
            let inv_w = 1.0 / weights[pos];
            let val = block_coeffs[pos] * inv_w * qac * qm_multiplier;
            let v = if val.abs() < thresholds[hfix] {
                0.0
            } else {
                // round-to-even matches libjxl rintf / Highway Round
                let r = (val * 0.5).round() * 2.0;
                let alt = val.round_ties_even();
                debug_assert!((r - alt).abs() <= 1.0); // sanity
                alt
            };
            let error = (val - v).abs();
            stats.sum_of_error += error;
            stats.sum_of_vals += v.abs();

            if c == 1 && v == 0.0 && stats.hf_max_error[hfix] < error {
                stats.hf_max_error[hfix] = error;
            }
            if v != 0.0 {
                stats.hf_nonzeros[hfix] += v.abs();
                let in_corner = y >= 7 * ysize && x >= 7 * xsize;
                let on_border = y == block_height - 1 || x == block_width - 1;
                let in_larger_corner = x >= 4 * xsize && y >= 4 * ysize;
                if in_corner || (on_border && in_larger_corner) {
                    stats.sum_of_highest_freq += val.abs();
                }
            }
        }
    }

    Some(stats)
}

/// AdjustQuantBlockAC heuristic A — threshold reduction for large
/// transforms. Applied unconditionally before the pre-scan in
/// upstream's `adjust_quant_block_ac`.
///
/// Behavior (matches upstream lines 167-176):
/// - If `xsize > 1 || ysize > 1` (any covered_blocks > 1×1):
///   - Compute `adj = clamp(0.003 * xsize * ysize, 0.0, 0.08)`.
///   - For each of the 4 thresholds: `t = max(t - adj, 0.54)`.
///   - Returns `true` (heuristic fired).
/// - Otherwise: thresholds unchanged, returns `false`.
///
/// The returned bool corresponds to `heuristics_fired & 0x01` in
/// upstream's bitfield convention.
pub fn apply_heuristic_a_thresholds(
    thresholds: &mut [f32; 4],
    xsize: usize,
    ysize: usize,
) -> bool {
    if xsize > 1 || ysize > 1 {
        let adj = (0.003 * (xsize * ysize) as f32).clamp(0.0, 0.08);
        for t in thresholds.iter_mut() {
            *t -= adj;
            if *t < 0.54 {
                *t = 0.54;
            }
        }
        true
    } else {
        false
    }
}

/// `quant` cap matching upstream `QUANT_MAX = 256`. The heuristics
/// clamp at `QUANT_MAX - 1` (255) when their increment would
/// otherwise exceed.
pub const QUANT_MAX: i32 = 256;

/// AdjustQuantBlockAC heuristic F — activity-based quant reduction.
/// Mirrors upstream lines 342-372 (the final block in
/// `adjust_quant_block_ac`).
///
/// Always runs (no skip condition). Computes `activity` from the
/// minimum of the four `hf_nonzeros` quadrants:
/// - If `min(hf_nonzeros) < 15 * (xsize * ysize)`:
///   `activity = (min_nonzeros + div/2) / div` (integer division;
///   matches libjxl's overflow-safe form from commit ae5cb19 — uses
///   float min then i32 cast to avoid HDR overflow).
/// - Otherwise: `activity = 15` (capped).
///
/// Then `quant -= activity`, clamped at `max(quant_orig / 2, 4)`.
/// For Y channel only, the upper 3 thresholds get `t += 0.01 *
/// activity`.
///
/// Returns `(fired, activity)` where `fired` is the bit
/// `0x20` upstream — true iff `quant` actually changed (i.e.,
/// `activity > 0` and the floor didn't bite). `activity` is the
/// same value upstream returns alongside `heuristics_fired`.
pub fn apply_heuristic_f_activity(
    quant: &mut i32,
    thresholds: &mut [f32; 4],
    stats: &AdjustQuantBlockStats,
    c: usize,
    xsize: usize,
    ysize: usize,
) -> (bool, i32) {
    let div = (xsize * ysize) as i32;
    let min_hf_nonzeros = stats.hf_nonzeros[0]
        .min(stats.hf_nonzeros[1])
        .min(stats.hf_nonzeros[2])
        .min(stats.hf_nonzeros[3]);
    let activity = if min_hf_nonzeros < 15.0 * div as f32 {
        ((min_hf_nonzeros as i32) + div / 2) / div
    } else {
        15
    };
    let orig = *quant;
    let orig_qp_limit = (orig / 2).max(4);
    let mut qp = orig - activity;
    if c == 1 {
        for t in thresholds[1..4].iter_mut() {
            *t += 0.01 * activity as f32;
        }
    }
    if qp < orig_qp_limit {
        qp = orig_qp_limit;
    }
    let fired = qp != orig;
    *quant = qp;
    (fired, activity)
}

/// AdjustQuantBlockAC heuristic E — large-transform error correction.
/// Mirrors upstream lines 282-340.
///
/// Only fires for the DCT16+ family (DCT16x16, DCT32x32, DCT16x8,
/// DCT8x16, DCT64x64, DCT64x32, DCT32x64, DCT32x16, DCT16x32). Uses
/// per-strategy K_MUL1/K_MUL2 tables (4 rows × 3 channels) and
/// K_QUANT_NORMALIZER to compute a `threshold = K_MUL1 * area +
/// K_MUL2 * norm_vals`. When `norm_error > threshold`, increments
/// `quant` by `clamp((norm_error / threshold) as i32, 0, 2)`.
///
/// Returns `true` when the heuristic fired (bit `0x10` upstream).
///
/// Strategy → table-row index mapping (matching upstream):
/// - DCT16X16 → 0
/// - DCT32X16, DCT16X32 → 1
/// - DCT32X32 → 2
/// - DCT16X8, DCT8X16, DCT64X*, DCT*X64 → 3 (default for "large but
///   not in the named buckets")
pub fn apply_heuristic_e_large_transform(
    quant: &mut i32,
    stats: &AdjustQuantBlockStats,
    c: usize,
    raw_strategy: u8,
    xsize: usize,
    ysize: usize,
) -> bool {
    use crate::forks::transform::{
        RAW_STRATEGY_DCT16X16, RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT16X8,
        RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64,
        RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64, RAW_STRATEGY_DCT8X16,
    };

    #[allow(clippy::excessive_precision)]
    const K_MUL1: [[f64; 3]; 4] = [
        [0.220_806_157_538_484_04, 0.457_974_798_242_620_11, 0.298_592_350_959_779_65],
        [0.701_094_865_102_868_34, 0.161_852_813_055_126_39, 0.143_876_917_300_354_73],
        [0.114_985_964_456_218_64, 0.446_568_404_410_277_0, 0.105_876_582_151_490_48],
        [0.468_496_652_644_093_96, 0.412_390_779_377_819_54, 0.088_667_407_767_185_44],
    ];
    #[allow(clippy::excessive_precision)]
    const K_MUL2: [[f64; 3]; 4] = [
        [0.274_502_819_418_222_0, 1.125_576_654_998_500, 0.989_504_591_341_283_9],
        [0.465_216_867_559_828_5, 0.409_458_079_834_558_2, 0.365_818_998_117_513_67],
        [0.280_349_724_247_157_15, 0.918_265_320_192_973_8, 1.558_153_154_305_741_6],
        [0.268_731_181_140_337_28, 0.688_637_123_903_924_84, 1.208_218_540_866_678_6],
    ];
    const K_QUANT_NORMALIZER: f64 = 2.294_270_834_328_472;
    const BLOCK_DIM: usize = 8;

    let is_large = matches!(
        raw_strategy,
        RAW_STRATEGY_DCT16X16
            | RAW_STRATEGY_DCT32X32
            | RAW_STRATEGY_DCT16X8
            | RAW_STRATEGY_DCT8X16
            | RAW_STRATEGY_DCT64X64
            | RAW_STRATEGY_DCT64X32
            | RAW_STRATEGY_DCT32X64
            | RAW_STRATEGY_DCT32X16
            | RAW_STRATEGY_DCT16X32
    );
    if !is_large {
        return false;
    }
    let ix = match raw_strategy {
        RAW_STRATEGY_DCT16X16 => 0,
        RAW_STRATEGY_DCT32X16 | RAW_STRATEGY_DCT16X32 => 1,
        RAW_STRATEGY_DCT32X32 => 2,
        _ => 3,
    };
    let norm_error = stats.sum_of_error as f64 * K_QUANT_NORMALIZER;
    let norm_vals = stats.sum_of_vals as f64 * K_QUANT_NORMALIZER;
    let area = (xsize * ysize * BLOCK_DIM * BLOCK_DIM) as f64;
    let threshold = K_MUL1[ix][c] * area + K_MUL2[ix][c] * norm_vals;
    if norm_error > threshold {
        let step = ((norm_error / threshold) as i32).clamp(0, 2);
        *quant += step;
        if *quant >= QUANT_MAX {
            *quant = QUANT_MAX - 1;
        }
        true
    } else {
        false
    }
}

/// AdjustQuantBlockAC heuristic B — sparse Y-channel handling.
/// Mirrors upstream lines 222-255.
///
/// Only runs for the Y channel (`c == 1`) when the block is sparse
/// (`sum_of_vals * 8 < xsize * ysize`). When fired, may increment
/// `quant` by 1 (if any non-DC quadrant has zero non-zeros AND a
/// per-quadrant `hf_max_error > K_LIMIT[i] = 0.46`), and may set
/// one of the threshold positions based on which quadrant ranges
/// match.
///
/// Returns `true` when the heuristic fired (bit `0x02` in
/// upstream's bitfield), regardless of whether the quant or
/// thresholds actually changed.
///
/// Threshold-setting precedence (upstream's `if/else if/else if`
/// cascade):
/// 1. Quadrant 3 (high-x, high-y) → updates `thresholds[3]`.
/// 2. Otherwise quadrant 1 or 2 (one high coord) → updates
///    `thresholds[1] = thresholds[2]` to the same value.
/// 3. Otherwise quadrant 0 (DC quadrant) → updates `thresholds[0]`.
pub fn apply_heuristic_b_sparse_y(
    quant: &mut i32,
    thresholds: &mut [f32; 4],
    stats: &AdjustQuantBlockStats,
    c: usize,
    xsize: usize,
    ysize: usize,
) -> bool {
    if c != 1 {
        return false;
    }
    if stats.sum_of_vals * 8.0 >= (xsize * ysize) as f32 {
        return false;
    }

    const K_LIMIT: [f64; 4] = [0.46, 0.46, 0.46, 0.46];
    const K_MUL: [f64; 4] = [0.9999, 0.9999, 0.9999, 0.9999];

    let orig_quant = *quant;
    let mut new_quant = *quant;
    for i in 1..4 {
        if stats.hf_nonzeros[i] == 0.0 && (stats.hf_max_error[i] as f64) > K_LIMIT[i] {
            new_quant = orig_quant + 1;
            break;
        }
    }
    *quant = new_quant;

    if stats.hf_nonzeros[3] == 0.0 && (stats.hf_max_error[3] as f64) > K_LIMIT[3] {
        thresholds[3] = (K_MUL[3] * stats.hf_max_error[3] as f64 * new_quant as f64
            / orig_quant as f64) as f32;
    } else if (stats.hf_nonzeros[1] == 0.0 && (stats.hf_max_error[1] as f64) > K_LIMIT[1])
        || (stats.hf_nonzeros[2] == 0.0 && (stats.hf_max_error[2] as f64) > K_LIMIT[2])
    {
        let max_err = stats.hf_max_error[1].max(stats.hf_max_error[2]);
        thresholds[1] =
            (K_MUL[1] * max_err as f64 * new_quant as f64 / orig_quant as f64) as f32;
        thresholds[2] = thresholds[1];
    } else if stats.hf_nonzeros[0] == 0.0 && (stats.hf_max_error[0] as f64) > K_LIMIT[0] {
        thresholds[0] = (K_MUL[0] * stats.hf_max_error[0] as f64 * new_quant as f64
            / orig_quant as f64) as f32;
    }
    true
}

/// AdjustQuantBlockAC heuristic C — high-frequency corner penalty.
/// Mirrors upstream lines 257-269.
///
/// Computes `all = sum(hf_nonzeros) + 1.0`. If
/// `mul[c] * sum_of_highest_freq >= all` (where `mul = [70, 30, 60]`
/// for X/Y/B), increments `quant` by `(mul[c] * sum_of_highest_freq
/// / all) as i32`, clamped to `QUANT_MAX - 1`. Returns `true` when
/// the heuristic fired (bit `0x04` upstream).
///
/// `c` is the channel index (0=X, 1=Y, 2=B). The `mul` per-channel
/// constants are reproduced verbatim from upstream.
pub fn apply_heuristic_c_corner_penalty(
    quant: &mut i32,
    stats: &AdjustQuantBlockStats,
    c: usize,
) -> bool {
    let all = stats.hf_nonzeros[0]
        + stats.hf_nonzeros[1]
        + stats.hf_nonzeros[2]
        + stats.hf_nonzeros[3]
        + 1.0;
    let mul = [70.0_f32, 30.0, 60.0];
    if mul[c] * stats.sum_of_highest_freq >= all {
        let bump = (mul[c] * stats.sum_of_highest_freq / all) as i32;
        *quant += bump;
        if *quant >= QUANT_MAX {
            *quant = QUANT_MAX - 1;
        }
        true
    } else {
        false
    }
}

/// AdjustQuantBlockAC heuristic D — DCT8 flatness detection.
/// Mirrors upstream lines 271-280.
///
/// Only fires for `raw_strategy == RAW_STRATEGY_DCT` (8×8 transform).
/// If `sum(hf_nonzeros) < 11.0`, increments `quant` by 1 (clamped
/// to `QUANT_MAX - 1`). Returns `true` when the heuristic fired
/// (bit `0x08` upstream).
///
/// The intuition: a very flat DCT8 block (few non-zero AC values)
/// can blur visibly with insufficient quantization; bumping `quant`
/// reduces the chance of blocking artifacts.
pub fn apply_heuristic_d_dct8_flatness(
    quant: &mut i32,
    stats: &AdjustQuantBlockStats,
    raw_strategy: u8,
) -> bool {
    use crate::forks::transform::RAW_STRATEGY_DCT;
    if raw_strategy != RAW_STRATEGY_DCT {
        return false;
    }
    let sum = stats.hf_nonzeros[0]
        + stats.hf_nonzeros[1]
        + stats.hf_nonzeros[2]
        + stats.hf_nonzeros[3];
    if sum < 11.0 {
        *quant += 1;
        if *quant >= QUANT_MAX {
            *quant = QUANT_MAX - 1;
        }
        true
    } else {
        false
    }
}

/// Per-block AdjustQuantBlockAC return value, mirroring upstream's
/// 4-tuple `(u8, f32, f32, i32)` return:
/// - `heuristics_fired`: bitfield with bit `1 << k` for heuristic
///   k. Bit 0 = A, 1 = B, 2 = C, 3 = D, 4 = E, 5 = F.
/// - `sum_of_vals` / `sum_of_error`: per-block stats from the
///   pre-scan (zero when the strategy is a "partial block kind"
///   that skips the pre-scan entirely).
/// - `activity`: heuristic F's activity value (always set, even
///   if F didn't change quant — matches upstream).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdjustQuantOutcome {
    pub heuristics_fired: u8,
    pub sum_of_vals: f32,
    pub sum_of_error: f32,
    pub activity: i32,
}

/// Compose pre-scan + heuristics A-F into the full per-block
/// AdjustQuantBlockAC orchestration. Mirrors upstream
/// `jxl_encoder::vardct::quantize::adjust_quant_block_ac`.
///
/// Order matches upstream exactly:
/// 1. **A** (threshold reduction for `xsize > 1 || ysize > 1`)
/// 2. Pre-scan over non-LLF coefficients
/// 3. **B** (sparse Y handling, c=1 only)
/// 4. **C** (HF corner penalty)
/// 5. **D** (DCT8 flatness)
/// 6. **E** (large-transform error correction)
/// 7. **F** (activity-based reduction)
///
/// For "partial block kinds" (IDENTITY, DCT2X2, DCT4X4, DCT4X8,
/// DCT8X4) upstream returns `(0, 0.0, 0.0, 0)` immediately, never
/// running A-F. We match that exactly via `adjust_quant_prescan`'s
/// `None` return.
///
/// `thresholds` and `quant` are modified in place; the
/// `AdjustQuantOutcome` is returned for stats/logging.
#[allow(clippy::too_many_arguments)]
pub fn adjust_quant_block_ac_host(
    block_coeffs: &[f32],
    weights: &[f32],
    qac: f32,
    qm_multiplier: f32,
    c: usize,
    raw_strategy: u8,
    block_width: usize,
    block_height: usize,
    xsize: usize,
    ysize: usize,
    thresholds: &mut [f32; 4],
    quant: &mut i32,
) -> AdjustQuantOutcome {
    let mut fired: u8 = 0;

    // (A) — runs before the pre-scan in upstream.
    if apply_heuristic_a_thresholds(thresholds, xsize, ysize) {
        fired |= 0x01;
    }

    // Pre-scan. Returns None for partial block kinds — upstream
    // skips A-F entirely on those (heuristics_fired stays 0 because
    // we early-return *before* running A; but the order in upstream
    // is: skip → return (0,0,0,0). To match that semantics for the
    // partial-block kinds, fold A's output away and return zeros.
    let stats = match adjust_quant_prescan(
        block_coeffs,
        weights,
        qac,
        qm_multiplier,
        c,
        raw_strategy,
        block_width,
        block_height,
        xsize,
        ysize,
        thresholds,
    ) {
        Some(s) => s,
        None => {
            return AdjustQuantOutcome::default();
        }
    };

    // (B)
    if apply_heuristic_b_sparse_y(quant, thresholds, &stats, c, xsize, ysize) {
        fired |= 0x02;
    }
    // (C)
    if apply_heuristic_c_corner_penalty(quant, &stats, c) {
        fired |= 0x04;
    }
    // (D)
    if apply_heuristic_d_dct8_flatness(quant, &stats, raw_strategy) {
        fired |= 0x08;
    }
    // (E)
    if apply_heuristic_e_large_transform(quant, &stats, c, raw_strategy, xsize, ysize) {
        fired |= 0x10;
    }
    // (F)
    let (f_fired, activity) = apply_heuristic_f_activity(quant, thresholds, &stats, c, xsize, ysize);
    if f_fired {
        fired |= 0x20;
    }

    AdjustQuantOutcome {
        heuristics_fired: fired,
        sum_of_vals: stats.sum_of_vals,
        sum_of_error: stats.sum_of_error,
        activity,
    }
}

/// Convenience: returns a length-`num_blocks * 64` vec of all-1.0
/// inverse quant matrix entries. Useful for tests where you don't
/// care about the actual quant matrix.
#[doc(hidden)]
pub fn unit_weights(num_blocks: usize) -> Vec<f32> {
    alloc::vec![1.0_f32; num_blocks * 64]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjust_quant_prescan_skips_partial_block_kinds() {
        use crate::forks::transform::{
            RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8,
            RAW_STRATEGY_DCT8X4, RAW_STRATEGY_IDENTITY,
        };
        let coeffs = [0.5_f32; 64];
        let weights = [1.0_f32; 64];
        let thresholds = [0.6_f32; 4];
        for &kind in &[
            RAW_STRATEGY_IDENTITY,
            RAW_STRATEGY_DCT2X2,
            RAW_STRATEGY_DCT4X4,
            RAW_STRATEGY_DCT4X8,
            RAW_STRATEGY_DCT8X4,
        ] {
            let r = adjust_quant_prescan(
                &coeffs, &weights, 1.0, 1.0, 1, kind, 8, 8, 1, 1, &thresholds,
            );
            assert!(r.is_none(), "partial block kind {kind} should skip prescan");
        }
    }

    #[test]
    fn test_adjust_quant_prescan_dct8_zero_block() {
        use crate::forks::transform::RAW_STRATEGY_DCT;
        // All-zero coefficients → all stats = 0. This is the trivial
        // sanity check matching upstream's behavior on an empty block.
        let coeffs = [0.0_f32; 64];
        let weights = [1.0_f32; 64];
        let thresholds = [0.6_f32; 4];
        let s = adjust_quant_prescan(
            &coeffs,
            &weights,
            10.0,
            1.0,
            1,
            RAW_STRATEGY_DCT,
            8,
            8,
            1,
            1,
            &thresholds,
        )
        .expect("DCT8 should not skip");
        assert_eq!(s.sum_of_highest_freq, 0.0);
        assert_eq!(s.sum_of_error, 0.0);
        assert_eq!(s.sum_of_vals, 0.0);
        assert_eq!(s.hf_nonzeros, [0.0_f32; 4]);
        assert_eq!(s.hf_max_error, [0.0_f32; 4]);
    }

    #[test]
    fn test_adjust_quant_prescan_threshold_zeros() {
        use crate::forks::transform::RAW_STRATEGY_DCT;
        // Coefficients that quantize below threshold (after scaling)
        // contribute to hf_max_error[hfix] when channel is Y.
        // Set qac so (val = coeff * qac * inv_w) is just below threshold.
        let mut coeffs = [0.0_f32; 64];
        coeffs[63] = 0.05; // far in HF quadrant 3 (bottom-right)
        let weights = [1.0_f32; 64];
        let thresholds = [0.6_f32; 4];
        let s = adjust_quant_prescan(
            &coeffs,
            &weights,
            10.0,
            1.0,
            1,
            RAW_STRATEGY_DCT,
            8,
            8,
            1,
            1,
            &thresholds,
        )
        .unwrap();
        // val = 0.05 * 10 * 1 = 0.5, abs(0.5) < 0.6 → quantizes to 0,
        // contributes to hf_max_error[3] = 0.5.
        assert!((s.hf_max_error[3] - 0.5).abs() < 1e-6);
        assert_eq!(s.sum_of_vals, 0.0);
        assert!((s.sum_of_error - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_heuristic_a_no_op_for_1x1() {
        let mut t = [0.62_f32; 4];
        let fired = apply_heuristic_a_thresholds(&mut t, 1, 1);
        assert!(!fired);
        assert_eq!(t, [0.62_f32; 4]);
    }

    #[test]
    fn test_heuristic_a_2x2() {
        // adj = 0.003 * 4 = 0.012 (under 0.08 cap), each threshold drops by 0.012.
        let mut t = [0.62_f32; 4];
        let fired = apply_heuristic_a_thresholds(&mut t, 2, 2);
        assert!(fired);
        for v in &t {
            assert!((v - (0.62 - 0.012)).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn test_heuristic_a_caps_at_008() {
        // For xsize*ysize >= 27 (= 0.08 / 0.003), adj caps at 0.08.
        // 8x8 → 64 → adj = 0.192 capped to 0.08.
        let mut t = [0.62_f32; 4];
        let fired = apply_heuristic_a_thresholds(&mut t, 8, 8);
        assert!(fired);
        for v in &t {
            assert!((v - (0.62 - 0.08)).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn test_heuristic_a_clamps_at_054_floor() {
        // Start at the floor — can't go lower. xsize=2, ysize=2 → adj = 0.012.
        // 0.55 - 0.012 = 0.538 → clamped up to 0.54.
        let mut t = [0.55_f32; 4];
        let _ = apply_heuristic_a_thresholds(&mut t, 2, 2);
        for v in &t {
            assert_eq!(*v, 0.54);
        }
    }

    #[test]
    fn test_heuristic_f_activity_zero_yields_no_fire() {
        // min hf_nonzeros = 0 → activity = (0 + div/2) / div = 0 (div=1).
        // qp = quant - 0 = quant unchanged → not fired.
        let mut q = 100;
        let mut t = [0.62_f32; 4];
        let stats = AdjustQuantBlockStats {
            hf_nonzeros: [0.0; 4],
            ..Default::default()
        };
        let (fired, activity) = apply_heuristic_f_activity(&mut q, &mut t, &stats, 1, 1, 1);
        assert_eq!(activity, 0);
        assert!(!fired);
        assert_eq!(q, 100);
        // Y-channel: thresholds[1..4] += 0.01 * 0 = no change
        assert_eq!(t, [0.62_f32; 4]);
    }

    #[test]
    fn test_heuristic_f_activity_5_y_channel_bumps_thresholds() {
        // div=1, min hf_nonzeros = 5 → activity = (5 + 0) / 1 = 5
        // qp = 100 - 5 = 95, qp_limit = 50, 95 > 50 → quant = 95.
        // Y channel: thresholds[1..4] += 0.05.
        let mut q = 100;
        let mut t = [0.62_f32; 4];
        let stats = AdjustQuantBlockStats {
            hf_nonzeros: [5.0, 7.0, 8.0, 6.0],
            ..Default::default()
        };
        let (fired, activity) = apply_heuristic_f_activity(&mut q, &mut t, &stats, 1, 1, 1);
        assert_eq!(activity, 5);
        assert!(fired);
        assert_eq!(q, 95);
        assert_eq!(t[0], 0.62);
        for v in &t[1..4] {
            assert!((v - 0.67).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn test_heuristic_f_x_channel_thresholds_unchanged() {
        // X channel (c=0): no threshold modification.
        let mut q = 100;
        let mut t = [0.62_f32; 4];
        let stats = AdjustQuantBlockStats {
            hf_nonzeros: [5.0; 4],
            ..Default::default()
        };
        let (_, activity) = apply_heuristic_f_activity(&mut q, &mut t, &stats, 0, 1, 1);
        assert_eq!(activity, 5);
        assert_eq!(t, [0.62_f32; 4]);
    }

    #[test]
    fn test_heuristic_f_floor_clamps_quant() {
        // activity = 15 (capped: min hf_nonzeros = 100 > 15 * 1).
        // quant 6 - 15 = -9, qp_limit = max(3, 4) = 4 → clamped to 4.
        let mut q = 6;
        let mut t = [0.62_f32; 4];
        let stats = AdjustQuantBlockStats {
            hf_nonzeros: [100.0; 4],
            ..Default::default()
        };
        let (fired, activity) = apply_heuristic_f_activity(&mut q, &mut t, &stats, 1, 1, 1);
        assert_eq!(activity, 15);
        assert_eq!(q, 4);
        assert!(fired); // qp != orig
    }

    #[test]
    fn test_heuristic_e_skips_small_transforms() {
        use crate::forks::transform::{RAW_STRATEGY_DCT, RAW_STRATEGY_DCT4X4};
        let q = 100;
        let stats = AdjustQuantBlockStats {
            sum_of_error: 1e6, // huge
            ..Default::default()
        };
        for &strat in &[RAW_STRATEGY_DCT, RAW_STRATEGY_DCT4X4] {
            let mut q2 = q;
            assert!(!apply_heuristic_e_large_transform(&mut q2, &stats, 1, strat, 1, 1));
            assert_eq!(q2, q);
        }
    }

    #[test]
    fn test_heuristic_e_dct16_fires_on_large_error() {
        use crate::forks::transform::RAW_STRATEGY_DCT16X16;
        // DCT16x16 (xsize=2, ysize=2). area = 4*64 = 256.
        // K_MUL1[0][1] (Y) = 0.4579748, K_MUL2[0][1] = 1.1255767.
        // With sum_of_vals = 0: threshold = 0.4579748 * 256 + 0 = 117.24.
        // For norm_error > 117.24, need sum_of_error > 117.24 / 2.2942 ≈ 51.1
        let mut q = 100;
        let stats = AdjustQuantBlockStats {
            sum_of_error: 1000.0, // norm_error = 2294 > threshold
            sum_of_vals: 0.0,
            ..Default::default()
        };
        let fired = apply_heuristic_e_large_transform(
            &mut q,
            &stats,
            1,
            RAW_STRATEGY_DCT16X16,
            2,
            2,
        );
        assert!(fired);
        assert!(q > 100, "quant should be bumped, got {q}");
        assert!(q <= 100 + 2, "quant bump capped at 2, got {q}");
    }

    #[test]
    fn test_heuristic_e_no_fire_on_small_error() {
        use crate::forks::transform::RAW_STRATEGY_DCT16X16;
        let mut q = 100;
        let stats = AdjustQuantBlockStats {
            sum_of_error: 1.0, // tiny
            sum_of_vals: 100.0, // raises threshold
            ..Default::default()
        };
        let fired = apply_heuristic_e_large_transform(
            &mut q,
            &stats,
            1,
            RAW_STRATEGY_DCT16X16,
            2,
            2,
        );
        assert!(!fired);
        assert_eq!(q, 100);
    }

    #[test]
    fn test_heuristic_b_skips_non_y_channel() {
        let q = 100;
        let t = [0.62_f32; 4];
        let stats = AdjustQuantBlockStats {
            sum_of_vals: 0.0, // sparse
            hf_nonzeros: [0.0; 4],
            hf_max_error: [1.0; 4], // big enough to fire
            ..Default::default()
        };
        for c in [0_usize, 2] {
            let mut q2 = q;
            let mut t2 = t;
            assert!(!apply_heuristic_b_sparse_y(&mut q2, &mut t2, &stats, c, 2, 2));
            assert_eq!(q2, q);
            assert_eq!(t2, t);
        }
    }

    #[test]
    fn test_heuristic_b_skips_non_sparse() {
        let mut q = 100;
        let mut t = [0.62_f32; 4];
        let stats = AdjustQuantBlockStats {
            sum_of_vals: 1.0, // 1.0 * 8 = 8 >= 4 (xsize*ysize for 2x2) → not sparse
            ..Default::default()
        };
        assert!(!apply_heuristic_b_sparse_y(&mut q, &mut t, &stats, 1, 2, 2));
        assert_eq!(q, 100);
        assert_eq!(t, [0.62_f32; 4]);
    }

    #[test]
    fn test_heuristic_b_quadrant_3_updates_thresholds_3() {
        let mut q = 100;
        let mut t = [0.62_f32; 4];
        let stats = AdjustQuantBlockStats {
            sum_of_vals: 0.0,
            hf_nonzeros: [0.0; 4],
            hf_max_error: [0.0, 0.0, 0.0, 1.0], // Q3 fires
            ..Default::default()
        };
        let fired = apply_heuristic_b_sparse_y(&mut q, &mut t, &stats, 1, 2, 2);
        assert!(fired);
        assert_eq!(q, 101); // bumped via quadrant 3 fire
        // thresholds[3] = 0.9999 * 1.0 * 101/100 = 1.009899
        assert!((t[3] - 1.0099_f32).abs() < 1e-3);
        // unchanged positions
        assert_eq!(t[0], 0.62);
        assert_eq!(t[1], 0.62);
        assert_eq!(t[2], 0.62);
    }

    #[test]
    fn test_heuristic_c_no_fire_when_no_hf_signal() {
        // sum_of_highest_freq=0 → never fires.
        let q = 100;
        let stats = AdjustQuantBlockStats {
            sum_of_highest_freq: 0.0,
            hf_nonzeros: [10.0, 10.0, 10.0, 10.0],
            ..Default::default()
        };
        for c in 0..3 {
            let mut q2 = q;
            assert!(!apply_heuristic_c_corner_penalty(&mut q2, &stats, c));
            assert_eq!(q2, q);
        }
    }

    #[test]
    fn test_heuristic_c_fires_y_channel() {
        // c=1, mul=30. all = sum(hf_nonzeros) + 1 = 1.0 (no nonzeros) + 1 = 1.0
        // Wait: hf_nonzeros all 0 → all = 1.0. mul[1]=30. sum_of_highest_freq=1.
        // 30*1 >= 1 → fires. bump = (30 * 1 / 1) = 30. quant 100 + 30 = 130.
        let mut q = 100;
        let stats = AdjustQuantBlockStats {
            sum_of_highest_freq: 1.0,
            hf_nonzeros: [0.0; 4],
            ..Default::default()
        };
        let fired = apply_heuristic_c_corner_penalty(&mut q, &stats, 1);
        assert!(fired);
        assert_eq!(q, 130);
    }

    #[test]
    fn test_heuristic_c_clamps_at_quant_max() {
        // Force a huge bump to verify clamping.
        let mut q = 200;
        let stats = AdjustQuantBlockStats {
            sum_of_highest_freq: 100.0,
            hf_nonzeros: [0.0; 4],
            ..Default::default()
        };
        let _ = apply_heuristic_c_corner_penalty(&mut q, &stats, 0);
        assert_eq!(q, QUANT_MAX - 1);
    }

    #[test]
    fn test_heuristic_d_only_for_dct8() {
        use crate::forks::transform::{RAW_STRATEGY_DCT, RAW_STRATEGY_DCT16X16};
        let mut q = 100;
        // hf_nonzeros sum = 0 → < 11, would fire if DCT8
        let stats = AdjustQuantBlockStats {
            hf_nonzeros: [0.0; 4],
            ..Default::default()
        };
        assert!(!apply_heuristic_d_dct8_flatness(
            &mut q,
            &stats,
            RAW_STRATEGY_DCT16X16
        ));
        assert_eq!(q, 100);
        assert!(apply_heuristic_d_dct8_flatness(&mut q, &stats, RAW_STRATEGY_DCT));
        assert_eq!(q, 101);
    }

    #[test]
    fn test_heuristic_d_no_fire_when_active() {
        use crate::forks::transform::RAW_STRATEGY_DCT;
        let mut q = 100;
        let stats = AdjustQuantBlockStats {
            hf_nonzeros: [4.0, 4.0, 4.0, 4.0], // sum = 16 > 11
            ..Default::default()
        };
        assert!(!apply_heuristic_d_dct8_flatness(&mut q, &stats, RAW_STRATEGY_DCT));
        assert_eq!(q, 100);
    }

    #[test]
    fn test_orchestrator_partial_block_kind_returns_zeros() {
        use crate::forks::transform::RAW_STRATEGY_DCT4X4;
        let coeffs = [0.5_f32; 64];
        let weights = [1.0_f32; 64];
        let mut thresholds = [0.62_f32; 4];
        let mut quant = 100;
        let out = adjust_quant_block_ac_host(
            &coeffs,
            &weights,
            1.0,
            1.0,
            1,
            RAW_STRATEGY_DCT4X4,
            8,
            8,
            1,
            1,
            &mut thresholds,
            &mut quant,
        );
        assert_eq!(out, AdjustQuantOutcome::default());
        // Quant unchanged for partial kinds (no heuristics ran).
        assert_eq!(quant, 100);
        // Thresholds unchanged: A wouldn't have fired anyway (xsize=ysize=1)
        assert_eq!(thresholds, [0.62_f32; 4]);
    }

    #[test]
    fn test_orchestrator_dct8_zero_block() {
        use crate::forks::transform::RAW_STRATEGY_DCT;
        // Zero block, DCT8: pre-scan runs, all stats zero. A skipped
        // (xsize=ysize=1). B skipped (sum_of_vals=0 BUT c=1 and 0*8 < 1
        // would be true... wait: sum_of_vals=0, 0*8=0 < 1 → sparse, fires).
        // hf_max_error all 0 → no quant bump from B, no threshold update.
        // C: sum_of_highest_freq=0 → no fire.
        // D: sum(hf_nonzeros)=0 < 11 → fires, +1.
        // E: not large strategy → skip.
        // F: min hf_nonzeros=0, activity=0, qp=quant unchanged → no fire.
        let coeffs = [0.0_f32; 64];
        let weights = [1.0_f32; 64];
        let mut thresholds = [0.62_f32; 4];
        let mut quant = 100;
        let out = adjust_quant_block_ac_host(
            &coeffs,
            &weights,
            1.0,
            1.0,
            1,
            RAW_STRATEGY_DCT,
            8,
            8,
            1,
            1,
            &mut thresholds,
            &mut quant,
        );
        // B fired (sparse) + D fired (flat). C no, E no, F activity=0 no fire.
        // After D's +1: quant = 101.
        // After F (activity=0): quant unchanged.
        assert_eq!(quant, 101);
        assert_eq!(out.activity, 0);
        // Bits: B=0x02, D=0x08 → 0x0A
        assert_eq!(out.heuristics_fired, 0x02 | 0x08);
        assert_eq!(out.sum_of_vals, 0.0);
        assert_eq!(out.sum_of_error, 0.0);
    }

    #[test]
    fn test_default_thresholds_y() {
        let t = default_thresholds(1, 1, 1);
        assert_eq!(t, [0.56, 0.62, 0.62, 0.62]);
    }

    #[test]
    fn test_default_thresholds_x_single_block() {
        let t = default_thresholds(0, 1, 1);
        assert_eq!(t, [0.58, 0.62, 0.62, 0.62]);
    }

    #[test]
    fn test_default_thresholds_x_multi_block_reduction() {
        // covered_x=2, covered_y=2 → 4 blocks → adj = 0.00744 * 4 = 0.02976
        let t = default_thresholds(0, 2, 2);
        let adj = 0.00744 * 4.0;
        assert!((t[0] - (0.58 - adj)).abs() < 1e-6);
        assert!((t[1] - (0.62 - adj)).abs() < 1e-6);
    }

    #[test]
    fn test_default_thresholds_y_no_multi_block_reduction() {
        // Y channel never gets multi-block adjustment.
        let t = default_thresholds(1, 4, 4);
        assert_eq!(t, [0.56, 0.62, 0.62, 0.62]);
    }

    #[test]
    fn test_default_thresholds_clamp() {
        // Very large coverage would push thresholds below 0.5 — should clamp.
        let t = default_thresholds(0, 16, 16);
        for &v in &t {
            assert!(v >= 0.5, "threshold {v} below 0.5 floor");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_quantize_dct8_zero_input_gpu() {
        // Zero coefficients → all-zero quantized output.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 4_usize;
        let coeffs = vec![0.0_f32; nb * 64];
        let weights = unit_weights(nb);
        let qac_qm = vec![1.0_f32; nb];
        let thr = [0.56_f32, 0.62, 0.62, 0.62];
        let q = quantize_dct8_blocks_gpu(&enc, &coeffs, &weights, &qac_qm, &thr);
        assert_eq!(q.len(), nb * 64);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_quantize_dct8_threshold_dead_zone_gpu() {
        // Coefficients below threshold should be zeroed out.
        // With weight=1, qac_qm=1, threshold=0.62, a coefficient of 0.5
        // multiplies through to 0.5 → below 0.62 → quantize to 0.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 1_usize;
        let mut coeffs = vec![0.0_f32; 64];
        // First quadrant uses thresholds[0] = 0.56; values 0.5 < 0.56 → zero.
        // Set position (0,0) (DC; the kernel actually skips DC, but we
        // also poke a few AC positions in each quadrant).
        coeffs[1] = 0.5; // (0,1): top-left quadrant (TL): below 0.56 → 0
        coeffs[8] = 0.4; // (1,0): top-left quadrant (TL): below 0.56 → 0
        coeffs[5] = 0.7; // (0,5): top-right (>= block_w/2): TR threshold = 0.62, 0.7 >= 0.62 → 1
        let weights = vec![1.0_f32; 64];
        let qac_qm = vec![1.0_f32; nb];
        let thr = [0.56_f32, 0.62, 0.62, 0.62];
        let q = quantize_dct8_blocks_gpu(&enc, &coeffs, &weights, &qac_qm, &thr);
        // Position 1 and 8 should be zero (below threshold).
        assert_eq!(q[1], 0, "coeff 0.5 at TL should be zeroed");
        assert_eq!(q[8], 0, "coeff 0.4 at TL should be zeroed");
        // Position 5 should round to 1 (0.7 → 1).
        assert_eq!(q[5], 1, "coeff 0.7 should quantize to 1, got {}", q[5]);
    }
}
