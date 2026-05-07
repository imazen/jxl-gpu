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
