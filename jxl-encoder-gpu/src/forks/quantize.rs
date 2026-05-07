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
//! Not yet covered (no GPU kernel for these):
//! - Larger-strategy quantize (DCT16+, AFV, IDENTITY, DCT2X2)
//! - `adjust_quant_block_ac` heuristics (sparse-block boost, HF corner
//!   increase, flatness detection, etc.) — stay on CPU
//! - Error diffusion in zigzag order — stay on CPU (libjxl never
//!   uses ED in QuantizeBlockAC anyway, despite accepting the param)

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
