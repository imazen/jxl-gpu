// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/{quantize.rs, reconstruct.rs}
// (BSD-3-Clause via libjxl + AGPL/commercial), with the 3-channel
// DequantBlock SIMD substituted for a batched GPU launch.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted DCT8 dequantization (decoder-side reconstruction).
//!
//! Currently covers:
//! - `adjust_quant_bias` — pure scalar, duplicated bit-for-bit. Used
//!   by upstream during coefficient-level dequant before applying
//!   the inverse quant matrix. (~ns per call; no GPU win.)
//! - `dequant_dct8_blocks_gpu` — 3-channel batched DCT8 dequant via
//!   GPU. One launch covers ALL DCT8 blocks across X/Y/B channels.
//!
//! Not yet covered (no GPU kernel): dequant for non-DCT8 strategies
//! (DCT16+, AFV, IDENTITY). The CPU path stays for those — relatively
//! few blocks per image use the larger transforms.
//!
//! Reshape vs upstream:
//! - Upstream's DequantBlock dispatches per-strategy, per-block, in a
//!   nested loop over the strategy bucket layout. Each block's coefficients
//!   are scaled by `weight[i] / (qac * qm_mul)` element-wise.
//! - Our GPU dequant_dct8_blocks_gpu takes ALL blocks of all 3 channels
//!   concatenated, plus per-block qac_qm and per-coefficient weights.
//!   One launch covers everything.
//!
//! ## Output layout
//!
//! For each block index `b` in `[0, num_blocks)`, output coefficients
//! live at `output[b*64 .. (b+1)*64]` in 8×8 row-major order. Same
//! layout as the input quantized blocks.

use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

/// kDefaultQuantBias from libjxl-tiny enc_group.cc — used during
/// integer→float dequantization to bias ±1 values toward the
/// distribution's expected magnitude.
///
/// `[0..2]` = channel-specific bias for ±1 values (X, Y, B);
/// `[3]` = reciprocal correction factor for `|q| >= 2`.
const QUANT_BIAS: [f32; 4] = [
    1.0 - 0.05465007330715401,  // [0] X channel ±1 → 0.945349
    1.0 - 0.07005449891748593,  // [1] Y channel ±1 → 0.929946
    1.0 - 0.049935103337343655, // [2] B channel ±1 → 0.950065
    0.145,                      // [3] reciprocal correction
];

/// Adjust a quantized integer for dequantization bias.
///
/// Pure scalar — bit-for-bit copy of upstream
/// `jxl_encoder::vardct::quantize::adjust_quant_bias`.
///
/// - For `quantized == 0` → `0.0`.
/// - For `|quantized| == 1` → `±BIAS[channel]` (channel-dependent
///   center value).
/// - For `|quantized| >= 2` → `q - BIAS[3] / q` (reciprocal taper).
///
/// ```
/// use jxl_encoder_gpu::forks::dequant::adjust_quant_bias;
///
/// // Zero stays zero.
/// assert_eq!(adjust_quant_bias(0, 0), 0.0);
/// assert_eq!(adjust_quant_bias(0, 1), 0.0);
/// assert_eq!(adjust_quant_bias(0, 2), 0.0);
///
/// // ±1 → channel-specific bias (X=0.945, Y=0.930, B=0.950).
/// // Y has the largest correction (smallest bias center).
/// assert!(adjust_quant_bias(1, 1) < adjust_quant_bias(1, 0));
/// assert!(adjust_quant_bias(1, 1) < adjust_quant_bias(1, 2));
/// assert!((adjust_quant_bias(1, 0) + adjust_quant_bias(-1, 0)).abs() < 1e-6);
///
/// // |q| >= 2 → q - 0.145/q (reciprocal taper).
/// // For q=5: 5 - 0.145/5 = 4.971.
/// assert!((adjust_quant_bias(5, 0) - (5.0 - 0.145 / 5.0)).abs() < 1e-6);
/// ```
#[inline]
pub fn adjust_quant_bias(quantized: i32, channel: usize) -> f32 {
    if quantized == 0 {
        return 0.0;
    }
    let q = quantized as f32;
    if q.abs() < 1.125 {
        q.signum() * QUANT_BIAS[channel]
    } else {
        q - QUANT_BIAS[3] / q
    }
}

/// 3-channel batched DCT8 dequant on GPU. Returns the dequantized
/// coefficient blocks for X, Y, B in the same order as the input.
///
/// All inputs MUST be aligned per-block:
/// - `quant_x/y/b`: integer quant values, length `num_blocks * 64`
/// - `weights_x/y/b`: per-coefficient inverse-dequant matrix entries
///   (same layout as quant)
/// - `qac_qm_x/y/b`: per-block scale factors (length `num_blocks`)
/// - `x_factor`, `b_factor`: per-block CfL factors (length `num_blocks`)
///
/// Mirrors the dequant step that upstream's `reconstruct_xyb` performs
/// during the encoder-side reconstruction loop (used for pixel-domain
/// loss estimation and EPF prep).
#[allow(clippy::too_many_arguments)]
pub fn dequant_dct8_blocks_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    quant_x: &[i32],
    quant_y: &[i32],
    quant_b: &[i32],
    weights_x: &[f32],
    weights_y: &[f32],
    weights_b: &[f32],
    qac_qm_x: &[f32],
    qac_qm_y: &[f32],
    qac_qm_b: &[f32],
    x_factor: &[f32],
    b_factor: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    enc.dequant_dct8_blocks(
        quant_x, quant_y, quant_b, weights_x, weights_y, weights_b, qac_qm_x, qac_qm_y, qac_qm_b,
        x_factor, b_factor,
    )
}

/// Strategy-aware simple dequant (no CfL, no adjust_quant_bias).
/// Computes `output[i] = quant[i] * weights[i]` for arbitrary
/// `block_size` (DCT8 → 64, DCT16x16 → 256, DCT32x32 → 1024, etc.).
///
/// Use this for the larger-strategy decoder paths where the
/// DCT8-specific CfL + adjust_quant_bias adjustments don't apply.
pub fn dequant_blocks_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    quant: &[i32],
    weights: &[f32],
    block_size: u32,
) -> Vec<f32> {
    enc.dequant_simple_blocks(quant, weights, block_size)
}

/// Broadcast-weights variant of [`dequant_blocks_gpu`].
/// `weights_template` is exactly `block_size` f32 (one quant matrix);
/// the kernel broadcasts across all blocks. Saves
/// `(num_blocks - 1) * block_size * 4` bytes of upload traffic when
/// callers were previously replicating the matrix per-block.
pub fn dequant_blocks_gpu_broadcast_w<R: Runtime>(
    enc: &GpuEncoder<R>,
    quant: &[i32],
    weights_template: &[f32],
    block_size: u32,
) -> Vec<f32> {
    enc.dequant_simple_blocks_broadcast_w(quant, weights_template, block_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjust_quant_bias_zero() {
        for c in 0..3 {
            assert_eq!(adjust_quant_bias(0, c), 0.0);
        }
    }

    #[test]
    fn test_adjust_quant_bias_pm_one() {
        // ±1 → ±BIAS[channel]
        for (c, &expected) in QUANT_BIAS[..3].iter().enumerate() {
            assert_eq!(adjust_quant_bias(1, c), expected);
            assert_eq!(adjust_quant_bias(-1, c), -expected);
        }
    }

    #[test]
    fn test_adjust_quant_bias_large_q() {
        // |q| >= 2: q - 0.145/q
        let q = 5_i32;
        let expected = 5.0 - 0.145 / 5.0;
        let got = adjust_quant_bias(q, 0);
        assert!(
            (got - expected).abs() < 1e-6,
            "got={got}, expected={expected}"
        );

        let qn = -10_i32;
        let expected_n = -10.0 - 0.145 / -10.0;
        let got_n = adjust_quant_bias(qn, 1);
        assert!((got_n - expected_n).abs() < 1e-6);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_dequant_dct8_zero_input_gpu() {
        // All-zero quantized input → all-zero output regardless of
        // weights/qac_qm/cfl factors.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 4_usize;
        let qx = alloc::vec![0_i32; nb * 64];
        let qy = qx.clone();
        let qb = qx.clone();
        let weights = alloc::vec![1.0_f32; nb * 64];
        let qac_qm = alloc::vec![1.0_f32; nb];
        let xf = alloc::vec![0.0_f32; nb];
        let bf = alloc::vec![0.0_f32; nb];
        let (ox, oy, ob) = dequant_dct8_blocks_gpu(
            &enc, &qx, &qy, &qb, &weights, &weights, &weights, &qac_qm, &qac_qm, &qac_qm, &xf, &bf,
        );
        assert!(ox.iter().all(|&v| v == 0.0));
        assert!(oy.iter().all(|&v| v == 0.0));
        assert!(ob.iter().all(|&v| v == 0.0));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_dequant_dct8_finite_output_gpu() {
        // Random-ish quantized input with varied weights → finite output.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 8_usize;
        let qx: Vec<i32> = (0..nb * 64).map(|i| (i as i32 % 7) - 3).collect();
        let qy = qx.clone();
        let qb = qx.clone();
        let weights: Vec<f32> = (0..nb * 64)
            .map(|i| 1.0 + (i as f32 * 0.013).sin())
            .collect();
        let qac_qm: Vec<f32> = (0..nb).map(|i| 1.0 + i as f32 * 0.1).collect();
        let xf = alloc::vec![0.0_f32; nb];
        let bf = alloc::vec![0.0_f32; nb];
        let (ox, oy, ob) = dequant_dct8_blocks_gpu(
            &enc, &qx, &qy, &qb, &weights, &weights, &weights, &qac_qm, &qac_qm, &qac_qm, &xf, &bf,
        );
        assert!(ox.iter().all(|v| v.is_finite()));
        assert!(oy.iter().all(|v| v.is_finite()));
        assert!(ob.iter().all(|v| v.is_finite()));
    }
}
