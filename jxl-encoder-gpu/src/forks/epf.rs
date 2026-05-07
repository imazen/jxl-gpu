// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/epf.rs (BSD-3-Clause via libjxl
// + AGPL/commercial), with the SIMD step1 + step2 calls substituted
// for GPU launches.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted Edge-Preserving Filter passes.
//!
//! Currently covers:
//! - `compute_inv_sigma_map` — pure scalar, duplicated bit-for-bit
//!   (~10 microseconds for a 1024×1024 image; no GPU win)
//! - `apply_epf_step1_gpu` — 5×5 plus-shaped kernel (the strong pass)
//! - `apply_epf_step2_gpu` — 3×3 cross kernel (the weak pass)
//!
//! Not yet covered (no GPU kernel for these):
//! - `epf_step0` — 12-tap kernel (the strongest pass, between two passes
//!   of EPF in the decoder). Used at higher EPF iter counts.
//! - `compute_epf_sharpness` — per-block sharpness selection
//!
//! Reshape vs upstream `apply_epf`:
//! - Upstream orchestrates step0 + step1 + step2 via the SIMD dispatch
//!   trampoline, with one scratch buffer per step.
//! - Our forks expose step1 and step2 as INDIVIDUAL passes the caller
//!   chains. Because GPU launches return new `Vec<f32>`, the caller can
//!   feed step1's output directly into step2's input without managing
//!   intermediate scratch.

use alloc::vec;
use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

// === Constants from libjxl loop_filter.cc, duplicated bit-for-bit ===

const K_INV_SIGMA_NUM: f32 = -1.171_572_9;

/// Default EPF parameters.
pub const EPF_QUANT_MUL: f32 = 0.46;
pub const EPF_PASS0_SIGMA_SCALE: f32 = 0.9;
pub const EPF_PASS2_SIGMA_SCALE: f32 = 6.5;
pub const EPF_BORDER_SAD_MUL: f32 = 2.0 / 3.0;

/// Default sharpness LUT: `EPF_SHARP_LUT[i] = i / 7.0`.
pub const EPF_SHARP_LUT: [f32; 8] = [
    0.0,
    1.0 / 7.0,
    2.0 / 7.0,
    3.0 / 7.0,
    4.0 / 7.0,
    5.0 / 7.0,
    6.0 / 7.0,
    1.0,
];

/// Pure-scalar copy of upstream `compute_inv_sigma_map`. Returns one
/// inv_sigma per 8×8 block.
///
/// Computes `sigma = (EPF_QUANT_MUL / (quant_scale * raw_quant *
/// K_INV_SIGMA_NUM)) * EPF_SHARP_LUT[sharpness]` per block, then
/// returns `1 / sigma` (or 0 when sigma underflows).
///
/// Two guard cases produce zero output:
/// 1. `raw_quant == 0` (would divide by zero in sigma).
/// 2. `EPF_SHARP_LUT[sharpness] == 0` (i.e., `sharpness == 0`).
///
/// ```
/// use jxl_encoder_gpu::forks::epf::compute_inv_sigma_map;
///
/// // raw_quant=0 → guarded, output 0.
/// // sharpness=0 → EPF_SHARP_LUT[0]=0 → sigma=0 → guarded, output 0.
/// // raw_quant>0 + sharpness>0 → finite negative inv_sigma
/// // (K_INV_SIGMA_NUM is negative).
/// let qf = vec![0_u8, 128, 128];
/// let sm = vec![0_u8, 0, 4];
/// let inv = compute_inv_sigma_map(&qf, &sm, 1.0, 3, 1);
/// assert_eq!(inv[0], 0.0);  // raw_quant=0 → guarded
/// assert_eq!(inv[1], 0.0);  // sharpness=0 → sigma=0 → guarded
/// assert!(inv[2].is_finite() && inv[2] != 0.0);  // non-trivial
/// // Sharpness clamped at 7 (LUT length).
/// let inv7 = compute_inv_sigma_map(&[128_u8], &[7_u8], 1.0, 1, 1);
/// let inv99 = compute_inv_sigma_map(&[128_u8], &[99_u8], 1.0, 1, 1);
/// assert_eq!(inv7[0], inv99[0]);
/// ```
pub fn compute_inv_sigma_map(
    quant_field: &[u8],
    sharpness_map: &[u8],
    quant_scale: f32,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> Vec<f32> {
    assert_eq!(quant_field.len(), xsize_blocks * ysize_blocks);
    assert_eq!(sharpness_map.len(), xsize_blocks * ysize_blocks);
    let mut inv_sigma = vec![0.0_f32; xsize_blocks * ysize_blocks];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let idx = by * xsize_blocks + bx;
            let raw_quant = quant_field[idx] as f32;
            let sharpness = sharpness_map[idx].min(7) as usize;
            let sigma_quant = EPF_QUANT_MUL / (quant_scale * raw_quant * K_INV_SIGMA_NUM);
            let sigma = sigma_quant * EPF_SHARP_LUT[sharpness];
            if sigma.abs() > 1e-10 {
                inv_sigma[idx] = 1.0 / sigma;
            }
        }
    }
    inv_sigma
}

/// EPF Step 1 on GPU — 5×5 plus-shaped kernel applied to all 3 channels.
/// Inputs are PADDED (width = unpadded_width + 2*pad).
///
/// Mirrors the upstream Step 1 pass invoked by `apply_epf`. Returns
/// three new unpadded buffers `(out_x, out_y, out_b)` of size
/// `unpadded_width * unpadded_height`.
///
/// `sigma_scale` is typically 1.0 for Step 1; `border_sigma_mul` is
/// `EPF_BORDER_SAD_MUL` (2/3).
#[allow(clippy::too_many_arguments)]
pub fn apply_epf_step1_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    inv_sigma: &[f32],
    width: u32,
    height: u32,
    xsize_blocks: u32,
    ysize_blocks: u32,
    pad: u32,
    sigma_scale: f32,
    border_sigma_mul: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    enc.epf_step1_channels(
        in_x,
        in_y,
        in_b,
        inv_sigma,
        width,
        height,
        xsize_blocks,
        ysize_blocks,
        pad,
        sigma_scale,
        border_sigma_mul,
    )
}

/// EPF Step 2 on GPU — 3×3 cross kernel.
/// Same I/O shape as `apply_epf_step1_gpu`. `sigma_scale` is typically
/// `EPF_PASS2_SIGMA_SCALE` (6.5).
#[allow(clippy::too_many_arguments)]
pub fn apply_epf_step2_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    inv_sigma: &[f32],
    width: u32,
    height: u32,
    xsize_blocks: u32,
    ysize_blocks: u32,
    pad: u32,
    sigma_scale: f32,
    border_sigma_mul: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    enc.epf_step2_channels(
        in_x,
        in_y,
        in_b,
        inv_sigma,
        width,
        height,
        xsize_blocks,
        ysize_blocks,
        pad,
        sigma_scale,
        border_sigma_mul,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inv_sigma_map_zero_quant_safe() {
        // raw_quant=0 → sigma=∞ (divide-by-zero would happen but we
        // gate on sigma.abs() > 1e-10), inv_sigma stays 0.
        // raw_quant=128, sharpness=0 → sigma=0 (because EPF_SHARP_LUT[0]=0),
        // inv_sigma stays 0.
        // raw_quant=128, sharpness=4 → non-zero negative sigma → finite
        // negative inv_sigma.
        let qf = vec![0_u8, 128, 128];
        let sm = vec![0_u8, 0, 4];
        let inv = compute_inv_sigma_map(&qf, &sm, 1.0, 3, 1);
        assert_eq!(inv.len(), 3);
        assert_eq!(inv[0], 0.0); // raw_quant=0 → guarded
        assert_eq!(inv[1], 0.0); // sharpness=0 → sigma=0 → guarded
        assert!(inv[2].is_finite());
        assert!(inv[2] != 0.0); // non-trivial result
    }

    #[test]
    fn test_inv_sigma_map_dimensions() {
        let xb = 8;
        let yb = 4;
        let qf = vec![64_u8; xb * yb];
        let sm = vec![3_u8; xb * yb];
        let inv = compute_inv_sigma_map(&qf, &sm, 1.0, xb, yb);
        assert_eq!(inv.len(), xb * yb);
        // All entries should match (uniform input).
        let v0 = inv[0];
        for &v in &inv {
            assert_eq!(v, v0);
        }
        assert!(v0.is_finite() && v0 != 0.0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_epf_step1_uniform_passthrough_gpu() {
        // EPF on a uniform image should not change pixel values
        // (all SAD=0, all weights=1, weighted average == original).
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 16_u32;
        let h = 16_u32;
        let pad = 4_u32;
        let pw = (w + 2 * pad) as usize;
        let ph = (h + 2 * pad) as usize;
        let in_x = vec![0.5_f32; pw * ph];
        let in_y = vec![0.3_f32; pw * ph];
        let in_b = vec![0.7_f32; pw * ph];
        let xb = (w / 8) as usize;
        let yb = (h / 8) as usize;
        let inv_sigma = vec![-1.0_f32; xb * yb]; // arbitrary negative
        let (ox, oy, ob) = apply_epf_step1_gpu(
            &enc,
            &in_x,
            &in_y,
            &in_b,
            &inv_sigma,
            w,
            h,
            xb as u32,
            yb as u32,
            pad,
            1.0,
            EPF_BORDER_SAD_MUL,
        );
        assert_eq!(ox.len(), (w * h) as usize);
        for &v in &ox {
            assert!((v - 0.5).abs() < 1e-4, "X drifted: {v}");
        }
        for &v in &oy {
            assert!((v - 0.3).abs() < 1e-4, "Y drifted: {v}");
        }
        for &v in &ob {
            assert!((v - 0.7).abs() < 1e-4, "B drifted: {v}");
        }
    }
}
