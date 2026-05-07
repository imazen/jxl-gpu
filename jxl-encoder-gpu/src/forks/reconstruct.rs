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
