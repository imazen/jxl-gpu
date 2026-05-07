// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/adaptive_quant.rs (BSD-3-Clause via
// libjxl + AGPL/commercial), with the SIMD calls substituted for GPU
// launches.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted pieces of the adaptive-quant pipeline.
//!
//! Currently covers the cleanest substitutable units:
//! - `compute_mask1x1_gpu` — Y-plane Laplacian + Symmetric5 blur
//! - `compute_pre_erosion_gpu` — Y-plane limit clamp + 4× downsample
//! - `fuzzy_erosion_gpu` — 3×3 min-of-4 weighted sum + 2× downsample
//! - `per_block_modulations_gpu` — mask/gamma/hf/blue modulations on aq_map
//!
//! With `fuzzy_erosion_gpu` (added 2026-05-07), the full
//! adaptive_quant chain (`mask1x1 → pre_erosion → fuzzy_erosion →
//! per_block_modulations`) runs end-to-end on GPU.
//!
//! Reshape vs upstream `jxl_encoder::vardct::adaptive_quant`:
//! - `compute_mask1x1`: original SIMD allocates a scratch buffer per
//!   call. GPU version goes (Y → mask) → (mask → blurred-mask) as two
//!   back-to-back GPU launches; the intermediate is a `Vec<f32>` we
//!   re-upload. Future fusion: a single GPU kernel that applies the
//!   blur on-die before download.
//! - `per_block_modulations`: same per-block math, run on the GPU. Full
//!   image rect (rect_x0/y0=0, rect_w/h=full) since GPU doesn't benefit
//!   from per-tile launches at the sizes we care about.

use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

/// libjxl mask1x1 Symmetric5 blur weights (from
/// `enc_adaptive_quantization.cc::kFilterMask1x1`).
const W_R: f32 = 0.364_911_248;
const W_D: f32 = 0.05;
const W_R2: f32 = 0.168_888_802_1;
const W_L: f32 = 0.221_069_183;
const W_D2: f32 = 0.306_563_504;

/// GPU `compute_mask1x1`. Mirrors upstream
/// `jxl_encoder::vardct::adaptive_quant::compute_mask1x1`.
///
/// 1. Per-pixel Laplacian masking field on the Y plane (GPU mask1x1_field).
/// 2. Symmetric5 blur with mask1x1-specific weights (GPU
///    gaborish_5x5_channel; same kernel structure, different weights).
pub fn compute_mask1x1_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_y: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    assert_eq!(xyb_y.len(), width * height);

    // Step 1: Laplacian per-pixel masking field.
    let raw_mask = enc.mask1x1_field(xyb_y, width as u32, height as u32);

    // Step 2: Symmetric5 blur with mask1x1-specific weights.
    let sum = 1.0 + 4.0 * (W_R + W_D + W_R2 + W_D2 + 2.0 * W_L);
    let inv_sum = 1.0 / sum;
    enc.gaborish_5x5_channel(
        &raw_mask,
        width as u32,
        height as u32,
        inv_sum,        // wc
        inv_sum * W_R,  // wr
        inv_sum * W_D,  // wd
        inv_sum * W_R2, // w_big_r
        inv_sum * W_L,  // wl
        inv_sum * W_D2, // w_big_d
    )
}

/// GPU `compute_pre_erosion`. Direct passthrough to the GPU kernel.
/// Returns `(pre_erosion, pre_erosion_w, pre_erosion_h)` matching upstream.
pub fn compute_pre_erosion_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_y: &[f32],
    width: usize,
    height: usize,
    tile_x0_pixels: usize,
    tile_y0_pixels: usize,
    tile_w: usize,
    tile_h: usize,
) -> (Vec<f32>, usize, usize) {
    // libjxl pre-erosion downsamples 4× in each direction.
    // tile_w / 4 (rounded up) gives the output width; same for height.
    let pre_erosion_w = tile_w.div_ceil(4);
    let pre_erosion_h = tile_h.div_ceil(4);
    let out = enc.pre_erosion(
        xyb_y,
        width as u32,
        height as u32,
        tile_x0_pixels as u32,
        tile_y0_pixels as u32,
        pre_erosion_w as u32,
        pre_erosion_h as u32,
    );
    (out, pre_erosion_w, pre_erosion_h)
}

/// GPU `fuzzy_erosion`. Mirrors upstream
/// `jxl_encoder::vardct::adaptive_quant::fuzzy_erosion`. Returns
/// `(out, out_w, out_h)` where `out_w = region_w / 2` and
/// `out_h = region_h / 2`.
///
/// `butteraugli_target` derives the per-position k_mul weights via
/// the same formula libjxl uses (see `forks::adaptive_quant_kmul`
/// for the formula).
#[allow(clippy::too_many_arguments)]
pub fn fuzzy_erosion_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    src: &[f32],
    src_w: usize,
    src_h: usize,
    from_x0: usize,
    from_y0: usize,
    region_w: usize,
    region_h: usize,
    butteraugli_target: f32,
) -> (Vec<f32>, usize, usize) {
    let (out, out_w, out_h) = enc.fuzzy_erosion_plane(
        src,
        src_w as u32,
        src_h as u32,
        from_x0 as u32,
        from_y0 as u32,
        region_w as u32,
        region_h as u32,
        butteraugli_target,
    );
    (out, out_w as usize, out_h as usize)
}

/// GPU `per_block_modulations`. Mirrors upstream signature; mutates
/// `aq_map` in place via the encoder's apply_per_block_modulations.
#[allow(clippy::too_many_arguments)]
pub fn per_block_modulations_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    butteraugli_target: f32,
    scale: f32,
    rect_x0_blocks: usize,
    rect_y0_blocks: usize,
    rect_w_blocks: usize,
    rect_h_blocks: usize,
    aq_map: &mut [f32],
    aq_map_w: usize,
) {
    enc.apply_per_block_modulations(
        xyb_x,
        xyb_y,
        xyb_b,
        aq_map,
        stride as u32,
        aq_map_w as u32,
        rect_x0_blocks as u32,
        rect_y0_blocks as u32,
        rect_w_blocks as u32,
        rect_h_blocks as u32,
        butteraugli_target,
        scale,
    );
}

/// Convert float quant field to u8 raw_quant — bit-for-bit copy of
/// upstream `quantize_quant_field` (no GPU substitution; pure scalar
/// loop, runs in microseconds).
pub fn quantize_quant_field(quant_field_float: &[f32], inv_scale: f32) -> Vec<u8> {
    quant_field_float
        .iter()
        .map(|&qf| {
            let val = (qf * inv_scale + 0.5) as i32;
            val.clamp(1, 255) as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_mask1x1_uniform_image_gpu() {
        // Constant Y plane → constant mask after blur.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 16;
        let h = 16;
        let y = vec![0.5_f32; w * h];
        let mask = compute_mask1x1_gpu(&enc, &y, w, h);
        let first = mask[0];
        for &m in &mask {
            assert!(
                (m - first).abs() < 1e-3,
                "mask diverges on uniform input: {m} vs {first}"
            );
            assert!(m.is_finite() && m > 0.0);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_pre_erosion_passthrough_gpu() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        // 32x32 → 8x8 (4× downsample)
        let w = 32;
        let h = 32;
        let y: Vec<f32> = (0..w * h).map(|i| (i as f32) * 0.001).collect();
        let (out, ow, oh) = compute_pre_erosion_gpu(&enc, &y, w, h, 0, 0, w, h);
        assert_eq!(ow, 8);
        assert_eq!(oh, 8);
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fuzzy_erosion_gpu_shape() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        // 32x32 src, region 16x16 → output 8x8 (2× downsample on the region)
        let w = 32;
        let h = 32;
        let src: Vec<f32> = (0..w * h)
            .map(|i| 0.3 + 0.4 * (i as f32 * 0.01).sin())
            .collect();
        let (out, ow, oh) = fuzzy_erosion_gpu(&enc, &src, w, h, 4, 4, 16, 16, 1.0);
        assert_eq!(ow, 8);
        assert_eq!(oh, 8);
        assert_eq!(out.len(), 64);
        // fuzzy_erosion is a weighted sum of mins — sign tracks input.
        for &v in &out {
            assert!(v.is_finite(), "fuzzy_erosion output must be finite");
        }
    }

    /// End-to-end full adaptive_quant chain on GPU:
    ///   mask1x1 → pre_erosion → fuzzy_erosion → (per_block_modulations
    ///   path stays in the upper-level encoder logic for now)
    /// Just verifies the compositions produce finite values.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_full_aq_chain_gpu() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 64;
        let h = 64;
        let xyb_y: Vec<f32> = (0..w * h)
            .map(|i| 0.3 + 0.2 * (i as f32 * 0.013).sin())
            .collect();
        // Step 1: mask1x1
        let mask = compute_mask1x1_gpu(&enc, &xyb_y, w, h);
        assert_eq!(mask.len(), w * h);
        assert!(mask.iter().all(|v| v.is_finite()));
        // Step 2: pre_erosion (4× downsample)
        let (pre, pw, ph) = compute_pre_erosion_gpu(&enc, &xyb_y, w, h, 0, 0, w, h);
        assert_eq!(pw, 16);
        assert_eq!(ph, 16);
        // Step 3: fuzzy_erosion (2× downsample on the pre_erosion output)
        let (fuzz, fw, fh) = fuzzy_erosion_gpu(&enc, &pre, pw, ph, 0, 0, pw, ph, 1.0);
        assert_eq!(fw, 8);
        assert_eq!(fh, 8);
        assert!(fuzz.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_quantize_quant_field_matches_upstream_logic() {
        // Spot-check the bit-for-bit copy is correct.
        let qf = [0.5_f32, 1.0, 2.0, 100.0, 0.0];
        let inv_scale = 2.0;
        let out = quantize_quant_field(&qf, inv_scale);
        // 0.5*2 + 0.5 = 1.5 → 1
        // 1.0*2 + 0.5 = 2.5 → 2
        // 2.0*2 + 0.5 = 4.5 → 4
        // 100*2 + 0.5 = 200.5 → 200
        // 0*2 + 0.5 = 0.5 → 0 → clamped to 1
        assert_eq!(out, [1, 2, 4, 200, 1]);
    }
}
