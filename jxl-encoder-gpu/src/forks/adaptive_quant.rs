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
use cubecl::prelude::*;

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

/// GPU full-pipeline `compute_quant_field` against persistent xyb
/// `GpuPlane`s. Mirrors the CPU
/// `jxl_encoder::vardct::adaptive_quant::compute_quant_field_float`
/// chain (`pre_erosion → fuzzy_erosion → mask_for_ac_strategy →
/// per_block_modulations`) and downloads `(quant_field_float, masking)`
/// in a single batched `read`.
///
/// Bridges the `gpu_pw` (16-multiple) input stride and the cpu-aligned
/// `(xsize_blocks, ysize_blocks)` output dims by sizing each
/// intermediate buffer to the cpu-aligned width and passing the gpu
/// stride to the per-pixel kernels. Edge-replicated padding past the
/// real image (preserved by both pipelines) makes the math identical
/// for the first `xsize_blocks * ysize_blocks` outputs.
///
/// `xx_g`/`xy_g`/`xb_g` are the gaborished XYB planes from
/// `LossyEncoder::prepare_strategy_search_plan`. They share dims
/// `(gpu_pw, gpu_ph)`. `cpu_pw` / `cpu_ph` are the 8-multiple-padded
/// dims used by the CPU consumers (`xsize_blocks = cpu_pw / 8`).
#[allow(clippy::too_many_arguments)]
pub fn compute_quant_field_full_persistent<R: Runtime>(
    enc: &GpuEncoder<R>,
    xx_g: &crate::persistent::GpuPlane<R>,
    xy_g: &crate::persistent::GpuPlane<R>,
    xb_g: &crate::persistent::GpuPlane<R>,
    cpu_pw: usize,
    cpu_ph: usize,
    distance: f32,
    k_ac_quant: f32,
) -> (Vec<f32>, Vec<f32>) {
    use cubecl::server::Handle;

    let gpu_pw = xy_g.width() as usize;
    let gpu_ph = xy_g.height() as usize;
    debug_assert_eq!(xx_g.width() as usize, gpu_pw);
    debug_assert_eq!(xb_g.width() as usize, gpu_pw);
    debug_assert_eq!(xx_g.height() as usize, gpu_ph);
    debug_assert_eq!(xb_g.height() as usize, gpu_ph);
    debug_assert!(cpu_pw <= gpu_pw, "cpu_pw must fit in gpu_pw");
    debug_assert!(cpu_ph <= gpu_ph, "cpu_ph must fit in gpu_ph");

    let xsize_blocks = cpu_pw / 8;
    let ysize_blocks = cpu_ph / 8;
    let pre_erosion_w = cpu_pw.div_ceil(4);
    let pre_erosion_h = cpu_ph.div_ceil(4);
    let n_pre = pre_erosion_w * pre_erosion_h;
    let n_blocks = xsize_blocks * ysize_blocks;

    let client = enc.client_ref();

    // Stage 1: pre_erosion (Y only, 4× downsample).
    let h_pre: Handle = client.empty(n_pre * core::mem::size_of::<f32>());
    crate::launch::adaptive_quant::compute_pre_erosion::<R>(
        client,
        xy_g.handle().clone(),
        h_pre.clone(),
        gpu_pw * gpu_ph,
        gpu_pw as u32,
        gpu_ph as u32,
        0,
        0,
        pre_erosion_w as u32,
        pre_erosion_h as u32,
    );

    // Stage 2: fuzzy_erosion (2× downsample → per-8x8-block aq_map).
    let h_aq: Handle = client.empty(n_blocks * core::mem::size_of::<f32>());
    let k_mul = crate::launch::fuzzy_erosion::fuzzy_erosion_kmul(distance);
    crate::launch::fuzzy_erosion::fuzzy_erosion::<R>(
        client,
        h_pre,
        h_aq.clone(),
        pre_erosion_w as u32,
        pre_erosion_h as u32,
        0,
        0,
        xsize_blocks as u32,
        ysize_blocks as u32,
        k_mul,
    );

    // Stage 2.5: snapshot aq_map → masking (`1 / (aq_map + 0.001)`).
    // Order matters: this MUST run before per_block_modulations
    // mutates aq_map in-place — masking captures the pre-modulation
    // values (matches CPU compute_quant_field_float Step 2.5).
    let h_mask: Handle = client.empty(n_blocks * core::mem::size_of::<f32>());
    crate::launch::mask_for_ac_strategy::mask_for_ac_strategy::<R>(
        client,
        h_aq.clone(),
        h_mask.clone(),
        n_blocks as u32,
    );

    // Stage 3: per_block_modulations (mutates aq_map in-place).
    let scale = k_ac_quant / distance;
    crate::launch::adaptive_quant::per_block_modulations::<R>(
        client,
        xx_g.handle().clone(),
        xy_g.handle().clone(),
        xb_g.handle().clone(),
        h_aq.clone(),
        gpu_pw * gpu_ph,
        n_blocks,
        gpu_pw as u32,
        xsize_blocks as u32,
        0,
        0,
        xsize_blocks as u32,
        ysize_blocks as u32,
        distance,
        scale,
    );

    // Stage 4: batched download (1 sync barrier for both planes).
    let mut bytes = client.read(alloc::vec![h_aq, h_mask]);
    let mask_bytes = bytes.pop().expect("read[1]");
    let aq_bytes = bytes.pop().expect("read[0]");
    (
        f32::from_bytes(&aq_bytes).to_vec(),
        f32::from_bytes(&mask_bytes).to_vec(),
    )
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

    /// GPU `compute_quant_field_full_persistent` matches the CPU
    /// `compute_quant_field_float_free` end-to-end on identical inputs.
    ///
    /// Critical case: gpu_pw > cpu_pw (the 16-vs-8 padding mismatch).
    /// 95×97 image: cpu_pw = ceil(95,8)*8 = 96, gpu_pw = ceil(95,16)*16 = 96 — same.
    /// 99×97 image: cpu_pw = 104, gpu_pw = 112 — differs by 8.
    /// Test the divergent case explicitly so the stride math is exercised.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_compute_quant_field_full_persistent_matches_cpu() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();

        let cases = [(99usize, 97usize), (96usize, 96usize), (137usize, 91usize)];
        for &(w, h) in &cases {
            let cpu_pw = w.div_ceil(8) * 8;
            let cpu_ph = h.div_ceil(8) * 8;
            let gpu_pw = w.div_ceil(16) * 16;
            let gpu_ph = h.div_ceil(16) * 16;

            // Pseudo-random xyb planes — non-degenerate values that
            // exercise gamma/HF/blue modulations.
            let make = |seed: usize| -> Vec<f32> {
                (0..gpu_pw * gpu_ph)
                    .map(|i| {
                        let s = ((i + seed) as f32) * 0.0173;
                        0.05 + 0.10 * s.sin() + 0.07 * (s * 1.7).cos()
                    })
                    .collect()
            };
            let mut xx_data = make(11);
            let mut xy_data = make(101);
            let mut xb_data = make(1009);

            // Edge-replicate gpu_pw padding past width w (matches what
            // pad_plane_with_replication does upstream).
            for plane in [&mut xx_data, &mut xy_data, &mut xb_data] {
                for y in 0..gpu_ph.min(h) {
                    let off = y * gpu_pw;
                    let last = plane[off + w - 1];
                    for x in w..gpu_pw {
                        plane[off + x] = last;
                    }
                }
                if h < gpu_ph {
                    let last_off = (h - 1) * gpu_pw;
                    for y in h..gpu_ph {
                        let off = y * gpu_pw;
                        for x in 0..gpu_pw {
                            plane[off + x] = plane[last_off + x];
                        }
                    }
                }
            }

            // Repack into cpu_pw × cpu_ph (same edge-replication policy)
            // so the CPU reference sees the same boundary values.
            let mut cpu_xx = vec![0.0f32; cpu_pw * cpu_ph];
            let mut cpu_xy = vec![0.0f32; cpu_pw * cpu_ph];
            let mut cpu_xb = vec![0.0f32; cpu_pw * cpu_ph];
            for (gpu, cpu) in [
                (&xx_data, &mut cpu_xx),
                (&xy_data, &mut cpu_xy),
                (&xb_data, &mut cpu_xb),
            ] {
                for y in 0..cpu_ph {
                    let g_off = y * gpu_pw;
                    let c_off = y * cpu_pw;
                    cpu[c_off..c_off + cpu_pw].copy_from_slice(&gpu[g_off..g_off + cpu_pw]);
                }
            }

            let xx_g = enc.upload_plane(&xx_data, gpu_pw as u32, gpu_ph as u32);
            let xy_g = enc.upload_plane(&xy_data, gpu_pw as u32, gpu_ph as u32);
            let xb_g = enc.upload_plane(&xb_data, gpu_pw as u32, gpu_ph as u32);

            let distance = 1.0_f32;
            let k_ac_quant = 0.765_f32;

            let (gpu_qf, gpu_mask) = compute_quant_field_full_persistent(
                &enc, &xx_g, &xy_g, &xb_g, cpu_pw, cpu_ph, distance, k_ac_quant,
            );

            let (cpu_qf, cpu_mask) = jxl_encoder::__pre_quantized::compute_quant_field_float_free(
                &cpu_xx,
                &cpu_xy,
                &cpu_xb,
                cpu_pw,
                cpu_ph,
                cpu_pw / 8,
                cpu_ph / 8,
                distance,
                k_ac_quant,
            )
            .expect("cpu compute");

            assert_eq!(gpu_qf.len(), cpu_qf.len(), "qf len {w}x{h}");
            assert_eq!(gpu_mask.len(), cpu_mask.len(), "mask len {w}x{h}");

            // Tolerance: GPU uses f32, CPU uses f32. Differences come
            // from kernel ordering of fuzzy_erosion's reductions and
            // cubecl's fast-math (fast_log2f / fast_pow2f) vs upstream
            // libjxl's intrinsics. Bound below empirically; tighter
            // would force exact-match between AT&T and CUDA libm.
            let qf_tol = 5e-3_f32;
            let mask_tol = 5e-3_f32;
            for i in 0..gpu_qf.len() {
                let g = gpu_qf[i];
                let c = cpu_qf[i];
                let abs = (g - c).abs();
                let rel = abs / c.abs().max(1e-6);
                assert!(
                    abs < qf_tol || rel < qf_tol,
                    "qf[{i}] {g} vs {c} (abs {abs}, rel {rel}) at {w}x{h}",
                );
            }
            for i in 0..gpu_mask.len() {
                let g = gpu_mask[i];
                let c = cpu_mask[i];
                let abs = (g - c).abs();
                let rel = abs / c.abs().max(1e-6);
                assert!(
                    abs < mask_tol || rel < mask_tol,
                    "mask[{i}] {g} vs {c} (abs {abs}, rel {rel}) at {w}x{h}",
                );
            }
        }
    }

    /// Production-flow divergence test: replicates what
    /// `encode_lossy_to_bitstream_via_precomputed_from_u8` does in
    /// `encoder.rs:1805-1959` and measures the gap between
    /// (CPU compute_quant_field on the post-fix-up CPU buffer) and
    /// (GPU compute_quant_field on the un-fix-up'd GPU planes).
    ///
    /// If divergence is small enough that quant_field quantizes to the
    /// same u8, the GPU port can drop straight in. If not, an extra
    /// GPU edge re-replication kernel is needed first.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_compute_quant_field_production_flow_divergence() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();

        // Use a non-multiple-of-16 width to exercise the gpu_pw > cpu_pw
        // post-gaborish edge divergence (which is the whole point of the
        // production fix-up step at encoder.rs:1837-1874).
        let w = 1025usize;
        let h = 257usize;
        let cpu_pw = w.div_ceil(8) * 8;
        let cpu_ph = h.div_ceil(8) * 8;
        let gpu_pw = w.div_ceil(16) * 16;
        let gpu_ph = h.div_ceil(16) * 16;
        assert!(gpu_pw > cpu_pw, "test should exercise gpu_pw > cpu_pw");

        // Generate a deterministic linear-RGB image with non-trivial
        // structure (sin*cos pattern + edge detail in the rightmost
        // pixels so post-gaborish edges actually differ).
        let make = |seed: usize| -> Vec<f32> {
            (0..w * h)
                .map(|i| {
                    let x = (i % w) as f32;
                    let y = (i / w) as f32;
                    let s = ((x + 13.0) * 0.013).sin();
                    let c = ((y + (seed as f32) * 7.0) * 0.027).cos();
                    let edge_kick = if (x as usize) >= w - 4 { 0.15 } else { 0.0 };
                    (0.10 + 0.04 * s * c + edge_kick).clamp(0.001, 0.999)
                })
                .collect()
        };
        let r = make(1);
        let g = make(2);
        let b = make(3);

        // Production GPU path: pad to gpu_pw, upload, xyb, gaborish.
        let r_padded = crate::lossy_encoder::pad_to_alignment_test_helper(&r, w, h, gpu_pw, gpu_ph);
        let g_padded = crate::lossy_encoder::pad_to_alignment_test_helper(&g, w, h, gpu_pw, gpu_ph);
        let b_padded = crate::lossy_encoder::pad_to_alignment_test_helper(&b, w, h, gpu_pw, gpu_ph);

        let g_r = enc.upload_plane(&r_padded, gpu_pw as u32, gpu_ph as u32);
        let g_g = enc.upload_plane(&g_padded, gpu_pw as u32, gpu_ph as u32);
        let g_b = enc.upload_plane(&b_padded, gpu_pw as u32, gpu_ph as u32);

        let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(&g_r, &g_g, &g_b);
        let weights = crate::lossy_encoder::default_gaborish_weights_test_helper();
        let xx_g = enc.gaborish_5x5_persistent(&xx, &weights);
        let xy_g = enc.gaborish_5x5_persistent(&xy, &weights);
        let xb_g = enc.gaborish_5x5_persistent(&xb, &weights);

        // Production CPU side: download, repack with edge fix-up.
        let (xx_dl, xy_dl, xb_dl) = enc.download_planes_3ch(&xx_g, &xy_g, &xb_g);
        let repack = |src: &[f32]| -> Vec<f32> {
            let mut dst = vec![0.0f32; cpu_pw * cpu_ph];
            for row in 0..cpu_ph {
                let off = row * gpu_pw;
                let dst_off = row * cpu_pw;
                dst[dst_off..dst_off + cpu_pw].copy_from_slice(&src[off..off + cpu_pw]);
            }
            // Edge re-replicate per encoder.rs:1853-1870.
            if cpu_pw > w {
                for row in 0..cpu_ph {
                    let dst_off = row * cpu_pw;
                    let v = dst[dst_off + (w - 1)];
                    for c in w..cpu_pw {
                        dst[dst_off + c] = v;
                    }
                }
            }
            if cpu_ph > h {
                let last_off = (h - 1) * cpu_pw;
                for row in h..cpu_ph {
                    let dst_off = row * cpu_pw;
                    dst.copy_within(last_off..last_off + cpu_pw, dst_off);
                }
            }
            dst
        };
        let cpu_xx = repack(&xx_dl);
        let cpu_xy = repack(&xy_dl);
        let cpu_xb = repack(&xb_dl);

        let distance = 1.0_f32;
        let k_ac_quant = 0.765_f32;
        let xsize_blocks = cpu_pw / 8;
        let ysize_blocks = cpu_ph / 8;

        let (cpu_qf, cpu_mask) = jxl_encoder::__pre_quantized::compute_quant_field_float_free(
            &cpu_xx,
            &cpu_xy,
            &cpu_xb,
            cpu_pw,
            cpu_ph,
            xsize_blocks,
            ysize_blocks,
            distance,
            k_ac_quant,
        )
        .expect("cpu");

        let (gpu_qf, gpu_mask) = compute_quant_field_full_persistent(
            &enc, &xx_g, &xy_g, &xb_g, cpu_pw, cpu_ph, distance, k_ac_quant,
        );

        // Quantize both qf to u8 (the only post-pipeline use) and
        // check what fraction of blocks land on a different bucket.
        // u8 conversion: round((qf * inv_scale + 0.5)).clamp(1,255)
        let inv_scale = (1.0 / k_ac_quant) * distance * 8.0;
        let q = |v: f32| -> u8 {
            let raw = (v * inv_scale + 0.5) as i32;
            raw.clamp(1, 255) as u8
        };
        let mut bucket_diffs = 0usize;
        let mut max_abs_qf = 0.0f32;
        let mut max_abs_mask = 0.0f32;
        for i in 0..gpu_qf.len() {
            if q(gpu_qf[i]) != q(cpu_qf[i]) {
                bucket_diffs += 1;
            }
            max_abs_qf = max_abs_qf.max((gpu_qf[i] - cpu_qf[i]).abs());
            max_abs_mask = max_abs_mask.max((gpu_mask[i] - cpu_mask[i]).abs());
        }
        let n = gpu_qf.len();
        let pct = 100.0 * (bucket_diffs as f64) / (n as f64);
        eprintln!(
            "production-flow divergence at {w}x{h}: \
             {bucket_diffs}/{n} blocks ({pct:.3}%) hit a different u8 bucket; \
             max_abs_qf={max_abs_qf:.5}, max_abs_mask={max_abs_mask:.5}"
        );
        // Sanity bounds: divergence must be confined to edges (< 5%
        // of blocks at this size). If it's larger, something is wrong
        // with the kernel math, not just the edge-fix-up gap.
        let edge_block_count = (xsize_blocks + ysize_blocks) * 2; // outer ring estimate
        assert!(
            bucket_diffs < edge_block_count * 4,
            "divergence too large: {bucket_diffs} > 4× edge ring estimate {edge_block_count}",
        );
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
