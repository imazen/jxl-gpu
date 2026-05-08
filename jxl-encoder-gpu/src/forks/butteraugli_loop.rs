// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/butteraugli_loop.rs (BSD-3-Clause via libjxl
// + AGPL/commercial), with the per-iteration butteraugli call substituted
// for `zenmetrics/butteraugli-gpu`.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted butteraugli quant-refinement loop.
//!
//! Mirrors upstream `jxl_encoder::vardct::butteraugli_loop::
//! VarDctEncoder::butteraugli_refine_quant_field` shape but with
//! the per-iteration distance compute on GPU via
//! [`zenmetrics::butteraugli-gpu`](https://github.com/imazen/zenmetrics).
//!
//! ## Status
//!
//! **Scaffold landed; full loop port pending.** The
//! [`ButteraugliLoopGpu`] struct holds a persistent
//! `butteraugli_gpu::Butteraugli` compute instance (so the reference
//! image is uploaded once and cached across iterations). The
//! per-iteration `compute_distmap` returns a per-pixel diffmap that
//! a future `refine_quant_field_gpu` orchestrator will consume to
//! adjust per-block quant_field, mirroring upstream's
//! FindBestQuantization algorithm.
//!
//! ## Why a fork instead of editing jxl-encoder
//!
//! Same pattern as `forks::epf::compute_epf_sharpness_dct8_gpu` and
//! the rest of `forks::*` — re-implement upstream's orchestration
//! in fork-space using GPU primitives. Zero edits to jxl-encoder.
//!
//! ## Reshape vs upstream
//!
//! - Upstream calls CPU butteraugli per iteration, which dominates
//!   encode time at effort >= 8 (100-500 ms per iter at 1MP × 2-4
//!   iters per encode). The GPU variant is single-digit ms per iter.
//! - Persistent reference: `set_reference(orig)` once per encode,
//!   then `compute_with_reference(recon)` per iteration. Caches the
//!   reference's opsin/blur intermediates so re-running on the same
//!   ref is fast.
//! - The diffmap is per-pixel f32; a host-side reduction maps it to
//!   per-block tile distances for the quant-field perturbation
//!   (matching upstream's TileDistance computation).

use alloc::vec::Vec;

use butteraugli_gpu::{Butteraugli, ButteraugliParams, GpuButteraugliResult};
use cubecl::Runtime;

use crate::encoder::GpuEncoder;

/// Persistent butteraugli compute state for the iterative quant
/// refinement loop. Construct once per encode (with the original
/// image as reference); re-use across iterations.
///
/// Wraps `butteraugli_gpu::Butteraugli`; exposes a fork-friendly API
/// that takes our `GpuEncoder` for the cubecl client.
pub struct ButteraugliLoopGpu<R: Runtime> {
    inner: Butteraugli<R>,
    width: u32,
    height: u32,
}

impl<R: Runtime> ButteraugliLoopGpu<R> {
    /// Construct a new butteraugli compute state for an `(width, height)`
    /// image. Allocates persistent GPU buffers; reuse via
    /// [`Self::set_reference`] + [`Self::compute_with_reference`] per
    /// iteration.
    pub fn new(enc: &GpuEncoder<R>, width: u32, height: u32) -> Self {
        let inner = Butteraugli::new(enc.client().clone(), width, height);
        Self {
            inner,
            width,
            height,
        }
    }

    /// Multi-resolution variant — matches upstream's default for the
    /// butteraugli-loop feature.
    pub fn new_multires(enc: &GpuEncoder<R>, width: u32, height: u32) -> Self {
        let inner = Butteraugli::new_multires(enc.client().clone(), width, height);
        Self {
            inner,
            width,
            height,
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Upload the reference (original) image once. The internal
    /// opsin/blur intermediates are cached so subsequent
    /// [`Self::compute_with_reference`] calls only need to re-run on
    /// the distorted image.
    ///
    /// `ref_srgb` is interleaved sRGB U8 (`width * height * 3` bytes).
    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> butteraugli_gpu::Result<()> {
        self.inner.set_reference(ref_srgb)
    }

    /// Compute butteraugli result against the cached reference. Call
    /// after [`Self::set_reference`]. Returns the per-pixel diffmap +
    /// score.
    pub fn compute_with_reference(
        &mut self,
        dist_srgb: &[u8],
    ) -> butteraugli_gpu::Result<GpuButteraugliResult> {
        self.inner.compute_with_reference(dist_srgb)
    }

    /// One-shot compute (no caching). For non-iterative callers.
    pub fn compute(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
    ) -> butteraugli_gpu::Result<GpuButteraugliResult> {
        self.inner.compute(ref_srgb, dist_srgb)
    }

    /// One-shot compute with explicit params override. Pass-through to
    /// `butteraugli_gpu::Butteraugli::compute_with_options`.
    pub fn compute_with_options(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        params: &ButteraugliParams,
    ) -> butteraugli_gpu::Result<GpuButteraugliResult> {
        self.inner.compute_with_options(ref_srgb, dist_srgb, params)
    }
}

/// Per-block AC strategy info for tile-distance reduction.
///
/// AC strategies cover variable block extents (DCT8 = 1×1, DCT16x16 =
/// 2×2, DCT32x32 = 4×4, etc.). Upstream's tile-distance computation
/// visits only the FIRST 8×8 block of each AC strategy, accumulates the
/// 16th-power norm over the full pixel rectangle the strategy covers,
/// and splats the resulting tile distance to every covered (8×8) block.
///
/// Callers pass `is_first[bi]` (true for the first 8×8 block of an AC
/// strategy) plus `covered_x[bi]` / `covered_y[bi]` (in 8×8 block units;
/// 1 for DCT8, 2 for DCT16, 4 for DCT32, etc.). For DCT8-only callers,
/// pass `is_first=[true; N], covered_x=[1; N], covered_y=[1; N]`.
pub struct AcStrategyInfo<'a> {
    pub is_first: &'a [bool],
    pub covered_x: &'a [u8],
    pub covered_y: &'a [u8],
}

/// Convenience: build an [`AcStrategyInfo`] view treating every (8×8)
/// block as its own DCT8 strategy. Backing storage must outlive the
/// returned info.
///
/// Use [`dct8_only_storage`] to allocate the three vecs in one call.
pub fn dct8_only_info<'a>(
    is_first: &'a [bool],
    covered_x: &'a [u8],
    covered_y: &'a [u8],
) -> AcStrategyInfo<'a> {
    AcStrategyInfo {
        is_first,
        covered_x,
        covered_y,
    }
}

/// Allocate the three backing vecs for a DCT8-only [`AcStrategyInfo`].
pub fn dct8_only_storage(num_blocks: usize) -> (Vec<bool>, Vec<u8>, Vec<u8>) {
    (
        alloc::vec![true; num_blocks],
        alloc::vec![1u8; num_blocks],
        alloc::vec![1u8; num_blocks],
    )
}

/// Per-block tile-distance constant. Matches upstream
/// `K_TILE_NORM = 1.2` in `vardct::butteraugli_loop` line 231.
pub const K_TILE_NORM: f32 = 1.2;

/// Compute per-(8×8) tile distances from a per-pixel butteraugli diffmap,
/// faithful to upstream's `vardct::butteraugli_loop::butteraugli_refine_quant_field`
/// lines 230-271.
///
/// For each AC strategy (one per `is_first[bi] == true`), accumulates the
/// f64 16th-power-mean over the strategy's pixel rectangle, takes the
/// 16th root, scales by [`K_TILE_NORM`], and splats the result to every
/// (8×8) block the strategy covers.
///
/// Algorithm: `td = K_TILE_NORM * (mean_pixels(v^16))^(1/16)`.
/// (Equivalently, the L16 norm divided by `pixels^(1/16)` and scaled by
/// 1.2.) The 16th-power emphasis makes a single peak pixel dominate the
/// per-tile distance, matching butteraugli's perceptual prioritization.
///
/// `width` / `height` are the DIFFMAP dimensions (= source image dims).
/// Strategy rectangles are clipped to `(width, height)` — partial blocks
/// at the right/bottom contribute fewer pixels but the same accumulator.
pub fn compute_tile_distances(
    diffmap: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    info: &AcStrategyInfo<'_>,
) -> Vec<f32> {
    debug_assert_eq!(diffmap.len(), width * height);
    let num_blocks = xsize_blocks * ysize_blocks;
    debug_assert_eq!(info.is_first.len(), num_blocks);
    debug_assert_eq!(info.covered_x.len(), num_blocks);
    debug_assert_eq!(info.covered_y.len(), num_blocks);

    let mut out = alloc::vec![0.0_f32; num_blocks];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let bi = by * xsize_blocks + bx;
            if !info.is_first[bi] {
                continue;
            }
            let cx = info.covered_x[bi] as usize;
            let cy = info.covered_y[bi] as usize;
            let px_start_x = bx * 8;
            let px_start_y = by * 8;
            let px_end_x = ((bx + cx) * 8).min(width);
            let px_end_y = ((by + cy) * 8).min(height);
            if px_start_x >= width || px_start_y >= height {
                continue;
            }
            let mut dist_norm = 0.0_f64;
            let mut pixels = 0.0_f64;
            for py in px_start_y..px_end_y {
                for px in px_start_x..px_end_x {
                    let v = diffmap[py * width + px] as f64;
                    let v2 = v * v;
                    let v4 = v2 * v2;
                    let v8 = v4 * v4;
                    let v16 = v8 * v8;
                    dist_norm += v16;
                    pixels += 1.0;
                }
            }
            if pixels == 0.0 {
                pixels = 1.0;
            }
            let td = K_TILE_NORM * (dist_norm / pixels).sqrt().sqrt().sqrt().sqrt() as f32;
            for sy in 0..cy {
                for sx in 0..cx {
                    let oy = by + sy;
                    let ox = bx + sx;
                    if oy < ysize_blocks && ox < xsize_blocks {
                        out[oy * xsize_blocks + ox] = td;
                    }
                }
            }
        }
    }
    out
}

/// Per-block deviation bounds for the per-iteration quant-field
/// adjustment. Matches upstream `vardct::butteraugli_loop` lines 105-122.
///
/// The bounds are derived from the FLOAT initial quant field (range
/// /// before any per-iteration adjustment); they prevent the per-iteration
/// adjuster from diverging too far from the initial calibration.
#[derive(Debug, Clone, Copy)]
pub struct DeviationBounds {
    pub qf_lower: f32,
    pub qf_higher: f32,
}

impl DeviationBounds {
    /// Compute deviation bounds from the float initial quant field.
    /// Mirrors upstream lines 107-122 exactly.
    pub fn compute(initial_quant_field_float: &[f32]) -> Self {
        let initial_qf_min = initial_quant_field_float
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or(0.01)
            .max(1e-6);
        let initial_qf_max = initial_quant_field_float
            .iter()
            .copied()
            .reduce(f32::max)
            .unwrap_or(1.0);
        let initial_qf_ratio = initial_qf_max / initial_qf_min;
        let qf_max_deviation_low = (250.0_f32 / initial_qf_ratio).sqrt();
        let asymmetry = 2.0_f32.min(qf_max_deviation_low);
        let qf_lower = initial_qf_min / (asymmetry * qf_max_deviation_low);
        let qf_higher = initial_qf_max * (qf_max_deviation_low / asymmetry);
        Self {
            qf_lower,
            qf_higher,
        }
    }
}

/// Constant for the kOriginalComparisonRound clamp blend. Matches
/// upstream `K_INIT_MUL = 0.6` in `vardct::butteraugli_loop` line 319.
pub const K_INIT_MUL: f64 = 0.6;

/// Iteration index at which the kOriginalComparisonRound clamp fires.
/// Upstream applies the clamp once, on iter == 1 (before the iter == 1
/// adjustment).
pub const K_ORIGINAL_COMPARISON_ROUND: usize = 1;

/// Apply the kOriginalComparisonRound clamp toward the initial field.
/// Mirrors upstream lines 314-336.
///
/// Only fires when the current per-block qf has DROPPED below the
/// blended target `(1 - K_INIT_MUL) * cur + K_INIT_MUL * init`. Bumps
/// it back up to the blend, then clamps to the deviation bounds. This
/// prevents oscillation in subsequent iterations by keeping qf from
/// diverging too far below its initial calibration.
pub fn clamp_toward_initial(
    quant_field_float: &mut [f32],
    initial_quant_field_float: &[f32],
    bounds: DeviationBounds,
) {
    debug_assert_eq!(quant_field_float.len(), initial_quant_field_float.len());
    let one_minus_init_mul = 1.0 - K_INIT_MUL;
    for bi in 0..quant_field_float.len() {
        let init_qf = initial_quant_field_float[bi] as f64;
        let cur_qf = quant_field_float[bi] as f64;
        let clamp_val = one_minus_init_mul * cur_qf + K_INIT_MUL * init_qf;
        if cur_qf < clamp_val {
            let mut v = clamp_val as f32;
            if v > bounds.qf_higher {
                v = bounds.qf_higher;
            }
            if v < bounds.qf_lower {
                v = bounds.qf_lower;
            }
            quant_field_float[bi] = v;
        }
    }
}

/// Per-iteration quant-field adjustment based on per-block tile
/// distances. Mirrors upstream `vardct::butteraugli_loop` lines 338-406.
///
/// Two regimes selected by `iter`:
/// - `iter < 2` → `cur_pow = 0.2`: adjust BOTH directions. Good blocks
///   (`tile_dist[bi] / target_distance <= 1`) are gently softened to
///   save bits via `qf *= diff^0.2`. Bad blocks get the `qf *= diff`
///   bump plus the integer-quantizer-step minimum guarantee.
/// - `iter >= 2` → `cur_pow = 0.0`: adjust ONLY bad blocks. Same `qf
///   *= diff` plus integer-step bump; good blocks are left alone.
///
/// `inv_global_scale` and `quantizer_scale` come from the CURRENT
/// iteration's [`DistanceParams`] (recomputed each iteration as the
/// global_scale shifts with the quant field).
///
/// The integer-step bump (lines 366-371, 392-397) ensures that a bad
/// block's adjustment changes the rounded integer quantizer value by
/// at least one step — preventing a "floating-point bump that rounds
/// to the same integer" no-op.
pub fn adjust_quant_field(
    quant_field_float: &mut [f32],
    tile_dist: &[f32],
    target_distance: f32,
    iter: usize,
    bounds: DeviationBounds,
    inv_global_scale: f32,
    quantizer_scale: f32,
) {
    debug_assert_eq!(quant_field_float.len(), tile_dist.len());
    let cur_pow: f64 = if iter < 2 { 0.2 } else { 0.0 };

    if cur_pow == 0.0 {
        for bi in 0..quant_field_float.len() {
            let diff = tile_dist[bi] / target_distance;
            if diff > 1.0 {
                let old = quant_field_float[bi];
                quant_field_float[bi] = old * diff;
                let qf_old = (old * inv_global_scale + 0.5).floor() as i32;
                let qf_new = (quant_field_float[bi] * inv_global_scale + 0.5).floor() as i32;
                if qf_old == qf_new {
                    quant_field_float[bi] = old + quantizer_scale;
                }
            }
            if quant_field_float[bi] > bounds.qf_higher {
                quant_field_float[bi] = bounds.qf_higher;
            }
            if quant_field_float[bi] < bounds.qf_lower {
                quant_field_float[bi] = bounds.qf_lower;
            }
        }
    } else {
        for bi in 0..quant_field_float.len() {
            let diff = tile_dist[bi] / target_distance;
            if diff <= 1.0 {
                quant_field_float[bi] *= (diff as f64).powf(cur_pow) as f32;
            } else {
                let old = quant_field_float[bi];
                quant_field_float[bi] = old * diff;
                let qf_old = (old * inv_global_scale + 0.5).floor() as i32;
                let qf_new = (quant_field_float[bi] * inv_global_scale + 0.5).floor() as i32;
                if qf_old == qf_new {
                    quant_field_float[bi] = old + quantizer_scale;
                }
            }
            if quant_field_float[bi] > bounds.qf_higher {
                quant_field_float[bi] = bounds.qf_higher;
            }
            if quant_field_float[bi] < bounds.qf_lower {
                quant_field_float[bi] = bounds.qf_lower;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== compute_tile_distances =====

    /// Uniform diffmap → every tile distance == K_TILE_NORM * v.
    /// (The 16th-power-mean of a constant equals that constant; 16th
    /// root recovers it; K_TILE_NORM = 1.2 multiplies.)
    #[test]
    fn test_compute_tile_distances_uniform() {
        let w = 16;
        let h = 16;
        let xb = 2;
        let yb = 2;
        let diffmap = alloc::vec![0.42_f32; w * h];
        let (is_first, cx, cy) = dct8_only_storage(xb * yb);
        let info = dct8_only_info(&is_first, &cx, &cy);
        let out = compute_tile_distances(&diffmap, w, h, xb, yb, &info);
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!((v - K_TILE_NORM * 0.42).abs() < 1e-5, "got {v}");
        }
    }

    /// One block has a single peak pixel of 4.0 (rest 0). The 16th-power
    /// mean of [4^16, 0..0] over 64 pixels is 4^16/64. The 16th root is
    /// 4 / 64^(1/16) = 4 * 2^(-6/16) = 4 / 2^(0.375). Times K_TILE_NORM.
    #[test]
    fn test_compute_tile_distances_single_peak() {
        let w = 8;
        let h = 8;
        let xb = 1;
        let yb = 1;
        let mut diffmap = alloc::vec![0.0_f32; w * h];
        diffmap[0] = 4.0;
        let (is_first, cx, cy) = dct8_only_storage(1);
        let info = dct8_only_info(&is_first, &cx, &cy);
        let out = compute_tile_distances(&diffmap, w, h, xb, yb, &info);
        let expected = K_TILE_NORM * 4.0_f32 / 64.0_f32.powf(1.0 / 16.0);
        assert!(
            (out[0] - expected).abs() < 1e-4,
            "got {} expected {}",
            out[0],
            expected
        );
    }

    /// AC strategy splat: a single 16x16 strategy at (0,0) covers
    /// blocks (0,0)/(1,0)/(0,1)/(1,1). All four output entries should
    /// match the strategy's tile distance.
    #[test]
    fn test_compute_tile_distances_strategy_splat() {
        let w = 16;
        let h = 16;
        let xb = 2;
        let yb = 2;
        let n = xb * yb;
        let diffmap = alloc::vec![0.5_f32; w * h];
        // Block (0,0) is the first of a 2x2 (DCT16) strategy. Other
        // blocks are NOT first.
        let mut is_first = alloc::vec![false; n];
        let mut cx = alloc::vec![0u8; n];
        let mut cy = alloc::vec![0u8; n];
        is_first[0] = true;
        cx[0] = 2;
        cy[0] = 2;
        let info = dct8_only_info(&is_first, &cx, &cy);
        let out = compute_tile_distances(&diffmap, w, h, xb, yb, &info);
        let expected = K_TILE_NORM * 0.5;
        for &v in &out {
            assert!((v - expected).abs() < 1e-5, "got {v}");
        }
    }

    /// Edge-clipped strategy: 16x16 strategy near a 12x12 image clips
    /// to the image bounds. Result should still equal K_TILE_NORM * v
    /// for uniform diffmap (mean is invariant to pixel count).
    #[test]
    fn test_compute_tile_distances_clipped_to_image_bounds() {
        let w = 12; // not a multiple of 8
        let h = 12;
        let xb = 2;
        let yb = 2;
        let n = xb * yb;
        let diffmap = alloc::vec![0.7_f32; w * h];
        let mut is_first = alloc::vec![false; n];
        let mut cx = alloc::vec![0u8; n];
        let mut cy = alloc::vec![0u8; n];
        is_first[0] = true;
        cx[0] = 2;
        cy[0] = 2;
        let info = dct8_only_info(&is_first, &cx, &cy);
        let out = compute_tile_distances(&diffmap, w, h, xb, yb, &info);
        let expected = K_TILE_NORM * 0.7;
        // Block (0,0) is the splatted strategy; all four covered blocks
        // get the same value (the mean is per-pixel, so even with
        // clipping the value is constant).
        for &v in &out {
            assert!((v - expected).abs() < 1e-5, "got {v}");
        }
    }

    // ===== DeviationBounds =====

    /// Uniform initial qf → ratio = 1, qf_max_deviation_low = sqrt(250),
    /// asymmetry = 2 (since sqrt(250) ≈ 15.8 > 2). Lower = qf / (2 *
    /// sqrt(250)), Higher = qf * (sqrt(250) / 2).
    #[test]
    fn test_deviation_bounds_uniform() {
        let init = alloc::vec![0.5_f32; 16];
        let b = DeviationBounds::compute(&init);
        let s = 250.0_f32.sqrt();
        let asym = 2.0_f32.min(s); // = 2.0
        let expected_lower = 0.5 / (asym * s);
        let expected_higher = 0.5 * (s / asym);
        assert!((b.qf_lower - expected_lower).abs() < 1e-6);
        assert!((b.qf_higher - expected_higher).abs() < 1e-6);
    }

    /// Lower bound is clamped to >= 1e-6 even if the field has near-zero
    /// values.
    #[test]
    fn test_deviation_bounds_min_floor() {
        let init = alloc::vec![1e-8_f32, 1.0];
        let b = DeviationBounds::compute(&init);
        // Floor is 1e-6 internally; output is divided by some positive
        // factor so qf_lower must be > 0 and finite.
        assert!(b.qf_lower > 0.0);
        assert!(b.qf_lower.is_finite());
        assert!(b.qf_higher > b.qf_lower);
    }

    /// Wide ratio → qf_max_deviation_low becomes small (sqrt(250/ratio)),
    /// asymmetry = min(2, that). For ratio = 250, deviation = 1, asym
    /// = 1, lower = qf_min / 1, higher = qf_max / 1 (no widening).
    #[test]
    fn test_deviation_bounds_wide_ratio() {
        let init = alloc::vec![0.01_f32, 2.5_f32];
        let b = DeviationBounds::compute(&init);
        // ratio = 250, deviation = 1, asymmetry = 1, lower = 0.01 / 1
        // = 0.01, higher = 2.5 / 1 = 2.5 (within float epsilon)
        assert!((b.qf_lower - 0.01).abs() < 1e-6);
        assert!((b.qf_higher - 2.5).abs() < 1e-5);
    }

    // ===== clamp_toward_initial =====

    /// When current >= blend, no change. When current < blend, bump up
    /// to blend (clamped to bounds).
    #[test]
    fn test_clamp_toward_initial_basic() {
        // init = 1.0 everywhere; current dropped to 0.4 in block 0,
        // unchanged at 1.0 in block 1.
        let init = alloc::vec![1.0_f32; 2];
        let mut cur = alloc::vec![0.4_f32, 1.0_f32];
        let bounds = DeviationBounds {
            qf_lower: 0.0,
            qf_higher: 100.0,
        };
        clamp_toward_initial(&mut cur, &init, bounds);
        // Blend = 0.4 * cur + 0.6 * init = 0.4 * 0.4 + 0.6 * 1.0
        //       = 0.16 + 0.6 = 0.76. cur < blend, so bump to 0.76.
        assert!((cur[0] - 0.76).abs() < 1e-5);
        // Block 1: cur (1.0) >= blend (1.0), no change.
        assert_eq!(cur[1], 1.0);
    }

    /// When the blend exceeds qf_higher, output is clamped to qf_higher.
    #[test]
    fn test_clamp_toward_initial_clips_to_higher() {
        let init = alloc::vec![10.0_f32];
        let mut cur = alloc::vec![5.0_f32];
        let bounds = DeviationBounds {
            qf_lower: 0.0,
            qf_higher: 7.0,
        };
        clamp_toward_initial(&mut cur, &init, bounds);
        // Blend = 0.4 * 5 + 0.6 * 10 = 2 + 6 = 8.0. Clamped to 7.0.
        assert_eq!(cur[0], 7.0);
    }

    // ===== adjust_quant_field =====

    /// iter == 2 (cur_pow=0.0): only bad blocks adjust. Good block left
    /// untouched.
    #[test]
    fn test_adjust_quant_field_iter2_only_bad_blocks() {
        let mut qf = alloc::vec![1.0_f32, 1.0_f32];
        let tile_dist = alloc::vec![0.5_f32, 2.0_f32]; // diff = 0.5, 2.0
        let bounds = DeviationBounds {
            qf_lower: 0.0,
            qf_higher: 100.0,
        };
        adjust_quant_field(&mut qf, &tile_dist, 1.0, 2, bounds, 1.0, 1.0);
        assert_eq!(qf[0], 1.0); // good block, unchanged
        // Bad block: qf = 1.0 * 2.0 = 2.0; integer step from 1.5 to 2.5
        // = 1 vs 2, different → no extra bump.
        assert_eq!(qf[1], 2.0);
    }

    /// iter == 2 (cur_pow=0.0): integer-step minimum bump fires when
    /// the diff bump rounds to the same integer.
    #[test]
    fn test_adjust_quant_field_integer_step_bump() {
        // qf = 1.0, diff = 1.001, inv_global_scale = 1.0
        // qf_old = (1.0 * 1.0 + 0.5).floor() = 1
        // qf_new = (1.001 * 1.0 + 0.5).floor() = 1
        // → integer bump fires: qf = old + quantizer_scale = 1.0 + 0.5 = 1.5
        let mut qf = alloc::vec![1.0_f32];
        let tile_dist = alloc::vec![1.001_f32];
        let bounds = DeviationBounds {
            qf_lower: 0.0,
            qf_higher: 100.0,
        };
        adjust_quant_field(&mut qf, &tile_dist, 1.0, 2, bounds, 1.0, 0.5);
        assert_eq!(qf[0], 1.5);
    }

    /// iter < 2 (cur_pow=0.2): good blocks soften by diff^0.2. Bad
    /// blocks bump as in iter>=2 case.
    #[test]
    fn test_adjust_quant_field_iter1_softens_good() {
        let mut qf = alloc::vec![1.0_f32, 1.0_f32];
        let tile_dist = alloc::vec![0.5_f32, 2.0_f32];
        let bounds = DeviationBounds {
            qf_lower: 0.0,
            qf_higher: 100.0,
        };
        adjust_quant_field(&mut qf, &tile_dist, 1.0, 1, bounds, 1.0, 1.0);
        // Good block: qf *= 0.5^0.2 ≈ 0.8706
        let expected_good = 0.5_f64.powf(0.2) as f32;
        assert!((qf[0] - expected_good).abs() < 1e-5);
        // Bad block: 2.0 (no integer-step bump needed)
        assert_eq!(qf[1], 2.0);
    }

    /// Bounds clamp the result regardless of branch.
    #[test]
    fn test_adjust_quant_field_clamps_to_bounds() {
        let mut qf = alloc::vec![10.0_f32];
        let tile_dist = alloc::vec![10.0_f32];
        let bounds = DeviationBounds {
            qf_lower: 0.0,
            qf_higher: 50.0,
        };
        adjust_quant_field(&mut qf, &tile_dist, 1.0, 0, bounds, 1.0, 1.0);
        // diff = 10 → qf = 100; clamped to 50.
        assert_eq!(qf[0], 50.0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_butteraugli_loop_gpu_constructs_smoke() {
        // Smoke test: the wrapper constructs without panicking. Doesn't
        // run the full distance compute (would need a fixture image).
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let bg = ButteraugliLoopGpu::new(&enc, 64, 64);
        assert_eq!(bg.dimensions(), (64, 64));
        let bg2 = ButteraugliLoopGpu::new_multires(&enc, 128, 128);
        assert_eq!(bg2.dimensions(), (128, 128));
    }
}
