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
use crate::lossy_encoder::{LossyEncoder, distance_to_qac};

/// Convert a linear-light f32 value (clamped to [0, 1]) to an sRGB U8
/// byte using the IEC 61966-2-1 piecewise transfer function.
///
/// Matches the inverse of [`butteraugli_gpu::kernels::colors::
/// srgb_byte_to_linear`] — round-tripping linear→u8→linear introduces
/// only quantization error (no transfer-function mismatch).
///
/// **Use this helper, not the simplified gamma-2.4 form**, when feeding
/// pixels to butteraugli-gpu. The CLAUDE.md note "PNG Color Metadata
/// Causes Bogus Butteraugli Scores" is the same root cause class:
/// transfer-function mismatch between input and what the metric assumes.
#[inline]
pub fn linear_f32_to_srgb_u8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// Interleave three linear-light f32 planes into an interleaved sRGB U8
/// buffer (`width * height * 3` bytes), suitable for feeding directly
/// into [`ButteraugliLoopGpu::set_reference`] /
/// [`ButteraugliLoopGpu::compute_with_reference`].
///
/// Caller-supplied `dst` must be exactly `width * height * 3` bytes.
/// Use [`linear_planar_to_srgb_u8_interleaved`] for the allocating
/// convenience wrapper.
pub fn linear_planar_to_srgb_u8_interleaved_into(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    dst: &mut [u8],
) {
    let n = width * height;
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(g.len(), n);
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(dst.len(), n * 3);
    for i in 0..n {
        let i3 = i * 3;
        dst[i3] = linear_f32_to_srgb_u8(r[i]);
        dst[i3 + 1] = linear_f32_to_srgb_u8(g[i]);
        dst[i3 + 2] = linear_f32_to_srgb_u8(b[i]);
    }
}

/// Allocating convenience wrapper around
/// [`linear_planar_to_srgb_u8_interleaved_into`]. Returns a fresh
/// `width * height * 3`-byte buffer.
pub fn linear_planar_to_srgb_u8_interleaved(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut dst = alloc::vec![0u8; width * height * 3];
    linear_planar_to_srgb_u8_interleaved_into(r, g, b, width, height, &mut dst);
    dst
}

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

    /// Read the per-pixel diffmap (post-`compute*`) into a caller-supplied
    /// buffer — no allocation. `dst.len()` must be ≥ `width × height`.
    /// Use this in the iterative refinement loop to avoid per-iter Vec
    /// allocation overhead.
    pub fn copy_diffmap_to(&self, dst: &mut [f32]) -> butteraugli_gpu::Result<()> {
        self.inner.copy_diffmap_to(dst)
    }

    /// Read the per-pixel diffmap (post-`compute*`) into a fresh Vec.
    /// Allocating variant; prefer [`Self::copy_diffmap_to`] in hot loops.
    pub fn copy_diffmap(&self) -> Vec<f32> {
        self.inner.copy_diffmap()
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

/// Constant per-encode parameters for the refinement loop. Holds the
/// inputs that don't change across iterations (image dims, block grid,
/// AC strategy info, target_distance, iters, deviation bounds). Built
/// once at the top of the loop; passed to every per-iteration call.
///
/// The `info` borrow keeps the AC strategy slices alive for the duration
/// of the loop; callers pass `&dct8_only_info(...)` or a real strategy
/// view backed by their AcStrategyMap state.
pub struct RefineConfig<'a> {
    pub width: usize,
    pub height: usize,
    pub xsize_blocks: usize,
    pub ysize_blocks: usize,
    pub info: AcStrategyInfo<'a>,
    pub target_distance: f32,
    /// Total iterations the loop will run. Used to detect the final
    /// "compare-only" iteration (no adjustment).
    pub iters: usize,
    pub bounds: DeviationBounds,
}

/// One iteration of the host-side quant-field refinement.
///
/// Composes the four helpers in the order upstream uses (lines 230-406):
/// 1. `compute_tile_distances(diffmap)` → per-block tile distances
/// 2. If `iter == cfg.iters`, return early (last iter is compare-only)
/// 3. If `iter == K_ORIGINAL_COMPARISON_ROUND` (==1),
///    `clamp_toward_initial` to prevent oscillation
/// 4. `adjust_quant_field` with the iter-appropriate cur_pow regime
///
/// `inv_global_scale` and `quantizer_scale` come from the CURRENT
/// iteration's [`jxl_encoder::vardct::frame::DistanceParams`] — recomputed
/// from `quant_field_float` at the top of each iteration via
/// `DistanceParams::compute_from_quant_field` BEFORE this call.
///
/// Returns the per-block `tile_dist` slice (caller may log it for
/// per-iteration diagnostics; mirrors upstream's `bfly/iter` debug_rect).
#[allow(clippy::too_many_arguments)]
pub fn refine_quant_field_one_iter(
    quant_field_float: &mut [f32],
    initial_quant_field_float: &[f32],
    diffmap: &[f32],
    iter: usize,
    inv_global_scale: f32,
    quantizer_scale: f32,
    cfg: &RefineConfig<'_>,
) -> Vec<f32> {
    debug_assert_eq!(quant_field_float.len(), initial_quant_field_float.len());
    debug_assert_eq!(
        quant_field_float.len(),
        cfg.xsize_blocks * cfg.ysize_blocks
    );
    debug_assert_eq!(diffmap.len(), cfg.width * cfg.height);

    let tile_dist = compute_tile_distances(
        diffmap,
        cfg.width,
        cfg.height,
        cfg.xsize_blocks,
        cfg.ysize_blocks,
        &cfg.info,
    );

    // Last iteration is compare-only — no adjustment. Caller still gets
    // tile_dist for logging.
    if iter == cfg.iters {
        return tile_dist;
    }

    if iter == K_ORIGINAL_COMPARISON_ROUND {
        clamp_toward_initial(quant_field_float, initial_quant_field_float, cfg.bounds);
    }

    adjust_quant_field(
        quant_field_float,
        &tile_dist,
        cfg.target_distance,
        iter,
        cfg.bounds,
        inv_global_scale,
        quantizer_scale,
    );

    tile_dist
}

/// Per-iteration trace event from [`refine_aq_field_gpu`]. Caller may
/// inspect (e.g., to log butteraugli score progression) or ignore.
#[derive(Debug, Clone)]
pub struct RefineIterTrace {
    /// 0-based iteration index. Loop runs `iters + 1` iterations total
    /// (final one is compare-only).
    pub iter: usize,
    /// Total iters scheduled (== `cfg.iters`).
    pub iters: usize,
    /// Butteraugli max-norm score for this iteration's reconstruction.
    pub score: f32,
    /// Butteraugli libjxl 3-norm score for this iteration's reconstruction.
    pub pnorm_3: f32,
    /// Per-block tile distances after the (8×8) reduction.
    pub tile_dist: Vec<f32>,
}

/// Empirically-derived distance threshold above which butteraugli
/// refinement does not generalize as a quality win across the
/// CLIC2025-1024 corpus. The threshold is conservative: at higher
/// distances refinement wins on the majority of images by a small
/// margin, but the few losses can be catastrophic (worst case:
/// +1.021 butteraugli score at d=4.0). Mean-vs-uniform regresses
/// at d > 1.5 because the rare big losses dominate the small wins.
///
/// Source data (16-image CLIC2025-1024 sweep, archived at
/// `/mnt/v/output/jxl-encoder-gpu/butteraugli-refinement-sweep/
/// sweep_clic_16imgs_2026-05-08.log`):
///
/// | dist | uniform µ | refined µ | rf<un wins | rf>un losses | worst loss |
/// |------|-----------|-----------|------------|--------------|------------|
/// | 1.0  | 1.2386    | 1.2105    | 7/16       | 9/16         | +0.133     |
/// | 2.0  | 2.0245    | 2.1195    | 11/16      | 5/16         | +0.362     |
/// | 4.0  | 3.2582    | 3.5160    | 12/16      | 3/16         | +1.021     |
///
/// At d=1.0 win rate is ~50/50 but worst loss is small (+0.133); at
/// d≥2.0 win rate is 60–75% but worst losses balloon (+0.362, +1.021).
/// Production code that can't tolerate occasional catastrophic
/// regressions should keep this conservative threshold; benchmarks or
/// content-known callers can override with [`refine_aq_field_gpu`].
///
/// A future per-content gating heuristic (e.g., skip refinement if
/// initial AQ score > uniform score × 1.1, indicating the AQ field
/// itself regresses uniform on this image) would give better win
/// rates at higher distances. See the corpus sweep TSV for the
/// per-image data.
///
/// Use [`should_refine_at_distance`] to query, or
/// [`refine_aq_field_gpu_auto`] for the full auto-gated wrapper.
pub const REFINEMENT_DISTANCE_THRESHOLD: f32 = 1.5;

/// Returns `true` when butteraugli refinement is empirically expected
/// to improve quality at the given target distance. See
/// [`REFINEMENT_DISTANCE_THRESHOLD`] for the source data.
#[inline]
pub fn should_refine_at_distance(distance: f32) -> bool {
    distance <= REFINEMENT_DISTANCE_THRESHOLD
}

/// Threshold ratio: if `initial_aq_score > uniform_score * SMART_GATE_AQ_REGRESSION_RATIO`,
/// the smart gate concludes AQ is hurting on this image and falls
/// back to a uniform qac field instead of refining a doomed initial.
///
/// Empirically chosen at 1.10 (10% AQ-vs-uniform regression
/// tolerance) from the corpus sweep: at d=4.0 the catastrophic
/// refinement losses (e.g., +1.021 score on image 2) all occur on
/// images where initial AQ already regresses uniform by > 10%.
pub const SMART_GATE_AQ_REGRESSION_RATIO: f32 = 1.10;

/// Outcome from [`refine_aq_field_gpu_smart`] — exposes which path the
/// content-aware gate selected and the diagnostic scores it measured.
#[derive(Debug, Clone)]
pub struct SmartGateOutcome {
    /// The selected qac field. One of:
    /// - refined output of [`refine_aq_field_gpu`] (gate let it through)
    /// - the input `initial_aq_field` (distance > threshold, refinement skipped)
    /// - a uniform field at `distance_to_qac(target_distance)` (AQ regression detected)
    pub aq_field: Vec<f32>,
    /// Path the gate took. Useful for telemetry / per-image debugging.
    pub path: SmartGatePath,
    /// Initial-AQ butteraugli score (always measured). `None` only
    /// when the distance gate fired before the AQ measurement.
    pub initial_aq_score: Option<f32>,
    /// Uniform-qac butteraugli score (measured iff content gate
    /// considered firing). `None` when distance gate skipped early.
    pub uniform_score: Option<f32>,
}

/// Which path the smart gate chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmartGatePath {
    /// AQ acceptable but distance > [`REFINEMENT_DISTANCE_THRESHOLD`];
    /// returned `initial_aq_field` as-is. Cost: 2 baseline encodes +
    /// 2 measures (AQ + uniform).
    DistanceGated,
    /// AQ regresses uniform by > [`SMART_GATE_AQ_REGRESSION_RATIO`]
    /// (10%); returned a uniform qac field. Fires regardless of
    /// distance — refinement can't recover from a doomed initial.
    /// Cost: same as DistanceGated (2 encodes + 2 measures, no
    /// refinement run).
    AqRegressedFallToUniform,
    /// AQ acceptable AND distance ≤ threshold; ran full refinement
    /// loop. Cost: 2 baseline encodes + the refinement loop.
    Refined,
}

/// Content-aware auto-gated refinement.
///
/// Thin delegate to [`refine_aq_field_gpu_smart_with_threshold`]
/// using the default [`SMART_GATE_AQ_REGRESSION_RATIO`] (1.10). Use
/// the `_with_threshold` variant when you need a different
/// AQ-regression tolerance (e.g., 1.30 for SSIM2-targeted output
/// where AQ is more often genuinely better, +infinity to never fall
/// back, 0 to always fall back to uniform).
///
/// ## Production usage
///
/// ```ignore
/// use jxl_encoder_gpu::encoder::GpuEncoder;
/// use jxl_encoder_gpu::lossy_encoder::LossyEncoder;
/// use jxl_encoder_gpu::forks::butteraugli_loop::{
///     ButteraugliLoopGpu, refine_aq_field_gpu_smart, SmartGatePath,
/// };
///
/// // One-time setup per (width, height) — instantiate the encoder
/// // and the butteraugli compute state once, reuse across encodes.
/// type Backend = cubecl::cuda::CudaRuntime;
/// let enc: GpuEncoder<Backend> = GpuEncoder::new();
/// let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);
/// let mut bg = ButteraugliLoopGpu::new_multires(&enc, w, h);
///
/// // Per-image: derive initial AQ, run smart gate, encode with the
/// // selected field. ref_srgb is the ORIGINAL sRGB bytes (not a
/// // re-encoding of the linear planes — see CLAUDE.md note).
/// let initial_aq = lossy.compute_aq_field(&enc, &r, &g, &b, distance);
/// let outcome = refine_aq_field_gpu_smart(
///     &enc, &lossy, &mut bg, &r, &g, &b, &original_srgb,
///     &initial_aq, distance, /* iters = */ 2, |_| (),
/// )?;
///
/// // Optional: log which path the gate chose for telemetry.
/// match outcome.path {
///     SmartGatePath::DistanceGated         => log::info!("d>1.5, kept AQ"),
///     SmartGatePath::AqRegressedFallToUniform => log::info!("AQ regressed, used uniform"),
///     SmartGatePath::Refined               => log::info!("refined AQ via butteraugli loop"),
/// }
///
/// let (rec_r, rec_g, rec_b) = lossy.encode_one_adaptive(
///     &enc, &r, &g, &b, &outcome.aq_field,
/// );
/// ```
///
/// Always measures AQ vs uniform baselines first, then chooses one
/// of three paths:
///
/// 1. If AQ score > uniform × [`SMART_GATE_AQ_REGRESSION_RATIO`]
///    (10% regression), the AQ field is hurting on this image —
///    return a uniform qac field. Refinement at any distance can't
///    recover from a regressing initial AQ; better to skip AQ
///    entirely.
/// 2. Else if `target_distance > REFINEMENT_DISTANCE_THRESHOLD` (1.5),
///    return the initial AQ field as-is (refinement doesn't help on
///    average and risks catastrophic regressions at high d).
/// 3. Else (low distance, AQ acceptable), run [`refine_aq_field_gpu`].
///
/// Costs 2 baseline encode+measure cycles always (one each for AQ
/// and uniform). At 1024×1024 on RTX 5070 ≈ 100 ms additional vs
/// [`refine_aq_field_gpu_auto`]. The benefit is that at high d this
/// catches AQ-regressing content that distance-gated returned
/// blindly. Empirically this is the best-mean-quality path on
/// butteraugli in the 16-image CLIC corpus sweep at d=1.0
/// (1.1886 vs 1.2105 refined, 1.2386 uniform, 1.4221 AQ). On
/// SSIMULACRA2 the optimal threshold is higher (less aggressive
/// fallback) — see CHANGELOG `0e470164` for the metric-disagreement
/// finding.
///
/// Returns a [`SmartGateOutcome`] with the selected field plus
/// diagnostics so callers can log / decide. The trace callback is
/// invoked iff path == `Refined`.
#[allow(clippy::too_many_arguments)]
pub fn refine_aq_field_gpu_smart<R: Runtime>(
    enc: &GpuEncoder<R>,
    lossy: &LossyEncoder<R>,
    bg: &mut ButteraugliLoopGpu<R>,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    ref_srgb: &[u8],
    initial_aq_field: &[f32],
    target_distance: f32,
    iters: usize,
    trace: impl FnMut(RefineIterTrace),
) -> butteraugli_gpu::Result<SmartGateOutcome> {
    refine_aq_field_gpu_smart_with_threshold(
        enc,
        lossy,
        bg,
        r,
        g,
        b,
        ref_srgb,
        initial_aq_field,
        target_distance,
        iters,
        SMART_GATE_AQ_REGRESSION_RATIO,
        trace,
    )
}

/// Generalized smart gate with explicit AQ-regression threshold.
///
/// Same logic as [`refine_aq_field_gpu_smart`] but the
/// `aq_regression_ratio` is a parameter instead of the
/// [`SMART_GATE_AQ_REGRESSION_RATIO`] const. Use this when targeting
/// a different metric than butteraugli's default 1.10:
///
/// - **1.10 (default)**: butteraugli-targeted; tight fallback to
///   uniform on AQ regression. Optimal mean butteraugli on the
///   CLIC sweep but loses some SSIM2 wins.
/// - **1.30**: more permissive; lets more AQ through. Better SSIM2
///   trade-off because AQ wins on SSIM2 even when butteraugli sees
///   it as a small regression.
/// - **`f32::INFINITY`**: never fall back to uniform. Equivalent to
///   distance-only gating ([`refine_aq_field_gpu_auto`]).
/// - **`0.0`**: always fall back to uniform. Equivalent to never
///   using AQ at all — useful as a "uniform baseline" path that
///   still routes through the smart-gate API for logging
///   consistency.
#[allow(clippy::too_many_arguments)]
pub fn refine_aq_field_gpu_smart_with_threshold<R: Runtime>(
    enc: &GpuEncoder<R>,
    lossy: &LossyEncoder<R>,
    bg: &mut ButteraugliLoopGpu<R>,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    ref_srgb: &[u8],
    initial_aq_field: &[f32],
    target_distance: f32,
    iters: usize,
    aq_regression_ratio: f32,
    trace: impl FnMut(RefineIterTrace),
) -> butteraugli_gpu::Result<SmartGateOutcome> {
    let (width, height) = lossy.dimensions();
    bg.set_reference(ref_srgb)?;

    // Measure initial AQ score
    let (rec_r, rec_g, rec_b) = lossy.encode_one_adaptive(enc, r, g, b, initial_aq_field);
    let recon_srgb = linear_planar_to_srgb_u8_interleaved(
        &rec_r,
        &rec_g,
        &rec_b,
        width as usize,
        height as usize,
    );
    let s_aq = bg.compute_with_reference(&recon_srgb)?.score;

    // Measure uniform score for comparison
    let qac_uniform = distance_to_qac(target_distance);
    let (rec_r, rec_g, rec_b) = lossy.encode_one(enc, r, g, b, qac_uniform);
    let recon_srgb = linear_planar_to_srgb_u8_interleaved(
        &rec_r,
        &rec_g,
        &rec_b,
        width as usize,
        height as usize,
    );
    let s_un = bg.compute_with_reference(&recon_srgb)?.score;

    // Path 1: AQ regresses uniform — fall back to uniform regardless
    // of distance. Refinement can't recover from a doomed initial.
    if s_aq > s_un * aq_regression_ratio {
        return Ok(SmartGateOutcome {
            aq_field: alloc::vec![qac_uniform; initial_aq_field.len()],
            path: SmartGatePath::AqRegressedFallToUniform,
            initial_aq_score: Some(s_aq),
            uniform_score: Some(s_un),
        });
    }

    // Path 2: AQ acceptable but distance too high for refinement — use
    // initial AQ as-is.
    if !should_refine_at_distance(target_distance) {
        return Ok(SmartGateOutcome {
            aq_field: initial_aq_field.to_vec(),
            path: SmartGatePath::DistanceGated,
            initial_aq_score: Some(s_aq),
            uniform_score: Some(s_un),
        });
    }

    // Path 3: refine.
    let refined = refine_aq_field_gpu(
        enc,
        lossy,
        bg,
        r,
        g,
        b,
        ref_srgb,
        initial_aq_field,
        target_distance,
        iters,
        trace,
    )?;
    Ok(SmartGateOutcome {
        aq_field: refined,
        path: SmartGatePath::Refined,
        initial_aq_score: Some(s_aq),
        uniform_score: Some(s_un),
    })
}

/// Distance-aware auto-gated wrapper around [`refine_aq_field_gpu`].
///
/// Calls the full refinement loop iff [`should_refine_at_distance`]
/// returns true; otherwise returns `initial_aq_field` unchanged
/// without spending GPU time on doomed iteration. The trace callback
/// is NOT invoked when refinement is skipped (no per-iteration
/// measurements happened).
///
/// Sets the production-ready default policy from the corpus sweep:
/// refine at `distance ≤ 1.5`, skip otherwise. Override with
/// [`refine_aq_field_gpu`] directly if you have content-specific
/// knowledge that justifies a different threshold.
#[allow(clippy::too_many_arguments)]
pub fn refine_aq_field_gpu_auto<R: Runtime>(
    enc: &GpuEncoder<R>,
    lossy: &LossyEncoder<R>,
    bg: &mut ButteraugliLoopGpu<R>,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    ref_srgb: &[u8],
    initial_aq_field: &[f32],
    target_distance: f32,
    iters: usize,
    trace: impl FnMut(RefineIterTrace),
) -> butteraugli_gpu::Result<Vec<f32>> {
    if should_refine_at_distance(target_distance) {
        refine_aq_field_gpu(
            enc,
            lossy,
            bg,
            r,
            g,
            b,
            ref_srgb,
            initial_aq_field,
            target_distance,
            iters,
            trace,
        )
    } else {
        Ok(initial_aq_field.to_vec())
    }
}

/// End-to-end butteraugli refinement loop with GPU-substituted
/// per-iteration distance compute.
///
/// **Qac-domain adaptation.** Operates on the per-block `aq_field` (qac
/// multiplier) directly — diverges from upstream's `quant_field_float`
/// + `global_scale` + integer rounding model. Specifically:
/// - `inv_global_scale = 1.0`, `quantizer_scale = 0.0` are passed to
///   the per-iter helper, neutering the upstream "integer-step minimum
///   bump" check (lines 366-371, 392-397). Our pipeline uses float qac
///   directly so there's no integer step to round-into.
/// - Deviation bounds are still computed from the initial aq_field,
///   keeping iteration adjustments bounded relative to the calibration.
///
/// The other three helpers (`compute_tile_distances`,
/// `clamp_toward_initial`, `adjust_quant_field`) operate identically.
///
/// **Callers**: pass `r/g/b` as linear-light f32 planes (the same input
/// you'd hand to [`LossyEncoder::encode_one_adaptive`]) plus
/// `ref_srgb` — the **actual original sRGB U8 bytes** for the
/// butteraugli reference. The reference must be the source bytes, not
/// a re-encoding of the linear planes — round-tripping through any
/// transfer function (especially the simplified `powf(2.4)` ↔ IEC
/// piecewise asymmetry) inflates butteraugli scores even when the
/// reconstruction is bit-perfect. See CLAUDE.md "PNG Color Metadata
/// Causes Bogus Butteraugli Scores" for the same root-cause class.
///
/// Returns the refined per-block `aq_field` after `iters + 1` iterations.
///
/// **`trace`**: per-iteration callback invoked with the
/// reconstruction's butteraugli score + tile distances. Pass `|_| ()`
/// to ignore.
#[allow(clippy::too_many_arguments)]
pub fn refine_aq_field_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    lossy: &LossyEncoder<R>,
    bg: &mut ButteraugliLoopGpu<R>,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    ref_srgb: &[u8],
    initial_aq_field: &[f32],
    target_distance: f32,
    iters: usize,
    trace: impl FnMut(RefineIterTrace),
) -> butteraugli_gpu::Result<Vec<f32>> {
    refine_aq_field_gpu_with_encode(
        lossy,
        bg,
        ref_srgb,
        initial_aq_field,
        target_distance,
        iters,
        |aq| lossy.encode_one_adaptive(enc, r, g, b, aq),
        trace,
    )
}

/// Strat-search variant of [`refine_aq_field_gpu`].
///
/// Identical control flow, but the per-iteration encode step calls
/// [`LossyEncoder::encode_one_with_strategy_search_dct8_16_adaptive`]
/// instead of [`LossyEncoder::encode_one_adaptive`]. Strategy
/// assignments are stable across iterations because the cost-grid
/// stage scales with `target_distance` (constant) — only per-block
/// quantization changes via the working `aq_field`.
///
/// **Quality target** (CLIC 1024×1024 @ d=1.0): combines strat-search
/// transform picks with butteraugli AQ refinement's per-block qac
/// tuning. Both are independently working. The combined-mode
/// expectation is to match-or-beat the refine-only score (1.1475 in
/// the source-tracking entry) by giving the loop a transform palette
/// to work with instead of forcing DCT8 everywhere.
///
/// **Cost** (per iteration): roughly 5-7× more GPU work than the
/// uniform-DCT8 variant, because each iter re-runs the full
/// strat-search pipeline (cost grids + selector + mixed encode-recon).
/// At 4 iters this is currently ~250-350 ms vs ~45-60 ms for the
/// uniform variant on a 1024² CLIC photo. Future optimization
/// (#41/#42): cache cost-grid output across iters since assignments
/// are stable when only `aq_field` varies.
#[allow(clippy::too_many_arguments)]
pub fn refine_aq_field_gpu_with_strategy_search<R: Runtime>(
    enc: &GpuEncoder<R>,
    lossy: &LossyEncoder<R>,
    bg: &mut ButteraugliLoopGpu<R>,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    ref_srgb: &[u8],
    initial_aq_field: &[f32],
    target_distance: f32,
    iters: usize,
    trace: impl FnMut(RefineIterTrace),
) -> butteraugli_gpu::Result<Vec<f32>> {
    refine_aq_field_gpu_with_encode(
        lossy,
        bg,
        ref_srgb,
        initial_aq_field,
        target_distance,
        iters,
        |aq| {
            lossy.encode_one_with_strategy_search_dct8_16_adaptive(
                enc,
                r,
                g,
                b,
                aq,
                target_distance,
            )
        },
        trace,
    )
}

/// Inner refinement loop parameterized on the encode step. Both
/// [`refine_aq_field_gpu`] (uniform DCT8) and
/// [`refine_aq_field_gpu_with_strategy_search`] delegate here. Keeps
/// the iteration semantics (deviation bounds, qac update rule, trace
/// callback shape) identical between the two encoders — there's only
/// one place that defines "what does a butteraugli refinement
/// iteration mean."
fn refine_aq_field_gpu_with_encode<R: Runtime, E>(
    lossy: &LossyEncoder<R>,
    bg: &mut ButteraugliLoopGpu<R>,
    ref_srgb: &[u8],
    initial_aq_field: &[f32],
    target_distance: f32,
    iters: usize,
    mut encode_step: E,
    mut trace: impl FnMut(RefineIterTrace),
) -> butteraugli_gpu::Result<Vec<f32>>
where
    E: FnMut(&[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>),
{
    let (width, height) = lossy.dimensions();
    let (xsize_blocks, ysize_blocks) = {
        let (pw, ph) = lossy.padded_dimensions();
        (pw as usize / 8, ph as usize / 8)
    };
    let n_pixels = (width as usize) * (height as usize);

    debug_assert_eq!(
        ref_srgb.len(),
        n_pixels * 3,
        "ref_srgb must be {n_pixels} * 3 bytes"
    );

    // Upload reference (original) sRGB U8 once. Cached internally for
    // the lifetime of all subsequent compute_with_reference calls.
    bg.set_reference(ref_srgb)?;

    // Per-encode invariants: deviation bounds + AC strategy info.
    let bounds = DeviationBounds::compute(initial_aq_field);
    let (is_first_storage, cx_storage, cy_storage) =
        dct8_only_storage(xsize_blocks * ysize_blocks);
    let cfg = RefineConfig {
        width: width as usize,
        height: height as usize,
        xsize_blocks,
        ysize_blocks,
        info: dct8_only_info(&is_first_storage, &cx_storage, &cy_storage),
        target_distance,
        iters,
        bounds,
    };

    // Mutable per-iteration state. `aq_field` is the working field;
    // `initial_aq_field` is preserved for the kOriginalComparisonRound
    // clamp.
    let mut aq_field = initial_aq_field.to_vec();
    let mut diffmap = alloc::vec![0.0_f32; n_pixels];
    let mut recon_srgb = alloc::vec![0u8; n_pixels * 3];

    for iter in 0..=iters {
        // Step 1: encode at current aq_field, get reconstructed linear RGB.
        let (rec_r, rec_g, rec_b) = encode_step(&aq_field);

        // Step 2: convert recon → sRGB U8 → butteraugli compute.
        linear_planar_to_srgb_u8_interleaved_into(
            &rec_r,
            &rec_g,
            &rec_b,
            width as usize,
            height as usize,
            &mut recon_srgb,
        );
        let result = bg.compute_with_reference(&recon_srgb)?;

        // Step 3: pull diffmap to host (no-alloc into preallocated buf).
        bg.copy_diffmap_to(&mut diffmap)?;

        // Step 4: reduce diffmap → per-block tile distances → adjust qf.
        // Force iter >= 2 semantics in adjust_quant_field (cur_pow=0.0,
        // only-bad-blocks regime). The cur_pow=0.2 path softens GOOD
        // blocks (`qac *= diff^0.2 < 1`) to "save bits" — but our
        // qac-domain pipeline has no bit budget, so softening just
        // degrades good blocks unnecessarily and the regression
        // observed at d=4.0 (refined +19.7% vs uniform) traces to
        // exactly this path. Skipping it also means
        // `clamp_toward_initial` (which fires at iter==1 to undo
        // softening overshoot) becomes a no-op, so we bypass
        // refine_quant_field_one_iter entirely and call the helpers
        // directly.
        let tile_dist = compute_tile_distances(
            &diffmap,
            cfg.width,
            cfg.height,
            cfg.xsize_blocks,
            cfg.ysize_blocks,
            &cfg.info,
        );
        if iter < iters {
            // Cap the per-iter multiplier at 1.5 to prevent compound
            // upward drift across iterations on consistently-bad blocks.
            // Without an integer-quant ceiling like upstream's
            // raw_quant ∈ [1, 255], `qac *= diff` with diff > 4 (common
            // at d=4.0 since target_distance=4 means diff=tile_dist/4)
            // would push qac to qf_higher ≈ 6.05 in one iteration on
            // every "bad" block, causing the catastrophic +19.7% AQ
            // regression at high d.
            for bi in 0..aq_field.len() {
                let diff = (tile_dist[bi] / target_distance).min(1.5);
                if diff > 1.0 {
                    aq_field[bi] *= diff;
                }
                if aq_field[bi] > bounds.qf_higher {
                    aq_field[bi] = bounds.qf_higher;
                }
                if aq_field[bi] < bounds.qf_lower {
                    aq_field[bi] = bounds.qf_lower;
                }
            }
        }

        trace(RefineIterTrace {
            iter,
            iters,
            score: result.score,
            pnorm_3: result.pnorm_3,
            tile_dist,
        });
    }

    Ok(aq_field)
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

    // ===== refine_quant_field_one_iter =====

    /// Helper: build a uniform-diffmap test config.
    fn make_uniform_cfg<'a>(
        is_first: &'a [bool],
        cx: &'a [u8],
        cy: &'a [u8],
        target_distance: f32,
        iters: usize,
        bounds: DeviationBounds,
    ) -> RefineConfig<'a> {
        RefineConfig {
            width: 16,
            height: 16,
            xsize_blocks: 2,
            ysize_blocks: 2,
            info: dct8_only_info(is_first, cx, cy),
            target_distance,
            iters,
            bounds,
        }
    }

    /// Final iteration (iter == iters): no adjustment, only return tile_dist.
    #[test]
    fn test_refine_one_iter_final_no_adjustment() {
        let n = 4;
        let mut qf = alloc::vec![1.0_f32; n];
        let init = qf.clone();
        let diffmap = alloc::vec![0.5_f32; 16 * 16];
        let (is_first, cx, cy) = dct8_only_storage(n);
        let bounds = DeviationBounds {
            qf_lower: 0.0,
            qf_higher: 100.0,
        };
        let cfg = make_uniform_cfg(&is_first, &cx, &cy, 1.0, 2, bounds);
        let td = refine_quant_field_one_iter(&mut qf, &init, &diffmap, 2, 1.0, 1.0, &cfg);
        // qf unchanged
        for &v in &qf {
            assert_eq!(v, 1.0);
        }
        // td reflects compute_tile_distances result
        for &v in &td {
            assert!((v - K_TILE_NORM * 0.5).abs() < 1e-5);
        }
    }

    /// iter == 1 fires the kOriginalComparisonRound clamp BEFORE the
    /// adjustment. Verify by setting cur << init so clamp bumps qf up,
    /// then a uniform diff < target keeps the bump (cur_pow=0.2 softens).
    #[test]
    fn test_refine_one_iter_iter1_applies_clamp() {
        let n = 4;
        let mut qf = alloc::vec![0.4_f32; n];
        let init = alloc::vec![1.0_f32; n];
        // Diff = 0.5 (< target) → cur_pow=0.2 path softens by 0.5^0.2.
        let diffmap = alloc::vec![0.5_f32 / K_TILE_NORM; 16 * 16];
        let (is_first, cx, cy) = dct8_only_storage(n);
        let bounds = DeviationBounds {
            qf_lower: 0.0,
            qf_higher: 100.0,
        };
        let cfg = make_uniform_cfg(&is_first, &cx, &cy, 1.0, 3, bounds);
        let _td = refine_quant_field_one_iter(&mut qf, &init, &diffmap, 1, 1.0, 1.0, &cfg);
        // After clamp: qf bumped to 0.4 * 0.4 + 0.6 * 1.0 = 0.76
        // After adjust (diff=0.5, cur_pow=0.2): qf *= 0.5^0.2 ≈ 0.8706
        // Final: 0.76 * 0.8706 ≈ 0.6617
        let expected = 0.76_f32 * 0.5_f32.powf(0.2);
        for &v in &qf {
            assert!((v - expected).abs() < 1e-4, "got {v} expected {expected}");
        }
    }

    /// iter == 0 does NOT fire the clamp (only iter == 1 does).
    /// Verify by setting cur << init: without clamp, qf stays at cur
    /// then adjust softens.
    #[test]
    fn test_refine_one_iter_iter0_no_clamp() {
        let n = 4;
        let mut qf = alloc::vec![0.4_f32; n];
        let init = alloc::vec![1.0_f32; n];
        let diffmap = alloc::vec![0.5_f32 / K_TILE_NORM; 16 * 16];
        let (is_first, cx, cy) = dct8_only_storage(n);
        let bounds = DeviationBounds {
            qf_lower: 0.0,
            qf_higher: 100.0,
        };
        let cfg = make_uniform_cfg(&is_first, &cx, &cy, 1.0, 3, bounds);
        let _td = refine_quant_field_one_iter(&mut qf, &init, &diffmap, 0, 1.0, 1.0, &cfg);
        // No clamp: qf stays 0.4. Then adjust: 0.4 * 0.5^0.2 ≈ 0.348
        let expected = 0.4_f32 * 0.5_f32.powf(0.2);
        for &v in &qf {
            assert!((v - expected).abs() < 1e-4, "got {v} expected {expected}");
        }
    }

    // ===== sRGB conversion helpers =====

    /// Round-trip the full IEC sRGB transfer: linear f32 → sRGB U8 →
    /// linear f32 (via butteraugli-gpu's reference inverse). Should
    /// match within quantization error (~1/255).
    #[test]
    fn test_linear_f32_to_srgb_u8_iec_roundtrip() {
        let inverse = |b: u8| {
            let f = b as f32 / 255.0;
            if f <= 0.04045 {
                f / 12.92
            } else {
                ((f + 0.055) / 1.055).powf(2.4)
            }
        };
        for &v in &[0.0_f32, 0.001, 0.01, 0.04045, 0.1, 0.5, 0.9, 1.0] {
            let u = linear_f32_to_srgb_u8(v);
            let v2 = inverse(u);
            // Quantization error at most ~1/255 = 0.004 in linear.
            // Be generous near black where 1 byte == many linear LSBs.
            let tol = if v < 0.01 { 0.0005 } else { 0.005 };
            assert!(
                (v - v2).abs() < tol,
                "v={v} u={u} v2={v2} diff={}",
                (v - v2).abs()
            );
        }
    }

    /// Linear 0.0 → 0; linear 1.0 → 255. Endpoints exact.
    #[test]
    fn test_linear_f32_to_srgb_u8_endpoints() {
        assert_eq!(linear_f32_to_srgb_u8(0.0), 0);
        assert_eq!(linear_f32_to_srgb_u8(1.0), 255);
        assert_eq!(linear_f32_to_srgb_u8(-0.5), 0);
        assert_eq!(linear_f32_to_srgb_u8(1.5), 255);
    }

    #[test]
    fn test_linear_planar_to_srgb_u8_interleaved_basic() {
        let r = alloc::vec![1.0_f32; 4];
        let g = alloc::vec![0.0_f32; 4];
        let b = alloc::vec![0.5_f32; 4];
        let out = linear_planar_to_srgb_u8_interleaved(&r, &g, &b, 2, 2);
        assert_eq!(out.len(), 12);
        for i in 0..4 {
            assert_eq!(out[i * 3], 255);
            assert_eq!(out[i * 3 + 1], 0);
            // 0.5 linear → ~0.7354 sRGB → ~187
            assert!((out[i * 3 + 2] as i32 - 188).abs() <= 1);
        }
    }

    // ===== should_refine_at_distance =====

    #[test]
    fn test_should_refine_at_distance_thresholds() {
        // d <= 1.5 → refine; d > 1.5 → skip
        assert!(should_refine_at_distance(0.5));
        assert!(should_refine_at_distance(1.0));
        assert!(should_refine_at_distance(1.5));
        assert!(!should_refine_at_distance(1.6));
        assert!(!should_refine_at_distance(2.0));
        assert!(!should_refine_at_distance(4.0));
        assert!(!should_refine_at_distance(8.0));
        // Edge cases
        assert!(should_refine_at_distance(0.0));
        assert!(!should_refine_at_distance(f32::INFINITY));
        // NaN: comparison returns false → not refined (safe default)
        assert!(!should_refine_at_distance(f32::NAN));
    }

    /// Smart gate with threshold = +infinity: AQ never falls back,
    /// behaves like distance-only auto-gate. Verifies the new
    /// `_with_threshold` parameter degrades gracefully at the extreme.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_refine_aq_field_gpu_smart_threshold_infinity_no_fallback() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, 64, 64);
        let mut bg = ButteraugliLoopGpu::new(&enc, 64, 64);
        let n = 64 * 64;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        let ref_srgb = linear_planar_to_srgb_u8_interleaved(&r, &g, &b, 64, 64);
        let initial = lossy.compute_aq_field(&enc, &r, &g, &b, 4.0);

        let outcome = refine_aq_field_gpu_smart_with_threshold(
            &enc, &lossy, &mut bg, &r, &g, &b, &ref_srgb, &initial, 4.0, 2,
            f32::INFINITY, |_| (),
        )
        .expect("smart");

        // With threshold=inf, never falls back. At d=4.0 distance gate
        // fires → DistanceGated, never AqRegressedFallToUniform.
        assert!(matches!(
            outcome.path,
            SmartGatePath::DistanceGated | SmartGatePath::Refined
        ));
    }

    /// Smart gate with threshold = 0.0: AQ always falls back to uniform
    /// (any AQ score > uniform * 0 = AQ score > 0 fires). The result
    /// should be a uniform field.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_refine_aq_field_gpu_smart_threshold_zero_always_fallback() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, 64, 64);
        let mut bg = ButteraugliLoopGpu::new(&enc, 64, 64);
        let n = 64 * 64;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        let ref_srgb = linear_planar_to_srgb_u8_interleaved(&r, &g, &b, 64, 64);
        let initial = lossy.compute_aq_field(&enc, &r, &g, &b, 1.0);

        let outcome = refine_aq_field_gpu_smart_with_threshold(
            &enc, &lossy, &mut bg, &r, &g, &b, &ref_srgb, &initial, 1.0, 2,
            0.0, |_| (),
        )
        .expect("smart");

        assert_eq!(outcome.path, SmartGatePath::AqRegressedFallToUniform);
        // Returned field is uniform — all elements equal.
        let first = outcome.aq_field[0];
        for &v in &outcome.aq_field {
            assert!((v - first).abs() < 1e-6);
        }
    }

    /// Smart gate at high distance: never refines (refinement gated by
    /// distance). Path is either DistanceGated (AQ acceptable, return
    /// initial AQ) or AqRegressedFallToUniform (AQ regresses, fall back
    /// to uniform). Both score AQ + uniform, neither runs refinement.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_refine_aq_field_gpu_smart_high_distance_never_refines() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, 64, 64);
        let mut bg = ButteraugliLoopGpu::new(&enc, 64, 64);
        let n = 64 * 64;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        let ref_srgb = linear_planar_to_srgb_u8_interleaved(&r, &g, &b, 64, 64);
        let initial = lossy.compute_aq_field(&enc, &r, &g, &b, 4.0);

        let outcome = refine_aq_field_gpu_smart(
            &enc, &lossy, &mut bg, &r, &g, &b, &ref_srgb, &initial, 4.0, 2, |_| (),
        )
        .expect("smart");

        // Always measures both at high distance now.
        assert!(outcome.initial_aq_score.is_some());
        assert!(outcome.uniform_score.is_some());
        // Path is never Refined at high distance.
        assert!(matches!(
            outcome.path,
            SmartGatePath::DistanceGated | SmartGatePath::AqRegressedFallToUniform
        ));
        assert_eq!(outcome.aq_field.len(), initial.len());
    }

    /// Smart gate at low distance: measures both, picks one of the two
    /// downstream paths. Just verify it completes without panic — the
    /// specific path chosen depends on the gradient image's content.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_refine_aq_field_gpu_smart_low_distance_completes() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, 64, 64);
        let mut bg = ButteraugliLoopGpu::new(&enc, 64, 64);
        let n = 64 * 64;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        let ref_srgb = linear_planar_to_srgb_u8_interleaved(&r, &g, &b, 64, 64);
        let initial = lossy.compute_aq_field(&enc, &r, &g, &b, 1.0);

        let outcome = refine_aq_field_gpu_smart(
            &enc, &lossy, &mut bg, &r, &g, &b, &ref_srgb, &initial, 1.0, 2, |_| (),
        )
        .expect("smart");

        // Distance below threshold → both AQ + uniform measured
        assert!(outcome.initial_aq_score.is_some());
        assert!(outcome.uniform_score.is_some());
        // Path is either Refined or AqRegressedFallToUniform
        assert!(matches!(
            outcome.path,
            SmartGatePath::Refined | SmartGatePath::AqRegressedFallToUniform
        ));
        assert_eq!(outcome.aq_field.len(), initial.len());
    }

    /// Auto-gate at high distance returns initial unchanged with no
    /// GPU work. Pure-CPU test (no cuda feature needed).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_refine_aq_field_gpu_auto_gates_off_at_high_distance() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, 64, 64);
        let mut bg = ButteraugliLoopGpu::new(&enc, 64, 64);
        let n = 64 * 64;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        let ref_srgb = linear_planar_to_srgb_u8_interleaved(&r, &g, &b, 64, 64);
        let initial = lossy.compute_aq_field(&enc, &r, &g, &b, 4.0); // high d

        let mut trace_count = 0;
        let refined = refine_aq_field_gpu_auto(
            &enc,
            &lossy,
            &mut bg,
            &r,
            &g,
            &b,
            &ref_srgb,
            &initial,
            4.0, // > REFINEMENT_DISTANCE_THRESHOLD
            2,
            |_| trace_count += 1,
        )
        .expect("auto-gated refine");

        // Gated off: no trace events, identical field.
        assert_eq!(trace_count, 0);
        assert_eq!(refined.len(), initial.len());
        assert_eq!(refined, initial);
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

    /// End-to-end smoke test: refine_aq_field_gpu runs without panic on
    /// a small synthetic gradient. No assertion on the refined field's
    /// shape (calibration not yet validated against real images), only
    /// on completion + the trace callback firing the expected number of
    /// times.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_refine_aq_field_gpu_smoke() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, 64, 64);
        let mut bg = ButteraugliLoopGpu::new(&enc, 64, 64);
        let n = 64 * 64;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        let initial = lossy.compute_aq_field(&enc, &r, &g, &b, 1.0);
        let ref_srgb = linear_planar_to_srgb_u8_interleaved(&r, &g, &b, 64, 64);
        let mut traces = alloc::vec![];
        let refined = refine_aq_field_gpu(
            &enc,
            &lossy,
            &mut bg,
            &r,
            &g,
            &b,
            &ref_srgb,
            &initial,
            1.0,
            2,
            |t| traces.push(t),
        )
        .expect("refine_aq_field_gpu should not error");
        // 2 iters → 3 trace events (iter 0, 1, 2).
        assert_eq!(traces.len(), 3);
        // Refined field same shape as initial.
        assert_eq!(refined.len(), initial.len());
        // Score progression: trace[0].score >= trace[2].score most of
        // the time but synthetic gradient is unstable; just check finite.
        for t in &traces {
            assert!(t.score.is_finite());
            assert!(t.pnorm_3.is_finite());
        }
    }
}
