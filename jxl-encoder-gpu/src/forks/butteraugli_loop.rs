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

/// Reduce a per-pixel diffmap to per-(8×8 block) tile distances.
///
/// Mirrors upstream's tile-distance reduction (a per-block aggregate
/// of the diffmap used to perturb `quant_field` per iteration).
/// Returns `xsize_blocks * ysize_blocks` floats; entry `[by, bx]` is
/// the maximum diffmap value within the 8×8 block at `(bx*8, by*8)`.
///
/// Pure-CPU reduction over the (already small) diffmap; per-block work
/// is one max over 64 pixels — negligible vs the diffmap compute.
pub fn diffmap_to_tile_distances(
    diffmap: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> Vec<f32> {
    debug_assert_eq!(diffmap.len(), width * height);
    debug_assert!(xsize_blocks * 8 <= width.max(xsize_blocks * 8));
    let mut out = alloc::vec![0.0_f32; xsize_blocks * ysize_blocks];
    for by in 0..ysize_blocks {
        let py0 = by * 8;
        let py1 = (py0 + 8).min(height);
        for bx in 0..xsize_blocks {
            let px0 = bx * 8;
            let px1 = (px0 + 8).min(width);
            let mut m = 0.0_f32;
            for py in py0..py1 {
                let row = &diffmap[py * width + px0..py * width + px1];
                for &v in row {
                    if v > m {
                        m = v;
                    }
                }
            }
            out[by * xsize_blocks + bx] = m;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diffmap_to_tile_distances_uniform() {
        // Uniform diffmap → every tile distance equals the constant.
        let w = 16;
        let h = 16;
        let xb = 2;
        let yb = 2;
        let diffmap = alloc::vec![0.42_f32; w * h];
        let out = diffmap_to_tile_distances(&diffmap, w, h, xb, yb);
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert_eq!(v, 0.42);
        }
    }

    #[test]
    fn test_diffmap_to_tile_distances_per_block_max() {
        // 16×16 diffmap = 4 blocks; each block has one peak pixel
        // distinguishing it from the others.
        let w = 16;
        let h = 16;
        let xb = 2;
        let yb = 2;
        let mut diffmap = alloc::vec![0.0_f32; w * h];
        // Block (0,0) peak = 1.0 at pixel (0,0)
        diffmap[0] = 1.0;
        // Block (1,0) peak = 2.0 at pixel (8,0)
        diffmap[8] = 2.0;
        // Block (0,1) peak = 3.0 at pixel (0,8)
        diffmap[8 * w] = 3.0;
        // Block (1,1) peak = 4.0 at pixel (8,8)
        diffmap[8 * w + 8] = 4.0;
        let out = diffmap_to_tile_distances(&diffmap, w, h, xb, yb);
        assert_eq!(out, alloc::vec![1.0, 2.0, 3.0, 4.0]);
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
