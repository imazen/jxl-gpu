// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/chroma_from_luma.rs (BSD-3-Clause via
// libjxl + AGPL/commercial), reshaped for batched per-image GPU dispatch.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted Chroma-from-Luma multiplier search.
//!
//! ## Reshape vs upstream `chroma_from_luma::find_best_multiplier`
//!
//! Upstream solves the CfL fit ONE tile at a time, parallelized across
//! tiles via rayon::par_chunks. Each tile is small (~64×64 pixels =
//! 64×64 = 4096 AC coefficients), so the per-call SIMD work is short.
//!
//! On GPU the per-tile launch overhead dominates: one launch per tile
//! is microseconds of bookkeeping for ~10 microseconds of compute.
//! The GPU shape is to flatten the tile loop into one big launch:
//! pass `(values_m, values_s)` for ALL tiles concatenated, plus a
//! `bases` array of length num_tiles, and run one kernel that does
//! all tiles in parallel.
//!
//! This module exposes:
//! - `find_best_multiplier_gpu` — single-tile API mirroring upstream
//!   for drop-in substitution at existing call sites
//! - `find_best_multipliers_batch_gpu` — multi-tile batched API
//!   exposing the GPU's natural shape (one launch covers all tiles)
//! - `find_best_multiplier_newton_gpu` / `_batch_gpu` — same for the
//!   Newton-refined variant
//!
//! The clamping to i8 happens here (the GPU returns i32 for runtime
//! convenience).

use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

#[inline]
fn clamp_to_i8(v: i32) -> i8 {
    v.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

/// Single-tile CfL multiplier via regularized least-squares on GPU.
/// Mirrors upstream `jxl_encoder::vardct::chroma_from_luma::find_best_multiplier`
/// (use_newton=false branch).
pub fn find_best_multiplier_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
) -> i8 {
    assert!(values_m.len() >= num);
    assert!(values_s.len() >= num);
    let bases = [base];
    let raw = enc.cfl_multipliers(
        &values_m[..num],
        &values_s[..num],
        &bases,
        num as u32,
        distance_mul,
    );
    debug_assert_eq!(raw.len(), 1);
    clamp_to_i8(raw[0])
}

/// Single-tile CfL multiplier via Newton's method on GPU.
/// Mirrors upstream `find_best_multiplier` (use_newton=true branch).
pub fn find_best_multiplier_newton_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    values_m: &[f32],
    values_s: &[f32],
    num: usize,
    base: f32,
    distance_mul: f32,
    newton_eps: f32,
    newton_max_iters: usize,
) -> i8 {
    assert!(values_m.len() >= num);
    assert!(values_s.len() >= num);
    let bases = [base];
    let raw = enc.cfl_multipliers_newton(
        &values_m[..num],
        &values_s[..num],
        &bases,
        num as u32,
        distance_mul,
        newton_eps,
        newton_max_iters as u32,
    );
    debug_assert_eq!(raw.len(), 1);
    clamp_to_i8(raw[0])
}

/// Batched CfL search over many tiles in one GPU launch.
///
/// `values_m` and `values_s` are concatenated tile data: tile `t`'s
/// values live at `[t*num_per_tile, (t+1)*num_per_tile)`. `bases.len()`
/// determines the tile count.
///
/// Returns `bases.len()` clamped i8 multipliers.
///
/// This is the GPU's natural shape — one kernel launch covers all
/// tiles instead of one launch per tile.
pub fn find_best_multipliers_batch_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    values_m: &[f32],
    values_s: &[f32],
    bases: &[f32],
    num_per_tile: usize,
    distance_mul: f32,
) -> Vec<i8> {
    let n = bases.len() * num_per_tile;
    assert_eq!(values_m.len(), n);
    assert_eq!(values_s.len(), n);
    let raw = enc.cfl_multipliers(values_m, values_s, bases, num_per_tile as u32, distance_mul);
    raw.into_iter().map(clamp_to_i8).collect()
}

/// Batched Newton-refined CfL search.
#[allow(clippy::too_many_arguments)]
pub fn find_best_multipliers_newton_batch_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    values_m: &[f32],
    values_s: &[f32],
    bases: &[f32],
    num_per_tile: usize,
    distance_mul: f32,
    newton_eps: f32,
    newton_max_iters: usize,
) -> Vec<i8> {
    let n = bases.len() * num_per_tile;
    assert_eq!(values_m.len(), n);
    assert_eq!(values_s.len(), n);
    let raw = enc.cfl_multipliers_newton(
        values_m,
        values_s,
        bases,
        num_per_tile as u32,
        distance_mul,
        newton_eps,
        newton_max_iters as u32,
    );
    raw.into_iter().map(clamp_to_i8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_find_best_multiplier_zero_input() {
        // values_m and values_s both zero → result should be base (0).
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let m = vec![0.0_f32; 64];
        let s = vec![0.0_f32; 64];
        let r = find_best_multiplier_gpu(&enc, &m, &s, 64, 0.0, 1.0);
        assert_eq!(r, 0, "zero input → multiplier should be 0, got {r}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_find_best_multiplier_correlated_in_range() {
        // Strongly correlated input → result is in valid i8 range.
        // (Exact magnitude depends on K_DISTANCE_MULTIPLIER conventions
        // and the LS regularizer; we just check the call succeeds and
        // the result is in [-128, 127].)
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n = 256;
        let m: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin()).collect();
        let s: Vec<f32> = m.iter().map(|&x| 0.5 * x).collect();
        let r = find_best_multiplier_gpu(&enc, &m, &s, n, 0.0, 1.0);
        // i8 already constrains the range; this test confirms non-panic.
        let _ = r;
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_batch_matches_single_tile() {
        // Batched call over 3 tiles should match 3 single-tile calls.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n_per_tile = 64;
        let m1: Vec<f32> = (0..n_per_tile).map(|i| (i as f32 * 0.07).sin()).collect();
        let s1: Vec<f32> = m1.iter().map(|x| 0.3 * x).collect();
        let m2: Vec<f32> = (0..n_per_tile)
            .map(|i| ((i + 5) as f32 * 0.11).cos())
            .collect();
        let s2: Vec<f32> = m2.iter().map(|x| -0.2 * x).collect();
        let m3 = vec![0.5_f32; n_per_tile];
        let s3 = vec![0.5_f32; n_per_tile];

        let r1 = find_best_multiplier_gpu(&enc, &m1, &s1, n_per_tile, 0.0, 1.0);
        let r2 = find_best_multiplier_gpu(&enc, &m2, &s2, n_per_tile, 0.0, 1.0);
        let r3 = find_best_multiplier_gpu(&enc, &m3, &s3, n_per_tile, 0.0, 1.0);

        let mut m_all = Vec::with_capacity(3 * n_per_tile);
        m_all.extend_from_slice(&m1);
        m_all.extend_from_slice(&m2);
        m_all.extend_from_slice(&m3);
        let mut s_all = Vec::with_capacity(3 * n_per_tile);
        s_all.extend_from_slice(&s1);
        s_all.extend_from_slice(&s2);
        s_all.extend_from_slice(&s3);
        let bases = [0.0_f32; 3];
        let batched =
            find_best_multipliers_batch_gpu(&enc, &m_all, &s_all, &bases, n_per_tile, 1.0);
        assert_eq!(batched, vec![r1, r2, r3]);
    }
}
