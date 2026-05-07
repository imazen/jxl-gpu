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
//! - `apply_epf_step0_gpu` — 5×5 plus kernel with 12-neighbor SAD
//!   (the heaviest pass)
//! - `apply_epf_step1_gpu` — 3×3 cross kernel with 3×3-plus SAD
//!   (the strong pass)
//! - `apply_epf_step2_gpu` — 3×3 cross kernel with single-pixel SAD
//!   (the weak pass)
//! - `select_sharpness_two_pass` — pure-CPU two-pass sharpness
//!   selection (greedy with neighbor bias + context refinement),
//!   ready to plug into `compute_epf_sharpness_gpu` once a
//!   reconstruct_xyb GPU orchestrator exists.
//!
//! Not yet covered (no orchestrator wrapper for this):
//! - `compute_epf_sharpness` — needs a `reconstruct_xyb` callable
//!   to produce the per-candidate base reconstructions. The
//!   per-candidate EPF + L2 chain is fully GPU-resident already
//!   (steps 0/1/2 + `block_l2_errors`); the missing piece is a
//!   reconstruct fork.
//!
//! Reshape vs upstream `apply_epf`:
//! - Upstream orchestrates step0 + step1 + step2 via the SIMD dispatch
//!   trampoline, with one scratch buffer per step.
//! - Our forks expose each step as an INDIVIDUAL pass the caller
//!   chains. Because GPU launches return new `Vec<f32>`, the caller
//!   can feed one step's output directly into the next step's input
//!   without managing intermediate scratch.

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

/// EPF Step 0 on GPU — 5×5 plus kernel with 3×3-plus SAD over 12
/// neighbors. The heaviest of the three EPF passes.
///
/// `pad` MUST be at least 3 (the 5×5 plus reaches ±2 from center;
/// the 3×3-plus SAD adds another ±1 around each neighbor → ±3
/// total reach). `sigma_scale` is the per-call multiplier — for
/// step 0 callers typically use `EPF_PASS0_SIGMA_SCALE * 1.65 = 1.485`
/// (matching upstream); `border_sigma_mul` is `EPF_BORDER_SAD_MUL`
/// (2/3).
#[allow(clippy::too_many_arguments)]
pub fn apply_epf_step0_gpu<R: Runtime>(
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
    enc.epf_step0_channels(
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

/// EPF candidate-list selector. Mirrors upstream
/// `compute_epf_sharpness` lines 774-778 — chooses
/// `[0, 4]` at high distance (> 4.5) and `[0, 2, 7]` otherwise.
///
/// Returned slice is one of two `'static` arrays so callers can
/// avoid allocating.
pub fn epf_sharpness_candidates(distance: f32) -> &'static [u8] {
    if distance > 4.5 {
        const HIGH_D: [u8; 2] = [0, 4];
        &HIGH_D
    } else {
        const NORMAL: [u8; 3] = [0, 2, 7];
        &NORMAL
    }
}

/// Apply the full EPF chain (step0 + step1 + step2 in sequence per
/// `epf_iters`) on GPU. Mirrors upstream `apply_epf` end-to-end.
///
/// Iteration semantics (matching upstream):
/// - `epf_iters == 0`: planes returned unchanged
/// - `epf_iters >= 3`: step 0 runs first (5×5 plus, 12 neighbors)
/// - `epf_iters >= 1`: step 1 always runs (3×3 cross + 5-pos SAD)
/// - `epf_iters >= 2`: step 2 runs last (3×3 cross + single-point SAD)
///
/// Each step pads its input independently (pad = 3 / 2 / 1) and
/// returns unpadded output. `inv_sigma` is reused across all steps
/// (one per 8×8 block).
///
/// Sigma scales applied:
/// - Step 0: `EPF_PASS0_SIGMA_SCALE * 1.65 = 1.485`
/// - Step 1: `1.65`
/// - Step 2: `EPF_PASS2_SIGMA_SCALE * 1.65 = 10.725`
///
/// All steps use `EPF_BORDER_SAD_MUL = 2/3` for the block-edge
/// reduction.
///
/// Returns `[plane_x, plane_y, plane_b]` (each `width * height`).
#[allow(clippy::too_many_arguments)]
pub fn apply_epf_chain_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    plane_x: &[f32],
    plane_y: &[f32],
    plane_b: &[f32],
    inv_sigma: &[f32],
    epf_iters: u32,
    width: u32,
    height: u32,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> [Vec<f32>; 3] {
    if epf_iters == 0 {
        return [plane_x.to_vec(), plane_y.to_vec(), plane_b.to_vec()];
    }

    let mut cur_x: Vec<f32> = plane_x.to_vec();
    let mut cur_y: Vec<f32> = plane_y.to_vec();
    let mut cur_b: Vec<f32> = plane_b.to_vec();

    let pad_for = |p: u32, x: &[f32]| -> Vec<f32> {
        enc.pad_plane_channel(x, width, height, p)
    };

    if epf_iters >= 3 {
        let pad = 3_u32;
        let px = pad_for(pad, &cur_x);
        let py = pad_for(pad, &cur_y);
        let pb = pad_for(pad, &cur_b);
        let (ox, oy, ob) = apply_epf_step0_gpu(
            enc,
            &px, &py, &pb,
            inv_sigma,
            width, height, xsize_blocks, ysize_blocks, pad,
            EPF_PASS0_SIGMA_SCALE * 1.65,
            EPF_BORDER_SAD_MUL,
        );
        cur_x = ox;
        cur_y = oy;
        cur_b = ob;
    }
    if epf_iters >= 1 {
        let pad = 2_u32;
        let px = pad_for(pad, &cur_x);
        let py = pad_for(pad, &cur_y);
        let pb = pad_for(pad, &cur_b);
        let (ox, oy, ob) = apply_epf_step1_gpu(
            enc,
            &px, &py, &pb,
            inv_sigma,
            width, height, xsize_blocks, ysize_blocks, pad,
            1.65,
            EPF_BORDER_SAD_MUL,
        );
        cur_x = ox;
        cur_y = oy;
        cur_b = ob;
    }
    if epf_iters >= 2 {
        let pad = 1_u32;
        let px = pad_for(pad, &cur_x);
        let py = pad_for(pad, &cur_y);
        let pb = pad_for(pad, &cur_b);
        let (ox, oy, ob) = apply_epf_step2_gpu(
            enc,
            &px, &py, &pb,
            inv_sigma,
            width, height, xsize_blocks, ysize_blocks, pad,
            EPF_PASS2_SIGMA_SCALE * 1.65,
            EPF_BORDER_SAD_MUL,
        );
        cur_x = ox;
        cur_y = oy;
        cur_b = ob;
    }

    [cur_x, cur_y, cur_b]
}

/// Two-pass per-block sharpness selection.
///
/// Pure-CPU port of the selection logic from
/// `jxl_encoder::vardct::epf::compute_epf_sharpness` — *only* the
/// part that consumes per-candidate error maps. The expensive part
/// (running EPF + block_l2 per candidate) is the caller's job:
/// run the full GPU EPF chain once per `candidate` value with a
/// uniform sharpness map, then call `block_l2_errors` to get the
/// per-block errors for that candidate.
///
/// `candidates` is the candidate sharpness list (typically `[0, 2, 7]`
/// or `[0, 4]` at high distance). `error_maps[ci][block_idx]` is the
/// per-block reconstruction error when the entire image was filtered
/// with `candidates[ci]`. `clamped_distance` is `params.distance`
/// clamped to `[0.5, 10.0]`.
///
/// Returns one `u8` per block in row-major order; values are
/// drawn from `candidates`.
///
/// Mirrors upstream's two-pass algorithm exactly:
/// 1. Greedy pass with `K_FAVOR_NO_SMOOTHING = 0.99` bias toward
///    sharpness=0 and a neighbor-preference fallback.
/// 2. Context-based reweighting using top/left neighbor sharpness as
///    context, with libjxl's signature `size_t / size_t` integer
///    division (which makes the entropy term a no-op for most
///    contexts — only the `c3` bias on sharpness=0 has real effect).
pub fn select_sharpness_two_pass(
    error_maps: &[Vec<f32>],
    candidates: &[u8],
    clamped_distance: f32,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> Vec<u8> {
    let nblocks = xsize_blocks * ysize_blocks;
    let num_candidates = candidates.len();
    debug_assert_eq!(error_maps.len(), num_candidates);
    for em in error_maps {
        debug_assert_eq!(em.len(), nblocks);
    }

    // Map candidate value → context-LUT index.
    let candidate_lut: Vec<usize> = candidates
        .iter()
        .map(|&v| match v {
            0 => 0,
            2 | 4 => 1,
            7 => 2,
            _ => 0,
        })
        .collect();

    // Pass 1: greedy with neighbor preference + favor-no-smoothing bias.
    const K_FAVOR_NO_SMOOTHING: f32 = 0.99;
    let mut sharpness_map = vec![4_u8; nblocks];
    let num_contexts = num_candidates * num_candidates;
    let mut histo = vec![vec![0_u32; num_candidates]; num_contexts];

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let block_idx = by * xsize_blocks + bx;

            let (top_val, top_err) = if by > 0 {
                let top_idx = (by - 1) * xsize_blocks + bx;
                let top_s = sharpness_map[top_idx];
                let top_ci = candidates.iter().position(|&c| c == top_s).unwrap_or(0);
                (top_ci, error_maps[top_ci][top_idx])
            } else {
                (0, f32::MAX)
            };
            let (left_val, left_err) = if bx > 0 {
                let left_idx = by * xsize_blocks + bx - 1;
                let left_s = sharpness_map[left_idx];
                let left_ci = candidates.iter().position(|&c| c == left_s).unwrap_or(0);
                (left_ci, error_maps[left_ci][left_idx])
            } else {
                (0, f32::MAX)
            };

            let mut best_ci = 0;
            let mut best_err = f32::MAX;
            for ci in 0..num_candidates {
                let mut err = error_maps[ci][block_idx];
                if candidates[ci] == 0 {
                    err *= K_FAVOR_NO_SMOOTHING;
                }
                if err < best_err {
                    best_err = err;
                    best_ci = ci;
                }
            }
            let selected_ci = if best_err < top_err.min(left_err) {
                best_ci
            } else if top_err < left_err {
                top_val
            } else {
                left_val
            };
            sharpness_map[block_idx] = candidates[selected_ci];
            let ctx = candidate_lut[top_val] * num_candidates + candidate_lut[left_val];
            if ctx < num_contexts {
                histo[ctx][selected_ci] += 1;
            }
        }
    }

    // Pass 2: context-based reweighting.
    let clamped_d = clamped_distance.clamp(0.5, 10.0);
    let c3base: f32 = 0.980_172;
    let c3clamp: f32 = 0.859_703_4;
    let c3 = c3clamp.max(c3base.powf(clamped_d));
    let c5: f32 = 0.108_769_04;

    // libjxl init: size_t totals[ctx] = 1, then accumulate counts.
    let mut totals = vec![1_usize; num_contexts];
    for ctx in 0..num_contexts {
        for &count in &histo[ctx][..num_candidates] {
            totals[ctx] += count as usize;
        }
    }

    // Multipliers per (ctx, ci). Integer division matches libjxl exactly:
    // for count < total → ratio = 0 → log1p(0) = 0 → mul stays 1.0
    // (which makes the entropy term a no-op for most contexts).
    let mut muls = vec![vec![1.0_f32; num_candidates]; num_contexts];
    for ctx in 0..num_contexts {
        for ci in 0..num_candidates {
            let count = histo[ctx][ci] as usize;
            let ratio = count / totals[ctx]; // integer division
            let mut mul = 1.0 / (1.0 + c5 * (1.0 + ratio as f32).ln() / clamped_d);
            if candidates[ci] == 0 {
                mul *= c3;
            }
            muls[ctx][ci] = mul;
        }
    }

    // Re-scan with context multipliers.
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let block_idx = by * xsize_blocks + bx;
            let top_ci = if by > 0 {
                let top_s = sharpness_map[(by - 1) * xsize_blocks + bx];
                candidates.iter().position(|&c| c == top_s).unwrap_or(0)
            } else {
                0
            };
            let left_ci = if bx > 0 {
                let left_s = sharpness_map[by * xsize_blocks + bx - 1];
                candidates.iter().position(|&c| c == left_s).unwrap_or(0)
            } else {
                0
            };
            let ctx = candidate_lut[top_ci] * num_candidates + candidate_lut[left_ci];
            let ctx_clamped = ctx.min(num_contexts - 1);
            let mut best_ci = 0;
            let mut best_err = f32::MAX;
            for ci in 0..num_candidates {
                let err = error_maps[ci][block_idx] * muls[ctx_clamped][ci];
                if err < best_err {
                    best_err = err;
                    best_ci = ci;
                }
            }
            sharpness_map[block_idx] = candidates[best_ci];
        }
    }

    sharpness_map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epf_sharpness_candidates_normal() {
        for d in [0.5_f32, 1.0, 2.0, 4.0, 4.5] {
            assert_eq!(epf_sharpness_candidates(d), &[0_u8, 2, 7]);
        }
    }

    #[test]
    fn test_epf_sharpness_candidates_high_distance() {
        for d in [4.51_f32, 5.0, 7.0, 100.0] {
            assert_eq!(epf_sharpness_candidates(d), &[0_u8, 4]);
        }
    }

    #[test]
    fn test_select_sharpness_two_pass_clear_winner() {
        // Construct a 4×3 grid where candidate 1 (sharpness=2) is
        // strictly best for every block. The selector must pick it
        // for every block on both passes.
        let xb = 4_usize;
        let yb = 3_usize;
        let nb = xb * yb;
        let candidates = [0_u8, 2_u8, 7_u8];
        // Errors: candidate 0 = 10.0, candidate 1 = 1.0, candidate 2 = 50.0
        // (candidate 0 gets * K_FAVOR_NO_SMOOTHING = 0.99 → 9.9 still > 1.0).
        let error_maps = vec![
            vec![10.0_f32; nb],
            vec![1.0_f32; nb],
            vec![50.0_f32; nb],
        ];
        let out = select_sharpness_two_pass(&error_maps, &candidates, 1.0, xb, yb);
        assert_eq!(out.len(), nb);
        for &v in &out {
            assert_eq!(v, 2);
        }
    }

    #[test]
    fn test_select_sharpness_two_pass_no_smoothing_bias() {
        // Candidate 0 (sharpness=0) error = 1.005, candidate 1 = 1.000.
        // After 0.99 bias: 1.005 * 0.99 = 0.99495 < 1.0 → picks 0.
        let xb = 2_usize;
        let yb = 2_usize;
        let nb = xb * yb;
        let candidates = [0_u8, 2_u8, 7_u8];
        let error_maps = vec![
            vec![1.005_f32; nb],
            vec![1.0_f32; nb],
            vec![100.0_f32; nb],
        ];
        let out = select_sharpness_two_pass(&error_maps, &candidates, 1.0, xb, yb);
        // Pass-2 c3 multiplier on sharpness=0 only strengthens this.
        for &v in &out {
            assert_eq!(v, 0);
        }
    }

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
