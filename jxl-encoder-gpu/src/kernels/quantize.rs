// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! AC coefficient quantization (DCT8) with dead-zone thresholding.
//!
//! Mirrors `jxl_encoder_simd::quantize::quantize_dct8_scalar`.
//!
//! Per-block: 64 coefficients, 4 per-quadrant thresholds, one
//! per-block `qac_qm` scalar. DC (index 0) is forced to 0.
//!
//! ## Rounding mode caveat
//!
//! The CPU `_scalar` reference uses `f32::round_ties_even()` (matching
//! libjxl `rintf` / Highway `Round`). cubecl 0.10 doesn't expose a
//! ties-to-even op; we use `round_ties_even` implemented in user code
//! to preserve bit-exact parity for half-integer values. This matters
//! for AdjustQuantBlockAC heuristics in the encoder (see jxl-encoder
//! CLAUDE.md → "Rounding mode mismatch (9ef2819)").

use cubecl::prelude::*;

/// Round a finite f32 to the nearest integer with ties-to-even (banker's
/// rounding). Matches `f32::round_ties_even()`. For values whose absolute
/// value is < 2^31 (always true for quantized DCT coeffs in practice).
#[cube]
fn round_ties_even_to_i32(x: f32) -> i32 {
    // `as i32` truncates toward zero. We add a half-aware adjustment
    // before the cast.
    let trunc_i = x as i32;
    let trunc_f = trunc_i as f32;
    let frac = x - trunc_f; // sign matches sign(x)

    // Determine adjustment without branching where possible.
    let abs_frac = f32::abs(frac);
    let trunc_is_odd = (trunc_i & 1i32) != 0i32;

    // Decide whether to bump trunc by ±1.
    let bump_unconditional = abs_frac > 0.5f32;
    let bump_tie = (abs_frac == 0.5f32) && trunc_is_odd;
    let do_bump = bump_unconditional || bump_tie;

    if do_bump {
        if frac > 0.0f32 {
            trunc_i + 1i32
        } else {
            trunc_i - 1i32
        }
    } else {
        trunc_i
    }
}

/// Per-block generic quantize with dead-zone. One cube per block.
/// All blocks share the same `grid_width`/`grid_height`/`llf_x`/`llf_y`
/// (passed as scalars). Coefficients within the LLF rectangle are forced
/// to 0; remaining coefs use the same dead-zone math as DCT8.
///
/// Mirrors `jxl_encoder_simd::quantize::quantize_large_scalar`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn quantize_large_kernel(
    coeffs: &Array<f32>,
    weights: &Array<f32>,
    qac_qm: &Array<f32>,
    thresholds: &Array<f32>,
    output: &mut Array<i32>,
    grid_width: u32,
    grid_height: u32,
    llf_x: u32,
    llf_y: u32,
) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = qac_qm.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let gw = grid_width as usize;
    let gh = grid_height as usize;
    let lx = llf_x as usize;
    let ly = llf_y as usize;
    let half_h = gh / 2usize;
    let half_w = gw / 2usize;
    let size = gw * gh;
    let off = block_idx * size;
    let qac = qac_qm[block_idx];

    let t0 = thresholds[0usize];
    let t1 = thresholds[1usize];
    let t2 = thresholds[2usize];
    let t3 = thresholds[3usize];

    let mut idx: u32 = 0u32;
    while (idx as usize) < size {
        let iu = idx as usize;
        let y = iu / gw;
        let x = iu - y * gw;
        // LLF skip
        if y < ly && x < lx {
            output[off + iu] = i32::new(0);
        } else {
            let row_hi = y >= half_h;
            let col_hi = x >= half_w;
            let thr = if row_hi {
                if col_hi { t3 } else { t2 }
            } else if col_hi {
                t1
            } else {
                t0
            };
            let val = coeffs[off + iu] * (1.0f32 / weights[off + iu]) * qac;
            let absv = f32::abs(val);
            output[off + iu] = if absv < thr {
                i32::new(0)
            } else {
                round_ties_even_to_i32(val)
            };
        }
        idx += 1u32;
    }
}

/// Per-block DCT8 quantize with dead-zone. One cube per block.
///
/// Layout:
/// - `coeffs`: `num_blocks * 64` f32 (per-block DCT coefficients)
/// - `weights`: `num_blocks * 64` f32 (per-block dequant weights — same
///   table replicated per block in the simple case, but the kernel allows
///   per-block variation)
/// - `qac_qm`: `num_blocks` f32 (per-block `qac * qm_mul` scalar)
/// - `thresholds`: 4 f32 (per-quadrant dead-zone thresholds, broadcast)
/// - `output`: `num_blocks * 64` i32 (quantized AC coefficients)
#[cube(launch_unchecked)]
pub fn quantize_dct8_kernel(
    coeffs: &Array<f32>,
    weights: &Array<f32>,
    qac_qm: &Array<f32>,
    thresholds: &Array<f32>,
    output: &mut Array<i32>,
) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = qac_qm.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;
    let qac = qac_qm[block_idx];

    let t0 = thresholds[0usize];
    let t1 = thresholds[1usize];
    let t2 = thresholds[2usize];
    let t3 = thresholds[3usize];

    output[off] = 0i32; // DC

    let mut idx: u32 = 1u32;
    while idx < 64u32 {
        let iu = idx as usize;
        let y = idx / 8u32;
        let x = idx - y * 8u32;
        let row_hi = y >= 4u32;
        let col_hi = x >= 4u32;
        // 2x2 quadrant -> threshold index
        let thr = if row_hi {
            if col_hi { t3 } else { t2 }
        } else if col_hi {
            t1
        } else {
            t0
        };
        let val = coeffs[off + iu] * (1.0f32 / weights[off + iu]) * qac;
        let absv = f32::abs(val);
        output[off + iu] = if absv < thr {
            i32::new(0)
        } else {
            round_ties_even_to_i32(val)
        };
        idx += 1u32;
    }
}

/// Broadcast-weights variant of [`quantize_dct8_kernel`]. Same shape
/// except `weights` is exactly 64 f32 (one DCT8 quant matrix, broadcast
/// across all blocks). Saves `num_blocks - 1` copies of the per-block
/// weights buffer (e.g., 12 MB at 1024² → 256 bytes per channel) and
/// has uniformly-coalesced weight reads inside a warp.
#[cube(launch_unchecked)]
pub fn quantize_dct8_kernel_broadcast_w(
    coeffs: &Array<f32>,
    weights: &Array<f32>,
    qac_qm: &Array<f32>,
    thresholds: &Array<f32>,
    output: &mut Array<i32>,
) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = qac_qm.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;
    let qac = qac_qm[block_idx];

    let t0 = thresholds[0usize];
    let t1 = thresholds[1usize];
    let t2 = thresholds[2usize];
    let t3 = thresholds[3usize];

    output[off] = 0i32; // DC

    let mut idx: u32 = 1u32;
    while idx < 64u32 {
        let iu = idx as usize;
        let y = idx / 8u32;
        let x = idx - y * 8u32;
        let row_hi = y >= 4u32;
        let col_hi = x >= 4u32;
        let thr = if row_hi {
            if col_hi { t3 } else { t2 }
        } else if col_hi {
            t1
        } else {
            t0
        };
        // Broadcast: weights[iu] not weights[off + iu].
        let val = coeffs[off + iu] * (1.0f32 / weights[iu]) * qac;
        let absv = f32::abs(val);
        output[off + iu] = if absv < thr {
            i32::new(0)
        } else {
            round_ties_even_to_i32(val)
        };
        idx += 1u32;
    }
}
