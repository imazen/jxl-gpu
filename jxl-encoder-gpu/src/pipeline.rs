// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Phase 3: whole-image cost-grid composition + host-side partition
//! selector.
//!
//! Implements the **whole-image-per-strategy** AC strategy search
//! pattern documented in `CLAUDE.md`. For each candidate strategy S,
//! one coherent whole-image pipeline computes per-block cost. The
//! host-side partition selector then enumerates legal partitions per
//! 16×16 / 32×32 / 64×64 region and picks the min-cost partition.
//!
//! ## Cost-grid coverage
//!
//! Per-strategy cost grids are available in two flavors:
//!
//! - **Single-channel**: simplified Y-only proxy. Cheaper, useful for
//!   smoke tests and as a strategy-pick predictor where the full
//!   3-channel signal isn't needed (turns out to be a near-equivalent
//!   predictor on natural photos).
//! - **3-channel XYB + mask1x1**: full XYB-weighted reconstruction
//!   error masked by per-pixel perceptual sensitivity. Matches what
//!   the VarDCT encoder actually pays for in production.
//!
//! | Strategy   | Single-ch        | 3-channel        | Sub-cells / region |
//! |---|---|---|---|
//! | DCT8       | `*_dct8_*`       | `*_dct8_xyb`     | 1                  |
//! | DCT16×8    | `*_dct16x8_*`    | `*_dct16x8_xyb`  | 2 (2×1)            |
//! | DCT8×16    | `*_dct8x16_*`    | `*_dct8x16_xyb`  | 2 (1×2)            |
//! | DCT16×16   | `*_dct16x16_*`   | `*_dct16x16_xyb` | 4 (2×2)            |
//! | DCT32×16   | `*_dct32x16_*`   | `*_dct32x16_xyb` | 8 (4×2)            |
//! | DCT16×32   | `*_dct16x32_*`   | `*_dct16x32_xyb` | 8 (2×4)            |
//! | DCT32×32   | `*_dct32x32_*`   | `*_dct32x32_xyb` | 16 (4×4)           |
//! | DCT64×32   | `*_dct64x32_*`   | `*_dct64x32_xyb` | 32 (8×4)           |
//! | DCT32×64   | `*_dct32x64_*`   | `*_dct32x64_xyb` | 32 (4×8)           |
//! | DCT64×64   | `*_dct64x64_*`   | `*_dct64x64_xyb` | 64 (8×8)           |
//! | DCT4×4     | `*_dct4x4_*`     | `*_dct4x4_xyb`   | 1 (sub-block)      |
//! | DCT4×8     | `*_dct4x8_*`     | `*_dct4x8_xyb`   | 1 (sub-block)      |
//! | DCT8×4     | `*_dct8x4_*`     | `*_dct8x4_xyb`   | 1 (sub-block)      |
//! | IDENTITY   | `*_identity_*`   | `*_identity_xyb` | 1 (sub-block DC)   |
//! | DCT2X2     | `*_dct2x2_*`     | `*_dct2x2_xyb`   | 1 (Hadamard cascade)|
//!
//! Full DCT4/8/16/32/64 family + IDENTITY + DCT2X2 covered (15
//! strategies × 2 flavors = 30 cost-grid functions). All standard
//! JXL AC strategies except AFV0-3 are wired. Remaining:
//! CfL-aware variants + AFV (kernels not yet ported).
//!
//! ## Partition selectors
//!
//! - [`select_partitions_16x16_full`] — picks per 16×16 region across
//!   {DCT16×16, 2×DCT16×8, 2×DCT8×16, 4×DCT8}.
//! - [`select_partitions_32x32_full`] — picks per 32×32 region across
//!   {DCT32×32, 2×DCT32×16, 2×DCT16×32, 4×Sub16×16}.
//! - [`select_partitions_64x64`] — picks per 64×64 region across
//!   {DCT64×64, 2×DCT64×32, 2×DCT32×64, 4×Sub32×32}. All three
//!   selectors compose recursively for full hierarchical strategy
//!   selection.

// `if let Some(c) = opt { if c < best { ... } }` pattern is intentionally
// nested for readability over collapsed `if let Some(c) = opt && c < best`
// (which lacks the explicit cost variable in the inner block).
#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_match)]

use alloc::vec;
use alloc::vec::Vec;

use cubecl::Runtime;
use cubecl::prelude::*;
use cubecl::server::Handle;

#[allow(dead_code)]
fn _unused_vec() -> Vec<f32> {
    Vec::new()
}

use crate::launch::{
    block_l2::block_l2,
    dct2x2::{dct2x2_forward, dct2x2_inverse},
    dct4::{dct_4x4_full, dct_4x8_full, dct_8x4_full, idct_4x4_full, idct_4x8_full, idct_8x4_full},
    dct8::{dct_8x8, idct_8x8},
    dct16::{dct_8x16, dct_16x8, dct_16x16, idct_8x16, idct_16x8, idct_16x16},
    dct32::{dct_16x32, dct_32x16, dct_32x32, idct_16x32, idct_32x16, idct_32x32},
    dct64::{dct_32x64, dct_64x32, dct_64x64, idct_32x64, idct_64x32, idct_64x64},
    identity::{identity_forward, identity_inverse},
    quantize::{quantize_dct8, quantize_large},
};

/// Output of a single-strategy cost grid evaluation.
pub struct CostGrid {
    /// Per-block scalar cost (entropy + pixel-loss). Length = `num_blocks`.
    /// Block ordering: row-major, `block_y * xsize_blocks + block_x`.
    pub costs: Handle,
    pub xsize_blocks: u32,
    pub ysize_blocks: u32,
}

/// Compute the DCT8 strategy's whole-image cost grid for a single channel.
///
/// **Inputs:**
/// - `original`: contiguous block-major Y-channel pixels. Shape:
///   `num_blocks * 64`. Each 8x8 block laid out as 64 contiguous floats
///   (row-major within block).
/// - `weights`: per-block dequant weights. Shape: `num_blocks * 64`.
///   Typically the same DCT8 quant matrix replicated per block.
/// - `qac_qm`: per-block `qac * qm_mul` scalar. Shape: `num_blocks`.
/// - `thresholds`: 4 dead-zone thresholds (one per quadrant).
///
/// **Output:** `CostGrid` with one f32 cost per block.
///
/// **Note:** This prototype uses a simple cost = block_l2(orig - reconstructed).
/// Full Phase 3 will combine `entropy_coeffs_pixel` output with `pixel_loss`
/// via documented entropy_mul / loss multipliers per strategy.
pub fn compute_cost_grid_dct8_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks * ysize_blocks;
    let nb = num_blocks as usize;
    let n_coef = nb * 64;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    // Allocate intermediate buffers
    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    // Step 1: Forward DCT
    dct_8x8::<R>(client, original.clone(), h_dct.clone(), num_blocks);

    // Step 2: Quantize
    quantize_dct8::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
    );

    // Step 3: Dequantize via scalar mul
    //
    // Full dequant_dct8 is 3-channel + CfL. For this single-channel
    // prototype, we just apply the inverse of quantize: dequant = quant
    // * weights (treating qac_qm as already absorbed). The proper
    // implementation would call dequant_dct8 with bias adjustment.
    dequant_simple_dct8::<R>(client, h_quant, weights, h_dequant.clone(), num_blocks);

    // Step 4: Inverse DCT
    idct_8x8::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    // Step 5: Per-block L2 vs original.
    // block_l2 expects 3-channel inputs (orig_x/y/b + recon_x/y/b + mask1x1)
    // — for the single-channel prototype, we pass the same plane for all
    // 3 channels with mask=1.0. The result is Y * (W_X + W_Y + W_B) summed
    // for the diff. Caller should interpret this as a proxy.
    let n_pixels = nb * 64;
    let h_zero = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n_pixels]));
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb]));

    // For the prototype, treat block_l2's 3-channel input as
    // (orig, orig, orig) and (recon, recon, recon) — measures Y-channel
    // diff weighted by sum of channel weights.
    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks,
        ysize_blocks,
        xsize_blocks * 8,
    );
    let _ = h_zero;

    CostGrid {
        costs: h_costs,
        xsize_blocks,
        ysize_blocks,
    }
}

/// Simple per-block dequantize: dequant[i] = quant[i] * weights[i].
/// (No bias adjustment, no CfL — for the single-channel prototype.)
#[cube(launch_unchecked)]
fn dequant_simple_dct8_kernel(quant: &Array<i32>, weights: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = quant.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;
    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        output[off + iu] = (quant[off + iu] as f32) * weights[off + iu];
        i += 1u32;
    }
}

/// Generic per-block dequantize for arbitrary `block_size` (256 for DCT16,
/// 1024 for DCT32, 4096 for DCT64).
#[cube(launch_unchecked)]
fn dequant_simple_generic_kernel(
    quant: &Array<i32>,
    weights: &Array<f32>,
    output: &mut Array<f32>,
    block_size: u32,
) {
    let block_idx = ABSOLUTE_POS;
    let bs = block_size as usize;
    let n_blocks = quant.len() / bs;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * bs;
    let mut i: u32 = 0u32;
    while (i as usize) < bs {
        let iu = i as usize;
        output[off + iu] = (quant[off + iu] as f32) * weights[off + iu];
        i += 1u32;
    }
}

fn dequant_simple_generic<R: Runtime>(
    client: &ComputeClient<R>,
    quant: Handle,
    weights: Handle,
    output: Handle,
    num_blocks: u32,
    block_size: u32,
) {
    let n = (num_blocks as usize) * (block_size as usize);
    let cubes = num_blocks.max(1);
    unsafe {
        dequant_simple_generic_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(quant, n),
            ArrayArg::from_raw_parts(weights, n),
            ArrayArg::from_raw_parts(output, n),
            block_size,
        );
    }
}

/// Compute the DCT16×16 strategy's whole-image cost grid for a single channel.
///
/// Counterpart to `compute_cost_grid_dct8_single_channel`. Each input
/// 16×16-block is laid out as 256 contiguous floats in `original`. Output
/// is one f32 cost per 16×16 block.
///
/// Pipeline: dct_16x16 → quantize_large(grid_width=16, grid_height=16, llf=2x2)
/// → dequant_simple_generic → idct_16x16 → block_l2 (16x16 region).
pub fn compute_cost_grid_dct16x16_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks_16: u32,
    ysize_blocks_16: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_16 * ysize_blocks_16;
    let nb = num_blocks as usize;
    let n_coef = nb * 256;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_16x16::<R>(client, original.clone(), h_dct.clone(), num_blocks);

    quantize_large::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
        16, // grid_width
        16, // grid_height
        2,  // llf_x
        2,  // llf_y
    );

    dequant_simple_generic::<R>(client, h_quant, weights, h_dequant.clone(), num_blocks, 256);

    idct_16x16::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    // For block_l2, treat each 16x16 region as 4 8x8 blocks then sum the
    // four costs back per-region. Simpler: re-use block_l2's per-8x8 layout
    // and let the demo aggregate. The cleanest map is: 16x16 region cost =
    // sum of 4 underlying 8x8 block_l2 costs in that region.
    //
    // For this prototype, return per-16x16-block cost approximated as:
    //   sum_over_pixels (orig - recon)^2 in each 16x16 block.
    // The block_l2 launcher operates on per-8x8 blocks; we expose 4× more
    // blocks then aggregate on host. Keeping the prototype self-contained:
    let n_pixels = nb * 256;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 4]));

    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks_16 * 2, // each 16x16 = 2x2 8x8 blocks
        ysize_blocks_16 * 2,
        xsize_blocks_16 * 16,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_16,
        ysize_blocks: ysize_blocks_16,
    }
}

/// 3-channel DCT8 cost grid with proper XYB weighting.
///
/// Mirrors `compute_cost_grid_dct8_single_channel` but runs the full
/// pipeline once per channel and combines them via `block_l2` (which
/// internally weights X/Y/B by libjxl's perceptual constants and
/// multiplies by the per-pixel `mask1x1` field).
///
/// Each channel gets its own block-major input + per-block quant
/// weights + per-block `qac_qm` scalar. `mask1x1` is per-pixel
/// (length = xsize_blocks * ysize_blocks * 64).
///
/// This is the cost grid you want in production: matches what the
/// VarDCT encoder actually pays for each AC strategy (sum of channel
/// reconstruction errors, masked by the perceptual signal).
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct8_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks * ysize_blocks;
    let nb = num_blocks as usize;
    let n_coef = nb * 64;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    // Per-channel intermediate buffers.
    let h_dct_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_q_x = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_y = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_b = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dq_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_b = client.create_from_slice(f32::as_bytes(&zero_f));

    // Per-channel DCT.
    dct_8x8::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
    dct_8x8::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
    dct_8x8::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);

    // Per-channel quantize.
    quantize_dct8::<R>(
        client,
        h_dct_x.clone(),
        weights_x.clone(),
        qac_qm_x,
        thresholds_x,
        h_q_x.clone(),
        num_blocks,
    );
    quantize_dct8::<R>(
        client,
        h_dct_y.clone(),
        weights_y.clone(),
        qac_qm_y,
        thresholds_y,
        h_q_y.clone(),
        num_blocks,
    );
    quantize_dct8::<R>(
        client,
        h_dct_b.clone(),
        weights_b.clone(),
        qac_qm_b,
        thresholds_b,
        h_q_b.clone(),
        num_blocks,
    );

    // Per-channel dequant + IDCT.
    dequant_simple_dct8::<R>(client, h_q_x, weights_x, h_dq_x.clone(), num_blocks);
    dequant_simple_dct8::<R>(client, h_q_y, weights_y, h_dq_y.clone(), num_blocks);
    dequant_simple_dct8::<R>(client, h_q_b, weights_b, h_dq_b.clone(), num_blocks);
    idct_8x8::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
    idct_8x8::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
    idct_8x8::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);

    // 3-channel masked block L2.
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb]));
    block_l2::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        h_recon_x,
        h_recon_y,
        h_recon_b,
        mask1x1,
        h_costs.clone(),
        xsize_blocks,
        ysize_blocks,
        xsize_blocks * 8,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks,
        ysize_blocks,
    }
}

/// 3-channel DCT16×16 cost grid with XYB weighting + per-pixel mask.
///
/// Same shape as `compute_cost_grid_dct8_xyb` but for the DCT16×16
/// strategy. Returns one cost per 16×16 region. Each region is
/// internally evaluated as 4 8×8 sub-cells in `block_l2` (the
/// per-rect aggregate must be done by the caller, exactly as for
/// `compute_cost_grid_dct16x16_single_channel`).
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct16x16_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks_16: u32,
    ysize_blocks_16: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_16 * ysize_blocks_16;
    let nb = num_blocks as usize;
    let n_coef = nb * 256;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_q_x = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_y = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_b = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dq_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_b = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_16x16::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
    dct_16x16::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
    dct_16x16::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);

    let qlarge = |dct: Handle, w: Handle, qac: Handle, thr: Handle, q: Handle| {
        quantize_large::<R>(client, dct, w, qac, thr, q, num_blocks, 16, 16, 2, 2);
    };
    qlarge(
        h_dct_x.clone(),
        weights_x.clone(),
        qac_qm_x,
        thresholds_x,
        h_q_x.clone(),
    );
    qlarge(
        h_dct_y.clone(),
        weights_y.clone(),
        qac_qm_y,
        thresholds_y,
        h_q_y.clone(),
    );
    qlarge(
        h_dct_b.clone(),
        weights_b.clone(),
        qac_qm_b,
        thresholds_b,
        h_q_b.clone(),
    );

    dequant_simple_generic::<R>(client, h_q_x, weights_x, h_dq_x.clone(), num_blocks, 256);
    dequant_simple_generic::<R>(client, h_q_y, weights_y, h_dq_y.clone(), num_blocks, 256);
    dequant_simple_generic::<R>(client, h_q_b, weights_b, h_dq_b.clone(), num_blocks, 256);
    idct_16x16::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
    idct_16x16::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
    idct_16x16::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);

    // 3-channel masked block_l2 returns per-8x8 sub-cell costs; aggregate
    // 4 per 16x16 region on host.
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 4]));
    block_l2::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        h_recon_x,
        h_recon_y,
        h_recon_b,
        mask1x1,
        h_costs.clone(),
        xsize_blocks_16 * 2,
        ysize_blocks_16 * 2,
        xsize_blocks_16 * 16,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_16,
        ysize_blocks: ysize_blocks_16,
    }
}

/// 3-channel DCT32×32 cost grid with XYB weighting + per-pixel mask.
///
/// Shape mirrors `compute_cost_grid_dct16x16_xyb` scaled to 32×32:
/// each region = 1024 floats per channel, LLF region 4×4, sub-cells
/// 4×4 = 16 per region. block_l2 returns per-8×8 sub-cell costs;
/// caller aggregates 16 to get the per-region total.
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct32x32_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks_32: u32,
    ysize_blocks_32: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_32 * ysize_blocks_32;
    let nb = num_blocks as usize;
    let n_coef = nb * 1024;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_q_x = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_y = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_b = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dq_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_b = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_32x32::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
    dct_32x32::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
    dct_32x32::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);

    let qlarge = |dct: Handle, w: Handle, qac: Handle, thr: Handle, q: Handle| {
        quantize_large::<R>(client, dct, w, qac, thr, q, num_blocks, 32, 32, 4, 4);
    };
    qlarge(
        h_dct_x.clone(),
        weights_x.clone(),
        qac_qm_x,
        thresholds_x,
        h_q_x.clone(),
    );
    qlarge(
        h_dct_y.clone(),
        weights_y.clone(),
        qac_qm_y,
        thresholds_y,
        h_q_y.clone(),
    );
    qlarge(
        h_dct_b.clone(),
        weights_b.clone(),
        qac_qm_b,
        thresholds_b,
        h_q_b.clone(),
    );

    dequant_simple_generic::<R>(client, h_q_x, weights_x, h_dq_x.clone(), num_blocks, 1024);
    dequant_simple_generic::<R>(client, h_q_y, weights_y, h_dq_y.clone(), num_blocks, 1024);
    dequant_simple_generic::<R>(client, h_q_b, weights_b, h_dq_b.clone(), num_blocks, 1024);
    idct_32x32::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
    idct_32x32::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
    idct_32x32::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);

    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 16]));
    block_l2::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        h_recon_x,
        h_recon_y,
        h_recon_b,
        mask1x1,
        h_costs.clone(),
        xsize_blocks_32 * 4,
        ysize_blocks_32 * 4,
        xsize_blocks_32 * 32,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_32,
        ysize_blocks: ysize_blocks_32,
    }
}

/// 3-channel DCT64×64 cost grid with XYB weighting + per-pixel mask.
///
/// Shape mirrors `compute_cost_grid_dct32x32_xyb` scaled to 64×64:
/// each region = 4096 floats per channel, LLF region 8×8, sub-cells
/// 8×8 = 64 per region. block_l2 returns per-8×8 sub-cell costs;
/// caller aggregates 64 to get the per-region total.
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct64x64_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks_64: u32,
    ysize_blocks_64: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_64 * ysize_blocks_64;
    let nb = num_blocks as usize;
    let n_coef = nb * 4096;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_q_x = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_y = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_b = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dq_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_b = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_64x64::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
    dct_64x64::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
    dct_64x64::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);

    let qlarge = |dct: Handle, w: Handle, qac: Handle, thr: Handle, q: Handle| {
        quantize_large::<R>(client, dct, w, qac, thr, q, num_blocks, 64, 64, 8, 8);
    };
    qlarge(
        h_dct_x.clone(),
        weights_x.clone(),
        qac_qm_x,
        thresholds_x,
        h_q_x.clone(),
    );
    qlarge(
        h_dct_y.clone(),
        weights_y.clone(),
        qac_qm_y,
        thresholds_y,
        h_q_y.clone(),
    );
    qlarge(
        h_dct_b.clone(),
        weights_b.clone(),
        qac_qm_b,
        thresholds_b,
        h_q_b.clone(),
    );

    dequant_simple_generic::<R>(client, h_q_x, weights_x, h_dq_x.clone(), num_blocks, 4096);
    dequant_simple_generic::<R>(client, h_q_y, weights_y, h_dq_y.clone(), num_blocks, 4096);
    dequant_simple_generic::<R>(client, h_q_b, weights_b, h_dq_b.clone(), num_blocks, 4096);
    idct_64x64::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
    idct_64x64::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
    idct_64x64::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);

    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 64]));
    block_l2::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        h_recon_x,
        h_recon_y,
        h_recon_b,
        mask1x1,
        h_costs.clone(),
        xsize_blocks_64 * 8,
        ysize_blocks_64 * 8,
        xsize_blocks_64 * 64,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_64,
        ysize_blocks: ysize_blocks_64,
    }
}

// =============================================================================
// 3-channel cost grids — rectangular DCT16/32/64 family
// =============================================================================
//
// All six functions share the same shape as their square 3-channel
// counterparts, differing only in DCT/IDCT kernel + quantize_large grid
// + sub-cell layout. Each one runs the per-channel pipeline (DCT →
// quantize → dequant → IDCT) three times then calls block_l2 with all
// six channel handles + per-pixel mask. block_l2 returns per-8×8
// sub-cell costs; caller aggregates per rect block.

/// 3-channel DCT16×8 cost grid (16-tall × 8-wide rect, 128 floats per
/// rect block, 2×1 = 2 sub-cells per rect).
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct16x8_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks_16x8: u32,
    ysize_blocks_16x8: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_16x8 * ysize_blocks_16x8;
    let nb = num_blocks as usize;
    let n_coef = nb * 128;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];
    let h_dct_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_q_x = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_y = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_b = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dq_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_b = client.create_from_slice(f32::as_bytes(&zero_f));
    dct_16x8::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
    dct_16x8::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
    dct_16x8::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);
    let qlarge = |dct: Handle, w: Handle, qac: Handle, thr: Handle, q: Handle| {
        quantize_large::<R>(client, dct, w, qac, thr, q, num_blocks, 16, 8, 2, 1);
    };
    qlarge(
        h_dct_x.clone(),
        weights_x.clone(),
        qac_qm_x,
        thresholds_x,
        h_q_x.clone(),
    );
    qlarge(
        h_dct_y.clone(),
        weights_y.clone(),
        qac_qm_y,
        thresholds_y,
        h_q_y.clone(),
    );
    qlarge(
        h_dct_b.clone(),
        weights_b.clone(),
        qac_qm_b,
        thresholds_b,
        h_q_b.clone(),
    );
    dequant_simple_generic::<R>(client, h_q_x, weights_x, h_dq_x.clone(), num_blocks, 128);
    dequant_simple_generic::<R>(client, h_q_y, weights_y, h_dq_y.clone(), num_blocks, 128);
    dequant_simple_generic::<R>(client, h_q_b, weights_b, h_dq_b.clone(), num_blocks, 128);
    idct_16x8::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
    idct_16x8::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
    idct_16x8::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 2]));
    block_l2::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        h_recon_x,
        h_recon_y,
        h_recon_b,
        mask1x1,
        h_costs.clone(),
        xsize_blocks_16x8 * 2,
        ysize_blocks_16x8,
        xsize_blocks_16x8 * 16,
    );
    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_16x8,
        ysize_blocks: ysize_blocks_16x8,
    }
}

/// 3-channel DCT8×16 cost grid (8-tall × 16-wide rect, 128 floats per
/// rect block, 1×2 = 2 sub-cells per rect).
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct8x16_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks_8x16: u32,
    ysize_blocks_8x16: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_8x16 * ysize_blocks_8x16;
    let nb = num_blocks as usize;
    let n_coef = nb * 128;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];
    let h_dct_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_q_x = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_y = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_b = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dq_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_b = client.create_from_slice(f32::as_bytes(&zero_f));
    dct_8x16::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
    dct_8x16::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
    dct_8x16::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);
    let qlarge = |dct: Handle, w: Handle, qac: Handle, thr: Handle, q: Handle| {
        quantize_large::<R>(client, dct, w, qac, thr, q, num_blocks, 8, 16, 1, 2);
    };
    qlarge(
        h_dct_x.clone(),
        weights_x.clone(),
        qac_qm_x,
        thresholds_x,
        h_q_x.clone(),
    );
    qlarge(
        h_dct_y.clone(),
        weights_y.clone(),
        qac_qm_y,
        thresholds_y,
        h_q_y.clone(),
    );
    qlarge(
        h_dct_b.clone(),
        weights_b.clone(),
        qac_qm_b,
        thresholds_b,
        h_q_b.clone(),
    );
    dequant_simple_generic::<R>(client, h_q_x, weights_x, h_dq_x.clone(), num_blocks, 128);
    dequant_simple_generic::<R>(client, h_q_y, weights_y, h_dq_y.clone(), num_blocks, 128);
    dequant_simple_generic::<R>(client, h_q_b, weights_b, h_dq_b.clone(), num_blocks, 128);
    idct_8x16::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
    idct_8x16::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
    idct_8x16::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 2]));
    block_l2::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        h_recon_x,
        h_recon_y,
        h_recon_b,
        mask1x1,
        h_costs.clone(),
        xsize_blocks_8x16,
        ysize_blocks_8x16 * 2,
        xsize_blocks_8x16 * 8,
    );
    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_8x16,
        ysize_blocks: ysize_blocks_8x16,
    }
}

/// 3-channel DCT32×16 cost grid (32-tall × 16-wide rect, 512 floats
/// per rect, 4×2 = 8 sub-cells per rect).
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct32x16_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks_32x16: u32,
    ysize_blocks_32x16: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_32x16 * ysize_blocks_32x16;
    let nb = num_blocks as usize;
    let n_coef = nb * 512;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];
    let h_dct_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_q_x = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_y = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_b = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dq_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_b = client.create_from_slice(f32::as_bytes(&zero_f));
    dct_32x16::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
    dct_32x16::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
    dct_32x16::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);
    let qlarge = |dct: Handle, w: Handle, qac: Handle, thr: Handle, q: Handle| {
        quantize_large::<R>(client, dct, w, qac, thr, q, num_blocks, 32, 16, 4, 2);
    };
    qlarge(
        h_dct_x.clone(),
        weights_x.clone(),
        qac_qm_x,
        thresholds_x,
        h_q_x.clone(),
    );
    qlarge(
        h_dct_y.clone(),
        weights_y.clone(),
        qac_qm_y,
        thresholds_y,
        h_q_y.clone(),
    );
    qlarge(
        h_dct_b.clone(),
        weights_b.clone(),
        qac_qm_b,
        thresholds_b,
        h_q_b.clone(),
    );
    dequant_simple_generic::<R>(client, h_q_x, weights_x, h_dq_x.clone(), num_blocks, 512);
    dequant_simple_generic::<R>(client, h_q_y, weights_y, h_dq_y.clone(), num_blocks, 512);
    dequant_simple_generic::<R>(client, h_q_b, weights_b, h_dq_b.clone(), num_blocks, 512);
    idct_32x16::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
    idct_32x16::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
    idct_32x16::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 8]));
    block_l2::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        h_recon_x,
        h_recon_y,
        h_recon_b,
        mask1x1,
        h_costs.clone(),
        xsize_blocks_32x16 * 4,
        ysize_blocks_32x16 * 2,
        xsize_blocks_32x16 * 32,
    );
    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_32x16,
        ysize_blocks: ysize_blocks_32x16,
    }
}

/// 3-channel DCT16×32 cost grid (16-tall × 32-wide rect, 512 floats
/// per rect, 2×4 = 8 sub-cells per rect).
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct16x32_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks_16x32: u32,
    ysize_blocks_16x32: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_16x32 * ysize_blocks_16x32;
    let nb = num_blocks as usize;
    let n_coef = nb * 512;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];
    let h_dct_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_q_x = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_y = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_b = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dq_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_b = client.create_from_slice(f32::as_bytes(&zero_f));
    dct_16x32::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
    dct_16x32::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
    dct_16x32::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);
    let qlarge = |dct: Handle, w: Handle, qac: Handle, thr: Handle, q: Handle| {
        quantize_large::<R>(client, dct, w, qac, thr, q, num_blocks, 16, 32, 2, 4);
    };
    qlarge(
        h_dct_x.clone(),
        weights_x.clone(),
        qac_qm_x,
        thresholds_x,
        h_q_x.clone(),
    );
    qlarge(
        h_dct_y.clone(),
        weights_y.clone(),
        qac_qm_y,
        thresholds_y,
        h_q_y.clone(),
    );
    qlarge(
        h_dct_b.clone(),
        weights_b.clone(),
        qac_qm_b,
        thresholds_b,
        h_q_b.clone(),
    );
    dequant_simple_generic::<R>(client, h_q_x, weights_x, h_dq_x.clone(), num_blocks, 512);
    dequant_simple_generic::<R>(client, h_q_y, weights_y, h_dq_y.clone(), num_blocks, 512);
    dequant_simple_generic::<R>(client, h_q_b, weights_b, h_dq_b.clone(), num_blocks, 512);
    idct_16x32::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
    idct_16x32::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
    idct_16x32::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 8]));
    block_l2::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        h_recon_x,
        h_recon_y,
        h_recon_b,
        mask1x1,
        h_costs.clone(),
        xsize_blocks_16x32 * 2,
        ysize_blocks_16x32 * 4,
        xsize_blocks_16x32 * 16,
    );
    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_16x32,
        ysize_blocks: ysize_blocks_16x32,
    }
}

/// 3-channel DCT64×32 cost grid (64-tall × 32-wide rect, 2048 floats
/// per rect, 8×4 = 32 sub-cells per rect).
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct64x32_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks_64x32: u32,
    ysize_blocks_64x32: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_64x32 * ysize_blocks_64x32;
    let nb = num_blocks as usize;
    let n_coef = nb * 2048;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];
    let h_dct_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_q_x = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_y = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_b = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dq_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_b = client.create_from_slice(f32::as_bytes(&zero_f));
    dct_64x32::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
    dct_64x32::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
    dct_64x32::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);
    let qlarge = |dct: Handle, w: Handle, qac: Handle, thr: Handle, q: Handle| {
        quantize_large::<R>(client, dct, w, qac, thr, q, num_blocks, 64, 32, 8, 4);
    };
    qlarge(
        h_dct_x.clone(),
        weights_x.clone(),
        qac_qm_x,
        thresholds_x,
        h_q_x.clone(),
    );
    qlarge(
        h_dct_y.clone(),
        weights_y.clone(),
        qac_qm_y,
        thresholds_y,
        h_q_y.clone(),
    );
    qlarge(
        h_dct_b.clone(),
        weights_b.clone(),
        qac_qm_b,
        thresholds_b,
        h_q_b.clone(),
    );
    dequant_simple_generic::<R>(client, h_q_x, weights_x, h_dq_x.clone(), num_blocks, 2048);
    dequant_simple_generic::<R>(client, h_q_y, weights_y, h_dq_y.clone(), num_blocks, 2048);
    dequant_simple_generic::<R>(client, h_q_b, weights_b, h_dq_b.clone(), num_blocks, 2048);
    idct_64x32::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
    idct_64x32::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
    idct_64x32::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 32]));
    block_l2::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        h_recon_x,
        h_recon_y,
        h_recon_b,
        mask1x1,
        h_costs.clone(),
        xsize_blocks_64x32 * 8,
        ysize_blocks_64x32 * 4,
        xsize_blocks_64x32 * 64,
    );
    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_64x32,
        ysize_blocks: ysize_blocks_64x32,
    }
}

/// 3-channel DCT32×64 cost grid (32-tall × 64-wide rect, 2048 floats
/// per rect, 4×8 = 32 sub-cells per rect).
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct32x64_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks_32x64: u32,
    ysize_blocks_32x64: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_32x64 * ysize_blocks_32x64;
    let nb = num_blocks as usize;
    let n_coef = nb * 2048;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];
    let h_dct_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_q_x = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_y = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_b = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dq_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_b = client.create_from_slice(f32::as_bytes(&zero_f));
    dct_32x64::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
    dct_32x64::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
    dct_32x64::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);
    let qlarge = |dct: Handle, w: Handle, qac: Handle, thr: Handle, q: Handle| {
        quantize_large::<R>(client, dct, w, qac, thr, q, num_blocks, 32, 64, 4, 8);
    };
    qlarge(
        h_dct_x.clone(),
        weights_x.clone(),
        qac_qm_x,
        thresholds_x,
        h_q_x.clone(),
    );
    qlarge(
        h_dct_y.clone(),
        weights_y.clone(),
        qac_qm_y,
        thresholds_y,
        h_q_y.clone(),
    );
    qlarge(
        h_dct_b.clone(),
        weights_b.clone(),
        qac_qm_b,
        thresholds_b,
        h_q_b.clone(),
    );
    dequant_simple_generic::<R>(client, h_q_x, weights_x, h_dq_x.clone(), num_blocks, 2048);
    dequant_simple_generic::<R>(client, h_q_y, weights_y, h_dq_y.clone(), num_blocks, 2048);
    dequant_simple_generic::<R>(client, h_q_b, weights_b, h_dq_b.clone(), num_blocks, 2048);
    idct_32x64::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
    idct_32x64::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
    idct_32x64::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 32]));
    block_l2::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        h_recon_x,
        h_recon_y,
        h_recon_b,
        mask1x1,
        h_costs.clone(),
        xsize_blocks_32x64 * 4,
        ysize_blocks_32x64 * 8,
        xsize_blocks_32x64 * 32,
    );
    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_32x64,
        ysize_blocks: ysize_blocks_32x64,
    }
}

// =============================================================================
// 3-channel cost grids — DCT4 family (sub-block transforms, 64-coeff layout)
// =============================================================================
//
// Per the single-channel variants: DCT4x4/4x8/8x4 all operate on 64-
// coeff blocks, so they reuse quantize_dct8 + dequant_simple_dct8
// rather than quantize_large. Each 8×8 block is independent → one
// per-region cost output.

/// 3-channel DCT4×4 cost grid (sub-block transform on 8×8 layout).
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct4x4_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> CostGrid {
    cost_grid_dct4_xyb_impl::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        weights_x,
        weights_y,
        weights_b,
        qac_qm_x,
        qac_qm_y,
        qac_qm_b,
        thresholds_x,
        thresholds_y,
        thresholds_b,
        mask1x1,
        xsize_blocks,
        ysize_blocks,
        Dct4Variant::Dct4x4,
    )
}

/// 3-channel DCT4×8 cost grid.
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct4x8_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> CostGrid {
    cost_grid_dct4_xyb_impl::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        weights_x,
        weights_y,
        weights_b,
        qac_qm_x,
        qac_qm_y,
        qac_qm_b,
        thresholds_x,
        thresholds_y,
        thresholds_b,
        mask1x1,
        xsize_blocks,
        ysize_blocks,
        Dct4Variant::Dct4x8,
    )
}

/// 3-channel DCT8×4 cost grid.
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct8x4_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> CostGrid {
    cost_grid_dct4_xyb_impl::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        weights_x,
        weights_y,
        weights_b,
        qac_qm_x,
        qac_qm_y,
        qac_qm_b,
        thresholds_x,
        thresholds_y,
        thresholds_b,
        mask1x1,
        xsize_blocks,
        ysize_blocks,
        Dct4Variant::Dct8x4,
    )
}

#[derive(Clone, Copy)]
enum Dct4Variant {
    Dct4x4,
    Dct4x8,
    Dct8x4,
}

#[allow(clippy::too_many_arguments)]
fn cost_grid_dct4_xyb_impl<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
    variant: Dct4Variant,
) -> CostGrid {
    let num_blocks = xsize_blocks * ysize_blocks;
    let nb = num_blocks as usize;
    let n_coef = nb * 64;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];
    let h_dct_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_q_x = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_y = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_b = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dq_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_b = client.create_from_slice(f32::as_bytes(&zero_f));

    match variant {
        Dct4Variant::Dct4x4 => {
            dct_4x4_full::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
            dct_4x4_full::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
            dct_4x4_full::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);
        }
        Dct4Variant::Dct4x8 => {
            dct_4x8_full::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
            dct_4x8_full::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
            dct_4x8_full::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);
        }
        Dct4Variant::Dct8x4 => {
            dct_8x4_full::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
            dct_8x4_full::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
            dct_8x4_full::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);
        }
    }

    let q = |dct: Handle, w: Handle, qac: Handle, thr: Handle, qh: Handle| {
        quantize_dct8::<R>(client, dct, w, qac, thr, qh, num_blocks);
    };
    q(
        h_dct_x.clone(),
        weights_x.clone(),
        qac_qm_x,
        thresholds_x,
        h_q_x.clone(),
    );
    q(
        h_dct_y.clone(),
        weights_y.clone(),
        qac_qm_y,
        thresholds_y,
        h_q_y.clone(),
    );
    q(
        h_dct_b.clone(),
        weights_b.clone(),
        qac_qm_b,
        thresholds_b,
        h_q_b.clone(),
    );

    dequant_simple_dct8::<R>(client, h_q_x, weights_x, h_dq_x.clone(), num_blocks);
    dequant_simple_dct8::<R>(client, h_q_y, weights_y, h_dq_y.clone(), num_blocks);
    dequant_simple_dct8::<R>(client, h_q_b, weights_b, h_dq_b.clone(), num_blocks);
    match variant {
        Dct4Variant::Dct4x4 => {
            idct_4x4_full::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
            idct_4x4_full::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
            idct_4x4_full::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);
        }
        Dct4Variant::Dct4x8 => {
            idct_4x8_full::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
            idct_4x8_full::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
            idct_4x8_full::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);
        }
        Dct4Variant::Dct8x4 => {
            idct_8x4_full::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
            idct_8x4_full::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
            idct_8x4_full::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);
        }
    }

    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb]));
    block_l2::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        h_recon_x,
        h_recon_y,
        h_recon_b,
        mask1x1,
        h_costs.clone(),
        xsize_blocks,
        ysize_blocks,
        xsize_blocks * 8,
    );
    CostGrid {
        costs: h_costs,
        xsize_blocks,
        ysize_blocks,
    }
}

/// 3-channel IDENTITY cost grid. 64-coeff layout, reuses
/// quantize_dct8 + dequant_simple_dct8 + 3-channel masked block_l2.
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_identity_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> CostGrid {
    cost_grid_64coef_xyb_impl::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        weights_x,
        weights_y,
        weights_b,
        qac_qm_x,
        qac_qm_y,
        qac_qm_b,
        thresholds_x,
        thresholds_y,
        thresholds_b,
        mask1x1,
        xsize_blocks,
        ysize_blocks,
        Coef64Variant::Identity,
    )
}

/// 3-channel DCT2X2 cost grid. 64-coeff layout, hierarchical 2×2
/// Hadamard at scales 8/4/2 (forward) and 2/4/8 (inverse).
#[allow(clippy::too_many_arguments)]
pub fn compute_cost_grid_dct2x2_xyb<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> CostGrid {
    cost_grid_64coef_xyb_impl::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        weights_x,
        weights_y,
        weights_b,
        qac_qm_x,
        qac_qm_y,
        qac_qm_b,
        thresholds_x,
        thresholds_y,
        thresholds_b,
        mask1x1,
        xsize_blocks,
        ysize_blocks,
        Coef64Variant::Dct2x2,
    )
}

#[derive(Clone, Copy)]
enum Coef64Variant {
    Identity,
    Dct2x2,
}

#[allow(clippy::too_many_arguments)]
fn cost_grid_64coef_xyb_impl<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    mask1x1: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
    variant: Coef64Variant,
) -> CostGrid {
    let num_blocks = xsize_blocks * ysize_blocks;
    let nb = num_blocks as usize;
    let n_coef = nb * 64;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];
    let h_dct_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dct_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_q_x = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_y = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_q_b = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dq_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_dq_b = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_x = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_y = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon_b = client.create_from_slice(f32::as_bytes(&zero_f));

    match variant {
        Coef64Variant::Identity => {
            identity_forward::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
            identity_forward::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
            identity_forward::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);
        }
        Coef64Variant::Dct2x2 => {
            dct2x2_forward::<R>(client, orig_x.clone(), h_dct_x.clone(), num_blocks);
            dct2x2_forward::<R>(client, orig_y.clone(), h_dct_y.clone(), num_blocks);
            dct2x2_forward::<R>(client, orig_b.clone(), h_dct_b.clone(), num_blocks);
        }
    }

    let q = |dct: Handle, w: Handle, qac: Handle, thr: Handle, qh: Handle| {
        quantize_dct8::<R>(client, dct, w, qac, thr, qh, num_blocks);
    };
    q(
        h_dct_x.clone(),
        weights_x.clone(),
        qac_qm_x,
        thresholds_x,
        h_q_x.clone(),
    );
    q(
        h_dct_y.clone(),
        weights_y.clone(),
        qac_qm_y,
        thresholds_y,
        h_q_y.clone(),
    );
    q(
        h_dct_b.clone(),
        weights_b.clone(),
        qac_qm_b,
        thresholds_b,
        h_q_b.clone(),
    );

    dequant_simple_dct8::<R>(client, h_q_x, weights_x, h_dq_x.clone(), num_blocks);
    dequant_simple_dct8::<R>(client, h_q_y, weights_y, h_dq_y.clone(), num_blocks);
    dequant_simple_dct8::<R>(client, h_q_b, weights_b, h_dq_b.clone(), num_blocks);

    match variant {
        Coef64Variant::Identity => {
            identity_inverse::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
            identity_inverse::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
            identity_inverse::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);
        }
        Coef64Variant::Dct2x2 => {
            dct2x2_inverse::<R>(client, h_dq_x, h_recon_x.clone(), num_blocks);
            dct2x2_inverse::<R>(client, h_dq_y, h_recon_y.clone(), num_blocks);
            dct2x2_inverse::<R>(client, h_dq_b, h_recon_b.clone(), num_blocks);
        }
    }

    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb]));
    block_l2::<R>(
        client,
        orig_x,
        orig_y,
        orig_b,
        h_recon_x,
        h_recon_y,
        h_recon_b,
        mask1x1,
        h_costs.clone(),
        xsize_blocks,
        ysize_blocks,
        xsize_blocks * 8,
    );
    CostGrid {
        costs: h_costs,
        xsize_blocks,
        ysize_blocks,
    }
}

/// IDENTITY single-channel cost grid. 64-coeff layout (8×8 sub-block
/// structure), reuses quantize_dct8 + dequant_simple_dct8.
pub fn compute_cost_grid_identity_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks * ysize_blocks;
    let nb = num_blocks as usize;
    let n_coef = nb * 64;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    identity_forward::<R>(client, original.clone(), h_dct.clone(), num_blocks);
    quantize_dct8::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
    );
    dequant_simple_dct8::<R>(client, h_quant, weights, h_dequant.clone(), num_blocks);
    identity_inverse::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    let n_pixels = nb * 64;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb]));
    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks,
        ysize_blocks,
        xsize_blocks * 8,
    );
    CostGrid {
        costs: h_costs,
        xsize_blocks,
        ysize_blocks,
    }
}

/// DCT2X2 single-channel cost grid. 64-coeff layout, hierarchical
/// 2×2 Hadamard at scales 8/4/2 (forward) and 2/4/8 (inverse).
pub fn compute_cost_grid_dct2x2_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks * ysize_blocks;
    let nb = num_blocks as usize;
    let n_coef = nb * 64;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct2x2_forward::<R>(client, original.clone(), h_dct.clone(), num_blocks);
    quantize_dct8::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
    );
    dequant_simple_dct8::<R>(client, h_quant, weights, h_dequant.clone(), num_blocks);
    dct2x2_inverse::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    let n_pixels = nb * 64;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb]));
    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks,
        ysize_blocks,
        xsize_blocks * 8,
    );
    CostGrid {
        costs: h_costs,
        xsize_blocks,
        ysize_blocks,
    }
}

/// Compute the DCT4×4 strategy's whole-image cost grid for a single channel.
///
/// DCT4×4 is a sub-block transform: each 8×8 block is treated as 4
/// independent 4×4 sub-blocks (in a 2×2 grid). All sub-blocks share
/// the 8×8 quant layout (64 coeffs), so we reuse `quantize_dct8` and
/// `dequant_simple_dct8` rather than `quantize_large`.
///
/// Each cost-grid entry corresponds to one 8×8 block. Pipeline:
/// `dct_4x4_full → quantize_dct8 → dequant_simple_dct8 → idct_4x4_full
/// → block_l2 (per-8×8)`.
pub fn compute_cost_grid_dct4x4_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks * ysize_blocks;
    let nb = num_blocks as usize;
    let n_coef = nb * 64;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_4x4_full::<R>(client, original.clone(), h_dct.clone(), num_blocks);
    quantize_dct8::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
    );
    dequant_simple_dct8::<R>(client, h_quant, weights, h_dequant.clone(), num_blocks);
    idct_4x4_full::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    let n_pixels = nb * 64;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb]));

    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks,
        ysize_blocks,
        xsize_blocks * 8,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks,
        ysize_blocks,
    }
}

/// Compute the DCT4×8 strategy's whole-image cost grid for a single channel.
/// Same as `compute_cost_grid_dct4x4_single_channel` with the 4×8
/// transform instead of 4×4. 64-coeff layout, reuses dct8 quant path.
pub fn compute_cost_grid_dct4x8_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks * ysize_blocks;
    let nb = num_blocks as usize;
    let n_coef = nb * 64;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_4x8_full::<R>(client, original.clone(), h_dct.clone(), num_blocks);
    quantize_dct8::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
    );
    dequant_simple_dct8::<R>(client, h_quant, weights, h_dequant.clone(), num_blocks);
    idct_4x8_full::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    let n_pixels = nb * 64;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb]));

    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks,
        ysize_blocks,
        xsize_blocks * 8,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks,
        ysize_blocks,
    }
}

/// Compute the DCT8×4 strategy's whole-image cost grid for a single channel.
/// Counterpart to `compute_cost_grid_dct4x8_*` with the rectangle transposed.
pub fn compute_cost_grid_dct8x4_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks * ysize_blocks;
    let nb = num_blocks as usize;
    let n_coef = nb * 64;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_8x4_full::<R>(client, original.clone(), h_dct.clone(), num_blocks);
    quantize_dct8::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
    );
    dequant_simple_dct8::<R>(client, h_quant, weights, h_dequant.clone(), num_blocks);
    idct_8x4_full::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    let n_pixels = nb * 64;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb]));

    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks,
        ysize_blocks,
        xsize_blocks * 8,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks,
        ysize_blocks,
    }
}

/// Compute the DCT16×8 strategy's whole-image cost grid for a single channel.
///
/// Each block covers a 16×8 pixel region (= 2×1 grid of 8×8 blocks).
/// LLF region is 2×1. Pipeline:
/// `dct_16x8 → quantize_large(grid_w=16, grid_h=8, llf_x=2, llf_y=1)
/// → dequant_simple_generic(coef=128) → idct_16x8 → block_l2 (16×8 region)`.
///
/// Block ordering: row-major, `block_y * xsize_blocks_16x8 + block_x`,
/// where `xsize_blocks_16x8 = ceil(image_w / 16)` and
/// `ysize_blocks_16x8 = ceil(image_h / 8)`.
pub fn compute_cost_grid_dct16x8_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks_16x8: u32,
    ysize_blocks_16x8: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_16x8 * ysize_blocks_16x8;
    let nb = num_blocks as usize;
    let n_coef = nb * 128;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_16x8::<R>(client, original.clone(), h_dct.clone(), num_blocks);

    quantize_large::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
        16, // grid_width
        8,  // grid_height
        2,  // llf_x
        1,  // llf_y
    );

    dequant_simple_generic::<R>(client, h_quant, weights, h_dequant.clone(), num_blocks, 128);

    idct_16x8::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    // Each 16×8 region = 2×1 grid of 8×8 blocks. block_l2 returns
    // per-8×8 costs; the demo aggregates 2 adjacent costs to get the
    // 16×8 cost. We expose the 2× block grid here.
    let n_pixels = nb * 128;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 2]));

    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks_16x8 * 2, // each 16×8 = 2×1 8×8 blocks
        ysize_blocks_16x8,
        xsize_blocks_16x8 * 16,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_16x8,
        ysize_blocks: ysize_blocks_16x8,
    }
}

/// Compute the DCT8×16 strategy's whole-image cost grid for a single channel.
///
/// Each block covers an 8×16 pixel region (= 1×2 grid of 8×8 blocks).
/// LLF region is 1×2. Pipeline mirrors `compute_cost_grid_dct16x8_*`
/// with the rectangle transposed.
pub fn compute_cost_grid_dct8x16_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks_8x16: u32,
    ysize_blocks_8x16: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_8x16 * ysize_blocks_8x16;
    let nb = num_blocks as usize;
    let n_coef = nb * 128;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_8x16::<R>(client, original.clone(), h_dct.clone(), num_blocks);

    quantize_large::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
        8,  // grid_width
        16, // grid_height
        1,  // llf_x
        2,  // llf_y
    );

    dequant_simple_generic::<R>(client, h_quant, weights, h_dequant.clone(), num_blocks, 128);

    idct_8x16::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    let n_pixels = nb * 128;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 2]));

    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks_8x16, // each 8×16 = 1×2 8×8 blocks
        ysize_blocks_8x16 * 2,
        xsize_blocks_8x16 * 8,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_8x16,
        ysize_blocks: ysize_blocks_8x16,
    }
}

/// Compute the DCT32×16 strategy's whole-image cost grid for a single channel.
///
/// Each block covers a 32×16 pixel region (= 4×2 grid of 8×8 blocks).
/// LLF region is 4×2. Pipeline mirrors the square DCT32x32 grid with the
/// rectangle dimensions; sub-cell count per rect block is 4×2 = 8.
pub fn compute_cost_grid_dct32x16_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks_32x16: u32,
    ysize_blocks_32x16: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_32x16 * ysize_blocks_32x16;
    let nb = num_blocks as usize;
    let n_coef = nb * 512;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_32x16::<R>(client, original.clone(), h_dct.clone(), num_blocks);
    quantize_large::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
        32, // grid_width
        16, // grid_height
        4,  // llf_x
        2,  // llf_y
    );
    dequant_simple_generic::<R>(client, h_quant, weights, h_dequant.clone(), num_blocks, 512);
    idct_32x16::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    let n_pixels = nb * 512;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 8]));

    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks_32x16 * 4, // each 32×16 = 4×2 8×8 blocks
        ysize_blocks_32x16 * 2,
        xsize_blocks_32x16 * 32,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_32x16,
        ysize_blocks: ysize_blocks_32x16,
    }
}

/// Compute the DCT16×32 strategy's whole-image cost grid for a single channel.
/// Counterpart to `compute_cost_grid_dct32x16_*` with the rectangle transposed.
/// LLF region is 2×4. Sub-cell count per rect block is 2×4 = 8.
pub fn compute_cost_grid_dct16x32_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks_16x32: u32,
    ysize_blocks_16x32: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_16x32 * ysize_blocks_16x32;
    let nb = num_blocks as usize;
    let n_coef = nb * 512;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_16x32::<R>(client, original.clone(), h_dct.clone(), num_blocks);
    quantize_large::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
        16, // grid_width
        32, // grid_height
        2,  // llf_x
        4,  // llf_y
    );
    dequant_simple_generic::<R>(client, h_quant, weights, h_dequant.clone(), num_blocks, 512);
    idct_16x32::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    let n_pixels = nb * 512;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 8]));

    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks_16x32 * 2, // each 16×32 = 2×4 8×8 blocks
        ysize_blocks_16x32 * 4,
        xsize_blocks_16x32 * 16,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_16x32,
        ysize_blocks: ysize_blocks_16x32,
    }
}

/// Compute the DCT32×32 strategy's whole-image cost grid for a single channel.
/// Same pattern as DCT16x16. Each block = 1024 floats.
pub fn compute_cost_grid_dct32x32_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks_32: u32,
    ysize_blocks_32: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_32 * ysize_blocks_32;
    let nb = num_blocks as usize;
    let n_coef = nb * 1024;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_32x32::<R>(client, original.clone(), h_dct.clone(), num_blocks);
    quantize_large::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
        32,
        32,
        4,
        4,
    );
    dequant_simple_generic::<R>(
        client,
        h_quant,
        weights,
        h_dequant.clone(),
        num_blocks,
        1024,
    );
    idct_32x32::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    let n_pixels = nb * 1024;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 16]));

    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks_32 * 4,
        ysize_blocks_32 * 4,
        xsize_blocks_32 * 32,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_32,
        ysize_blocks: ysize_blocks_32,
    }
}

/// Compute the DCT64×32 strategy's whole-image cost grid for a single channel.
///
/// Each rect block = 64×32 = 2048 floats = 8×4 underlying 8×8 sub-cells.
/// LLF region is 8×4. Pipeline mirrors the DCT64x64 grid with rectangle dims.
pub fn compute_cost_grid_dct64x32_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks_64x32: u32,
    ysize_blocks_64x32: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_64x32 * ysize_blocks_64x32;
    let nb = num_blocks as usize;
    let n_coef = nb * 2048;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_64x32::<R>(client, original.clone(), h_dct.clone(), num_blocks);
    quantize_large::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
        64, // grid_width
        32, // grid_height
        8,  // llf_x
        4,  // llf_y
    );
    dequant_simple_generic::<R>(
        client,
        h_quant,
        weights,
        h_dequant.clone(),
        num_blocks,
        2048,
    );
    idct_64x32::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    let n_pixels = nb * 2048;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 32]));

    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks_64x32 * 8, // each 64×32 = 8×4 8×8 blocks
        ysize_blocks_64x32 * 4,
        xsize_blocks_64x32 * 64,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_64x32,
        ysize_blocks: ysize_blocks_64x32,
    }
}

/// Compute the DCT32×64 strategy's whole-image cost grid for a single channel.
/// Counterpart to `compute_cost_grid_dct64x32_*` with the rectangle transposed.
/// LLF region is 4×8. Sub-cell count per rect block is 4×8 = 32.
pub fn compute_cost_grid_dct32x64_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks_32x64: u32,
    ysize_blocks_32x64: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_32x64 * ysize_blocks_32x64;
    let nb = num_blocks as usize;
    let n_coef = nb * 2048;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_32x64::<R>(client, original.clone(), h_dct.clone(), num_blocks);
    quantize_large::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
        32, // grid_width
        64, // grid_height
        4,  // llf_x
        8,  // llf_y
    );
    dequant_simple_generic::<R>(
        client,
        h_quant,
        weights,
        h_dequant.clone(),
        num_blocks,
        2048,
    );
    idct_32x64::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    let n_pixels = nb * 2048;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 32]));

    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks_32x64 * 4, // each 32×64 = 4×8 8×8 blocks
        ysize_blocks_32x64 * 8,
        xsize_blocks_32x64 * 32,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_32x64,
        ysize_blocks: ysize_blocks_32x64,
    }
}

/// DCT64×64 cost grid. Each block = 4096 floats.
pub fn compute_cost_grid_dct64x64_single_channel<R: Runtime>(
    client: &ComputeClient<R>,
    original: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    xsize_blocks_64: u32,
    ysize_blocks_64: u32,
) -> CostGrid {
    let num_blocks = xsize_blocks_64 * ysize_blocks_64;
    let nb = num_blocks as usize;
    let n_coef = nb * 4096;
    let zero_f = vec![0.0f32; n_coef];
    let zero_i = vec![0i32; n_coef];

    let h_dct = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_quant = client.create_from_slice(i32::as_bytes(&zero_i));
    let h_dequant = client.create_from_slice(f32::as_bytes(&zero_f));
    let h_recon = client.create_from_slice(f32::as_bytes(&zero_f));

    dct_64x64::<R>(client, original.clone(), h_dct.clone(), num_blocks);
    quantize_large::<R>(
        client,
        h_dct.clone(),
        weights.clone(),
        qac_qm,
        thresholds,
        h_quant.clone(),
        num_blocks,
        64,
        64,
        8,
        8,
    );
    dequant_simple_generic::<R>(
        client,
        h_quant,
        weights,
        h_dequant.clone(),
        num_blocks,
        4096,
    );
    idct_64x64::<R>(client, h_dequant, h_recon.clone(), num_blocks);

    let n_pixels = nb * 4096;
    let h_mask = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n_pixels]));
    let h_costs = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nb * 64]));

    block_l2::<R>(
        client,
        original.clone(),
        original.clone(),
        original,
        h_recon.clone(),
        h_recon.clone(),
        h_recon,
        h_mask,
        h_costs.clone(),
        xsize_blocks_64 * 8,
        ysize_blocks_64 * 8,
        xsize_blocks_64 * 64,
    );

    CostGrid {
        costs: h_costs,
        xsize_blocks: xsize_blocks_64,
        ysize_blocks: ysize_blocks_64,
    }
}

fn dequant_simple_dct8<R: Runtime>(
    client: &ComputeClient<R>,
    quant: Handle,
    weights: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        dequant_simple_dct8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(quant, n),
            ArrayArg::from_raw_parts(weights, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}

// =============================================================================
// Phase 3 Component 2: Host-side partition selector (prototype)
// =============================================================================

/// Per-region strategy choice for a 16×16 region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition16x16 {
    /// One DCT16×16 covering the whole 16×16 region.
    Dct16x16,
    /// Two DCT16x8 side-by-side (each is 16 tall × 8 wide; two horizontally
    /// adjacent fill a 16x16 region).
    TwoDct16x8Horizontal,
    /// Two DCT8x16 stacked vertically (each is 8 tall × 16 wide; two
    /// stacked fill a 16x16 region).
    TwoDct8x16Vertical,
    /// Four DCT8×8 blocks (TL, TR, BL, BR). Use this when only the
    /// DCT8 8×8-tier strategy is being considered. For per-cell choice
    /// from the full 8×8-tier set (DCT8 / DCT4x4 / DCT4x8 / DCT8x4 /
    /// IDENTITY / DCT2X2), use [`Partition16x16::FourSubBlocks`].
    FourDct8x8,
    /// Four 8×8 sub-blocks, each independently picked from the full
    /// 8×8-tier strategy set. Index order matches `FourDct8x8`:
    /// `[TL, TR, BL, BR]`.
    FourSubBlocks([SubStrategy; 4]),
}

/// Per-cell 8×8-tier AC strategy choice, used inside
/// [`Partition16x16::FourSubBlocks`] to drive per-cell strategy
/// selection beyond the all-DCT8 default.
///
/// Mirrors the libjxl `RAW_STRATEGY_*` codes for the 64-coeff family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubStrategy {
    Dct8,
    Dct4x4,
    Dct4x8,
    Dct8x4,
    Identity,
    Dct2x2,
}

/// Optional per-cell 8×8-tier cost grids for use with
/// [`pick_subblock_strategies`] / [`Partition16x16::FourSubBlocks`].
///
/// Each grid is per-8×8-cell, same shape as the canonical DCT8 cost
/// grid: `(xsize_blocks_8, ysize_blocks_8)` row-major. Pass `None` to
/// drop a strategy from consideration (it'll be skipped in the picker).
#[derive(Default, Clone, Copy)]
pub struct SubBlockCostGrids<'a> {
    pub dct4x4: Option<&'a [f32]>,
    pub dct4x8: Option<&'a [f32]>,
    pub dct8x4: Option<&'a [f32]>,
    pub identity: Option<&'a [f32]>,
    pub dct2x2: Option<&'a [f32]>,
}

/// Pick the lowest-cost 8×8-tier strategy for each of the 4 cells in a
/// 16×16 region. Always considers DCT8; the optional grids are
/// considered if present. Returns `(total_cost, [TL, TR, BL, BR])` so
/// the caller can compare against the larger-transform partitions.
///
/// `(bx0, by0)` is the top-left 8×8-block coordinate of the 16×16
/// region (so the four cells are at offsets (0,0), (1,0), (0,1), (1,1)).
pub fn pick_subblock_strategies(
    cost_dct8: &[f32],
    extra: SubBlockCostGrids<'_>,
    xsize_blocks_8: usize,
    bx0: usize,
    by0: usize,
) -> (f32, [SubStrategy; 4]) {
    let cell_idx = |i: usize| {
        let dx = i % 2;
        let dy = i / 2;
        (by0 + dy) * xsize_blocks_8 + bx0 + dx
    };
    let pick_one = |i: usize| -> (f32, SubStrategy) {
        let idx = cell_idx(i);
        let mut best = (cost_dct8[idx], SubStrategy::Dct8);
        if let Some(g) = extra.dct4x4 {
            if g[idx] < best.0 {
                best = (g[idx], SubStrategy::Dct4x4);
            }
        }
        if let Some(g) = extra.dct4x8 {
            if g[idx] < best.0 {
                best = (g[idx], SubStrategy::Dct4x8);
            }
        }
        if let Some(g) = extra.dct8x4 {
            if g[idx] < best.0 {
                best = (g[idx], SubStrategy::Dct8x4);
            }
        }
        if let Some(g) = extra.identity {
            if g[idx] < best.0 {
                best = (g[idx], SubStrategy::Identity);
            }
        }
        if let Some(g) = extra.dct2x2 {
            if g[idx] < best.0 {
                best = (g[idx], SubStrategy::Dct2x2);
            }
        }
        best
    };
    let cells = [pick_one(0), pick_one(1), pick_one(2), pick_one(3)];
    let total = cells[0].0 + cells[1].0 + cells[2].0 + cells[3].0;
    (total, [cells[0].1, cells[1].1, cells[2].1, cells[3].1])
}

/// Optional cost grids for extra strategies. Pass `None` to skip
/// considering that strategy (e.g. when starting from DCT8/DCT16x16 only).
#[derive(Default, Clone, Copy)]
pub struct CostGrids16x16<'a> {
    /// Per-16×8-block costs, shape `(xsize_blocks_8, ysize_blocks_8 / 2)`.
    /// 16x8 blocks tile the image with the SHORT dimension along x.
    pub dct_16x8: Option<&'a [f32]>,
    /// Per-8×16-block costs, shape `(xsize_blocks_8 / 2, ysize_blocks_8)`.
    pub dct_8x16: Option<&'a [f32]>,
    /// Per-cell 8×8-tier alternative cost grids (DCT4x4, DCT4x8,
    /// DCT8x4, IDENTITY, DCT2X2). Pass any subset; the picker
    /// considers DCT8 + whichever sub-block grids are present and
    /// emits [`Partition16x16::FourSubBlocks`] when the per-cell
    /// total beats the other 3 partition strategies.
    ///
    /// Each grid is per-8×8-cell, shape
    /// `(xsize_blocks_8, ysize_blocks_8)` row-major.
    pub sub_blocks: SubBlockCostGrids<'a>,
}

/// Pick the lower-cost partition for each 16×16 region. Considers up to
/// 4 strategies: DCT16×16, two DCT16×8 vertical, two DCT8×16 horizontal,
/// four DCT8×8. Tie-break order favors larger transforms (per encoder
/// convention to minimize AC-strategy encoding overhead).
///
/// **Inputs:**
/// - `cost_dct8`: per-8×8-block costs, shape `(xsize_blocks_8, ysize_blocks_8)`,
///   row-major. Both dimensions must be even.
/// - `cost_dct16x16`: per-16×16-block costs, shape
///   `(xsize_blocks_8 / 2, ysize_blocks_8 / 2)`, row-major.
/// - `extra`: optional per-strategy cost grids (DCT16×8, DCT8×16).
pub fn select_partitions_16x16(
    cost_dct8: &[f32],
    cost_dct16x16: &[f32],
    xsize_blocks_8: usize,
    ysize_blocks_8: usize,
) -> Vec<Partition16x16> {
    select_partitions_16x16_full(
        cost_dct8,
        cost_dct16x16,
        CostGrids16x16::default(),
        xsize_blocks_8,
        ysize_blocks_8,
    )
}

/// Full per-region selector with optional rectangular DCT16 strategies.
pub fn select_partitions_16x16_full(
    cost_dct8: &[f32],
    cost_dct16x16: &[f32],
    extra: CostGrids16x16<'_>,
    xsize_blocks_8: usize,
    ysize_blocks_8: usize,
) -> Vec<Partition16x16> {
    assert!(xsize_blocks_8.is_multiple_of(2));
    assert!(ysize_blocks_8.is_multiple_of(2));
    let xsize_blocks_16 = xsize_blocks_8 / 2;
    let ysize_blocks_16 = ysize_blocks_8 / 2;
    assert_eq!(cost_dct8.len(), xsize_blocks_8 * ysize_blocks_8);
    assert_eq!(cost_dct16x16.len(), xsize_blocks_16 * ysize_blocks_16);

    let xsize_blocks_16x8 = xsize_blocks_8;
    let ysize_blocks_16x8 = ysize_blocks_8 / 2;
    let xsize_blocks_8x16 = xsize_blocks_8 / 2;
    let ysize_blocks_8x16 = ysize_blocks_8;

    if let Some(g) = extra.dct_16x8 {
        assert_eq!(g.len(), xsize_blocks_16x8 * ysize_blocks_16x8);
    }
    if let Some(g) = extra.dct_8x16 {
        assert_eq!(g.len(), xsize_blocks_8x16 * ysize_blocks_8x16);
    }

    let mut partitions = Vec::with_capacity(xsize_blocks_16 * ysize_blocks_16);
    for ry in 0..ysize_blocks_16 {
        for rx in 0..xsize_blocks_16 {
            let bx0 = rx * 2;
            let by0 = ry * 2;
            let cost_4_dct8 = cost_dct8[by0 * xsize_blocks_8 + bx0]
                + cost_dct8[by0 * xsize_blocks_8 + bx0 + 1]
                + cost_dct8[(by0 + 1) * xsize_blocks_8 + bx0]
                + cost_dct8[(by0 + 1) * xsize_blocks_8 + bx0 + 1];
            let cost_dct16 = cost_dct16x16[ry * xsize_blocks_16 + rx];

            // Two DCT16×8 side-by-side. Each is 16 tall × 8 wide pixels →
            // 1-wide × 2-tall in 8x8-block units. In 16x8 grid (xsize =
            // xsize_blocks_8, ysize = ysize_blocks_8/2), the two blocks
            // occupy grid cells (2*rx, ry) and (2*rx+1, ry).
            let cost_two_16x8 = extra.dct_16x8.map(|g| {
                g[ry * xsize_blocks_16x8 + 2 * rx] + g[ry * xsize_blocks_16x8 + 2 * rx + 1]
            });

            // Two DCT8×16 stacked vertically. Each is 8 tall × 16 wide →
            // 2-wide × 1-tall in 8x8-block units. In 8x16 grid (xsize =
            // xsize_blocks_8/2, ysize = ysize_blocks_8), the two blocks
            // occupy grid cells (rx, 2*ry) and (rx, 2*ry+1).
            let cost_two_8x16 = extra.dct_8x16.map(|g| {
                g[2 * ry * xsize_blocks_8x16 + rx] + g[(2 * ry + 1) * xsize_blocks_8x16 + rx]
            });

            // If any sub-block alternative grids are provided, also
            // compute a per-cell strategy choice. The picker may then
            // emit FourSubBlocks instead of FourDct8x8.
            let any_sub = extra.sub_blocks.dct4x4.is_some()
                || extra.sub_blocks.dct4x8.is_some()
                || extra.sub_blocks.dct8x4.is_some()
                || extra.sub_blocks.identity.is_some()
                || extra.sub_blocks.dct2x2.is_some();
            let cost_sub = if any_sub {
                Some(pick_subblock_strategies(
                    cost_dct8,
                    extra.sub_blocks,
                    xsize_blocks_8,
                    bx0,
                    by0,
                ))
            } else {
                None
            };

            // Pick the strategy with lowest cost. Tie-break: prefer larger
            // transform (DCT16x16 > rectangular > four DCT8x8) to minimize
            // AC-strategy encoding overhead.
            let mut best = (cost_dct16, Partition16x16::Dct16x16);
            if let Some(c) = cost_two_16x8 {
                if c < best.0 {
                    best = (c, Partition16x16::TwoDct16x8Horizontal);
                }
            }
            if let Some(c) = cost_two_8x16 {
                if c < best.0 {
                    best = (c, Partition16x16::TwoDct8x16Vertical);
                }
            }
            if cost_4_dct8 < best.0 {
                best = (cost_4_dct8, Partition16x16::FourDct8x8);
            }
            // Sub-block per-cell pick. Note: when cost_sub.0 == cost_4_dct8
            // (i.e., all 4 cells picked DCT8 anyway), the picker collapses
            // to FourDct8x8 — saving the AC-strategy bits per cell.
            if let Some((c, subs)) = cost_sub {
                if c < best.0 {
                    if subs.iter().all(|&s| s == SubStrategy::Dct8) {
                        best = (c, Partition16x16::FourDct8x8);
                    } else {
                        best = (c, Partition16x16::FourSubBlocks(subs));
                    }
                }
            }
            partitions.push(best.1);
        }
    }
    partitions
}

/// Per-region 32×32 partition. Recursive — Sub16x16 contains four
/// independently-chosen Partition16x16 entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition32x32 {
    /// One DCT32×32 covering the whole region.
    Dct32x32,
    /// Two DCT32×16 side-by-side (each 32 tall × 16 wide; two horizontally
    /// adjacent fill a 32x32 region).
    TwoDct32x16Horizontal,
    /// Two DCT16×32 stacked vertically (each 16 tall × 32 wide).
    TwoDct16x32Vertical,
    /// Four 16×16 sub-regions, each with its own partition choice.
    /// Index 0=TL, 1=TR, 2=BL, 3=BR within the 32×32 region.
    Sub16x16([Partition16x16; 4]),
}

/// Optional cost grids for 32×32-tier rectangular strategies.
#[derive(Default, Clone, Copy)]
pub struct CostGrids32x32<'a> {
    /// Per-32×16-block costs, shape `(xsize_blocks_8 / 2, ysize_blocks_8 / 4)`.
    /// Each DCT32×16 block is 32 tall × 16 wide pixels = 2-wide × 4-tall in
    /// 8x8 blocks.
    pub dct_32x16: Option<&'a [f32]>,
    /// Per-16×32-block costs, shape `(xsize_blocks_8 / 4, ysize_blocks_8 / 2)`.
    /// Each DCT16×32 block is 16 tall × 32 wide pixels = 4-wide × 2-tall.
    pub dct_16x32: Option<&'a [f32]>,
}

/// 2-strategy 32×32 selector (DCT32×32 vs 4 sub-16×16 partitions).
pub fn select_partitions_32x32(
    cost_dct8: &[f32],
    cost_dct16x16: &[f32],
    cost_dct32x32: &[f32],
    xsize_blocks_8: usize,
    ysize_blocks_8: usize,
) -> Vec<Partition32x32> {
    select_partitions_32x32_full(
        cost_dct8,
        cost_dct16x16,
        cost_dct32x32,
        CostGrids32x32::default(),
        xsize_blocks_8,
        ysize_blocks_8,
    )
}

/// Full 4-strategy 32×32 selector with optional rectangular DCT32 grids.
pub fn select_partitions_32x32_full(
    cost_dct8: &[f32],
    cost_dct16x16: &[f32],
    cost_dct32x32: &[f32],
    extra: CostGrids32x32<'_>,
    xsize_blocks_8: usize,
    ysize_blocks_8: usize,
) -> Vec<Partition32x32> {
    select_partitions_32x32_with_extras16(
        cost_dct8,
        cost_dct16x16,
        cost_dct32x32,
        extra,
        CostGrids16x16::default(),
        xsize_blocks_8,
        ysize_blocks_8,
    )
}

/// Like [`select_partitions_32x32_full`] but threads `CostGrids16x16`
/// through the inner 16×16-tier picks. Use this when you've computed
/// rectangular DCT16 / sub-block cost grids and want them to influence
/// the Sub16x16 sub-region picks inside 32×32 regions.
#[allow(clippy::too_many_arguments)]
pub fn select_partitions_32x32_with_extras16(
    cost_dct8: &[f32],
    cost_dct16x16: &[f32],
    cost_dct32x32: &[f32],
    extra: CostGrids32x32<'_>,
    extra16: CostGrids16x16<'_>,
    xsize_blocks_8: usize,
    ysize_blocks_8: usize,
) -> Vec<Partition32x32> {
    assert!(xsize_blocks_8.is_multiple_of(4));
    assert!(ysize_blocks_8.is_multiple_of(4));
    let xsize_blocks_32 = xsize_blocks_8 / 4;
    let ysize_blocks_32 = ysize_blocks_8 / 4;
    let xsize_blocks_16 = xsize_blocks_8 / 2;
    assert_eq!(cost_dct32x32.len(), xsize_blocks_32 * ysize_blocks_32);

    // Rectangular DCT32 grid dimensions
    let xsize_blocks_32x16 = xsize_blocks_8 / 2;
    let ysize_blocks_32x16 = ysize_blocks_8 / 4;
    let xsize_blocks_16x32 = xsize_blocks_8 / 4;
    let ysize_blocks_16x32 = ysize_blocks_8 / 2;
    if let Some(g) = extra.dct_32x16 {
        assert_eq!(g.len(), xsize_blocks_32x16 * ysize_blocks_32x16);
    }
    if let Some(g) = extra.dct_16x32 {
        assert_eq!(g.len(), xsize_blocks_16x32 * ysize_blocks_16x32);
    }

    let sub16 = select_partitions_16x16_full(
        cost_dct8,
        cost_dct16x16,
        extra16,
        xsize_blocks_8,
        ysize_blocks_8,
    );

    let mut partitions = Vec::with_capacity(xsize_blocks_32 * ysize_blocks_32);
    for ry in 0..ysize_blocks_32 {
        for rx in 0..xsize_blocks_32 {
            // Compute cost for the four 16×16 sub-regions inside this 32×32
            let r16x = rx * 2;
            let r16y = ry * 2;
            let sub_choices = [
                sub16[r16y * xsize_blocks_16 + r16x],
                sub16[r16y * xsize_blocks_16 + r16x + 1],
                sub16[(r16y + 1) * xsize_blocks_16 + r16x],
                sub16[(r16y + 1) * xsize_blocks_16 + r16x + 1],
            ];
            let sub_costs = [
                partition_16x16_cost_with_extras(
                    sub_choices[0],
                    cost_dct8,
                    cost_dct16x16,
                    extra16,
                    xsize_blocks_8,
                    xsize_blocks_16,
                    r16x,
                    r16y,
                ),
                partition_16x16_cost_with_extras(
                    sub_choices[1],
                    cost_dct8,
                    cost_dct16x16,
                    extra16,
                    xsize_blocks_8,
                    xsize_blocks_16,
                    r16x + 1,
                    r16y,
                ),
                partition_16x16_cost_with_extras(
                    sub_choices[2],
                    cost_dct8,
                    cost_dct16x16,
                    extra16,
                    xsize_blocks_8,
                    xsize_blocks_16,
                    r16x,
                    r16y + 1,
                ),
                partition_16x16_cost_with_extras(
                    sub_choices[3],
                    cost_dct8,
                    cost_dct16x16,
                    extra16,
                    xsize_blocks_8,
                    xsize_blocks_16,
                    r16x + 1,
                    r16y + 1,
                ),
            ];
            let sub_cost_total: f32 = sub_costs.iter().sum();
            let cost_32 = cost_dct32x32[ry * xsize_blocks_32 + rx];

            // Two DCT32×16 side-by-side. Each 32 tall × 16 wide pixels →
            // 2-wide × 4-tall in 8x8 blocks. In 32x16 grid: cells (2*rx, ry)
            // and (2*rx+1, ry).
            let cost_two_32x16 = extra.dct_32x16.map(|g| {
                g[ry * xsize_blocks_32x16 + 2 * rx] + g[ry * xsize_blocks_32x16 + 2 * rx + 1]
            });

            // Two DCT16×32 stacked vertically. Each 16 tall × 32 wide
            // pixels → 4-wide × 2-tall. In 16x32 grid: cells (rx, 2*ry)
            // and (rx, 2*ry+1).
            let cost_two_16x32 = extra.dct_16x32.map(|g| {
                g[2 * ry * xsize_blocks_16x32 + rx] + g[(2 * ry + 1) * xsize_blocks_16x32 + rx]
            });

            // Pick lowest-cost. Tie-break favors larger transforms.
            let mut best = (cost_32, Partition32x32::Dct32x32);
            if let Some(c) = cost_two_32x16 {
                if c < best.0 {
                    best = (c, Partition32x32::TwoDct32x16Horizontal);
                }
            }
            if let Some(c) = cost_two_16x32 {
                if c < best.0 {
                    best = (c, Partition32x32::TwoDct16x32Vertical);
                }
            }
            if sub_cost_total < best.0 {
                best = (sub_cost_total, Partition32x32::Sub16x16(sub_choices));
            }
            partitions.push(best.1);
        }
    }
    partitions
}

/// Per-region 64×64 partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition64x64 {
    Dct64x64,
    TwoDct64x32Horizontal,
    TwoDct32x64Vertical,
    /// Four 32×32 sub-regions (TL, TR, BL, BR).
    Sub32x32([Partition32x32; 4]),
}

#[derive(Default, Clone, Copy)]
pub struct CostGrids64x64<'a> {
    /// Per-64×32 block: 64 tall × 32 wide pixels = 4-wide × 8-tall in 8x8 units.
    /// Grid xsize = xsize_blocks_8 / 4, ysize = ysize_blocks_8 / 8.
    pub dct_64x32: Option<&'a [f32]>,
    /// Per-32×64 block: 32 tall × 64 wide pixels = 8-wide × 4-tall.
    /// Grid xsize = xsize_blocks_8 / 8, ysize = ysize_blocks_8 / 4.
    pub dct_32x64: Option<&'a [f32]>,
}

/// 64×64 region selector. Considers DCT64×64, 2×DCT64×32, 2×DCT32×64,
/// and 4 sub-32×32 regions.
#[allow(clippy::too_many_arguments)]
pub fn select_partitions_64x64(
    cost_dct8: &[f32],
    cost_dct16x16: &[f32],
    cost_dct32x32: &[f32],
    cost_dct64x64: &[f32],
    extra32: CostGrids32x32<'_>,
    extra64: CostGrids64x64<'_>,
    xsize_blocks_8: usize,
    ysize_blocks_8: usize,
) -> Vec<Partition64x64> {
    select_partitions_64x64_with_extras16(
        cost_dct8,
        cost_dct16x16,
        cost_dct32x32,
        cost_dct64x64,
        extra32,
        extra64,
        CostGrids16x16::default(),
        xsize_blocks_8,
        ysize_blocks_8,
    )
}

/// Like [`select_partitions_64x64`] but threads `CostGrids16x16` through
/// the inner 16×16-tier picks (for FourSubBlocks etc.).
#[allow(clippy::too_many_arguments)]
pub fn select_partitions_64x64_with_extras16(
    cost_dct8: &[f32],
    cost_dct16x16: &[f32],
    cost_dct32x32: &[f32],
    cost_dct64x64: &[f32],
    extra32: CostGrids32x32<'_>,
    extra64: CostGrids64x64<'_>,
    extra16: CostGrids16x16<'_>,
    xsize_blocks_8: usize,
    ysize_blocks_8: usize,
) -> Vec<Partition64x64> {
    assert!(xsize_blocks_8.is_multiple_of(8));
    assert!(ysize_blocks_8.is_multiple_of(8));
    let xsize_blocks_64 = xsize_blocks_8 / 8;
    let ysize_blocks_64 = ysize_blocks_8 / 8;
    let xsize_blocks_32 = xsize_blocks_8 / 4;
    assert_eq!(cost_dct64x64.len(), xsize_blocks_64 * ysize_blocks_64);

    let xsize_blocks_64x32 = xsize_blocks_8 / 4;
    let ysize_blocks_64x32 = ysize_blocks_8 / 8;
    let xsize_blocks_32x64 = xsize_blocks_8 / 8;
    let ysize_blocks_32x64 = ysize_blocks_8 / 4;
    if let Some(g) = extra64.dct_64x32 {
        assert_eq!(g.len(), xsize_blocks_64x32 * ysize_blocks_64x32);
    }
    if let Some(g) = extra64.dct_32x64 {
        assert_eq!(g.len(), xsize_blocks_32x64 * ysize_blocks_32x64);
    }

    let sub32 = select_partitions_32x32_with_extras16(
        cost_dct8,
        cost_dct16x16,
        cost_dct32x32,
        extra32,
        extra16,
        xsize_blocks_8,
        ysize_blocks_8,
    );

    let mut partitions = Vec::with_capacity(xsize_blocks_64 * ysize_blocks_64);
    for ry in 0..ysize_blocks_64 {
        for rx in 0..xsize_blocks_64 {
            let r32x = rx * 2;
            let r32y = ry * 2;
            let sub_choices = [
                sub32[r32y * xsize_blocks_32 + r32x],
                sub32[r32y * xsize_blocks_32 + r32x + 1],
                sub32[(r32y + 1) * xsize_blocks_32 + r32x],
                sub32[(r32y + 1) * xsize_blocks_32 + r32x + 1],
            ];
            let sub_cost_total: f32 = (0..4)
                .map(|i| {
                    let (sx, sy) = match i {
                        0 => (r32x, r32y),
                        1 => (r32x + 1, r32y),
                        2 => (r32x, r32y + 1),
                        _ => (r32x + 1, r32y + 1),
                    };
                    partition_32x32_cost(
                        sub_choices[i],
                        cost_dct8,
                        cost_dct16x16,
                        cost_dct32x32,
                        xsize_blocks_8,
                        xsize_blocks_8 / 2,
                        xsize_blocks_32,
                        sx,
                        sy,
                    )
                })
                .sum();
            let cost_64 = cost_dct64x64[ry * xsize_blocks_64 + rx];

            let cost_two_64x32 = extra64.dct_64x32.map(|g| {
                g[ry * xsize_blocks_64x32 + 2 * rx] + g[ry * xsize_blocks_64x32 + 2 * rx + 1]
            });
            let cost_two_32x64 = extra64.dct_32x64.map(|g| {
                g[2 * ry * xsize_blocks_32x64 + rx] + g[(2 * ry + 1) * xsize_blocks_32x64 + rx]
            });

            let mut best = (cost_64, Partition64x64::Dct64x64);
            if let Some(c) = cost_two_64x32 {
                if c < best.0 {
                    best = (c, Partition64x64::TwoDct64x32Horizontal);
                }
            }
            if let Some(c) = cost_two_32x64 {
                if c < best.0 {
                    best = (c, Partition64x64::TwoDct32x64Vertical);
                }
            }
            if sub_cost_total < best.0 {
                best = (sub_cost_total, Partition64x64::Sub32x32(sub_choices));
            }
            partitions.push(best.1);
        }
    }
    partitions
}

#[allow(clippy::too_many_arguments)]
fn partition_32x32_cost(
    p: Partition32x32,
    cost_dct8: &[f32],
    cost_dct16x16: &[f32],
    cost_dct32x32: &[f32],
    xsize_blocks_8: usize,
    xsize_blocks_16: usize,
    xsize_blocks_32: usize,
    rx32: usize,
    ry32: usize,
) -> f32 {
    match p {
        Partition32x32::Dct32x32 => cost_dct32x32[ry32 * xsize_blocks_32 + rx32],
        Partition32x32::Sub16x16(subs) => {
            let r16x = rx32 * 2;
            let r16y = ry32 * 2;
            let coords = [
                (r16x, r16y),
                (r16x + 1, r16y),
                (r16x, r16y + 1),
                (r16x + 1, r16y + 1),
            ];
            (0..4)
                .map(|i| {
                    let (sx, sy) = coords[i];
                    partition_16x16_cost(
                        subs[i],
                        cost_dct8,
                        cost_dct16x16,
                        xsize_blocks_8,
                        xsize_blocks_16,
                        sx,
                        sy,
                    )
                })
                .sum()
        }
        // Rectangular DCT32 not stored — partition_64x64 caller path can't
        // recover them without the cost grids. Treat as infinite (matches
        // partition_16x16_cost's treatment of rectangulars).
        Partition32x32::TwoDct32x16Horizontal | Partition32x32::TwoDct16x32Vertical => {
            f32::INFINITY
        }
    }
}

fn partition_16x16_cost(
    p: Partition16x16,
    cost_dct8: &[f32],
    cost_dct16x16: &[f32],
    xsize_blocks_8: usize,
    xsize_blocks_16: usize,
    rx16: usize,
    ry16: usize,
) -> f32 {
    partition_16x16_cost_with_extras(
        p,
        cost_dct8,
        cost_dct16x16,
        CostGrids16x16::default(),
        xsize_blocks_8,
        xsize_blocks_16,
        rx16,
        ry16,
    )
}

/// Like [`partition_16x16_cost`] but consumes an extras struct so it
/// can compute exact costs for `TwoDct16x8Horizontal`,
/// `TwoDct8x16Vertical`, and `FourSubBlocks(...)` partitions when the
/// caller has the underlying grids. Returns `f32::INFINITY` when a
/// partition's required grid is `None` (treating it as not-considered).
#[allow(clippy::too_many_arguments)]
fn partition_16x16_cost_with_extras(
    p: Partition16x16,
    cost_dct8: &[f32],
    cost_dct16x16: &[f32],
    extra: CostGrids16x16<'_>,
    xsize_blocks_8: usize,
    xsize_blocks_16: usize,
    rx16: usize,
    ry16: usize,
) -> f32 {
    let bx = rx16 * 2;
    let by = ry16 * 2;
    let cell_cost = |g: &[f32], stride: usize, cx: usize, cy: usize| -> f32 {
        g[cy * stride + cx]
    };
    match p {
        Partition16x16::Dct16x16 => cost_dct16x16[ry16 * xsize_blocks_16 + rx16],
        Partition16x16::FourDct8x8 => {
            cost_dct8[by * xsize_blocks_8 + bx]
                + cost_dct8[by * xsize_blocks_8 + bx + 1]
                + cost_dct8[(by + 1) * xsize_blocks_8 + bx]
                + cost_dct8[(by + 1) * xsize_blocks_8 + bx + 1]
        }
        Partition16x16::TwoDct16x8Horizontal => {
            // Two 16x8 blocks at (bx, by) and (bx+1, by) in the
            // 16x8 grid (xsize = xsize_blocks_8).
            match extra.dct_16x8 {
                Some(g) => cell_cost(g, xsize_blocks_8, bx, ry16)
                    + cell_cost(g, xsize_blocks_8, bx + 1, ry16),
                None => f32::INFINITY,
            }
        }
        Partition16x16::TwoDct8x16Vertical => {
            // Two 8x16 blocks at (rx16, by) and (rx16, by+1) in the
            // 8x16 grid (xsize = xsize_blocks_8 / 2 = xsize_blocks_16).
            match extra.dct_8x16 {
                Some(g) => cell_cost(g, xsize_blocks_16, rx16, by)
                    + cell_cost(g, xsize_blocks_16, rx16, by + 1),
                None => f32::INFINITY,
            }
        }
        Partition16x16::FourSubBlocks(subs) => {
            let pick_one = |sub: SubStrategy, dx: usize, dy: usize| -> f32 {
                let idx = (by + dy) * xsize_blocks_8 + bx + dx;
                match sub {
                    SubStrategy::Dct8 => cost_dct8[idx],
                    SubStrategy::Dct4x4 => extra
                        .sub_blocks
                        .dct4x4
                        .map(|g| g[idx])
                        .unwrap_or(f32::INFINITY),
                    SubStrategy::Dct4x8 => extra
                        .sub_blocks
                        .dct4x8
                        .map(|g| g[idx])
                        .unwrap_or(f32::INFINITY),
                    SubStrategy::Dct8x4 => extra
                        .sub_blocks
                        .dct8x4
                        .map(|g| g[idx])
                        .unwrap_or(f32::INFINITY),
                    SubStrategy::Identity => extra
                        .sub_blocks
                        .identity
                        .map(|g| g[idx])
                        .unwrap_or(f32::INFINITY),
                    SubStrategy::Dct2x2 => extra
                        .sub_blocks
                        .dct2x2
                        .map(|g| g[idx])
                        .unwrap_or(f32::INFINITY),
                }
            };
            pick_one(subs[0], 0, 0)
                + pick_one(subs[1], 1, 0)
                + pick_one(subs[2], 0, 1)
                + pick_one(subs[3], 1, 1)
        }
    }
}

// =============================================================================
// Phase 3 Component 3: Partition → block-strategy assignment
// =============================================================================

/// Per-block AC strategy assignment: for each first-block of a strategy,
/// records the strategy code and where its top-left 8×8-block sits in
/// the image's 8x8 grid.
///
/// Use this as the bridge between the host-side partition selector
/// (`select_partitions_*`) and the GPU-side mixed-strategy reconstruct
/// (`forks::reconstruct::reconstruct_mixed_strategy_gpu`). The
/// `BlockRecipe` struct that reconstruct takes carries `coeffs`
/// alongside `(bx, by, raw_strategy)`; this struct is the
/// strategy-only metadata so callers can build coefficient batches
/// per-strategy before assembling final recipes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StrategyAssignment {
    /// Top-left 8×8-block coordinate (in the image's 8x8 grid).
    pub bx: usize,
    pub by: usize,
    /// Strategy code (matching `forks::transform::RAW_STRATEGY_*`).
    pub raw_strategy: u8,
}

/// Walk a `Vec<Partition16x16>` to a flat list of strategy assignments.
/// Each Partition16x16 occupies 2×2 8×8-blocks in the image.
///
/// Mirrors libjxl's `AcStrategy::Set(bx, by, kind)` flat assignments —
/// this is the GPU-side equivalent of the per-block strategy map that
/// upstream stores in `AcStrategyMap`.
///
/// `xsize_blocks_8` / `ysize_blocks_8` are the image's 8x8-block
/// grid dimensions. Both must be even (`select_partitions_16x16`
/// already asserts this).
///
/// Note: `Partition16x16::FourSubBlocks` emits per-cell strategies
/// (DCT8 / DCT4×4 / DCT4×8 / DCT8×4 / IDENTITY / DCT2X2). For the
/// MVP we only walk DCT8 and DCT16x16; the other variants are
/// translated to their `RAW_STRATEGY_*` codes for completeness so
/// future Phase D work can drop in without touching this walker.
pub fn partitions_16x16_to_assignments(
    partitions: &[Partition16x16],
    xsize_blocks_8: usize,
    ysize_blocks_8: usize,
) -> Vec<StrategyAssignment> {
    use crate::forks::transform::{
        RAW_STRATEGY_DCT, RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8,
        RAW_STRATEGY_DCT8X4, RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16,
        RAW_STRATEGY_IDENTITY,
    };

    assert!(xsize_blocks_8.is_multiple_of(2));
    assert!(ysize_blocks_8.is_multiple_of(2));
    let xsize_blocks_16 = xsize_blocks_8 / 2;
    let ysize_blocks_16 = ysize_blocks_8 / 2;
    assert_eq!(partitions.len(), xsize_blocks_16 * ysize_blocks_16);

    let mut out = Vec::with_capacity(xsize_blocks_8 * ysize_blocks_8);
    for ry in 0..ysize_blocks_16 {
        for rx in 0..xsize_blocks_16 {
            let bx = rx * 2;
            let by = ry * 2;
            match partitions[ry * xsize_blocks_16 + rx] {
                Partition16x16::Dct16x16 => {
                    out.push(StrategyAssignment {
                        bx,
                        by,
                        raw_strategy: RAW_STRATEGY_DCT16X16,
                    });
                }
                Partition16x16::TwoDct16x8Horizontal => {
                    // Two 16×8 (16 tall × 8 wide) blocks side-by-side.
                    // Each occupies 1×2 in the 8x8 grid: (bx, by..by+1) and (bx+1, by..by+1).
                    out.push(StrategyAssignment {
                        bx,
                        by,
                        raw_strategy: RAW_STRATEGY_DCT16X8,
                    });
                    out.push(StrategyAssignment {
                        bx: bx + 1,
                        by,
                        raw_strategy: RAW_STRATEGY_DCT16X8,
                    });
                }
                Partition16x16::TwoDct8x16Vertical => {
                    // Two 8×16 (8 tall × 16 wide) blocks stacked.
                    // Each occupies 2×1 in the 8x8 grid.
                    out.push(StrategyAssignment {
                        bx,
                        by,
                        raw_strategy: RAW_STRATEGY_DCT8X16,
                    });
                    out.push(StrategyAssignment {
                        bx,
                        by: by + 1,
                        raw_strategy: RAW_STRATEGY_DCT8X16,
                    });
                }
                Partition16x16::FourDct8x8 => {
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        out.push(StrategyAssignment {
                            bx: bx + dx,
                            by: by + dy,
                            raw_strategy: RAW_STRATEGY_DCT,
                        });
                    }
                }
                Partition16x16::FourSubBlocks(subs) => {
                    for ((dx, dy), sub) in
                        [(0, 0), (1, 0), (0, 1), (1, 1)].iter().zip(subs.iter())
                    {
                        let raw = match sub {
                            SubStrategy::Dct8 => RAW_STRATEGY_DCT,
                            SubStrategy::Dct4x4 => RAW_STRATEGY_DCT4X4,
                            SubStrategy::Dct4x8 => RAW_STRATEGY_DCT4X8,
                            SubStrategy::Dct8x4 => RAW_STRATEGY_DCT8X4,
                            SubStrategy::Identity => RAW_STRATEGY_IDENTITY,
                            SubStrategy::Dct2x2 => RAW_STRATEGY_DCT2X2,
                        };
                        out.push(StrategyAssignment {
                            bx: bx + dx,
                            by: by + dy,
                            raw_strategy: raw,
                        });
                    }
                }
            }
        }
    }
    out
}

/// Convert a list of `Partition32x32` partitions (in 32×32-grid raster
/// order) into a flat list of strategy assignments at the 8×8-block
/// grid level.
///
/// Same shape as [`partitions_16x16_to_assignments`] but operates at
/// the 32×32 tier. For `Sub16x16` partitions, recurses through the
/// 4 contained `Partition16x16` choices via the existing 16×16 lowering.
///
/// `xsize_blocks_8` and `ysize_blocks_8` must both be multiples of 4
/// (otherwise the 32×32 grid doesn't tile cleanly).
pub fn partitions_32x32_to_assignments(
    partitions: &[Partition32x32],
    xsize_blocks_8: usize,
    ysize_blocks_8: usize,
) -> Vec<StrategyAssignment> {
    use crate::forks::transform::{
        RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32,
    };

    assert!(xsize_blocks_8.is_multiple_of(4));
    assert!(ysize_blocks_8.is_multiple_of(4));
    let xsize_blocks_32 = xsize_blocks_8 / 4;
    let ysize_blocks_32 = ysize_blocks_8 / 4;
    let xsize_blocks_16 = xsize_blocks_8 / 2;
    assert_eq!(partitions.len(), xsize_blocks_32 * ysize_blocks_32);

    let mut out = Vec::with_capacity(xsize_blocks_8 * ysize_blocks_8);
    for ry in 0..ysize_blocks_32 {
        for rx in 0..xsize_blocks_32 {
            let bx = rx * 4;
            let by = ry * 4;
            match partitions[ry * xsize_blocks_32 + rx] {
                Partition32x32::Dct32x32 => {
                    out.push(StrategyAssignment {
                        bx,
                        by,
                        raw_strategy: RAW_STRATEGY_DCT32X32,
                    });
                }
                Partition32x32::TwoDct32x16Horizontal => {
                    // Two 32×16 (32 tall × 16 wide) side-by-side. Each
                    // covers 2×4 in the 8x8 grid.
                    out.push(StrategyAssignment {
                        bx,
                        by,
                        raw_strategy: RAW_STRATEGY_DCT32X16,
                    });
                    out.push(StrategyAssignment {
                        bx: bx + 2,
                        by,
                        raw_strategy: RAW_STRATEGY_DCT32X16,
                    });
                }
                Partition32x32::TwoDct16x32Vertical => {
                    // Two 16×32 (16 tall × 32 wide) stacked. Each covers
                    // 4×2 in the 8x8 grid.
                    out.push(StrategyAssignment {
                        bx,
                        by,
                        raw_strategy: RAW_STRATEGY_DCT16X32,
                    });
                    out.push(StrategyAssignment {
                        bx,
                        by: by + 2,
                        raw_strategy: RAW_STRATEGY_DCT16X32,
                    });
                }
                Partition32x32::Sub16x16(subs) => {
                    // Recurse: build a 2x2 mini-grid of Partition16x16
                    // and call the 16x16 lowering on it. Position
                    // ordering matches Partition16x16 conventions: TL,
                    // TR, BL, BR within the 32×32 region.
                    let mini = [subs[0], subs[1], subs[2], subs[3]];
                    let mini_assignments =
                        partitions_16x16_to_assignments(&mini, 4, 4);
                    // mini_assignments is in 8×8-block coords relative to
                    // a 4×4-block (= 32×32 pixel) region; offset by (bx, by).
                    for a in mini_assignments {
                        out.push(StrategyAssignment {
                            bx: bx + a.bx,
                            by: by + a.by,
                            raw_strategy: a.raw_strategy,
                        });
                    }
                }
            }
        }
    }
    let _ = xsize_blocks_16;
    out
}

/// Convert a list of `Partition64x64` partitions (in 64×64-grid raster
/// order) into a flat list of strategy assignments at the 8×8-block
/// grid level.
///
/// Recurses through Sub32x32 → Partition32x32 via the existing
/// [`partitions_32x32_to_assignments`] helper.
///
/// `xsize_blocks_8` and `ysize_blocks_8` must both be multiples of 8
/// (otherwise the 64×64 grid doesn't tile cleanly).
pub fn partitions_64x64_to_assignments(
    partitions: &[Partition64x64],
    xsize_blocks_8: usize,
    ysize_blocks_8: usize,
) -> Vec<StrategyAssignment> {
    use crate::forks::transform::{
        RAW_STRATEGY_DCT32X64, RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64,
    };

    assert!(xsize_blocks_8.is_multiple_of(8));
    assert!(ysize_blocks_8.is_multiple_of(8));
    let xsize_blocks_64 = xsize_blocks_8 / 8;
    let ysize_blocks_64 = ysize_blocks_8 / 8;
    assert_eq!(partitions.len(), xsize_blocks_64 * ysize_blocks_64);

    let mut out = Vec::with_capacity(xsize_blocks_8 * ysize_blocks_8);
    for ry in 0..ysize_blocks_64 {
        for rx in 0..xsize_blocks_64 {
            let bx = rx * 8;
            let by = ry * 8;
            match partitions[ry * xsize_blocks_64 + rx] {
                Partition64x64::Dct64x64 => {
                    out.push(StrategyAssignment {
                        bx,
                        by,
                        raw_strategy: RAW_STRATEGY_DCT64X64,
                    });
                }
                Partition64x64::TwoDct64x32Horizontal => {
                    // Two 64×32 (64 tall × 32 wide) side-by-side. Each
                    // covers 4×8 in the 8x8 grid.
                    out.push(StrategyAssignment {
                        bx,
                        by,
                        raw_strategy: RAW_STRATEGY_DCT64X32,
                    });
                    out.push(StrategyAssignment {
                        bx: bx + 4,
                        by,
                        raw_strategy: RAW_STRATEGY_DCT64X32,
                    });
                }
                Partition64x64::TwoDct32x64Vertical => {
                    // Two 32×64 (32 tall × 64 wide) stacked. Each covers
                    // 8×4 in the 8x8 grid.
                    out.push(StrategyAssignment {
                        bx,
                        by,
                        raw_strategy: RAW_STRATEGY_DCT32X64,
                    });
                    out.push(StrategyAssignment {
                        bx,
                        by: by + 4,
                        raw_strategy: RAW_STRATEGY_DCT32X64,
                    });
                }
                Partition64x64::Sub32x32(subs) => {
                    // Recurse: build 4 Partition32x32 entries in raster
                    // order [TL, TR, BL, BR] within the 64×64 region
                    // and call the 32x32 lowering on the 8×8-block-sized
                    // mini-grid (8 blocks per side = 4×4 in 32x32-grid).
                    let mini = [subs[0], subs[1], subs[2], subs[3]];
                    let mini_assignments =
                        partitions_32x32_to_assignments(&mini, 8, 8);
                    for a in mini_assignments {
                        out.push(StrategyAssignment {
                            bx: bx + a.bx,
                            by: by + a.by,
                            raw_strategy: a.raw_strategy,
                        });
                    }
                }
            }
        }
    }
    out
}

/// Group strategy assignments by `raw_strategy` code. Returns a
/// `Vec<(raw_strategy, Vec<(bx, by)>)>` with strategies in
/// ascending raw_strategy order — convenient for the per-strategy
/// encode pass (each entry batches all blocks of one strategy
/// into a single DCT + quantize + dequant launch).
///
/// The returned `(bx, by)` pairs preserve the order they appeared
/// in `assignments` (within each strategy group), so callers can
/// match them back to the original assignment list.
pub fn group_assignments_by_strategy(
    assignments: &[StrategyAssignment],
) -> Vec<(u8, Vec<(usize, usize)>)> {
    use alloc::collections::BTreeMap;
    let mut by_strat: BTreeMap<u8, Vec<(usize, usize)>> = BTreeMap::new();
    for a in assignments {
        by_strat
            .entry(a.raw_strategy)
            .or_default()
            .push((a.bx, a.by));
    }
    by_strat.into_iter().collect()
}

#[cfg(test)]
mod strategy_assignment_tests {
    use super::*;
    use crate::forks::transform::{
        RAW_STRATEGY_DCT, RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16,
    };

    #[test]
    fn test_group_assignments_by_strategy() {
        let assignments = vec![
            StrategyAssignment {
                bx: 0,
                by: 0,
                raw_strategy: RAW_STRATEGY_DCT16X16,
            },
            StrategyAssignment {
                bx: 2,
                by: 0,
                raw_strategy: RAW_STRATEGY_DCT,
            },
            StrategyAssignment {
                bx: 3,
                by: 0,
                raw_strategy: RAW_STRATEGY_DCT,
            },
            StrategyAssignment {
                bx: 0,
                by: 2,
                raw_strategy: RAW_STRATEGY_DCT16X16,
            },
        ];
        let groups = group_assignments_by_strategy(&assignments);
        // BTreeMap orders by raw_strategy: DCT (0) before DCT16X16 (3)
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].0, RAW_STRATEGY_DCT);
        assert_eq!(groups[0].1, vec![(2, 0), (3, 0)]);
        assert_eq!(groups[1].0, RAW_STRATEGY_DCT16X16);
        assert_eq!(groups[1].1, vec![(0, 0), (0, 2)]);
    }

    #[test]
    fn test_partitions_to_assignments_all_dct16() {
        // 4×4 image (in 8x8 blocks) = 2×2 in 16x16 regions.
        let partitions = vec![Partition16x16::Dct16x16; 4];
        let out = partitions_16x16_to_assignments(&partitions, 4, 4);
        assert_eq!(out.len(), 4);
        for (i, a) in out.iter().enumerate() {
            let rx = (i % 2) * 2;
            let ry = (i / 2) * 2;
            assert_eq!(a.bx, rx);
            assert_eq!(a.by, ry);
            assert_eq!(a.raw_strategy, RAW_STRATEGY_DCT16X16);
        }
    }

    #[test]
    fn test_partitions_to_assignments_four_dct8() {
        // One 16x16 region = four 8x8 blocks.
        let partitions = vec![Partition16x16::FourDct8x8];
        let out = partitions_16x16_to_assignments(&partitions, 2, 2);
        assert_eq!(out.len(), 4);
        let coords: Vec<(usize, usize)> = out.iter().map(|a| (a.bx, a.by)).collect();
        assert_eq!(coords, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
        for a in &out {
            assert_eq!(a.raw_strategy, RAW_STRATEGY_DCT);
        }
    }

    #[test]
    fn test_partitions_to_assignments_rectangular() {
        // Test both rectangular variants in a 4×2 image (2×1 16x16-regions).
        let partitions = vec![
            Partition16x16::TwoDct16x8Horizontal,
            Partition16x16::TwoDct8x16Vertical,
        ];
        let out = partitions_16x16_to_assignments(&partitions, 4, 2);
        assert_eq!(out.len(), 4);
        // First region: two DCT16x8 at (0,0) and (1,0)
        assert_eq!(out[0].raw_strategy, RAW_STRATEGY_DCT16X8);
        assert_eq!((out[0].bx, out[0].by), (0, 0));
        assert_eq!(out[1].raw_strategy, RAW_STRATEGY_DCT16X8);
        assert_eq!((out[1].bx, out[1].by), (1, 0));
        // Second region: two DCT8x16 at (2,0) and (2,1)
        assert_eq!(out[2].raw_strategy, RAW_STRATEGY_DCT8X16);
        assert_eq!((out[2].bx, out[2].by), (2, 0));
        assert_eq!(out[3].raw_strategy, RAW_STRATEGY_DCT8X16);
        assert_eq!((out[3].bx, out[3].by), (2, 1));
    }
}
