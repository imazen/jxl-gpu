// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Phase 3 prototype: whole-image cost-grid composition.
//!
//! Demonstrates the **whole-image-per-strategy** AC strategy search
//! pattern documented in `CLAUDE.md`. For each candidate strategy S
//! (here DCT8 only — the most common case), one coherent whole-image
//! pipeline computes per-block cost. A future host-side partition
//! selector can then enumerate legal partitions per 32×32 / 64×64
//! region and pick the min-cost partition.
//!
//! Pipeline for DCT8 single-channel (Y), no CfL:
//! 1. Forward DCT 8x8 → coefficients
//! 2. Quantize → quantized integer coeffs
//! 3. Dequantize → reconstructed coeffs
//! 4. Inverse DCT → reconstructed pixels
//! 5. Compute per-block L2 error vs original (proxy for pixel_loss)
//! 6. Combine into per-block cost
//!
//! This is the **simplest** proof-of-concept; full Phase 3 needs:
//! - 3-channel coverage with CfL decorrelation
//! - Multiple candidate strategies (DCT8, DCT16, DCT32, DCT64, DCT4, AFV)
//! - Host-side partition selector reading the per-strategy cost grids
//! - Refactor of `ac_strategy_search.rs` in `jxl-encoder` crate

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
    dct8::{dct_8x8, idct_8x8},
    dct16::{dct_16x16, idct_16x16},
    dct32::{dct_32x32, idct_32x32},
    dct64::{dct_64x64, idct_64x64},
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
    /// Four DCT8×8 blocks (TL, TR, BL, BR).
    FourDct8x8,
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

    let sub16 = select_partitions_16x16(cost_dct8, cost_dct16x16, xsize_blocks_8, ysize_blocks_8);

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
                partition_16x16_cost(
                    sub_choices[0],
                    cost_dct8,
                    cost_dct16x16,
                    xsize_blocks_8,
                    xsize_blocks_16,
                    r16x,
                    r16y,
                ),
                partition_16x16_cost(
                    sub_choices[1],
                    cost_dct8,
                    cost_dct16x16,
                    xsize_blocks_8,
                    xsize_blocks_16,
                    r16x + 1,
                    r16y,
                ),
                partition_16x16_cost(
                    sub_choices[2],
                    cost_dct8,
                    cost_dct16x16,
                    xsize_blocks_8,
                    xsize_blocks_16,
                    r16x,
                    r16y + 1,
                ),
                partition_16x16_cost(
                    sub_choices[3],
                    cost_dct8,
                    cost_dct16x16,
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

    let sub32 = select_partitions_32x32_full(
        cost_dct8,
        cost_dct16x16,
        cost_dct32x32,
        extra32,
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
    match p {
        Partition16x16::Dct16x16 => cost_dct16x16[ry16 * xsize_blocks_16 + rx16],
        Partition16x16::FourDct8x8 => {
            let bx = rx16 * 2;
            let by = ry16 * 2;
            cost_dct8[by * xsize_blocks_8 + bx]
                + cost_dct8[by * xsize_blocks_8 + bx + 1]
                + cost_dct8[(by + 1) * xsize_blocks_8 + bx]
                + cost_dct8[(by + 1) * xsize_blocks_8 + bx + 1]
        }
        // Rectangular DCT16 variants: in this simplified prototype,
        // partition_16x16_cost is only called from select_partitions_32x32
        // which uses select_partitions_16x16 (without rect variants).
        // Treat as infinite cost so they're never picked at this level.
        // Full Phase 3 would extend select_partitions_32x32 to pass through
        // the extra cost grids and recompute these costs precisely.
        Partition16x16::TwoDct16x8Horizontal | Partition16x16::TwoDct8x16Vertical => f32::INFINITY,
    }
}
