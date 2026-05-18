// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Integration test for the AFV-preservation diagnostic sink shipped
//! in chunk 1 of the "GPU patches AFV preservation across case-1
//! recompute" follow-on to W7-3 (`406b40bb`).
//!
//! Proves:
//! 1. The diagnostic sink is populated after every encode through
//!    [`jxl_encoder_gpu::encoder::GpuEncoder::encode_lossy_to_bitstream_via_precomputed`].
//! 2. On a synthetic image where patches detection produces nothing
//!    (smooth gradient), `patches_recompute_fired = false` and
//!    `gpu_afv_picks_pre == cpu_afv_picks_post` (baseline snapshot).
//! 3. The total_blocks field matches the padded grid dimensions.
//! 4. Per-class histogram sums to total first-blocks (which is
//!    ≤ total_blocks; covered blocks of multi-block transforms don't
//!    count).
//!
//! The chunk-1 bench
//! (`benchmarks/afv_preservation_diagnostic_d1_2026-05-17.{txt,meta}`)
//! is the production validation against the W7-3 sweep image set;
//! this test is the synthetic Layer-1 invariant.

#![cfg(all(feature = "cuda", feature = "encoder"))]

use jxl_encoder_gpu::diagnostics::{
    LastAfvPreservationStats, take_last_afv_preservation_stats,
};
use jxl_encoder_gpu::encoder::GpuEncoder;
use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

type Backend = cubecl::cuda::CudaRuntime;

const W: u32 = 64;
const H: u32 = 64;
const N: usize = (W * H) as usize;

/// Build a smooth gradient — guaranteed to produce no patches and no
/// AFV picks (smooth content → DCT8/larger).
fn smooth_gradient() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = Vec::with_capacity(N);
    let mut g = Vec::with_capacity(N);
    let mut b = Vec::with_capacity(N);
    for y in 0..H {
        for x in 0..W {
            let v = (x as f32 + y as f32) / (2.0 * (W - 1) as f32);
            r.push(v);
            g.push(v);
            b.push(v);
        }
    }
    (r, g, b)
}

/// Smoke test: the sink is populated after every encode (regardless
/// of whether patches were detected). On a smooth gradient patches
/// detection produces nothing so the baseline snapshot lands
/// (`patches_recompute_fired = false`).
#[test]
fn test_diagnostic_sink_populated_after_encode() {
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let (r, g, b) = smooth_gradient();
    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, W, H);
    let _bs = enc
        .encode_lossy_to_bitstream_via_precomputed(&lossy, &r, &g, &b, 1.0)
        .expect("encode must succeed");

    let stats = take_last_afv_preservation_stats()
        .expect("diagnostic sink must be populated by encode");

    // Padded grid: 64×64 → 8×8 blocks = 64 blocks total.
    let cpu_pw = (W as usize).div_ceil(8) * 8;
    let cpu_ph = (H as usize).div_ceil(8) * 8;
    let expected_blocks = ((cpu_pw / 8) * (cpu_ph / 8)) as u32;
    assert_eq!(
        stats.total_blocks, expected_blocks,
        "total_blocks must match the padded 8x8-block grid"
    );

    // Smooth gradient → no patches → recompute did not fire.
    assert!(
        !stats.patches_recompute_fired,
        "patches recompute must not fire on smooth gradient (no patches)"
    );

    // Baseline contract: when the recompute didn't fire, pre == post.
    assert_eq!(
        stats.gpu_afv_picks_pre, stats.cpu_afv_picks_post,
        "AFV pre / post must be equal when no recompute fired"
    );
    assert_eq!(
        stats.gpu_dct8_picks_pre, stats.cpu_dct8_picks_post,
        "DCT8 pre / post must be equal when no recompute fired"
    );
    assert_eq!(
        stats.gpu_strategy_histogram_pre, stats.cpu_strategy_histogram_post,
        "Full histogram must be unchanged when no recompute fired"
    );

    // First-block sum bound: sum of histogram first-blocks ≤ total_blocks
    // (multi-block transforms occupy multiple physical blocks but only
    // one first-block entry).
    let first_block_total: u32 = stats.gpu_strategy_histogram_pre.iter().sum();
    assert!(
        first_block_total <= stats.total_blocks,
        "first-block sum {first_block_total} must be ≤ total_blocks {}",
        stats.total_blocks
    );

    // AFV count is the sum of indices 12..=15.
    let afv_sum: u32 = stats.gpu_strategy_histogram_pre[12..=15].iter().sum();
    assert_eq!(
        afv_sum, stats.gpu_afv_picks_pre,
        "gpu_afv_picks_pre must equal histogram[12..=15].sum()"
    );
}

/// `take_last_afv_preservation_stats` consumes the slot — a second
/// call without an intervening encode returns `None`.
#[test]
fn test_diagnostic_sink_consumes_on_take() {
    // Stash a baseline so the slot is populated.
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let (r, g, b) = smooth_gradient();
    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, W, H);
    let _bs = enc
        .encode_lossy_to_bitstream_via_precomputed(&lossy, &r, &g, &b, 1.0)
        .expect("encode must succeed");

    let first: Option<LastAfvPreservationStats> = take_last_afv_preservation_stats();
    assert!(first.is_some(), "first take must return Some");
    let second: Option<LastAfvPreservationStats> = take_last_afv_preservation_stats();
    assert!(
        second.is_none(),
        "second take (no intervening encode) must return None"
    );
}
