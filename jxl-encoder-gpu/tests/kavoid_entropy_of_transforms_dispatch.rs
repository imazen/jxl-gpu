// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! API + gate-semantics tests for the chunk-1 GPU port of libjxl's
//! `kAvoidEntropyOfTransforms` heuristic
//! (`LossyEncoder::with_enable_kavoid_entropy_of_transforms`).
//!
//! Chunk 1 scope:
//! - Helper [`forks::cost::avoid_entropy_of_transforms_mul`] formula matches
//!   libjxl `enc_ac_strategy.cc::FindBest8x8Transform` (covered in
//!   `forks/cost.rs::tests::test_avoid_entropy_of_transforms_mul_libjxl_parity`,
//!   no CUDA required).
//! - Const [`forks::cost::K_AVOID_TRANSFORMS_BASE`] matches libjxl
//!   `kAvoidEntropyOfTransforms = 0.5` and the CPU encoder's
//!   `EffortProfile::k_avoid_transforms_base`.
//! - Builder method round-trips; default is `false` (opt-in).
//! - Dispatch runs without panic in both flag states at d=5.0 (where the
//!   penalty fires) and d=1.0 (where the formula returns 0.0 and the path
//!   is a structural no-op).
//!
//! What's NOT in chunk 1 (deferred):
//! - AFV cost-path integration (AFV picks flow through a separate path).
//! - Auto-dispatch wired into `auto_libjxl_entropy_mul_on_photos`. This
//!   chunk adds the missing counterweight; a follow-on can re-enable the
//!   libjxl-faithful entropy_mul branch on photos.
//! - The X-channel multi-block weight (also missing from the GPU cost
//!   grids — see `dropped_optimizations_for_parity_2026-05-15.md` item #3
//!   for the bundle context).
//! - Byte-Δ measurement on a photo corpus — that lives in an example
//!   harness (see `examples/auto_entropy_mul_bytes_ab.rs` for the
//!   sibling pattern).

#![cfg(all(feature = "cuda", feature = "encoder"))]

use jxl_encoder_gpu::encoder::GpuEncoder;
use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

type Backend = cubecl::cuda::CudaRuntime;

const W: u32 = 16;
const H: u32 = 16;

fn dummy_rgb_planes() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = (W * H) as usize;
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for y in 0..H {
        for x in 0..W {
            // Mid-grey with a tiny gradient — content is irrelevant for an
            // API / panic-free smoke test.
            let v = ((x + y) as f32 / 32.0).clamp(0.05, 0.95);
            r.push(v);
            g.push(v);
            b.push(v);
        }
    }
    (r, g, b)
}

#[test]
fn test_enable_kavoid_entropy_of_transforms_default_is_off() {
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, W, H);
    // Default MUST be `false`. Chunk-1 ships the port behind an opt-in
    // flag so production at distance > 4.0 stays byte-identical until
    // the full multi-week bundle (AFV path + X-channel multi-block
    // weight) lands. At distance <= 4.0 the formula returns 0.0 so the
    // path is a structural no-op regardless of the flag.
    assert!(
        !lossy.enable_kavoid_entropy_of_transforms(),
        "enable_kavoid_entropy_of_transforms() MUST default to false — \
         see field docs for the chunk-1 POC rationale"
    );
}

#[test]
fn test_enable_kavoid_entropy_of_transforms_with_toggle_round_trips() {
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, W, H);

    let on = lossy.with_enable_kavoid_entropy_of_transforms(true);
    assert!(on.enable_kavoid_entropy_of_transforms());

    let off = on.with_enable_kavoid_entropy_of_transforms(false);
    assert!(!off.enable_kavoid_entropy_of_transforms());
}

#[test]
fn test_dispatch_runs_without_panic_low_d_no_op_branch() {
    // At d=1.0 the formula returns 0.0 — the path is a structural
    // no-op even when the flag is on. Both states must produce the
    // same cost-grid shape (and identical output downstream).
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let (r, g, b) = dummy_rgb_planes();

    let off: LossyEncoder<Backend> = LossyEncoder::new(&enc, W, H);
    let _ = off.prepare_strategy_search_plan(&enc, &r, &g, &b, 1.0);

    let on: LossyEncoder<Backend> =
        LossyEncoder::new(&enc, W, H).with_enable_kavoid_entropy_of_transforms(true);
    let _ = on.prepare_strategy_search_plan(&enc, &r, &g, &b, 1.0);
}

#[test]
fn test_dispatch_runs_without_panic_high_d_penalty_branch() {
    // At d=5.0 the formula returns (12-4)/(5-4) = 8.0; the per-strategy
    // adjustment is 0.5 * 8.0 = 4.0 added to DCT4X4 / DCT4X8 / DCT8X4
    // entropy_mul. Confirms the plan can still be prepared with the
    // boosted entropy_mul values (max(0.01) clamp keeps them positive).
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let (r, g, b) = dummy_rgb_planes();

    let on: LossyEncoder<Backend> =
        LossyEncoder::new(&enc, W, H).with_enable_kavoid_entropy_of_transforms(true);
    let _ = on.prepare_strategy_search_plan(&enc, &r, &g, &b, 5.0);
}
