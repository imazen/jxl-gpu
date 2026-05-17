// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Smoke tests for the opt-in entropy_mul + dist_bias content-discriminated
//! bundle dispatch (`LossyEncoder::with_auto_libjxl_entropy_mul_on_photos`).
//!
//! The 2026-05-17 A/B run on 3 CLIC photos + 3 GB82-SC screenshots refuted
//! the audit hypothesis: the libjxl-faithful entropy_mul + dropped
//! `dist_bias` branch is Pareto-worse on photos (bytes +2.5% to +8.5%,
//! butteraugli +0.11 to +0.24, SSIM2 −0.17 to −0.42 at d=1.0). The
//! dispatch is therefore opt-in (default `false`). These tests only assert
//! the API surface + default-off contract; they do NOT re-run the metric
//! sweep (that lives in `examples/auto_entropy_mul_bytes_ab.rs`).
//!
//! Screenshot byte-identity is enforced by the `corpus_regression`
//! test (gb82-sc rows) — the discriminator picks the GPU-lifted branch
//! on screenshots regardless of the auto flag, so the toggle is a no-op
//! there.

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
            // Mid-grey with a tiny gradient so the encoder has *something*
            // to process; content doesn't matter — we're only exercising
            // the API surface.
            let v = ((x + y) as f32 / 32.0).clamp(0.05, 0.95);
            r.push(v);
            g.push(v);
            b.push(v);
        }
    }
    (r, g, b)
}

#[test]
fn test_auto_libjxl_entropy_mul_default_is_off() {
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, W, H);
    // Default MUST be `false`. The 2026-05-17 A/B refuted the audit
    // hypothesis; flipping this to `true` reintroduces Pareto-worse
    // photo behavior. Update this assertion ONLY after the GPU port of
    // `kAvoidEntropyOfTransforms` + X-channel multi-block weight
    // (cross-ref dropped log item #4) and a fresh A/B confirming the
    // photo branch is no longer Pareto-worse.
    assert!(
        !lossy.auto_libjxl_entropy_mul_on_photos(),
        "auto_libjxl_entropy_mul_on_photos() MUST default to false — \
         see field docs + benchmarks/entropy_mul_bundle_ab_*_2026-05-17.txt"
    );
}

#[test]
fn test_auto_libjxl_entropy_mul_with_toggle_round_trips() {
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, W, H);

    let on = lossy.with_auto_libjxl_entropy_mul_on_photos(true);
    assert!(on.auto_libjxl_entropy_mul_on_photos());

    let off = on.with_auto_libjxl_entropy_mul_on_photos(false);
    assert!(!off.auto_libjxl_entropy_mul_on_photos());
}

#[test]
fn test_auto_libjxl_entropy_mul_dispatch_runs_without_panic() {
    // Sanity: exercising the dispatch on both branches must not panic.
    // The actual byte / quality outcome is measured by the example
    // harness (`auto_entropy_mul_bytes_ab.rs`); here we only confirm
    // the strat-search plan can be prepared in both modes.
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let (r, g, b) = dummy_rgb_planes();

    let off: LossyEncoder<Backend> = LossyEncoder::new(&enc, W, H);
    let _ = off.prepare_strategy_search_plan(&enc, &r, &g, &b, 1.0);

    let on: LossyEncoder<Backend> =
        LossyEncoder::new(&enc, W, H).with_auto_libjxl_entropy_mul_on_photos(true);
    let _ = on.prepare_strategy_search_plan(&enc, &r, &g, &b, 1.0);
}
