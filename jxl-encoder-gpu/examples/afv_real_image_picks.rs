// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chunk 4 validation: report AFV pick distribution on real
//! diagonal-frequency CLIC images, comparing default-off vs
//! `with_evaluate_afv(true)` under the new libjxl-faithful per-block
//! cost formula (chunks 2a + 2b).
//!
//! Expected behavior:
//! - default-off: 0 AFV picks (corpus_regression invariant).
//! - opt-in (true): AFV picks are now possible — the picker compares
//!   AFV costs on the same scale as DCT8 / DCT4x4 / etc. natively.
//!   On `a365e654...` (CLIC's diagonal-frequency lockup image,
//!   chunk lineage's reference), some 8x8 blocks should pick AFV
//!   where the diagonal cuts a 4×4 corner.
//!
//! Usage:
//!   cargo run --release --features cuda --example afv_real_image_picks
//!
//! Optional env vars:
//!   CORPUS_IMAGE  Path to PNG (default:
//!     /home/lilith/work/codec-corpus/clic2025-1024/
//!       a365e6541bab5c0f4e01bf43a0c3a655d88292a8ac45403a889c308d11854555.png)
//!   DISTANCE      Target distance (default: 1.0)

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(all(not(feature = "cuda"), not(feature = "wgpu"), feature = "cpu"))]
type Backend = cubecl::cpu::CpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "cpu")))]
fn main() {
    eprintln!("enable one of: --features cuda | wgpu | cpu");
    std::process::exit(2);
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::transform::{
        RAW_STRATEGY_AFV0, RAW_STRATEGY_AFV1, RAW_STRATEGY_AFV2, RAW_STRATEGY_AFV3,
        RAW_STRATEGY_DCT, RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8,
        RAW_STRATEGY_DCT8X4, RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16,
        RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64,
        RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64, RAW_STRATEGY_IDENTITY,
        RAW_STRATEGY_DCT16X32,
    };
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

    let image_path = std::env::var("CORPUS_IMAGE").unwrap_or_else(|_| {
        "/home/lilith/work/codec-corpus/clic2025-1024/\
        a365e6541bab5c0f4e01bf43a0c3a655d88292a8ac45403a889c308d11854555.png"
            .to_string()
    });
    let distance: f32 = std::env::var("DISTANCE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    println!("[afv-real-pick-diff] image: {image_path}");
    println!("[afv-real-pick-diff] distance: {distance}");
    if !std::path::Path::new(&image_path).exists() {
        eprintln!("ERROR: image not found at {image_path}");
        std::process::exit(2);
    }

    // Load image as RGB8 → linear f32 (sRGB inverse EOTF).
    let img = image::open(&image_path).expect("image open").to_rgb8();
    let (w, h) = img.dimensions();
    let pixels: Vec<u8> = img.into_raw();
    let n = (w * h) as usize;
    let to_linear = |c: u8| -> f32 {
        let f = c as f32 / 255.0;
        if f <= 0.04045 {
            f / 12.92
        } else {
            ((f + 0.055) / 1.055).powf(2.4)
        }
    };
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for chunk in pixels.chunks_exact(3) {
        r.push(to_linear(chunk[0]));
        g.push(to_linear(chunk[1]));
        b.push(to_linear(chunk[2]));
    }
    println!("[afv-real-pick-diff] image dims: {w}×{h} ({} blocks 8x8)", n / 64);

    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let label_for = |s: u8| -> &'static str {
        match s {
            RAW_STRATEGY_DCT => "DCT8",
            RAW_STRATEGY_DCT4X4 => "DCT4x4",
            RAW_STRATEGY_DCT4X8 => "DCT4x8",
            RAW_STRATEGY_DCT8X4 => "DCT8x4",
            RAW_STRATEGY_DCT16X8 => "DCT16x8",
            RAW_STRATEGY_DCT8X16 => "DCT8x16",
            RAW_STRATEGY_DCT16X16 => "DCT16x16",
            RAW_STRATEGY_DCT32X32 => "DCT32x32",
            RAW_STRATEGY_DCT32X16 => "DCT32x16",
            RAW_STRATEGY_DCT16X32 => "DCT16x32",
            RAW_STRATEGY_DCT64X64 => "DCT64x64",
            RAW_STRATEGY_DCT64X32 => "DCT64x32",
            RAW_STRATEGY_DCT32X64 => "DCT32x64",
            RAW_STRATEGY_DCT2X2 => "DCT2x2",
            RAW_STRATEGY_IDENTITY => "IDENTITY",
            RAW_STRATEGY_AFV0 => "AFV0",
            RAW_STRATEGY_AFV1 => "AFV1",
            RAW_STRATEGY_AFV2 => "AFV2",
            RAW_STRATEGY_AFV3 => "AFV3",
            _ => "??",
        }
    };

    let run = |afv_on: bool| -> (std::collections::BTreeMap<u8, usize>, f64, std::time::Duration) {
        let lossy: LossyEncoder<Backend> =
            LossyEncoder::new(&enc, w, h).with_evaluate_afv(afv_on);
        // Warm up once to amortize first-call upload/kernel costs.
        let _ = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
        let t0 = std::time::Instant::now();
        let plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
        let dt = t0.elapsed();
        let mut counts = std::collections::BTreeMap::<u8, usize>::new();
        for a in &plan.assignments {
            *counts.entry(a.raw_strategy).or_insert(0) += 1;
        }
        let total = plan.assignments.len() as f64;
        (counts, total, dt)
    };

    println!("\n========== AFV OFF (production default) ==========");
    let (counts_off, total_off, dt_off) = run(false);
    println!("[afv-real-pick-diff] prepare time: {:.1} ms", dt_off.as_secs_f64() * 1000.0);
    println!("[afv-real-pick-diff] total assignments: {total_off}");
    for (k, n) in &counts_off {
        let pct = 100.0 * (*n as f64) / total_off;
        println!("  {} : {} ({:.1}%)", label_for(*k), n, pct);
    }
    let n_afv_off = counts_off
        .iter()
        .filter(|(k, _)| {
            **k == RAW_STRATEGY_AFV0
                || **k == RAW_STRATEGY_AFV1
                || **k == RAW_STRATEGY_AFV2
                || **k == RAW_STRATEGY_AFV3
        })
        .map(|(_, n)| n)
        .sum::<usize>();
    println!("[afv-real-pick-diff] AFV picks (default-off): {n_afv_off}");

    println!("\n========== AFV ON (with libjxl-faithful formula) ==========");
    let (counts_on, total_on, dt_on) = run(true);
    println!("[afv-real-pick-diff] prepare time: {:.1} ms", dt_on.as_secs_f64() * 1000.0);
    println!("[afv-real-pick-diff] total assignments: {total_on}");
    for (k, n) in &counts_on {
        let pct = 100.0 * (*n as f64) / total_on;
        println!("  {} : {} ({:.1}%)", label_for(*k), n, pct);
    }
    let n_afv_on = counts_on
        .iter()
        .filter(|(k, _)| {
            **k == RAW_STRATEGY_AFV0
                || **k == RAW_STRATEGY_AFV1
                || **k == RAW_STRATEGY_AFV2
                || **k == RAW_STRATEGY_AFV3
        })
        .map(|(_, n)| n)
        .sum::<usize>();
    println!("[afv-real-pick-diff] AFV picks (opt-in): {n_afv_on}");

    println!("\n========== Summary ==========");
    let dt_diff = dt_on.as_secs_f64() * 1000.0 - dt_off.as_secs_f64() * 1000.0;
    println!("[afv-real-pick-diff] AFV cost grid wall-clock added: {dt_diff:.1} ms");
    println!("[afv-real-pick-diff] AFV picks: {n_afv_off} → {n_afv_on}");
    if n_afv_on > 0 {
        println!(
            "[afv-real-pick-diff] AFV pick fraction: {:.2}% of blocks",
            100.0 * (n_afv_on as f64) / total_on
        );
    }

    // Layer-2 invariants
    assert_eq!(
        n_afv_off, 0,
        "default-off must NEVER produce AFV picks (got {n_afv_off}, expected 0)"
    );
    println!("\n[invariant] default-off produces 0 AFV picks: PASS");
    if total_off as i64 == total_on as i64 {
        println!("[invariant] assignment-count parity (off vs on): PASS");
    } else {
        println!(
            "[invariant] WARN: assignment-count diff off={} on={}",
            total_off, total_on
        );
    }
}
