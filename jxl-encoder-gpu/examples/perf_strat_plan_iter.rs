// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-iter wall-clock measurement for the strat-search encode path.
//!
//! What this captures:
//! - prepare_strategy_search_plan one-shot cost
//! - encode_with_strategy_plan_adaptive_traced N iterations after a
//!   warmup, with per-stage timing via the `mark` callback.
//!
//! The N-iter loop simulates the butteraugli refinement loop's hot
//! path. Useful as a smoke benchmark to catch regressions in the
//! GPU-resident encode pipeline (the work behind commits a0489e58 →
//! 5127c040 — full per-strategy GPU chain + GpuPlane plumbing).
//!
//! Run with:
//!   cargo run -p jxl-encoder-gpu --features 'cuda encoder' \
//!     --release --example perf_strat_plan_iter [width [height [iters]]]
//!
//! Defaults: 1024 × 1024, 5 iters.
//!
//! Synthetic input only (gradient + noise) — no corpus dependency.
//! Reports wall-clock per stage so a future run on this commit's
//! baseline can be compared against later changes.

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("This example requires both `cuda` and `encoder` features.");
    eprintln!(
        "Run: cargo run -p jxl-encoder-gpu --features 'cuda encoder' --release --example perf_strat_plan_iter"
    );
}

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use std::time::Instant;

    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::{distance_to_qac, LossyEncoder};

    type B = cubecl::cuda::CudaRuntime;

    // Parse args: width height iters.
    let args: Vec<String> = std::env::args().collect();
    let width: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let height: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let distance: f32 = 1.0;

    println!(
        "perf_strat_plan_iter: {width}×{height}, {iters} iters at distance={distance}"
    );

    // Synthetic image: low-frequency gradient + per-pixel jitter (hash).
    let n = (width as usize) * (height as usize);
    let make_channel = |seed: u32| -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i as u32 % width) as f32 / width as f32;
            let y = (i as u32 / width) as f32 / height as f32;
            let jitter = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) as f32
                / u32::MAX as f32
                - 0.5)
                * 0.04;
            v.push((0.20 + 0.60 * x + 0.10 * y + jitter).clamp(0.0, 1.0));
        }
        v
    };
    let r = make_channel(11);
    let g = make_channel(23);
    let b = make_channel(37);

    let enc: GpuEncoder<B> = GpuEncoder::new();
    let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, width, height);
    let nb8 = (lossy.padded_dimensions().0 as usize / 8)
        * (lossy.padded_dimensions().1 as usize / 8);
    let aq_field = vec![distance_to_qac(distance); nb8];

    // Warmup: build a plan + run one encode to JIT/upload static buffers.
    let warm_plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
    let _ = lossy.encode_with_strategy_plan_adaptive(&enc, &warm_plan, &aq_field);
    drop(warm_plan);

    // ── prepare_strategy_search_plan timing ──
    let t0 = Instant::now();
    let plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
    let dt_prepare = t0.elapsed();
    println!("\nprepare_strategy_search_plan: {:.2} ms", dt_prepare.as_secs_f64() * 1000.0);

    // Plan-side strategy histogram (helps explain timings — DCT8-heavy
    // photos hit the 1×1-LLF GPU path, large-transform-heavy hit the
    // bigger LLF kernels).
    use std::collections::BTreeMap;
    let mut histo: BTreeMap<u8, usize> = BTreeMap::new();
    for a in &plan.assignments {
        *histo.entry(a.raw_strategy).or_insert(0) += 1;
    }
    let total_assignments: usize = histo.values().sum();
    println!("plan: {} strategy assignments", total_assignments);
    use jxl_encoder_gpu::forks::transform::*;
    let label = |s: u8| -> &'static str {
        match s {
            RAW_STRATEGY_DCT => "DCT8",
            RAW_STRATEGY_DCT16X8 => "DCT16x8",
            RAW_STRATEGY_DCT8X16 => "DCT8x16",
            RAW_STRATEGY_DCT16X16 => "DCT16x16",
            RAW_STRATEGY_DCT32X32 => "DCT32x32",
            RAW_STRATEGY_DCT4X8 => "DCT4x8",
            RAW_STRATEGY_DCT8X4 => "DCT8x4",
            RAW_STRATEGY_DCT4X4 => "DCT4x4",
            RAW_STRATEGY_DCT32X16 => "DCT32x16",
            RAW_STRATEGY_DCT16X32 => "DCT16x32",
            RAW_STRATEGY_DCT64X64 => "DCT64x64",
            RAW_STRATEGY_DCT64X32 => "DCT64x32",
            RAW_STRATEGY_DCT32X64 => "DCT32x64",
            RAW_STRATEGY_IDENTITY => "IDENT",
            RAW_STRATEGY_DCT2X2 => "DCT2x2",
            RAW_STRATEGY_AFV0 => "AFV0",
            RAW_STRATEGY_AFV1 => "AFV1",
            RAW_STRATEGY_AFV2 => "AFV2",
            RAW_STRATEGY_AFV3 => "AFV3",
            _ => "??",
        }
    };
    for (s, n) in &histo {
        println!(
            "  {:8} = {:5} ({:.1}%)",
            label(*s),
            n,
            100.0 * (*n as f64) / (total_assignments as f64)
        );
    }

    // ── encode_with_strategy_plan_adaptive_traced N iters with marks ──
    println!("\nencode_with_strategy_plan_adaptive_traced × {iters}:");
    let mut iter_times = Vec::with_capacity(iters);
    let mut stage_totals: BTreeMap<&'static str, f64> = BTreeMap::new();
    for it in 0..iters {
        let mut last = Instant::now();
        let t_iter = Instant::now();
        let _ = lossy.encode_with_strategy_plan_adaptive_traced(
            &enc,
            &plan,
            &aq_field,
            &mut |label: &'static str| {
                let now = Instant::now();
                let elapsed = now.duration_since(last).as_secs_f64() * 1000.0;
                *stage_totals.entry(label).or_insert(0.0) += elapsed;
                last = now;
            },
        );
        let dt = t_iter.elapsed();
        iter_times.push(dt);
        println!("  iter {}: {:.2} ms", it, dt.as_secs_f64() * 1000.0);
    }

    // ── Summary ──
    let total_ms: f64 = iter_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum();
    let mean_ms = total_ms / iters as f64;
    let min_ms = iter_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .fold(f64::INFINITY, f64::min);
    println!("\nencode-with-plan summary: mean {:.2} ms, min {:.2} ms (over {iters} iters)", mean_ms, min_ms);

    println!("\nstage breakdown (mean across {iters} iters, sorted by total):");
    let mut entries: Vec<(&&'static str, &f64)> = stage_totals.iter().collect();
    entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (label, total_ms) in entries {
        let mean = total_ms / iters as f64;
        println!("  {:32} {:7.2} ms/iter", label, mean);
    }
}
