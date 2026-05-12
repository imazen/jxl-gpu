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
//!   cargo run -p jxl-encoder-gpu --features 'cuda encoder' \
//!     --release --example perf_strat_plan_iter --image PATH [iters]
//!
//! Defaults: 1024 × 1024 synthetic gradient, 5 iters.
//!
//! With `--image PATH`, decodes the PNG/JPEG/etc and converts to
//! linear RGB; dimensions come from the image. Use a real CLIC photo
//! to get realistic strategy distributions (synthetic gradients hit
//! >50% DCT64x64; real photos are DCT8-heavy).
//!
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
    use std::collections::BTreeMap;
    use std::time::Instant;

    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::{LossyEncoder, distance_to_qac};

    type B = cubecl::cuda::CudaRuntime;

    // Parse args: either positional [width [height [iters]]] or
    // --image PATH [iters].
    let raw_args: Vec<String> = std::env::args().collect();
    let distance: f32 = 1.0;

    let (width, height, r, g, b, source_label): (u32, u32, Vec<f32>, Vec<f32>, Vec<f32>, String);
    let iters: usize;

    if raw_args.len() >= 3 && raw_args[1] == "--image" {
        let path = &raw_args[2];
        // Optional --target-mp M after the path: upscale (Triangle) or
        // downscale (Lanczos3) to that megapixel target. Useful for
        // testing the same source at multiple sizes without staging
        // intermediate PNGs.
        let mut target_mp: Option<f32> = None;
        let mut iter_arg_pos = 3;
        if raw_args.get(3).map(|s| s.as_str()) == Some("--target-mp") {
            target_mp = raw_args.get(4).and_then(|s| s.parse().ok());
            iter_arg_pos = 5;
        }
        iters = raw_args
            .get(iter_arg_pos)
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let mut img = image::open(path)
            .unwrap_or_else(|e| panic!("failed to open {path}: {e}"))
            .to_rgb8();
        if let Some(mp) = target_mp {
            let (sw, sh) = img.dimensions();
            let cur_mp = (sw as f32 * sh as f32) / 1_000_000.0;
            let scale = (mp / cur_mp).sqrt();
            let nw = ((sw as f32 * scale).round() as u32).max(8);
            let nh = ((sh as f32 * scale).round() as u32).max(8);
            let filter = if nw * nh < (sw * sh) {
                image::imageops::FilterType::Lanczos3
            } else {
                image::imageops::FilterType::Triangle
            };
            img = image::imageops::resize(&img, nw, nh, filter);
        }
        let (w, h) = img.dimensions();
        width = w;
        height = h;
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
        let mut rr = Vec::with_capacity(n);
        let mut gg = Vec::with_capacity(n);
        let mut bb = Vec::with_capacity(n);
        for chunk in pixels.chunks_exact(3) {
            rr.push(to_linear(chunk[0]));
            gg.push(to_linear(chunk[1]));
            bb.push(to_linear(chunk[2]));
        }
        r = rr;
        g = gg;
        b = bb;
        source_label = if let Some(mp) = target_mp {
            format!("image={path} (resized to {} MP)", mp)
        } else {
            format!("image={path}")
        };
    } else {
        width = raw_args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
        height = raw_args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
        iters = raw_args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
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
        r = make_channel(11);
        g = make_channel(23);
        b = make_channel(37);
        source_label = "synthetic gradient".to_string();
    }

    println!(
        "perf_strat_plan_iter: {width}×{height} ({source_label}), {iters} iters at distance={distance}"
    );

    let enc: GpuEncoder<B> = GpuEncoder::new();
    let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, width, height);
    let nb8 =
        (lossy.padded_dimensions().0 as usize / 8) * (lossy.padded_dimensions().1 as usize / 8);
    let aq_field = vec![distance_to_qac(distance); nb8];

    // Warmup: build a plan + run one encode to JIT/upload static buffers.
    let warm_plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
    let _ = lossy.encode_with_strategy_plan_adaptive(&enc, &warm_plan, &aq_field);
    drop(warm_plan);

    // ── prepare_strategy_search_plan timing ──
    // Run prepare 3 times back-to-back to detect cubecl pool-warmup
    // behavior at large image sizes (xyb_gab regression at 16+ MP
    // appears to be cudaMalloc per encode for ~64 MB output buffers).
    let mut prep_stages: BTreeMap<&'static str, f64> = BTreeMap::new();
    let mut prep_totals: Vec<f64> = Vec::new();
    let mut plan_holder: Option<_> = None;
    for prep_idx in 0..3 {
        let mut last_prep = Instant::now();
        let t0 = Instant::now();
        let mut local_stages: BTreeMap<&'static str, f64> = BTreeMap::new();
        let plan = lossy.prepare_strategy_search_plan_traced(
            &enc,
            &r,
            &g,
            &b,
            distance,
            &mut |label: &'static str| {
                let now = Instant::now();
                let elapsed = now.duration_since(last_prep).as_secs_f64() * 1000.0;
                *local_stages.entry(label).or_insert(0.0) += elapsed;
                last_prep = now;
            },
        );
        let dt_prepare = t0.elapsed().as_secs_f64() * 1000.0;
        prep_totals.push(dt_prepare);
        println!("\nprepare run {prep_idx}: {:.2} ms", dt_prepare);
        for (label, ms) in &local_stages {
            *prep_stages.entry(label).or_insert(0.0) += ms;
        }
        if prep_idx == 0 {
            // Show the first-run breakdown (worst case, before any pool warmup).
            let mut entries: Vec<_> = local_stages.iter().collect();
            entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            for (label, ms) in entries {
                println!("    {:32} {:7.2} ms", label, ms);
            }
        }
        if prep_idx == 2 {
            plan_holder = Some(plan);
        }
    }
    let plan = plan_holder.unwrap();
    let dt_prepare = std::time::Duration::from_secs_f64(prep_totals[2] / 1000.0);
    println!(
        "\nprepare_strategy_search_plan (last of 3): {:.2} ms (totals: {:?})",
        dt_prepare.as_secs_f64() * 1000.0,
        prep_totals
    );
    println!("  prepare stage breakdown (sorted by total):");
    let mut prep_entries: Vec<(&&'static str, &f64)> = prep_stages.iter().collect();
    prep_entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (label, ms) in prep_entries {
        println!("    {:32} {:7.2} ms", label, ms);
    }

    // Plan-side strategy histogram (helps explain timings — DCT8-heavy
    // photos hit the 1×1-LLF GPU path, large-transform-heavy hit the
    // bigger LLF kernels).
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
    let total_ms: f64 = iter_times.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
    let mean_ms = total_ms / iters as f64;
    let min_ms = iter_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .fold(f64::INFINITY, f64::min);
    println!(
        "\nencode-with-plan summary: mean {:.2} ms, min {:.2} ms (over {iters} iters)",
        mean_ms, min_ms
    );

    println!("\nstage breakdown (mean across {iters} iters, sorted by total):");
    let mut entries: Vec<(&&'static str, &f64)> = stage_totals.iter().collect();
    entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (label, total_ms) in entries {
        let mean = total_ms / iters as f64;
        println!("  {:32} {:7.2} ms/iter", label, mean);
    }
}
