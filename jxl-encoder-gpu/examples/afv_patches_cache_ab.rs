// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! A/B wall-clock comparison for the W12-2 chunk-2 follow-on
//! (2026-05-18): plumb the `PatchesData` from
//! `prepare_strategy_search_plan_inner`'s auto-AFV gate pre-check
//! through `StrategySearchPlan` so the encoder slow path's case-1
//! reuses it instead of re-running `find_and_build_patches` on the
//! same pre-gaborish XYB.
//!
//! Why bytes-identical: the cache holds the SAME `PatchesData` the
//! encoder.rs case-1 would have built on its own. `take()`-then-use
//! is observationally indistinguishable from recompute-then-use
//! (the only difference is the wall-clock of one BFS L1-distance
//! text-like-patch search).
//!
//! Methodology:
//! - Path A (baseline): `with_auto_skip_afv_when_patches(false)`.
//!   The W12-2 gate does NOT run, so `plan.patches_data_cache` stays
//!   `None`, and the encoder slow path always recomputes patches.
//!   This is the **pre-chunk-2** behaviour as well: prior to this
//!   commit the gate threw away its `PatchesData` after the boolean
//!   `is_some()` check, so even with the gate ON the encoder still
//!   recomputed. Path A here mirrors that double-detection cost.
//! - Path B (new): default-on `with_auto_skip_afv_when_patches(true)`
//!   AND the W12-2 chunk-2 cache plumbing. The gate stores its
//!   `PatchesData` in the plan; the encoder slow path `take()`s it
//!   on the patches case-1 branch.
//!
//! Wall-clock measured is the FULL ENCODE (not just prepare) since
//! the chunk-2 savings live inside the encoder.rs patches case-1
//! block, not the strat-search plan stage.
//!
//! Expected outcome on the 3+3 default set at d=1.0:
//! - Photos: gate never fires; cache stays `None`; encoder.rs
//!   recomputes anyway. Time-identical to Path A.
//! - Patches-fired screenshots (terminal, windows, imac_g3): gate
//!   fires and finds patches; cache is populated. Path B saves the
//!   second `find_and_build_patches` call in encoder.rs (BFS + L1)
//!   AND keeps the W12-2 AFV cost-grid skip.
//! - Patches-NOT-fired screenshots (gmessages, graph, gui): gate
//!   fires but finds no patches; cache is `Some(None)`. Path B
//!   saves the encoder.rs's redundant `find_and_build_patches` call
//!   that would otherwise return None too. This is the "recover the
//!   +125-170 ms cost from W12-2" win.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder' \
//!     --example afv_patches_cache_ab -- [--distance D] [--iters N] [--image PATH]...

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
}

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;
    use std::time::Instant;

    type B = cubecl::cuda::CudaRuntime;

    // Parse args.
    let raw: Vec<String> = std::env::args().collect();
    let mut distance: f32 = 1.0;
    let mut iters: usize = 3;
    let mut images: Vec<String> = Vec::new();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--distance" => {
                distance = raw[i + 1].parse().expect("--distance D");
                i += 2;
            }
            "--iters" => {
                iters = raw[i + 1].parse().expect("--iters N");
                i += 2;
            }
            "--image" => {
                images.push(raw[i + 1].clone());
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    if images.is_empty() {
        let base = "/home/lilith/work/codec-corpus";
        let candidates = [
            // Patches-fired screenshots (per W11-2 diagnostic):
            "gb82-sc/terminal.png",
            "gb82-sc/windows.png",
            "gb82-sc/imac_g3.png",
            // Patches-NOT-fired screenshots:
            "gb82-sc/gmessages.png",
            "gb82-sc/graph.png",
            "gb82-sc/gui.png",
        ];
        for c in &candidates {
            let path = format!("{base}/{c}");
            if std::path::Path::new(&path).exists() {
                images.push(path);
            } else {
                eprintln!("[skip] {c} not found in {base}");
            }
        }
        if images.is_empty() {
            panic!("no default images found and no --image given");
        }
    }

    println!(
        "[afv_patches_cache_ab] distance={} iters={} images={} (path A: skip-off baseline; path B: skip-on + cache)",
        distance,
        iters,
        images.len()
    );
    println!(
        "{:<60} {:>5} {:>12} {:>12} {:>7} {:>9} {:>9} {:>+9}",
        "image", "MP", "A_bytes", "B_bytes", "Δ_bytes", "A_ms", "B_ms", "Δ_ms"
    );

    let mut total_a_bytes: u64 = 0;
    let mut total_b_bytes: u64 = 0;
    let mut total_pixels: u64 = 0;
    let mut total_a_ms: f64 = 0.0;
    let mut total_b_ms: f64 = 0.0;

    for path in &images {
        let img = match image::open(path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                eprintln!("[skip] {path}: {e}");
                continue;
            }
        };
        let (w, h) = img.dimensions();
        let n = (w * h) as usize;
        let pixels_u8 = img.into_raw();
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
        for chunk in pixels_u8.chunks_exact(3) {
            r.push(to_linear(chunk[0]));
            g.push(to_linear(chunk[1]));
            b.push(to_linear(chunk[2]));
        }

        let enc: GpuEncoder<B> = GpuEncoder::new();

        // Helper: full-encode wall-clock + final byte count. Warms
        // once (kernel compile / pool prime), then takes the min of
        // `iters` measured runs to filter out one-off scheduler
        // hiccups (the chunk-2 savings are 10-50ms in absolute terms
        // — too tight to average over noisy single runs).
        let full_encode_min_ms = |lossy: &LossyEncoder<B>| -> (f64, usize) {
            // Warm.
            let warm = enc
                .encode_lossy_to_bitstream_via_precomputed(lossy, &r, &g, &b, distance)
                .unwrap();
            let bytes = warm.len();
            let mut best_ms = f64::INFINITY;
            for _ in 0..iters {
                let t0 = Instant::now();
                let bs = enc
                    .encode_lossy_to_bitstream_via_precomputed(lossy, &r, &g, &b, distance)
                    .unwrap();
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(
                    bs.len(),
                    bytes,
                    "non-deterministic byte count between iters"
                );
                if ms < best_ms {
                    best_ms = ms;
                }
            }
            (best_ms, bytes)
        };

        // Path A: baseline (skip-when-patches OFF — no cache, no gate).
        let lossy_a: LossyEncoder<B> =
            LossyEncoder::new(&enc, w, h).with_auto_skip_afv_when_patches(false);
        assert!(!lossy_a.auto_skip_afv_when_patches());
        let (a_ms, a_bytes) = full_encode_min_ms(&lossy_a);

        // Path B: new default (skip-when-patches ON + chunk-2 cache).
        let lossy_b: LossyEncoder<B> = LossyEncoder::new(&enc, w, h);
        assert!(lossy_b.auto_skip_afv_when_patches());
        let (b_ms, b_bytes) = full_encode_min_ms(&lossy_b);

        let a = a_bytes as i64;
        let b_b = b_bytes as i64;
        let dbytes = b_b - a;
        let dms = b_ms - a_ms;
        let short = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        let mp = n as f32 / 1e6;
        println!(
            "{:<60} {:>5.2} {:>12} {:>12} {:>+7} {:>9.1} {:>9.1} {:>+9.1}",
            short, mp, a, b_b, dbytes, a_ms, b_ms, dms
        );

        total_a_bytes += a as u64;
        total_b_bytes += b_b as u64;
        total_pixels += n as u64;
        total_a_ms += a_ms;
        total_b_ms += b_ms;
    }

    println!();
    let total_dbytes = total_b_bytes as i64 - total_a_bytes as i64;
    let total_dms = total_b_ms - total_a_ms;
    println!(
        "TOTAL (n={} images, {:.2} MP): bytes {} → {} ({:+}); full-encode ms {:.1} → {:.1} ({:+.1})",
        images.len(),
        total_pixels as f32 / 1e6,
        total_a_bytes,
        total_b_bytes,
        total_dbytes,
        total_a_ms,
        total_b_ms,
        total_dms,
    );
    if total_dbytes != 0 {
        eprintln!(
            "WARNING: bytes differ ({:+} total) — the cache plumbing must be byte-identical",
            total_dbytes
        );
    }
}
