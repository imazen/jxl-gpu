// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! A/B byte-size comparison for the auto-AFV-on-screenshots dispatch
//! introduced in 2026-05-17. Encodes each image with auto-AFV **OFF**
//! (baseline — `with_auto_evaluate_afv_on_screenshots(false)`) and
//! auto-AFV **ON** (the new default), reports byte counts and
//! `(auto_on - auto_off) / auto_off` percentage delta.
//!
//! Drives [`GpuEncoder::encode_lossy_to_bitstream_via_precomputed`]
//! which exercises the [`LossyEncoder::prepare_strategy_search_plan`]
//! pipeline where the auto-AFV gate fires (the
//! `refine_and_encode_smart` path skips strat-search for screenshots
//! entirely and would not exercise the dispatch).
//!
//! Expected outcomes per the conditional-resurrection audit
//! (2026-05-17):
//! - Screenshots (mask1x1 median > 95, effort >= 7): auto-on wins by
//!   ~0.3-0.5% bytes (small but real, content-dependent).
//! - Photos: auto-on byte-identical to auto-off (gate never fires;
//!   median is well under 95 on photo content).
//!
//! Usage:
//!   cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder' \
//!     --example auto_afv_bytes_ab -- [--distance D] [--image PATH]...
//!
//! With no `--image` args, uses a default 3-screenshot + 3-photo
//! corpus subset to match the audit's expected validation.

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
    let mut images: Vec<String> = Vec::new();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--distance" => {
                distance = raw[i + 1].parse().expect("--distance D");
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
        // Default validation set: 3 screenshots + 3 photos from the
        // standard `codec-corpus` tree.
        let base = "/home/lilith/work/codec-corpus";
        let candidates = [
            // Screenshots — auto-AFV gate should fire.
            "gb82-sc/terminal.png",
            "gb82-sc/imac_g3.png",
            "gb82-sc/windows95.png",
            // Photos — auto-AFV gate should NOT fire (byte-identical).
            "clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
            "clic2025-1024/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png",
            "clic2025-1024/22ea12c903e41583b7c469cb86040157.png",
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
        "[auto_afv_bytes_ab] distance={} images={} (auto-AFV OFF baseline vs ON)",
        distance,
        images.len()
    );
    println!(
        "{:<60} {:>5} {:>12} {:>12} {:>9} {:>8} {:>5} {:>5} {:>8}",
        "image", "MP", "off_bytes", "on_bytes", "Δ_bytes", "Δ_pct", "afv_o", "afv_n", "ms_on"
    );

    let mut total_off: u64 = 0;
    let mut total_on: u64 = 0;
    let mut total_pixels: u64 = 0;

    for path in &images {
        // Load image → planar linear f32.
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

        // Helper: count AFV picks in a strat-search plan for an AB pair.
        let count_afv = |lossy: &LossyEncoder<B>| -> usize {
            use jxl_encoder_gpu::forks::transform::{
                RAW_STRATEGY_AFV0, RAW_STRATEGY_AFV1, RAW_STRATEGY_AFV2, RAW_STRATEGY_AFV3,
            };
            let plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
            plan.assignments
                .iter()
                .filter(|a| {
                    matches!(
                        a.raw_strategy,
                        RAW_STRATEGY_AFV0
                            | RAW_STRATEGY_AFV1
                            | RAW_STRATEGY_AFV2
                            | RAW_STRATEGY_AFV3
                    )
                })
                .count()
        };

        // Path A: auto-AFV OFF (baseline; reproduces pre-2026-05-17 behavior).
        let lossy_off: LossyEncoder<B> =
            LossyEncoder::new(&enc, w, h).with_auto_evaluate_afv_on_screenshots(false);
        let afv_off = count_afv(&lossy_off);
        let bs_off = enc
            .encode_lossy_to_bitstream_via_precomputed(&lossy_off, &r, &g, &b, distance)
            .unwrap_or_else(|e| panic!("off-path encode failed for {path}: {e:?}"));

        // Path B: auto-AFV ON (the new default).
        let lossy_on: LossyEncoder<B> = LossyEncoder::new(&enc, w, h);
        // Sanity: the new default is on.
        assert!(
            lossy_on.auto_evaluate_afv_on_screenshots(),
            "new default must be auto-AFV ON"
        );
        let afv_on = count_afv(&lossy_on);
        let t0 = Instant::now();
        let bs_on = enc
            .encode_lossy_to_bitstream_via_precomputed(&lossy_on, &r, &g, &b, distance)
            .unwrap_or_else(|e| panic!("on-path encode failed for {path}: {e:?}"));
        let ms_on = t0.elapsed().as_secs_f64() * 1000.0;

        let off = bs_off.len() as i64;
        let on = bs_on.len() as i64;
        let dbytes = on - off;
        let dpct = (dbytes as f64) / (off as f64) * 100.0;
        let short = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        let mp = n as f32 / 1e6;
        println!(
            "{:<60} {:>5.2} {:>12} {:>12} {:>+9} {:>+7.3}% {:>5} {:>5} {:>7.1}",
            short, mp, off, on, dbytes, dpct, afv_off, afv_on, ms_on
        );

        total_off += off as u64;
        total_on += on as u64;
        total_pixels += n as u64;
    }

    println!();
    let total_dbytes = total_on as i64 - total_off as i64;
    let total_dpct = (total_dbytes as f64) / (total_off as f64) * 100.0;
    println!(
        "TOTAL (n={} images, {:.2} MP): {} → {} bytes ({:+} bytes, {:+.3}%)",
        images.len(),
        total_pixels as f32 / 1e6,
        total_off,
        total_on,
        total_dbytes,
        total_dpct,
    );
}
