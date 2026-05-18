// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! A/B byte-size + wall-clock comparison for the patches-absence gate
//! on the auto-AFV cost-grid evaluation (W11-2 follow-on, 2026-05-18).
//!
//! Encodes each image with `auto_skip_afv_when_patches` **OFF**
//! (baseline — pre-2026-05-18 W7-3 unconditional auto-AFV) and **ON**
//! (the new default — skip AFV cost grid when patches detection fires
//! on the same image). Reports byte counts (must be identical),
//! per-image wall-clock (`prepare_strategy_search_plan` only, since
//! that's where the AFV cost grid lives), and the wall-clock delta.
//!
//! Why bytes-identical: the W11-2 diagnostic confirmed that the GPU
//! AFV picks made by the W7-3 auto-AFV dispatch are wiped on patches-
//! fired screenshots — the slow-path patches case-1 in `encoder.rs`
//! runs an independent CPU `compute_ac_strategy` on patches-subtracted
//! XYB that produces 5-10× more AFV picks (terminal: 40 → 214,
//! windows: 264 → 449). The GPU's AFV contribution to those bitstreams
//! is 0 bytes. Skipping the AFV cost grid only saves wall-clock; it
//! produces byte-identical output.
//!
//! Expected outcome on the 3-screenshot + 3-photo default set at d=1.0:
//! - Photos: gate stays off (auto-AFV doesn't fire on photo content
//!   anyway — `median(mask1x1) < 95`); byte- and time-identical.
//! - Screenshots that fire patches (terminal, windows, imac_g3):
//!   bytes identical, `prepare_strategy_search_plan` faster by the
//!   AFV-cost-grid wall-clock (~100 ms / 5 MP).
//! - Screenshots that do NOT fire patches (gmessages, graph, gui):
//!   bytes identical, time identical to W7-3 baseline (AFV cost
//!   grid still runs, since patches pre-check returned None).
//!
//! Usage:
//!   cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder' \
//!     --example afv_gate_by_patches_ab -- [--distance D] [--image PATH]...

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
        let base = "/home/lilith/work/codec-corpus";
        let candidates = [
            // Screenshots — patches fire on these (terminal, windows,
            // imac_g3 per W11-2 diagnostic; codec_wiki/imac_dark/
            // imessage/windows95 also fire but are larger).
            "gb82-sc/terminal.png",
            "gb82-sc/windows.png",
            "gb82-sc/imac_g3.png",
            // Screenshots — patches do NOT fire (gmessages, graph, gui
            // per W11-2 diagnostic). Auto-AFV gate fires here and gives
            // the actual byte wins under W7-3.
            "gb82-sc/gmessages.png",
            "gb82-sc/graph.png",
            "gb82-sc/gui.png",
            // Photos — auto-AFV gate never fires; gate is a no-op.
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
        "[afv_gate_by_patches_ab] distance={} images={} (skip-when-patches OFF baseline vs ON)",
        distance,
        images.len()
    );
    println!(
        "{:<60} {:>5} {:>12} {:>12} {:>7} {:>5} {:>5} {:>9} {:>9} {:>8}",
        "image",
        "MP",
        "off_bytes",
        "on_bytes",
        "Δ_bytes",
        "afv_o",
        "afv_n",
        "off_ms",
        "on_ms",
        "Δ_ms"
    );

    let mut total_off: u64 = 0;
    let mut total_on: u64 = 0;
    let mut total_pixels: u64 = 0;
    let mut total_off_ms: f64 = 0.0;
    let mut total_on_ms: f64 = 0.0;

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

        // Helper: time prepare_strategy_search_plan + count AFV picks.
        // The AFV cost grid runs inside prepare_strategy_search_plan —
        // that's the only place the wall-clock delta lives. Calling
        // the full bitstream encode would dominate with downstream
        // entropy coding, masking the small AFV savings.
        let prepare_time_and_afv = |lossy: &LossyEncoder<B>| -> (f64, usize) {
            use jxl_encoder_gpu::forks::transform::{
                RAW_STRATEGY_AFV0, RAW_STRATEGY_AFV1, RAW_STRATEGY_AFV2, RAW_STRATEGY_AFV3,
            };
            // Warm pass to amortize first-encode setup (kernel compile, etc.).
            let _ = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
            // Measured pass.
            let t0 = Instant::now();
            let plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let afv = plan
                .assignments
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
                .count();
            (ms, afv)
        };

        // Path A: baseline (skip-when-patches OFF).
        let lossy_off: LossyEncoder<B> =
            LossyEncoder::new(&enc, w, h).with_auto_skip_afv_when_patches(false);
        assert!(!lossy_off.auto_skip_afv_when_patches());
        let (off_ms, afv_off) = prepare_time_and_afv(&lossy_off);
        let bs_off = enc
            .encode_lossy_to_bitstream_via_precomputed(&lossy_off, &r, &g, &b, distance)
            .unwrap_or_else(|e| panic!("off-path encode failed for {path}: {e:?}"));

        // Path B: new default (skip-when-patches ON).
        let lossy_on: LossyEncoder<B> = LossyEncoder::new(&enc, w, h);
        assert!(lossy_on.auto_skip_afv_when_patches());
        let (on_ms, afv_on) = prepare_time_and_afv(&lossy_on);
        let bs_on = enc
            .encode_lossy_to_bitstream_via_precomputed(&lossy_on, &r, &g, &b, distance)
            .unwrap_or_else(|e| panic!("on-path encode failed for {path}: {e:?}"));

        let off = bs_off.len() as i64;
        let on = bs_on.len() as i64;
        let dbytes = on - off;
        let dms = on_ms - off_ms;
        let short = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        let mp = n as f32 / 1e6;
        println!(
            "{:<60} {:>5.2} {:>12} {:>12} {:>+7} {:>5} {:>5} {:>9.1} {:>9.1} {:>+7.1}",
            short, mp, off, on, dbytes, afv_off, afv_on, off_ms, on_ms, dms
        );

        total_off += off as u64;
        total_on += on as u64;
        total_pixels += n as u64;
        total_off_ms += off_ms;
        total_on_ms += on_ms;
    }

    println!();
    let total_dbytes = total_on as i64 - total_off as i64;
    let total_dms = total_on_ms - total_off_ms;
    println!(
        "TOTAL (n={} images, {:.2} MP): bytes {} → {} ({:+}); prepare ms {:.1} → {:.1} ({:+.1})",
        images.len(),
        total_pixels as f32 / 1e6,
        total_off,
        total_on,
        total_dbytes,
        total_off_ms,
        total_on_ms,
        total_dms,
    );
    if total_dbytes != 0 {
        eprintln!(
            "WARNING: bytes differ ({:+} total) — the gate must be byte-identical",
            total_dbytes
        );
    }
}
