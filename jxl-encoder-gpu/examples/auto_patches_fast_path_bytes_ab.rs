// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! A/B byte-size comparison for the auto-patches-on-fast-path
//! dispatch introduced in 2026-05-17. Encodes each image with
//! auto-patches **OFF**
//! (baseline — `with_auto_patches_on_fast_path(false)`) and
//! auto-patches **ON** (the new default), reports byte counts and
//! `(on - off) / off` percentage delta.
//!
//! Drives
//! [`GpuEncoder::encode_lossy_to_bitstream_via_precomputed_from_u8`]
//! — the u8 entry where the GPU pre-quantized AC fast path lives
//! (the f32 entry has no fast path).
//!
//! When the all-DCT8 fast path is selected on a small
//! (`< 1_000_000` pixels) screenshot-like image (median per-block
//! mask1x1 > 95) at effort >= 5, the gate forces the slow path,
//! which runs `find_and_build_patches` and emits a patches reference
//! frame. The fast path entry point ignores
//! `EncoderPrecomputed::patches_data` (jxl-encoder
//! `vardct/encoder.rs:2602` hardcodes `None`), so the only way to
//! ship patches on an all-DCT8 image is to fall through.
//!
//! Expected outcomes per the conditional-resurrection audit
//! (2026-05-17, item #5):
//! - Small screenshots with all-DCT8 picks: auto-on can save
//!   30-50% bytes when patches actually detect.
//! - Photos: byte-identical (median mask1x1 well under 95).
//!
//! Usage:
//!   cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder' \
//!     --example auto_patches_fast_path_bytes_ab -- \
//!     [--distance D] [--image PATH]...
//!
//! With no `--image` args, uses a default 3-screenshot + 3-photo
//! corpus subset to match the audit's expected validation. Defaults
//! to distance 0.5 (the regime where all-DCT8 selection is most
//! likely on screenshot content).

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

    let raw: Vec<String> = std::env::args().collect();
    let mut distance: f32 = 0.5;
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
            // Screenshots — auto-patches gate may fire when all-DCT8
            // picks AND pixel_count < 1MP AND median(mask1x1) > 95.
            // Sub-MP screenshots from the standard corpus:
            "gb82-sc/windows95.png", // 640x480 = 307 200 px
            "gb82-sc/graph.png",     // 796x481 = 382 876 px
            // 1.75 MP — over the pixel-count gate; included for the
            // negative case (gate cannot fire even if strategy picks
            // happen to be all-DCT8).
            "gb82-sc/terminal.png",
            // Photos — gate must NOT fire (median mask1x1 ≤ 87 on
            // CLIC photos per audit). Photos here are 1.05 MP which
            // alone disqualifies the gate (>= 1_000_000), but the
            // median check is the load-bearing photo guard if the
            // pixel-count gate is ever relaxed.
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
        "[auto_patches_fast_path_bytes_ab] distance={} images={} \
         (auto-patches OFF baseline vs ON)",
        distance,
        images.len()
    );
    println!(
        "{:<60} {:>5} {:>12} {:>12} {:>9} {:>8} {:>8}",
        "image", "MP", "off_bytes", "on_bytes", "Δ_bytes", "Δ_pct", "ms_on"
    );

    let mut total_off: u64 = 0;
    let mut total_on: u64 = 0;
    let mut total_pixels: u64 = 0;

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

        // The u8 entry takes interleaved sRGB R G B u8 — exactly
        // what `image::open(...).to_rgb8().into_raw()` produces.
        let enc: GpuEncoder<B> = GpuEncoder::new();

        // Path A: auto-patches OFF (baseline; pre-2026-05-17 behavior).
        let lossy_off: LossyEncoder<B> =
            LossyEncoder::new(&enc, w, h).with_auto_patches_on_fast_path(false);
        let bs_off = enc
            .encode_lossy_to_bitstream_via_precomputed_from_u8(&lossy_off, &pixels_u8, distance)
            .unwrap_or_else(|e| panic!("off-path encode failed for {path}: {e:?}"));

        // Path B: auto-patches ON (new default).
        let lossy_on: LossyEncoder<B> = LossyEncoder::new(&enc, w, h);
        assert!(
            lossy_on.auto_patches_on_fast_path(),
            "new default must be auto-patches ON"
        );
        let t0 = Instant::now();
        let bs_on = enc
            .encode_lossy_to_bitstream_via_precomputed_from_u8(&lossy_on, &pixels_u8, distance)
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
            "{:<60} {:>5.2} {:>12} {:>12} {:>+9} {:>+7.3}% {:>7.1}",
            short, mp, off, on, dbytes, dpct, ms_on
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
