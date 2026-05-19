// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chunk-3 re-verification: does the `auto_libjxl_entropy_mul_on_photos`
//! photo-branch dispatch (W8-5, commit `f7677ac4`) now stop regressing
//! once the chunks 1+2 counterweights are in place?
//!
//! ## Context
//!
//! W8-5 (commit `f7677ac4`, 2026-05-17 evening) added the opt-in
//! `with_auto_libjxl_entropy_mul_on_photos(true)` builder. When on,
//! photo-class images (`mask1x1` median < `SCREENSHOT_MEDIAN_MASK_THRESHOLD`)
//! get the libjxl-faithful per-strategy `entropy_mul`
//! (IDENTITY 1.0428, DCT4x8/DCT8x4 0.859316) and have `dist_bias`
//! disabled (= 1.0). An A/B at d=1.0 measured strictly Pareto-worse on
//! every axis (+2.5% to +8.5% bytes, +0.11 to +0.24 butteraugli, −0.17
//! to −0.42 SSIM2). The default was therefore left at `false`.
//!
//! The Pareto-loss root cause was that the GPU encoder lacked libjxl's
//! `kAvoidEntropyOfTransforms` heuristic and the X-channel multi-block
//! weight — counterweights that prevent over-pick of large transforms
//! when the lifted entropy_mul values are removed. Chunk 1
//! (commit `f5d3703`) ported `kAvoidEntropyOfTransforms` into the
//! sub-block (DCT4X4/DCT4X8/DCT8X4) cost path. Chunk 2 (commit
//! `dd4af71`) extended it into AFV0-3, and audited the X-channel
//! multi-block weight as already-applied for every multi-block strategy
//! via `per_block_upstream_cost_per_block`.
//!
//! ## What this harness does
//!
//! For each image × distance ∈ {0.5, 1.0, 2.0, 5.0}, encode twice:
//!
//! - **OFF**: production default. `auto_libjxl_entropy_mul_on_photos`
//!   off, `enable_kavoid_entropy_of_transforms` off, AFV explicitly on
//!   so the AFV cost path runs (matches chunk-2's bench harness so the
//!   only varying axis is the bundle + counterweights).
//!
//! - **ON**: both flags on together.
//!   `with_auto_libjxl_entropy_mul_on_photos(true)` selects the
//!   libjxl-faithful entropy_mul branch on photos (median < 95) and
//!   drops `dist_bias`; `with_enable_kavoid_entropy_of_transforms(true)`
//!   adds the per-distance penalty to DCT4X4/DCT4X8/DCT8X4 + AFV0-3
//!   above d > 4.0. AFV evaluation is also explicitly on so the chunk-2
//!   wiring is exercised.
//!
//! Decision rule for the default flip:
//!
//! - **Flip to `true`** if photos are bytes-neutral-or-better (Δb_pct
//!   ≤ 0 on the photo total) AND butteraugli stays within +0.05
//!   AND SSIM2 within −0.20 on every photo cell.
//! - **Keep default `false`** otherwise. If photos still regress, the
//!   chunks 1+2 counterweights were insufficient — the entropy_mul
//!   over-pick goes elsewhere (e.g. DCT16/DCT32/DCT64 large transforms
//!   that don't get the kAvoid penalty).
//!
//! ## Reproducer
//!
//! ```text
//! cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder' \
//!     --example kavoid_entropy_chunk3_bytes_ab -- \
//!     [--distance D] [--image PATH]...
//! ```
//!
//! Default sweep covers 3 CLIC photos × 4 distances (matches the
//! chunk-3 task description) when `--distance` is omitted by looping.
//! Screenshots are deliberately excluded — they have no bytes movement
//! since the dispatch picks the GPU-lifted branch on screenshots
//! (median > 95). The screenshot-byte-identity invariant was already
//! verified in W8-5's bench output.

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
}

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use butteraugli::{ButteraugliParams, butteraugli_linear};
    use imgref::Img;
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;
    use rgb::RGB;

    type B = cubecl::cuda::CudaRuntime;

    fn decode_linear(bytes: &[u8]) -> (usize, usize, Vec<f32>) {
        let reader = std::io::Cursor::new(bytes);
        let mut img = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("jxl-oxide read");
        img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let frame = img.render_frame(0).expect("render_frame");
        let fb = frame.image_all_channels();
        (fb.width(), fb.height(), fb.buf().to_vec())
    }

    fn linear_to_srgb_u8(c: f32) -> u8 {
        let c = c.clamp(0.0, 1.0);
        let v = if c <= 0.003_130_8 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    }

    let raw: Vec<String> = std::env::args().collect();
    // Default sweep: full 4-distance per-image grid the task requested.
    // `--distance D` overrides to a single distance for spot checks.
    let mut distances: Vec<f32> = vec![0.5, 1.0, 2.0, 5.0];
    let mut images: Vec<String> = Vec::new();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--distance" => {
                distances = vec![raw[i + 1].parse().expect("--distance D")];
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
        // Default validation set: the same 3 CLIC photos W8-5 measured
        // the regression on. Same image set lets us compare directly to
        // the W8-5 bench numbers.
        let base = "/home/lilith/work/codec-corpus";
        let candidates = [
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
        "[kavoid_chunk3_ab] distances={:?} images={} (off = production default; on = both flags + AFV)",
        distances,
        images.len()
    );
    println!(
        "{:<46} {:>4} {:>5} {:>10} {:>10} {:>+8} {:>7} {:>7} {:>7} {:>+6} {:>6} {:>6} {:>+6}",
        "image",
        "d",
        "MP",
        "off_b",
        "on_b",
        "Δb",
        "Δb_pct",
        "bfly_o",
        "bfly_n",
        "Δbfly",
        "ss2_o",
        "ss2_n",
        "Δss2",
    );

    let bfly_params = ButteraugliParams::default();

    // Per-distance running totals.
    let mut total_off_by_d: std::collections::BTreeMap<u32, u64> = Default::default();
    let mut total_on_by_d: std::collections::BTreeMap<u32, u64> = Default::default();
    let mut total_pixels: u64 = 0;
    // Decision-gate trackers: max Δbfly and min Δssim2 per distance.
    let mut max_dbfly_by_d: std::collections::BTreeMap<u32, f64> = Default::default();
    let mut min_dssim2_by_d: std::collections::BTreeMap<u32, f64> = Default::default();

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

        let orig_lin_pixels: Vec<RGB<f32>> = (0..n).map(|i| RGB::new(r[i], g[i], b[i])).collect();
        let orig_lin_img = Img::new(orig_lin_pixels, w as usize, h as usize);
        let orig_srgb_pixels: Vec<[u8; 3]> = pixels_u8
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let orig_srgb_img = Img::new(orig_srgb_pixels, w as usize, h as usize);

        let enc: GpuEncoder<B> = GpuEncoder::new();

        for &distance in &distances {
            // Path A: pre-W44-41 baseline. Both flags explicitly OFF
            // to isolate the lifted entropy_mul + dist_bias bundle from
            // the W44-41 chunk A flip. AFV explicitly on so the chunk-2
            // AFV cost path is exercised in both branches (only varying
            // axis remains the bundle + counterweights).
            let lossy_off: LossyEncoder<B> = LossyEncoder::new(&enc, w, h)
                .with_auto_evaluate_afv_on_screenshots(false)
                .with_evaluate_afv(true)
                .with_enable_kavoid_entropy_of_transforms(false);
            assert!(
                !lossy_off.auto_libjxl_entropy_mul_on_photos(),
                "production default for auto_libjxl_entropy_mul_on_photos must be OFF"
            );
            assert!(
                !lossy_off.enable_kavoid_entropy_of_transforms(),
                "pre-W44-41 baseline must have kAvoidEntropyOfTransforms OFF"
            );
            let bs_off = enc
                .encode_lossy_to_bitstream_via_precomputed(&lossy_off, &r, &g, &b, distance)
                .unwrap_or_else(|e| {
                    panic!("off-path encode failed for {path}@d={distance}: {e:?}")
                });

            // Path B: both flags on. libjxl-faithful entropy_mul +
            // dropped dist_bias on photo branch, kAvoid penalty
            // counterweight on DCT4X4/DCT4X8/DCT8X4 + AFV.
            let lossy_on: LossyEncoder<B> = LossyEncoder::new(&enc, w, h)
                .with_auto_evaluate_afv_on_screenshots(false)
                .with_evaluate_afv(true)
                .with_auto_libjxl_entropy_mul_on_photos(true)
                .with_enable_kavoid_entropy_of_transforms(true);
            let bs_on = enc
                .encode_lossy_to_bitstream_via_precomputed(&lossy_on, &r, &g, &b, distance)
                .unwrap_or_else(|e| panic!("on-path encode failed for {path}@d={distance}: {e:?}"));

            let off = bs_off.len() as i64;
            let on = bs_on.len() as i64;
            let dbytes = on - off;
            let dpct = (dbytes as f64) / (off as f64) * 100.0;

            let metric = |bytes: &[u8]| -> (f64, f64) {
                let (dw, dh, dec_lin) = decode_linear(bytes);
                assert_eq!(dw, w as usize);
                assert_eq!(dh, h as usize);
                let dec_lin_pixels: Vec<RGB<f32>> = dec_lin
                    .chunks_exact(3)
                    .map(|c| RGB::new(c[0], c[1], c[2]))
                    .collect();
                let dec_lin_img = Img::new(dec_lin_pixels, dw, dh);
                let bfly =
                    butteraugli_linear(orig_lin_img.as_ref(), dec_lin_img.as_ref(), &bfly_params)
                        .map(|s| s.score as f64)
                        .unwrap_or(f64::NAN);
                let dec_srgb: Vec<[u8; 3]> = dec_lin
                    .chunks_exact(3)
                    .map(|c| {
                        [
                            linear_to_srgb_u8(c[0]),
                            linear_to_srgb_u8(c[1]),
                            linear_to_srgb_u8(c[2]),
                        ]
                    })
                    .collect();
                let dec_srgb_img = Img::new(dec_srgb, dw, dh);
                let ssim2 =
                    fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
                        .unwrap_or(f64::NAN);
                (bfly, ssim2)
            };
            let (bfly_off, ssim2_off) = metric(&bs_off);
            let (bfly_on, ssim2_on) = metric(&bs_on);
            let dbfly = bfly_on - bfly_off;
            let dssim2 = ssim2_on - ssim2_off;

            let short = std::path::Path::new(path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(path);
            // Truncate hash names so the column stays readable.
            let short_disp: String = if short.len() > 44 {
                format!("{}…", &short[..43])
            } else {
                short.to_string()
            };
            let mp = n as f32 / 1e6;
            println!(
                "{:<46} {:>4.1} {:>5.2} {:>10} {:>10} {:>+8} {:>+6.2}% {:>7.3} {:>7.3} {:>+6.3} {:>6.2} {:>6.2} {:>+6.2}",
                short_disp,
                distance,
                mp,
                off,
                on,
                dbytes,
                dpct,
                bfly_off,
                bfly_on,
                dbfly,
                ssim2_off,
                ssim2_on,
                dssim2,
            );

            let dkey = (distance * 10.0) as u32;
            *total_off_by_d.entry(dkey).or_default() += off as u64;
            *total_on_by_d.entry(dkey).or_default() += on as u64;
            let dbfly_entry = max_dbfly_by_d.entry(dkey).or_insert(f64::NEG_INFINITY);
            if dbfly > *dbfly_entry {
                *dbfly_entry = dbfly;
            }
            let dssim2_entry = min_dssim2_by_d.entry(dkey).or_insert(f64::INFINITY);
            if dssim2 < *dssim2_entry {
                *dssim2_entry = dssim2;
            }
        }
        total_pixels += n as u64;
    }

    println!();
    println!(
        "per-distance totals (n={} photos, {:.2} MP each):",
        images.len(),
        (total_pixels as f64 / images.len() as f64) / 1e6,
    );
    for (dkey, off_sum) in &total_off_by_d {
        let on_sum = total_on_by_d.get(dkey).copied().unwrap_or(0);
        let d_disp = *dkey as f32 / 10.0;
        let dbytes = on_sum as i64 - *off_sum as i64;
        let dpct = (dbytes as f64) / (*off_sum as f64) * 100.0;
        let max_dbfly = max_dbfly_by_d.get(dkey).copied().unwrap_or(f64::NAN);
        let min_dssim2 = min_dssim2_by_d.get(dkey).copied().unwrap_or(f64::NAN);
        println!(
            "  d={:<4} off={:>10} on={:>10} ({:+} bytes, {:+.3}%)  max Δbfly={:+.3}  min Δss2={:+.2}",
            d_disp, off_sum, on_sum, dbytes, dpct, max_dbfly, min_dssim2,
        );
    }

    println!();
    println!("decision rule (per task spec):");
    println!(
        "  flip default to `true` if ALL distances satisfy: Δb_pct ≤ 0 AND max Δbfly ≤ +0.05 AND min Δss2 ≥ −0.20",
    );
    let mut flip_ok = true;
    let mut blockers: Vec<String> = Vec::new();
    for (dkey, off_sum) in &total_off_by_d {
        let on_sum = total_on_by_d.get(dkey).copied().unwrap_or(0);
        let d_disp = *dkey as f32 / 10.0;
        let dpct = (on_sum as i64 - *off_sum as i64) as f64 / (*off_sum as f64) * 100.0;
        let max_dbfly = max_dbfly_by_d.get(dkey).copied().unwrap_or(f64::NAN);
        let min_dssim2 = min_dssim2_by_d.get(dkey).copied().unwrap_or(f64::NAN);
        if dpct > 0.0 {
            flip_ok = false;
            blockers.push(format!("  d={d_disp}: bytes regressed {dpct:+.3}%"));
        }
        if max_dbfly > 0.05 {
            flip_ok = false;
            blockers.push(format!("  d={d_disp}: max Δbfly {max_dbfly:+.3} > +0.05"));
        }
        if min_dssim2 < -0.20 {
            flip_ok = false;
            blockers.push(format!("  d={d_disp}: min Δss2 {min_dssim2:+.2} < −0.20"));
        }
    }
    if flip_ok {
        println!("  RESULT: FLIP — chunks 1+2 counterweights unlocked the photo-branch dispatch.");
    } else {
        println!("  RESULT: KEEP DEFAULT FALSE — gates failed:");
        for b in &blockers {
            println!("{b}");
        }
    }
}
