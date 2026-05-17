// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! A/B byte-size + butteraugli / SSIM2 comparison for the opt-in
//! entropy_mul + dist_bias bundle dispatch introduced in 2026-05-17.
//! Encodes each image with the dispatch OFF (production default — the
//! GPU-lifted entropy_mul + distance-scaled dist_bias values applied
//! uniformly) and ON (experimental opt-in — libjxl-faithful values +
//! dist_bias = 1.0 on photo content, GPU-lifted values + distance-scaled
//! dist_bias on screenshots).
//!
//! Audit hypothesis
//! (`vardct_gpu_dropped_optimizations_resurrection_2026-05-17.md` item #3+#10):
//!
//! - Screenshots (mask1x1 median > 95): on byte-identical to off
//!   (the dispatch picks the GPU-lifted branch on screenshots).
//! - Photos: on saves 0.5-1% bytes at slight bfly cost, because
//!   libjxl-faithful entropy_mul values let larger transforms win on
//!   smooth regions where they are actually optimal.
//!
//! Measured 2026-05-17 (3 CLIC photos + 3 GB82-SC screenshots at d=1.0):
//!
//! - Screenshots: byte-identical, bfly identical, ssim2 identical. OK.
//! - Photos: bytes +2.5% to +8.5%, butteraugli +0.11 to +0.24, SSIM2
//!   −0.17 to −0.42. **Pareto-worse on every axis** — refutes the
//!   audit hypothesis. Root cause is the original drop rationale
//!   re-validated: this GPU encoder lacks `kAvoidEntropyOfTransforms`
//!   and the X-channel multi-block weight (cross-ref dropped log
//!   item #4), so removing the GPU-lifted counterweights causes
//!   over-pick of large transforms regardless of content class.
//!
//! Disposition: dispatch infrastructure left in place as an opt-in;
//! production default flipped to OFF. Re-validate once the missing
//! counterweights land.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder' \
//!     --example auto_entropy_mul_bytes_ab -- [--distance D] [--image PATH]...

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
    use std::time::Instant;

    type B = cubecl::cuda::CudaRuntime;

    // Decode a JXL bitstream to linear-light RGB f32 (interleaved). Uses
    // jxl-oxide's `srgb_linear(Relative)` color encoding request so
    // butteraugli sees consistent sRGB TF on both branches; per project
    // CLAUDE.md this is the metadata-immune comparison path.
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
            // Screenshots — dispatch picks GPU-lifted branch → byte-identical to off.
            "gb82-sc/terminal.png",
            "gb82-sc/imac_g3.png",
            "gb82-sc/windows95.png",
            // Photos — dispatch picks libjxl-faithful branch; bytes may shift.
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
        "[auto_entropy_mul_bytes_ab] distance={} images={} (auto-dispatch OFF baseline vs ON)",
        distance,
        images.len()
    );
    println!(
        "{:<60} {:>5} {:>10} {:>10} {:>+7} {:>7} {:>7} {:>7} {:>+6} {:>6} {:>6} {:>+6}",
        "image",
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
        "Δss2"
    );

    let bfly_params = ButteraugliParams::default();

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

        // Source for butteraugli (linear interleaved RGB f32 pixels) and
        // for SSIM2 (sRGB u8 — re-use original pixels_u8 reshape).
        let orig_lin_pixels: Vec<RGB<f32>> = (0..n)
            .map(|i| RGB::new(r[i], g[i], b[i]))
            .collect();
        let orig_lin_img = Img::new(orig_lin_pixels, w as usize, h as usize);
        let orig_srgb_pixels: Vec<[u8; 3]> = pixels_u8
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let orig_srgb_img = Img::new(orig_srgb_pixels, w as usize, h as usize);

        let enc: GpuEncoder<B> = GpuEncoder::new();

        // Path A: production default (auto-dispatch OFF — the GPU-lifted
        // entropy_mul + distance-scaled dist_bias values applied uniformly).
        // Disable auto-AFV too so the only varying axis is the entropy_mul +
        // dist_bias bundle.
        let lossy_off: LossyEncoder<B> = LossyEncoder::new(&enc, w, h)
            .with_auto_evaluate_afv_on_screenshots(false);
        assert!(
            !lossy_off.auto_libjxl_entropy_mul_on_photos(),
            "production default must be OFF (refuted 2026-05-17 A/B)"
        );
        let bs_off = enc
            .encode_lossy_to_bitstream_via_precomputed(&lossy_off, &r, &g, &b, distance)
            .unwrap_or_else(|e| panic!("off-path encode failed for {path}: {e:?}"));

        // Path B: experimental opt-in (auto-dispatch ON). Photos take the
        // libjxl-faithful branch (1.0428 / 0.859316 + dist_bias=1.0);
        // screenshots stay on GPU-lifted values (byte-identical to OFF).
        // Currently Pareto-worse on photos — kept for re-validation once
        // kAvoidEntropyOfTransforms + X-channel multi-block weight land.
        let lossy_on: LossyEncoder<B> = LossyEncoder::new(&enc, w, h)
            .with_auto_evaluate_afv_on_screenshots(false)
            .with_auto_libjxl_entropy_mul_on_photos(true);
        let t0 = Instant::now();
        let bs_on = enc
            .encode_lossy_to_bitstream_via_precomputed(&lossy_on, &r, &g, &b, distance)
            .unwrap_or_else(|e| panic!("on-path encode failed for {path}: {e:?}"));
        let ms_on = t0.elapsed().as_secs_f64() * 1000.0;

        let off = bs_off.len() as i64;
        let on = bs_on.len() as i64;
        let dbytes = on - off;
        let dpct = (dbytes as f64) / (off as f64) * 100.0;

        // Decode both branches → linear RGB → butteraugli + ssim2.
        let metric = |bytes: &[u8]| -> (f64, f64) {
            let (dw, dh, dec_lin) = decode_linear(bytes);
            assert_eq!(dw, w as usize);
            assert_eq!(dh, h as usize);
            let dec_lin_pixels: Vec<RGB<f32>> = dec_lin
                .chunks_exact(3)
                .map(|c| RGB::new(c[0], c[1], c[2]))
                .collect();
            let dec_lin_img = Img::new(dec_lin_pixels, dw, dh);
            let bfly = butteraugli_linear(
                orig_lin_img.as_ref(),
                dec_lin_img.as_ref(),
                &bfly_params,
            )
            .map(|s| s.score as f64)
            .unwrap_or(f64::NAN);
            let dec_srgb: Vec<[u8; 3]> = dec_lin
                .chunks_exact(3)
                .map(|c| [
                    linear_to_srgb_u8(c[0]),
                    linear_to_srgb_u8(c[1]),
                    linear_to_srgb_u8(c[2]),
                ])
                .collect();
            let dec_srgb_img = Img::new(dec_srgb, dw, dh);
            let ssim2 = fast_ssim2::compute_ssimulacra2(
                orig_srgb_img.as_ref(),
                dec_srgb_img.as_ref(),
            )
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
        let mp = n as f32 / 1e6;
        let _ = ms_on;
        println!(
            "{:<60} {:>5.2} {:>10} {:>10} {:>+7} {:>+6.2}% {:>7.3} {:>7.3} {:>+6.3} {:>6.2} {:>6.2} {:>+6.2}",
            short, mp, off, on, dbytes, dpct,
            bfly_off, bfly_on, dbfly,
            ssim2_off, ssim2_on, dssim2,
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
