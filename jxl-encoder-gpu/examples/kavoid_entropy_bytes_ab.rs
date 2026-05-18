// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! A/B byte-size + butteraugli / SSIM2 comparison for the chunk-1 GPU port
//! of libjxl's `kAvoidEntropyOfTransforms` heuristic
//! (`LossyEncoder::with_enable_kavoid_entropy_of_transforms`).
//!
//! Chunk 1 ports the formula + a narrow gate (`distance > 4.0`) onto the
//! GPU encoder's sub-block cost-grid path (DCT4X4 / DCT4X8 / DCT8X4).
//! At `distance <= 4.0` the formula returns 0.0 and the path is a
//! structural no-op — production at low distances stays byte-identical
//! regardless of the flag.
//!
//! At `distance > 4.0` the per-strategy `entropy_mul` of DCT4X4 / DCT4X8
//! / DCT8X4 is bumped by `0.5 * (12-4) / (d-4)` (clamped at d >= 12 to
//! 0.5). Expected direction at d=5.0: fewer DCT4/DCT4x8 picks, more
//! DCT8 picks, slightly smaller files at slight bfly delta. Matches the
//! CPU encoder's
//! `jxl_encoder::vardct::ac_strategy_search::avoid_entropy_of_transforms_mul`
//! exactly (unit-test-verified parity in `forks/cost.rs`).
//!
//! Usage:
//!   cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder' \
//!     --example kavoid_entropy_bytes_ab -- [--distance D] [--image PATH]...

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
    let mut distance: f32 = 5.0;
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
        // Default validation set: 3 CLIC photos (kAvoidEntropyOfTransforms
        // is a photo-content heuristic; libjxl scopes it to 8×8-class
        // sub-blocks where photo cost models over-pick DCT4/DCT4x8
        // without the penalty).
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
        "[kavoid_entropy_bytes_ab] distance={} images={} (kAvoidEntropyOfTransforms OFF baseline vs ON)",
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

        let orig_lin_pixels: Vec<RGB<f32>> = (0..n).map(|i| RGB::new(r[i], g[i], b[i])).collect();
        let orig_lin_img = Img::new(orig_lin_pixels, w as usize, h as usize);
        let orig_srgb_pixels: Vec<[u8; 3]> = pixels_u8
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let orig_srgb_img = Img::new(orig_srgb_pixels, w as usize, h as usize);

        let enc: GpuEncoder<B> = GpuEncoder::new();

        // Path A: chunk-1 flag OFF (production default). All cost-grid
        // tuning matches the pre-2026-05-17 GPU baseline.
        let lossy_off: LossyEncoder<B> =
            LossyEncoder::new(&enc, w, h).with_auto_evaluate_afv_on_screenshots(false);
        assert!(
            !lossy_off.enable_kavoid_entropy_of_transforms(),
            "production default must be OFF"
        );
        let bs_off = enc
            .encode_lossy_to_bitstream_via_precomputed(&lossy_off, &r, &g, &b, distance)
            .unwrap_or_else(|e| panic!("off-path encode failed for {path}: {e:?}"));

        // Path B: chunk-1 flag ON. At distance > 4.0 the per-strategy
        // entropy_mul of DCT4X4/DCT4X8/DCT8X4 is bumped by
        // `0.5 * (12-4) / (d-4)`. At d <= 4.0 the path is a structural
        // no-op (bytes will equal off-path).
        let lossy_on: LossyEncoder<B> = LossyEncoder::new(&enc, w, h)
            .with_auto_evaluate_afv_on_screenshots(false)
            .with_enable_kavoid_entropy_of_transforms(true);
        let bs_on = enc
            .encode_lossy_to_bitstream_via_precomputed(&lossy_on, &r, &g, &b, distance)
            .unwrap_or_else(|e| panic!("on-path encode failed for {path}: {e:?}"));

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
        let mp = n as f32 / 1e6;
        println!(
            "{:<60} {:>5.2} {:>10} {:>10} {:>+7} {:>+6.2}% {:>7.3} {:>7.3} {:>+6.3} {:>6.2} {:>6.2} {:>+6.2}",
            short, mp, off, on, dbytes, dpct, bfly_off, bfly_on, dbfly, ssim2_off, ssim2_on, dssim2,
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
