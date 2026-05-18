// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chunk-2 byte-size + butteraugli / SSIM2 comparison for the GPU port of
//! libjxl's `kAvoidEntropyOfTransforms` heuristic.
//!
//! Chunk 1 wired DCT4X4 / DCT4X8 / DCT8X4 (commit `f5d3703`). Chunk 2
//! extends the same per-distance penalty into AFV's cost path via the
//! new `entropy_mul_adjust` parameter on
//! `forks::afv::afv_per_block_upstream_cost_xyb_host`. To exercise the
//! chunk-2 wiring this harness EXPLICITLY enables AFV evaluation
//! (`with_evaluate_afv(true)`) — the chunk-1 harness only hit the
//! sub-block specs because production AFV is screenshot-gated and the
//! chunk-1 corpus was all photos.
//!
//! At `distance <= 4.0` the formula returns 0.0 and both the
//! sub-block and AFV paths are structural no-ops — bytes/quality match
//! the off-path regardless of the flag. At `distance > 4.0`:
//! - Sub-block DCT4X4/DCT4X8/DCT8X4 get `0.5 * (12-4)/(d-4)` added
//!   (chunk 1).
//! - AFV0-3 get the same addition (chunk 2) — folded into AFV's
//!   `entropy_mul` via `(entropy_mul + entropy_mul_adjust).max(0.01)`.
//!
//! Expected direction at d=5.0: fewer AFV picks on screenshots (AFV's
//! cost relative to DCT8 climbs because its `entropy_mul` ~= 1.022 +
//! 4.0 = 5.022, ~5× higher), slightly different bytes / bfly tradeoff.
//! Photos see no AFV picks even with AFV ON (verified in
//! `afv_disabled_assumption.md` + the chunk-1 sweep), so photos at
//! d > 4.0 should remain byte-identical between OFF and ON.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder' \
//!     --example kavoid_entropy_chunk2_bytes_ab -- [--distance D] [--image PATH]...

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
        // Default validation set: 3 CLIC photos + 3 GB82-SC screenshots.
        // Photos test the no-AFV-picks invariant (chunks 1+2 both 0
        // bytes Δ); screenshots are where AFV actually fires under the
        // production discriminator. We pick 3 of each so the sweep
        // exercises both paths in one run.
        let base = "/home/lilith/work/codec-corpus";
        let candidates = [
            "clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
            "clic2025-1024/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png",
            "clic2025-1024/22ea12c903e41583b7c469cb86040157.png",
            "gb82-sc/terminal.png",
            "gb82-sc/codec_wiki.png",
            "gb82-sc/imac_g3.png",
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
        "[kavoid_chunk2_ab] distance={} images={} (chunks 1+2: sub-blocks + AFV)",
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

        // Path A: chunks 1+2 flag OFF. AFV is force-evaluated so the
        // AFV cost path runs (chunk-2 wiring is exercised even though
        // the entropy_mul_adjust = 0.0 makes it a no-op).
        let lossy_off: LossyEncoder<B> = LossyEncoder::new(&enc, w, h)
            .with_auto_evaluate_afv_on_screenshots(false)
            .with_evaluate_afv(true);
        assert!(
            !lossy_off.enable_kavoid_entropy_of_transforms(),
            "production default must be OFF"
        );
        let bs_off = enc
            .encode_lossy_to_bitstream_via_precomputed(&lossy_off, &r, &g, &b, distance)
            .unwrap_or_else(|e| panic!("off-path encode failed for {path}: {e:?}"));

        // Path B: chunks 1+2 flag ON. At distance > 4.0:
        //   - DCT4X4 / DCT4X8 / DCT8X4 entropy_mul bumped (chunk 1)
        //   - AFV0-3 entropy_mul bumped (chunk 2)
        // At d <= 4.0 the path is a structural no-op (bytes will equal
        // off-path).
        let lossy_on: LossyEncoder<B> = LossyEncoder::new(&enc, w, h)
            .with_auto_evaluate_afv_on_screenshots(false)
            .with_evaluate_afv(true)
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
