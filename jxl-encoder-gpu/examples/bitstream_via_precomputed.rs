// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Smoke test for `encode_lossy_to_bitstream_via_precomputed` — runs
//! the GPU strat-search pipeline + CPU bitstream emit via the
//! `__pre_quantized` seam and verifies the output bitstream is
//! non-empty (and ideally decodes cleanly).
//!
//! This is the end-to-end validation that GPU work finally reaches
//! the bitstream stage instead of being discarded by the prior
//! `encode_lossy_via_cpu` pure-CPU passthrough.
//!
//! Run:
//!   cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder' \
//!     --example bitstream_via_precomputed -- --image PATH \
//!     [--target-mp M] [--distance D]

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
}

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use std::time::Instant;

    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

    type B = cubecl::cuda::CudaRuntime;

    let raw: Vec<String> = std::env::args().collect();
    let mut image_path: Option<String> = None;
    let mut target_mp: Option<f32> = None;
    let mut distance: f32 = 1.0;
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--image" => {
                image_path = Some(raw[i + 1].clone());
                i += 2;
            }
            "--target-mp" => {
                target_mp = Some(raw[i + 1].parse().expect("--target-mp M"));
                i += 2;
            }
            "--distance" => {
                distance = raw[i + 1].parse().expect("--distance D");
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    let image_path = image_path.expect("--image PATH required");

    let mut img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("open {image_path}: {e}"))
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
    let pixels_u8: Vec<u8> = img.into_raw();
    let n = (w as usize) * (h as usize);

    println!(
        "bitstream_via_precomputed: src {}×{} ({:.2} MP), distance={}",
        w,
        h,
        n as f32 / 1e6,
        distance,
    );

    // sRGB → linear f32 (the GPU prepare path's input format).
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
    let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, w, h);

    println!("[run] encode_lossy_to_bitstream_via_precomputed …");
    let t0 = Instant::now();
    let bitstream = enc
        .encode_lossy_to_bitstream_via_precomputed(&lossy, &r, &g, &b, distance)
        .unwrap_or_else(|e| panic!("encode failed: {e:?}"));
    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!(
        "  bitstream: {} bytes, {:.2} bpp, encode wall-clock {:.1} ms",
        bitstream.len(),
        (bitstream.len() as f64 * 8.0) / (n as f64),
        dt_ms,
    );

    // Sanity: JXL signature 0xFF 0x0A or container 0x00 0x00 0x00 0x0C / 0x4A 0x58 0x4C 0x20
    let head = bitstream.first_chunk::<8>().expect("bitstream too short");
    let is_codestream = head[0] == 0xFF && head[1] == 0x0A;
    let is_container = head[0] == 0x00
        && head[1] == 0x00
        && head[2] == 0x00
        && head[3] == 0x0C
        && head[4] == 0x4A
        && head[5] == 0x58
        && head[6] == 0x4C
        && head[7] == 0x20;
    if !is_codestream && !is_container {
        eprintln!(
            "WARNING: bitstream does NOT start with JXL signature (got {head:02x?}) — possibly invalid"
        );
        std::process::exit(2);
    }
    println!(
        "  [sig] valid {} signature",
        if is_container { "container" } else { "codestream" }
    );
}
