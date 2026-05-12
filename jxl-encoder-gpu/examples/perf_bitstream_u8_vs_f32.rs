// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! End-to-end bitstream encode A/B: f32 path vs u8 fast path.
//!
//! Runs `encode_lossy_to_bitstream_via_precomputed` (the production
//! endpoint that takes 3× pre-converted f32 linear planes) and the
//! new `encode_lossy_to_bitstream_via_precomputed_from_u8` (raw
//! interleaved sRGB u8) on the same image, prints encode wall-clock
//! and bitstream sizes for both. Confirms the u8 path's measured
//! prepare-stage win flows through to the full encode end-to-end.
//!
//! Math note: the two paths use slightly different sRGB→linear
//! conversions (the f32 path's host helper here uses the proper
//! sRGB EOTF; the u8 GPU kernel ALSO uses the proper sRGB EOTF).
//! Bitstreams should be very close in size, possibly bit-identical
//! depending on f32 rounding.
//!
//! Run:
//!   cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder' \
//!     --example perf_bitstream_u8_vs_f32 -- \
//!     --image PATH [--target-mp M] [--distance D] [--runs N]

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
    let mut runs: usize = 5;
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
            "--runs" => {
                runs = raw[i + 1].parse().expect("--runs N");
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
        "perf_bitstream_u8_vs_f32: src {}×{} ({:.2} MP), distance={}, runs={}",
        w,
        h,
        n as f32 / 1e6,
        distance,
        runs,
    );

    // Pre-convert u8 → host f32 linear (proper sRGB EOTF) for the f32 path.
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

    // ── Warm both paths (cubecl pool) ─────────────────────────────
    let _ = enc
        .encode_lossy_to_bitstream_via_precomputed(&lossy, &r, &g, &b, distance)
        .expect("warm f32");
    let _ = enc
        .encode_lossy_to_bitstream_via_precomputed_from_u8(&lossy, &pixels_u8, distance)
        .expect("warm u8");

    // ── Path A: f32 ──────────────────────────────────────────────
    let mut a_times: Vec<f64> = Vec::with_capacity(runs);
    let mut a_bytes_last = 0usize;
    for _ in 0..runs {
        let t0 = Instant::now();
        let bitstream = enc
            .encode_lossy_to_bitstream_via_precomputed(&lossy, &r, &g, &b, distance)
            .expect("f32 encode");
        a_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        a_bytes_last = bitstream.len();
    }

    // ── Path B: u8 fast path ─────────────────────────────────────
    let mut b_times: Vec<f64> = Vec::with_capacity(runs);
    let mut b_bytes_last = 0usize;
    for _ in 0..runs {
        let t0 = Instant::now();
        let bitstream = enc
            .encode_lossy_to_bitstream_via_precomputed_from_u8(&lossy, &pixels_u8, distance)
            .expect("u8 encode");
        b_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        b_bytes_last = bitstream.len();
    }

    let summary = |name: &str, times: &[f64]| {
        let mut s: Vec<f64> = times.iter().copied().collect();
        s.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let min = s[0];
        let med = s[runs / 2];
        let mean = times.iter().sum::<f64>() / runs as f64;
        println!(
            "  {} : min {:.1} ms / median {:.1} ms / mean {:.1} ms",
            name, min, med, mean,
        );
    };

    println!("\nResults ({}× per path, paired):", runs);
    summary("f32 path", &a_times);
    summary("u8  path", &b_times);

    let f32_med = {
        let mut s: Vec<f64> = a_times.iter().copied().collect();
        s.sort_by(|x, y| x.partial_cmp(y).unwrap());
        s[runs / 2]
    };
    let u8_med = {
        let mut s: Vec<f64> = b_times.iter().copied().collect();
        s.sort_by(|x, y| x.partial_cmp(y).unwrap());
        s[runs / 2]
    };
    println!(
        "  speedup u8/f32 (median): {:.2}× ({:+.1} ms)",
        f32_med / u8_med,
        u8_med - f32_med,
    );

    println!("\nBitstream sizes:");
    println!(
        "  f32 path: {} bytes ({:.3} bpp)",
        a_bytes_last,
        a_bytes_last as f64 * 8.0 / n as f64
    );
    println!(
        "  u8  path: {} bytes ({:.3} bpp)",
        b_bytes_last,
        b_bytes_last as f64 * 8.0 / n as f64
    );
    let size_delta = b_bytes_last as i64 - a_bytes_last as i64;
    println!(
        "  u8 vs f32 delta: {:+} bytes ({:+.2}%)",
        size_delta,
        100.0 * size_delta as f64 / a_bytes_last as f64,
    );
}
