// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Measures the production encoder paths at e7/e8/e9.
//!
//! - e7: encode_lossy_to_bitstream_via_precomputed_from_u8
//!       (no butteraugli loop)
//! - e8: encode_lossy_to_bitstream_via_precomputed_with_butteraugli
//!       (2 iters)
//! - e9: same with 4 iters
//!
//! Reports min/median/mean wall-clock + bitstream size for each.

use std::time::Instant;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(all(not(feature = "cuda"), not(feature = "wgpu"), feature = "cpu"))]
type Backend = cubecl::cpu::CpuRuntime;

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]
use jxl_encoder_gpu::encoder::GpuEncoder;
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]
use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

#[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "cpu")))]
fn main() {
    eprintln!("enable one of: --features cuda | wgpu | cpu");
    std::process::exit(2);
}

#[cfg(all(
    any(feature = "cuda", feature = "wgpu", feature = "cpu"),
    not(feature = "butteraugli-loop")
))]
fn main() {
    eprintln!("perf_production_e7_e8_e9 needs --features butteraugli-loop");
}

#[cfg(all(
    any(feature = "cuda", feature = "wgpu", feature = "cpu"),
    feature = "butteraugli-loop"
))]
fn main() {
    use jxl_encoder_gpu::forks::butteraugli_loop::ButteraugliLoopGpu;

    let mut args = std::env::args().skip(1);
    let mut image_path: Option<String> = None;
    let mut target_mp: Option<f32> = None;
    let mut distance: f32 = 1.0;
    let mut runs: usize = 3;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--image" => image_path = args.next(),
            "--target-mp" => target_mp = args.next().and_then(|s| s.parse().ok()),
            "--distance" => distance = args.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "--runs" => runs = args.next().and_then(|s| s.parse().ok()).unwrap_or(3),
            _ => {}
        }
    }
    let image_path = image_path.expect("--image PATH required");
    let img = image::open(&image_path).expect("decode").to_rgb8();
    let (orig_w, orig_h) = img.dimensions();
    let (w, h, pixels_u8) = if let Some(mp) = target_mp {
        let target_pixels = (mp * 1e6).round() as u32;
        let scale = (target_pixels as f64 / (orig_w as f64 * orig_h as f64)).sqrt();
        let nw = ((orig_w as f64 * scale).round() as u32).max(8);
        let nh = ((orig_h as f64 * scale).round() as u32).max(8);
        let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Lanczos3);
        (nw, nh, resized.into_raw())
    } else {
        (orig_w, orig_h, img.into_raw())
    };
    let n = (w as usize) * (h as usize);

    // sRGB → linear f32 (e8/e9 paths take linear f32 RGB).
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

    println!(
        "perf_production_e7_e8_e9: src {}×{} ({:.2} MP), distance={}, runs={}",
        w,
        h,
        n as f32 / 1e6,
        distance,
        runs,
    );

    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);

    // Warm.
    let _ = enc.encode_lossy_to_bitstream_via_precomputed_from_u8(&lossy, &pixels_u8, distance);

    // ── e7: production fast path, no butteraugli ─────────────────
    let mut e7_times = Vec::new();
    let mut e7_bytes = 0usize;
    for _ in 0..runs {
        let t = Instant::now();
        let bs = enc
            .encode_lossy_to_bitstream_via_precomputed_from_u8(&lossy, &pixels_u8, distance)
            .expect("e7");
        e7_times.push(t.elapsed().as_secs_f64() * 1000.0);
        e7_bytes = bs.len();
    }

    // ── e8: butteraugli loop, 2 iters ────────────────────────────
    let mut bg: ButteraugliLoopGpu<Backend> = ButteraugliLoopGpu::new_multires(&enc, w, h);
    bg.set_reference(&pixels_u8).expect("set_reference");

    // Warm.
    let _ = enc.encode_lossy_to_bitstream_via_precomputed_with_butteraugli(
        &lossy, &mut bg, &r, &g, &b, &pixels_u8, distance, 2,
    );
    let mut e8_times = Vec::new();
    let mut e8_bytes = 0usize;
    for _ in 0..runs {
        let t = Instant::now();
        let bs = enc
            .encode_lossy_to_bitstream_via_precomputed_with_butteraugli(
                &lossy, &mut bg, &r, &g, &b, &pixels_u8, distance, 2,
            )
            .expect("e8");
        e8_times.push(t.elapsed().as_secs_f64() * 1000.0);
        e8_bytes = bs.len();
    }

    // ── e9: butteraugli loop, 4 iters ────────────────────────────
    let _ = enc.encode_lossy_to_bitstream_via_precomputed_with_butteraugli(
        &lossy, &mut bg, &r, &g, &b, &pixels_u8, distance, 4,
    );
    let mut e9_times = Vec::new();
    let mut e9_bytes = 0usize;
    for _ in 0..runs {
        let t = Instant::now();
        let bs = enc
            .encode_lossy_to_bitstream_via_precomputed_with_butteraugli(
                &lossy, &mut bg, &r, &g, &b, &pixels_u8, distance, 4,
            )
            .expect("e9");
        e9_times.push(t.elapsed().as_secs_f64() * 1000.0);
        e9_bytes = bs.len();
    }

    fn summary(name: &str, times: &[f64], bytes: usize, mp: f32) {
        let mut s: Vec<f64> = times.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = s[0];
        let med = s[s.len() / 2];
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        println!(
            "  {} : min {:7.1} ms / median {:7.1} ms / mean {:7.1} ms / {} bytes ({:.3} bpp)",
            name,
            min,
            med,
            mean,
            bytes,
            (bytes as f32 * 8.0) / (mp * 1e6),
        );
    }
    let mp = (w as f32 * h as f32) / 1e6;
    println!("\nResults:");
    summary("e7 (no buttloop)        ", &e7_times, e7_bytes, mp);
    summary("e8 (buttloop, 2 iters)  ", &e8_times, e8_bytes, mp);
    summary("e9 (buttloop, 4 iters)  ", &e9_times, e9_bytes, mp);

    let med = |t: &[f64]| -> f64 {
        let mut s: Vec<f64> = t.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    let e7m = med(&e7_times);
    let e8m = med(&e8_times);
    let e9m = med(&e9_times);
    println!(
        "\n  e8 vs e7: {:.2}× slower, {:+.2}% bytes",
        e8m / e7m,
        (e8_bytes as f64 - e7_bytes as f64) / e7_bytes as f64 * 100.0
    );
    println!(
        "  e9 vs e7: {:.2}× slower, {:+.2}% bytes",
        e9m / e7m,
        (e9_bytes as f64 - e7_bytes as f64) / e7_bytes as f64 * 100.0
    );
}
