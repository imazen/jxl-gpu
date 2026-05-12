// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! End-to-end paired A/B for `prepare_strategy_search_plan` with two
//! input forms:
//!
//! - **Path A — f32 host preprocess + 3× padded plane upload.** Caller
//!   converts sRGB → linear in u8→f32 host code, pads, and calls
//!   [`LossyEncoder::prepare_strategy_search_plan_traced`] (the
//!   pre-existing entry point).
//! - **Path B — raw u8 + GPU-fused sRGB→linear+pad.** Caller hands the
//!   raw interleaved u8 RGB buffer directly to
//!   [`LossyEncoder::prepare_strategy_search_plan_traced_from_u8`].
//!   One fused GPU launch does the conversion and padding; upload is
//!   3× smaller (u8 vs f32) AND skips the host-side
//!   `pad_to_alignment` × 3.
//!
//! Both paths are timed end-to-end (entire prepare_strategy_search_plan
//! call), not just the upload step. This catches any downstream
//! divergence (e.g. cubecl re-syncing on mixed launches) and shows the
//! "real" production benefit.
//!
//! Run:
//!   cargo run --release -p jxl-encoder-gpu --features 'cuda encoder' \
//!     --example perf_strat_plan_u8_vs_f32 -- --image PATH \
//!     [--target-mp M] [--runs N] [--distance D]

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

    // Args.
    let raw: Vec<String> = std::env::args().collect();
    let mut image_path: Option<String> = None;
    let mut target_mp: Option<f32> = None;
    let mut runs: usize = 5;
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
            "--runs" => {
                runs = raw[i + 1].parse().expect("--runs N");
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

    // Load + optionally resize.
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
        "perf_strat_plan_u8_vs_f32: src {}×{} ({:.2} MP), distance={}, runs={}",
        w,
        h,
        n as f32 / 1e6,
        distance,
        runs,
    );

    let enc: GpuEncoder<B> = GpuEncoder::new();
    let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, w, h);

    // Pre-convert u8 → host f32 linear (Path A's host-side cost is
    // OUTSIDE the timed region because that's how production callers
    // would already be holding the f32 — but we time the prepare call
    // itself which includes the pad + upload).
    let to_linear = |c: u8| -> f32 {
        let f = c as f32 / 255.0;
        if f <= 0.04045 {
            f / 12.92
        } else {
            ((f + 0.055) / 1.055).powf(2.4)
        }
    };
    let mut r_lin = Vec::with_capacity(n);
    let mut g_lin = Vec::with_capacity(n);
    let mut b_lin = Vec::with_capacity(n);
    for chunk in pixels_u8.chunks_exact(3) {
        r_lin.push(to_linear(chunk[0]));
        g_lin.push(to_linear(chunk[1]));
        b_lin.push(to_linear(chunk[2]));
    }

    // Warm both paths (and the cubecl pool).
    let _ = lossy.prepare_strategy_search_plan_traced(
        &enc,
        &r_lin,
        &g_lin,
        &b_lin,
        distance,
        &mut |_| {},
    );
    let _ =
        lossy.prepare_strategy_search_plan_traced_from_u8(&enc, &pixels_u8, distance, &mut |_| {});

    // ── Path A: f32 ──────────────────────────────────────────────
    let mut a_times: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = lossy.prepare_strategy_search_plan_traced(
            &enc,
            &r_lin,
            &g_lin,
            &b_lin,
            distance,
            &mut |_| {},
        );
        a_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // ── Path B: u8 fused ─────────────────────────────────────────
    let mut b_times: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = lossy.prepare_strategy_search_plan_traced_from_u8(
            &enc,
            &pixels_u8,
            distance,
            &mut |_| {},
        );
        b_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let summary = |name: &str, times: &[f64]| {
        let mut s: Vec<f64> = times.iter().copied().collect();
        s.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let min = s[0];
        let med = s[runs / 2];
        let mean: f64 = s.iter().sum::<f64>() / runs as f64;
        println!(
            "  {:36}  min={:8.2} ms  median={:8.2} ms  mean={:8.2} ms",
            name, min, med, mean
        );
    };
    println!();
    summary("Path A: prepare(f32, 3×plane upload)", &a_times);
    summary("Path B: prepare_from_u8 (fused)     ", &b_times);
    let a_min = *a_times
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let b_min = *b_times
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let a_med = a_times[runs / 2];
    let b_med = b_times[runs / 2];
    println!(
        "\n  speedup (min):     {:.2}×  (Δ {:+.2} ms)",
        a_min / b_min,
        a_min - b_min,
    );
    println!(
        "  speedup (median):  {:.2}×  (Δ {:+.2} ms)",
        a_med / b_med,
        a_med - b_med,
    );
}
