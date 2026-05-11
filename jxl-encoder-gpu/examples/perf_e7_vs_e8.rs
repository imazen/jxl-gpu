// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! True e7 vs e8 wall-clock measurement on the GPU lossy pipeline.
//!
//! - **e7 (Squirrel)**: prepare strat-search plan + 1 single
//!   `encode_with_strategy_plan_adaptive` call. No butteraugli loop
//!   (libjxl gates the loop at `speed_tier <= kKitten` = effort >= 8).
//! - **e8 (Kitten)**: full
//!   `refine_aq_field_gpu_with_strategy_search` butteraugli refinement
//!   loop with N_e8 iters. libjxl default for kKitten is 2 iters.
//!
//! The bench warms up both paths first (cubecl pool + jit cache), then
//! does paired timed runs.
//!
//! Run:
//!   cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder butteraugli-loop' \
//!     --example perf_e7_vs_e8 -- --image PATH \
//!     [--target-mp M] [--runs N] [--distance D]

#[cfg(not(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop")))]
fn main() {
    eprintln!("requires --features 'cuda encoder butteraugli-loop'");
}

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn main() {
    use std::time::Instant;

    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::butteraugli_loop::{
        ButteraugliLoopGpu, refine_aq_field_gpu_with_strategy_search,
    };
    use jxl_encoder_gpu::lossy_encoder::{LossyEncoder, distance_to_qac};

    type B = cubecl::cuda::CudaRuntime;

    // ── args ─────────────────────────────────────────────────────
    let raw: Vec<String> = std::env::args().collect();
    let mut image_path: Option<String> = None;
    let mut target_mp: Option<f32> = None;
    let mut runs: usize = 5;
    let mut distance: f32 = 1.0;
    let mut iters_e8: usize = 2; // libjxl Kitten default
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
            "--iters-e8" => {
                iters_e8 = raw[i + 1].parse().expect("--iters-e8 N");
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    let image_path = image_path.expect("--image PATH required");

    // ── load + resize ────────────────────────────────────────────
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

    // sRGB → linear f32 (e7+e8 paths take linear f32 RGB).
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
        "perf_e7_vs_e8: src {}×{} ({:.2} MP), distance={}, runs={}, iters_e8={}",
        w,
        h,
        n as f32 / 1e6,
        distance,
        runs,
        iters_e8,
    );

    let enc: GpuEncoder<B> = GpuEncoder::new();
    let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, w, h);
    let nb8 = (lossy.padded_dimensions().0 as usize / 8)
        * (lossy.padded_dimensions().1 as usize / 8);
    let initial_aq = vec![distance_to_qac(distance); nb8];

    // ── warmup both paths ────────────────────────────────────────
    eprintln!("[warmup] e7 path …");
    let warm_plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
    let _ = lossy.encode_with_strategy_plan_adaptive(&enc, &warm_plan, &initial_aq);
    drop(warm_plan);
    eprintln!("[warmup] e8 path …");
    let mut bg: ButteraugliLoopGpu<B> = ButteraugliLoopGpu::new_multires(&enc, w, h);
    bg.set_reference(&pixels_u8).expect("set_reference warmup");
    let _ = refine_aq_field_gpu_with_strategy_search(
        &enc,
        &lossy,
        &mut bg,
        &r,
        &g,
        &b,
        &pixels_u8,
        &initial_aq,
        distance,
        iters_e8,
        |_| {},
    )
    .expect("refine warmup");

    // ── e7: prepare + 1 encode (no butteraugli loop) ─────────────
    let mut e7_times: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
        let _ = lossy.encode_with_strategy_plan_adaptive(&enc, &plan, &initial_aq);
        e7_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // ── e8: full butteraugli refinement loop ─────────────────────
    let mut e8_times: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = refine_aq_field_gpu_with_strategy_search(
            &enc,
            &lossy,
            &mut bg,
            &r,
            &g,
            &b,
            &pixels_u8,
            &initial_aq,
            distance,
            iters_e8,
            |_| {},
        )
        .expect("refine timed");
        e8_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let summary = |name: &str, times: &[f64]| {
        let mut s: Vec<f64> = times.iter().copied().collect();
        s.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let min = s[0];
        let med = s[runs / 2];
        let mean: f64 = s.iter().sum::<f64>() / runs as f64;
        println!(
            "  {:42}  min={:8.2} ms  median={:8.2} ms  mean={:8.2} ms",
            name, min, med, mean
        );
    };
    println!();
    summary("e7 (prepare + 1 encode, no buttloop)     ", &e7_times);
    summary(&format!("e8 ({iters_e8}× butteraugli refinement) "), &e8_times);
    let e7_med = e7_times.iter().copied().fold(f64::INFINITY, f64::min);
    let e8_med = e8_times.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "\n  e8/e7 ratio (min):  {:.2}×  (Δ {:+.2} ms)",
        e8_med / e7_med,
        e8_med - e7_med,
    );
}
