// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Apples-ish CPU vs GPU encode latency comparison.
//!
//! - CPU: full pipeline via jxl-encoder (encode_lossy_via_cpu →
//!   produces JXL bytes).
//! - GPU: prepare_strategy_search_plan + N × encode_with_strategy_plan_adaptive
//!   (the lossy-encode pipeline only — does NOT include bitstream
//!   output yet, so this is "GPU lossy pipeline" vs "CPU full
//!   encode"). Treat the comparison as a lower bound on the CPU side
//!   and a partial number on the GPU side.
//!
//! Run:
//!   cargo run -p jxl-encoder-gpu --features 'cuda encoder' \
//!     --release --example perf_cpu_vs_gpu \
//!     --image PATH [--effort N] [--distance D] [--iters N]

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
}

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use std::time::Instant;

    use jxl_encoder::api::{LossyConfig, PixelLayout};
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::{distance_to_qac, LossyEncoder};

    type B = cubecl::cuda::CudaRuntime;

    let raw_args: Vec<String> = std::env::args().collect();
    let mut image_path: Option<String> = None;
    let mut effort: u8 = 7;
    let mut distance: f32 = 1.0;
    let mut iters: usize = 4; // butteraugli refinement at e8+
    let mut runs: usize = 5;
    let mut i = 1;
    while i < raw_args.len() {
        match raw_args[i].as_str() {
            "--image" => {
                image_path = Some(raw_args[i + 1].clone());
                i += 2;
            }
            "--effort" => {
                effort = raw_args[i + 1].parse().expect("--effort N");
                i += 2;
            }
            "--distance" => {
                distance = raw_args[i + 1].parse().expect("--distance D");
                i += 2;
            }
            "--iters" => {
                iters = raw_args[i + 1].parse().expect("--iters N");
                i += 2;
            }
            "--runs" => {
                runs = raw_args[i + 1].parse().expect("--runs N");
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    let image_path = image_path.expect("--image PATH required");

    println!(
        "perf_cpu_vs_gpu: image={image_path} effort={effort} distance={distance} \
         iters={iters} runs={runs}"
    );

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let pixels_u8: Vec<u8> = img.into_raw();
    let n = (w * h) as usize;
    println!("  dims: {w}×{h} = {n} pixels");

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

    // ── CPU path ──────────────────────────────────────────────────
    let cpu_config = LossyConfig::new(distance).with_effort(effort);
    let enc: GpuEncoder<B> = GpuEncoder::new();

    // Warmup
    let _warm = enc
        .encode_lossy_via_cpu(&cpu_config, &pixels_u8, w, h, PixelLayout::Rgb8)
        .expect("cpu warmup");

    let mut cpu_times: Vec<f64> = Vec::with_capacity(runs);
    let mut cpu_bytes_seen: Option<usize> = None;
    for _ in 0..runs {
        let t0 = Instant::now();
        let bytes = enc
            .encode_lossy_via_cpu(&cpu_config, &pixels_u8, w, h, PixelLayout::Rgb8)
            .expect("cpu encode");
        cpu_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        if cpu_bytes_seen.is_none() {
            cpu_bytes_seen = Some(bytes.len());
        }
    }
    cpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cpu_min = cpu_times[0];
    let cpu_med = cpu_times[runs / 2];
    let cpu_mean: f64 = cpu_times.iter().sum::<f64>() / (runs as f64);

    // ── GPU lossy-pipeline path ──────────────────────────────────
    // Mirrors what encode_with_strategy_plan_adaptive_traced does
    // (no bitstream output yet — that step is upstream from this
    // pipeline and not yet ported to GPU).
    let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, w, h);
    let nb8 = (lossy.padded_dimensions().0 as usize / 8)
        * (lossy.padded_dimensions().1 as usize / 8);
    let aq_field = vec![distance_to_qac(distance); nb8];

    // Warmup
    let warm_plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
    let _ = lossy.encode_with_strategy_plan_adaptive(&enc, &warm_plan, &aq_field);
    drop(warm_plan);

    let mut gpu_times: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
        for _ in 0..iters {
            let _ = lossy.encode_with_strategy_plan_adaptive(&enc, &plan, &aq_field);
        }
        gpu_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let gpu_min = gpu_times[0];
    let gpu_med = gpu_times[runs / 2];
    let gpu_mean: f64 = gpu_times.iter().sum::<f64>() / (runs as f64);

    println!();
    println!(
        "CPU encode_lossy_via_cpu (full pipeline → {} bytes):",
        cpu_bytes_seen.unwrap_or(0)
    );
    println!("  min={:.1} ms  median={:.1} ms  mean={:.1} ms", cpu_min, cpu_med, cpu_mean);
    println!();
    println!(
        "GPU lossy pipeline (prepare + {iters}× iter, NO bitstream output yet):"
    );
    println!("  min={:.1} ms  median={:.1} ms  mean={:.1} ms", gpu_min, gpu_med, gpu_mean);
    println!();
    let ratio_min = cpu_min / gpu_min;
    let ratio_med = cpu_med / gpu_med;
    println!(
        "Ratio CPU/GPU: min={:.2}× median={:.2}× (>1 means GPU pipeline is faster than CPU full)",
        ratio_min, ratio_med
    );
    println!();
    println!(
        "Caveats:\n  * CPU number includes bitstream output; GPU number doesn't.\n  * GPU prepare\
         is one-shot; in a real encode the prepare cost is amortized only if the loop iters > 1.\n  \
         * Both warmed up; cubecl pool + jit cache are hot."
    );
}
