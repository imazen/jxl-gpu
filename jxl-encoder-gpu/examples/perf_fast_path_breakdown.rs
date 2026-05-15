// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-stage breakdown of the production all-DCT8 fast path
//! (encode_lossy_to_bitstream_via_precomputed_from_u8 with GPU CfL).

use std::time::Instant;

use cubecl::cuda::CudaRuntime as Backend;
use jxl_encoder::__pre_quantized::{
    AcStrategyMap, DistanceParams, EncoderPrecomputed, VarDctEncoder, quantize_quant_field,
};
use jxl_encoder_gpu::encoder::GpuEncoder;
use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut image_path: Option<String> = None;
    let mut target_mp: Option<f32> = None;
    let mut distance: f32 = 4.0;
    let mut runs: usize = 5;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--image" => image_path = args.next(),
            "--target-mp" => target_mp = args.next().and_then(|s| s.parse().ok()),
            "--distance" => distance = args.next().and_then(|s| s.parse().ok()).unwrap_or(4.0),
            "--runs" => runs = args.next().and_then(|s| s.parse().ok()).unwrap_or(5),
            _ => {}
        }
    }
    let image_path = image_path.expect("--image PATH required");

    let img = image::open(&image_path).expect("decode image").to_rgb8();
    let (orig_w, orig_h) = img.dimensions();
    let (w, h, pixels) = if let Some(mp) = target_mp {
        let target_pixels = (mp * 1e6).round() as u32;
        let scale = (target_pixels as f64 / (orig_w as f64 * orig_h as f64)).sqrt();
        let nw = ((orig_w as f64 * scale).round() as u32).max(8);
        let nh = ((orig_h as f64 * scale).round() as u32).max(8);
        let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Lanczos3);
        (nw, nh, resized.into_raw())
    } else {
        (orig_w, orig_h, img.into_raw())
    };
    println!(
        "perf_fast_path_breakdown: {}x{} ({:.2} MP), distance={}, runs={}",
        w,
        h,
        (w as f32 * h as f32) / 1e6,
        distance,
        runs
    );

    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);

    // Warm up.
    let _ = enc.encode_lossy_to_bitstream_via_precomputed_from_u8(&lossy, &pixels, distance);

    let xsize_blocks = (w as usize).div_ceil(8);
    let ysize_blocks = (h as usize).div_ceil(8);
    let cpu_pw = xsize_blocks * 8;
    let cpu_ph = ysize_blocks * 8;

    let mut tot = vec![0.0_f64; 8];
    for _ in 0..runs {
        let t0 = Instant::now();
        let plan = lossy.prepare_strategy_search_plan_from_u8(&enc, &pixels, distance);
        let t_plan = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();

        let mut ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);
        for a in &plan.assignments {
            let bx = a.bx as usize;
            let by = a.by as usize;
            if bx >= xsize_blocks || by >= ysize_blocks {
                continue;
            }
            use jxl_encoder_gpu::forks::transform::*;
            let (cx, cy): (usize, usize) = match a.raw_strategy {
                RAW_STRATEGY_DCT16X8 => (1, 2),
                RAW_STRATEGY_DCT8X16 => (2, 1),
                RAW_STRATEGY_DCT16X16 => (2, 2),
                RAW_STRATEGY_DCT32X16 => (2, 4),
                RAW_STRATEGY_DCT16X32 => (4, 2),
                RAW_STRATEGY_DCT32X32 => (4, 4),
                RAW_STRATEGY_DCT64X32 => (4, 8),
                RAW_STRATEGY_DCT32X64 => (8, 4),
                RAW_STRATEGY_DCT64X64 => (8, 8),
                _ => (1, 1),
            };
            if bx + cx > xsize_blocks || by + cy > ysize_blocks {
                continue;
            }
            if a.raw_strategy != 0 {
                ac_strategy.set(bx, by, a.raw_strategy);
            }
        }
        let t_ac_strat = t1.elapsed().as_secs_f64() * 1000.0;
        let t2 = Instant::now();

        let cfl_result = jxl_encoder_gpu::forks::cfl_map_gpu::compute_cfl_map_gpu_persistent(
            &enc,
            &plan.xyb_x_gpu,
            &plan.xyb_y_gpu,
            &plan.xyb_b_gpu,
            xsize_blocks,
            ysize_blocks,
            true,
            1e-3,
            10,
        );
        let cfl_map = jxl_encoder::__pre_quantized::CflMap {
            ytox: cfl_result.ytox,
            ytob: cfl_result.ytob,
            xsize_tiles: cfl_result.xsize_tiles,
            ysize_tiles: cfl_result.ysize_tiles,
        };
        let t_cfl = t2.elapsed().as_secs_f64() * 1000.0;
        let t3 = Instant::now();

        let vardct = VarDctEncoder::new(distance);
        let params = DistanceParams::compute_for_profile(distance, &vardct.profile);
        let quant_field_float = plan.quant_field_float.clone();
        let quant_field_u8 = quantize_quant_field(&quant_field_float, params.inv_scale);
        let masking = plan.masking.clone();
        let _ = (
            xsize_blocks,
            ysize_blocks,
            cpu_pw,
            cpu_ph,
            ac_strategy,
            cfl_map,
            quant_field_u8,
            quant_field_float,
            masking,
        );
        let t_post = t3.elapsed().as_secs_f64() * 1000.0;

        tot[0] += t_plan;
        tot[1] += t_ac_strat;
        tot[2] += t_cfl;
        tot[3] += t_post;

        // Production end-to-end (includes a fresh prepare + cfl etc.)
        // — we just want the producer + entropy_encode delta.
        let t_full_start = Instant::now();
        let bs = enc
            .encode_lossy_to_bitstream_via_precomputed_from_u8(&lossy, &pixels, distance)
            .expect("encode");
        let t_full = t_full_start.elapsed().as_secs_f64() * 1000.0;
        let _ = bs.len();
        tot[4] += t_full;
    }
    let n = runs as f64;
    println!("\nMean per-stage at {:.2} MP:", (w as f32 * h as f32) / 1e6);
    println!("  prepare_strategy_search    {:7.1} ms", tot[0] / n);
    println!("  build ac_strategy          {:7.1} ms", tot[1] / n);
    println!("  GPU compute_cfl_map        {:7.1} ms", tot[2] / n);
    println!(
        "  vardct/params/quant_field  {:7.1} ms (CPU postwork before producer)",
        tot[3] / n
    );
    println!(
        "  full production end-to-end {:7.1} ms (includes everything)",
        tot[4] / n
    );
    let above = tot[0] + tot[1] + tot[2] + tot[3];
    let producer_plus_entropy = tot[4] - above;
    println!(
        "  → implied producer+entropy {:7.1} ms (full - above)",
        producer_plus_entropy / n
    );
}
