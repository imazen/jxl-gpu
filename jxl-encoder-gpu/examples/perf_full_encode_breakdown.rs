// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Time each stage of `encode_lossy_to_bitstream_via_precomputed_from_u8`
//! to find where the post-prepare time goes (the u8 fast-path saved
//! 234 ms in prepare but only 63 ms end-to-end at 12 MP — somewhere
//! in the post-prepare stages, ~170 ms is being spent that we
//! couldn't see in the prepare-only benchmark).
//!
//! Calls each step inline with `Instant::now()` between, mirroring
//! the production `encode_lossy_to_bitstream_via_precomputed_from_u8`
//! structure step-for-step.

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
}

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use std::time::Instant;

    use jxl_encoder::__pre_quantized::{
        AcStrategyMap, DistanceParams, EncoderPrecomputed, VarDctEncoder, compute_cfl_map,
        quantize_quant_field,
    };
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

    println!(
        "perf_full_encode_breakdown: src {}×{} ({:.2} MP), distance={}, runs={}",
        w,
        h,
        (w as f32 * h as f32) / 1e6,
        distance,
        runs,
    );

    let enc: GpuEncoder<B> = GpuEncoder::new();
    let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, w, h);
    let (gpu_pw, gpu_ph) = lossy.padded_dimensions();
    let cpu_pw = (w as usize).div_ceil(8) * 8;
    let cpu_ph = (h as usize).div_ceil(8) * 8;
    let xsize_blocks = cpu_pw / 8;
    let ysize_blocks = cpu_ph / 8;

    let repack = |src: &[f32]| -> Vec<f32> {
        if cpu_pw == gpu_pw as usize && cpu_ph == gpu_ph as usize {
            src.to_vec()
        } else {
            let mut dst = vec![0.0_f32; cpu_pw * cpu_ph];
            let w_us = w as usize;
            let h_us = h as usize;
            for row in 0..cpu_ph {
                let off = row * (gpu_pw as usize);
                let dst_off = row * cpu_pw;
                dst[dst_off..dst_off + cpu_pw].copy_from_slice(&src[off..off + cpu_pw]);
            }
            if cpu_pw > w_us {
                for row in 0..cpu_ph {
                    let dst_off = row * cpu_pw;
                    let last_real = dst[dst_off + w_us - 1];
                    for col in w_us..cpu_pw {
                        dst[dst_off + col] = last_real;
                    }
                }
            }
            if cpu_ph > h_us {
                let last_real_off = (h_us - 1) * cpu_pw;
                for row in h_us..cpu_ph {
                    let dst_off = row * cpu_pw;
                    dst.copy_within(last_real_off..last_real_off + cpu_pw, dst_off);
                }
            }
            dst
        }
    };

    // Warm up.
    let _ = enc
        .encode_lossy_to_bitstream_via_precomputed_from_u8(&lossy, &pixels_u8, distance)
        .expect("warm");

    // Report per-prepare-stage marks once on a warm run so we can see
    // where the 250+ ms goes inside `prepare_strategy_search_plan`.
    {
        use std::time::Instant as I;
        let mut last = I::now();
        let mut marks: Vec<(&'static str, f64)> = Vec::new();
        let mut on_mark = |name: &'static str| {
            let now = I::now();
            marks.push((name, now.duration_since(last).as_secs_f64() * 1000.0));
            last = now;
        };
        let _plan = lossy.prepare_strategy_search_plan_traced_from_u8(
            &enc,
            &pixels_u8,
            distance,
            &mut on_mark,
        );
        eprintln!("--- prepare per-stage (one warm run) ---");
        for (name, ms) in &marks {
            eprintln!("  {name:30} {ms:7.2} ms");
        }
        eprintln!("----------------------------------------");
    }

    let mut tot = vec![0.0f64; 8];
    for _ in 0..runs {
        let t0 = Instant::now();

        let plan = lossy.prepare_strategy_search_plan_from_u8(&enc, &pixels_u8, distance);
        let t_prepare = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();

        // Match production path: batched 3-plane download (one
        // sync round-trip), then parallel repack via rayon::join×3.
        let (xyb_x_gpu, xyb_y_gpu, xyb_b_gpu) = enc.download_planes_3ch(
            &plan.xyb_x_gpu,
            &plan.xyb_y_gpu,
            &plan.xyb_b_gpu,
        );
        let t_download = t1.elapsed().as_secs_f64() * 1000.0;
        let t2 = Instant::now();

        let (xyb_x, (xyb_y, xyb_b)) = rayon::join(
            || repack(&xyb_x_gpu),
            || rayon::join(|| repack(&xyb_y_gpu), || repack(&xyb_b_gpu)),
        );
        let t_repack = t2.elapsed().as_secs_f64() * 1000.0;
        let t3 = Instant::now();

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
        let t_ac_strat = t3.elapsed().as_secs_f64() * 1000.0;
        let t4 = Instant::now();

        let cfl_map = compute_cfl_map(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            cpu_pw,
            cpu_ph,
            xsize_blocks,
            ysize_blocks,
            true,
            1e-3,
            10,
        );
        let t_cfl = t4.elapsed().as_secs_f64() * 1000.0;
        let t5 = Instant::now();

        // Production now reads quant_field/masking from the strategy
        // plan (compute_quant_field_full_persistent ran in prepare).
        // The CPU compute_quant_field_float_free path is dead — kept
        // imported below for non-prod harnesses but not exercised
        // here.
        let quant_field_float = plan.quant_field_float.clone();
        let masking = plan.masking.clone();
        let t_qfield = t5.elapsed().as_secs_f64() * 1000.0;
        let t6 = Instant::now();

        let precomputed = EncoderPrecomputed::from_parts(
            w as usize,
            h as usize,
            xsize_blocks,
            ysize_blocks,
            cpu_pw,
            cpu_ph,
            xyb_x,
            xyb_y,
            xyb_b,
            Vec::new(),
            cfl_map,
            None,
            quant_field_float.clone(),
            masking,
            None,
            ac_strategy,
            true,
            distance,
            0,
            0,
        );
        let vardct = VarDctEncoder::new(distance);
        let params = DistanceParams::compute_for_profile(distance, &vardct.profile);
        let quant_field_u8 = quantize_quant_field(&quant_field_float, params.inv_scale);
        let bitstream = vardct
            .encode_from_precomputed(&precomputed, &quant_field_u8)
            .expect("encode");
        let t_bitstream = t6.elapsed().as_secs_f64() * 1000.0;

        let total = t0.elapsed().as_secs_f64() * 1000.0;
        let _ = bitstream.len();

        tot[0] += t_prepare;
        tot[1] += t_download;
        tot[2] += t_repack;
        tot[3] += t_ac_strat;
        tot[4] += t_cfl;
        tot[5] += t_qfield;
        tot[6] += t_bitstream;
        tot[7] += total;
    }
    let n = runs as f64;
    println!("\nMean per-stage at {} MP:", (w as f32 * h as f32) / 1e6);
    println!("  prepare (GPU)          {:7.1} ms", tot[0] / n);
    println!("  download xyb (sync)    {:7.1} ms", tot[1] / n);
    println!("  repack                 {:7.1} ms", tot[2] / n);
    println!("  build ac_strategy      {:7.1} ms", tot[3] / n);
    println!("  compute_cfl_map        {:7.1} ms", tot[4] / n);
    println!("  compute_quant_field    {:7.1} ms", tot[5] / n);
    println!("  vardct.encode_from_p…  {:7.1} ms", tot[6] / n);
    println!("  ─────────────────────────────");
    println!("  total                  {:7.1} ms", tot[7] / n);
}
