// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! GPU vs CPU butteraugli per-tile-distance parity test.
//!
//! Encodes one image via the GPU e7 fast path
//! (`encode_lossy_to_bitstream_via_precomputed_from_u8`), decodes the
//! resulting JXL via jxl-rs to linear-RGB f32, then runs the SAME
//! reduction pipeline (per-pixel diffmap → `compute_tile_distances`)
//! through TWO independent butteraugli implementations:
//!
//! - **GPU**: `ButteraugliLoopGpu::new_multires` +
//!   `set_reference(srgb_u8)` + `compute_with_reference(srgb_u8_recon)`.
//!   Uses the same code path the e8/e9 refinement loop calls
//!   (`forks::butteraugli_loop`).
//! - **CPU**: `butteraugli::butteraugli` (sRGB u8 in, max-norm + diffmap
//!   out). Independent Rust port of libjxl, used by jxl-encoder's
//!   `butteraugli-loop` feature.
//!
//! Both engines see byte-identical sRGB u8 inputs; sRGB→linear and the
//! whole pipeline run inside each engine. Differences therefore come
//! from independent ports of libjxl's reference algorithm — exactly
//! the question we're asking ("are the GPU butteraugli scores
//! systematically biased relative to a known-good CPU implementation?").
//!
//! Output: counts of tiles diverging by >10% (above a `distance/10`
//! noise floor), mean/max abs-fractional delta, and the 5 worst
//! `(gpu, cpu, x_block, y_block)` for inspection.
//!
//! ## Phase-1 finding (2026-05-14, see issue imazen/zenmetrics#…)
//!
//! Across 25+ CLIC-2025 images at d ∈ {0.5, 1.0, 2.0, 3.0, 5.0}:
//! - mean |Δfrac| = 0.02–0.03% per tile
//! - max |Δfrac| = 1.6–3.7% (a single near-edge tile per image)
//! - **0** tiles diverge by >10% (above noise floor) on any image
//! - bias(GPU − CPU) = +0.006…+0.010 (uniform tiny positive offset,
//!   most likely from sRGB→linear conversion precision drift between
//!   `colors::srgb_u8_to_linear_planar_kernel` and CPU's table)
//!
//! GPU and CPU butteraugli AGREE on `tile_dist` to within FP noise.
//! The 5–14% byte gap between `refine_aq_field_gpu_with_strategy_*`
//! and the CPU butteraugli loop on adversarial CLIC images is therefore
//! NOT caused by GPU butteraugli divergence; it must come from the
//! still-unported CPU-loop features
//! (`kOriginalComparisonRound` + per-iter `SetQuantField` recompute +
//! min-step + convergence early-exit; see
//! `forks::butteraugli_loop::refine_aq_field_gpu_with_strategy_search_persistent`
//! and `e8_e9_perf_2026-05-14.md` / `buttloop_rd_gap_2026-05-14.md`).
//!
//! Usage:
//!   cargo run -p jxl-encoder-gpu --release \
//!       --features 'cuda encoder butteraugli-loop' \
//!       --example buttloop_parity -- \
//!       --image PATH.png --distance 1.0

#![allow(clippy::needless_range_loop)]

#[cfg(not(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop")))]
fn main() {
    eprintln!("buttloop_parity requires --features 'cuda encoder butteraugli-loop'");
    std::process::exit(2);
}

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn main() {
    use butteraugli::{ButteraugliParams, Img, RGB8, butteraugli};
    use cubecl::cuda::CudaRuntime as Backend;
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::butteraugli_loop::{
        AcStrategyInfo, ButteraugliLoopGpu, K_TILE_NORM, compute_tile_distances, dct8_only_storage,
    };
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

    let mut args = std::env::args().skip(1);
    let mut image_path: Option<String> = None;
    let mut distance: f32 = 1.0;
    let mut compare_mode: &'static str = "srgb";
    while let Some(a) = args.next() {
        match a.as_str() {
            "--image" => image_path = args.next(),
            "--distance" => distance = args.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "--mode" => {
                let m = args.next().unwrap_or_else(|| "srgb".to_string());
                compare_mode = match m.as_str() {
                    "linear" => "linear",
                    _ => "srgb",
                };
            }
            _ => {}
        }
    }
    let image_path = image_path.expect("--image PATH required");

    println!("=== buttloop_parity: GPU vs CPU butteraugli per-tile distance ===");
    println!("Image:    {image_path}");
    println!("Distance: {distance}");
    println!("Mode:     {compare_mode} (sRGB-u8-in for both engines)\n");

    // ── Load source PNG ────────────────────────────────────────────────
    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("failed to open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let src_srgb_u8: Vec<u8> = img.into_raw();
    println!("Loaded:   {w}x{h} sRGB u8 ({} bytes)", src_srgb_u8.len());

    // ── Encode (GPU e7 fast path) ──────────────────────────────────────
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);
    let t0 = std::time::Instant::now();
    let bytes = enc
        .encode_lossy_to_bitstream_via_precomputed_from_u8(&lossy, &src_srgb_u8, distance)
        .expect("encode failed");
    let encode_dt = t0.elapsed();
    let bpp = (bytes.len() * 8) as f64 / (w as f64 * h as f64);
    println!(
        "Encoded:  {} bytes ({bpp:.3} bpp) in {:.2}s",
        bytes.len(),
        encode_dt.as_secs_f64()
    );

    // ── Decode via jxl-rs to linear RGB f32 ────────────────────────────
    let t1 = std::time::Instant::now();
    let recon_lin_f32 = decode_jxl_rs_linear_f32(&bytes, w, h);
    let decode_dt = t1.elapsed();
    println!(
        "Decoded:  {} f32 samples in {:.2}s",
        recon_lin_f32.len(),
        decode_dt.as_secs_f64()
    );

    // ── Convert decoded linear → sRGB u8 (for both engines' input) ────
    let n_pix = (w as usize) * (h as usize);
    let mut recon_srgb_u8 = vec![0u8; n_pix * 3];
    for i in 0..n_pix {
        let i3 = i * 3;
        let i_in = i * 3;
        recon_srgb_u8[i3] = linear_f32_to_srgb_u8(recon_lin_f32[i_in]);
        recon_srgb_u8[i3 + 1] = linear_f32_to_srgb_u8(recon_lin_f32[i_in + 1]);
        recon_srgb_u8[i3 + 2] = linear_f32_to_srgb_u8(recon_lin_f32[i_in + 2]);
    }

    // ── GPU butteraugli ────────────────────────────────────────────────
    let t2 = std::time::Instant::now();
    let mut bg: ButteraugliLoopGpu<Backend> = ButteraugliLoopGpu::new_multires(&enc, w, h);
    bg.set_reference(&src_srgb_u8).expect("gpu set_reference");
    let gpu_result = bg
        .compute_with_reference(&recon_srgb_u8)
        .expect("gpu compute_with_reference");
    let mut gpu_diffmap = vec![0.0_f32; n_pix];
    bg.copy_diffmap_to(&mut gpu_diffmap)
        .expect("gpu copy_diffmap");
    let gpu_dt = t2.elapsed();
    println!(
        "GPU:      score={:.4} pnorm_3={:.4} in {:.3}s",
        gpu_result.score,
        gpu_result.pnorm_3,
        gpu_dt.as_secs_f64(),
    );

    // ── CPU butteraugli ────────────────────────────────────────────────
    // CPU butteraugli wants `ImgRef<RGB8>` for sRGB u8 input.
    let src_rgb8: Vec<RGB8> = src_srgb_u8
        .chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let recon_rgb8: Vec<RGB8> = recon_srgb_u8
        .chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let src_img = Img::new(src_rgb8, w as usize, h as usize);
    let recon_img = Img::new(recon_rgb8, w as usize, h as usize);
    let mut params = ButteraugliParams::default();
    params = params.with_compute_diffmap(true);
    let t3 = std::time::Instant::now();
    let cpu_result =
        butteraugli(src_img.as_ref(), recon_img.as_ref(), &params).expect("cpu butteraugli");
    let cpu_dt = t3.elapsed();
    let cpu_diffmap_imgvec = cpu_result.diffmap.expect("compute_diffmap=true");
    println!(
        "CPU:      score={:.4} pnorm_3={:.4} in {:.3}s",
        cpu_result.score,
        cpu_result.pnorm_3,
        cpu_dt.as_secs_f64(),
    );

    let cpu_diffmap_buf = cpu_diffmap_imgvec.buf();
    let cpu_dm_w = cpu_diffmap_imgvec.width();
    let cpu_dm_h = cpu_diffmap_imgvec.height();
    assert_eq!(cpu_dm_w, w as usize, "CPU diffmap width mismatch");
    assert_eq!(cpu_dm_h, h as usize, "CPU diffmap height mismatch");
    assert_eq!(
        cpu_diffmap_buf.len(),
        n_pix,
        "CPU diffmap buf len {} vs n_pix {}",
        cpu_diffmap_buf.len(),
        n_pix
    );

    // ── Compute per-tile distances on both diffmaps (DCT8-only grid) ──
    let xsize_blocks = (w as usize).div_ceil(8);
    let ysize_blocks = (h as usize).div_ceil(8);
    let num_blocks = xsize_blocks * ysize_blocks;
    let (is_first, cx, cy) = dct8_only_storage(num_blocks);
    let info = AcStrategyInfo {
        is_first: &is_first,
        covered_x: &cx,
        covered_y: &cy,
    };

    let gpu_tile_dist = compute_tile_distances(
        &gpu_diffmap,
        w as usize,
        h as usize,
        xsize_blocks,
        ysize_blocks,
        &info,
    );
    let cpu_tile_dist = compute_tile_distances(
        cpu_diffmap_buf,
        w as usize,
        h as usize,
        xsize_blocks,
        ysize_blocks,
        &info,
    );

    // ── Per-pixel diffmap divergence ──────────────────────────────────
    {
        let mut sum_abs = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut max_abs = 0.0_f32;
        let mut sum_gpu = 0.0_f64;
        let mut sum_cpu = 0.0_f64;
        for i in 0..n_pix {
            let g = gpu_diffmap[i];
            let c = cpu_diffmap_buf[i];
            sum_gpu += g as f64;
            sum_cpu += c as f64;
            let d = (g - c).abs();
            sum_abs += d as f64;
            sum_sq += (d as f64) * (d as f64);
            if d > max_abs {
                max_abs = d;
            }
        }
        let mean_abs = sum_abs / n_pix as f64;
        let rmse = (sum_sq / n_pix as f64).sqrt();
        println!(
            "\n[per-pixel diffmap] mean_abs={mean_abs:.6} rmse={rmse:.6} max_abs={max_abs:.4} \
             | mean_gpu={:.6} mean_cpu={:.6}",
            sum_gpu / n_pix as f64,
            sum_cpu / n_pix as f64,
        );
    }

    // ── Per-tile divergence ───────────────────────────────────────────
    //
    // Denominator clamp: the loop's only meaningful distance threshold
    // is `tile_dist > target_distance` (so blocks with butteraugli
    // headroom at the target). We therefore filter the divergence
    // statistics by `cpu_tile_dist > MIN_MEANINGFUL` — anything below
    // is below the loop's noise floor and a relative-difference
    // computation there is pure FP epsilon noise (e.g. 1e-7 / 1e-7 ≈
    // 1.0). Use 1.0 (target_distance baseline) since loop decisions
    // only care about diff > 1.0 vs diff < 1.0.
    let min_meaningful = 0.10_f32 * (distance.max(0.10_f32));

    let mut sum_abs_frac = 0.0_f64;
    let mut max_abs_frac = 0.0_f32;
    let mut max_loc = (0usize, 0usize, 0.0_f32, 0.0_f32);
    let mut diverge_count = 0usize;
    let mut filtered_count = 0usize;
    let mut sum_gpu = 0.0_f64;
    let mut sum_cpu = 0.0_f64;
    let mut worst: Vec<(f32, f32, f32, usize, usize)> = Vec::new(); // (frac, gpu, cpu, bx, by)

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let bi = by * xsize_blocks + bx;
            let g = gpu_tile_dist[bi];
            let c = cpu_tile_dist[bi];
            sum_gpu += g as f64;
            sum_cpu += c as f64;
            // Skip below-noise tiles for divergence statistics; the loop
            // ignores diff < 1.0 anyway.
            if c < min_meaningful && g < min_meaningful {
                continue;
            }
            filtered_count += 1;
            let denom = c.max(min_meaningful);
            let frac = ((g - c) / denom).abs();
            sum_abs_frac += frac as f64;
            if frac > max_abs_frac {
                max_abs_frac = frac;
                max_loc = (bx, by, g, c);
            }
            if frac > 0.10 {
                diverge_count += 1;
            }
            worst.push((frac, g, c, bx, by));
        }
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let n_blocks = num_blocks as f64;
    let n_meaningful = (filtered_count as f64).max(1.0);
    let mean_abs_frac = sum_abs_frac / n_meaningful;
    let mean_gpu = sum_gpu / n_blocks;
    let mean_cpu = sum_cpu / n_blocks;
    let signed_bias = (sum_gpu - sum_cpu) / n_blocks;

    println!("\n[per-tile distance] (DCT8-only, {num_blocks} tiles)");
    println!(
        "  mean(GPU) = {mean_gpu:.4}    mean(CPU) = {mean_cpu:.4}    \
         bias(GPU-CPU) = {signed_bias:+.4}"
    );
    println!(
        "  meaningful tiles = {filtered_count} / {num_blocks} ({:.1}%, threshold > {min_meaningful:.4})",
        100.0 * filtered_count as f64 / n_blocks,
    );
    println!(
        "  mean abs Δ frac = {:.2}%    max abs Δ frac = {:.2}%    \
         tiles >10% diverge = {} / {} ({:.2}%)",
        mean_abs_frac * 100.0,
        max_abs_frac * 100.0,
        diverge_count,
        filtered_count,
        100.0 * diverge_count as f64 / n_meaningful,
    );
    println!(
        "  worst tile: ({}, {})  GPU={:.4}  CPU={:.4}",
        max_loc.0, max_loc.1, max_loc.2, max_loc.3,
    );
    println!("\n  Top 5 worst tiles by Δ frac (above noise floor):");
    for (i, (frac, g, c, bx, by)) in worst.iter().take(5).enumerate() {
        println!(
            "    [{i}]  ({bx:3}, {by:3})  GPU={g:.4}  CPU={c:.4}  \
             Δ={:+.4}  frac={:+.2}%",
            g - c,
            frac * 100.0,
        );
    }

    let _ = K_TILE_NORM; // referenced in compute_tile_distances; silence unused

    // One-line summary at the end (easy to grep from CI logs).
    println!(
        "\nSUMMARY: GPU vs CPU per-tile distance: mean Δ = {:.2}%, max = {:.2}%, {} / {} meaningful tiles diverge by >10% (bias GPU-CPU = {:+.4})",
        mean_abs_frac * 100.0,
        max_abs_frac * 100.0,
        diverge_count,
        filtered_count,
        signed_bias,
    );
}

/// IEC 61966-2-1 sRGB linearization for u8 → f32.
#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn linear_f32_to_srgb_u8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn decode_jxl_rs_linear_f32(bytes: &[u8], w: u32, h: u32) -> Vec<f32> {
    use jxl::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input: &[u8] = bytes;
    let options = JxlDecoderOptions::default();
    let initialized: JxlDecoder<states::Initialized> = JxlDecoder::new(options);

    let mut decoder_with_image_info = match initialized.process(&mut input).expect("decode init") {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => {
            panic!("decoder reported NeedsMoreInput on initial process");
        }
    };

    let basic_info = decoder_with_image_info.basic_info().clone();
    assert_eq!(basic_info.size, (w as usize, h as usize));

    let default_format = decoder_with_image_info.current_pixel_format().clone();
    let requested_format = JxlPixelFormat {
        color_type: default_format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: default_format
            .extra_channel_format
            .iter()
            .map(|_| Some(JxlDataFormat::f32()))
            .collect(),
    };
    decoder_with_image_info.set_pixel_format(requested_format);

    let pixel_format = decoder_with_image_info.current_pixel_format().clone();
    let num_channels = pixel_format.color_type.samples_per_pixel();
    assert!(
        num_channels == 3,
        "expected 3-channel RGB output, got {num_channels}"
    );

    let buffer_w = basic_info.size.0;
    let buffer_h = basic_info.size.1;

    let mut color_buf = Image::<f32>::new_with_value((buffer_w * num_channels, buffer_h), f32::NAN)
        .expect("alloc color buf");

    let extra_buf_count = pixel_format
        .extra_channel_format
        .iter()
        .filter(|x| x.is_some())
        .count();
    let mut extra_bufs: Vec<Image<f32>> = (0..extra_buf_count)
        .map(|_| Image::<f32>::new_with_value((buffer_w, buffer_h), f32::NAN).expect("alloc extra"))
        .collect();

    let mut all_imgs: Vec<&mut Image<f32>> = std::iter::once(&mut color_buf)
        .chain(extra_bufs.iter_mut())
        .collect();
    let mut api_buffers: Vec<JxlOutputBuffer<'_>> = all_imgs
        .iter_mut()
        .map(|b| {
            let size = b.size();
            JxlOutputBuffer::from_image_rect_mut(
                b.get_rect_mut(Rect {
                    origin: (0, 0),
                    size,
                })
                .into_raw(),
            )
        })
        .collect();

    let decoder_with_frame_info = match decoder_with_image_info
        .process(&mut input)
        .expect("decode frame info")
    {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("NeedsMoreInput at frame info"),
    };
    let _ = match decoder_with_frame_info
        .process(&mut input, &mut api_buffers)
        .expect("decode pixels")
    {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("NeedsMoreInput at frame body"),
    };
    drop(api_buffers);

    // Flatten Image (rows of (W*3) f32 interleaved RGB) into a Vec<f32>.
    let (xs, ys) = color_buf.size();
    let mut out = Vec::with_capacity(xs * ys);
    for y in 0..ys {
        out.extend_from_slice(color_buf.row(y));
    }
    out
}
