// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Quality-regression gate: GPU output may not score lower than CPU
//! output by more than the per-distance tolerance (see
//! [`tolerance_for_distance`]) on any (image, effort, distance) cell.
//!
//! Encodes the same input through both
//! `jxl-encoder` (CPU, via `LossyConfig::with_effort(...)`) and
//! `jxl-encoder-gpu` (GPU, via the matching
//! `encode_lossy_to_bitstream_via_precomputed*` entry points), decodes
//! both bitstreams in linear sRGB via jxl-oxide, and asserts
//! `gpu_zensim >= cpu_zensim - tolerance_for_distance(d)` per cell.
//!
//! ## Why zensim
//!
//! - It's a perceptual metric (sibling to SSIMULACRA2 with stronger
//!   correlation to human MOS in the project's CID22/KADID/TID
//!   evaluation tables — see zensim README).
//! - It runs in ~22 ms at 1080p (faster than butteraugli-rs or
//!   ssimulacra2-rs), so 30+ cells finish in under a minute.
//! - It's the project's preferred regression metric (CLAUDE.md
//!   "use Rust zensim/butteraugli/ssim2" rule).
//!
//! ## Tolerance
//!
//! Distance-keyed tolerance — the gap between GPU and CPU zensim
//! grows with distance because the cubecl-vs-jxl_simd DCT FP
//! precision difference (~2% chroma AC coefs flipping by 1 near
//! rounding ties — see `encoder.rs:2290`) compounds with quantizer
//! aggressiveness. Calibrated 2026-05-15 from the
//! `cpu_vs_gpu_e7_e8_e9_2026-05-15.tsv` baseline:
//!
//! | distance | observed worst Δ | tolerance |
//! |----------|------------------|-----------|
//! | 1.0      | -3.6             | 4.0       |
//! | 2.0      | -6.4             | 7.0       |
//!
//! Tolerances are 1 zensim-point above the worst observed delta on
//! the 4-image baseline at the time of calibration. The test catches
//! NEW regressions (any cell exceeding the tolerance) — the existing
//! delta is a known quality gap documented in the bench TSV header
//! caveats. Tightening these tolerances over time as the GPU encoder
//! catches up to CPU is a CLAUDE.md "never relax tests" win, not a
//! relaxation.
//!
//! ## What this is NOT
//!
//! This test does NOT enforce byte-identical output. cubecl-vs-jxl_simd
//! DCT precision means the bitstreams differ by up to ~2% chroma AC
//! coefficients near rounding ties. The test asserts that the
//! perceptual quality of the decoded output is within tolerance —
//! which is the actual user-visible contract.
//!
//! ## Running
//!
//! ```bash
//! cargo test -p jxl-encoder-gpu \
//!   --features gpu-zensim-regress \
//!   --test cpu_vs_gpu_zensim_regress -- --nocapture
//! ```
//!
//! Per CLAUDE.md "NO 'GRACEFUL SKIPS' IN TESTS": if the corpus tree is
//! missing the test PANICS rather than silently passing. The skip
//! decision lives at the cargo invocation level (don't enable the
//! `gpu-zensim-regress` feature).

#![cfg(feature = "gpu-zensim-regress")]

use butteraugli::srgb_to_linear;
use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use jxl_encoder_gpu::encoder::GpuEncoder;
use jxl_encoder_gpu::forks::butteraugli_loop::ButteraugliLoopGpu;
use jxl_encoder_gpu::lossy_encoder::LossyEncoder;
use zensim::{RgbSlice, Zensim, ZensimProfile};

type Backend = cubecl::cuda::CudaRuntime;

const CORPUS_ROOT: &str = "/home/lilith/work/codec-corpus";

/// Per-distance tolerance in zensim points (full scale 0..=100). The
/// cubecl-vs-jxl_simd DCT FP gap compounds with quantizer
/// aggressiveness, so tolerance scales with distance. See module
/// docstring for the full calibration table.
fn tolerance_for_distance(d: f32) -> f64 {
    if d <= 1.0 { 4.0 } else { 7.0 }
}

/// Image set used by the regress gate. Small intentionally — this test
/// runs 4 corpus images × 3 efforts × 2 distances = 24 cells × 2
/// encodes = 48 encodes. With CPU e9 ~5s/MP and GPU e9 ~1.5s/MP at
/// 1024², the suite completes in ~5-8 minutes.
///
/// The set covers the four BestOfBothPath quadrants:
/// - 02809272 — RefineDct8 photo (the "easy" case)
/// - 22ea12c9 — RefineStratSearch photo (DCT64 actively wins)
/// - 1cba10ad — RefineStratSearch photo (DCT64 wins on textured detail)
/// - 1e2f9d41 — RefineDct8 photo (uniform actively wins)
const IMAGES: &[&str] = &[
    "clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
    "clic2025-1024/22ea12c903e41583.png",
    "clic2025-1024/1cba10ad9bb4ced57e42f7656c5f2a58d32dc6bad084957d2f8d1c78e0fcd224.png",
    "clic2025-1024/1e2f9d41529197f1.png",
];

const EFFORTS: &[u8] = &[7, 8, 9];
const DISTANCES: &[f32] = &[1.0_f32, 2.0];

#[test]
fn gpu_zensim_does_not_regress_vs_cpu() {
    if !std::path::Path::new(CORPUS_ROOT).exists() {
        panic!(
            "cpu_vs_gpu_zensim_regress: corpus tree not found at {CORPUS_ROOT}. \
             This test requires the codec-corpus tree; either symlink it at \
             the expected path or run without --features gpu-zensim-regress."
        );
    }

    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let zensim = Zensim::new(ZensimProfile::latest()).with_max_pixels(64 * 1024 * 1024);
    let cpu_limits = Limits::default().with_max_memory_bytes(64 * 1024 * 1024 * 1024);

    let mut failures: Vec<String> = Vec::new();
    let mut ran = 0usize;

    for subpath in IMAGES {
        let mut full_path = format!("{CORPUS_ROOT}/{subpath}");
        if !std::path::Path::new(&full_path).exists() {
            // CLIC fuzzy-match by hash prefix (variants exist in some
            // corpus snapshots).
            let parent = std::path::Path::new(&full_path)
                .parent()
                .map(|p| p.to_path_buf());
            let stem_prefix = std::path::Path::new(subpath)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.chars().take(16).collect::<String>())
                .unwrap_or_default();
            if let Some(parent) = parent
                && let Ok(entries) = std::fs::read_dir(&parent)
            {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.starts_with(&stem_prefix) && name.ends_with(".png") {
                        full_path = entry.path().to_string_lossy().into_owned();
                        break;
                    }
                }
            }
        }
        if !std::path::Path::new(&full_path).exists() {
            failures.push(format!("missing image: {subpath}"));
            continue;
        }

        let img = match image::open(&full_path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                failures.push(format!("image load failed for {subpath}: {e}"));
                continue;
            }
        };
        let (w, h) = img.dimensions();
        let pixels_u8: Vec<u8> = img.into_raw();
        let n = (w * h) as usize;

        let mut r_lin = Vec::with_capacity(n);
        let mut g_lin = Vec::with_capacity(n);
        let mut b_lin = Vec::with_capacity(n);
        for chunk in pixels_u8.chunks_exact(3) {
            r_lin.push(srgb_to_linear(chunk[0]));
            g_lin.push(srgb_to_linear(chunk[1]));
            b_lin.push(srgb_to_linear(chunk[2]));
        }

        let orig_srgb_arr: Vec<[u8; 3]> = pixels_u8
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let orig_zensim = RgbSlice::new(&orig_srgb_arr, w as usize, h as usize);

        // Per-image GPU resources.
        let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);
        let mut bg: ButteraugliLoopGpu<Backend> = ButteraugliLoopGpu::new_multires(&enc, w, h);
        bg.set_reference(&pixels_u8).expect("bg.set_reference");

        for &effort in EFFORTS {
            for &dist in DISTANCES {
                ran += 1;

                let cpu_cfg = LossyConfig::new(dist).with_effort(effort);
                let cpu_bytes = match cpu_cfg
                    .encode_request(w, h, PixelLayout::Rgb8)
                    .with_limits(&cpu_limits)
                    .encode(&pixels_u8)
                    .map_err(|e| e.decompose().0)
                {
                    Ok(b) => b,
                    Err(e) => {
                        failures.push(format!(
                            "{subpath} e{effort} d={dist}: cpu encode failed: {e:?}"
                        ));
                        continue;
                    }
                };

                let gpu_bytes_res = match effort {
                    7 => enc
                        .encode_lossy_to_bitstream_via_precomputed_from_u8(&lossy, &pixels_u8, dist)
                        .map_err(|e| format!("{e:?}")),
                    8 => enc
                        .encode_lossy_to_bitstream_via_precomputed_with_butteraugli(
                            &lossy, &mut bg, &r_lin, &g_lin, &b_lin, &pixels_u8, dist, 2,
                        )
                        .map_err(|e| format!("{e:?}")),
                    _ => enc
                        .encode_lossy_to_bitstream_via_precomputed_with_butteraugli(
                            &lossy, &mut bg, &r_lin, &g_lin, &b_lin, &pixels_u8, dist, 4,
                        )
                        .map_err(|e| format!("{e:?}")),
                };
                let gpu_bytes = match gpu_bytes_res {
                    Ok(b) => b,
                    Err(e) => {
                        failures.push(format!(
                            "{subpath} e{effort} d={dist}: gpu encode failed: {e}"
                        ));
                        continue;
                    }
                };

                let (Some(cpu_zensim), Some(gpu_zensim)) = (
                    decode_and_zensim(&cpu_bytes, w, h, &zensim, &orig_zensim),
                    decode_and_zensim(&gpu_bytes, w, h, &zensim, &orig_zensim),
                ) else {
                    failures.push(format!(
                        "{subpath} e{effort} d={dist}: decode/zensim failed (cpu_bytes={} gpu_bytes={})",
                        cpu_bytes.len(),
                        gpu_bytes.len(),
                    ));
                    continue;
                };

                let delta = gpu_zensim - cpu_zensim;
                let tol = tolerance_for_distance(dist);
                let cpu_b = cpu_bytes.len();
                let gpu_b = gpu_bytes.len();
                eprintln!(
                    "  {subpath} e{effort} d={dist}: \
                     cpu zensim={cpu_zensim:.3} ({cpu_b}B) \
                     gpu zensim={gpu_zensim:.3} ({gpu_b}B) \
                     Δ={delta:+.3} (tol={tol:.2})",
                );

                if delta < -tol {
                    failures.push(format!(
                        "REGRESS {subpath} e{effort} d={dist}: \
                         cpu={cpu_zensim:.3} gpu={gpu_zensim:.3} Δ={delta:+.3} (tol={tol:.2})",
                    ));
                }
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "cpu_vs_gpu_zensim_regress: {} failure(s) of {} cell(s) ran:\n  {}",
            failures.len(),
            ran,
            failures.join("\n  "),
        );
    }
    assert!(ran > 0, "no cells ran");
    eprintln!("cpu_vs_gpu_zensim_regress: {ran} cell(s) all within distance-keyed tolerance",);
}

/// Decode a JXL bitstream and compute zensim against the source.
/// Returns `None` on decode failure.
fn decode_and_zensim(
    bytes: &[u8],
    src_w: u32,
    src_h: u32,
    zensim_engine: &Zensim,
    orig_zensim: &RgbSlice,
) -> Option<f64> {
    let (dw, dh, decoded_linear) = decode_jxl_linear(bytes)?;
    if dw != src_w as usize || dh != src_h as usize {
        return None;
    }
    let dec_srgb_arr: Vec<[u8; 3]> = decoded_linear
        .chunks_exact(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_zensim = RgbSlice::new(&dec_srgb_arr, dw, dh);
    zensim_engine
        .compute(orig_zensim, &dec_zensim)
        .ok()
        .map(|r| r.score())
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

/// Decode a JXL bitstream to interleaved linear RGB f32. Tries
/// jxl-oxide first (`request_color_encoding(srgb_linear(Relative))`) for
/// the metadata-immune path documented in project CLAUDE.md; falls back
/// to jxl-rs for the known jxl-oxide 0.12.5 multi-group ANS modular
/// limitation.
fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    if let Some(t) = decode_via_jxl_oxide(bytes) {
        return Some(t);
    }
    decode_via_jxl_rs(bytes)
}

fn decode_via_jxl_oxide(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = std::io::Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let frame = img.render_frame(0).ok()?;
    let fb = frame.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn decode_via_jxl_rs(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    use jxl::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input: &[u8] = bytes;
    let initialized: JxlDecoder<states::Initialized> =
        JxlDecoder::new(JxlDecoderOptions::default());
    let mut with_image_info = match initialized.process(&mut input).ok()? {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => return None,
    };
    let basic = with_image_info.basic_info().clone();
    let (w, h) = (basic.size.0, basic.size.1);
    let default_fmt = with_image_info.current_pixel_format().clone();
    let requested_fmt = JxlPixelFormat {
        color_type: default_fmt.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: default_fmt
            .extra_channel_format
            .iter()
            .map(|_| Some(JxlDataFormat::f32()))
            .collect(),
    };
    with_image_info.set_pixel_format(requested_fmt);
    let pixel_format = with_image_info.current_pixel_format().clone();
    let num_channels = pixel_format.color_type.samples_per_pixel();
    if num_channels != 3 && num_channels != 4 {
        return None;
    }
    let mut color_buf = Image::<f32>::new_with_value((w * num_channels, h), f32::NAN).ok()?;
    let extra_count = pixel_format
        .extra_channel_format
        .iter()
        .filter(|x| x.is_some())
        .count();
    let mut extra_bufs: Vec<Image<f32>> = (0..extra_count)
        .filter_map(|_| Image::<f32>::new_with_value((w, h), f32::NAN).ok())
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
    let with_frame_info = match with_image_info.process(&mut input).ok()? {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => return None,
    };
    let _back = match with_frame_info.process(&mut input, &mut api_buffers).ok()? {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => return None,
    };
    drop(api_buffers);
    let (xs, ys) = color_buf.size();
    if xs != w * num_channels || ys != h {
        return None;
    }
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let row = color_buf.row(y);
        for x in 0..w {
            let base = x * num_channels;
            let r = row[base];
            let g = row[base + 1];
            let b = row[base + 2];
            if !r.is_finite() || !g.is_finite() || !b.is_finite() {
                return None;
            }
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    Some((w, h, out))
}
