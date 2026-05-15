// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Real RD-pareto comparison: GPU encoder (e7/e8/e9) vs cjxl (e7/e8/e9).
//!
//! Per-image, per-distance, per-encoder data points capturing **measured**
//! butteraugli + SSIM2 — not the encoder's claimed quality, the actually-
//! decoded quality. Outputs a TSV that can be loaded into pandas /
//! analysed for RD-pareto and RD-time pareto.
//!
//! Why this matters: absolute-byte comparisons at the same `--distance`
//! setting are misleading. cjxl and our encoder may emit different
//! *achieved* qualities at the same nominal distance. The only valid
//! comparison is: at the same MEASURED butteraugli, who has fewer bytes?
//!
//! Decode is via jxl-oxide with `request_color_encoding(srgb_linear)` per
//! project CLAUDE.md — this enforces consistent sRGB transfer function
//! on both bitstreams, immune to PNG color metadata mismatches that
//! plague `butteraugli_main`.
//!
//! Sweeps:
//! - Distances: per CLAUDE.md sweep rule, include low-q (web-focused).
//!   Default {0.25, 0.5, 1.0, 2.0, 3.0, 5.0}.
//! - Encoders: gpu_e7, gpu_e8, gpu_e9, cjxl_e7, cjxl_e8 (e9 if available).
//! - Images: configurable via --image (repeatable) or --corpus DIR.
//!
//! TSV columns (tab-separated):
//!   image  width  height  encoder  effort  distance  bytes  measured_bfly
//!   measured_ssim2  encode_ms
//!
//! Usage:
//!   cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder butteraugli-loop' \
//!     --example rd_pareto_vs_cjxl -- \
//!     --out /home/lilith/work/zen/jxl-encoder-gpu/benchmarks/rd_pareto_$(date +%F).tsv \
//!     [--image PATH]... \
//!     [--distance D]... (default: 0.25 0.5 1.0 2.0 3.0 5.0) \
//!     [--encoder NAME]... (default: gpu_e7 gpu_e8 gpu_e9 cjxl_e7 cjxl_e8) \
//!     [--cjxl PATH] (default: /home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl) \
//!     [--runs N] (default: 1; >1 gives timing reproducibility)

#[cfg(not(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop")))]
fn main() {
    eprintln!("rd_pareto_vs_cjxl requires --features 'cuda encoder butteraugli-loop'");
    std::process::exit(2);
}

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn main() {
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
    use imgref::Img;
    use rgb::RGB;

    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::butteraugli_loop::ButteraugliLoopGpu;
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

    type Backend = cubecl::cuda::CudaRuntime;

    // ── Parse args ────────────────────────────────────────────────────
    let mut args: Vec<String> = std::env::args().collect();
    // Strip program name.
    args.remove(0);

    let mut images: Vec<PathBuf> = Vec::new();
    let mut distances: Vec<f32> = Vec::new();
    let mut encoders: Vec<String> = Vec::new();
    let mut cjxl_path: String = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl".to_string();
    let mut out_path: Option<PathBuf> = None;
    let mut runs: usize = 1;
    let mut help = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--image" => {
                let p = args
                    .get(i + 1)
                    .unwrap_or_else(|| panic!("--image requires a path"));
                images.push(PathBuf::from(p));
                i += 2;
            }
            "--corpus" => {
                let dir = args
                    .get(i + 1)
                    .unwrap_or_else(|| panic!("--corpus requires a directory"));
                let dir = PathBuf::from(dir);
                let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
                    .unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"))
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| {
                        matches!(
                            p.extension().and_then(|s| s.to_str()),
                            Some("png") | Some("PNG")
                        )
                    })
                    .collect();
                entries.sort();
                images.extend(entries);
                i += 2;
            }
            "--distance" => {
                let d: f32 = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| panic!("--distance needs a float"));
                distances.push(d);
                i += 2;
            }
            "--encoder" => {
                let e = args
                    .get(i + 1)
                    .unwrap_or_else(|| panic!("--encoder needs a name"));
                encoders.push(e.clone());
                i += 2;
            }
            "--cjxl" => {
                cjxl_path = args[i + 1].clone();
                i += 2;
            }
            "--out" => {
                out_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--runs" => {
                runs = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| panic!("--runs needs an int"));
                i += 2;
            }
            "-h" | "--help" => {
                help = true;
                i += 1;
            }
            other => panic!("unknown arg: {other}"),
        }
    }

    if help || images.is_empty() {
        eprintln!(
            "Usage:\n  cargo run --release -p jxl-encoder-gpu \\\n    --features 'cuda encoder butteraugli-loop' \\\n    --example rd_pareto_vs_cjxl -- \\\n      --out PATH.tsv \\\n      [--image PATH]... \\\n      [--corpus DIR] \\\n      [--distance D]... (default: 0.25 0.5 1.0 2.0 3.0 5.0) \\\n      [--encoder NAME]... (default: gpu_e7 gpu_e8 gpu_e9 cjxl_e7 cjxl_e8) \\\n      [--runs N] (default: 1)\n"
        );
        if images.is_empty() && !help {
            eprintln!("ERROR: --image PATH (or --corpus DIR) required");
            std::process::exit(2);
        }
        return;
    }

    // Defaults — apply CLAUDE.md sweep rules (web-focused: dense low-q).
    if distances.is_empty() {
        distances = vec![0.25_f32, 0.5, 1.0, 2.0, 3.0, 5.0];
    }
    if encoders.is_empty() {
        encoders = vec![
            "gpu_e7".to_string(),
            "gpu_e8".to_string(),
            "gpu_e9".to_string(),
            "cjxl_e7".to_string(),
            "cjxl_e8".to_string(),
        ];
    }

    let out_path = out_path.unwrap_or_else(|| {
        PathBuf::from(format!(
            "/tmp/rd_pareto_{}.tsv",
            chrono_today_or_placeholder()
        ))
    });
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // ── Header ────────────────────────────────────────────────────────
    let git_commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into());
    let hostname = std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into());
    let now_utc = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into());
    let cmdline = std::env::args().collect::<Vec<_>>().join(" ");

    let mut tsv =
        std::fs::File::create(&out_path).unwrap_or_else(|e| panic!("create {out_path:?}: {e}"));
    writeln!(tsv, "# rd_pareto_vs_cjxl TSV").unwrap();
    writeln!(tsv, "# generated_utc\t{now_utc}").unwrap();
    writeln!(tsv, "# git_commit\t{git_commit}").unwrap();
    writeln!(tsv, "# hostname\t{hostname}").unwrap();
    writeln!(tsv, "# cmdline\t{cmdline}").unwrap();
    writeln!(tsv, "# n_images\t{}", images.len()).unwrap();
    writeln!(tsv, "# distances\t{:?}", distances).unwrap();
    writeln!(tsv, "# encoders\t{:?}", encoders).unwrap();
    writeln!(tsv, "# runs\t{runs}").unwrap();
    writeln!(
        tsv,
        "image\twidth\theight\tencoder\teffort\tdistance\tbytes\tmeasured_bfly\tmeasured_ssim2\tencode_ms"
    )
    .unwrap();
    tsv.flush().ok();

    eprintln!(
        "[setup] {} images × {} distances × {} encoders × {} runs = {} encodes",
        images.len(),
        distances.len(),
        encoders.len(),
        runs,
        images.len() * distances.len() * encoders.len() * runs
    );
    eprintln!("[setup] out={}", out_path.display());

    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let total = images.len() * distances.len() * encoders.len();
    let mut done = 0usize;

    for img_path in &images {
        let img_stem = img_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        // Shorten long hex stems to first 8 chars for readability.
        let short_name = if img_stem.len() > 12 && img_stem.chars().all(|c| c.is_ascii_hexdigit()) {
            img_stem[..8].to_string()
        } else {
            img_stem.clone()
        };

        eprintln!("\n[image] {} ← {}", short_name, img_path.display());

        let src = match image::open(img_path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                eprintln!("  ERROR: open {img_path:?}: {e} — skipping");
                continue;
            }
        };
        let (w, h) = src.dimensions();
        if w < 16 || h < 16 {
            eprintln!("  WARN: {w}×{h} too small (LossyEncoder needs >=16) — skipping");
            continue;
        }

        // sRGB u8 for cjxl input + SSIM2 reference.
        let pixels_u8: Vec<u8> = src.clone().into_raw();

        // Linear f32 planar for GPU encoder.
        let n = (w as usize) * (h as usize);
        let mut r_lin = Vec::with_capacity(n);
        let mut g_lin = Vec::with_capacity(n);
        let mut b_lin = Vec::with_capacity(n);
        for chunk in pixels_u8.chunks_exact(3) {
            r_lin.push(srgb_to_linear(chunk[0]));
            g_lin.push(srgb_to_linear(chunk[1]));
            b_lin.push(srgb_to_linear(chunk[2]));
        }

        // Interleaved linear RGB for butteraugli reference.
        let orig_pixels_rgb: Vec<RGB<f32>> = (0..n)
            .map(|i| RGB::new(r_lin[i], g_lin[i], b_lin[i]))
            .collect();
        let orig_lin_img = Img::new(orig_pixels_rgb, w as usize, h as usize);

        // sRGB u8 reference for SSIM2.
        let orig_srgb: Vec<[u8; 3]> = pixels_u8
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let orig_srgb_img = Img::new(orig_srgb, w as usize, h as usize);

        let bfly_params = ButteraugliParams::default();

        // ── Build GPU LossyEncoder once per image (per-size resource) ─
        let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);
        let mut bg: ButteraugliLoopGpu<Backend> = ButteraugliLoopGpu::new_multires(&enc, w, h);
        bg.set_reference(&pixels_u8).expect("bg.set_reference");

        for &dist in &distances {
            for enc_name in &encoders {
                done += 1;
                eprintln!("  [{done}/{total}] {} d={} {}", short_name, dist, enc_name);

                // Encode N runs to get a stable time. Bytes are
                // deterministic so we only need a single bitstream for
                // metric measurement; we just record the min wall time.
                let mut min_ms = f64::INFINITY;
                let mut bytes_opt: Option<Vec<u8>> = None;

                for run in 0..runs {
                    let t0 = Instant::now();
                    let bs_result: Result<Vec<u8>, String> = match enc_name.as_str() {
                        "gpu_e7" => enc
                            .encode_lossy_to_bitstream_via_precomputed(
                                &lossy, &r_lin, &g_lin, &b_lin, dist,
                            )
                            .map_err(|e| format!("{e:?}")),
                        "gpu_e8" => enc
                            .encode_lossy_to_bitstream_via_precomputed_with_butteraugli(
                                &lossy, &mut bg, &r_lin, &g_lin, &b_lin, &pixels_u8, dist, 2,
                            )
                            .map_err(|e| format!("{e:?}")),
                        "gpu_e9" => enc
                            .encode_lossy_to_bitstream_via_precomputed_with_butteraugli(
                                &lossy, &mut bg, &r_lin, &g_lin, &b_lin, &pixels_u8, dist, 4,
                            )
                            .map_err(|e| format!("{e:?}")),
                        cjxl @ ("cjxl_e7" | "cjxl_e8" | "cjxl_e9") => {
                            let effort = match cjxl {
                                "cjxl_e7" => 7,
                                "cjxl_e8" => 8,
                                _ => 9,
                            };
                            run_cjxl(&cjxl_path, img_path, dist, effort).map_err(|e| e.to_string())
                        }
                        other => {
                            eprintln!("    unknown encoder: {other}");
                            continue;
                        }
                    };
                    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    match bs_result {
                        Ok(bs) => {
                            min_ms = min_ms.min(dt_ms);
                            // Reuse the last bitstream for metric measurement.
                            // (deterministic across runs for cjxl + our encoder).
                            if run + 1 == runs {
                                bytes_opt = Some(bs);
                            }
                        }
                        Err(e) => {
                            eprintln!("    encode failed: {e}");
                            break;
                        }
                    }
                }

                let Some(bytes) = bytes_opt else { continue };
                let bs_len = bytes.len();

                // ── Decode via jxl-oxide in linear sRGB ────────────────
                let (dw, dh, decoded_linear) = match decode_jxl_linear(&bytes) {
                    Some(t) => t,
                    None => {
                        eprintln!("    decode failed");
                        continue;
                    }
                };
                if dw != w as usize || dh != h as usize {
                    eprintln!("    WARN: decoded dims {dw}×{dh} != source {}×{}", w, h);
                }

                // Butteraugli on linear RGB.
                let dec_lin_pixels: Vec<RGB<f32>> = decoded_linear
                    .chunks_exact(3)
                    .map(|c| RGB::new(c[0], c[1], c[2]))
                    .collect();
                let dec_lin_img = Img::new(dec_lin_pixels, dw, dh);
                let bfly =
                    butteraugli_linear(orig_lin_img.as_ref(), dec_lin_img.as_ref(), &bfly_params)
                        .map(|s| s.score as f64)
                        .unwrap_or(f64::NAN);

                // SSIM2 on sRGB u8 (with correct sRGB TF).
                let decoded_srgb: Vec<[u8; 3]> = decoded_linear
                    .chunks_exact(3)
                    .map(|c| {
                        [
                            linear_to_srgb_u8(c[0]),
                            linear_to_srgb_u8(c[1]),
                            linear_to_srgb_u8(c[2]),
                        ]
                    })
                    .collect();
                let dec_srgb_img = Img::new(decoded_srgb, dw, dh);
                let ssim2 =
                    fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
                        .unwrap_or(f64::NAN);

                let effort_num = encoder_effort(enc_name);

                writeln!(
                    tsv,
                    "{short_name}\t{w}\t{h}\t{enc_name}\t{effort_num}\t{dist}\t{bs_len}\t{bfly:.6}\t{ssim2:.6}\t{min_ms:.2}"
                )
                .unwrap();
                tsv.flush().ok();

                eprintln!("      bytes={bs_len} bfly={bfly:.3} ssim2={ssim2:.2} t={min_ms:.1}ms");
            }
        }
    }

    eprintln!("\n[done] wrote {}", out_path.display());
}

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn encoder_effort(name: &str) -> u32 {
    match name {
        "gpu_e7" | "cjxl_e7" => 7,
        "gpu_e8" | "cjxl_e8" => 8,
        "gpu_e9" | "cjxl_e9" => 9,
        _ => 0,
    }
}

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn chrono_today_or_placeholder() -> String {
    std::process::Command::new("date")
        .args(["+%F"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "today".into())
}

/// Run cjxl as a subprocess and return the bitstream bytes.
#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn run_cjxl(
    cjxl_path: &str,
    input: &std::path::Path,
    distance: f32,
    effort: u32,
) -> Result<Vec<u8>, String> {
    // Per-call tmp output. Unique-enough via pid + nanos.
    let pid = std::process::id();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = std::env::temp_dir().join(format!("rd_pareto_cjxl_{pid}_{now}.jxl"));

    let out = std::process::Command::new(cjxl_path)
        .arg(input)
        .arg(&tmp)
        .arg("-d")
        .arg(distance.to_string())
        .arg("-e")
        .arg(effort.to_string())
        .arg("--quiet")
        .output()
        .map_err(|e| format!("spawn cjxl: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("cjxl status={}: {stderr}", out.status));
    }

    let bytes = std::fs::read(&tmp).map_err(|e| format!("read tmp: {e}"))?;
    let _ = std::fs::remove_file(&tmp);
    Ok(bytes)
}

/// Linear-light sRGB f32 → sRGB u8 using the correct sRGB transfer
/// function (linear segment near black, exponent 1/2.4 above). Per
/// project CLAUDE.md, NEVER use gamma 2.2 here — it'd produce ~5% darks
/// divergence vs the reference and bogus SSIM2 / butteraugli scores.
#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

/// Decode a JXL bitstream to interleaved linear-RGB f32.
///
/// Tries jxl-oxide first (request_color_encoding(srgb_linear(Relative)))
/// for the metadata-immune path documented in project CLAUDE.md.
///
/// Falls back to jxl-rs (the PRIMARY decoder per project CLAUDE.md) if
/// jxl-oxide rejects the bitstream — this catches the known jxl-oxide
/// 0.12.5 limitation with ANS in multi-group modular frames (unexpected
/// EOF). jxl-rs's `JxlDataFormat::f32()` decodes to LINEAR f32 by
/// default (per `examples/jxl_rs_roundtrip.rs:186`), so the output is
/// already in the format butteraugli + our SSIM2 path expect.
///
/// Returns (width, height, interleaved RGB f32 in linear light).
#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    if let Some(t) = decode_via_jxl_oxide(bytes) {
        return Some(t);
    }
    decode_via_jxl_rs(bytes)
}

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
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

/// jxl-rs fallback decode. Returns interleaved linear f32 RGB. Mirrors
/// the pattern in `examples/jxl_rs_roundtrip.rs` exactly — the
/// `JxlDataFormat::f32()` request gives LINEAR f32 (not sRGB) per the
/// comment at line 186 of that file. We only handle the color channels
/// (no extras) since metric measurement doesn't need alpha here.
#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
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

    // Project an interleaved RGB output. jxl-rs writes
    // (w * num_channels, h) — channels interleaved per row. For RGBA
    // we drop alpha. Validate finite values; bail if NaN.
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
