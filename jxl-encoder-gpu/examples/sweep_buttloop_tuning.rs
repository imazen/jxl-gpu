// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Sweep harness for [`crate::forks::butteraugli_loop`] tuning constants.
//!
//! Per-image, per-config encode at fixed distance(s); writes one TSV row per
//! (image, distance, encoder, config) so a downstream Python script can
//! compute RD-pareto ratio against an independent cjxl reference.
//!
//! **Why a separate harness from `rd_pareto_vs_cjxl`**: this one repeats each
//! GPU encode under different override values for `CUR_POW_X1000_*` and
//! `MAX_INCREASE_X1000_*`. The cjxl side never moves so we don't re-run it
//! here — we read it from a previously-captured reference TSV.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder-gpu \
//!     --features 'cuda encoder butteraugli-loop' \
//!     --example sweep_buttloop_tuning -- \
//!       --image PATH... \
//!       --distance 3.0 [--distance ...] \
//!       --config name=cur_pow_low:cur_pow_high:max_inc_low:max_inc_high:split \
//!       --config ... \
//!       --out PATH.tsv \
//!       [--encoder gpu_e8] (default e8)
//!
//! Each --config gets a row tagged with its name in the `config_name` column.

#[cfg(not(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop")))]
fn main() {
    eprintln!("sweep_buttloop_tuning requires --features 'cuda encoder butteraugli-loop'");
    std::process::exit(2);
}

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn main() {
    use core::sync::atomic::Ordering;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
    use imgref::Img;
    use rgb::RGB;

    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::butteraugli_loop::{
        ButteraugliLoopGpu, CUR_POW_X1000_HIGH, CUR_POW_X1000_LOW, DISTANCE_SPLIT_X1000,
        MAX_INCREASE_X1000_HIGH, MAX_INCREASE_X1000_LOW,
    };
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

    type Backend = cubecl::cuda::CudaRuntime;

    /// (name, cur_pow_low, cur_pow_high, max_inc_low, max_inc_high, split)
    /// Use f32::NAN to mean "leave default" for any value.
    type Config = (String, f32, f32, f32, f32, f32);

    fn parse_config(s: &str) -> Result<Config, String> {
        // Format: name=cpL:cpH:miL:miH:split   (any field may be 'd' for default)
        let (name, rest) = s
            .split_once('=')
            .ok_or_else(|| format!("config missing '=': {s}"))?;
        let parts: Vec<&str> = rest.split(':').collect();
        if parts.len() != 5 {
            return Err(format!(
                "config needs 5 fields after '=', got {} in {s}",
                parts.len()
            ));
        }
        let parse_one = |p: &str| -> Result<f32, String> {
            if p == "d" || p == "default" {
                Ok(f32::NAN)
            } else {
                p.parse::<f32>().map_err(|e| format!("parse '{p}': {e}"))
            }
        };
        Ok((
            name.to_string(),
            parse_one(parts[0])?,
            parse_one(parts[1])?,
            parse_one(parts[2])?,
            parse_one(parts[3])?,
            parse_one(parts[4])?,
        ))
    }

    fn apply_config(c: &Config) {
        // i32::MIN = "use default"
        let store = |slot: &core::sync::atomic::AtomicI32, v: f32| {
            if v.is_nan() {
                slot.store(i32::MIN, Ordering::Relaxed);
            } else {
                slot.store((v * 1000.0).round() as i32, Ordering::Relaxed);
            }
        };
        store(&CUR_POW_X1000_LOW, c.1);
        store(&CUR_POW_X1000_HIGH, c.2);
        store(&MAX_INCREASE_X1000_LOW, c.3);
        store(&MAX_INCREASE_X1000_HIGH, c.4);
        store(&DISTANCE_SPLIT_X1000, c.5);
    }

    fn decode_jxl_linear(bs: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
        let reader = std::io::Cursor::new(bs);
        let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
        img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let frame = img.render_frame(0).ok()?;
        let fb = frame.image_all_channels();
        Some((fb.width(), fb.height(), fb.buf().to_vec()))
    }

    // ── Parse args ────────────────────────────────────────────────────
    let mut images: Vec<PathBuf> = Vec::new();
    let mut distances: Vec<f32> = Vec::new();
    let mut configs: Vec<Config> = Vec::new();
    let mut out_path: Option<PathBuf> = None;
    let mut encoder_label = "gpu_e8".to_string();
    let mut iters: usize = 2;
    let mut help = false;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--image" => {
                images.push(PathBuf::from(args.get(i + 1).expect("--image PATH")));
                i += 2;
            }
            "--distance" => {
                distances.push(args[i + 1].parse().expect("distance float"));
                i += 2;
            }
            "--config" => {
                configs.push(parse_config(&args[i + 1]).expect("parse_config"));
                i += 2;
            }
            "--out" => {
                out_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--encoder" => {
                encoder_label = args[i + 1].clone();
                i += 2;
            }
            "--iters" => {
                iters = args[i + 1].parse().expect("iters int");
                i += 2;
            }
            "-h" | "--help" => {
                help = true;
                i += 1;
            }
            other => panic!("unknown arg: {other}"),
        }
    }

    if help || images.is_empty() || distances.is_empty() || configs.is_empty() {
        eprintln!(
            "usage: sweep_buttloop_tuning --image PATH... --distance D... \\\n               --config name=cpL:cpH:miL:miH:split [--config ...] \\\n               --out PATH.tsv [--encoder gpu_e8] [--iters 2]\n\nUse 'd' for any field to leave its compile-time default. Defaults are\ncur_pow=0.5, max_increase=1.3, split=2.0.\n"
        );
        std::process::exit(if help { 0 } else { 2 });
    }

    let out_path = out_path.expect("--out PATH.tsv");
    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let mut tsv =
        std::fs::File::create(&out_path).unwrap_or_else(|e| panic!("create {out_path:?}: {e}"));
    let now = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    writeln!(tsv, "# sweep_buttloop_tuning TSV").unwrap();
    writeln!(tsv, "# generated_utc\t{now}").unwrap();
    writeln!(tsv, "# encoder\t{encoder_label}").unwrap();
    writeln!(tsv, "# iters\t{iters}").unwrap();
    writeln!(tsv, "# n_images\t{}", images.len()).unwrap();
    writeln!(tsv, "# n_distances\t{}", distances.len()).unwrap();
    writeln!(tsv, "# n_configs\t{}", configs.len()).unwrap();
    for c in &configs {
        writeln!(
            tsv,
            "# config\t{}\tcpL={}\tcpH={}\tmiL={}\tmiH={}\tsplit={}",
            c.0, c.1, c.2, c.3, c.4, c.5
        )
        .unwrap();
    }
    writeln!(
        tsv,
        "image\twidth\theight\tencoder\tdistance\tconfig_name\tbytes\tmeasured_bfly\tencode_ms"
    )
    .unwrap();

    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    eprintln!(
        "[setup] {} images × {} distances × {} configs = {} encodes",
        images.len(),
        distances.len(),
        configs.len(),
        images.len() * distances.len() * configs.len()
    );

    for img_path in &images {
        let stem = img_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let short = if stem.len() > 12 && stem.chars().all(|c| c.is_ascii_hexdigit()) {
            stem[..8].to_string()
        } else {
            stem.clone()
        };
        eprintln!("[image] {} ← {}", short, img_path.display());

        let src = match image::open(img_path) {
            Ok(im) => im.to_rgb8(),
            Err(e) => {
                eprintln!("  ERROR open: {e} — skipping");
                continue;
            }
        };
        let (w, h) = src.dimensions();
        if w < 16 || h < 16 {
            eprintln!("  too small, skipping");
            continue;
        }

        let pixels_u8: Vec<u8> = src.into_raw();
        let n = (w as usize) * (h as usize);
        let mut r_lin = Vec::with_capacity(n);
        let mut g_lin = Vec::with_capacity(n);
        let mut b_lin = Vec::with_capacity(n);
        for chunk in pixels_u8.chunks_exact(3) {
            r_lin.push(srgb_to_linear(chunk[0]));
            g_lin.push(srgb_to_linear(chunk[1]));
            b_lin.push(srgb_to_linear(chunk[2]));
        }
        let orig_pixels_rgb: Vec<RGB<f32>> = (0..n)
            .map(|i| RGB::new(r_lin[i], g_lin[i], b_lin[i]))
            .collect();
        let orig_lin_img = Img::new(orig_pixels_rgb, w as usize, h as usize);
        let bfly_params = ButteraugliParams::default();

        let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);
        let mut bg: ButteraugliLoopGpu<Backend> = ButteraugliLoopGpu::new_multires(&enc, w, h);
        bg.set_reference(&pixels_u8).expect("bg.set_reference");

        for &dist in &distances {
            for cfg in &configs {
                apply_config(cfg);
                let t0 = Instant::now();
                let bs = match enc.encode_lossy_to_bitstream_via_precomputed_with_butteraugli(
                    &lossy, &mut bg, &r_lin, &g_lin, &b_lin, &pixels_u8, dist, iters,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!(
                            "    encode failed for {} @ d={} cfg={}: {e:?}",
                            short, dist, cfg.0
                        );
                        continue;
                    }
                };
                let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let (dw, dh, dec_lin) = match decode_jxl_linear(&bs) {
                    Some(t) => t,
                    None => {
                        eprintln!("    decode failed");
                        continue;
                    }
                };
                let dec_pixels: Vec<RGB<f32>> = dec_lin
                    .chunks_exact(3)
                    .map(|c| RGB::new(c[0], c[1], c[2]))
                    .collect();
                let dec_img = Img::new(dec_pixels, dw, dh);
                let bfly =
                    butteraugli_linear(orig_lin_img.as_ref(), dec_img.as_ref(), &bfly_params)
                        .map(|s| s.score as f64)
                        .unwrap_or(f64::NAN);
                writeln!(
                    tsv,
                    "{short}\t{w}\t{h}\t{encoder_label}\t{dist}\t{}\t{}\t{:.6}\t{:.2}",
                    cfg.0,
                    bs.len(),
                    bfly,
                    dt_ms
                )
                .unwrap();
                tsv.flush().ok();
                eprintln!(
                    "  {} d={} cfg={:<20} bytes={} bfly={:.3} t={:.0}ms",
                    short,
                    dist,
                    cfg.0,
                    bs.len(),
                    bfly,
                    dt_ms
                );
            }
        }
    }

    eprintln!("\n[done] wrote {}", out_path.display());
}
