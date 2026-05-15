// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Apples-to-apples CPU vs GPU bench across e7 / e8 / e9.
//!
//! For every (image, effort, distance) cell, encodes the same input
//! through both the CPU `jxl-encoder` API and the GPU
//! `jxl-encoder-gpu` `encode_lossy_to_bitstream_via_precomputed*` path,
//! decodes both bitstreams via jxl-rs in linear sRGB, and reports
//! butteraugli + ssim2 + zensim. Output is a TSV suitable for downstream
//! analysis (pandas / matplotlib / pareto plots).
//!
//! ## Apples-to-apples discipline
//!
//! 1. **Identical input pixel data.** Both encoders receive the SAME
//!    sRGB u8 RGB pixel bytes (decoded from the source PNG via the
//!    `image` crate, no re-encode). The CPU encoder takes
//!    `PixelLayout::Rgb8`; the GPU encoder takes the same `&[u8]`
//!    via `encode_lossy_to_bitstream_via_precomputed_from_u8` and
//!    converts on-GPU. If both use the IEC 61966-2-1 sRGB EOTF
//!    (linear segment + powf 2.4), the linear values match to within
//!    f32 precision.
//! 2. **Identical effort↔buttloop mapping.** CPU `LossyConfig::with_effort(7)`
//!    sets `butteraugli_iters=0` (libjxl `enc_adaptive_quantization.cc:1282`
//!    gates `FindBestQuantization` at `speed_tier <= kKitten`, i.e.
//!    effort >= 8), CPU e8 = 2 iters, CPU e9 = 4 iters. GPU mirrors
//!    EXACTLY: e7 = `_from_u8` no-loop path, e8 =
//!    `_with_butteraugli` 2 iters, e9 = `_with_butteraugli` 4 iters.
//! 3. **Identical distance.** Same `--distance D` flag; no per-encoder
//!    rescaling.
//! 4. **Identical decode path.** Both bitstreams decoded via jxl-rs in
//!    linear sRGB f32 (per project CLAUDE.md "PNG color metadata"
//!    section — only metadata-immune comparison).
//! 5. **Identical metric implementations.** Rust butteraugli +
//!    fast-ssim2 + zensim — no `butteraugli_main` CLI (see CLAUDE.md
//!    PNG metadata section for why).
//!
//! ## Documented apples-to-apples caveats (see TSV footer)
//!
//! - **Gaborish gate**: CPU `api.rs:3842` skips encoder gaborish when
//!   `distance <= 0.5`. GPU mirrors this gate as of commit `4f3203d8`
//!   (`prepare_strategy_search_plan_inner`,
//!   `run_pipeline_with_qac`, `encode_lossy_to_bitstream_via_precomputed*`).
//!   Both paths are equivalent at d <= 0.5 (gaborish off) and at d > 0.5
//!   (gaborish on).
//! - **Patches**: GPU pre-quantized fast path for all-DCT8 images skips
//!   patch detection (commit comment in `encoder.rs:2280-2293`). CPU
//!   path runs patch detection unconditionally at e7+. This means GPU
//!   and CPU bytes can differ on patch-heavy inputs (most photos are
//!   patch-empty so this is a no-op for the photo cells in the default
//!   image set).
//! - **DCT FP precision**: GPU uses cubecl-generated DCT vs CPU
//!   `jxl-encoder-simd` AVX-512. Quoted at "~2% chroma AC coefs flip
//!   by 1 near rounding ties" in `encoder.rs:2290`. Bytes are NOT
//!   byte-identical even when all other state matches. Quality
//!   tolerance is captured by the zensim regress gate
//!   (`tests/cpu_vs_gpu_zensim_regress.rs`).
//!
//! ## Output TSV columns
//!
//! `image, megapixels, effort, distance, encoder, wallclock_ms, bytes,
//!  butteraugli, ssim2, zensim`
//!
//! Rows for `encoder = cpu` and `encoder = gpu` are paired by
//! (image, effort, distance) — easy to pivot.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release -p jxl-encoder-gpu \
//!   --features 'cuda encoder butteraugli-loop' \
//!   --example cpu_vs_gpu_e7_e8_e9_bench -- \
//!     --out PATH.tsv \
//!     [--image PATH]... [--corpus DIR] \
//!     [--effort 7 --effort 8 --effort 9] \
//!     [--distance 0.5 --distance 1.0 ...] \
//!     [--runs 3] [--cell-budget 30]
//! ```
//!
//! `--runs N` does N timed runs per cell (min wall-clock reported).
//! `--cell-budget N` caps total wall-clock per cell at N seconds (skip
//! to next image if exceeded). Default: no cap.

#[cfg(not(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop")))]
fn main() {
    eprintln!("cpu_vs_gpu_e7_e8_e9_bench requires --features 'cuda encoder butteraugli-loop'");
    std::process::exit(2);
}

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn main() {
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    use butteraugli::{ButteraugliParams, srgb_to_linear};
    use imgref::Img;
    use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::butteraugli_loop::ButteraugliLoopGpu;
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;
    use rgb::RGB;
    use zensim::{RgbSlice, Zensim, ZensimProfile};

    type Backend = cubecl::cuda::CudaRuntime;

    // ── parse args ────────────────────────────────────────────────────
    let raw: Vec<String> = std::env::args().collect();
    let mut images: Vec<PathBuf> = Vec::new();
    let mut distances: Vec<f32> = Vec::new();
    let mut efforts: Vec<u8> = Vec::new();
    let mut out_path: Option<PathBuf> = None;
    let mut runs: usize = 3;
    let mut cell_budget_secs: Option<f64> = None;
    let mut help = false;
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--image" => {
                images.push(PathBuf::from(&raw[i + 1]));
                i += 2;
            }
            "--corpus" => {
                let dir = PathBuf::from(&raw[i + 1]);
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
            "--effort" => {
                let e: u8 = raw[i + 1]
                    .parse()
                    .unwrap_or_else(|_| panic!("--effort N (got {})", raw[i + 1]));
                efforts.push(e);
                i += 2;
            }
            "--distance" => {
                let d: f32 = raw[i + 1]
                    .parse()
                    .unwrap_or_else(|_| panic!("--distance D (got {})", raw[i + 1]));
                distances.push(d);
                i += 2;
            }
            "--out" => {
                out_path = Some(PathBuf::from(&raw[i + 1]));
                i += 2;
            }
            "--runs" => {
                runs = raw[i + 1].parse().unwrap_or(3);
                i += 2;
            }
            "--cell-budget" => {
                cell_budget_secs = Some(raw[i + 1].parse().unwrap_or(30.0));
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
            "Usage: cpu_vs_gpu_e7_e8_e9_bench --out PATH.tsv \
                [--image P]... [--corpus DIR] \
                [--effort 7 --effort 8 --effort 9] \
                [--distance 0.5 --distance 1.0 ...] \
                [--runs N] [--cell-budget SECS]"
        );
        if !help {
            std::process::exit(2);
        }
        return;
    }
    if efforts.is_empty() {
        efforts = vec![7, 8, 9];
    }
    if distances.is_empty() {
        distances = vec![0.5_f32, 1.0, 2.0, 4.0];
    }
    let out_path = out_path.unwrap_or_else(|| PathBuf::from("/tmp/cpu_vs_gpu_e7_e8_e9.tsv"));
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // ── header ────────────────────────────────────────────────────────
    // git: when running from a jj workspace (no .git here), fall back
    // to `jj log` for the parent commit's git short id.
    let git_commit = {
        let g = run_cmd_trim("git", &["rev-parse", "--short", "HEAD"]);
        if g != "unknown" {
            g
        } else {
            // jj log -r @- shows the parent change_id; for the git
            // short hash, ask for the commit_id template (first 7 chars).
            let j = run_cmd_trim(
                "jj",
                &["log", "-r", "@-", "--no-graph", "-T", "commit_id.short()"],
            );
            if j.is_empty() {
                "unknown".to_string()
            } else {
                j
            }
        }
    };
    let hostname = run_cmd_trim("hostname", &[]);
    let now_utc = run_cmd_trim("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]);
    let cmdline = std::env::args().collect::<Vec<_>>().join(" ");

    let mut tsv =
        std::fs::File::create(&out_path).unwrap_or_else(|e| panic!("create {out_path:?}: {e}"));
    writeln!(tsv, "# cpu_vs_gpu_e7_e8_e9_bench TSV").unwrap();
    writeln!(tsv, "# generated_utc\t{now_utc}").unwrap();
    writeln!(tsv, "# git_commit\t{git_commit}").unwrap();
    writeln!(tsv, "# hostname\t{hostname}").unwrap();
    writeln!(tsv, "# cmdline\t{cmdline}").unwrap();
    writeln!(tsv, "# n_images\t{}", images.len()).unwrap();
    writeln!(tsv, "# efforts\t{:?}", efforts).unwrap();
    writeln!(tsv, "# distances\t{:?}", distances).unwrap();
    writeln!(tsv, "# runs\t{runs}").unwrap();
    writeln!(tsv, "# cell_budget_secs\t{:?}", cell_budget_secs).unwrap();
    writeln!(tsv, "# CAVEATS — apples-to-apples scope").unwrap();
    writeln!(
        tsv,
        "#   1. CPU sRGB EOTF: jxl-encoder applies IEC 61966-2-1 sRGB EOTF on input. GPU \
         applies the SAME math in `upload_u8_rgb_to_linear_planar_padded`. Linear values \
         match to within f32 precision."
    )
    .unwrap();
    writeln!(
        tsv,
        "#   2. CPU butteraugli_iters per effort: e7=0, e8=2, e9=4 (libjxl \
         enc_adaptive_quantization.cc:1282). GPU mirrors EXACTLY: e7=`_from_u8` no-loop, \
         e8=`_with_butteraugli` 2 iters, e9=`_with_butteraugli` 4 iters."
    )
    .unwrap();
    writeln!(
        tsv,
        "#   3. Gaborish gate at d <= 0.5: CPU api.rs:3842, GPU mirrored in \
         `prepare_strategy_search_plan_inner` + `run_pipeline_with_qac` + \
         `encode_lossy_to_bitstream_via_precomputed*` (commit 4f3203d8)."
    )
    .unwrap();
    writeln!(
        tsv,
        "#   4. NOT apples-to-apples — patches: GPU all-DCT8 fast path skips patch detection \
         (encoder.rs:2280-2293). CPU runs patches at e7+. Photo content is patch-empty so \
         this is a no-op for clic2025-1024 cells; matters for screenshots."
    )
    .unwrap();
    writeln!(
        tsv,
        "#   5. NOT apples-to-apples — DCT FP precision: cubecl DCT vs jxl-encoder-simd \
         AVX-512 differ by ~2% chroma AC coefs flipping by 1 near rounding ties \
         (encoder.rs:2290). Bytes are NOT byte-identical even when all other state matches."
    )
    .unwrap();
    writeln!(
        tsv,
        "#   6. The zensim regress test \
         (`tests/cpu_vs_gpu_zensim_regress.rs`, --features gpu-zensim-regress) \
         asserts gpu_zensim >= cpu_zensim - tolerance per cell."
    )
    .unwrap();
    writeln!(
        tsv,
        "image\tmegapixels\teffort\tdistance\tencoder\twallclock_ms\tbytes\tbutteraugli\tssim2\tzensim"
    )
    .unwrap();
    tsv.flush().ok();

    // ── set up encoders + zensim ──────────────────────────────────────
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    // Cap zensim to something reasonable — refuse 100 MP+ images so a
    // bad input doesn't OOM the bench.
    let zensim = Zensim::new(ZensimProfile::latest()).with_max_pixels(64 * 1024 * 1024);
    // CPU memory budget: 64 GiB. Some images at e9 d=0.5 use ~5 GB of
    // scratch; the default 2 GB cap rejects them.
    let cpu_limits = Limits::default().with_max_memory_bytes(64 * 1024 * 1024 * 1024);

    let total_cells = images.len() * efforts.len() * distances.len();
    let mut cell = 0usize;
    let bench_t0 = Instant::now();

    eprintln!(
        "[setup] {} images × {} efforts × {} distances = {} cells × 2 encoders",
        images.len(),
        efforts.len(),
        distances.len(),
        total_cells,
    );
    eprintln!("[setup] out={}", out_path.display());

    for img_path in &images {
        let img_stem = img_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let short_name = if img_stem.len() > 12 && img_stem.chars().all(|c| c.is_ascii_hexdigit()) {
            img_stem[..8].to_string()
        } else {
            img_stem.clone()
        };

        let src = match image::open(img_path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                eprintln!("[image] ERROR open {img_path:?}: {e} — skipping");
                continue;
            }
        };
        let (w, h) = src.dimensions();
        if w < 16 || h < 16 {
            eprintln!("[image] skip {} (too small {w}×{h})", short_name);
            continue;
        }
        let pixels_u8: Vec<u8> = src.clone().into_raw();
        let mp = (w as f32 * h as f32) / 1e6;

        // Linear f32 planar for the GPU buttloop entry point (which
        // takes linear planar). Matches `to_linear` used in the rest of
        // the codebase (perf_e7_vs_e8.rs:97-104, etc.).
        let n = (w as usize) * (h as usize);
        let mut r_lin = Vec::with_capacity(n);
        let mut g_lin = Vec::with_capacity(n);
        let mut b_lin = Vec::with_capacity(n);
        for chunk in pixels_u8.chunks_exact(3) {
            r_lin.push(srgb_to_linear(chunk[0]));
            g_lin.push(srgb_to_linear(chunk[1]));
            b_lin.push(srgb_to_linear(chunk[2]));
        }

        // Source materialised in the metric formats once per image.
        let orig_lin_pixels: Vec<RGB<f32>> = (0..n)
            .map(|i| RGB::new(r_lin[i], g_lin[i], b_lin[i]))
            .collect();
        let orig_lin_img = Img::new(orig_lin_pixels, w as usize, h as usize);
        let orig_srgb_arr: Vec<[u8; 3]> = pixels_u8
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let orig_srgb_img = Img::new(orig_srgb_arr.clone(), w as usize, h as usize);
        // zensim wants `&[[u8; 3]]` — reuse `orig_srgb_arr`.
        let orig_srgb_for_zensim = RgbSlice::new(&orig_srgb_arr, w as usize, h as usize);

        let bfly_params = ButteraugliParams::default();

        // Per-image GPU resources (per-size).
        let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);
        let mut bg: ButteraugliLoopGpu<Backend> = ButteraugliLoopGpu::new_multires(&enc, w, h);
        bg.set_reference(&pixels_u8).expect("bg.set_reference");

        eprintln!(
            "\n[image {}/{}] {} ({}×{}, {:.2} MP) [src={}]",
            cell + 1,
            total_cells * 2,
            short_name,
            w,
            h,
            mp,
            img_path.display(),
        );

        for &effort in &efforts {
            for &dist in &distances {
                cell += 1;
                let cell_t0 = Instant::now();
                eprintln!("  [{cell}/{total_cells}] e{effort} d={dist}",);

                // ── CPU encode ──────────────────────────────────────
                let cpu_cfg = LossyConfig::new(dist).with_effort(effort);
                // Warm.
                let _warm = cpu_cfg
                    .encode_request(w, h, PixelLayout::Rgb8)
                    .with_limits(&cpu_limits)
                    .encode(&pixels_u8)
                    .map_err(|e| e.decompose().0);
                let mut cpu_min_ms = f64::INFINITY;
                let mut cpu_bytes_opt: Option<Vec<u8>> = None;
                for run in 0..runs {
                    let t = Instant::now();
                    let result = cpu_cfg
                        .encode_request(w, h, PixelLayout::Rgb8)
                        .with_limits(&cpu_limits)
                        .encode(&pixels_u8)
                        .map_err(|e| e.decompose().0);
                    let dt = t.elapsed().as_secs_f64() * 1000.0;
                    match result {
                        Ok(b) => {
                            cpu_min_ms = cpu_min_ms.min(dt);
                            if run + 1 == runs {
                                cpu_bytes_opt = Some(b);
                            }
                        }
                        Err(e) => {
                            eprintln!("    cpu encode failed: {e:?}");
                            break;
                        }
                    }
                    if let Some(budget) = cell_budget_secs
                        && cell_t0.elapsed().as_secs_f64() > budget
                    {
                        eprintln!("    cell-budget {budget}s hit during cpu run; aborting");
                        break;
                    }
                }

                // ── GPU encode ──────────────────────────────────────
                // Warm.
                let _ = match effort {
                    7 => enc
                        .encode_lossy_to_bitstream_via_precomputed_from_u8(&lossy, &pixels_u8, dist)
                        .ok(),
                    8 => enc
                        .encode_lossy_to_bitstream_via_precomputed_with_butteraugli(
                            &lossy, &mut bg, &r_lin, &g_lin, &b_lin, &pixels_u8, dist, 2,
                        )
                        .ok(),
                    _ => enc
                        .encode_lossy_to_bitstream_via_precomputed_with_butteraugli(
                            &lossy, &mut bg, &r_lin, &g_lin, &b_lin, &pixels_u8, dist, 4,
                        )
                        .ok(),
                };
                let mut gpu_min_ms = f64::INFINITY;
                let mut gpu_bytes_opt: Option<Vec<u8>> = None;
                for run in 0..runs {
                    let t = Instant::now();
                    let result: Result<Vec<u8>, String> = match effort {
                        7 => enc
                            .encode_lossy_to_bitstream_via_precomputed_from_u8(
                                &lossy, &pixels_u8, dist,
                            )
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
                    let dt = t.elapsed().as_secs_f64() * 1000.0;
                    match result {
                        Ok(b) => {
                            gpu_min_ms = gpu_min_ms.min(dt);
                            if run + 1 == runs {
                                gpu_bytes_opt = Some(b);
                            }
                        }
                        Err(e) => {
                            eprintln!("    gpu encode failed: {e}");
                            break;
                        }
                    }
                    if let Some(budget) = cell_budget_secs
                        && cell_t0.elapsed().as_secs_f64() > budget
                    {
                        eprintln!("    cell-budget {budget}s hit during gpu run; aborting");
                        break;
                    }
                }

                // ── Decode + score ─────────────────────────────────
                let cpu_metrics = match &cpu_bytes_opt {
                    Some(b) => measure(
                        b,
                        w,
                        h,
                        &orig_lin_img,
                        &orig_srgb_img,
                        &orig_srgb_for_zensim,
                        &bfly_params,
                        &zensim,
                    ),
                    None => Metrics::nan(),
                };
                let gpu_metrics = match &gpu_bytes_opt {
                    Some(b) => measure(
                        b,
                        w,
                        h,
                        &orig_lin_img,
                        &orig_srgb_img,
                        &orig_srgb_for_zensim,
                        &bfly_params,
                        &zensim,
                    ),
                    None => Metrics::nan(),
                };

                let cpu_bytes = cpu_bytes_opt.as_ref().map(|b| b.len()).unwrap_or(0);
                let gpu_bytes = gpu_bytes_opt.as_ref().map(|b| b.len()).unwrap_or(0);

                writeln!(
                    tsv,
                    "{short_name}\t{mp:.4}\t{effort}\t{dist}\tcpu\t{cpu_min_ms:.2}\t{cpu_bytes}\t{:.6}\t{:.6}\t{:.6}",
                    cpu_metrics.bfly, cpu_metrics.ssim2, cpu_metrics.zensim,
                )
                .unwrap();
                writeln!(
                    tsv,
                    "{short_name}\t{mp:.4}\t{effort}\t{dist}\tgpu\t{gpu_min_ms:.2}\t{gpu_bytes}\t{:.6}\t{:.6}\t{:.6}",
                    gpu_metrics.bfly, gpu_metrics.ssim2, gpu_metrics.zensim,
                )
                .unwrap();
                tsv.flush().ok();

                eprintln!(
                    "    cpu: {cpu_min_ms:7.1} ms / {cpu_bytes:8} bytes / bfly {:.3} ssim2 {:5.2} zensim {:5.2}",
                    cpu_metrics.bfly, cpu_metrics.ssim2, cpu_metrics.zensim,
                );
                eprintln!(
                    "    gpu: {gpu_min_ms:7.1} ms / {gpu_bytes:8} bytes / bfly {:.3} ssim2 {:5.2} zensim {:5.2}",
                    gpu_metrics.bfly, gpu_metrics.ssim2, gpu_metrics.zensim,
                );
            }
        }
    }

    let total_secs = bench_t0.elapsed().as_secs_f64();
    eprintln!("\n[done] total wall-clock: {total_secs:.1}s");
    eprintln!("[done] wrote {}", out_path.display());
}

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
struct Metrics {
    bfly: f64,
    ssim2: f64,
    zensim: f64,
}

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
impl Metrics {
    fn nan() -> Self {
        Self {
            bfly: f64::NAN,
            ssim2: f64::NAN,
            zensim: f64::NAN,
        }
    }
}

/// Decode a JXL bitstream and compute butteraugli + ssim2 + zensim
/// against the source. Returns NaN-filled `Metrics` on decode failure.
#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
#[allow(clippy::too_many_arguments)]
fn measure(
    bytes: &[u8],
    src_w: u32,
    src_h: u32,
    orig_lin_img: &imgref::Img<Vec<rgb::RGB<f32>>>,
    orig_srgb_img: &imgref::Img<Vec<[u8; 3]>>,
    orig_srgb_for_zensim: &zensim::RgbSlice,
    bfly_params: &butteraugli::ButteraugliParams,
    zensim_engine: &zensim::Zensim,
) -> Metrics {
    use imgref::Img;
    use rgb::RGB;

    let Some((dw, dh, decoded_linear)) = decode_jxl_linear(bytes) else {
        return Metrics::nan();
    };
    if dw != src_w as usize || dh != src_h as usize {
        return Metrics::nan();
    }

    // Butteraugli on linear RGB.
    let dec_lin: Vec<RGB<f32>> = decoded_linear
        .chunks_exact(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img = Img::new(dec_lin, dw, dh);
    let bfly =
        butteraugli::butteraugli_linear(orig_lin_img.as_ref(), dec_lin_img.as_ref(), bfly_params)
            .map(|s| s.score)
            .unwrap_or(f64::NAN);

    // Convert decoded linear → sRGB u8 for ssim2 + zensim (sRGB-domain
    // metrics).
    let dec_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks_exact(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(dec_srgb.clone(), dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);

    // Zensim takes `&[[u8; 3]]` per-pixel. `dec_srgb` is already in
    // that layout; reuse it directly.
    let dec_zensim = zensim::RgbSlice::new(&dec_srgb, dw, dh);
    let zensim_score = zensim_engine
        .compute(orig_srgb_for_zensim, &dec_zensim)
        .map(|r| r.score())
        .unwrap_or(f64::NAN);

    Metrics {
        bfly,
        ssim2,
        zensim: zensim_score,
    }
}

/// Linear-light sRGB f32 → sRGB u8, IEC 61966-2-1.
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

/// Decode a JXL bitstream to interleaved linear RGB f32.
///
/// Tries jxl-oxide first (`request_color_encoding(srgb_linear(Relative))`)
/// for the metadata-immune path documented in project CLAUDE.md.
///
/// Falls back to jxl-rs (the PRIMARY decoder per project CLAUDE.md) if
/// jxl-oxide rejects the bitstream — this catches the known jxl-oxide
/// 0.12.5 limitation with ANS in multi-group modular frames (unexpected
/// EOF). jxl-rs returns linear f32 by default for `JxlDataFormat::f32()`
/// but its color-encoding handling is less explicit than jxl-oxide's
/// `request_color_encoding`, so jxl-oxide is the canonical path.
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

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn run_cmd_trim(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd)
        .args(args)
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
        .unwrap_or_else(|| "unknown".into())
}
