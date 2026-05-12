// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Paired A/B for u8-input fused-prep vs f32-input three-plane upload.
//!
//! Both paths produce 3 padded GpuPlanes of linear f32 RGB. They
//! differ only in upload cost + host-side preprocessing:
//!
//! - Path A (existing): host srgb→linear (per-pixel powf), host
//!   pad_to_alignment, upload 3 × padded f32 planes via
//!   upload_planes_3ch.
//! - Path B (fused): upload raw interleaved u8 RGB, run
//!   upload_u8_rgb_to_linear_planar_padded (one fused GPU launch
//!   doing srgb→linear + pad).
//!
//! Run:
//!   cargo run --release -p jxl-encoder-gpu --features 'cuda encoder' \
//!     --example perf_u8_upload_vs_f32 -- --image PATH \
//!     [--target-mp M] [--runs N]

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
}

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use std::time::Instant;

    use jxl_encoder_gpu::encoder::GpuEncoder;

    type B = cubecl::cuda::CudaRuntime;

    // Args.
    let raw: Vec<String> = std::env::args().collect();
    let mut image_path: Option<String> = None;
    let mut target_mp: Option<f32> = None;
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
            "--runs" => {
                runs = raw[i + 1].parse().expect("--runs N");
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

    // Padded dims = align_up to 16.
    let pad_to_16 = |v: u32| -> u32 { (v + 15) / 16 * 16 };
    let pw = pad_to_16(w);
    let ph = pad_to_16(h);
    println!(
        "perf_u8_upload_vs_f32: src {}×{} ({:.2} MP), padded {}×{}, runs={}",
        w,
        h,
        n as f32 / 1e6,
        pw,
        ph,
        runs
    );

    let enc: GpuEncoder<B> = GpuEncoder::new();

    // Pre-convert u8 → host f32 linear + pad-to-alignment for path A.
    // (This work is OUTSIDE the timed region — represents the
    // status-quo work the host has done before calling LossyEncoder.)
    let to_linear = |c: u8| -> f32 {
        let f = c as f32 / 255.0;
        if f <= 0.04045 {
            f / 12.92
        } else {
            ((f + 0.055) / 1.055).powf(2.4)
        }
    };
    let mut r_unpadded = Vec::with_capacity(n);
    let mut g_unpadded = Vec::with_capacity(n);
    let mut b_unpadded = Vec::with_capacity(n);
    for chunk in pixels_u8.chunks_exact(3) {
        r_unpadded.push(to_linear(chunk[0]));
        g_unpadded.push(to_linear(chunk[1]));
        b_unpadded.push(to_linear(chunk[2]));
    }
    let pad_plane = |src: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0_f32; (pw as usize) * (ph as usize)];
        for y in 0..h as usize {
            let so = y * w as usize;
            let dout = y * pw as usize;
            out[dout..dout + w as usize].copy_from_slice(&src[so..so + w as usize]);
            let last = src[so + w as usize - 1];
            for x in w as usize..pw as usize {
                out[dout + x] = last;
            }
        }
        if (ph as usize) > h as usize {
            let last_row_start = (h as usize - 1) * pw as usize;
            for y in h as usize..ph as usize {
                let dout = y * pw as usize;
                out.copy_within(last_row_start..last_row_start + pw as usize, dout);
            }
        }
        out
    };
    let r_padded = pad_plane(&r_unpadded);
    let g_padded = pad_plane(&g_unpadded);
    let b_padded = pad_plane(&b_unpadded);

    // Warm both paths.
    let _ = enc.upload_planes_3ch(&r_padded, &g_padded, &b_padded, pw, ph);
    let _ = enc.upload_u8_rgb_to_linear_planar_padded(&pixels_u8, w, h, pw, ph);

    // ── Path A: f32 upload (3× padded planes) ───────────────────
    let mut a_times: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = enc.upload_planes_3ch(&r_padded, &g_padded, &b_padded, pw, ph);
        // Force sync via small read on each handle would be needed
        // for accurate timing, but upload_planes_3ch's create_tensors
        // is already synchronous wrt the upload completion (per the
        // perf_upload_plane microbench). So end-of-call wall-clock
        // captures the upload time.
        a_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // ── Path B: u8 upload + GPU srgb→linear + pad ────────────────
    let mut b_times: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = enc.upload_u8_rgb_to_linear_planar_padded(&pixels_u8, w, h, pw, ph);
        b_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let summary = |name: &str, times: &[f64]| {
        let mut s: Vec<f64> = times.iter().copied().collect();
        s.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let min = s[0];
        let med = s[runs / 2];
        let mean: f64 = s.iter().sum::<f64>() / runs as f64;
        println!(
            "  {:30}  min={:6.2} ms  median={:6.2} ms  mean={:6.2} ms",
            name, min, med, mean
        );
    };
    println!();
    summary("Path A (f32 upload, 3 planes)", &a_times);
    summary("Path B (u8 upload + fused)   ", &b_times);
    let a_min = *a_times
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let b_min = *b_times
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    println!(
        "\n  speedup (min):  {:.2}×  (Δ {:.2} ms)",
        a_min / b_min,
        a_min - b_min,
    );
    println!(
        "  upload bytes:  Path A = {:.1} MB (3× padded f32),  Path B = {:.1} MB (raw u8)",
        (3.0 * pw as f64 * ph as f64 * 4.0) / 1e6,
        (3.0 * w as f64 * h as f64) / 1e6,
    );
}
