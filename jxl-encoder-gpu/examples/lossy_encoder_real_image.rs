//! LossyEncoder on a real CLIC2025-1024 photo.
//!
//! Loads an actual photo via the `image` crate (sRGB U8 PNG),
//! converts to linear f32 RGB, runs the LossyEncoder pipeline,
//! and reports per-channel reconstruction quality (MAE, max).
//!
//! Validates that LossyEncoder works on real-world image data, not
//! just synthetic input. Output is the reconstructed RGB; this isn't
//! producing JXL bitstream bytes (that path is
//! `GpuEncoder::encode_lossy_via_cpu`).
//!
//! Usage:
//!   cargo run --release --features cuda --example lossy_encoder_real_image
//!
//! Optional env vars:
//!   IMAGE_PATH  Override the input PNG (default: a CLIC2025-1024 sample)
//!   QAC         Override the per-block quant scale (default: 4.0)

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let image_path = std::env::var("IMAGE_PATH").unwrap_or_else(|_| {
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
            .to_string()
    });
    let qac: f32 = std::env::var("QAC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4.0);

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("failed to open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    assert!(w.is_multiple_of(8) && h.is_multiple_of(8), "image dims must be 8-aligned");
    let pixels: Vec<u8> = img.into_raw();
    let n = (w * h) as usize;

    // sRGB U8 → linear f32 (per channel).
    let to_linear = |c: u8| (c as f32 / 255.0).powf(2.4);
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for chunk in pixels.chunks_exact(3) {
        r.push(to_linear(chunk[0]));
        g.push(to_linear(chunk[1]));
        b.push(to_linear(chunk[2]));
    }

    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);

    println!("=== LossyEncoder on real image ===");
    println!("Image: {image_path}");
    println!("Size:  {w}×{h} ({:.2} MP)", n as f64 / 1e6);
    println!("qac:   {qac}\n");

    // Cold encode (includes warm-up).
    let t0 = std::time::Instant::now();
    let (rec_r, rec_g, rec_b) = lossy.encode_one(&enc, &r, &g, &b, qac);
    let dt0 = t0.elapsed();
    println!("encode_one (cold): {:.2} ms", dt0.as_secs_f64() * 1000.0);

    // Warm encode (re-runs on same input — measures steady state).
    let t1 = std::time::Instant::now();
    let _ = lossy.encode_one(&enc, &r, &g, &b, qac);
    let dt1 = t1.elapsed();
    println!(
        "encode_one (warm): {:.2} ms ({:.0} MP/s)",
        dt1.as_secs_f64() * 1000.0,
        (n as f64 / 1e6) / dt1.as_secs_f64()
    );

    // Per-channel reconstruction quality.
    let mut sum_r = 0.0_f64;
    let mut sum_g = 0.0_f64;
    let mut sum_b = 0.0_f64;
    let mut max_r = 0.0_f32;
    let mut max_g = 0.0_f32;
    let mut max_b = 0.0_f32;
    for i in 0..n {
        let dr = (r[i] - rec_r[i]).abs();
        let dg = (g[i] - rec_g[i]).abs();
        let db = (b[i] - rec_b[i]).abs();
        sum_r += dr as f64;
        sum_g += dg as f64;
        sum_b += db as f64;
        max_r = max_r.max(dr);
        max_g = max_g.max(dg);
        max_b = max_b.max(db);
    }
    let nf = n as f64;
    println!("\nReconstruction quality (linear f32):");
    println!("  R: MAE={:.4e}, max={:.4e}", sum_r / nf, max_r);
    println!("  G: MAE={:.4e}, max={:.4e}", sum_g / nf, max_g);
    println!("  B: MAE={:.4e}, max={:.4e}", sum_b / nf, max_b);

    for v in rec_r.iter().chain(&rec_g).chain(&rec_b) {
        assert!(v.is_finite(), "non-finite pixel in reconstruction");
    }

    println!("\n✓ LossyEncoder ran end-to-end on real {w}×{h} input.");
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
