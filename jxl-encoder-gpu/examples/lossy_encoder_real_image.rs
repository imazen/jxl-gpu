//! LossyEncoder on a real CLIC2025-1024 photo.
//!
//! Loads an actual photo via the `image` crate (sRGB U8 PNG),
//! crops to a non-multiple-of-8 size to exercise the arbitrary-size
//! padding path, runs the LossyEncoder pipeline using the
//! `encode_one_srgb_u8` U8 → U8 convenience helper, optionally
//! writes the reconstructed PNG to disk.
//!
//! Usage:
//!   cargo run --release --features cuda --example lossy_encoder_real_image
//!
//! Optional env vars:
//!   IMAGE_PATH  Override the input PNG (default: a CLIC2025-1024 sample)
//!   QAC         Override the per-block quant scale (default: 4.0)
//!   CROP_W      Crop width (default: src.width - 7)
//!   CROP_H     Crop height (default: src.height - 11)
//!   OUT_PATH    Write reconstructed PNG to this path (default: skip)

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
    let out_path = std::env::var("OUT_PATH").ok();

    let img_raw = image::open(&image_path)
        .unwrap_or_else(|e| panic!("failed to open {image_path}: {e}"))
        .to_rgb8();
    let crop_w: u32 = std::env::var("CROP_W")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or((img_raw.width() - 7).min(1019));
    let crop_h: u32 = std::env::var("CROP_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or((img_raw.height() - 11).min(1013));
    let img = image::imageops::crop_imm(&img_raw, 0, 0, crop_w, crop_h).to_image();
    let (w, h) = img.dimensions();
    let rgb_in: Vec<u8> = img.into_raw();
    let n = (w * h) as usize;

    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);

    println!("=== LossyEncoder::encode_one_srgb_u8 on real image ===");
    println!("Image: {image_path}");
    println!(
        "Size:  {w}×{h} ({:.2} MP), padded internally to {:?}",
        n as f64 / 1e6,
        lossy.padded_dimensions()
    );
    println!("qac:   {qac}\n");

    // Cold + warm encode (sRGB U8 in/out — no host-side conversion in caller).
    let t0 = std::time::Instant::now();
    let _rgb_out_cold = lossy.encode_one_srgb_u8(&enc, &rgb_in, qac);
    let dt0 = t0.elapsed();
    println!("encode_one_srgb_u8 (cold): {:.2} ms", dt0.as_secs_f64() * 1000.0);

    let t1 = std::time::Instant::now();
    let rgb_out = lossy.encode_one_srgb_u8(&enc, &rgb_in, qac);
    let dt1 = t1.elapsed();
    println!(
        "encode_one_srgb_u8 (warm): {:.2} ms ({:.0} MP/s)",
        dt1.as_secs_f64() * 1000.0,
        (n as f64 / 1e6) / dt1.as_secs_f64()
    );

    assert_eq!(rgb_out.len(), n * 3);

    // Per-channel U8 reconstruction quality.
    let mut sum: [u64; 3] = [0; 3];
    let mut max: [u8; 3] = [0; 3];
    for (i, (&src, &dst)) in rgb_in.iter().zip(&rgb_out).enumerate() {
        let c = i % 3;
        let d = src.abs_diff(dst);
        sum[c] += d as u64;
        if d > max[c] {
            max[c] = d;
        }
    }
    println!("\nReconstruction quality (sRGB U8):");
    println!(
        "  R: MAE={:.2} bytes, max={}",
        sum[0] as f64 / n as f64,
        max[0]
    );
    println!(
        "  G: MAE={:.2} bytes, max={}",
        sum[1] as f64 / n as f64,
        max[1]
    );
    println!(
        "  B: MAE={:.2} bytes, max={}",
        sum[2] as f64 / n as f64,
        max[2]
    );

    if let Some(path) = out_path {
        let img_buf =
            image::RgbImage::from_raw(w, h, rgb_out).expect("buf size matches");
        img_buf.save(&path).expect("save output PNG");
        println!("\n✓ Wrote reconstructed PNG to {path}");
    } else {
        println!(
            "\n✓ LossyEncoder::encode_one_srgb_u8 ran end-to-end on real {w}×{h} input.\n  Set OUT_PATH=/tmp/recon.png to write the reconstructed image."
        );
    }
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
