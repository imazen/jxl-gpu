//! Real-image encode integration test using a CLIC2025-1024 photo.
//!
//! Loads a 1024×1024 PNG via the `image` crate, encodes it through
//! `GpuEncoder::encode_lossy_via_cpu` (delegates to jxl-encoder for
//! the full CPU lossy path — proves the dependency works on real
//! images), and writes the result to /tmp for manual verification
//! with djxl or jxl-rs CLI.
//!
//! Usage:
//!   cargo run --release --features cuda --example real_image_encode
//!
//! Optional env vars:
//!   IMAGE_PATH        Override the input PNG (default: a CLIC sample)
//!   OUT_PATH          Override output JXL path (default: /tmp/jxl-gpu-real.jxl)
//!   DISTANCE          Override distance (default: 1.0 ≈ q92)
//!   EFFORT            Override effort 1-9 (default: 5)
//!
//! Verify the output:
//!   djxl /tmp/jxl-gpu-real.jxl /tmp/jxl-gpu-real.png
//!   # or: cargo run -p jxl_cli --release -- /tmp/jxl-gpu-real.jxl /tmp/out.png

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder::api::{LossyConfig, PixelLayout};
    use jxl_encoder_gpu::encoder::GpuEncoder;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let image_path = std::env::var("IMAGE_PATH").unwrap_or_else(|_| {
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
            .to_string()
    });
    let out_path =
        std::env::var("OUT_PATH").unwrap_or_else(|_| "/tmp/jxl-gpu-real.jxl".to_string());
    let distance: f32 = std::env::var("DISTANCE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let effort: u8 = std::env::var("EFFORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    println!("=== real-image encode test ===");
    println!("Image:    {image_path}");
    println!("Output:   {out_path}");
    println!("Distance: {distance}");
    println!("Effort:   {effort}\n");

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("failed to open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let pixels: Vec<u8> = img.into_raw();
    println!("Loaded {w}×{h} RGB8 ({} bytes)", pixels.len());

    let config = LossyConfig::new(distance).with_effort(effort);

    let t0 = std::time::Instant::now();
    let bytes = enc
        .encode_lossy_via_cpu(&config, &pixels, w, h, PixelLayout::Rgb8)
        .expect("encode failed");
    let dt = t0.elapsed();

    let bpp = (bytes.len() * 8) as f64 / (w as f64 * h as f64);
    println!(
        "Encoded:  {} bytes ({:.3} bpp) in {:.2}s",
        bytes.len(),
        bpp,
        dt.as_secs_f64()
    );

    // Sanity: starts with JXL signature (0xFF 0x0A) or container box.
    let starts_ok = bytes.starts_with(&[0xFF, 0x0A])
        || bytes.starts_with(&[0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ']);
    assert!(starts_ok, "output doesn't start with JXL signature");

    std::fs::write(&out_path, &bytes).unwrap_or_else(|e| panic!("write failed: {e}"));
    println!("\n✓ Wrote {} bytes to {out_path}", bytes.len());
    println!("  Verify: djxl {out_path} /tmp/decoded.png");

    // Standalone GPU XYB on the same image to prove the GPU side handles
    // 1024×1024 input. This is a sanity check that the GPU path scales
    // beyond the 64×64 demos.
    let n = (w * h) as usize;
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    // Convert sRGB 8-bit → linear f32. Use the same simple gamma curve
    // jxl-encoder applies internally (good enough for a sanity check;
    // real encode handles the proper sRGB transfer function).
    for chunk in pixels.chunks_exact(3) {
        let to_linear = |c: u8| (c as f32 / 255.0).powf(2.4);
        r.push(to_linear(chunk[0]));
        g.push(to_linear(chunk[1]));
        b.push(to_linear(chunk[2]));
    }
    let t_gpu = std::time::Instant::now();
    let (xx, xy, xb) = enc.xyb_from_linear_rgb(&r, &g, &b);
    let dt_gpu = t_gpu.elapsed();
    assert_eq!(xx.len(), n);
    assert_eq!(xy.len(), n);
    assert_eq!(xb.len(), n);
    assert!(xx.iter().all(|v| v.is_finite()));
    assert!(xy.iter().all(|v| v.is_finite()));
    assert!(xb.iter().all(|v| v.is_finite()));
    println!(
        "\n✓ GPU XYB on {w}×{h} ({}MP) in {:.3}ms — all finite",
        (w as f64 * h as f64 / 1e6),
        dt_gpu.as_secs_f64() * 1000.0
    );
    println!(
        "  Sample: X[0]={:.4}, Y[0]={:.4}, B[0]={:.4}",
        xx[0], xy[0], xb[0]
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
