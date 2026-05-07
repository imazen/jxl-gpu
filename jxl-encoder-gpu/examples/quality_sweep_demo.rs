//! Quality sweep on a real image — demonstrates the recommended
//! API pattern: `quality_to_qac` + `encode_many_srgb_u8`.
//!
//! Encodes a single image at JPEG-quality settings 95, 80, 60, 40,
//! 20 (mapped to `qac_qm` via [`quality_to_qac`]) and reports
//! per-quality reconstruction MAE. Optionally writes one PNG per
//! quality so you can visually compare degradation.
//!
//! This is the "production" usage pattern. Construct one
//! [`LossyEncoder`] per image size, then call [`encode_many_srgb_u8`]
//! to run the sweep with a single host→GPU upload of the input.
//!
//! Usage:
//!   cargo run --release --features cuda --example quality_sweep_demo
//!
//! Optional env vars:
//!   IMAGE_PATH  Override the input PNG (default: a CLIC2025-1024 sample)
//!   OUT_DIR     Write per-quality PNGs to this dir (default: skip)

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::{LossyEncoder, quality_to_qac};

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let image_path = std::env::var("IMAGE_PATH").unwrap_or_else(|_| {
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
            .to_string()
    });
    let out_dir = std::env::var("OUT_DIR").ok();

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("failed to open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let rgb_in: Vec<u8> = img.into_raw();
    let n = (w * h) as usize;

    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);

    // The five JPEG-quality settings to sweep.
    let qualities = [95.0_f32, 80.0, 60.0, 40.0, 20.0];
    let qacs: Vec<f32> = qualities.iter().copied().map(quality_to_qac).collect();

    println!("=== quality_sweep_demo ===");
    println!(
        "Image: {image_path}\nSize:  {w}×{h} ({:.2} MP), padded to {:?}\n",
        n as f64 / 1e6,
        lossy.padded_dimensions()
    );

    println!("Quality → qac mapping:");
    for (q, qac) in qualities.iter().zip(&qacs) {
        println!("  q={q:>4.0}  →  qac={qac:.3}");
    }

    let t0 = std::time::Instant::now();
    let outputs = lossy.encode_many_srgb_u8(&enc, &rgb_in, &qacs);
    let dt = t0.elapsed();
    println!(
        "\nencode_many_srgb_u8: {} settings in {:.2} ms ({:.2} ms/encode)",
        qualities.len(),
        dt.as_secs_f64() * 1000.0,
        dt.as_secs_f64() * 1000.0 / qualities.len() as f64,
    );

    println!("\nReconstruction quality (sRGB U8 byte MAE):");
    println!("  {:>5}  {:>8}  {:>6}  {:>6}  {:>6}", "qual", "qac", "R MAE", "G MAE", "B MAE");
    for ((q, qac), rgb_out) in qualities.iter().zip(&qacs).zip(&outputs) {
        let mut sum = [0_u64; 3];
        for (i, (&src, &dst)) in rgb_in.iter().zip(rgb_out).enumerate() {
            sum[i % 3] += src.abs_diff(dst) as u64;
        }
        let nf = n as f64;
        println!(
            "  {:>5.0}  {:>8.3}  {:>6.2}  {:>6.2}  {:>6.2}",
            q,
            qac,
            sum[0] as f64 / nf,
            sum[1] as f64 / nf,
            sum[2] as f64 / nf,
        );
    }

    if let Some(dir) = out_dir {
        std::fs::create_dir_all(&dir).expect("create output dir");
        for (q, rgb_out) in qualities.iter().zip(&outputs) {
            let path = format!("{dir}/q{:03}.png", *q as u32);
            let img_buf =
                image::RgbImage::from_raw(w, h, rgb_out.clone()).expect("buf size matches");
            img_buf.save(&path).expect("save PNG");
        }
        println!("\n✓ Wrote {} PNGs to {dir}/", qualities.len());
    } else {
        println!("\n✓ Sweep complete. Set OUT_DIR=/tmp/sweep to write PNGs per quality.");
    }
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
