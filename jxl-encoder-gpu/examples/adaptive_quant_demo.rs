//! Demonstrates [`LossyEncoder::encode_one_adaptive`] with a
//! synthetic per-block qac field.
//!
//! Splits the image vertically: the LEFT half gets gentle quant
//! (qac = distance_to_qac(0.5), high quality) and the RIGHT half
//! gets aggressive quant (qac = distance_to_qac(8.0), heavily
//! compressed). The output PNG should show a visible quality split
//! down the middle — the right half blockier/blurrier.
//!
//! Usage:
//!   cargo run --release --features cuda --example adaptive_quant_demo
//!
//! Optional env vars:
//!   IMAGE_PATH  Override input PNG (default: a CLIC2025 sample)
//!   OUT_PATH    Write reconstructed PNG (default: skip)

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::{LossyEncoder, distance_to_qac};

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let image_path = std::env::var("IMAGE_PATH").unwrap_or_else(|_| {
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
            .to_string()
    });
    let out_path = std::env::var("OUT_PATH").ok();

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("failed to open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let pixels: Vec<u8> = img.into_raw();
    let n = (w * h) as usize;

    // sRGB U8 → linear f32 planar.
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
    let (pw, ph) = lossy.padded_dimensions();
    let nb = ((pw / 8) * (ph / 8)) as usize;

    // Build a per-block aq field: left half gets light quant (high-q),
    // right half gets heavy quant (low-q).
    let qac_high = distance_to_qac(0.5); // light quant (high quality)
    let qac_low = distance_to_qac(8.0); // heavy quant (low quality)
    let blocks_per_row = (pw / 8) as usize;
    let mut aq_field = vec![0.0_f32; nb];
    for i in 0..nb {
        let bx = i % blocks_per_row;
        aq_field[i] = if bx < blocks_per_row / 2 {
            qac_high
        } else {
            qac_low
        };
    }

    println!("=== adaptive_quant_demo ===");
    println!("Image: {image_path}");
    println!(
        "Size: {w}x{h} ({:.2} MP), padded to {pw}x{ph}",
        n as f64 / 1e6
    );
    println!(
        "Left half:  qac={qac_high:.3} (distance=0.5, high quality)\nRight half: qac={qac_low:.3} (distance=8.0, low quality)\n"
    );

    let t0 = std::time::Instant::now();
    let (rec_r, rec_g, rec_b) = lossy.encode_one_adaptive(&enc, &r, &g, &b, &aq_field);
    let dt = t0.elapsed();
    println!(
        "encode_one_adaptive: {:.2} ms ({} blocks, half at each setting)",
        dt.as_secs_f64() * 1000.0,
        nb
    );

    // Per-half MAE.
    let half_w = (w / 2) as usize;
    let mut sum_left = [0.0_f64; 3];
    let mut sum_right = [0.0_f64; 3];
    for y in 0..(h as usize) {
        for x in 0..(w as usize) {
            let i = y * (w as usize) + x;
            let dr = (r[i] - rec_r[i]).abs() as f64;
            let dg = (g[i] - rec_g[i]).abs() as f64;
            let db = (b[i] - rec_b[i]).abs() as f64;
            if x < half_w {
                sum_left[0] += dr;
                sum_left[1] += dg;
                sum_left[2] += db;
            } else {
                sum_right[0] += dr;
                sum_right[1] += dg;
                sum_right[2] += db;
            }
        }
    }
    let half_n = (half_w * h as usize) as f64;
    println!("\nPer-half reconstruction MAE (linear f32):");
    println!(
        "  Left  (high-q):  R={:.4e} G={:.4e} B={:.4e}",
        sum_left[0] / half_n,
        sum_left[1] / half_n,
        sum_left[2] / half_n
    );
    println!(
        "  Right (low-q):   R={:.4e} G={:.4e} B={:.4e}",
        sum_right[0] / half_n,
        sum_right[1] / half_n,
        sum_right[2] / half_n
    );
    println!("\nRight should have ~3-5× higher MAE (heavier quant).");

    if let Some(path) = out_path {
        // sRGB encode.
        let to_srgb_u8 = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.4) * 255.0).round() as u8;
        let mut rgb_out = Vec::with_capacity(n * 3);
        for i in 0..n {
            rgb_out.push(to_srgb_u8(rec_r[i]));
            rgb_out.push(to_srgb_u8(rec_g[i]));
            rgb_out.push(to_srgb_u8(rec_b[i]));
        }
        let img_buf = image::RgbImage::from_raw(w, h, rgb_out).expect("buf size matches");
        img_buf.save(&path).expect("save output PNG");
        println!("\n✓ Wrote split-quality reconstruction to {path}");
    } else {
        println!("\n✓ Adaptive quant ran end-to-end. Set OUT_PATH=/tmp/aq.png to write the PNG.");
    }
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
