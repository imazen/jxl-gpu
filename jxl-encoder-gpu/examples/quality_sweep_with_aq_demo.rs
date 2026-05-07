//! Distance sweep with content-driven AQ vs uniform quant — the
//! batch sRGB-U8 form. Demonstrates [`LossyEncoder::encode_many_with_aq_srgb_u8`]
//! alongside [`LossyEncoder::encode_many_srgb_u8`] for direct
//! comparison.
//!
//! For each libjxl distance in {0.5, 1.0, 2.0, 4.0, 8.0}, encodes
//! the same image two ways:
//!   1. Uniform quant at `qac = distance_to_qac(distance)`.
//!   2. Content-driven AQ centered on `distance` (per-block field
//!      derived from mask1x1, varying in a 4× range).
//!
//! Reports per-distance sRGB-U8 byte MAE for each. Optionally writes
//! `q{distance}_uniform.png` and `q{distance}_aq.png` per setting so
//! you can visually compare smooth-region preservation in dark areas
//! vs detail preservation at edges.
//!
//! This is the "AQ in production" usage pattern: one batch call gets
//! you N PNGs across a distance grid with the mask1x1 prepass paid
//! exactly once.
//!
//! Usage:
//!   cargo run --release --features cuda --example quality_sweep_with_aq_demo
//!
//! Optional env vars:
//!   IMAGE_PATH  Override the input PNG (default: a CLIC2025-1024 sample)
//!   OUT_DIR     Write per-distance PNGs to this dir (default: skip)

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
    let out_dir = std::env::var("OUT_DIR").ok();

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("failed to open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let rgb_in: Vec<u8> = img.into_raw();
    let n = (w * h) as usize;

    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);

    let distances = [0.5_f32, 1.0, 2.0, 4.0, 8.0];
    let qacs_uniform: Vec<f32> = distances.iter().copied().map(distance_to_qac).collect();

    println!("=== quality_sweep_with_aq_demo ===");
    println!(
        "Image: {image_path}\nSize:  {w}×{h} ({:.2} MP), padded to {:?}\n",
        n as f64 / 1e6,
        lossy.padded_dimensions()
    );

    let t_un = std::time::Instant::now();
    let out_uniform = lossy.encode_many_srgb_u8(&enc, &rgb_in, &qacs_uniform);
    let dt_un = t_un.elapsed();

    let t_aq = std::time::Instant::now();
    let out_aq = lossy.encode_many_with_aq_srgb_u8(&enc, &rgb_in, &distances);
    let dt_aq = t_aq.elapsed();

    println!(
        "encode_many_srgb_u8 (uniform):     {} encodes in {:>6.1} ms ({:.2} ms/encode)",
        distances.len(),
        dt_un.as_secs_f64() * 1000.0,
        dt_un.as_secs_f64() * 1000.0 / distances.len() as f64,
    );
    println!(
        "encode_many_with_aq_srgb_u8 (AQ):  {} encodes in {:>6.1} ms ({:.2} ms/encode, incl. mask prepass)",
        distances.len(),
        dt_aq.as_secs_f64() * 1000.0,
        dt_aq.as_secs_f64() * 1000.0 / distances.len() as f64,
    );

    let mae = |out: &[u8]| -> [f64; 3] {
        let mut sum = [0_u64; 3];
        for (i, (&src, &dst)) in rgb_in.iter().zip(out).enumerate() {
            sum[i % 3] += src.abs_diff(dst) as u64;
        }
        let nf = n as f64;
        [sum[0] as f64 / nf, sum[1] as f64 / nf, sum[2] as f64 / nf]
    };

    // SSIMULACRA2 perceptual score (100 = identical, 90+ imperceptible).
    let ssim2 = |out: &[u8]| -> f64 {
        let to_rgb3 = |buf: &[u8]| -> Vec<[u8; 3]> {
            buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
        };
        let src = to_rgb3(&rgb_in);
        let dst = to_rgb3(out);
        let src_img =
            imgref::ImgVec::new(src, w as usize, h as usize);
        let dst_img =
            imgref::ImgVec::new(dst, w as usize, h as usize);
        fast_ssim2::compute_ssimulacra2(src_img.as_ref(), dst_img.as_ref())
            .expect("ssimulacra2") as f64
    };

    println!("\nPer-distance metrics:");
    println!(
        "  {:>5}  {:>8}  | {:>16}  {:>5}  | {:>16}  {:>5}  | {:>5}",
        "dist", "qac_un", "uniform R/G/B MAE", "ssim2", "AQ R/G/B MAE", "ssim2", "Δssim2"
    );
    for (i, d) in distances.iter().enumerate() {
        let m_un = mae(&out_uniform[i]);
        let m_aq = mae(&out_aq[i]);
        let s_un = ssim2(&out_uniform[i]);
        let s_aq = ssim2(&out_aq[i]);
        println!(
            "  {:>5.2}  {:>8.3}  | {:>4.2}/{:>4.2}/{:>4.2}    {:>5.2}  | {:>4.2}/{:>4.2}/{:>4.2}    {:>5.2}  | {:>+5.2}",
            d,
            qacs_uniform[i],
            m_un[0],
            m_un[1],
            m_un[2],
            s_un,
            m_aq[0],
            m_aq[1],
            m_aq[2],
            s_aq,
            s_aq - s_un,
        );
    }
    println!(
        "\nSSIMULACRA2: 100 = identical, 90+ imperceptible, 70 = noticeable,\n50 = significant degradation. Δssim2 > 0 means AQ wins perceptually.\nByte MAE alone often understates AQ benefit (AQ trades smooth-region\nbits for detail bits, which per-pixel L1 averages out)."
    );

    if let Some(dir) = out_dir {
        std::fs::create_dir_all(&dir).expect("create output dir");
        for (i, d) in distances.iter().enumerate() {
            let p_un = format!("{dir}/d{:.1}_uniform.png", d);
            let p_aq = format!("{dir}/d{:.1}_aq.png", d);
            image::RgbImage::from_raw(w, h, out_uniform[i].clone())
                .expect("buf size matches")
                .save(&p_un)
                .expect("save uniform PNG");
            image::RgbImage::from_raw(w, h, out_aq[i].clone())
                .expect("buf size matches")
                .save(&p_aq)
                .expect("save AQ PNG");
        }
        println!("\n✓ Wrote {} PNGs to {dir}/", 2 * distances.len());
    } else {
        println!("\n✓ Sweep complete. Set OUT_DIR=/tmp/aqsweep to write PNGs per distance.");
    }
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
