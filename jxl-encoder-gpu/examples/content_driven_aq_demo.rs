//! Content-driven adaptive quantization on a real image.
//!
//! Uses [`LossyEncoder::encode_one_with_aq`] — the turnkey API that
//! runs the full content-driven AQ chain (XYB → mask1x1 → per-block
//! reduce → derived qac field → adaptive encode) in one call.
//!
//! For comparison this demo also calls [`LossyEncoder::compute_aq_field`]
//! directly so we can print the qac range, then a uniform-qac baseline
//! (single scalar) to show what AQ buys you.
//!
//! See `adaptive_quant_demo` for a walkthrough of
//! `encode_one_adaptive` with a manually-built qac field.

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

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("failed to open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let pixels: Vec<u8> = img.into_raw();
    let n = (w * h) as usize;

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

    println!("=== content_driven_aq_demo ===");
    println!("Image: {image_path}");
    println!("Size: {w}×{h}, padded {pw}×{ph}, {nb} blocks\n");

    let distance = 2.0_f32;

    // Inspect the field that encode_one_with_aq will use internally.
    let aq_field = lossy.compute_aq_field(&enc, &r, &g, &b, distance);
    let qac_min = aq_field.iter().copied().fold(f32::INFINITY, f32::min);
    let qac_max = aq_field.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "Derived qac field: [{qac_min:.3}, {qac_max:.3}]  (centered on distance={distance})\n"
    );

    // Turnkey content-driven AQ.
    let t0 = std::time::Instant::now();
    let (rec_r_aq, rec_g_aq, rec_b_aq) = lossy.encode_one_with_aq(&enc, &r, &g, &b, distance);
    let dt_aq = t0.elapsed();

    // Uniform baseline at the same central distance.
    let qac_uniform = distance_to_qac(distance);
    let t1 = std::time::Instant::now();
    let (rec_r_un, rec_g_un, rec_b_un) = lossy.encode_one(&enc, &r, &g, &b, qac_uniform);
    let dt_un = t1.elapsed();

    let mut mae_aq = [0.0_f64; 3];
    let mut mae_un = [0.0_f64; 3];
    for i in 0..n {
        mae_aq[0] += (r[i] - rec_r_aq[i]).abs() as f64;
        mae_aq[1] += (g[i] - rec_g_aq[i]).abs() as f64;
        mae_aq[2] += (b[i] - rec_b_aq[i]).abs() as f64;
        mae_un[0] += (r[i] - rec_r_un[i]).abs() as f64;
        mae_un[1] += (g[i] - rec_g_un[i]).abs() as f64;
        mae_un[2] += (b[i] - rec_b_un[i]).abs() as f64;
    }
    let nf = n as f64;
    println!(
        "encode_one_with_aq (content-driven AQ): {:.2} ms",
        dt_aq.as_secs_f64() * 1000.0
    );
    println!(
        "  R={:.4e} G={:.4e} B={:.4e}",
        mae_aq[0] / nf,
        mae_aq[1] / nf,
        mae_aq[2] / nf
    );
    println!(
        "\nencode_one (uniform qac=distance_to_qac({distance})): {:.2} ms",
        dt_un.as_secs_f64() * 1000.0
    );
    println!(
        "  R={:.4e} G={:.4e} B={:.4e}",
        mae_un[0] / nf,
        mae_un[1] / nf,
        mae_un[2] / nf
    );
    println!(
        "\nThe AQ pass uses heavier quant in smooth regions (where the\nhigh mask values mean the eye is less sensitive), reclaiming\nbits for detail regions. MAE is not the right metric for AQ\nbenefit (you'd need SSIMULACRA2 etc.); but the demo confirms\nthe full content-driven AQ pipeline composes end-to-end."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
