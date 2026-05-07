//! Content-driven adaptive quantization on a real image.
//!
//! Derives a per-block qac field from the image's mask1x1 field
//! (a per-pixel masking signal that's high in smooth regions and
//! low at edges). Smooth blocks get heavier quant (smaller files),
//! detail blocks get lighter quant (preserved edges).
//!
//! Pipeline:
//!   linear-RGB → XYB → mask1x1 (per pixel, GPU) → per-block reduce
//!   (CPU mean over each 8×8 block) → scale to qac range
//!   → LossyEncoder::encode_one_adaptive with the derived field
//!
//! Compares against a uniform-quant baseline (single qac scalar)
//! and reports per-half MAE plus written-PNG size for both.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::adaptive_quant::compute_mask1x1_gpu;
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
    let blocks_per_row = (pw / 8) as usize;
    let blocks_per_col = (ph / 8) as usize;
    let nb = blocks_per_row * blocks_per_col;

    println!("=== content_driven_aq_demo ===");
    println!("Image: {image_path}");
    println!("Size: {w}×{h}, padded {pw}×{ph}, {nb} blocks\n");

    // ── Step 1: derive per-block qac field from mask1x1 ──────────
    // 1a. compute XYB Y plane on GPU (just for the mask1x1 input).
    let (_xyb_x, xyb_y, _xyb_b) = enc.xyb_from_linear_rgb(&r, &g, &b);
    // 1b. compute mask1x1 field (one f32 per pixel) on GPU.
    let mask = compute_mask1x1_gpu(&enc, &xyb_y, w as usize, h as usize);
    // 1c. per-block mean of the mask, normalized to qac range.
    //     Mask is HIGH in smooth regions (low gradient) and LOW at edges.
    //     Smooth → heavier quant (smaller qac); detail → lighter quant
    //     (larger qac). So qac inversely correlates with mask intensity.
    let mut block_means = vec![0.0_f32; nb];
    for by in 0..blocks_per_col {
        for bx in 0..blocks_per_row {
            // Average mask over the 8×8 block (clamped to image bounds).
            let mut sum = 0.0_f64;
            let mut count = 0_usize;
            for dy in 0..8 {
                let y = by * 8 + dy;
                if y >= h as usize {
                    break;
                }
                for dx in 0..8 {
                    let x = bx * 8 + dx;
                    if x >= w as usize {
                        break;
                    }
                    sum += mask[y * w as usize + x] as f64;
                    count += 1;
                }
            }
            block_means[by * blocks_per_row + bx] =
                if count > 0 { (sum / count as f64) as f32 } else { 1.0 };
        }
    }
    // Normalize: map block_means' range to [qac_min, qac_max].
    let mean_min = block_means.iter().copied().fold(f32::INFINITY, f32::min);
    let mean_max = block_means.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let qac_max = distance_to_qac(0.5); // detail → light quant
    let qac_min = distance_to_qac(4.0); // smooth → heavy quant
    println!(
        "Mask range: [{mean_min:.3}, {mean_max:.3}]  →  qac range [{qac_min:.3}, {qac_max:.3}]"
    );
    let aq_field: Vec<f32> = block_means
        .iter()
        .map(|&m| {
            let t = if mean_max > mean_min {
                (m - mean_min) / (mean_max - mean_min)
            } else {
                0.5
            };
            // Inverse: high mask (smooth) → low qac (heavy quant).
            qac_max + (qac_min - qac_max) * t
        })
        .collect();

    // ── Step 2: encode_one_adaptive with the content-driven field ──
    let t0 = std::time::Instant::now();
    let (rec_r_aq, rec_g_aq, rec_b_aq) =
        lossy.encode_one_adaptive(&enc, &r, &g, &b, &aq_field);
    let dt_aq = t0.elapsed();

    // ── Step 3: encode with uniform qac at midpoint distance for comparison ──
    let qac_uniform = distance_to_qac(2.0);
    let t1 = std::time::Instant::now();
    let (rec_r_un, rec_g_un, rec_b_un) =
        lossy.encode_one(&enc, &r, &g, &b, qac_uniform);
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
        "\nencode_one_adaptive (content-driven AQ): {:.2} ms",
        dt_aq.as_secs_f64() * 1000.0
    );
    println!(
        "  R={:.4e} G={:.4e} B={:.4e}",
        mae_aq[0] / nf,
        mae_aq[1] / nf,
        mae_aq[2] / nf
    );
    println!(
        "\nencode_one (uniform qac=distance_to_qac(2.0)): {:.2} ms",
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
