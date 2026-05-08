//! Butteraugli quant-refinement loop on a real image.
//!
//! Demonstrates [`forks::butteraugli_loop::refine_aq_field_gpu`] — the
//! end-to-end orchestrator that takes an initial per-block aq_field
//! (e.g., from content-driven AQ) and iteratively refines it by
//! measuring perceptual distance with `zenmetrics/butteraugli-gpu`.
//!
//! Reports per-iteration butteraugli score progression so you can see
//! whether the loop is actually moving scores in the expected direction
//! (lower = better quality at the same target distance).
//!
//! Set `IMAGE_PATH` env var to point at a different RGB image; defaults
//! to a CLIC 2025 photo from `~/work/codec-corpus/clic2025-1024/`.
//! Set `ITERS` env var to control loop length (default 2 = effort 8).
//! Set `DISTANCE` env var to control target butteraugli distance
//! (default 1.0).

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::butteraugli_loop::{
        ButteraugliLoopGpu, linear_planar_to_srgb_u8_interleaved, refine_aq_field_gpu,
    };
    use jxl_encoder_gpu::lossy_encoder::{LossyEncoder, distance_to_qac};

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let image_path = std::env::var("IMAGE_PATH").unwrap_or_else(|_| {
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
            .to_string()
    });
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let distance: f32 = std::env::var("DISTANCE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("failed to open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let pixels: Vec<u8> = img.into_raw();
    let n = (w * h) as usize;

    // Use the same IEC 61966-2-1 piecewise transfer that butteraugli-gpu
    // uses internally. Linearizing input via simplified powf(2.4) instead
    // creates an asymmetry: recon → IEC sRGB U8, butteraugli linearizes
    // via IEC, original via IEC — so the pipeline operating on a
    // gamma-2.4 linear interpretation doesn't match what butteraugli
    // sees as ground truth.
    let to_linear = |c: u8| -> f32 {
        let f = c as f32 / 255.0;
        if f <= 0.04045 {
            f / 12.92
        } else {
            ((f + 0.055) / 1.055).powf(2.4)
        }
    };
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
    let mut bg = ButteraugliLoopGpu::new_multires(&enc, w, h);

    println!("=== butteraugli_refinement_demo ===");
    println!("Image:         {image_path}");
    println!("Size:          {w}x{h} (padded {pw}x{ph}, {nb} blocks)");
    println!("Distance:      {distance:.2}");
    println!("Iters:         {iters} (loop runs {} iterations total)\n", iters + 1);

    // Helper: encode → linear sRGB U8 → butteraugli score against
    // ORIGINAL pixel bytes. Returns (score, pnorm_3).
    let measure_score = |bg: &mut ButteraugliLoopGpu<Backend>,
                         rec_r: &[f32],
                         rec_g: &[f32],
                         rec_b: &[f32]|
     -> (f32, f32) {
        let recon_srgb = linear_planar_to_srgb_u8_interleaved(
            rec_r,
            rec_g,
            rec_b,
            w as usize,
            h as usize,
        );
        let result = bg
            .compute_with_reference(&recon_srgb)
            .expect("compute_with_reference");
        (result.score, result.pnorm_3)
    };

    // Need the reference uploaded ONCE for all baseline measurements.
    bg.set_reference(&pixels)
        .expect("set_reference for baselines");

    // Baseline 1: encode with uniform qac (no AQ). This is what
    // distance_to_qac(distance) maps to.
    let qac_uniform = distance_to_qac(distance);
    println!(
        "Baseline measurements (qac_uniform = distance_to_qac({distance:.2}) = {qac_uniform:.3}):"
    );
    let (rec_r_un, rec_g_un, rec_b_un) = lossy.encode_one(&enc, &r, &g, &b, qac_uniform);
    let (score_un, pn3_un) = measure_score(&mut bg, &rec_r_un, &rec_g_un, &rec_b_un);
    println!("  uniform qac:    score={score_un:.4}  pnorm_3={pn3_un:.4}");

    // Baseline 2: initial content-driven AQ field (no refinement).
    let initial_aq = lossy.compute_aq_field(&enc, &r, &g, &b, distance);
    let qac_min = initial_aq.iter().copied().fold(f32::INFINITY, f32::min);
    let qac_max = initial_aq.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let qac_mean = initial_aq.iter().copied().sum::<f32>() / nb as f32;
    let (rec_r_aq, rec_g_aq, rec_b_aq) =
        lossy.encode_one_adaptive(&enc, &r, &g, &b, &initial_aq);
    let (score_aq, pn3_aq) = measure_score(&mut bg, &rec_r_aq, &rec_g_aq, &rec_b_aq);
    println!(
        "  initial AQ:     score={score_aq:.4}  pnorm_3={pn3_aq:.4}  (qac min={qac_min:.3} max={qac_max:.3} mean={qac_mean:.3})\n"
    );

    // Use the original sRGB U8 bytes directly as the butteraugli
    // reference. Round-tripping through transfer functions (esp. our
    // simplified powf(2.4) ↔ butteraugli-gpu's IEC piecewise) inflates
    // scores even on bit-perfect reconstructions.
    let mut traces = Vec::new();
    let t0 = std::time::Instant::now();
    let refined = refine_aq_field_gpu(
        &enc,
        &lossy,
        &mut bg,
        &r,
        &g,
        &b,
        &pixels,
        &initial_aq,
        distance,
        iters,
        |t| traces.push(t),
    )
    .expect("refine_aq_field_gpu");
    let dt = t0.elapsed();

    println!("Per-iteration butteraugli scores:");
    println!("  {:>4}  {:>9}  {:>9}  {:>9}", "iter", "score", "pnorm_3", "td_max");
    for t in &traces {
        let td_max = t.tile_dist.iter().copied().fold(0.0_f32, f32::max);
        let suffix = if t.iter == t.iters { "  (compare-only, no adjust)" } else { "" };
        println!(
            "  {:>4}  {:>9.4}  {:>9.4}  {:>9.4}{}",
            t.iter, t.score, t.pnorm_3, td_max, suffix
        );
    }

    let score_delta = traces
        .last()
        .map(|last| last.score - traces[0].score)
        .unwrap_or(0.0);
    let pnorm3_delta = traces
        .last()
        .map(|last| last.pnorm_3 - traces[0].pnorm_3)
        .unwrap_or(0.0);
    let qac_min_r = refined.iter().copied().fold(f32::INFINITY, f32::min);
    let qac_max_r = refined.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let qac_mean_r = refined.iter().copied().sum::<f32>() / nb as f32;
    let qac_drift = (refined.iter().zip(initial_aq.iter()))
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / nb as f64;

    println!(
        "\nRefined aq_field: min={qac_min_r:.3} max={qac_max_r:.3} mean={qac_mean_r:.3}"
    );
    println!("Mean |refined - initial|: {qac_drift:.4}");
    println!(
        "\nScore delta:    {score_delta:+.4}  (negative = better, target_distance={distance})"
    );
    println!("pnorm_3 delta:  {pnorm3_delta:+.4}");
    println!(
        "\nTotal refine_aq_field_gpu time: {:.0} ms ({:.0} ms/iter avg)",
        dt.as_secs_f64() * 1000.0,
        dt.as_secs_f64() * 1000.0 / (iters + 1) as f64
    );
    // Final comparison: refined AQ score vs baselines.
    let refined_score = traces.last().map(|t| t.score).unwrap_or(0.0);
    let refined_pn3 = traces.last().map(|t| t.pnorm_3).unwrap_or(0.0);
    println!("\n=== Quality summary (lower butteraugli = better) ===");
    println!("  uniform qac:    score={score_un:.4}  pnorm_3={pn3_un:.4}");
    println!("  initial AQ:     score={score_aq:.4}  pnorm_3={pn3_aq:.4}");
    println!("  refined AQ:     score={refined_score:.4}  pnorm_3={refined_pn3:.4}");
    println!(
        "  AQ vs uniform:    {:+.4} ({:+.1}%)",
        score_aq - score_un,
        100.0 * (score_aq - score_un) / score_un
    );
    println!(
        "  refined vs AQ:    {:+.4} ({:+.1}%)",
        refined_score - score_aq,
        100.0 * (refined_score - score_aq) / score_aq
    );
    println!(
        "  refined vs uniform: {:+.4} ({:+.1}%)",
        refined_score - score_un,
        100.0 * (refined_score - score_un) / score_un
    );
    println!(
        "\nNote: this is the qac-domain adaptation — the integer-step\n\
         minimum bump from upstream's float-qf model is neutered (our\n\
         pipeline uses float qac directly). Score progression should\n\
         still show the loop converging toward target_distance, but the\n\
         exact per-iter dynamics differ from upstream's CPU loop."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop")))]
fn main() {
    eprintln!("requires --features 'cuda encoder butteraugli-loop'");
    std::process::exit(2);
}
