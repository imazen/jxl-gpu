//! Combined-mode demo: strat-search + butteraugli AQ refinement loop.
//!
//! Compares 4 pipelines on a single image:
//!   1. uniform qac (encode_one)
//!   2. strat-search alone (uniform qac, transform palette)
//!   3. butteraugli AQ refinement on uniform-DCT8 encoder
//!   4. butteraugli AQ refinement on strat-search adaptive encoder ← NEW
//!
//! The expectation per CLAUDE.md "Combining strat-search with
//! butteraugli AQ refinement (open)" is that pipeline 4 should match
//! or beat pipeline 3 by giving the loop a transform palette to work
//! with instead of forcing DCT8 everywhere.
//!
//! `IMAGE_PATH` env var selects input (default: CLIC 2025 1024×1024
//! photo). `ITERS` controls loop length (default 4). `DISTANCE`
//! controls target butteraugli distance (default 1.0).

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::butteraugli_loop::{
        ButteraugliLoopGpu, RefineIterTrace, linear_planar_to_srgb_u8_interleaved,
        refine_and_encode_best_of_both, refine_aq_field_gpu,
        refine_aq_field_gpu_with_strategy_search,
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
        .unwrap_or(4);
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

    println!("=== combined_strat_search_aq_demo ===");
    println!("Image:    {image_path}");
    println!("Size:     {w}x{h} (padded {pw}x{ph}, {nb} blocks)");
    println!("Distance: {distance:.2}");
    println!("Iters:    {iters} (loop runs {} iterations total)\n", iters + 1);

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

    bg.set_reference(&pixels)
        .expect("set_reference for baselines");

    // Pipeline 1: uniform qac at distance.
    let qac_uniform = distance_to_qac(distance);
    let t = std::time::Instant::now();
    let (rec_r_un, rec_g_un, rec_b_un) = lossy.encode_one(&enc, &r, &g, &b, qac_uniform);
    let dt_un = t.elapsed();
    let (score_un, pn3_un) = measure_score(&mut bg, &rec_r_un, &rec_g_un, &rec_b_un);

    // Pipeline 2: strat-search alone.
    // Warm up first (JIT + persistent buffers).
    let _ =
        lossy.encode_one_with_strategy_search_dct8_16(&enc, &r, &g, &b, distance);
    let t = std::time::Instant::now();
    let (rec_r_s, rec_g_s, rec_b_s) =
        lossy.encode_one_with_strategy_search_dct8_16(&enc, &r, &g, &b, distance);
    let dt_s = t.elapsed();
    let (score_s, pn3_s) = measure_score(&mut bg, &rec_r_s, &rec_g_s, &rec_b_s);

    // Strategy histogram: prepare a plan and count assignments by raw_strategy.
    let plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
    let mut histo: std::collections::BTreeMap<u8, usize> =
        std::collections::BTreeMap::new();
    for a in &plan.assignments {
        *histo.entry(a.raw_strategy).or_insert(0) += 1;
    }
    let n_assignments: usize = histo.values().sum();
    let strat_name = |s: u8| -> &'static str {
        use jxl_encoder_gpu::forks::transform::*;
        match s {
            RAW_STRATEGY_DCT => "DCT8",
            RAW_STRATEGY_DCT16X8 => "DCT16x8",
            RAW_STRATEGY_DCT8X16 => "DCT8x16",
            RAW_STRATEGY_DCT16X16 => "DCT16x16",
            RAW_STRATEGY_DCT32X32 => "DCT32x32",
            RAW_STRATEGY_DCT4X8 => "DCT4x8",
            RAW_STRATEGY_DCT8X4 => "DCT8x4",
            RAW_STRATEGY_DCT4X4 => "DCT4x4",
            RAW_STRATEGY_DCT32X16 => "DCT32x16",
            RAW_STRATEGY_DCT16X32 => "DCT16x32",
            RAW_STRATEGY_DCT64X64 => "DCT64x64",
            RAW_STRATEGY_DCT64X32 => "DCT64x32",
            RAW_STRATEGY_DCT32X64 => "DCT32x64",
            RAW_STRATEGY_IDENTITY => "IDENT",
            RAW_STRATEGY_DCT2X2 => "DCT2x2",
            RAW_STRATEGY_AFV0 => "AFV0",
            RAW_STRATEGY_AFV1 => "AFV1",
            RAW_STRATEGY_AFV2 => "AFV2",
            RAW_STRATEGY_AFV3 => "AFV3",
            _ => "??",
        }
    };
    let mut histo_str = String::new();
    for (s, n) in &histo {
        histo_str.push_str(&format!(
            " {}={}({:.0}%)",
            strat_name(*s),
            n,
            100.0 * (*n as f32) / (n_assignments as f32)
        ));
    }
    println!("=== Strat-search assignments ({} regions): ===", n_assignments);
    println!("  {histo_str}\n");

    // Initial AQ field for the refinement loop.
    let initial_aq = lossy.compute_aq_field(&enc, &r, &g, &b, distance);

    // Pipeline 3: refine on uniform-DCT8 encoder.
    let mut traces_un: Vec<RefineIterTrace> = Vec::new();
    let t = std::time::Instant::now();
    let _refined_un = refine_aq_field_gpu(
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
        |trace| traces_un.push(trace),
    )
    .expect("refine_aq_field_gpu");
    let dt_refine_un = t.elapsed();
    let (score_refine_un, pn3_refine_un) = traces_un
        .last()
        .map(|t| (t.score, t.pnorm_3))
        .unwrap_or((f32::NAN, f32::NAN));

    // Pipeline 4: refine on strat-search adaptive encoder. ← NEW
    let mut traces_ss: Vec<RefineIterTrace> = Vec::new();
    let t = std::time::Instant::now();
    let refined_ss = refine_aq_field_gpu_with_strategy_search(
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
        |trace| traces_ss.push(trace),
    )
    .expect("refine_aq_field_gpu_with_strategy_search");
    let dt_refine_ss = t.elapsed();
    let (score_refine_ss, pn3_refine_ss) = traces_ss
        .last()
        .map(|t| (t.score, t.pnorm_3))
        .unwrap_or((f32::NAN, f32::NAN));

    // Stats on refined fields.
    let qac_min_ss = refined_ss.iter().copied().fold(f32::INFINITY, f32::min);
    let qac_max_ss = refined_ss.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let qac_mean_ss = refined_ss.iter().copied().sum::<f32>() / nb as f32;

    println!("=== Per-iteration progression ===");
    println!("Pipeline 3 (refine on uniform-DCT8 encoder):");
    println!("  {:>4}  {:>9}  {:>9}", "iter", "score", "pnorm_3");
    for t in &traces_un {
        println!("  {:>4}  {:>9.4}  {:>9.4}", t.iter, t.score, t.pnorm_3);
    }
    println!();
    println!("Pipeline 4 (refine on strat-search adaptive encoder):");
    println!("  {:>4}  {:>9}  {:>9}", "iter", "score", "pnorm_3");
    for t in &traces_ss {
        println!("  {:>4}  {:>9.4}  {:>9.4}", t.iter, t.score, t.pnorm_3);
    }
    println!();
    println!(
        "Refined-SS aq_field stats: min={qac_min_ss:.3} max={qac_max_ss:.3} mean={qac_mean_ss:.3}"
    );

    println!("\n=== Quality summary (lower butteraugli = better) ===");
    println!(
        "  1. uniform qac          score={score_un:.4}  pnorm_3={pn3_un:.4}  ({:.0} ms)",
        dt_un.as_secs_f64() * 1000.0
    );
    println!(
        "  2. strat-search alone   score={score_s:.4}  pnorm_3={pn3_s:.4}  ({:.0} ms)",
        dt_s.as_secs_f64() * 1000.0
    );
    println!(
        "  3. refine + DCT8        score={score_refine_un:.4}  pnorm_3={pn3_refine_un:.4}  ({:.0} ms total, {:.0} ms/iter)",
        dt_refine_un.as_secs_f64() * 1000.0,
        dt_refine_un.as_secs_f64() * 1000.0 / (iters + 1) as f64
    );
    println!(
        "  4. refine + strat-search score={score_refine_ss:.4}  pnorm_3={pn3_refine_ss:.4}  ({:.0} ms total, {:.0} ms/iter)",
        dt_refine_ss.as_secs_f64() * 1000.0,
        dt_refine_ss.as_secs_f64() * 1000.0 / (iters + 1) as f64
    );

    let pct = |x: f32, ref_: f32| 100.0 * (x - ref_) / ref_;
    println!("\n=== Combined-mode delta ===");
    println!(
        "  refine+strat vs refine+DCT8:  {:+.4} ({:+.2}%)",
        score_refine_ss - score_refine_un,
        pct(score_refine_ss, score_refine_un)
    );
    println!(
        "  refine+strat vs strat alone:  {:+.4} ({:+.2}%)",
        score_refine_ss - score_s,
        pct(score_refine_ss, score_s)
    );
    println!(
        "  refine+strat vs uniform:      {:+.4} ({:+.2}%)",
        score_refine_ss - score_un,
        pct(score_refine_ss, score_un)
    );
    let cost_overhead = dt_refine_ss.as_secs_f64() / dt_refine_un.as_secs_f64();
    println!(
        "  combined-mode encode cost:    {cost_overhead:.2}× refine+DCT8 cost"
    );

    // Pipeline 5: best-of-both (uncompromising-quality wrapper).
    // Runs both pipelines, picks the lower-butteraugli winner.
    let t = std::time::Instant::now();
    let (_bob_r, _bob_g, _bob_b, bob_path, bob_scores) = refine_and_encode_best_of_both(
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
    )
    .expect("refine_and_encode_best_of_both");
    let dt_bob = t.elapsed();

    println!("\n=== Best-of-both pipeline (uncompromising mode) ===");
    println!(
        "  picked: {bob_path:?}  (DCT8: {:.4}, strat: {:.4}; pnorm_3 DCT8: {:.4}, strat: {:.4})",
        bob_scores.dct8_score,
        bob_scores.strat_search_score,
        bob_scores.dct8_pnorm_3,
        bob_scores.strat_search_pnorm_3,
    );
    let bob_score = bob_scores
        .dct8_score
        .min(bob_scores.strat_search_score);
    println!(
        "  winning score: {bob_score:.4}  ({:.0} ms total, {:.2}× refine+DCT8)",
        dt_bob.as_secs_f64() * 1000.0,
        dt_bob.as_secs_f64() / dt_refine_un.as_secs_f64(),
    );
    println!(
        "  best-of-both vs refine+DCT8:  {:+.4} ({:+.2}%)",
        bob_score - score_refine_un,
        pct(bob_score, score_refine_un),
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop")))]
fn main() {
    eprintln!("requires --features 'cuda encoder butteraugli-loop'");
    std::process::exit(2);
}
