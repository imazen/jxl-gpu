//! Corpus-level butteraugli refinement loop measurement: runs uniform
//! qac vs initial AQ vs refined AQ over many images at multiple
//! distances and reports aggregate butteraugli statistics.
//!
//! For each image in the corpus directory, encodes three ways at each
//! distance (uniform / AQ / refined) and accumulates per-distance
//! butteraugli deltas. Reports mean, refined-vs-uniform win count,
//! refined-vs-AQ win count, and worst-case regression per distance.
//!
//! This validates whether the per-image refinement gain at d=1.0
//! (-11% on the single CLIC photo in `butteraugli_refinement_demo`)
//! generalizes across content.
//!
//! Usage:
//!   cargo run --release --features butteraugli-loop \
//!     --example butteraugli_refinement_corpus_sweep
//!
//! Optional env vars:
//!   CORPUS_DIR   Directory of PNGs (default: codec-corpus/clic2025-1024)
//!   MAX_IMAGES   Cap on images to process (default: 8)
//!   ITERS        Refinement iters (default: 2)

#[cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::butteraugli_loop::{
        ButteraugliLoopGpu, SmartGatePath, linear_planar_to_srgb_u8_interleaved,
        refine_aq_field_gpu, refine_aq_field_gpu_smart,
    };
    use jxl_encoder_gpu::lossy_encoder::{LossyEncoder, distance_to_qac};

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let corpus_dir = std::env::var("CORPUS_DIR")
        .unwrap_or_else(|_| "/home/lilith/work/codec-corpus/clic2025-1024".to_string());
    let max_images: usize = std::env::var("MAX_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&corpus_dir)
        .unwrap_or_else(|e| panic!("read_dir {corpus_dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
        .collect();
    paths.sort();
    paths.truncate(max_images);

    let distances = [1.0_f32, 2.0, 4.0];

    println!("=== butteraugli_refinement_corpus_sweep ===");
    println!(
        "Corpus:    {corpus_dir}\nImages:    {}\nDistances: {:?}\nIters:     {iters}\n",
        paths.len(),
        distances
    );

    // Per-distance aggregates
    let nd = distances.len();
    let mut sum_un = vec![0.0_f64; nd];
    let mut sum_aq = vec![0.0_f64; nd];
    let mut sum_rf = vec![0.0_f64; nd];
    let mut sum_sm = vec![0.0_f64; nd]; // smart-gate
    let mut wins_rf_vs_un = vec![0_usize; nd];
    let mut wins_rf_vs_aq = vec![0_usize; nd];
    let mut losses_rf_vs_un = vec![0_usize; nd];
    let mut worst_rf_vs_un: Vec<(f32, String)> = vec![(0.0, String::new()); nd];
    let mut smart_paths: Vec<[usize; 3]> = vec![[0; 3]; nd]; // [DistanceGated, AqRegressed, Refined]

    let to_linear = |c: u8| -> f32 {
        let f = c as f32 / 255.0;
        if f <= 0.04045 {
            f / 12.92
        } else {
            ((f + 0.055) / 1.055).powf(2.4)
        }
    };

    println!(
        "{:>3}  {:>15.15}  {}",
        "#",
        "image",
        distances
            .iter()
            .map(|d| format!("d={d:>3.1} (un / AQ / refined / smart / Δrf-un)"))
            .collect::<Vec<_>>()
            .join("  ")
    );
    for (img_idx, path) in paths.iter().enumerate() {
        let img = match image::open(path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                eprintln!("[skip] {path:?}: {e}");
                continue;
            }
        };
        let (w, h) = img.dimensions();
        let pixels: Vec<u8> = img.into_raw();
        let n = (w * h) as usize;

        // IEC linearize
        let mut r = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for chunk in pixels.chunks_exact(3) {
            r.push(to_linear(chunk[0]));
            g.push(to_linear(chunk[1]));
            b.push(to_linear(chunk[2]));
        }

        let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);
        let mut bg = ButteraugliLoopGpu::new_multires(&enc, w, h);

        let measure = |bg: &mut ButteraugliLoopGpu<Backend>,
                       rec_r: &[f32],
                       rec_g: &[f32],
                       rec_b: &[f32]|
         -> f32 {
            let recon_srgb = linear_planar_to_srgb_u8_interleaved(
                rec_r,
                rec_g,
                rec_b,
                w as usize,
                h as usize,
            );
            bg.compute_with_reference(&recon_srgb)
                .expect("compute_with_reference")
                .score
        };

        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        print!("{:>3}  {:>15.15}", img_idx + 1, name);
        bg.set_reference(&pixels)
            .expect("set_reference for baselines");

        for (di, &d) in distances.iter().enumerate() {
            // Uniform
            let q_un = distance_to_qac(d);
            let (rr, gg, bb) = lossy.encode_one(&enc, &r, &g, &b, q_un);
            let s_un = measure(&mut bg, &rr, &gg, &bb);
            // Initial AQ
            let initial_aq = lossy.compute_aq_field(&enc, &r, &g, &b, d);
            let (rr, gg, bb) = lossy.encode_one_adaptive(&enc, &r, &g, &b, &initial_aq);
            let s_aq = measure(&mut bg, &rr, &gg, &bb);
            // Refined
            let refined = refine_aq_field_gpu(
                &enc,
                &lossy,
                &mut bg,
                &r,
                &g,
                &b,
                &pixels,
                &initial_aq,
                d,
                iters,
                |_| (),
            )
            .expect("refine_aq_field_gpu");
            let (rr, gg, bb) = lossy.encode_one_adaptive(&enc, &r, &g, &b, &refined);
            let s_rf = measure(&mut bg, &rr, &gg, &bb);

            // Smart-gate: distance + content-aware. Use the same
            // initial AQ to avoid a second AQ encode (the smart fn
            // measures internally; we measure it by encoding the
            // resulting field).
            let smart_outcome = refine_aq_field_gpu_smart(
                &enc,
                &lossy,
                &mut bg,
                &r,
                &g,
                &b,
                &pixels,
                &initial_aq,
                d,
                iters,
                |_| (),
            )
            .expect("refine_aq_field_gpu_smart");
            // smart_outcome is correct as a "decision". Measure the
            // selected field's actual butteraugli score.
            let (rrs, ggs, bbs) =
                lossy.encode_one_adaptive(&enc, &r, &g, &b, &smart_outcome.aq_field);
            let s_sm = measure(&mut bg, &rrs, &ggs, &bbs);
            match smart_outcome.path {
                SmartGatePath::DistanceGated => smart_paths[di][0] += 1,
                SmartGatePath::AqRegressedFallToUniform => smart_paths[di][1] += 1,
                SmartGatePath::Refined => smart_paths[di][2] += 1,
            }

            sum_un[di] += s_un as f64;
            sum_aq[di] += s_aq as f64;
            sum_rf[di] += s_rf as f64;
            sum_sm[di] += s_sm as f64;
            let drf_un = s_rf - s_un;
            if s_rf < s_un - 0.005 {
                wins_rf_vs_un[di] += 1;
            } else if s_rf > s_un + 0.005 {
                losses_rf_vs_un[di] += 1;
                if drf_un > worst_rf_vs_un[di].0 {
                    worst_rf_vs_un[di] = (drf_un, name.to_string());
                }
            }
            if s_rf < s_aq - 0.005 {
                wins_rf_vs_aq[di] += 1;
            }
            print!(
                "  {:>5.2}/{:>5.2}/{:>5.2}/{:>5.2}/{:>+6.3}",
                s_un, s_aq, s_rf, s_sm, drf_un
            );
        }
        println!();
    }

    let nf = paths.len() as f64;
    println!("\n=== Aggregate (n={}, lower butteraugli = better) ===", paths.len());
    println!(
        "  {:>5}  {:>9}  {:>9}  {:>9}  {:>9}  {:>5}/{:>5}  {:>5}  {}",
        "dist", "uniform µ", "AQ µ", "refined µ", "smart µ", "rf>un", "rf<un", "rf<AQ",
        "smart paths [DistGate / AQ→un / Refined]"
    );
    for (di, d) in distances.iter().enumerate() {
        println!(
            "  {:>5.2}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}  {:>5}/{:>5}  {:>5}  {} / {} / {}",
            d,
            sum_un[di] / nf,
            sum_aq[di] / nf,
            sum_rf[di] / nf,
            sum_sm[di] / nf,
            wins_rf_vs_un[di],
            losses_rf_vs_un[di],
            wins_rf_vs_aq[di],
            smart_paths[di][0],
            smart_paths[di][1],
            smart_paths[di][2],
        );
    }
    println!("\nWorst refinement-vs-uniform regressions (per distance):");
    for (di, d) in distances.iter().enumerate() {
        let (worst_d, worst_name) = &worst_rf_vs_un[di];
        if losses_rf_vs_un[di] > 0 {
            println!(
                "  d={:.2}  {:>15.15} ({:+.3})",
                d,
                worst_name.chars().take(15).collect::<String>(),
                worst_d
            );
        } else {
            println!("  d={:.2}  (none)", d);
        }
    }
    println!(
        "\nrf<un / rf<AQ counts: 'wins' = score > 0.005 better. The refined\nloop is doing useful work if rf<un > rf>un at the same distance.\nWorst-loss column shows the worst refinement regression vs uniform\n— useful for identifying content types where refinement should be\nskipped or tuned differently."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop")))]
fn main() {
    eprintln!("requires --features 'cuda encoder butteraugli-loop'");
    std::process::exit(2);
}
