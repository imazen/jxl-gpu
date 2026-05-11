// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Diagnostic: measure the divergence between libjxl's `quant_norm16`
//! (L16 norm of per-8x8 adaptive_quant values over a strategy region)
//! and the scalar `distance_to_qac(distance)` that the GPU strat-search
//! currently passes as `quant_for_coeffs`.
//!
//! Hypothesis (recorded in
//! `~/.claude/.../memory/cost_model_wedge_analysis.md`): the wedge
//! that keeps DCT32's hardcoded entropy_mul at 3.0 vs libjxl's 1.48
//! is caused by this divergence. On photos with non-uniform adaptive
//! quant, libjxl's L16 norm biases toward the HIGHEST per-8x8 quant
//! value in the region → makes DCT32 look WORSE in libjxl than in our
//! GPU code → cjxl picks DCT32 less aggressively on edges → matches
//! our band-aid direction.
//!
//! This example loads a CLIC photo, computes aq_field, and reports:
//!   - For each 32×32 region: quant_norm16 vs scalar quant_y, ratio.
//!   - Histogram of ratios (how many regions deviate >5%, >10%, etc).
//!
//! If divergence is small (<5% for >90% of regions), the hypothesis is
//! wrong. If large, plumb per-block quant_norm16 through
//! per_block_upstream_cost next.
//!
//! Run:
//!   cargo run --release -p jxl-encoder-gpu --features 'cuda encoder' \
//!     --example quant_norm16_divergence -- --image PATH [--distance D]

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
}

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::{distance_to_qac, LossyEncoder};

    type B = cubecl::cuda::CudaRuntime;

    let raw: Vec<String> = std::env::args().collect();
    let mut image_path: Option<String> = None;
    let mut distance: f32 = 1.0;
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--image" => {
                image_path = Some(raw[i + 1].clone());
                i += 2;
            }
            "--distance" => {
                distance = raw[i + 1].parse().expect("--distance D");
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    let image_path = image_path.expect("--image PATH required");

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let pixels_u8: Vec<u8> = img.into_raw();
    let n = (w as usize) * (h as usize);

    println!(
        "quant_norm16_divergence: src {}×{} ({:.2} MP), distance={}",
        w, h, n as f32 / 1e6, distance,
    );

    let to_linear = |c: u8| -> f32 {
        let f = c as f32 / 255.0;
        if f <= 0.04045 { f / 12.92 } else { ((f + 0.055) / 1.055).powf(2.4) }
    };
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for chunk in pixels_u8.chunks_exact(3) {
        r.push(to_linear(chunk[0]));
        g.push(to_linear(chunk[1]));
        b.push(to_linear(chunk[2]));
    }

    let enc: GpuEncoder<B> = GpuEncoder::new();
    let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, w, h);

    let aq_field = lossy.compute_aq_field(&enc, &r, &g, &b, distance);
    let scalar_qac = distance_to_qac(distance);

    let (pw, ph) = lossy.padded_dimensions();
    let xs8 = (pw as usize) / 8;
    let ys8 = (ph as usize) / 8;
    assert_eq!(aq_field.len(), xs8 * ys8);

    // L16 norm over a region of N 8x8 blocks: pow(sum(q^16)/N, 1/16).
    // Implementing via pow chain `q*q`, `*=`, etc. — same as libjxl.
    let quant_norm16 = |aq: &[f32], x0: usize, y0: usize, cx: usize, cy: usize| -> f32 {
        if cx == 1 && cy == 1 {
            aq[y0 * xs8 + x0]
        } else if cx + cy == 3 {
            // num_blocks=2: libjxl uses max of the two
            let a = aq[y0 * xs8 + x0];
            let b = if cy == 2 { aq[(y0 + 1) * xs8 + x0] } else { aq[y0 * xs8 + (x0 + 1)] };
            a.max(b)
        } else {
            let mut acc = 0.0f32;
            let n = (cx * cy) as f32;
            for iy in 0..cy {
                for ix in 0..cx {
                    let q = aq[(y0 + iy) * xs8 + (x0 + ix)];
                    let mut q16 = q * q;
                    q16 *= q16;
                    q16 *= q16;
                    q16 *= q16; // q^16
                    acc += q16;
                }
            }
            (acc / n).powf(1.0 / 16.0)
        }
    };

    let strategies = [
        ("DCT16x16", 2usize, 2usize),
        ("DCT16x8 ", 1, 2),
        ("DCT8x16 ", 2, 1),
        ("DCT32x32", 4, 4),
        ("DCT32x16", 2, 4),
        ("DCT16x32", 4, 2),
        ("DCT64x64", 8, 8),
    ];

    println!(
        "scalar quant (distance_to_qac({})) = {:.5}",
        distance, scalar_qac
    );
    println!();
    println!("{:<10} {:>6} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7}",
             "strategy", "n", "mean", "median", "min", "max",
             "p5%>", "p10%>", "p20%>");

    for (name, cx, cy) in &strategies {
        if xs8 < *cx || ys8 < *cy { continue; }
        let mut ratios = Vec::new();
        let yb_steps = ys8.saturating_sub(*cy - 1);
        let xb_steps = xs8.saturating_sub(*cx - 1);
        // Step by region size (non-overlapping, as the strategy
        // assignments would actually use).
        for y0 in (0..yb_steps).step_by(*cy) {
            for x0 in (0..xb_steps).step_by(*cx) {
                let qn16 = quant_norm16(&aq_field, x0, y0, *cx, *cy);
                ratios.push(qn16 / scalar_qac);
            }
        }
        if ratios.is_empty() { continue; }
        let mut sorted = ratios.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let mean = ratios.iter().sum::<f32>() / ratios.len() as f32;
        let med = sorted[sorted.len() / 2];
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let p5 = ratios.iter().filter(|r| (**r - 1.0).abs() > 0.05).count();
        let p10 = ratios.iter().filter(|r| (**r - 1.0).abs() > 0.10).count();
        let p20 = ratios.iter().filter(|r| (**r - 1.0).abs() > 0.20).count();
        let n_total = ratios.len();
        println!(
            "{:<10} {:>6} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>5.1}% {:>5.1}% {:>5.1}%",
            name, n_total, mean, med, min, max,
            100.0 * p5 as f32 / n_total as f32,
            100.0 * p10 as f32 / n_total as f32,
            100.0 * p20 as f32 / n_total as f32,
        );
    }
    println!();
    println!("hypothesis: if median deviates >5% OR p10% > 30%, the wedge fix needs per-block quant_norm16.");
}
