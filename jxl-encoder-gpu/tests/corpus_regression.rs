// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Corpus regression test for `refine_and_encode_smart`.
//!
//! Runs the smart turnkey pipeline on a fixed set of images and
//! asserts butteraugli scores are within tolerance against committed
//! expectations. **Designed as a safety net for cost-model
//! experiments** — when we tune DCT64/DCT32 muls, re-enable AFV,
//! port libjxl counterweights, etc., this test catches silent
//! regressions on diverse content.
//!
//! ## Running
//!
//! Required cargo feature: `corpus` (which pulls in cuda + encoder +
//! butteraugli-loop). Off by default in CI to avoid the corpus
//! dependency.
//!
//! ```bash
//! cargo test -p jxl-encoder-gpu --features corpus --test corpus_regression
//! ```
//!
//! ## Failure mode
//!
//! Per CLAUDE.md "NO 'GRACEFUL SKIPS' IN TESTS" — if the corpus tree
//! isn't present, the test FAILS with a clear "corpus required at
//! /home/lilith/work/codec-corpus" message rather than silently
//! returning success. The skip decision is at the cargo invocation
//! level (don't enable the `corpus` feature) — visible in the build
//! chain.
//!
//! ## Updating expected scores
//!
//! When an INTENTIONAL cost-model change shifts scores, update the
//! `EXPECTED_SCORES` const after manually verifying the new scores
//! reflect a quality improvement (or accepted parity tradeoff). Don't
//! relax tolerances to make the test pass — that's a CLAUDE.md "never
//! relax test expectations" violation.

#![cfg(feature = "corpus")]

use jxl_encoder_gpu::encoder::GpuEncoder;
use jxl_encoder_gpu::forks::butteraugli_loop::{
    refine_and_encode_smart, ButteraugliLoopGpu, BestOfBothPath,
};
use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

type Backend = cubecl::cuda::CudaRuntime;

const CORPUS_ROOT: &str = "/home/lilith/work/codec-corpus";

/// Per-(image, distance) expected scores from `refine_and_encode_smart`
/// at 4 iters. Tolerance is 0.5% — tight enough to catch silent
/// regressions, loose enough to absorb GPU floating-point determinism
/// jitter across runs.
///
/// (image_subpath, distance, expected_score, expected_path)
///
/// All d=1.0 scores captured 2026-05-09 with commits up to 5c9e0910;
/// d=0.5 and d=2.0 captured the same evening after the cost-model
/// retune (commits 7a503551 / e813b414 / 65ba3074 / bc3393f0).
///
/// Distance coverage spans the production-relevant range:
/// - d=0.5: high-quality web (the "wow this looks great" tier)
/// - d=1.0: libjxl reference / "visually transparent" target
/// - d=2.0: aggressive web compression where bytes really matter
///
/// At d=2.0, refinement statistically regresses on many photos (the
/// `refine_aq_field_gpu_smart` distance-gate threshold is 1.5 for
/// exactly this reason). Smart's best-of-3 catches that by
/// comparing against uniform and picking the winner, so smart never
/// regresses uniform even at d=2.0.
const EXPECTED_SCORES: &[(&str, f32, f32, BestOfBothPath)] = &[
    // ===== d=1.0 (the original 11-image set) =====
    ("clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
        1.0, 1.1475, BestOfBothPath::Tie),
    ("clic2025-1024/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png",
        1.0, 1.2089, BestOfBothPath::RefineStratSearch),
    ("clic2025-1024/0d154749c7771f58e89ad343653ec4e20d6f037da829f47f5598e5d0a4ab61f0.png",
        1.0, 1.0999, BestOfBothPath::RefineDct8), // uniform won
    ("clic2025-1024/1e2f9d41529197f1.png",
        1.0, 0.8190, BestOfBothPath::RefineDct8), // uniform won
    // DCT64-sensitive photos (52/9/79 picks at mul=8 pre-fix)
    ("clic2025-1024/1cba10ad9bb4ced57e42f7656c5f2a58d32dc6bad084957d2f8d1c78e0fcd224.png",
        1.0, 1.1587, BestOfBothPath::RefineDct8), // FP-tied; epsilon flipped at DCT64=5
    ("clic2025-1024/0c49a5cce349020bbba2f97ae41e90ba.png",
        1.0, 1.1548, BestOfBothPath::RefineDct8),
    ("clic2025-1024/11f2b039b293758398b1a7a8afa64bb2.png",
        1.0, 1.1556, BestOfBothPath::RefineDct8),
    // Strat-search-winning photos
    ("clic2025-1024/22ea12c903e41583.png",
        1.0, 1.1716, BestOfBothPath::RefineStratSearch),
    ("clic2025-1024/2684452db505ddbb.png",
        1.0, 1.1868, BestOfBothPath::RefineStratSearch),
    // Screenshots — discriminator fires
    ("gb82-sc/graph.png", 1.0, 1.0528, BestOfBothPath::SkippedStratSearchAsScreenshot),
    ("gb82-sc/gmessages.png", 1.0, 0.9266, BestOfBothPath::SkippedStratSearchAsScreenshot),

    // ===== d=0.5 (high-quality web) =====
    ("clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
        0.5, 0.6641, BestOfBothPath::RefineStratSearch),
    ("clic2025-1024/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png",
        0.5, 0.6705, BestOfBothPath::Tie),
    ("clic2025-1024/0d154749c7771f58e89ad343653ec4e20d6f037da829f47f5598e5d0a4ab61f0.png",
        0.5, 0.5272, BestOfBothPath::Tie),
    ("clic2025-1024/1e2f9d41529197f1.png",
        0.5, 0.5071, BestOfBothPath::Tie),
    ("clic2025-1024/1cba10ad9bb4ced57e42f7656c5f2a58d32dc6bad084957d2f8d1c78e0fcd224.png",
        0.5, 0.6554, BestOfBothPath::RefineDct8),
    ("clic2025-1024/0c49a5cce349020bbba2f97ae41e90ba.png",
        0.5, 0.6380, BestOfBothPath::RefineDct8),
    ("clic2025-1024/11f2b039b293758398b1a7a8afa64bb2.png",
        0.5, 0.7292, BestOfBothPath::RefineDct8),
    // d=0.5 score improved 0.7397 → 0.7109 (-3.9%) when DCT64 mul
    // dropped 6 → 5. Strat-search picks DCT64 here for real gain.
    ("clic2025-1024/22ea12c903e41583.png",
        0.5, 0.7109, BestOfBothPath::RefineStratSearch),
    ("clic2025-1024/2684452db505ddbb.png",
        0.5, 0.6667, BestOfBothPath::RefineStratSearch),
    ("gb82-sc/graph.png", 0.5, 0.5271, BestOfBothPath::SkippedStratSearchAsScreenshot),
    ("gb82-sc/gmessages.png", 0.5, 0.5265, BestOfBothPath::SkippedStratSearchAsScreenshot),

    // ===== d=2.0 (aggressive web compression) =====
    // At d>1.5, refinement statistically regresses on most photos.
    // Smart's best-of-3 catches that — most cases pick the uniform
    // candidate (reported as RefineDct8 path label).
    ("clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
        2.0, 2.1525, BestOfBothPath::RefineDct8),
    ("clic2025-1024/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png",
        2.0, 2.0550, BestOfBothPath::RefineDct8),
    ("clic2025-1024/0d154749c7771f58e89ad343653ec4e20d6f037da829f47f5598e5d0a4ab61f0.png",
        2.0, 1.8807, BestOfBothPath::RefineDct8),
    ("clic2025-1024/1e2f9d41529197f1.png",
        2.0, 1.5000, BestOfBothPath::RefineDct8),
    ("clic2025-1024/1cba10ad9bb4ced57e42f7656c5f2a58d32dc6bad084957d2f8d1c78e0fcd224.png",
        2.0, 1.9975, BestOfBothPath::RefineDct8),
    ("clic2025-1024/0c49a5cce349020bbba2f97ae41e90ba.png",
        2.0, 2.0202, BestOfBothPath::RefineDct8),
    ("clic2025-1024/11f2b039b293758398b1a7a8afa64bb2.png",
        2.0, 2.0866, BestOfBothPath::RefineStratSearch),
    ("clic2025-1024/22ea12c903e41583.png",
        2.0, 2.0949, BestOfBothPath::RefineDct8),
    ("clic2025-1024/2684452db505ddbb.png",
        2.0, 2.1252, BestOfBothPath::RefineDct8),
    ("gb82-sc/graph.png", 2.0, 1.4447, BestOfBothPath::SkippedStratSearchAsScreenshot),
    ("gb82-sc/gmessages.png", 2.0, 1.6462, BestOfBothPath::SkippedStratSearchAsScreenshot),
];

/// Tolerance for score comparison. 0.5% is tight enough to catch
/// real regressions; absorbs GPU FP determinism jitter (typically
/// ≤0.01% on these workloads).
const TOLERANCE: f32 = 0.005;

const ITERS: usize = 4;

#[test]
fn corpus_regression_smart_turnkey_d1_iter4() {
    // Hard fail (not skip) if corpus root absent — per CLAUDE.md
    // "no graceful skips in tests".
    if !std::path::Path::new(CORPUS_ROOT).exists() {
        panic!(
            "corpus_regression: corpus tree not found at {CORPUS_ROOT}. \
             This test requires the codec-corpus tree; either symlink it \
             at the expected path or run without --features corpus."
        );
    }

    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    // Resolve any approximate filenames to actual on-disk names
    // (CLIC files have hash-suffix variants).
    let mut failures = Vec::<String>::new();
    let mut ran = 0usize;

    for (subpath, distance, expected_score, expected_path) in EXPECTED_SCORES {
        let mut full_path = format!("{CORPUS_ROOT}/{subpath}");
        if !std::path::Path::new(&full_path).exists() {
            // Try fuzzy match by hash prefix (CLIC images have varying
            // suffix lengths).
            let parent = std::path::Path::new(&full_path)
                .parent()
                .map(|p| p.to_path_buf());
            let stem_prefix = std::path::Path::new(subpath)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.chars().take(16).collect::<String>())
                .unwrap_or_default();
            if let Some(parent) = parent {
                if let Ok(entries) = std::fs::read_dir(&parent) {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().into_owned();
                        if name.starts_with(&stem_prefix) && name.ends_with(".png") {
                            full_path = entry.path().to_string_lossy().into_owned();
                            break;
                        }
                    }
                }
            }
        }
        if !std::path::Path::new(&full_path).exists() {
            failures.push(format!("missing image: {subpath}"));
            continue;
        }

        // Load image as RGB8 → linear f32 (IEC 61966-2-1 sRGB inverse
        // EOTF, matching the demo + butteraugli-gpu reference).
        let img = match image::open(&full_path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                failures.push(format!("image load failed for {subpath}: {e}"));
                continue;
            }
        };
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
        let mut bg = ButteraugliLoopGpu::new_multires(&enc, w, h);
        let initial_aq = lossy.compute_aq_field(&enc, &r, &g, &b, *distance);

        let result = refine_and_encode_smart(
            &enc, &lossy, &mut bg, &r, &g, &b, &pixels, &initial_aq, *distance, ITERS,
        );
        let (_rec_r, _rec_g, _rec_b, path, scores) = match result {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("smart turnkey failed for {subpath}: {e:?}"));
                continue;
            }
        };
        let actual_score = if scores.strat_search_score.is_nan() {
            scores.dct8_score
        } else {
            scores.dct8_score.min(scores.strat_search_score)
        };

        let rel_err = ((actual_score - expected_score) / expected_score).abs();
        let path_match = path == *expected_path;
        if rel_err > TOLERANCE || !path_match {
            failures.push(format!(
                "{subpath} @ d={distance}: score expected={expected_score:.4} actual={actual_score:.4} \
                 rel_err={rel_err:.4} (tol={TOLERANCE:.4}); \
                 path expected={expected_path:?} actual={path:?}",
            ));
        }
        ran += 1;
    }

    if !failures.is_empty() {
        panic!(
            "corpus_regression: {} failure(s) of {} image(s) ran:\n  {}",
            failures.len(),
            ran + failures.len(),
            failures.join("\n  "),
        );
    }
    assert!(ran > 0, "corpus_regression: no images ran (all missing?)");
    let tol_pct = TOLERANCE * 100.0;
    let distinct_imgs = EXPECTED_SCORES
        .iter()
        .map(|(p, _, _, _)| *p)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    eprintln!(
        "corpus_regression: {ran} (image, distance) cases ran across \
         {distinct_imgs} distinct images, all within {tol_pct:.1}% tolerance"
    );
}
