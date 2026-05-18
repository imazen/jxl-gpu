// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! AFV preservation across patches case-1 recompute — diagnostic readout.
//!
//! This is the chunk-1 deliverable of the "GPU patches AFV preservation
//! across case-1 recompute" follow-on to W7-3 (`406b40bb`). W7-3
//! shipped the auto-AFV dispatch on screenshots; this example quantifies
//! how many GPU AFV picks the patches case-1 recompute discards on
//! the same screenshot corpus, separated into the AFV→AFV, AFV→DCT8,
//! and AFV→other transition classes.
//!
//! ## What it measures
//!
//! For each image:
//! - Encodes once with `with_auto_evaluate_afv_on_screenshots(true)` so
//!   the GPU strategy-search plan produces AFV picks (when the
//!   median-mask-1x1 > 95 + e>=7 gate fires — i.e. on screenshot
//!   content).
//! - Reads `diagnostics::take_last_afv_preservation_stats()` after the
//!   encode completes. This sink is populated inside the patches case-1
//!   path in `encoder.rs:2120` and `encoder.rs:2639`:
//!     - `patches_recompute_fired` — whether
//!       `find_and_build_patches` returned `Some` AND we ran the CPU
//!       `compute_ac_strategy` recompute on patches-subtracted XYB.
//!     - `gpu_afv_picks_pre` / `cpu_afv_picks_post` — first-block AFV
//!       counts before/after the recompute.
//!     - `gpu_strategy_histogram_pre` / `cpu_strategy_histogram_post` —
//!       full first-block histograms (19 strategy classes).
//! - Computes (and prints) the wedge:
//!     - `afv_lost = gpu_afv_picks_pre - cpu_afv_picks_post` (clamped
//!       at 0 — CPU can also gain AFV picks the GPU didn't make, in
//!       which case `lost` is meaningless and we report `gained`).
//!     - Where did the lost picks go? Compare pre vs post DCT8,
//!       DCT4x4, DCT2x2, IDENTITY, DCT4x8, DCT8x4 counts and report
//!       the deltas. This is the input chunk-2 needs to decide
//!       between (A) pure cost-model precision merge, (B) per-class
//!       host-side re-evaluation on patches-subtracted XYB.
//!
//! ## Expected outcomes (chunk-1 hypothesis)
//!
//! Per the W7-3 commit message:
//!   - `terminal.png` d=1.0: 40 picks pre, ~0 post (full wipe).
//!   - `windows.png` d=1.0: 264 picks pre, ~0 post (full wipe).
//!   - `gmessages.png` / `graph.png` / `gui.png`: pre = post (no
//!     patches detected on these, so the recompute never fires — the
//!     baseline snapshot reports `patches_recompute_fired = false`
//!     and the counts are byte-identical).
//!   - Photo images (`02809272…`, `07b9f93f…`, `22ea12c9…`): both pre
//!     AND post are 0 — the auto-AFV gate (mask1x1 median > 95) never
//!     fires on photos so the GPU plan has no AFV picks to begin with.
//!
//! The histogram delta is the key chunk-2 input: if the wedge is
//! AFV→DCT8 (most), the merge can plausibly recover it (DCT8 is also
//! 1x1 covered — restoring AFV doesn't break covered-region
//! consistency, but the cost on patches-subtracted XYB must be
//! re-evaluated host-side to avoid quality regressions). If the wedge
//! is AFV→DCT4x4 / DCT2x2, the patches-subtracted content prefers
//! different sub-block transforms and the GPU AFV picks are no longer
//! optimal — chunk 2 should NOT try to restore them.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p jxl-encoder-gpu \
//!   --features 'cuda encoder' \
//!   --example afv_preservation_diagnostic -- [--distance D] [--image PATH]...
//! ```
//!
//! With no `--image` args, uses the same default corpus the W7-3 sweep
//! used (10 gb82-sc screenshots) at `--distance 1.0` so the readout
//! lines up with `benchmarks/auto_afv_screenshots_sweep_2026-05-17.txt`.

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
}

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::diagnostics::take_last_afv_preservation_stats;
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

    type B = cubecl::cuda::CudaRuntime;

    // Parse args.
    let raw: Vec<String> = std::env::args().collect();
    let mut distance: f32 = 1.0;
    let mut images: Vec<String> = Vec::new();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--distance" => {
                distance = raw[i + 1].parse().expect("--distance D");
                i += 2;
            }
            "--image" => {
                images.push(raw[i + 1].clone());
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    if images.is_empty() {
        // Default validation set: same 10 gb82-sc screenshots W7-3
        // used so the chunk-1 readout aligns with the W7-3 sweep at
        // `benchmarks/auto_afv_screenshots_sweep_2026-05-17.txt`.
        let base = "/home/lilith/work/codec-corpus/gb82-sc";
        let candidates = [
            "codec_wiki.png",
            "gmessages.png",
            "graph.png",
            "gui.png",
            "imac_dark.png",
            "imac_g3.png",
            "imessage.png",
            "terminal.png",
            "windows95.png",
            "windows.png",
        ];
        for c in &candidates {
            let path = format!("{base}/{c}");
            if std::path::Path::new(&path).exists() {
                images.push(path);
            } else {
                eprintln!("[skip-missing] {path}");
            }
        }
    }

    println!(
        "[afv_preservation_diagnostic] distance={distance} images={} (auto-AFV ON)",
        images.len()
    );
    println!(
        "{:<28} {:>5} {:>5} {:>7} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "image",
        "MP",
        "fired",
        "afv_pre",
        "afv_post",
        "afv_lost",
        "dct8_pre",
        "dct8_post",
        "afv→dct8?",
        "d4x4_post",
        "id_post",
    );

    let mut total_pre = 0u32;
    let mut total_post = 0u32;
    let mut total_dct8_pre = 0u32;
    let mut total_dct8_post = 0u32;

    for path in &images {
        let img = match image::open(path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                eprintln!("[skip] {path}: {e}");
                continue;
            }
        };
        let (w, h) = img.dimensions();
        let n = (w * h) as usize;
        let pixels_u8 = img.into_raw();

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
        for chunk in pixels_u8.chunks_exact(3) {
            r.push(to_linear(chunk[0]));
            g.push(to_linear(chunk[1]));
            b.push(to_linear(chunk[2]));
        }

        let enc: GpuEncoder<B> = GpuEncoder::new();
        // Auto-AFV ON path (new default since W7-3).
        let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, w, h);
        let _bs = enc
            .encode_lossy_to_bitstream_via_precomputed(&lossy, &r, &g, &b, distance)
            .unwrap_or_else(|e| panic!("encode failed for {path}: {e:?}"));
        let stats = take_last_afv_preservation_stats()
            .expect("diagnostics sink must be populated by encode");

        let afv_pre = stats.gpu_afv_picks_pre;
        let afv_post = stats.cpu_afv_picks_post;
        let afv_lost = afv_pre.saturating_sub(afv_post);
        let dct8_pre = stats.gpu_dct8_picks_pre;
        let dct8_post = stats.cpu_dct8_picks_post;
        let dct8_gain = dct8_post.saturating_sub(dct8_pre);
        // Upper-bound estimate: "afv→dct8" approximated as min(afv_lost, dct8_gain).
        // True value would need per-block tracking which AcStrategyMap
        // doesn't expose. Comparing to other class deltas below shows
        // the residual went elsewhere.
        let afv_to_dct8_upper_bound = afv_lost.min(dct8_gain);
        let d4x4_post = stats.cpu_strategy_histogram_post[7]; // RAW_STRATEGY_DCT4X4
        let id_post = stats.cpu_strategy_histogram_post[8]; // RAW_STRATEGY_IDENTITY

        let short = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        let mp = n as f32 / 1e6;
        println!(
            "{:<28} {:>5.2} {:>5} {:>7} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10}",
            short,
            mp,
            stats.patches_recompute_fired,
            afv_pre,
            afv_post,
            afv_lost,
            dct8_pre,
            dct8_post,
            afv_to_dct8_upper_bound,
            d4x4_post,
            id_post,
        );

        // Optional per-image deeper breakdown when there's a non-trivial wedge.
        if stats.patches_recompute_fired && afv_lost > 0 {
            print!("    histogram delta (post - pre): ");
            let names = [
                "DCT8", "DCT16X8", "DCT8X16", "DCT16X16", "DCT32X32", "DCT4X8", "DCT8X4", "DCT4X4",
                "IDENTITY", "DCT2X2", "DCT32X16", "DCT16X32", "AFV0", "AFV1", "AFV2", "AFV3",
                "DCT64X64", "DCT64X32", "DCT32X64",
            ];
            let mut shown = 0;
            for k in 0..19 {
                let delta = stats.cpu_strategy_histogram_post[k] as i64
                    - stats.gpu_strategy_histogram_pre[k] as i64;
                if delta != 0 {
                    if shown > 0 {
                        print!("  ");
                    }
                    print!("{}={:+}", names[k], delta);
                    shown += 1;
                }
            }
            println!();
        }

        total_pre += afv_pre;
        total_post += afv_post;
        total_dct8_pre += dct8_pre;
        total_dct8_post += dct8_post;
    }

    println!();
    println!(
        "TOTAL (n={} images): AFV pre = {}, AFV post = {}, AFV lost = {}",
        images.len(),
        total_pre,
        total_post,
        total_pre.saturating_sub(total_post),
    );
    println!(
        "TOTAL DCT8 pre = {}, DCT8 post = {}, DCT8 gained = {} (upper bound on AFV→DCT8)",
        total_dct8_pre,
        total_dct8_post,
        total_dct8_post.saturating_sub(total_dct8_pre),
    );
}
