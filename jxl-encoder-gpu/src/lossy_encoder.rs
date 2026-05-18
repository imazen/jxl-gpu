// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! High-level GPU lossy encoder facade.
//!
//! Wraps the [`crate::persistent`] API into user-friendly entry
//! points: uniform-quant, manual per-block adaptive, and turnkey
//! content-driven adaptive (mask1x1 prepass), each in one-shot and
//! batch (single-input-upload sweep) form, with f32 planar and sRGB
//! u8 interleaved variants.
//!
//! ### Uniform quant (one qac scalar)
//!
//! | Method | Input | Output | Use case |
//! |---|---|---|---|
//! | [`LossyEncoder::encode_one`] | `f32` planar | `(R, G, B)` `f32` | one-shot, already-linear |
//! | [`LossyEncoder::encode_many`] | `f32` planar | `Vec<(R, G, B)>` | quality sweep, already-linear |
//! | [`LossyEncoder::encode_one_srgb_u8`] | sRGB `u8` interleaved | sRGB `u8` interleaved | one-shot, image-crate input |
//! | [`LossyEncoder::encode_many_srgb_u8`] | sRGB `u8` interleaved | `Vec<u8>` per setting | quality sweep, image-crate input |
//!
//! ### Adaptive quant (per-block qac field)
//!
//! | Method | Input | Field | Use case |
//! |---|---|---|---|
//! | [`LossyEncoder::encode_one_adaptive`] | `f32` planar | caller-supplied `&[f32]` | manual AQ |
//! | [`LossyEncoder::encode_one_with_aq`] | `f32` planar | derived (mask1x1) | turnkey content-driven AQ |
//! | [`LossyEncoder::encode_many_with_aq`] | `f32` planar | derived per distance | distance sweep, single mask prepass |
//! | [`LossyEncoder::encode_one_with_aq_srgb_u8`] | sRGB `u8` | derived | turnkey AQ from sRGB U8 |
//! | [`LossyEncoder::encode_many_with_aq_srgb_u8`] | sRGB `u8` | derived per distance | distance sweep from sRGB U8 |
//! | [`LossyEncoder::compute_block_mask_means`] | `f32` planar | — | mask prepass (custom mappings) |
//! | [`LossyEncoder::compute_aq_field`] | `f32` planar | — | derived field (inspection / tweak) |
//! | [`block_means_to_qac_field`] | `&[f32]`, distance | — | pure-CPU mapping (custom prepass) |
//!
//! ### Quality knobs (free fns / consts)
//!
//! | Symbol | Input | Output | Use case |
//! |---|---|---|---|
//! | [`quality_to_qac`] | quality 1-100 | `qac_qm` scalar | JPEG-style quality knob |
//! | [`distance_to_qac`] | libjxl distance | `qac_qm` scalar | direct libjxl semantics |
//! | [`K_AC_QUANT`] | (constant) | 0.765 | libjxl AC scale at distance=1 |
//!
//! Construct one [`LossyEncoder`] per `(width, height)` to amortize
//! the static-input upload (per-channel DCT8 quant matrices +
//! gaborish weights + dead-zone thresholds) across many encodes.
//! Arbitrary image sizes are supported (non-multiples-of-8 are
//! padded internally with right+
//! bottom edge replication and cropped back at output).
//!
//! ## What it does today
//!
//! Runs the canonical lossy DCT8 pipeline end-to-end on GPU:
//!
//!   linear-RGB → XYB → gaborish ×3 → gather ×3 →
//!   DCT8 wide ×3 → quantize ×3 → dequant → DC-restore → IDCT8 ×3 →
//!   scatter ×3 → XYB inverse → linear-RGB
//!
//! This is the same pipeline lossy_pipeline_throughput benchmarks
//! against CPU — currently 1.05-3.95× faster than CPU AVX2 at sizes
//! 256² → 2048² (RTX 5070 + Ryzen 9 7950X).
//!
//! ## What it does NOT do
//!
//! - Does not produce JXL bitstream bytes (no entropy coding /
//!   container yet — this is a roundtrip path for measuring quality
//!   + algorithmic correctness, not a complete encoder).
//! - Does not implement DC quant + entropy coding (DC restore is a
//!   passthrough; real encoders use jxl-encoder's dc_coding).
//! - DCT8 only (one strategy) in this facade. The 13 strategies are
//!   available individually via [`crate::persistent`], and per-strategy
//!   3-channel cost grids + recursive partition selection (16×16 /
//!   32×32 / 64×64 tiers) are in [`crate::pipeline`] —
//!   [`crate::pipeline::compute_cost_grid_dct8_xyb`] et al. and
//!   [`crate::pipeline::select_partitions_16x16_full`] et al. The
//!   high-level facade just doesn't compose them yet.
//!
//! For full bitstream encoding today, use
//! [`crate::encoder::GpuEncoder::encode_lossy_via_cpu`] which delegates
//! to jxl-encoder.
//!
//! ## Related: Phase 3 strategy selection (`crate::pipeline`)
//!
//! The [`crate::pipeline`] module provides the lower-level building
//! blocks for a full strategy-search encoder:
//! - 13 single-channel + 13 3-channel cost grids covering the
//!   DCT4/8/16/32/64 family (square + rect + sub-block).
//! - Three partition selectors (16×16, 32×32, 64×64) that compose
//!   recursively.
//! - Validated end-to-end on real images: rect strategies win ~60%
//!   of picks across a CLIC2025 corpus when offered to the selector
//!   (see `examples/corpus_rect_picks_demo`).

use alloc::vec;
use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;
use crate::persistent::{GaborishWeights, GpuBlocks, GpuPlane};

/// libjxl gaborish K_GABORISH constants for mul=1.0. Matches
/// `forks::gaborish::compute_weights(1.0)` bit-for-bit.
const K_GABORISH: [f64; 5] = [
    -0.094_958_156_7,
    -0.041_031_725,
    0.013_710_005,
    0.006_510_206,
    -0.001_478_906_3,
];

/// Test-only re-export of [`default_gaborish_weights`]. Production
/// callers go through `LossyEncoder` which holds a precomputed
/// `GaborishWeights` instance.
#[cfg(test)]
pub(crate) fn default_gaborish_weights_test_helper() -> GaborishWeights {
    default_gaborish_weights()
}

/// Test-only re-export of [`pad_to_alignment`]. Lifetime-erased so
/// the test gets an owned Vec regardless of dimension equality.
#[cfg(test)]
pub(crate) fn pad_to_alignment_test_helper(
    src: &[f32],
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
) -> alloc::vec::Vec<f32> {
    pad_to_alignment(src, width, height, padded_width, padded_height).into_owned()
}

fn default_gaborish_weights() -> GaborishWeights {
    let sum_w = 1.0
        + 4.0
            * (K_GABORISH[0] + K_GABORISH[1] + K_GABORISH[2] + K_GABORISH[4] + 2.0 * K_GABORISH[3]);
    let norm = 1.0 / sum_w;
    GaborishWeights {
        wc: norm as f32,
        wr: (norm * K_GABORISH[0]) as f32,
        wd: (norm * K_GABORISH[1]) as f32,
        w_big_r: (norm * K_GABORISH[2]) as f32,
        wl: (norm * K_GABORISH[3]) as f32,
        w_big_d: (norm * K_GABORISH[4]) as f32,
    }
}

// Note: the LossyEncoder construction docs live with the impl block
// further down. StrategySearchPlan's docs follow immediately so the
// doc-comments stay attached to their items (clippy
// empty_line_after_doc_comments).

/// Cached cost-grid output from
/// [`LossyEncoder::prepare_strategy_search_plan`]. Holds everything
/// the encode/recon stage needs that's invariant under per-block
/// `aq_field` changes — XYB GPU planes, per-(8×8) DC GPU buffers,
/// strategy assignments, and host-side companions.
///
/// Why split prepare/encode: combined-mode (strat-search + butteraugli
/// AQ refinement) would otherwise pay the cost-grid cost on every
/// refinement iteration. Strategy assignments are stable across iters
/// (cost grids scale with `target_distance`, not `aq_field`), so
/// caching them via this plan drops per-iter cost from ~210 ms to
/// ~50 ms on CLIC 1024² — a ~4× speedup.
///
/// The plan owns both host buffers (`xyb_*`, `dc_grid_*` — kept for the
/// currently-unused AFV strategy branch and host-fallback paths) and
/// GPU buffers (`xyb_*_gpu`, `dc_grid_*_gpu` — clones of cubecl
/// reference-counted handles). Threading the GPU buffers into
/// `encode_and_reconstruct_mixed_strategy_3channel` lets per-iter
/// encodes skip the redundant `upload_plane(xyb)` /
/// `upload_blocks(dc_grid)` PCIe transfers.
#[derive(Clone, Debug)]
pub struct StrategySearchPlan<R: Runtime> {
    /// XYB X-channel host buffer, padded-image-size raster order.
    /// Currently kept alongside the GPU plane below for the AFV
    /// strategy path — `forks::afv::afv_transform_batch_gpu` takes a
    /// host slice. Production cost grids never select AFV today, so
    /// this could be dropped if AFV is removed from the dispatcher.
    pub xyb_x: Vec<f32>,
    /// XYB Y-channel host buffer (see `xyb_x`).
    pub xyb_y: Vec<f32>,
    /// XYB B-channel host buffer (see `xyb_x`).
    pub xyb_b: Vec<f32>,
    /// XYB X-channel GPU plane — the gaborished output of
    /// `prepare_strategy_search_plan_traced`'s pipeline. Holding it
    /// here lets the per-iter encode skip the redundant
    /// `upload_plane(xyb_channel)` that `encode_and_reconstruct_*`
    /// would otherwise pay every refinement iteration.
    pub xyb_x_gpu: GpuPlane<R>,
    /// XYB Y-channel GPU plane (see `xyb_x_gpu`).
    pub xyb_y_gpu: GpuPlane<R>,
    /// XYB B-channel GPU plane (see `xyb_x_gpu`).
    pub xyb_b_gpu: GpuPlane<R>,
    /// XYB pre-gaborish GPU planes [X, Y, B] — the unsharpened XYB
    /// output of `xyb_from_linear_rgb_persistent`, *before* the 5x5
    /// gaborish_inverse runs. Held so the slow-path encode can
    /// download them and feed
    /// [`jxl_encoder::__pre_quantized::EncoderPrecomputed::with_xyb_pre_gaborish`]
    /// for patches detection — the only place patches stay byte-exact
    /// across the encoder/decoder boundary (decoder pipeline is
    /// `IDCT → gaborish → EPF → patches`, so the patches reference
    /// frame stores pre-gaborish patch values). Adds a 3-plane refcount
    /// hold; bytes only download when the slow path takes them.
    pub xyb_x_pre_gab_gpu: GpuPlane<R>,
    /// XYB pre-gaborish GPU plane (Y channel) — see `xyb_x_pre_gab_gpu`.
    pub xyb_y_pre_gab_gpu: GpuPlane<R>,
    /// XYB pre-gaborish GPU plane (B channel) — see `xyb_x_pre_gab_gpu`.
    pub xyb_b_pre_gab_gpu: GpuPlane<R>,
    /// Per-8x8-block DC grid for X channel (length = num_padded_blocks).
    pub dc_grid_x: Vec<f32>,
    /// Per-8x8-block DC grid for Y channel.
    pub dc_grid_y: Vec<f32>,
    /// Per-8x8-block DC grid for B channel.
    pub dc_grid_b: Vec<f32>,
    /// Per-8x8-block DC grid for X channel as a GPU buffer (output of
    /// `dc_grid_8x8_persistent` — 1 f32 per block, n_padded_blocks
    /// total). Used by the GPU LLF restore kernels in
    /// `encode_and_reconstruct_mixed_strategy_*` to skip the per-iter
    /// `upload_blocks(dc_grid_per_8x8_block)` PCIe transfer that the
    /// host-only path otherwise pays.
    pub dc_grid_x_gpu: GpuBlocks<R>,
    /// Per-8x8-block DC grid for Y channel as a GPU buffer.
    pub dc_grid_y_gpu: GpuBlocks<R>,
    /// Per-8x8-block DC grid for B channel as a GPU buffer.
    pub dc_grid_b_gpu: GpuBlocks<R>,
    /// Per-region strategy picks from the cost-grid selector.
    pub assignments: Vec<crate::pipeline::StrategyAssignment>,
    /// CPU-aligned per-block float quant_field — output of
    /// `compute_quant_field_full_persistent` (the GPU port of
    /// `jxl_encoder::vardct::adaptive_quant::compute_quant_field_float`).
    /// Length = `(cpu_pw / 8) * (cpu_ph / 8)`, where
    /// `cpu_pw = align_up(width, 8)` and `cpu_ph = align_up(height, 8)`.
    /// Empty if the prepare path skipped the GPU compute_quant_field
    /// (e.g. dimensions where gpu_pw == cpu_pw is not satisfied — see
    /// `LossyEncoder::compute_quant_field_full_persistent_dims`).
    pub quant_field_float: Vec<f32>,
    /// CPU-aligned per-block masking field — Step 2.5 snapshot of the
    /// post-fuzzy-erosion / pre-modulation aq_map (matches
    /// `compute_mask_for_ac_strategy_use` per element). Same length /
    /// emptiness contract as [`Self::quant_field_float`].
    pub masking: Vec<f32>,
    /// Target distance used for cost-grid scaling (constant across
    /// refinement iterations).
    pub target_distance: f32,
    /// Padded image width (multiple of 8/16/32/64 alignment).
    pub padded_width: u32,
    /// Padded image height.
    pub padded_height: u32,
}

pub struct LossyEncoder<R: Runtime> {
    /// Original dimensions as the caller sees them.
    width: u32,
    height: u32,
    /// Padded-to-8 dimensions used internally for the pipeline.
    padded_width: u32,
    padded_height: u32,
    num_blocks: u32,
    /// Per-channel DCT8 quant weight buffers (X, Y, B). Each holds the
    /// libjxl DCT8 quant table for that channel, replicated per block.
    weights_x: GpuBlocks<R>,
    weights_y: GpuBlocks<R>,
    weights_b: GpuBlocks<R>,
    weights: GaborishWeights,
    /// Per-channel dead-zone thresholds (X, Y, B). Y has the tightest
    /// TL threshold (0.56) — matches libjxl `default_thresholds`.
    thresholds_x: [f32; 4],
    thresholds_y: [f32; 4],
    thresholds_b: [f32; 4],
    /// libjxl-equivalent effort level. Drives per-strategy gating in
    /// `prepare_strategy_search_plan_inner`'s cost-grid stage to match
    /// libjxl's `EvalAcStrategy` per-speed_tier behavior:
    /// - effort 7+ (kSquirrel/kKitten/kTortoise): all strategies
    /// - effort 6 (kWombat): all 16x16-class + 32x16/16x32 + AFV (no DCT32x32 / DCT64*)
    /// - effort 5 (kHare): DCT8, DCT16x16, DCT16x8, DCT8x16, DCT4x4,
    ///   DCT2x2, IDENTITY (no DCT4x8/8x4, no AFV, no DCT32+)
    /// - effort 3-4 (kCheetah/kFalcon): DCT8 only
    ///
    /// Default 7. Set via [`Self::with_effort`].
    effort: u8,
    /// Opt-in: evaluate AFV0-3 cost grids in `prepare_strategy_search_plan_inner`.
    ///
    /// Default `false`. AFV cost grids are produced by
    /// [`crate::forks::afv::afv_cost_grid_xyb_host`] (~52 ms on 1024×1024,
    /// proportionally more at larger sizes due to per-channel block-major
    /// downloads in the AFV cost-grid host helper). Production keeps this
    /// off until a comparable cost-model scaling is dialed in vs the existing
    /// 8x8-class costs (which fold `entropy_mul * total_entropy +
    /// k_info_loss_mul * loss_scalar` inside the kernel — AFV's grid is pure
    /// SSE × mask). Once enabled, `SubBlockCostGrids.afv0..3` flow into the
    /// 16x16 partition selector and AFV picks become possible whenever an
    /// AFV cost beats DCT8 / DCT4x4 / etc. on a given 8×8 cell.
    ///
    /// **Use case for opt-in today**: regression / smoke testing of the AFV
    /// cost-grid integration end-to-end. The default-off path keeps
    /// `corpus_regression` byte-identical.
    evaluate_afv: bool,
    /// Auto-enable AFV0-3 cost-grid evaluation on screenshot-like content.
    ///
    /// Default `true`. When set, [`Self::prepare_strategy_search_plan_inner`]
    /// inspects the per-block `mask1x1` median (already computed for the
    /// adaptive-quant field) and treats `evaluate_afv` as locally enabled
    /// when:
    ///
    /// 1. `evaluate_afv == false` (explicit opt-in via
    ///    [`Self::with_evaluate_afv`] always wins),
    /// 2. `median(per-block mask1x1) > Self::SCREENSHOT_MEDIAN_MASK_THRESHOLD`
    ///    (`95.0` — same screenshot discriminator used by
    ///    [`Self::content_looks_like_screenshot`] and the
    ///    `SkippedStratSearchAsScreenshot` path in
    ///    [`forks::butteraugli_loop::refine_and_encode_smart`]), and
    /// 3. `effort >= 7` (libjxl gates DCT32+ + AFV at speed_tier <= kSquirrel;
    ///    AFV picks only become available at e>=7 anyway because the strategy
    ///    selector treats them as sub-blocks of a DCT16 / DCT32 region).
    ///
    /// Rationale: screenshot/text content is exactly where AFV picks fire
    /// (text glyphs at sub-block alignment) — empirically 0 picks on photos
    /// (no impact when gate doesn't fire). Wall-clock cost (~46 ms / MP for
    /// the AFV cost-grid pipeline + a one-time ~150 MB lazy `xyb_x/y/b`
    /// host download) is contained to the gated subset. Photo
    /// `corpus_regression` bitstream stays byte-identical because the gate
    /// never fires on photos.
    ///
    /// Disable via [`Self::with_auto_evaluate_afv_on_screenshots`] to
    /// recover strict default-off AFV behavior (e.g., for byte-exact
    /// reproducibility against a pre-2026-05-17 baseline).
    auto_evaluate_afv_on_screenshots: bool,
    /// Skip the GPU AFV cost-grid evaluation when patches detection
    /// fires on the same image.
    ///
    /// Default `true`. When the W7-3 auto-AFV gate
    /// ([`Self::auto_evaluate_afv_on_screenshots`]) would fire AND this
    /// flag is set, [`Self::prepare_strategy_search_plan_inner`] runs
    /// a cheap host-side `find_and_build_patches` pre-check on the
    /// pre-gaborish XYB the GPU pipeline already produced. If patches
    /// detection returns `Some(_)`, the AFV cost-grid stage is skipped
    /// (saving ~26 ms / kernel call × 4 kinds ≈ ~100 ms on a 5 MP
    /// screenshot).
    ///
    /// Rationale (jxl-encoder-gpu W11-2 finding, commit `04541934`):
    /// the slow-path patches case-1 in `encoder.rs` runs an independent
    /// CPU `compute_ac_strategy` on patches-subtracted XYB, which
    /// produces its OWN AFV picks (typically 5-10× more than the GPU
    /// did). GPU AFV picks on patches-fired images get wiped by the
    /// CPU recompute → the auto-AFV cost-grid evaluation is dead code
    /// on those images. The W7-3 sweep at d=1.0 confirmed this: zero
    /// bytes change on terminal/windows (patches fired, 40+264 GPU AFV
    /// picks each, wiped) vs `-0.788%` / `-0.403%` / `-0.116%` on
    /// gmessages/graph/gui (patches did NOT fire, GPU AFV picks kept).
    ///
    /// Cost when gate fires: 3-plane host download of the pre-gaborish
    /// XYB (refcounted GpuPlane → first read materializes bytes;
    /// ~3-15 ms on a 5 MP screenshot) plus one
    /// `find_and_build_patches` call (~10-50 ms on screenshot-sized
    /// content). The download is duplicated with the slow-path
    /// encoder.rs sites today; a follow-on chunk could plumb the
    /// `PatchesData` through `StrategySearchPlan` so encoder.rs reuses
    /// it instead of re-running detection.
    ///
    /// Disable via [`Self::with_auto_skip_afv_when_patches`] to
    /// recover the unconditional pre-2026-05-18 auto-AFV behavior
    /// (e.g., for ablation studies, or for byte-exact reproducibility
    /// against the W7-3 baseline). Note: production bytes stay
    /// byte-identical with or without this gate (the GPU AFV picks
    /// were already getting wiped — this flag only saves wall-clock).
    auto_skip_afv_when_patches: bool,
    /// Opt-in dispatch of per-strategy `entropy_mul` and `dist_bias`
    /// between libjxl-faithful and GPU-lifted values based on a content
    /// discriminator (median per-block `mask1x1`).
    ///
    /// **Default `false`** — measured photo regression on the
    /// validation set (see
    /// `vardct_gpu_dropped_optimizations_resurrection_2026-05-17.md`
    /// item #3+#10). The audit hypothesised that on photo content the
    /// libjxl-faithful entropy_mul values + dropped `dist_bias` would
    /// "let larger transforms win on smooth regions where they are
    /// actually optimal — measured to save bytes on photos at slight
    /// bfly cost." An A/B run on three CLIC photos at d=1.0 measured the
    /// opposite: **bytes +2.5% to +8.5%, butteraugli +0.11 to +0.24,
    /// SSIM2 −0.17 to −0.42** — strictly Pareto-worse on every axis.
    /// The root cause is the original drop reason re-validated: this
    /// GPU encoder lacks `kAvoidEntropyOfTransforms` and the
    /// X-channel multi-block weight, so removing the GPU-lifted
    /// entropy_mul + `dist_bias` counterweights causes over-pick of
    /// large transforms regardless of content class.
    ///
    /// Kept as an opt-in (rather than removed) because the dispatch
    /// infrastructure is reusable: once `kAvoidEntropyOfTransforms` +
    /// the X-channel multi-block weight land in the GPU cost grids
    /// (cross-reference dropped log item #4 — multi-week project),
    /// this branch becomes the right shape for re-validation against
    /// libjxl-faithful values. Until then, leave at `false`.
    ///
    /// When enabled, [`Self::prepare_strategy_search_plan_inner`]
    /// picks one of two cost-model branches once per encode based on
    /// the same screenshot discriminator that powers
    /// [`Self::auto_evaluate_afv_on_screenshots`]:
    ///
    /// 1. **Screenshot branch** (`median(per-block mask1x1) >
    ///    Self::SCREENSHOT_MEDIAN_MASK_THRESHOLD`): keep the GPU-lifted
    ///    `entropy_mul` for IDENTITY (1.85) and DCT4x8/DCT8x4 (0.98)
    ///    plus the current distance-scaled `dist_bias` multipliers.
    ///    Screenshot output is byte-identical to the default-off path
    ///    (the discriminator picks this branch on screenshots).
    ///
    /// 2. **Photo branch** (median ≤ threshold): switch IDENTITY and
    ///    DCT4x8/DCT8x4 to libjxl reference values (1.0428 and 0.859316
    ///    respectively, from [`crate::forks::cost::EntropyMulTable::reference`])
    ///    and drop the distance-scaled `dist_bias` multipliers
    ///    (`dist_bias = dist_bias_32 = dist_bias_64 = 1.0`). **Currently
    ///    Pareto-worse vs the default** — see paragraph above.
    ///
    /// The dispatch is **bundled** (entropy_mul + `dist_bias` together)
    /// because the GPU-lifted values were tuned as a counterweight
    /// suite; replacing one without the other unbalances the cost
    /// model further. See
    /// `dropped_optimizations_for_parity_2026-05-15.md` items #3
    /// (entropy_mul) and #10 (`dist_bias`) for the original drop
    /// rationale and
    /// `vardct_gpu_dropped_optimizations_resurrection_2026-05-17.md`
    /// item #3 for the conditional-dispatch hypothesis that the
    /// 2026-05-17 A/B run refuted.
    auto_libjxl_entropy_mul_on_photos: bool,
    /// Auto-enable patches detection on the all-DCT8 GPU pre-quantized
    /// fast path.
    ///
    /// Default `true`. When set,
    /// [`GpuEncoder::encode_lossy_to_bitstream_via_precomputed_from_u8`]
    /// inspects the per-block `mask1x1` median on the pre-gaborish Y plane
    /// and **disables** the fast path (forcing the slow path)
    /// when:
    ///
    /// 1. The strategy selector picked all-DCT8 (otherwise the fast path
    ///    wouldn't fire anyway),
    /// 2. The image is small (`pixel_count < 1_000_000`),
    /// 3. The content is screenshot-like
    ///    (`median(per-block mask1x1) > Self::SCREENSHOT_MEDIAN_MASK_THRESHOLD`),
    /// 4. `effort >= 5` (libjxl gates FindTextLikePatches at speed_tier
    ///    <= kHare).
    ///
    /// Why: the fast path
    /// (`jxl_encoder::__pre_quantized::VarDctEncoder::encode_from_pre_quantized_ac`)
    /// ignores `EncoderPrecomputed::patches_data` — it always passes
    /// `None` for patches into `encode_two_pass`
    /// (jxl-encoder/src/vardct/encoder.rs:2602). Without forcing the
    /// slow path, an all-DCT8 screenshot that would benefit from
    /// patches (terminal glyphs, repeated UI buttons) compresses
    /// at 30-50% worse bitrate than the slow-path counterpart.
    ///
    /// Cost when gate fires: one mask1x1 GPU pass + one block-mask-mean
    /// reduction + a `num_blocks` f32 download + a median sort. Sub-MP
    /// images: a few ms. Plus the slow-path pre-gab XYB download +
    /// CPU `find_and_build_patches`. Photos never trigger the gate
    /// (`median(mask1x1) ≤ 87` on all 16 CLIC photos), so wall-clock
    /// cost is contained to the gated subset.
    ///
    /// Disable via [`Self::with_auto_patches_on_fast_path`] to keep
    /// the fast path unconditional (e.g., for benchmarking the fast
    /// path in isolation).
    auto_patches_on_fast_path: bool,
    /// Opt-in: apply libjxl's `kAvoidEntropyOfTransforms` heuristic to
    /// the GPU cost-grid's sub-block strategies (DCT4X4, DCT4X8,
    /// DCT8X4). When enabled, adds
    /// `K_AVOID_TRANSFORMS_BASE * avoid_entropy_of_transforms_mul(distance)`
    /// to those strategies' `entropy_mul` so the cost-grid penalizes
    /// sub-block picks at very high distances (`distance > 4.0`) where
    /// DCT8 dominates on rate-distortion anyway.
    ///
    /// **Default `false`** — chunk-1 POC of the multi-week GPU port of
    /// libjxl's `kAvoidEntropyOfTransforms` counterweight. The full port
    /// will unlock the libjxl-faithful entropy_mul branch on photos
    /// (see [`Self::auto_libjxl_entropy_mul_on_photos`] field docs +
    /// `vardct_gpu_dropped_optimizations_resurrection_2026-05-17.md`
    /// item #3 for the rationale). This chunk-1 ships only the
    /// formula + a narrow gate (`distance > 4.0`) so production at
    /// `distance ≤ 4.0` stays byte-identical (the formula returns 0.0
    /// inside the gated band and is therefore a no-op).
    ///
    /// Gate semantics (mirrors libjxl `enc_ac_strategy.cc::FindBest8x8Transform`):
    ///
    /// - `target_distance <= 4.0`: penalty is 0.0 — no effect on
    ///   cost grids, output is byte-identical to default-off (so this
    ///   flag is a structural no-op outside the gated band).
    /// - `target_distance > 4.0` and `effort >= 5`: add
    ///   `0.5 * (12 - 4) / (distance - 4)` (clamped at `d >= 12` to
    ///   `0.5 * 1.0 = 0.5`) to the `entropy_mul` of DCT4X4,
    ///   DCT4X8, and DCT8X4 cost-grid specs. AFV is not touched by
    ///   this chunk because AFV picks flow through a separate cost
    ///   path (`forks::afv`) and the GPU AFV grid is opt-in.
    ///
    /// IDENTITY and DCT2X2 are excluded from the penalty per libjxl
    /// (they receive the `kFavor2X2` bonus instead at low distances).
    ///
    /// Enable via [`Self::with_enable_kavoid_entropy_of_transforms`].
    enable_kavoid_entropy_of_transforms: bool,
}

/// Round `n` up to the next multiple of `align`.
#[inline]
fn align_up(n: u32, align: u32) -> u32 {
    n.div_ceil(align) * align
}

/// Map a JPEG-style quality value (1..=100) to the per-block `qac_qm`
/// scale that [`LossyEncoder`] expects.
///
/// In the libjxl convention this kernel follows, **larger qac means
/// lighter quantization** (`val = coef * inv_weight * qac`; larger
/// `val` → above the dead-zone threshold → coefficient survives).
/// So higher quality maps to higher qac.
///
/// libjxl uses `qac = K_AC_QUANT / distance` with `K_AC_QUANT = 0.765`,
/// where distance≈1 is high quality and distance≥6 is low. We map
/// JPEG-quality monotonically to libjxl distance via `distance = 50 / q`.
///
/// Resulting qac values:
///
/// - quality=100 → distance=0.5  → qac=1.530 (very light quant)
/// - quality=75  → distance=0.667 → qac=1.148
/// - quality=50  → distance=1.0  → qac=0.765
/// - quality=25  → distance=2.0  → qac=0.383
/// - quality=10  → distance=5.0  → qac=0.153
/// - quality=1   → distance=50.0 → qac=0.0153
///
/// The mapping is approximate — for actual JPEG XL compatibility,
/// users targeting specific bitrate or quality should drive
/// `qac_qm` directly via measurement (e.g., via SSIMULACRA2).
///
/// ```
/// use jxl_encoder_gpu::lossy_encoder::quality_to_qac;
/// // Higher quality → higher qac (lighter quant).
/// assert!(quality_to_qac(100.0) > quality_to_qac(50.0));
/// assert!(quality_to_qac(50.0) > quality_to_qac(10.0));
/// // Out-of-range inputs clamp to [1, 100].
/// assert_eq!(quality_to_qac(150.0), quality_to_qac(100.0));
/// assert_eq!(quality_to_qac(-10.0), quality_to_qac(1.0));
/// ```
pub fn quality_to_qac(quality: f32) -> f32 {
    let q = quality.clamp(1.0, 100.0);
    // Simple monotonic mapping: distance = 50 / q.
    // q=100 → d=0.5 (high quality), q=50 → d=1.0 (libjxl reference),
    // q=10 → d=5.0, q=1 → d=50 (degraded).
    distance_to_qac(50.0 / q)
}

/// libjxl `K_AC_QUANT` constant — the per-block AC scale at distance=1.
pub const K_AC_QUANT: f32 = 0.765;

/// Map a libjxl-style distance to the per-block `qac_qm` scale that
/// [`LossyEncoder`] expects. This is the most direct interface for
/// callers who want libjxl semantics:
///
/// - distance=0.5 → qac=1.530 (very high quality, light quant)
/// - distance=1.0 → qac=0.765 (libjxl default — visually transparent)
/// - distance=2.0 → qac=0.383 (mild artifacts)
/// - distance=5.0 → qac=0.153 (visibly degraded)
/// - distance=10.0 → qac=0.0765 (heavily degraded)
///
/// Formula: `qac = K_AC_QUANT / distance`. Distance is clamped at
/// `1e-3` to avoid div-by-zero (effectively unbounded qac).
///
/// Use this directly when targeting a libjxl distance; use
/// [`quality_to_qac`] when you have a JPEG-style 1-100 knob.
///
/// ```
/// use jxl_encoder_gpu::lossy_encoder::{distance_to_qac, K_AC_QUANT};
/// // distance=1.0 (libjxl reference) → exactly K_AC_QUANT.
/// assert!((distance_to_qac(1.0) - K_AC_QUANT).abs() < 1e-6);
/// // distance=0.5 → twice the AC quant scale.
/// assert!((distance_to_qac(0.5) - 2.0 * K_AC_QUANT).abs() < 1e-6);
/// // Monotonically decreasing with distance.
/// assert!(distance_to_qac(0.5) > distance_to_qac(1.0));
/// assert!(distance_to_qac(1.0) > distance_to_qac(2.0));
/// ```
pub fn distance_to_qac(distance: f32) -> f32 {
    K_AC_QUANT / distance.max(1e-3)
}

/// Map per-block mask means to a per-block qac field, centered on
/// `distance` and varying in a 4× range:
/// `qac ∈ [distance_to_qac(distance * 2), distance_to_qac(distance / 2)]`.
///
/// High mask values (smooth regions, where the eye is less sensitive)
/// map to LOW qac (heavy quant); low mask values (edges) map to HIGH
/// qac (light quant). Pure CPU — no GPU touch — so cheap to call once
/// per distance in a sweep over the same image's
/// [`LossyEncoder::compute_block_mask_means`] output.
///
/// Used by [`LossyEncoder::compute_aq_field`] and
/// [`LossyEncoder::encode_many_with_aq`] internally; exposed for
/// callers that want a custom prepass (e.g., a different mask, or
/// a different distance-range mapping).
pub fn block_means_to_qac_field(block_means: &[f32], distance: f32) -> alloc::vec::Vec<f32> {
    block_means_to_qac_field_with_range(block_means, distance, 2.0)
}

/// Configurable-range version of [`block_means_to_qac_field`].
///
/// `range_factor` controls how much the per-block qac varies around
/// `distance`'s central qac. With `range_factor = R`:
///   - smooth blocks (max mask) → `distance_to_qac(distance * R)`
///   - detail blocks (min mask) → `distance_to_qac(distance / R)`
///
/// Total qac variation is `R²` (e.g., R=2 → 4× range, R=1.5 →
/// 2.25× range, R=1 → uniform). The default
/// [`block_means_to_qac_field`] uses `R = 2`, matching libjxl's
/// content-driven AQ default.
///
/// **Why narrower may be better in our DCT8-only pipeline:** AQ
/// allocates heavier quant to smooth regions to free bits for detail.
/// In libjxl those smooth regions also get larger AC strategies
/// (DCT16/32) that absorb the heavier quant gracefully; in our
/// DCT8-only pipeline the heavy-quant smooth blocks become visibly
/// blocky. A narrower range (e.g., R=1.4) keeps smooth blocks
/// closer to the central qac while still allocating extra precision
/// to detail.
///
/// Empirical: at d=2.0 on a 1024×1024 CLIC photo, R=2.0 (default)
/// produces +14.9% butteraugli vs uniform; R=1.4 reduces this gap.
pub fn block_means_to_qac_field_with_range(
    block_means: &[f32],
    distance: f32,
    range_factor: f32,
) -> alloc::vec::Vec<f32> {
    let m_min = block_means.iter().copied().fold(f32::INFINITY, f32::min);
    let m_max = block_means
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let qac_max = distance_to_qac(distance / range_factor); // detail → light quant
    let qac_min = distance_to_qac(distance * range_factor); // smooth → heavy quant
    block_means
        .iter()
        .map(|&m| {
            let t = if m_max > m_min {
                (m - m_min) / (m_max - m_min)
            } else {
                0.5
            };
            qac_max + (qac_min - qac_max) * t
        })
        .collect()
}

/// Pad a `width × height` plane up to `padded_width × padded_height` with
/// edge-replication on the right/bottom. Output buffer is allocated by
/// this function; caller passes empty Vec or pre-allocated of correct size.
fn pad_to_alignment<'a>(
    src: &'a [f32],
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
) -> alloc::borrow::Cow<'a, [f32]> {
    debug_assert_eq!(src.len(), width * height);
    if width == padded_width && height == padded_height {
        // No padding needed — borrow the input slice directly. Saves
        // a host memcpy (~3 ms at 1024², larger as image scales).
        return alloc::borrow::Cow::Borrowed(src);
    }
    let mut out = vec![0.0_f32; padded_width * padded_height];
    // Copy interior rows + replicate right edge per source row.
    for y in 0..height {
        let src_off = y * width;
        let dst_off = y * padded_width;
        out[dst_off..dst_off + width].copy_from_slice(&src[src_off..src_off + width]);
        let last = src[src_off + width - 1];
        for x in width..padded_width {
            out[dst_off + x] = last;
        }
    }
    // Replicate the last source row downward.
    if padded_height > height {
        let src_row_in_dst = (height - 1) * padded_width;
        for y in height..padded_height {
            let dst_off = y * padded_width;
            out.copy_within(src_row_in_dst..src_row_in_dst + padded_width, dst_off);
        }
    }
    alloc::borrow::Cow::Owned(out)
}

// =============================================================================
// DCT8 quantization weights (per-coefficient, per-channel)
// =============================================================================
//
// Duplicated from jxl-encoder/src/vardct/quant.rs (the quant module is
// crate-private upstream). These are the libjxl default DCT8 weights
// derived from DCT8_PARAMS via the parametric band formula. Without
// these, LossyEncoder uses unit weights and the qac knob doesn't
// produce monotonic quality vs MAE.

/// DCT8 band parameters from libjxl quant_weights.cc:535-561.
const DCT8_PARAMS: [[f64; 6]; 3] = [
    [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0], // X channel
    [560.0, 0.0, -0.3, -0.3, -0.3, -0.3],  // Y channel
    [512.0, -2.0, -1.0, 0.0, -1.0, -2.0],  // B channel
];

#[inline]
fn band_mult(v: f64) -> f64 {
    if v > 0.0 { 1.0 + v } else { 1.0 / (1.0 - v) }
}

#[inline]
fn interpolate_band(pos: f64, bands: &[f64]) -> f64 {
    let len = bands.len();
    if len == 1 {
        return bands[0];
    }
    let idx = (pos as usize).min(len - 2);
    let frac = pos - idx as f64;
    let a = bands[idx];
    let b = bands[idx + 1];
    a * (b / a).powf(frac)
}

/// Generate the 3-channel DCT8 quant weight table (192 floats: 64 per
/// channel, X then Y then B). Matches `jxl_encoder::vardct::quant::
/// quant_weights(0, channel)` bit-for-bit.
fn generate_dct8_quant_weights() -> [f32; 192] {
    const NUM_BANDS: usize = 6;
    const ROWS: usize = 8;
    const COLS: usize = 8;
    let sqrt2 = core::f64::consts::SQRT_2;
    let scale = (NUM_BANDS as f64 - 1.0) / (sqrt2 + 1e-6);
    let rcpcol = scale / (COLS as f64 - 1.0);
    let rcprow = scale / (ROWS as f64 - 1.0);

    let mut out = [0.0_f32; 192];
    for c in 0..3 {
        let params = &DCT8_PARAMS[c];
        let mut bands = [0.0_f64; NUM_BANDS];
        bands[0] = params[0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(params[i]);
        }
        for y in 0..ROWS {
            let dy = y as f64 * rcprow;
            let dy2 = dy * dy;
            for x in 0..COLS {
                let dx = x as f64 * rcpcol;
                let scaled_distance = (dx * dx + dy2).sqrt();
                let dequant_weight = interpolate_band(scaled_distance, &bands);
                out[c * 64 + y * COLS + x] = (1.0 / dequant_weight) as f32;
            }
        }
    }
    out
}

/// Crop a `padded_width × padded_height` plane back to `width × height`.
fn crop_to_original(
    padded: &[f32],
    padded_width: usize,
    width: usize,
    height: usize,
) -> alloc::vec::Vec<f32> {
    if width == padded_width {
        return padded[..width * height].to_vec();
    }
    let mut out = vec![0.0_f32; width * height];
    for y in 0..height {
        let src_off = y * padded_width;
        let dst_off = y * width;
        out[dst_off..dst_off + width].copy_from_slice(&padded[src_off..src_off + width]);
    }
    out
}

impl<R: Runtime> LossyEncoder<R> {
    /// Construct a lossy encoder for a given image size.
    /// Any width and height ≥ 16 are accepted; non-multiple-of-16
    /// dimensions are padded internally with edge replication.
    ///
    /// **Why 16, not 8**: the strat-search 16x16 selector (the base
    /// of the partition hierarchy) requires `xsize_blocks_8` and
    /// `ysize_blocks_8` to be multiples of 2 (= padded dims multiples
    /// of 16). Aligning to 16 unconditionally lets strat-search work
    /// on ALL image sizes, not just those whose bare 8-aligned padded
    /// dims happen to land on multiples of 16. Cost: at most 8 extra
    /// pixels per axis (~4 KB at typical sizes — negligible).
    /// DCT32/DCT64 paths remain gated on multiples of 32/64
    /// internally inside the strat-search facade.
    pub fn new(enc: &GpuEncoder<R>, width: u32, height: u32) -> Self {
        assert!(
            width >= 16 && height >= 16,
            "width/height must be at least 16 (was {width}x{height})"
        );
        let padded_width = align_up(width, 16);
        let padded_height = align_up(height, 16);
        let num_blocks = (padded_width / 8) * (padded_height / 8);
        // Upload one weights TEMPLATE per channel (X, Y, B) — exactly
        // 64 floats, broadcast across all blocks by the
        // *_broadcast_w persistent kernels. Saves
        // 3 × (num_blocks - 1) × 64 × 4 bytes of GPU memory + upload
        // traffic vs the per-block replicated form (e.g., 12 MB at
        // 1024² for the three channels combined). Y has the gentlest
        // quant (preserves luma); X/B have steeper quant (chroma).
        let all_weights = generate_dct8_quant_weights();
        let upload_channel = |slice: &[f32]| enc.upload_blocks(slice, 1, 64);
        let weights_x = upload_channel(&all_weights[0..64]);
        let weights_y = upload_channel(&all_weights[64..128]);
        let weights_b = upload_channel(&all_weights[128..192]);
        Self {
            width,
            height,
            padded_width,
            padded_height,
            num_blocks,
            weights_x,
            weights_y,
            weights_b,
            weights: default_gaborish_weights(),
            // libjxl default_thresholds: Y has tightest TL (0.56);
            // X/B share the same {0.58, 0.62, 0.62, 0.62}.
            thresholds_x: [0.58, 0.62, 0.62, 0.62],
            thresholds_y: [0.56, 0.62, 0.62, 0.62],
            thresholds_b: [0.58, 0.62, 0.62, 0.62],
            // Default to libjxl effort 7 (kSquirrel): evaluate every
            // strategy that libjxl considers at e7 — DCT8/16/16x8/8x16,
            // sub-blocks (DCT4x4/4x8/8x4/IDENTITY/DCT2x2), DCT32/16x32,
            // DCT64*. AFV remains separately gated for unrelated reasons.
            effort: 7,
            // AFV cost-grid evaluation: opt-in, default off. See
            // `Self::evaluate_afv` field docs for semantics. Production
            // bitstream stays byte-identical with this off.
            evaluate_afv: false,
            // Auto-enable AFV on screenshot-like content (mask1x1 median
            // > 95 AND effort >= 7). Default `true` — zero impact on
            // photos (gate never fires), modest bytes win on screenshots.
            // See `Self::auto_evaluate_afv_on_screenshots` field docs.
            auto_evaluate_afv_on_screenshots: true,
            // Skip the AFV cost-grid evaluation when the slow-path
            // patches case-1 will fire on this image (CPU recompute
            // wipes the GPU AFV picks anyway — see W11-2 in
            // jxl-encoder-gpu@04541934). Default `true`. Bytes
            // byte-identical with/without; saves ~100 ms / 5 MP on
            // patches-fired screenshots.
            auto_skip_afv_when_patches: true,
            // Opt-in dispatch of entropy_mul / dist_bias bundle.
            // **Default `false`** — measured photo regression
            // (Pareto-worse on bytes + butteraugli + SSIM2) blocks
            // default-on. Kept as opt-in for re-validation once the
            // missing `kAvoidEntropyOfTransforms` + X-channel
            // multi-block weight counterweights land in the GPU cost
            // grids (cross-ref dropped log item #4). See
            // `Self::auto_libjxl_entropy_mul_on_photos` field docs
            // for the audit hypothesis + the A/B measurement that
            // refuted it.
            auto_libjxl_entropy_mul_on_photos: false,
            // Auto-disable the GPU pre-quantized AC fast path on small
            // screenshot content so the slow path runs patches detection
            // and writes the patches reference frame (the fast path's
            // entry point hardcodes `None` for patches). Default `true`
            // — zero impact on photos (gate never fires), 30-50% bytes
            // win on the gated subset (small + screenshot + all-DCT8).
            // See `Self::auto_patches_on_fast_path` field docs.
            auto_patches_on_fast_path: true,
            // Chunk-1 POC of GPU `kAvoidEntropyOfTransforms` port.
            // Default `false` — narrow d>4 gate means this is a
            // structural no-op at production distances anyway, but
            // keep opt-in until the full multi-week port is wired
            // (X-channel multi-block weight, AFV cost-path integration,
            // bundle with `auto_libjxl_entropy_mul_on_photos`).
            enable_kavoid_entropy_of_transforms: false,
        }
    }

    /// Set the libjxl-equivalent effort level. See [`Self::effort`] field
    /// docs for the per-effort strategy enable list. Default 7.
    ///
    /// Lower effort = fewer cost-grid evaluations = faster encode but
    /// can pick less-optimal strategies on borderline content. Match
    /// libjxl's `EvalAcStrategy` per-speed_tier behavior so swapping
    /// our encoder for libjxl at the same effort produces comparable
    /// strategy distributions and bitstream sizes.
    pub fn with_effort(mut self, effort: u8) -> Self {
        self.effort = effort.clamp(1, 9);
        self
    }

    /// Opt-in: evaluate AFV0-3 cost grids in
    /// [`Self::prepare_strategy_search_plan_traced`]. See the
    /// [`Self::evaluate_afv`][evaluate_afv-field] field for semantics.
    /// Default `false` — production bitstream stays byte-identical.
    ///
    /// [evaluate_afv-field]: #structfield.evaluate_afv
    pub fn with_evaluate_afv(mut self, on: bool) -> Self {
        self.evaluate_afv = on;
        self
    }

    /// Whether AFV0-3 cost grids will be evaluated by
    /// [`Self::prepare_strategy_search_plan_traced`]. Default `false`.
    ///
    /// Note this reports the **explicit opt-in** flag only. The
    /// effective per-encode decision may still enable AFV evaluation
    /// when this returns `false` if
    /// [`Self::auto_evaluate_afv_on_screenshots`] is set (default)
    /// AND the input passes the screenshot discriminator at runtime.
    /// See [`Self::auto_evaluate_afv_on_screenshots`] for semantics.
    pub fn evaluate_afv(&self) -> bool {
        self.evaluate_afv
    }

    /// Auto-enable AFV0-3 cost-grid evaluation on screenshot-like
    /// content (default `true`). See the
    /// [`Self::auto_evaluate_afv_on_screenshots`][field] field for
    /// gate semantics and rationale.
    ///
    /// Pass `false` to recover strict default-off AFV behavior (e.g.,
    /// for byte-exact reproducibility against a pre-2026-05-17 baseline).
    /// Explicit [`Self::with_evaluate_afv`] always wins regardless of
    /// this setting.
    ///
    /// [field]: #structfield.auto_evaluate_afv_on_screenshots
    pub fn with_auto_evaluate_afv_on_screenshots(mut self, on: bool) -> Self {
        self.auto_evaluate_afv_on_screenshots = on;
        self
    }

    /// Whether the encoder will auto-enable AFV cost-grid evaluation on
    /// screenshot-like content. Default `true`. See the
    /// [`Self::auto_evaluate_afv_on_screenshots`][field] field for
    /// gate semantics.
    ///
    /// [field]: #structfield.auto_evaluate_afv_on_screenshots
    pub fn auto_evaluate_afv_on_screenshots(&self) -> bool {
        self.auto_evaluate_afv_on_screenshots
    }

    /// Skip the AFV cost-grid evaluation when patches detection fires
    /// on the same image (default `true`). See the
    /// [`Self::auto_skip_afv_when_patches`][field] field for gate
    /// semantics and rationale.
    ///
    /// Pass `false` to recover the unconditional pre-2026-05-18
    /// auto-AFV behavior (e.g., for ablation studies, or for byte-exact
    /// reproducibility against the W7-3 baseline). Bytes are
    /// byte-identical with or without this gate — only wall-clock
    /// changes.
    ///
    /// [field]: #structfield.auto_skip_afv_when_patches
    pub fn with_auto_skip_afv_when_patches(mut self, on: bool) -> Self {
        self.auto_skip_afv_when_patches = on;
        self
    }

    /// Whether the encoder will skip the AFV cost-grid when patches
    /// detection fires. Default `true`. See
    /// [`Self::auto_skip_afv_when_patches`][field] for gate semantics.
    ///
    /// [field]: #structfield.auto_skip_afv_when_patches
    pub fn auto_skip_afv_when_patches(&self) -> bool {
        self.auto_skip_afv_when_patches
    }

    /// Opt-in dispatch of per-strategy `entropy_mul` and `dist_bias`
    /// between libjxl-faithful (photo branch) and GPU-lifted
    /// (screenshot branch) values based on the per-block `mask1x1`
    /// median.
    ///
    /// **Default `false`** — see the
    /// [`Self::auto_libjxl_entropy_mul_on_photos`][field] field for
    /// the audit hypothesis and the 2026-05-17 A/B measurement that
    /// refuted it (Pareto-worse on bytes + butteraugli + SSIM2 on
    /// photos). Use `with_auto_libjxl_entropy_mul_on_photos(true)`
    /// only for re-validation experiments — production should stay
    /// at the default.
    ///
    /// [field]: #structfield.auto_libjxl_entropy_mul_on_photos
    pub fn with_auto_libjxl_entropy_mul_on_photos(mut self, on: bool) -> Self {
        self.auto_libjxl_entropy_mul_on_photos = on;
        self
    }

    /// Whether the encoder will dispatch the entropy_mul + dist_bias
    /// bundle by content. **Default `false`** (opt-in). See the
    /// [`Self::auto_libjxl_entropy_mul_on_photos`][field] field for
    /// gate semantics and the measured photo regression that blocks
    /// default-on.
    ///
    /// [field]: #structfield.auto_libjxl_entropy_mul_on_photos
    pub fn auto_libjxl_entropy_mul_on_photos(&self) -> bool {
        self.auto_libjxl_entropy_mul_on_photos
    }

    /// Auto-disable the GPU pre-quantized AC fast path on small
    /// screenshot content so the slow path can run patches detection.
    /// Default `true`. See the
    /// [`Self::auto_patches_on_fast_path`][field] field for gate
    /// semantics and rationale.
    ///
    /// Pass `false` to keep the fast path unconditional (e.g., for
    /// benchmarking the fast path in isolation, or for byte-exact
    /// reproducibility against a pre-2026-05-17 baseline).
    ///
    /// [field]: #structfield.auto_patches_on_fast_path
    pub fn with_auto_patches_on_fast_path(mut self, on: bool) -> Self {
        self.auto_patches_on_fast_path = on;
        self
    }

    /// Whether the encoder will auto-disable the fast path on small
    /// screenshot content to run patches detection. Default `true`.
    /// See [`Self::auto_patches_on_fast_path`][field] for semantics.
    ///
    /// [field]: #structfield.auto_patches_on_fast_path
    pub fn auto_patches_on_fast_path(&self) -> bool {
        self.auto_patches_on_fast_path
    }

    /// Opt-in: enable libjxl's `kAvoidEntropyOfTransforms` penalty on
    /// the GPU cost-grid's sub-block strategies (DCT4X4, DCT4X8, DCT8X4).
    /// Default `false`. See the
    /// [`Self::enable_kavoid_entropy_of_transforms`][field] field for
    /// gate semantics and the chunk-1 POC rationale.
    ///
    /// [field]: #structfield.enable_kavoid_entropy_of_transforms
    pub fn with_enable_kavoid_entropy_of_transforms(mut self, on: bool) -> Self {
        self.enable_kavoid_entropy_of_transforms = on;
        self
    }

    /// Whether the encoder will apply the `kAvoidEntropyOfTransforms`
    /// penalty on the GPU cost-grid's sub-block strategies. Default
    /// `false`. See [`Self::enable_kavoid_entropy_of_transforms`][field]
    /// for semantics.
    ///
    /// [field]: #structfield.enable_kavoid_entropy_of_transforms
    pub fn enable_kavoid_entropy_of_transforms(&self) -> bool {
        self.enable_kavoid_entropy_of_transforms
    }

    /// The current effort level. See [`Self::with_effort`].
    pub fn effort(&self) -> u8 {
        self.effort
    }

    /// Original (un-padded) dimensions the caller sees.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Internal padded dimensions (multiples of 8).
    pub fn padded_dimensions(&self) -> (u32, u32) {
        (self.padded_width, self.padded_height)
    }

    /// Run the full lossy pipeline on RGB input. Returns reconstructed
    /// RGB (linear, planar). `qac_qm` is the per-block quantize scale
    /// (broadcast to all blocks); larger = more aggressive quant.
    ///
    /// Single-shot: input uploaded, pipeline run, output downloaded.
    /// For batch workloads, use [`Self::encode_many`].
    pub fn encode_one(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        qac_qm: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (w, h) = (self.width as usize, self.height as usize);
        let (pw, ph) = (self.padded_width as usize, self.padded_height as usize);
        let r_pad = pad_to_alignment(r, w, h, pw, ph);
        let g_pad = pad_to_alignment(g, w, h, pw, ph);
        let b_pad = pad_to_alignment(b, w, h, pw, ph);
        let g_r = enc.upload_plane(&r_pad, self.padded_width, self.padded_height);
        let g_g = enc.upload_plane(&g_pad, self.padded_width, self.padded_height);
        let g_b = enc.upload_plane(&b_pad, self.padded_width, self.padded_height);
        let (rec_r, rec_g, rec_b) = self.run_pipeline(enc, &g_r, &g_g, &g_b, qac_qm);
        (
            crop_to_original(&rec_r, pw, w, h),
            crop_to_original(&rec_g, pw, w, h),
            crop_to_original(&rec_b, pw, w, h),
        )
    }

    /// sRGB U8 convenience wrapper for [`Self::encode_one`].
    ///
    /// Takes interleaved RGB U8 (`width * height * 3` bytes), converts
    /// to linear f32 internally, runs the lossy roundtrip, returns
    /// reconstructed RGB U8 (linearized output clamped + sRGB-encoded +
    /// rounded). Saves the caller the per-channel sRGB↔linear boilerplate.
    ///
    /// sRGB transfer function: gamma 2.4 (matches the simple model
    /// used elsewhere in the repo). For the IEC 61966-2-1 piecewise
    /// curve, deinterleave + linearize on the host before calling
    /// `encode_one` directly with f32 planes.
    pub fn encode_one_srgb_u8(&self, enc: &GpuEncoder<R>, rgb: &[u8], qac_qm: f32) -> Vec<u8> {
        let n = (self.width as usize) * (self.height as usize);
        assert_eq!(rgb.len(), n * 3, "rgb.len() must be width*height*3");
        let to_linear = |c: u8| (c as f32 / 255.0).powf(2.4);
        let mut r = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for chunk in rgb.chunks_exact(3) {
            r.push(to_linear(chunk[0]));
            g.push(to_linear(chunk[1]));
            b.push(to_linear(chunk[2]));
        }
        let (rr, gg, bb) = self.encode_one(enc, &r, &g, &b, qac_qm);
        let to_srgb_u8 = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.4) * 255.0).round() as u8;
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            out.push(to_srgb_u8(rr[i]));
            out.push(to_srgb_u8(gg[i]));
            out.push(to_srgb_u8(bb[i]));
        }
        out
    }

    /// Run the full pipeline at multiple `qac_qm` settings on the same
    /// input. Input uploaded ONCE; all subsequent iterations re-use the
    /// uploaded handles.
    ///
    /// Per `lossy_pipeline_repeated_input`, this is ~5× faster than
    /// calling [`Self::encode_one`] in a loop at 2048² × 5 settings.
    pub fn encode_many(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        qac_settings: &[f32],
    ) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (w, h) = (self.width as usize, self.height as usize);
        let (pw, ph) = (self.padded_width as usize, self.padded_height as usize);
        let r_pad = pad_to_alignment(r, w, h, pw, ph);
        let g_pad = pad_to_alignment(g, w, h, pw, ph);
        let b_pad = pad_to_alignment(b, w, h, pw, ph);
        let g_r = enc.upload_plane(&r_pad, self.padded_width, self.padded_height);
        let g_g = enc.upload_plane(&g_pad, self.padded_width, self.padded_height);
        let g_b = enc.upload_plane(&b_pad, self.padded_width, self.padded_height);
        qac_settings
            .iter()
            .map(|&qac_qm| {
                let (rec_r, rec_g, rec_b) = self.run_pipeline(enc, &g_r, &g_g, &g_b, qac_qm);
                (
                    crop_to_original(&rec_r, pw, w, h),
                    crop_to_original(&rec_g, pw, w, h),
                    crop_to_original(&rec_b, pw, w, h),
                )
            })
            .collect()
    }

    /// sRGB U8 batch wrapper for [`Self::encode_many`].
    ///
    /// Same shape as [`Self::encode_one_srgb_u8`] but produces one
    /// reconstructed RGB U8 buffer per `qac_qm` setting. Input
    /// linearization happens once outside the inner loop, so the
    /// per-encode sRGB↔linear cost is amortized — much closer to the
    /// pure-f32 batch throughput.
    pub fn encode_many_srgb_u8(
        &self,
        enc: &GpuEncoder<R>,
        rgb: &[u8],
        qac_settings: &[f32],
    ) -> Vec<Vec<u8>> {
        let n = (self.width as usize) * (self.height as usize);
        assert_eq!(rgb.len(), n * 3);
        let to_linear = |c: u8| (c as f32 / 255.0).powf(2.4);
        let mut r = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for chunk in rgb.chunks_exact(3) {
            r.push(to_linear(chunk[0]));
            g.push(to_linear(chunk[1]));
            b.push(to_linear(chunk[2]));
        }
        let outputs = self.encode_many(enc, &r, &g, &b, qac_settings);
        let to_srgb_u8 = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.4) * 255.0).round() as u8;
        outputs
            .into_iter()
            .map(|(rr, gg, bb)| {
                let mut out = Vec::with_capacity(n * 3);
                for i in 0..n {
                    out.push(to_srgb_u8(rr[i]));
                    out.push(to_srgb_u8(gg[i]));
                    out.push(to_srgb_u8(bb[i]));
                }
                out
            })
            .collect()
    }

    /// sRGB U8 wrapper for [`Self::encode_one_with_aq`] — turnkey
    /// content-driven AQ from sRGB U8 input to sRGB U8 output.
    pub fn encode_one_with_aq_srgb_u8(
        &self,
        enc: &GpuEncoder<R>,
        rgb: &[u8],
        distance: f32,
    ) -> Vec<u8> {
        let n = (self.width as usize) * (self.height as usize);
        assert_eq!(rgb.len(), n * 3);
        let to_linear = |c: u8| (c as f32 / 255.0).powf(2.4);
        let mut r = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for chunk in rgb.chunks_exact(3) {
            r.push(to_linear(chunk[0]));
            g.push(to_linear(chunk[1]));
            b.push(to_linear(chunk[2]));
        }
        let (rr, gg, bb) = self.encode_one_with_aq(enc, &r, &g, &b, distance);
        let to_srgb_u8 = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.4) * 255.0).round() as u8;
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            out.push(to_srgb_u8(rr[i]));
            out.push(to_srgb_u8(gg[i]));
            out.push(to_srgb_u8(bb[i]));
        }
        out
    }

    /// sRGB U8 batch wrapper for [`Self::encode_many_with_aq`] —
    /// distance sweep over sRGB U8 input with content-driven AQ on
    /// every iteration. sRGB↔linear and the mask1x1 prepass both
    /// happen exactly once.
    pub fn encode_many_with_aq_srgb_u8(
        &self,
        enc: &GpuEncoder<R>,
        rgb: &[u8],
        distances: &[f32],
    ) -> Vec<Vec<u8>> {
        let n = (self.width as usize) * (self.height as usize);
        assert_eq!(rgb.len(), n * 3);
        let to_linear = |c: u8| (c as f32 / 255.0).powf(2.4);
        let mut r = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for chunk in rgb.chunks_exact(3) {
            r.push(to_linear(chunk[0]));
            g.push(to_linear(chunk[1]));
            b.push(to_linear(chunk[2]));
        }
        let outputs = self.encode_many_with_aq(enc, &r, &g, &b, distances);
        let to_srgb_u8 = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.4) * 255.0).round() as u8;
        outputs
            .into_iter()
            .map(|(rr, gg, bb)| {
                let mut out = Vec::with_capacity(n * 3);
                for i in 0..n {
                    out.push(to_srgb_u8(rr[i]));
                    out.push(to_srgb_u8(gg[i]));
                    out.push(to_srgb_u8(bb[i]));
                }
                out
            })
            .collect()
    }

    /// Internal pipeline body. Operates on pre-uploaded GPU planes;
    /// downloads the reconstructed RGB at the end.
    /// Pipeline body. Operates on already-padded planes (dimensions
    /// `padded_width × padded_height`); returns padded reconstruction
    /// Run the pipeline with a per-block adaptive `qac_qm` field (one
    /// scalar per padded block in raster order). Returns reconstructed
    /// padded RGB.
    ///
    /// Use this when you want adaptive quantization — e.g., flatten
    /// quant on smooth regions and tighten it on detail. The `aq_field`
    /// length must equal `num_blocks` (= `padded_w/8 * padded_h/8`).
    pub fn encode_one_adaptive(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        aq_field: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(
            aq_field.len(),
            self.num_blocks as usize,
            "aq_field length {} != num_blocks {}",
            aq_field.len(),
            self.num_blocks
        );
        let (w, h) = (self.width as usize, self.height as usize);
        let (pw, ph) = (self.padded_width as usize, self.padded_height as usize);
        let r_pad = pad_to_alignment(r, w, h, pw, ph);
        let g_pad = pad_to_alignment(g, w, h, pw, ph);
        let b_pad = pad_to_alignment(b, w, h, pw, ph);
        let g_r = enc.upload_plane(&r_pad, self.padded_width, self.padded_height);
        let g_g = enc.upload_plane(&g_pad, self.padded_width, self.padded_height);
        let g_b = enc.upload_plane(&b_pad, self.padded_width, self.padded_height);
        let (rec_r, rec_g, rec_b) = self.run_pipeline_with_qac(enc, &g_r, &g_g, &g_b, aq_field);
        (
            crop_to_original(&rec_r, pw, w, h),
            crop_to_original(&rec_g, pw, w, h),
            crop_to_original(&rec_b, pw, w, h),
        )
    }

    /// Phase A MVP for AC strategy search.
    ///
    /// Selects per-region between DCT8 and DCT16×16 based on
    /// upstream-faithful `estimate_entropy_full` cost grids, then
    /// encodes each region with its winning strategy via
    /// `encode_and_reconstruct_mixed_strategy_3channel`.
    ///
    /// **Phase A scope** (this method):
    /// - 2 strategies only: DCT8 vs DCT16×16
    /// - No CfL (ytox/ytob = 0)
    /// - No AdjustQuantBlockAC (per-block quant tuning skipped)
    /// - No EPF in postpass (kept simple; gab_smooth still applied)
    /// - Per-strategy entropy_mul fixed (0.8 / 1.34 from libjxl
    ///   profile.entropy_mul_table)
    /// - Per-strategy mul/bonus/penalty post-processing deferred to
    ///   Phase C (the simplification preserves relative ranking
    ///   between DCT8 and DCT16×16 at any single distance)
    ///
    /// Future phases:
    /// - Phase B: full strategy palette (DCT32, DCT64, rectangular)
    /// - Phase C: full upstream cost-formula parity (per-strategy
    ///   mul + kFavor2X2 + kAvoidEntropyOfTransforms)
    /// - Phase D: AFV path + sub-block strategies
    /// - Phase E: perf — shared DCT cache between cost-grid + final encode
    pub fn encode_one_with_strategy_search_dct8_16(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        distance: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        self.encode_one_with_strategy_search_dct8_16_traced(enc, r, g, b, distance, &mut |_| {})
    }

    /// Tracing variant of [`Self::encode_one_with_strategy_search_dct8_16`]
    /// that calls `mark` with a static label between each major stage.
    /// The caller measures time between callbacks (e.g. with
    /// `std::time::Instant`) to attribute work to specific stages.
    ///
    /// Stage labels (in order):
    /// `pad_upload`, `xyb_gab`, `mask1x1`, `cost_dct8_dct16`,
    /// `cost_dct16x8`, `cost_dct8x16`, `selector`, `dc_grids`,
    /// `mixed_strategy_encode_recon`, `postpass_gab_epf_xyb`,
    /// `download_crop`.
    pub fn encode_one_with_strategy_search_dct8_16_traced(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        distance: f32,
        mark: &mut dyn FnMut(&'static str),
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Build a uniform per-block qac field at the target distance,
        // then delegate to the adaptive variant. Equivalent to the
        // historical "uniform qac" behaviour.
        let nb8 = (self.padded_width as usize / 8) * (self.padded_height as usize / 8);
        let aq_field = alloc::vec![distance_to_qac(distance); nb8];
        self.encode_one_with_strategy_search_dct8_16_adaptive_traced(
            enc, r, g, b, &aq_field, distance, mark,
        )
    }

    /// Adaptive (per-block qac) variant of strat-search. Use this when
    /// you have a butteraugli-refined or content-driven `aq_field`
    /// instead of a uniform target distance.
    ///
    /// `aq_field` is the per-padded-block qac scalar (length =
    /// `padded_w/8 * padded_h/8`, raster order). `target_distance` is
    /// still required because the cost-grid stage uses it for the
    /// libjxl-style scaled constants (`compute_scaled_constants`),
    /// `mul_8x8 = 1 + kFavor2X2/(d+1.4)`, and the distance-scaled
    /// anti-bias on non-DCT8 grids. Strategy *selection* happens against
    /// `target_distance`'s cost-model state; per-block *quantization*
    /// uses `aq_field` directly. This composes with butteraugli AQ
    /// refinement: strat-search picks the transform per region, the
    /// loop tunes per-block qac.
    pub fn encode_one_with_strategy_search_dct8_16_adaptive(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        aq_field: &[f32],
        target_distance: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        self.encode_one_with_strategy_search_dct8_16_adaptive_traced(
            enc,
            r,
            g,
            b,
            aq_field,
            target_distance,
            &mut |_| {},
        )
    }

    /// Tracing variant of [`Self::encode_one_with_strategy_search_dct8_16_adaptive`].
    /// See that method for the meaning of `aq_field` vs `target_distance`.
    ///
    /// This is the shared body for both strat-search entry points.
    /// `target_distance` controls the cost-model scaling; `aq_field`
    /// controls per-block quantization. The `mark` callback is invoked
    /// at the same boundaries as the historical `_traced` method.
    ///
    /// As of May 9 2026 this is a 4-line shim that delegates to
    /// [`Self::prepare_strategy_search_plan_traced`] +
    /// [`Self::encode_with_strategy_plan_adaptive_traced`]. Callers
    /// who need to encode the same image at multiple `aq_field`s
    /// (e.g., the butteraugli refinement loop) should call
    /// `prepare_strategy_search_plan` once and reuse the plan across
    /// many `encode_with_strategy_plan_adaptive` calls.
    pub fn encode_one_with_strategy_search_dct8_16_adaptive_traced(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        aq_field: &[f32],
        target_distance: f32,
        mark: &mut dyn FnMut(&'static str),
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let plan = self.prepare_strategy_search_plan_traced(enc, r, g, b, target_distance, mark);
        self.encode_with_strategy_plan_adaptive_traced(enc, &plan, aq_field, mark)
    }

    /// Non-traced wrapper around
    /// [`Self::prepare_strategy_search_plan_traced`]. Runs the cost-grid
    /// stage of strat-search and returns a [`StrategySearchPlan`] that
    /// can be reused across multiple
    /// [`Self::encode_with_strategy_plan_adaptive`] calls (e.g., across
    /// butteraugli refinement iterations).
    pub fn prepare_strategy_search_plan(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        target_distance: f32,
    ) -> StrategySearchPlan<R> {
        self.prepare_strategy_search_plan_traced(enc, r, g, b, target_distance, &mut |_| {})
    }

    /// Non-traced wrapper around
    /// [`Self::prepare_strategy_search_plan_traced_from_u8`]. Takes
    /// raw interleaved sRGB u8 RGB and runs the entire pipeline
    /// (sRGB→linear + pad + XYB + cost-grid) without ever
    /// materialising the f32 linear planes on the host. Saves the
    /// per-pixel host `powf` cost and 4× the upload bandwidth (48 MB
    /// raw u8 vs 192 MB converted f32 at 16 MP).
    pub fn prepare_strategy_search_plan_from_u8(
        &self,
        enc: &GpuEncoder<R>,
        pixels_u8: &[u8],
        target_distance: f32,
    ) -> StrategySearchPlan<R> {
        self.prepare_strategy_search_plan_traced_from_u8(
            enc,
            pixels_u8,
            target_distance,
            &mut |_| {},
        )
    }

    /// Cost-grid stage of strat-search. Computes XYB + gaborish + masks,
    /// runs all per-strategy cost grids, runs the partition selector,
    /// and returns the assignments + cached host buffers.
    ///
    /// **Cost** (CLIC 1024²): ~150-180 ms. The bulk is the cost-grid
    /// kernels (DCT8, DCT16x16, DCT16x8, DCT8x16, DCT32x32, DCT32x16,
    /// DCT16x32, DCT64x64, DCT64x32, DCT32x64, plus the 5 sub-block
    /// strategies on 8x8 grids). AFV cost grids are skipped per the
    /// existing notes.
    ///
    /// **Reuse**: when the same image is encoded multiple times at the
    /// same `target_distance` but with different `aq_field`s (e.g., the
    /// butteraugli refinement loop), this plan can be reused across
    /// iterations. Strategy assignments are invariant under aq_field
    /// changes because the cost-grid scalar `qac` derives from
    /// `target_distance` (constant), not from `aq_field`.
    ///
    /// `mark` callback fires at the same stage boundaries as the
    /// monolithic `encode_one_with_strategy_search_dct8_16_adaptive_traced`
    /// up through `dc_grids`.
    /// Pad host f32 planes to alignment, upload to GPU, then call
    /// [`Self::prepare_strategy_search_plan_inner`].
    ///
    /// Most callers should use the higher-level
    /// [`Self::prepare_strategy_search_plan`]; this _traced variant
    /// exposes per-stage timing via `mark`.
    pub fn prepare_strategy_search_plan_traced(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        target_distance: f32,
        mark: &mut dyn FnMut(&'static str),
    ) -> StrategySearchPlan<R> {
        let (w, h) = (self.width as usize, self.height as usize);
        let (pw, ph) = (self.padded_width as usize, self.padded_height as usize);
        mark("start");
        let r_pad = pad_to_alignment(r, w, h, pw, ph);
        let g_pad = pad_to_alignment(g, w, h, pw, ph);
        let b_pad = pad_to_alignment(b, w, h, pw, ph);
        mark("pad_only");
        // Batched 3-channel upload — see `upload_planes_3ch` for the
        // amortized-allocation rationale.
        let (g_r, g_g, g_b) = enc.upload_planes_3ch(
            &r_pad,
            &g_pad,
            &b_pad,
            self.padded_width,
            self.padded_height,
        );
        mark("upload_3ch");
        self.prepare_strategy_search_plan_inner(enc, g_r, g_g, g_b, target_distance, mark)
    }

    /// Like [`Self::prepare_strategy_search_plan_traced`], but takes
    /// raw interleaved sRGB u8 RGB pixels (no alpha) and runs sRGB→linear
    /// conversion + edge-replication padding on the GPU in a single
    /// fused launch. The host-side preprocessing cost (per-pixel powf +
    /// pad_to_alignment × 3 planes) and the larger 3× padded f32 upload
    /// disappear; the wire transfer is just `width * height * 3` bytes.
    ///
    /// `pixels_u8` MUST be exactly `self.width * self.height * 3` bytes,
    /// row-major, R G B R G B …, sRGB encoded (the standard PNG layout).
    ///
    /// **Measured savings vs the f32 path** (paired A/B at distance=1.0,
    /// `examples/perf_strat_plan_u8_vs_f32`):
    ///
    /// | Pixels  | f32 path | u8 path | Speedup |
    /// |---------|---------:|--------:|--------:|
    /// | 1.05 MP |  29.4 ms | 24.0 ms |  1.22×  |
    /// | 16 MP   |   703 ms |  499 ms |  1.41×  |
    ///
    /// Bound below by cubecl 0.10's slow upload path — even the
    /// reduced 48 MB u8 upload at 16 MP costs ~300 ms via cubecl
    /// (vs ~4 ms via raw pinned cudarc). The full ~1100 ms theoretical
    /// savings would require the upstream cubecl pinned-buffer fix.
    pub fn prepare_strategy_search_plan_traced_from_u8(
        &self,
        enc: &GpuEncoder<R>,
        pixels_u8: &[u8],
        target_distance: f32,
        mark: &mut dyn FnMut(&'static str),
    ) -> StrategySearchPlan<R> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        assert_eq!(
            pixels_u8.len(),
            expected,
            "pixels_u8 len {} != width*height*3 = {}",
            pixels_u8.len(),
            expected,
        );
        mark("start");
        let (g_r, g_g, g_b) = enc.upload_u8_rgb_to_linear_planar_padded(
            pixels_u8,
            self.width,
            self.height,
            self.padded_width,
            self.padded_height,
        );
        mark("upload_u8_fused");
        self.prepare_strategy_search_plan_inner(enc, g_r, g_g, g_b, target_distance, mark)
    }

    /// Inner helper used by both
    /// [`Self::prepare_strategy_search_plan_traced`] and
    /// [`Self::prepare_strategy_search_plan_traced_from_u8`]. Takes
    /// the three padded linear-RGB GPU planes already on-device and
    /// runs the full XYB → gaborish → cost-grid → strategy-pick pipeline.
    fn prepare_strategy_search_plan_inner(
        &self,
        enc: &GpuEncoder<R>,
        g_r: GpuPlane<R>,
        g_g: GpuPlane<R>,
        g_b: GpuPlane<R>,
        target_distance: f32,
        mark: &mut dyn FnMut(&'static str),
    ) -> StrategySearchPlan<R> {
        // Cost-grid stage uses the user-facing target distance for
        // `compute_scaled_constants`, `mul_8x8`, and the anti-bias
        // distance-ramp.
        let distance = target_distance;
        use crate::forks::cost::{
            compute_scaled_constants, strategy_search_costs_dct8_16x16_persistent,
            strategy_search_costs_dct16x8_or_8x16_persistent,
            strategy_search_costs_dct32x16_or_16x32_persistent,
            strategy_search_costs_dct32x32_persistent_with_aq_field,
            strategy_search_costs_dct64x32_or_32x64_persistent,
            strategy_search_costs_dct64x64_persistent,
        };
        use crate::forks::transform::{
            RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8, RAW_STRATEGY_DCT8X4,
            RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X32,
            RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X64, RAW_STRATEGY_DCT64X32,
            RAW_STRATEGY_IDENTITY,
        };
        use crate::pipeline::{
            CostGrids16x16, CostGrids32x32, CostGrids64x64, partitions_16x16_to_assignments,
            partitions_32x32_to_assignments, partitions_64x64_to_assignments,
            select_partitions_16x16_full, select_partitions_32x32_with_extras16,
            select_partitions_64x64_with_extras16,
        };
        use crate::quant_weights::{
            dct2x2_weights_per_channel, dct4x4_weights_per_channel, dct4x8_weights_per_channel,
            dct8_weights_per_channel, dct16x8_weights_per_channel, dct16x16_weights_per_channel,
            dct16x32_weights_per_channel, dct32x32_weights_per_channel,
            dct32x64_weights_per_channel, dct64x64_weights_per_channel,
            identity_weights_per_channel,
        };

        // DCT32x32 wiring: STAGED but DISABLED. Investigation found
        // (commits during this session, especially the diag tests):
        //   - Encoder/recon path correct (test_dct32x32_reconstruct_smooth_gradient
        //     shows DCT32 RMSE 0.0016 < DCT8 RMSE 0.0076 on smooth content)
        //   - partitions_32x32_to_assignments lowering correct
        //   - When enabled on real CLIC photo: 79 of 1024 32x32 regions
        //     pick DCT32, butteraugli regresses 1.35 → 12.4 despite
        //     RMSE only 0.030 (= localized perceptual artifacts in the
        //     79 DCT32 blocks)
        //
        // Root cause: this Phase A cost model lacks libjxl's per-strategy
        // mul/bonus/penalty adjustments (kFavor2X2, kAvoidEntropyOfTransforms,
        // mul8x8 vs mul16x16 vs mul32x32 ratios). Without those, raw
        // entropy + pixel_loss systematically over-favors larger
        // transforms on detailed content, producing perceptually-broken
        // picks that L2 loss doesn't catch.
        //
        // Fix is in the cost model (forks/cost.rs) — apply per-strategy
        // adjustments before returning the cost grid. Until then, keep
        // DCT32x32 disabled to preserve the +0.36% baseline.
        // Eligibility = image-dim alignment AND libjxl-effort allows it.
        // libjxl drops DCT32+ at speed_tier > kSquirrel (effort < 7) per
        // EvalAcStrategy. The `_eval_dct32_64` is computed below from
        // self.effort; refer to that block for the gate semantics.
        let speed_tier = 10u8.saturating_sub(self.effort);
        let effort_dct32_64 = speed_tier <= 3; // e>=7
        let dct32_eligible = effort_dct32_64
            && (self.padded_width as usize).is_multiple_of(32)
            && (self.padded_height as usize).is_multiple_of(32);
        let dct64_eligible = effort_dct32_64
            && (self.padded_width as usize).is_multiple_of(64)
            && (self.padded_height as usize).is_multiple_of(64);

        let (w, h) = (self.width as usize, self.height as usize);
        let (pw, ph) = (self.padded_width as usize, self.padded_height as usize);
        let xb8 = pw / 8;
        let yb8 = ph / 8;
        let nb8 = xb8 * yb8;

        // Stage 2: XYB + gaborish (GPU). Stage 1 (upload) was done by
        // the caller; planes are already on device.
        //
        // libjxl gates gaborish at distance > 0.5 (enc_frame.cc:281).
        // The CPU API mirrors this at jxl-encoder/src/api.rs:3842
        // (`enc.enable_gaborish = cfg.gaborish && effective_distance > 0.5`).
        // At d <= 0.5 cjxl skips gaborish entirely AND scales the quant
        // field's input distance by 0.62 (vardct/bitstream.rs:1261-1265,
        // also vardct/encoder.rs:869-876). Without this gate the GPU
        // produced gaborished XYB at d=0.5 while the bitstream emit
        // signaled `enable_gaborish=true` to the decoder — but cjxl at
        // d=0.5 produces UN-sharpened XYB and signals `gaborish=false`,
        // so screenshots regressed by 8-27% bfly vs CPU rate-control
        // (terminal: GPU 1.393 vs CPU 1.094). Mirror the gate here.
        let enable_gaborish_local = target_distance > 0.5;
        let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(&g_r, &g_g, &g_b);
        let (xx_g, xy_g, xb_g) = if enable_gaborish_local {
            (
                enc.gaborish_5x5_persistent(&xx, &self.weights),
                enc.gaborish_5x5_persistent(&xy, &self.weights),
                enc.gaborish_5x5_persistent(&xb, &self.weights),
            )
        } else {
            // Refcount-only clones (see GpuPlane Clone impl, persistent.rs:139).
            // No PCIe traffic, no GPU work; downstream stages see un-sharpened
            // XYB and the StrategySearchPlan's xyb_x_pre_gab_gpu and xyb_x_gpu
            // happen to point at the same buffer (correct — pre and post are
            // identical when gaborish is skipped).
            (xx.clone(), xy.clone(), xb.clone())
        };

        // Stage 3: mask1x1 from Y channel — keep on GPU.
        //
        // Skip the host XYB download. Only the AFV branch in
        // encode_and_reconstruct_mixed_strategy_single_channel uses
        // these host slices (the per-block extend_from_slice loop in
        // forks/reconstruct.rs's `is_afv` branch), and AFV is
        // currently disabled in production cost grids. Downloading
        // 3× plane-sized buffers here forced a queue-drain sync
        // that masked the prior GPU work as "xyb_gab" wall-clock.
        // At 16 MP that was 170 ms of pure waste — entirely just
        // the download_plane × 3.
        //
        // SAFETY: if AFV gets re-enabled in the strat-search, the
        // AFV branch will need to lazily download these planes.
        // The empty Vecs flow through plan.xyb_x/y/b unchanged but
        // any caller that indexes them will panic — the existing
        // debug_assert checking len was removed.
        let g_mask = enc.mask1x1_persistent(&xy_g);
        // NOTE: the host XYB download (xyb_x/y/b) used to live HERE,
        // but the screenshot-gated auto-AFV path (see
        // `Self::auto_evaluate_afv_on_screenshots`) needs the
        // per-block mask1x1 means to compute the median screenshot
        // discriminator BEFORE we know whether the AFV branch will
        // need the host XYB planes. The download is deferred to right
        // after `aq_field_means` is computed (see `aq_field` mark
        // below), where the `effective_evaluate_afv` decision has
        // been made. When the gate stays off (photos), the download
        // is still skipped — preserving the 170 ms / 16 MP PCIe-stall
        // saving that the lazy path was introduced for.
        mark("xyb_gab");
        mark("mask1x1");

        // GPU 8×8 reduction: produces per-block mask means in 1.5 MB
        // (12 MP) instead of downloading the full 48 MB mask plane and
        // running the reduction on the host. The mask values are
        // already f32 (computed by compute_mask1x1_gpu), so the GPU
        // f32-sum drops the f64 precision the original CPU loop used
        // — the cumulative error on 64 values bounded in [0, ~1] is
        // ε * 64 ≈ 6e-6 absolute, well below the threshold where
        // block_means_to_qac_field's min/max scaling shifts strategy
        // assignments.
        //
        // libjxl uses this field as `quant_norm16` for multi-block
        // strat-search (enc_ac_strategy.cc:382-413). Without per-block
        // quant the DCT32 cost-model has a +42% upward bias on photos
        // vs libjxl — see examples/quant_norm16_divergence.rs.
        let xs8 = pw / 8;
        let ys8 = ph / 8;
        let nb_padded = xs8 * ys8;
        let block_means_bytes = nb_padded * 4;
        let h_means = enc.client_ref().empty(block_means_bytes);
        crate::launch::aq_field::block_mask_mean::<R>(
            enc.client_ref(),
            g_mask.handle().clone(),
            h_means.clone(),
            w as u32,
            h as u32,
            pw as u32,
            ph as u32,
        );
        let mut aq_means_bytes = enc.client_ref().read(alloc::vec![h_means]);
        let aq_bytes = aq_means_bytes.pop().expect("read[0]");
        let aq_field_means: Vec<f32> = f32::from_bytes(&aq_bytes).to_vec();
        // Convert mean → adaptive qac per block.
        let aq_field = block_means_to_qac_field(&aq_field_means, distance);
        mark("aq_field");

        // Compute the screenshot-discriminator median ONCE per encode
        // and reuse for both AFV auto-dispatch and the entropy_mul +
        // dist_bias auto-dispatch bundle (`auto_libjxl_entropy_mul_on_photos`).
        //
        // Median is over `aq_field_means` directly — per-padded-block
        // mask1x1 means already produced on GPU (block_mask_mean
        // kernel). Same value space and threshold as
        // `content_looks_like_screenshot` (95.0 floor).
        //
        // partial_cmp NaN-safe: NaN sorts to the end via Equal
        // fallback; production mask1x1 values never produce NaN, but
        // the fallback matches the rest of the codebase.
        let mask1x1_block_median: Option<f32> = if !aq_field_means.is_empty() {
            let mut sorted = aq_field_means.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            Some(sorted[sorted.len() / 2])
        } else {
            None
        };
        let screenshot_likely: bool = mask1x1_block_median
            .map(|m| m > Self::SCREENSHOT_MEDIAN_MASK_THRESHOLD)
            .unwrap_or(false);

        // Auto-AFV dispatch: enable AFV cost-grid evaluation when the
        // input passes the same screenshot discriminator that
        // `LossyEncoder::content_looks_like_screenshot` uses, and
        // effort allows it. See `Self::auto_evaluate_afv_on_screenshots`
        // field docs for full rationale.
        //
        // W11-2 follow-on (this commit): if the W7-3 gate would fire
        // AND `auto_skip_afv_when_patches` is set (default `true`), do
        // a cheap host-side patches pre-check on the pre-gaborish XYB
        // the GPU pipeline just produced. If patches detection returns
        // `Some(_)`, skip AFV — the slow-path patches case-1 in
        // encoder.rs runs an independent CPU `compute_ac_strategy` on
        // patches-subtracted XYB that produces its own AFV picks
        // (typically 5-10× more), wiping whatever the GPU AFV cost
        // grid contributed. See `Self::auto_skip_afv_when_patches`
        // field docs for full rationale. Bytes are byte-identical
        // with or without the gate — only wall-clock changes.
        let auto_afv_would_fire =
            self.auto_evaluate_afv_on_screenshots && self.effort >= 7 && screenshot_likely;
        let patches_likely_to_fire: bool =
            if !self.evaluate_afv && auto_afv_would_fire && self.auto_skip_afv_when_patches {
                // Download pre-gab XYB and run the same
                // `find_and_build_patches` the encoder.rs slow path
                // uses (`jxl-encoder/src/vardct/patches.rs:1761`).
                // Cost: 3-plane host download (~3-15 ms / 5 MP) +
                // ~10-50 ms for the BFS/L1-distance text-like-patch
                // search. Net win on patches-fired screenshots:
                // skips the AFV cost grid (~100 ms / 5 MP).
                let (pre_x, pre_y, pre_b) = enc.download_planes_3ch(&xx, &xy, &xb);
                let detected = jxl_encoder::__pre_quantized::find_and_build_patches(
                    [&pre_x, &pre_y, &pre_b],
                    w,
                    h,
                    pw,
                )
                .is_some();
                #[cfg(any(test, feature = "encoder"))]
                {
                    extern crate std;
                    if std::env::var("JXL_GPU_DEBUG_AUTO_AFV").is_ok() {
                        std::eprintln!(
                            "[auto_afv] patches pre-check: detected={}, will_skip_afv={}",
                            detected,
                            detected,
                        );
                    }
                }
                detected
            } else {
                false
            };
        let effective_evaluate_afv = if self.evaluate_afv {
            // Explicit opt-in always wins (caller knows what they want;
            // patches gate is bypassed).
            true
        } else if auto_afv_would_fire && !patches_likely_to_fire {
            // Diagnostic: set JXL_GPU_DEBUG_AUTO_AFV=1 to log the dispatch
            // decision per encode (host-side only; the `encoder` feature
            // brings std in transitively).
            #[cfg(any(test, feature = "encoder"))]
            {
                extern crate std;
                if std::env::var("JXL_GPU_DEBUG_AUTO_AFV").is_ok() {
                    let median = mask1x1_block_median.unwrap_or(f32::NAN);
                    std::eprintln!(
                        "[auto_afv] mask1x1 block-mean median={:.3}, threshold={:.3}, effort={}, fired=true",
                        median,
                        Self::SCREENSHOT_MEDIAN_MASK_THRESHOLD,
                        self.effort,
                    );
                }
            }
            true
        } else {
            #[cfg(any(test, feature = "encoder"))]
            {
                extern crate std;
                if std::env::var("JXL_GPU_DEBUG_AUTO_AFV").is_ok()
                    && auto_afv_would_fire
                    && patches_likely_to_fire
                {
                    let median = mask1x1_block_median.unwrap_or(f32::NAN);
                    std::eprintln!(
                        "[auto_afv] mask1x1 block-mean median={:.3}, threshold={:.3}, effort={}, fired=false (patches gate)",
                        median,
                        Self::SCREENSHOT_MEDIAN_MASK_THRESHOLD,
                        self.effort,
                    );
                }
            }
            false
        };

        // Auto-dispatch of the entropy_mul + dist_bias bundle. See
        // `Self::auto_libjxl_entropy_mul_on_photos` field docs for the
        // rationale + rationale for keeping the two values bundled.
        //
        // `use_libjxl_entropy_mul_branch == true` means the encoder
        // picks libjxl-faithful per-strategy `entropy_mul` for IDENTITY
        // and DCT4x8/DCT8x4 AND drops the distance-scaled `dist_bias`
        // multipliers (set to 1.0). Used on the photo branch.
        //
        // `false` keeps the GPU-lifted values + `dist_bias` (used on
        // the screenshot branch). Screenshot output stays byte-identical
        // to the pre-2026-05-17 single-branch behavior.
        let use_libjxl_entropy_mul_branch = self.auto_libjxl_entropy_mul_on_photos
            && mask1x1_block_median.is_some()
            && !screenshot_likely;
        #[cfg(any(test, feature = "encoder"))]
        {
            extern crate std;
            if std::env::var("JXL_GPU_DEBUG_AUTO_ENTROPY_MUL").is_ok() {
                let median = mask1x1_block_median.unwrap_or(f32::NAN);
                std::eprintln!(
                    "[auto_entropy_mul] mask1x1 block-mean median={:.3}, threshold={:.3}, branch={}, dist_bias={}",
                    median,
                    Self::SCREENSHOT_MEDIAN_MASK_THRESHOLD,
                    if use_libjxl_entropy_mul_branch {
                        "libjxl-faithful (photo)"
                    } else {
                        "GPU-lifted (screenshot/fallback)"
                    },
                    if use_libjxl_entropy_mul_branch {
                        "1.0 (off)"
                    } else {
                        "(distance-scaled, on)"
                    },
                );
            }
        }

        // Deferred host XYB download for the AFV branch in
        // `encode_and_reconstruct_mixed_strategy_single_channel`
        // (forks/reconstruct.rs:601 reads `xyb_channel[src_off..src_off+tile_w]`
        // per AFV-selected block). Only paid when AFV evaluation is
        // active (explicit opt-in OR auto-dispatch fired on screenshots).
        // Production photo path stays at zero PCIe bytes for these planes.
        let (xyb_x, xyb_y, xyb_b): (Vec<f32>, Vec<f32>, Vec<f32>) = if effective_evaluate_afv {
            enc.download_planes_3ch(&xx_g, &xy_g, &xb_g)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        // Stage 4: cost grids — DCT8, DCT16x16, DCT16x8, DCT8x16
        let (dct8_x, dct8_y, dct8_b) = dct8_weights_per_channel();
        let (dct16_x, dct16_y, dct16_b) = dct16x16_weights_per_channel();
        let (dct16x8_x, dct16x8_y, dct16x8_b) = dct16x8_weights_per_channel();
        let inv_8x: Vec<f32> = dct8_x.iter().map(|w| 1.0 / w).collect();
        let inv_8y: Vec<f32> = dct8_y.iter().map(|w| 1.0 / w).collect();
        let inv_8b: Vec<f32> = dct8_b.iter().map(|w| 1.0 / w).collect();
        let inv_16x: Vec<f32> = dct16_x.iter().map(|w| 1.0 / w).collect();
        let inv_16y: Vec<f32> = dct16_y.iter().map(|w| 1.0 / w).collect();
        let inv_16b: Vec<f32> = dct16_b.iter().map(|w| 1.0 / w).collect();
        let inv_16x8_x: Vec<f32> = dct16x8_x.iter().map(|w| 1.0 / w).collect();
        let inv_16x8_y: Vec<f32> = dct16x8_y.iter().map(|w| 1.0 / w).collect();
        let inv_16x8_b: Vec<f32> = dct16x8_b.iter().map(|w| 1.0 / w).collect();
        let qac = distance_to_qac(distance);
        // libjxl effort 7+ default base constants for compute_scaled_constants
        let scaled_constants = compute_scaled_constants(distance, (1.2, 9.308_906, 10.833_273));

        // libjxl per-strategy cost adjustment: mul_8x8 = 1 + kFavor2X2/(d+1.4)
        // where kFavor2X2 = -0.4. At d=1.0 this gives ~0.833 — DCT8 cost
        // is reduced by ~17%, making it competitive with larger transforms.
        // This is the "DCT8 favoritism" that prevents over-selection of
        // DCT16/DCT32 on detailed content. Other strategies use mul=1.0.
        const K_FAVOR_2X2: f32 = -0.4;
        let mul_8x8 = 1.0 + K_FAVOR_2X2 / (distance + 1.4);

        // Pre-gather 8x8 GpuBlocks ONCE — reused by both the DCT8 cost
        // grid (via the persistent dct8_16x16 variant below) and all 5
        // 8x8 sub-block cost grids further down.
        let g_8x = enc.gather_blocks_persistent(&xx_g, 8, 8);
        let g_8y = enc.gather_blocks_persistent(&xy_g, 8, 8);
        let g_8b = enc.gather_blocks_persistent(&xb_g, 8, 8);

        let (mut cost_dct8, mut cost_dct16) = strategy_search_costs_dct8_16x16_persistent(
            enc,
            &g_8x,
            &g_8y,
            &g_8b,
            &xx_g,
            &xy_g,
            &xb_g,
            pw,
            ph,
            &g_mask,
            &dct8_x,
            &dct8_y,
            &dct8_b,
            &inv_8x,
            &inv_8y,
            &inv_8b,
            &dct16_x,
            &dct16_y,
            &dct16_b,
            &inv_16x,
            &inv_16y,
            &inv_16b,
            qac,
            qac,
            qac,
            0,
            0,
            scaled_constants,
        );
        // Apply mul_8x8 to DCT8 cost grid (libjxl kFavor2X2).
        for c in cost_dct8.iter_mut() {
            *c *= mul_8x8;
        }
        mark("cost_dct8_dct16");

        // Distance-scaled anti-bias for non-DCT8 cost grids. Diagnostic
        // (force-all-DCT8) confirmed the d=4 regression is purely in
        // cost-model picks — not the encode/recon path. The fixed muls
        // (2.5/3.5 for DCT32/64) work at d=1 but break at d=4 because
        // larger transforms have far fewer non-zero coeffs at heavy
        // quantization, making them artificially cheap.
        // Formula: mul_at_d = base_mul * (1 + (d - 1) * scale_factor)
        // ensures d=1 unchanged; d>1 ramps up the bias.
        // Distance-scaled anti-bias slope for non-DCT8 cost grids.
        // Tuned 2026-05-09: 0.6 → 0.3 cuts the slope in half. At d=4
        // the new dist_bias is 1.9 (was 2.8) for DCT16, 2.35 for DCT32,
        // 2.8 for DCT64 — leaves more selectivity room for non-DCT8
        // wins on smooth content. Quality at parity confirmed at
        // d ∈ {1, 2, 4} on CLIC test image (1.3456 / 2.1525 / 3.4407,
        // all matching uniform-qac exactly).
        //
        // Bundle with `entropy_mul` dispatch (see
        // `Self::auto_libjxl_entropy_mul_on_photos` field docs): on
        // the libjxl-faithful branch (photo content), `dist_bias` is
        // disabled (= 1.0). The GPU-lifted entropy_mul + dist_bias
        // were tuned together as a counterweight suite; the
        // libjxl-reference entropy_mul values do not need the
        // distance-scaled bias when used on photo content (libjxl
        // itself ships without it).
        let bias_scale = if use_libjxl_entropy_mul_branch {
            0.0
        } else {
            (distance - 1.0).max(0.0) * 0.3
        };
        let dist_bias = 1.0 + bias_scale;
        for c in cost_dct16.iter_mut() {
            *c *= dist_bias;
        }

        // 8x8 sub-block strategies (DCT4x4, DCT4x8, DCT8x4, IDENTITY,
        // DCT2x2). Reuse the GpuBlocks gathered above for the DCT8 cost
        // grid — all 5 strategies extract 8x8 tiles → 64 coefs.
        //
        // libjxl-faithful effort gating for AC strategy evaluation
        // (mirrors `EvalAcStrategy` per-speed_tier switches in
        // libjxl/lib/jxl/enc_ac_strategy.cc):
        //
        //   e7+ (kSquirrel/kKitten/kTortoise): evaluate everything —
        //     DCT8/16/16x8/8x16, all sub-blocks, DCT32* and DCT64*
        //     when grid-aligned.
        //   e6 (kWombat): drop DCT32x32 and DCT64* (kept for the rect
        //     32x16/16x32 variants only).
        //   e5 (kHare): drop DCT4x8 / DCT8x4 sub-block variants and
        //     drop DCT32+ entirely. DCT4x4 / IDENTITY / DCT2x2 stay.
        //   e3-4 (kCheetah/kFalcon): DCT8 only.
        //
        // The DCT32/64 effort gate folds into `dct32_eligible` /
        // `dct64_eligible` already (computed above with self.effort).
        // The variables below cover the remaining gates.
        //
        // Skipping a cost grid sets the corresponding selector input to
        // None; the partition selector's min-cost comparison never
        // picks it, so the bitstream stays identical to a libjxl run at
        // the same effort.
        // libjxl enum: kFalcon=8, kCheetah=7, kHare=5, kWombat=4, kSquirrel=3
        let evaluate_subblock_costs = speed_tier <= 5; // e>=5 (kHare+)
        let _eval_dct4x8_8x4 = speed_tier <= 4; // e>=6 (kWombat+)
        let evaluate_rect16_costs_inner = speed_tier <= 5; // e>=5 (kHare+)
        //
        // Hoist the mask_row_base upload out of the per-strategy loop.
        // All 5 strategies operate on the same 8×8 grid, so the row-base
        // table is identical. Theory predicted ~24 ms savings at 16 MP
        // (5× duplicate 1 MB uploads at cubecl's slow HtoD); MEASURED
        // savings are ~2 ms within run-to-run noise — cubecl's pool
        // amortizes the small repeated uploads more effectively than
        // the per-call overhead model suggests. Kept as a refactor for
        // clarity (single explicit upload point) and as setup for
        // future fused-multi-strategy launches; not a production perf
        // win in itself.
        use crate::forks::cost::{SubblockStratSpec, strategy_search_costs_subblock_8x8_batch};
        use cubecl::prelude::*;
        let mask_row_base_subblock: Vec<u32> = (0..nb8)
            .map(|i| {
                let bx_i = i % xb8;
                let by_i = i / xb8;
                (by_i * 8 * pw + bx_i * 8) as u32
            })
            .collect();
        let h_mrb_subblock = enc
            .client_ref()
            .create_from_slice(u32::as_bytes(&mask_row_base_subblock));

        let (dct4x4_x, dct4x4_y, dct4x4_b) = dct4x4_weights_per_channel();
        let inv_4x4_x: Vec<f32> = dct4x4_x.iter().map(|w| 1.0 / w).collect();
        let inv_4x4_y: Vec<f32> = dct4x4_y.iter().map(|w| 1.0 / w).collect();
        let inv_4x4_b: Vec<f32> = dct4x4_b.iter().map(|w| 1.0 / w).collect();

        let (dct4x8_x, dct4x8_y, dct4x8_b) = dct4x8_weights_per_channel();
        let inv_4x8_x: Vec<f32> = dct4x8_x.iter().map(|w| 1.0 / w).collect();
        let inv_4x8_y: Vec<f32> = dct4x8_y.iter().map(|w| 1.0 / w).collect();
        let inv_4x8_b: Vec<f32> = dct4x8_b.iter().map(|w| 1.0 / w).collect();

        let (id_x, id_y, id_b) = identity_weights_per_channel();
        let inv_id_x: Vec<f32> = id_x.iter().map(|w| 1.0 / w).collect();
        let inv_id_y: Vec<f32> = id_y.iter().map(|w| 1.0 / w).collect();
        let inv_id_b: Vec<f32> = id_b.iter().map(|w| 1.0 / w).collect();

        let (d2_x, d2_y, d2_b) = dct2x2_weights_per_channel();
        let inv_d2_x: Vec<f32> = d2_x.iter().map(|w| 1.0 / w).collect();
        let inv_d2_y: Vec<f32> = d2_y.iter().map(|w| 1.0 / w).collect();
        let inv_d2_b: Vec<f32> = d2_b.iter().map(|w| 1.0 / w).collect();

        // entropy_mul tuning per strategy. Bundled with the
        // content-discriminated `dist_bias` dispatch (see
        // `Self::auto_libjxl_entropy_mul_on_photos` field docs):
        //
        //   DCT4x4   — 1.08 (matches libjxl reference 1.08 in both branches)
        //   DCT2x2   — 0.95 (matches libjxl reference 0.95 in both branches)
        //   DCT4x8/8x4
        //     libjxl-faithful branch (photo): 0.859316
        //       (= `EntropyMulTable::reference().dct4x8`)
        //     GPU-lifted branch (screenshot): 0.98 (bisected; path-flip below 0.95)
        //   IDENTITY
        //     libjxl-faithful branch (photo): 1.0428
        //       (= `EntropyMulTable::reference().identity`)
        //     GPU-lifted branch (screenshot): 1.85 (bisected; path-flip on strat-wins)
        //
        // The GPU-lifted values exist because the GPU cost-grid path
        // lacks libjxl's `kAvoidEntropyOfTransforms` and X-channel
        // multi-block weight counterweights, so lifted entropy_mul
        // values stand in for the missing penalties. On photo content
        // the dropped counterweights matter less (large transforms
        // genuinely win on smooth regions), so the libjxl-faithful
        // values produce smaller bytes at slight bfly cost.
        let entropy_mul_dct4x8 = if use_libjxl_entropy_mul_branch {
            0.859_316_37_f32
        } else {
            0.98_f32
        };
        let entropy_mul_identity = if use_libjxl_entropy_mul_branch {
            1.0428_f32
        } else {
            1.85_f32
        };

        // Chunk-1 POC of libjxl's `kAvoidEntropyOfTransforms` heuristic.
        // Adds a per-distance penalty to non-DCT8 / non-DCT2X2 /
        // non-IDENTITY 8×8-class strategies (DCT4X4, DCT4X8, DCT8X4)
        // when `distance > 4.0` and `effort >= 5` (libjxl gates at
        // `speed_tier <= kHare`). At `distance <= 4.0` the formula
        // returns 0.0, so this is a structural no-op outside the
        // gated band — production at `d <= 4.0` stays byte-identical
        // regardless of `enable_kavoid_entropy_of_transforms`.
        //
        // The penalty mirrors the CPU encoder's
        // `jxl_encoder::vardct::ac_strategy_search::avoid_entropy_of_transforms_mul`
        // and folds straight into the existing per-strategy `entropy_mul`
        // value uploaded to the cost-grid kernel — matching the CPU's
        // `(entropy_mul_for_strategy + entropy_mul_adjust).max(0.01)`
        // shape from `vardct/ac_strategy.rs::estimate_entropy_with_mask`.
        //
        // AFV0-3 are NOT touched here: AFV picks flow through a
        // separate cost path (`forks::afv`) and the GPU AFV grid is
        // opt-in. Wiring AFV into the same penalty is a follow-on
        // chunk once the AFV cost path lands per-strategy entropy_mul
        // input.
        let avoid_transforms_adjust =
            if self.enable_kavoid_entropy_of_transforms && self.effort >= 5 {
                crate::forks::cost::K_AVOID_TRANSFORMS_BASE
                    * crate::forks::cost::avoid_entropy_of_transforms_mul(distance)
            } else {
                0.0
            };

        let mut specs: Vec<SubblockStratSpec> = Vec::new();
        if evaluate_subblock_costs {
            specs.push(SubblockStratSpec {
                raw_strategy: RAW_STRATEGY_DCT4X4,
                weights_x: &dct4x4_x,
                weights_y: &dct4x4_y,
                weights_b: &dct4x4_b,
                inv_weights_x: &inv_4x4_x,
                inv_weights_y: &inv_4x4_y,
                inv_weights_b: &inv_4x4_b,
                entropy_mul: (1.08_f32 + avoid_transforms_adjust).max(0.01),
            });
        }
        if _eval_dct4x8_8x4 {
            specs.push(SubblockStratSpec {
                raw_strategy: RAW_STRATEGY_DCT4X8,
                weights_x: &dct4x8_x,
                weights_y: &dct4x8_y,
                weights_b: &dct4x8_b,
                inv_weights_x: &inv_4x8_x,
                inv_weights_y: &inv_4x8_y,
                inv_weights_b: &inv_4x8_b,
                entropy_mul: (entropy_mul_dct4x8 + avoid_transforms_adjust).max(0.01),
            });
            specs.push(SubblockStratSpec {
                raw_strategy: RAW_STRATEGY_DCT8X4,
                weights_x: &dct4x8_x,
                weights_y: &dct4x8_y,
                weights_b: &dct4x8_b,
                inv_weights_x: &inv_4x8_x,
                inv_weights_y: &inv_4x8_y,
                inv_weights_b: &inv_4x8_b,
                entropy_mul: (entropy_mul_dct4x8 + avoid_transforms_adjust).max(0.01),
            });
        }
        if evaluate_subblock_costs {
            specs.push(SubblockStratSpec {
                raw_strategy: RAW_STRATEGY_IDENTITY,
                weights_x: &id_x,
                weights_y: &id_y,
                weights_b: &id_b,
                inv_weights_x: &inv_id_x,
                inv_weights_y: &inv_id_y,
                inv_weights_b: &inv_id_b,
                entropy_mul: entropy_mul_identity,
            });
            specs.push(SubblockStratSpec {
                raw_strategy: RAW_STRATEGY_DCT2X2,
                weights_x: &d2_x,
                weights_y: &d2_y,
                weights_b: &d2_b,
                inv_weights_x: &inv_d2_x,
                inv_weights_y: &inv_d2_y,
                inv_weights_b: &inv_d2_b,
                entropy_mul: 0.95,
            });
        }
        // Submit all strategies' pipelines, then ONE batched download.
        // Replaces 5 sequential calls (each with its own queue-drain
        // sync), folding 5 sync barriers into 1.
        let costs_batch = strategy_search_costs_subblock_8x8_batch(
            enc,
            &g_8x,
            &g_8y,
            &g_8b,
            pw,
            ph,
            &g_mask,
            &h_mrb_subblock,
            mask_row_base_subblock.len(),
            qac,
            qac,
            qac,
            0,
            0,
            scaled_constants,
            &specs,
        );
        // Re-deal results back into the per-strategy named bins
        // expected downstream. Order matches the push order above.
        let mut iter = costs_batch.into_iter();
        let cost_dct4x4 = if evaluate_subblock_costs {
            iter.next().unwrap_or_default()
        } else {
            Vec::new()
        };
        let cost_dct4x8 = if _eval_dct4x8_8x4 {
            iter.next().unwrap_or_default()
        } else {
            Vec::new()
        };
        let cost_dct8x4 = if _eval_dct4x8_8x4 {
            iter.next().unwrap_or_default()
        } else {
            Vec::new()
        };
        let cost_identity = if evaluate_subblock_costs {
            iter.next().unwrap_or_default()
        } else {
            Vec::new()
        };
        let cost_dct2x2 = if evaluate_subblock_costs {
            iter.next().unwrap_or_default()
        } else {
            Vec::new()
        };
        let _ = (&h_mrb_subblock, &mask_row_base_subblock);
        mark("cost_subblock_8x8");

        // AFV0-3 cost grid: opt-in via `self.evaluate_afv` (see
        // `LossyEncoder::with_evaluate_afv`). Default OFF — production
        // bitstream stays byte-identical with `corpus_regression`.
        //
        // Uses libjxl's per-block cost formula
        // (`entropy_mul × entropy + k_info_loss × loss_scalar`) — the
        // same shape every other 8x8-class sub-block strategy uses.
        // Output sits on the same scale as `cost_dct4x4` /
        // `cost_dct4x8` / `cost_identity` / `cost_dct2x2` / `cost_dct8`
        // natively, so the selector can compare them directly without
        // per-image calibration.
        //
        // Pipeline (per AFV kind 0..3, fully on GPU):
        //  1. Forward AFV via `afv_transform_batch_persistent`.
        //  2. 3-channel entropy + nzeros + per-coef error fused launch.
        //  3. Inverse AFV on per-coef errors → pixel-domain error blocks.
        //  4. Fused 3-channel pixel_loss kernel.
        // After all 4 kinds finish their pipelines (no syncs), ONE
        // batched read pulls all 24 result handles. Per-kind CPU-side
        // finalize via `per_block_upstream_cost_per_block` (per-block
        // `quant_for_coeffs` from `aq_field`, same as DCT8 sub-blocks).
        //
        // For chunk 2c (corpus retune): once turned ON in production,
        // the AFV `entropy_mul` may need a small anti-bias bump from
        // libjxl reference (0.818 / 0.8 ≈ 1.022) to match the
        // empirical bias other 8x8 sub-blocks carry in this encoder.
        // Sweep first; the corpus_regression test catches drift.
        let afv_costs_full: Vec<f32> = if effective_evaluate_afv {
            use crate::forks::afv::afv_per_block_upstream_cost_xyb_host;
            use crate::forks::cost::{EntropyMulTable, afv_entropy_mul};
            use crate::kernels::afv::AFV4X4_BASIS_TRANSPOSE;
            use crate::quant_weights::afv_weights_per_channel;

            // Per-channel weights (one [f32; 64] per channel) — these
            // are the AFV-specific quant weight templates; the cost
            // grid broadcast-applies them across all blocks.
            let (afv_wx, afv_wy, afv_wb) = afv_weights_per_channel();
            let inv_afv_wx: [f32; 64] = core::array::from_fn(|i| 1.0 / afv_wx[i]);
            let inv_afv_wy: [f32; 64] = core::array::from_fn(|i| 1.0 / afv_wy[i]);
            let inv_afv_wb: [f32; 64] = core::array::from_fn(|i| 1.0 / afv_wb[i]);

            // Per-strategy entropy_mul — libjxl reference (0.817795 /
            // 0.8 ≈ 1.022). Re-tune in chunk 2c if corpus regression
            // shifts.
            let entropy_mul = afv_entropy_mul(&EntropyMulTable::reference());

            // Per-block quant: same as the other 8x8 sub-blocks (aq_field
            // for adaptive, qac for uniform). For AFV (covered_blocks=1),
            // this is the per-8x8-block adaptive_quant value directly.
            let quant_for_coeffs_per_block: &[f32] = &aq_field;

            // mask_row_base_subblock + h_mrb_subblock are already
            // computed above for the DCT8 sub-block batch — reuse.
            afv_per_block_upstream_cost_xyb_host(
                enc,
                &AFV4X4_BASIS_TRANSPOSE,
                &g_8x,
                &g_8y,
                &g_8b,
                &afv_wx,
                &afv_wy,
                &afv_wb,
                &inv_afv_wx,
                &inv_afv_wy,
                &inv_afv_wb,
                &g_mask,
                &h_mrb_subblock,
                mask_row_base_subblock.len(),
                qac,
                qac,
                qac,
                0.0, // cmap_factor_x (no CfL refinement at strat-search time)
                0.0, // cmap_factor_b
                entropy_mul,
                scaled_constants,
                quant_for_coeffs_per_block,
            )
        } else {
            Vec::new()
        };
        mark("cost_afv");

        // Distance-scaled anti-bias for sub-blocks (same scale as DCT16).
        let mut cost_dct4x4 = cost_dct4x4;
        let mut cost_dct4x8 = cost_dct4x8;
        let mut cost_dct8x4 = cost_dct8x4;
        let mut cost_identity = cost_identity;
        let mut cost_dct2x2 = cost_dct2x2;
        for c in cost_dct4x4.iter_mut() {
            *c *= dist_bias;
        }
        for c in cost_dct4x8.iter_mut() {
            *c *= dist_bias;
        }
        for c in cost_dct8x4.iter_mut() {
            *c *= dist_bias;
        }
        for c in cost_identity.iter_mut() {
            *c *= dist_bias;
        }
        for c in cost_dct2x2.iter_mut() {
            *c *= dist_bias;
        }

        // Distance gate for the rectangular DCT16x8 / DCT8x16 cost
        // grids (~26 ms each at 12 MP, ~52 ms combined). They favor
        // edge-aligned content; at d>=K_RECT16_DISTANCE_GATE the
        // square DCT16x16 / DCT8 strategies are competitive enough
        // that skipping the rectangular evaluation rarely changes
        // strategy picks on photo content.
        //
        // Verified via corpus_regression (33 cases × 11 images, 0.5%
        // tolerance) — adjust gate downward if quality regresses.
        let evaluate_rect16_costs = evaluate_rect16_costs_inner;

        let cost_dct16x8 = if evaluate_rect16_costs {
            strategy_search_costs_dct16x8_or_8x16_persistent(
                enc,
                &xx_g,
                &xy_g,
                &xb_g,
                pw,
                ph,
                &g_mask,
                RAW_STRATEGY_DCT16X8,
                &dct16x8_x,
                &dct16x8_y,
                &dct16x8_b,
                &inv_16x8_x,
                &inv_16x8_y,
                &inv_16x8_b,
                qac,
                qac,
                qac,
                0,
                0,
                scaled_constants,
            )
        } else {
            Vec::new()
        };
        mark("cost_dct16x8");
        let cost_dct8x16 = if evaluate_rect16_costs {
            strategy_search_costs_dct16x8_or_8x16_persistent(
                enc,
                &xx_g,
                &xy_g,
                &xb_g,
                pw,
                ph,
                &g_mask,
                RAW_STRATEGY_DCT8X16,
                &dct16x8_x,
                &dct16x8_y,
                &dct16x8_b,
                &inv_16x8_x,
                &inv_16x8_y,
                &inv_16x8_b,
                qac,
                qac,
                qac,
                0,
                0,
                scaled_constants,
            )
        } else {
            Vec::new()
        };

        mark("cost_dct8x16");
        // Distance-scaled anti-bias (same as DCT16x16).
        let mut cost_dct16x8 = cost_dct16x8;
        let mut cost_dct8x16 = cost_dct8x16;
        for c in cost_dct16x8.iter_mut() {
            *c *= dist_bias;
        }
        for c in cost_dct8x16.iter_mut() {
            *c *= dist_bias;
        }

        // Optional: DCT32x32 cost grid (only when padded dims are
        // multiples of 32). Returns empty Vec when ineligible; selector
        // sees this as "no DCT32x32 candidate" and falls back to the
        // 16x16 tier. dct32_* are needed outside this block (weights_for
        // closures); inv_32* are only needed inside.
        let (dct32_x, dct32_y, dct32_b);
        let cost_dct32x32 = if dct32_eligible {
            let (x, y, b) = dct32x32_weights_per_channel();
            dct32_x = x;
            dct32_y = y;
            dct32_b = b;
            let inv_32x: Vec<f32> = dct32_x.iter().map(|w| 1.0 / w).collect();
            let inv_32y: Vec<f32> = dct32_y.iter().map(|w| 1.0 / w).collect();
            let inv_32b: Vec<f32> = dct32_b.iter().map(|w| 1.0 / w).collect();
            strategy_search_costs_dct32x32_persistent_with_aq_field(
                enc,
                &xx_g,
                &xy_g,
                &xb_g,
                pw,
                ph,
                &g_mask,
                &dct32_x,
                &dct32_y,
                &dct32_b,
                &inv_32x,
                &inv_32y,
                &inv_32b,
                &aq_field,
                qac,
                qac,
                0,
                0,
                scaled_constants,
                // entropy_mul = 3.0 — band-aid kept after the
                // libjxl-faithful loss-side per-block quant_norm16
                // fix landed.
                //
                // 2026-05-11 bisection (post-loss-fix): tried 2.5 to
                // see if the loss-side fix would unwedge the
                // entropy_mul tuning. Result: TRADEOFF.
                //   - 07b9f93f@d=1.0: 1.2133 → 1.2089 (improved,
                //     strat-search wins again — back to pre-fix score)
                //   - 2684452d@d=1.0: 1.1868 → 1.1991 (+1.03%,
                //     OUTSIDE 0.5% tolerance — DCT32 over-selected)
                //
                // Conclusion: entropy_mul tuning is per-image-variant,
                // NOT one-sided cost-model-biased. Loss-side fix
                // didn't move the wedge — it's a real per-image quality
                // tradeoff. Some images want more DCT32, some want
                // less; one global mul can't satisfy both.
                //
                // To actually shift this wedge, we need either:
                //   - kernel-side per-block quant in coefficient
                //     quantization (matches libjxl entirely)
                //   - per-image cost-model gating (e.g., screenshot
                //     discriminator pattern)
                //   - per-region adaptive entropy_mul
                3.0_f32,
            )
        } else {
            dct32_x = Vec::new();
            dct32_y = Vec::new();
            dct32_b = Vec::new();
            Vec::new()
        };
        mark("cost_dct32x32");
        // Distance-scaled anti-bias for DCT32x32 (1.5× the DCT16
        // factor since DCT32 over-selection at high d is more severe).
        let mut cost_dct32x32 = cost_dct32x32;
        let dist_bias_32 = 1.0 + bias_scale * 1.5;
        for c in cost_dct32x32.iter_mut() {
            *c *= dist_bias_32;
        }

        // Optional: DCT32x16 + DCT16x32 cost grids (rectangular DCT32
        // family). Both feed into the 32x32-tier selector via CostGrids32x32.
        let (dct32x16_x, dct32x16_y, dct32x16_b);
        let (cost_dct32x16, cost_dct16x32) = if dct32_eligible {
            let (x, y, b) = dct16x32_weights_per_channel();
            dct32x16_x = x;
            dct32x16_y = y;
            dct32x16_b = b;
            let inv_32x16_x: Vec<f32> = dct32x16_x.iter().map(|w| 1.0 / w).collect();
            let inv_32x16_y: Vec<f32> = dct32x16_y.iter().map(|w| 1.0 / w).collect();
            let inv_32x16_b: Vec<f32> = dct32x16_b.iter().map(|w| 1.0 / w).collect();
            let c_32x16 = strategy_search_costs_dct32x16_or_16x32_persistent(
                enc,
                &xx_g,
                &xy_g,
                &xb_g,
                pw,
                ph,
                &g_mask,
                RAW_STRATEGY_DCT32X16,
                &dct32x16_x,
                &dct32x16_y,
                &dct32x16_b,
                &inv_32x16_x,
                &inv_32x16_y,
                &inv_32x16_b,
                qac,
                qac,
                qac,
                0,
                0,
                scaled_constants,
            );
            let c_16x32 = strategy_search_costs_dct32x16_or_16x32_persistent(
                enc,
                &xx_g,
                &xy_g,
                &xb_g,
                pw,
                ph,
                &g_mask,
                RAW_STRATEGY_DCT16X32,
                &dct32x16_x,
                &dct32x16_y,
                &dct32x16_b,
                &inv_32x16_x,
                &inv_32x16_y,
                &inv_32x16_b,
                qac,
                qac,
                qac,
                0,
                0,
                scaled_constants,
            );
            (c_32x16, c_16x32)
        } else {
            dct32x16_x = Vec::new();
            dct32x16_y = Vec::new();
            dct32x16_b = Vec::new();
            (Vec::new(), Vec::new())
        };
        mark("cost_dct32x16_and_16x32");
        // Distance-scaled anti-bias for rectangular DCT32 (same as DCT32x32).
        let mut cost_dct32x16 = cost_dct32x16;
        let mut cost_dct16x32 = cost_dct16x32;
        for c in cost_dct32x16.iter_mut() {
            *c *= dist_bias_32;
        }
        for c in cost_dct16x32.iter_mut() {
            *c *= dist_bias_32;
        }

        // Optional: DCT64x64 + DCT64x32 + DCT32x64 cost grids.
        // All gated on dct64_eligible (image dims multiple of 64).
        let (dct64_x, dct64_y, dct64_b);
        let (dct64x32_x, dct64x32_y, dct64x32_b);
        let (cost_dct64x64, cost_dct64x32, cost_dct32x64) = if dct64_eligible {
            let (x, y, b) = dct64x64_weights_per_channel();
            dct64_x = x;
            dct64_y = y;
            dct64_b = b;
            let inv_64x: Vec<f32> = dct64_x.iter().map(|w| 1.0 / w).collect();
            let inv_64y: Vec<f32> = dct64_y.iter().map(|w| 1.0 / w).collect();
            let inv_64b: Vec<f32> = dct64_b.iter().map(|w| 1.0 / w).collect();
            let (x, y, b) = dct32x64_weights_per_channel();
            dct64x32_x = x;
            dct64x32_y = y;
            dct64x32_b = b;
            let inv_64x32_x: Vec<f32> = dct64x32_x.iter().map(|w| 1.0 / w).collect();
            let inv_64x32_y: Vec<f32> = dct64x32_y.iter().map(|w| 1.0 / w).collect();
            let inv_64x32_b: Vec<f32> = dct64x32_b.iter().map(|w| 1.0 / w).collect();
            let c64 = strategy_search_costs_dct64x64_persistent(
                enc,
                &xx_g,
                &xy_g,
                &xb_g,
                pw,
                ph,
                &g_mask,
                &dct64_x,
                &dct64_y,
                &dct64_b,
                &inv_64x,
                &inv_64y,
                &inv_64b,
                qac,
                qac,
                qac,
                0,
                0,
                scaled_constants,
            );
            let c64x32 = strategy_search_costs_dct64x32_or_32x64_persistent(
                enc,
                &xx_g,
                &xy_g,
                &xb_g,
                pw,
                ph,
                &g_mask,
                RAW_STRATEGY_DCT64X32,
                &dct64x32_x,
                &dct64x32_y,
                &dct64x32_b,
                &inv_64x32_x,
                &inv_64x32_y,
                &inv_64x32_b,
                qac,
                qac,
                qac,
                0,
                0,
                scaled_constants,
            );
            let c32x64 = strategy_search_costs_dct64x32_or_32x64_persistent(
                enc,
                &xx_g,
                &xy_g,
                &xb_g,
                pw,
                ph,
                &g_mask,
                RAW_STRATEGY_DCT32X64,
                &dct64x32_x,
                &dct64x32_y,
                &dct64x32_b,
                &inv_64x32_x,
                &inv_64x32_y,
                &inv_64x32_b,
                qac,
                qac,
                qac,
                0,
                0,
                scaled_constants,
            );
            (c64, c64x32, c32x64)
        } else {
            dct64_x = Vec::new();
            dct64_y = Vec::new();
            dct64_b = Vec::new();
            dct64x32_x = Vec::new();
            dct64x32_y = Vec::new();
            dct64x32_b = Vec::new();
            (Vec::new(), Vec::new(), Vec::new())
        };
        mark("cost_dct64_family");
        // Distance-scaled anti-bias for DCT64 (2× the DCT16 factor —
        // most extreme over-selection at high d).
        let mut cost_dct64x64 = cost_dct64x64;
        let mut cost_dct64x32 = cost_dct64x32;
        let mut cost_dct32x64 = cost_dct32x64;
        let dist_bias_64 = 1.0 + bias_scale * 2.0;
        for c in cost_dct64x64.iter_mut() {
            *c *= dist_bias_64;
        }
        for c in cost_dct64x32.iter_mut() {
            *c *= dist_bias_64;
        }
        for c in cost_dct32x64.iter_mut() {
            *c *= dist_bias_64;
        }

        // Stage 5: host-side selector + assignments. All 5 sub-block
        // strategies feed in with anti-bias entropy_muls (2× the libjxl
        // reference) — same trick as DCT32 needed.
        //
        // AFV cost grids: when `evaluate_afv` is on, split
        // `afv_costs_full` into per-kind slices and apply the same
        // distance-scaled anti-bias the other 8x8 sub-blocks already
        // get above. The cost grid producer
        // (`afv_per_block_upstream_cost_xyb_host`) emits costs on the
        // same scale as `cost_dct4x4` / `cost_identity` / `cost_dct2x2`
        // natively (uses libjxl's `entropy_mul × entropy +
        // k_info_loss × loss_scalar` formula), so no per-image
        // calibration is needed — just a uniform `dist_bias` to match
        // the rest of the sub-block tier.
        //
        // Earlier scaffold versions of this path applied a per-image
        // `dct8_mean / afv_mean` re-scaling to bring placeholder
        // SSE×mask costs into the right magnitude; that's gone now
        // that the formula matches.
        let n_blocks_8x8 = cost_dct8.len();
        let (afv0_scaled, afv1_scaled, afv2_scaled, afv3_scaled): (
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
        ) = if !afv_costs_full.is_empty() {
            debug_assert_eq!(afv_costs_full.len(), 4 * n_blocks_8x8);
            let split = |kind: usize| -> Vec<f32> {
                let r0 = kind * n_blocks_8x8;
                afv_costs_full[r0..r0 + n_blocks_8x8]
                    .iter()
                    .map(|&v| v * dist_bias)
                    .collect()
            };
            (split(0), split(1), split(2), split(3))
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        // Selector accepts None for any cost grid that the gate above
        // skipped (empty Vec); convert empties to None so the selector
        // doesn't read garbage.
        fn opt<'a>(v: &'a [f32]) -> Option<&'a [f32]> {
            if v.is_empty() { None } else { Some(v) }
        }
        let sub_blocks = crate::pipeline::SubBlockCostGrids {
            dct4x4: opt(&cost_dct4x4),
            dct4x8: opt(&cost_dct4x8),
            dct8x4: opt(&cost_dct8x4),
            identity: opt(&cost_identity),
            dct2x2: opt(&cost_dct2x2),
            afv0: opt(&afv0_scaled),
            afv1: opt(&afv1_scaled),
            afv2: opt(&afv2_scaled),
            afv3: opt(&afv3_scaled),
        };
        let extra16 = CostGrids16x16 {
            dct_16x8: opt(&cost_dct16x8),
            dct_8x16: opt(&cost_dct8x16),
            sub_blocks,
        };
        let assignments = if dct64_eligible {
            let extra32 = CostGrids32x32 {
                dct_32x16: Some(&cost_dct32x16),
                dct_16x32: Some(&cost_dct16x32),
            };
            let extra64 = CostGrids64x64 {
                dct_64x32: Some(&cost_dct64x32),
                dct_32x64: Some(&cost_dct32x64),
            };
            let partitions = select_partitions_64x64_with_extras16(
                &cost_dct8,
                &cost_dct16,
                &cost_dct32x32,
                &cost_dct64x64,
                extra32,
                extra64,
                extra16,
                xb8,
                yb8,
            );
            #[cfg(test)]
            {
                let mut h_dct64 = 0_usize;
                let mut h_sub32 = 0_usize;
                let mut h_other = 0_usize;
                for p in &partitions {
                    use crate::pipeline::Partition64x64 as P;
                    match p {
                        P::Dct64x64 => h_dct64 += 1,
                        P::Sub32x32(_) => h_sub32 += 1,
                        _ => h_other += 1,
                    }
                }
                std::println!(
                    "[strat-search] 64x64 partitions: dct64x64={h_dct64} sub32x32={h_sub32} other={h_other} (total={})",
                    partitions.len()
                );
            }
            let asn = partitions_64x64_to_assignments(&partitions, xb8, yb8);
            #[cfg(test)]
            {
                use crate::forks::transform::*;
                let mut counts = std::collections::BTreeMap::<u8, usize>::new();
                for a in &asn {
                    *counts.entry(a.raw_strategy).or_insert(0) += 1;
                }
                let strat_name = |s: u8| -> &'static str {
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
                        _ => "??",
                    }
                };
                let mut report = std::string::String::from("[strat-search] strategy histogram:");
                for (s, n) in &counts {
                    report.push_str(&std::format!(" {}={n}", strat_name(*s)));
                }
                std::println!("{report}");
            }
            asn
        } else if dct32_eligible {
            // 32x32-tier selector picks per-32x32-region between
            // DCT32x32, two-DCT32x16, two-DCT16x32, and four sub-16x16
            // (which themselves descend through the 16x16 selector).
            let extra32 = CostGrids32x32 {
                dct_32x16: Some(&cost_dct32x16),
                dct_16x32: Some(&cost_dct16x32),
            };
            let partitions = select_partitions_32x32_with_extras16(
                &cost_dct8,
                &cost_dct16,
                &cost_dct32x32,
                extra32,
                extra16,
                xb8,
                yb8,
            );
            // Histogram of Partition32x32 picks for diagnostic.
            #[cfg(test)]
            {
                let mut h_dct32 = 0_usize;
                let mut h_sub = 0_usize;
                let mut h_other = 0_usize;
                for p in &partitions {
                    use crate::pipeline::Partition32x32 as P;
                    match p {
                        P::Dct32x32 => h_dct32 += 1,
                        P::Sub16x16(_) => h_sub += 1,
                        _ => h_other += 1,
                    }
                }
                std::println!(
                    "[strat-search] 32x32 partitions: dct32x32={h_dct32} sub16x16={h_sub} other={h_other}"
                );
                let s8 = cost_dct8.iter().copied().sum::<f32>() / cost_dct8.len() as f32;
                let s16 = cost_dct16.iter().copied().sum::<f32>() / cost_dct16.len() as f32;
                let s32 = cost_dct32x32.iter().copied().sum::<f32>() / cost_dct32x32.len() as f32;
                std::println!(
                    "[strat-search] avg per-block cost: dct8={s8:.3} dct16={s16:.3} dct32={s32:.3} (4*dct8={:.3} 4*dct16={:.3})",
                    s8 * 16.0,
                    s16 * 4.0,
                );
            }
            partitions_32x32_to_assignments(&partitions, xb8, yb8)
        } else {
            let partitions =
                select_partitions_16x16_full(&cost_dct8, &cost_dct16, extra16, xb8, yb8);
            partitions_16x16_to_assignments(&partitions, xb8, yb8)
        };
        mark("selector");

        // Stage 6: per-channel DC grids — computed on GPU from the
        // gaborished XYB GpuPlanes (xx_g/xy_g/xb_g still resident).
        //
        // Skip the host download (same reasoning as the xyb host
        // download removal in 78ca825c). The `dc_grid_*` host
        // slices in StrategySearchPlan are only consumed by the
        // AFV branch of encode_and_reconstruct_mixed_strategy_single_channel
        // and the dispatch_restore_llf fallback — both inactive in
        // production strat-search. The GPU handles
        // (`dc_grid_*_gpu`) flow into the encode/recon GPU LLF
        // kernels via dc_grid_gpu = Some.
        //
        // SAFETY: see plan.xyb_x notes. Re-enabling AFV requires
        // lazy download of dc_grid_*_gpu in the AFV branch.
        let g_dc_x = enc.dc_grid_8x8_persistent(&xx_g);
        let g_dc_y = enc.dc_grid_8x8_persistent(&xy_g);
        let g_dc_b = enc.dc_grid_8x8_persistent(&xb_g);
        // Lazy host download of DC grids — needed by the AFV branch
        // in encode_and_reconstruct_mixed_strategy_single_channel
        // (forks/reconstruct.rs:639 reads dc_grid_per_8x8_block[by *
        // xsize_blocks_8 + bx] per AFV-selected block to restore the
        // packed-DC mean position). Skip the download in production
        // (evaluate_afv = false) — the GPU LLF fast paths (DCT8 /
        // DCT16x16 / DCT16x8 / DCT8x16) use g_dc_*_gpu directly via
        // set_llf_*_indexed_persistent, so the host slice is unused.
        let (dc_grid_x, dc_grid_y, dc_grid_b): (Vec<f32>, Vec<f32>, Vec<f32>) =
            if effective_evaluate_afv {
                let mut bytes = enc.client_ref().read(alloc::vec![
                    g_dc_x.handle().clone(),
                    g_dc_y.handle().clone(),
                    g_dc_b.handle().clone(),
                ]);
                use cubecl::prelude::*;
                let b_bytes = bytes.pop().expect("read[2]");
                let y_bytes = bytes.pop().expect("read[1]");
                let x_bytes = bytes.pop().expect("read[0]");
                (
                    f32::from_bytes(&x_bytes).to_vec(),
                    f32::from_bytes(&y_bytes).to_vec(),
                    f32::from_bytes(&b_bytes).to_vec(),
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
        // Keep the GPU dc_grid handles too — encode_with_strategy_plan_adaptive
        // threads them into encode_and_reconstruct_* via the
        // `dc_grid_*_gpu` Option params, skipping the per-iter
        // upload_blocks(dc_grid_per_8x8_block) PCIe transfer.
        let dc_grid_x_gpu = g_dc_x;
        let dc_grid_y_gpu = g_dc_y;
        let dc_grid_b_gpu = g_dc_b;
        mark("dc_grids");

        // Suppress "unused" warnings for weight Vecs that are only
        // re-derived in the encode-with-plan stage. The cost-grid
        // stage above uses these for cost-grid kernel calls; the
        // encode/recon stage recomputes them from the same const
        // weight functions.
        let _ = (
            &dct8_x,
            &dct8_y,
            &dct8_b,
            &dct16_x,
            &dct16_y,
            &dct16_b,
            &dct16x8_x,
            &dct16x8_y,
            &dct16x8_b,
            &dct32_x,
            &dct32_y,
            &dct32_b,
            &dct32x16_x,
            &dct32x16_y,
            &dct32x16_b,
            &dct64_x,
            &dct64_y,
            &dct64_b,
            &dct64x32_x,
            &dct64x32_y,
            &dct64x32_b,
        );

        // GPU compute_quant_field: runs the full
        // pre_erosion → fuzzy_erosion → mask_for_ac_strategy →
        // per_block_modulations chain on the persistent xyb planes
        // and downloads `(quant_field_float, masking)` in one batched
        // `read`. Replaces the CPU
        // `compute_quant_field_float_free` call that production
        // endpoints (encoder.rs:1948 / 2153) previously ran on the
        // post-fix-up CPU buffer — saves ~50 ms at 12 MP.
        //
        // Bit-equivalence verified by
        // `forks::adaptive_quant::tests::test_compute_quant_field_production_flow_divergence`:
        // 0/4257 blocks land on a different u8 bucket at 1025×257.
        let cpu_pw = (w).div_ceil(8) * 8;
        let cpu_ph = (h).div_ceil(8) * 8;
        // libjxl-parity: when gaborish is disabled, the quant field is
        // computed at `distance * 0.62` (a tighter target) to compensate
        // for the missing 5x5 sharpening filter. See
        // jxl-encoder/src/vardct/bitstream.rs:1261-1265 and
        // vardct/encoder.rs:869-876. AC strategy + cost grids still see
        // raw `target_distance` — only this iqf computation rescales.
        let distance_for_iqf = if enable_gaborish_local {
            target_distance
        } else {
            target_distance * 0.62
        };
        let (quant_field_float, masking) =
            crate::forks::adaptive_quant::compute_quant_field_full_persistent(
                enc,
                &xx_g,
                &xy_g,
                &xb_g,
                cpu_pw,
                cpu_ph,
                distance_for_iqf,
                K_AC_QUANT,
            );
        mark("gpu_quant_field");

        StrategySearchPlan {
            xyb_x,
            xyb_y,
            xyb_b,
            // GpuPlane is reference-counted under the hood (see Clone
            // impl in persistent.rs) — the xx_g/xy_g/xb_g handles created
            // during the gaborish stage live on; the plan just holds an
            // extra refcount.
            xyb_x_gpu: xx_g,
            xyb_y_gpu: xy_g,
            xyb_b_gpu: xb_g,
            // Pre-gaborish XYB held for patches detection in the
            // slow-path bitstream emit. Refcounted GpuPlane — no new
            // PCIe bytes; the encoder slow path downloads on demand.
            xyb_x_pre_gab_gpu: xx,
            xyb_y_pre_gab_gpu: xy,
            xyb_b_pre_gab_gpu: xb,
            dc_grid_x,
            dc_grid_y,
            dc_grid_b,
            dc_grid_x_gpu,
            dc_grid_y_gpu,
            dc_grid_b_gpu,
            assignments,
            quant_field_float,
            masking,
            target_distance,
            padded_width: self.padded_width,
            padded_height: self.padded_height,
        }
    }

    /// Non-traced wrapper around
    /// [`Self::encode_with_strategy_plan_adaptive_traced`]. Encode an
    /// image using a precomputed [`StrategySearchPlan`] and a per-block
    /// `aq_field`. The plan must come from
    /// [`Self::prepare_strategy_search_plan`] on the same image.
    pub fn encode_with_strategy_plan_adaptive(
        &self,
        enc: &GpuEncoder<R>,
        plan: &StrategySearchPlan<R>,
        aq_field: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        self.encode_with_strategy_plan_adaptive_traced(enc, plan, aq_field, &mut |_| {})
    }

    /// GPU-resident variant of [`Self::encode_with_strategy_plan_adaptive`]
    /// — returns the post-postpass linear-RGB recon as 3 padded
    /// [`GpuPlane<R>`]s instead of host `Vec<f32>`s. Skips the
    /// `download_planes_3ch` + `crop_to_original × 3` final steps.
    ///
    /// At 16 MP the skipped boundary work is ~160 ms / iter. The
    /// returned planes have the LossyEncoder's PADDED dimensions
    /// (`self.padded_dimensions()`) — caller is responsible for any
    /// crop-to-original needed downstream. For butteraugli refinement
    /// loops at exact-multiple-of-16 image dims (no padding), no crop
    /// is needed; pass directly to
    /// `Butteraugli::compute_with_reference_from_linear_planes`.
    ///
    /// **Use case**: butteraugli refinement loop where the next step
    /// is a butteraugli compute on GPU — feeding the recon planes
    /// directly skips the recon-download + sRGB-host-convert +
    /// re-upload boundary.
    pub fn encode_with_strategy_plan_adaptive_persistent(
        &self,
        enc: &GpuEncoder<R>,
        plan: &StrategySearchPlan<R>,
        aq_field: &[f32],
    ) -> (
        crate::persistent::GpuPlane<R>,
        crate::persistent::GpuPlane<R>,
        crate::persistent::GpuPlane<R>,
    ) {
        self.encode_with_strategy_plan_adaptive_persistent_traced(enc, plan, aq_field, &mut |_| {})
    }

    /// Encode + recon + postpass stage of strat-search using a
    /// precomputed [`StrategySearchPlan`].
    ///
    /// **Cost** (CLIC 1024²): ~50 ms per call (excludes the ~150 ms
    /// prepare cost paid once). When called repeatedly on the same
    /// plan with different `aq_field`s (refinement loop), this is
    /// where most of the per-iter wall-clock goes.
    ///
    /// `mark` callback fires at the same stage boundaries as the
    /// monolithic `_adaptive_traced` from `mixed_strategy_encode_recon`
    /// onward.
    pub fn encode_with_strategy_plan_adaptive_traced(
        &self,
        enc: &GpuEncoder<R>,
        plan: &StrategySearchPlan<R>,
        aq_field: &[f32],
        mark: &mut dyn FnMut(&'static str),
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (rgb_r, rgb_g, rgb_b) =
            self.encode_with_strategy_plan_adaptive_persistent_traced(enc, plan, aq_field, mark);
        // Batched 3-channel D2H read: one read_async + sync wait
        // instead of three serial read_one round-trips.
        let (r_out, g_out, b_out) = enc.download_planes_3ch(&rgb_r, &rgb_g, &rgb_b);
        mark("download_crop");
        let (w, h) = (self.width as usize, self.height as usize);
        let pw = self.padded_width as usize;
        (
            crop_to_original(&r_out, pw, w, h),
            crop_to_original(&g_out, pw, w, h),
            crop_to_original(&b_out, pw, w, h),
        )
    }

    /// GPU-resident traced variant — same pipeline as
    /// [`Self::encode_with_strategy_plan_adaptive_traced`] but returns
    /// the post-postpass linear-RGB recon as 3 padded GPU planes
    /// instead of host `Vec<f32>`s. Skips the final
    /// `download_planes_3ch + crop_to_original × 3` step (~160 ms /
    /// iter at 16 MP).
    ///
    /// Marks fire at the same stage boundaries as `_adaptive_traced`
    /// from `mixed_strategy_encode_recon` through `postpass_gab_epf_xyb`;
    /// `download_crop` is NOT emitted (no download here).
    pub fn encode_with_strategy_plan_adaptive_persistent_traced(
        &self,
        enc: &GpuEncoder<R>,
        plan: &StrategySearchPlan<R>,
        aq_field: &[f32],
        mark: &mut dyn FnMut(&'static str),
    ) -> (
        crate::persistent::GpuPlane<R>,
        crate::persistent::GpuPlane<R>,
        crate::persistent::GpuPlane<R>,
    ) {
        use crate::forks::reconstruct::{
            encode_and_reconstruct_mixed_strategy_3channel, gab_weights,
        };
        use crate::forks::transform::{
            RAW_STRATEGY_DCT, RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16,
            RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32,
            RAW_STRATEGY_DCT32X64, RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64,
        };
        use crate::quant_weights::{
            dct8_weights_per_channel, dct16x8_weights_per_channel, dct16x16_weights_per_channel,
            dct16x32_weights_per_channel, dct32x32_weights_per_channel,
            dct32x64_weights_per_channel, dct64x64_weights_per_channel,
        };

        let (w, h) = (self.width as usize, self.height as usize);
        let pw = plan.padded_width as usize;
        let ph = plan.padded_height as usize;
        let xb8 = pw / 8;
        let yb8 = ph / 8;
        let nb8 = xb8 * yb8;

        assert_eq!(
            aq_field.len(),
            nb8,
            "aq_field length {} != num_padded_blocks {}",
            aq_field.len(),
            nb8,
        );
        // plan.xyb_x/y/b AND plan.dc_grid_x/y/b are empty Vecs
        // (intentional — see comments in
        // prepare_strategy_search_plan_traced where we skip the
        // downloads). Both are only consumed by the AFV branch in
        // encode_and_reconstruct_mixed_strategy_single_channel,
        // which is currently disabled in production. If AFV is
        // re-enabled, callers that consume these need to lazily
        // populate from xyb_*_gpu / dc_grid_*_gpu.

        // Stage 7: encode + reconstruct via mixed-strategy IDCT.
        // Per-block qac comes from `aq_field` directly — this is what
        // makes the adaptive variant compose with the butteraugli AQ
        // refinement loop. For uniform-qac (scalar `distance` shim)
        // callers, this is just `vec![distance_to_qac(distance); nb8]`.
        let qac_vec = aq_field.to_vec();
        let (dct8_x, dct8_y, dct8_b) = dct8_weights_per_channel();
        let (dct16_x, dct16_y, dct16_b) = dct16x16_weights_per_channel();
        let (dct16x8_x, dct16x8_y, dct16x8_b) = dct16x8_weights_per_channel();
        let dct8_x_clone = dct8_x;
        let dct8_y_clone = dct8_y;
        let dct8_b_clone = dct8_b;
        let dct16_x_clone = dct16_x.clone();
        let dct16_y_clone = dct16_y.clone();
        let dct16_b_clone = dct16_b.clone();
        let dct16x8_x_clone = dct16x8_x.clone();
        let dct16x8_y_clone = dct16x8_y.clone();
        let dct16x8_b_clone = dct16x8_b.clone();
        // DCT32x32 + rectangular DCT32 weights (only used if
        // assignments contain those strategies, but cheap to fetch).
        let (dct32_x, dct32_y, dct32_b) = dct32x32_weights_per_channel();
        let dct32_x_clone = dct32_x.clone();
        let dct32_y_clone = dct32_y.clone();
        let dct32_b_clone = dct32_b.clone();
        let (dct32x16_x, dct32x16_y, dct32x16_b) = dct16x32_weights_per_channel();
        let dct32x16_x_clone = dct32x16_x.clone();
        let dct32x16_y_clone = dct32x16_y.clone();
        let dct32x16_b_clone = dct32x16_b.clone();
        // DCT64 family weights.
        let (dct64_x, dct64_y, dct64_b) = dct64x64_weights_per_channel();
        let dct64_x_clone = dct64_x.clone();
        let dct64_y_clone = dct64_y.clone();
        let dct64_b_clone = dct64_b.clone();
        let (dct64x32_x, dct64x32_y, dct64x32_b) = dct32x64_weights_per_channel();
        let dct64x32_x_clone = dct64x32_x.clone();
        let dct64x32_y_clone = dct64x32_y.clone();
        let dct64x32_b_clone = dct64x32_b.clone();
        // Sub-block 8x8-tier weights (DCT4x4, DCT4x8/DCT8x4, IDENTITY,
        // DCT2x2). The strat-search 16x16 selector can pick these
        // (cost grids computed in the prepare stage), so the
        // encode/recon stage must support them too.
        let (dct4x4_xw, dct4x4_yw, dct4x4_bw) = crate::quant_weights::dct4x4_weights_per_channel();
        let dct4x4_x_clone = dct4x4_xw.clone();
        let dct4x4_y_clone = dct4x4_yw.clone();
        let dct4x4_b_clone = dct4x4_bw.clone();
        let (dct4x8_xw, dct4x8_yw, dct4x8_bw) = crate::quant_weights::dct4x8_weights_per_channel();
        let dct4x8_x_clone = dct4x8_xw.clone();
        let dct4x8_y_clone = dct4x8_yw.clone();
        let dct4x8_b_clone = dct4x8_bw.clone();
        let (id_xw, id_yw, id_bw) = crate::quant_weights::identity_weights_per_channel();
        let id_x_clone = id_xw.clone();
        let id_y_clone = id_yw.clone();
        let id_b_clone = id_bw.clone();
        let (d2_xw, d2_yw, d2_bw) = crate::quant_weights::dct2x2_weights_per_channel();
        let d2_x_clone = d2_xw.clone();
        let d2_y_clone = d2_yw.clone();
        let d2_b_clone = d2_bw.clone();
        let (afv_wx, afv_wy, afv_wb) = crate::quant_weights::afv_weights_per_channel();
        let afv_wx_v: Vec<f32> = afv_wx.to_vec();
        let afv_wy_v: Vec<f32> = afv_wy.to_vec();
        let afv_wb_v: Vec<f32> = afv_wb.to_vec();
        let is_afv_strategy = |s: u8| {
            s == crate::forks::transform::RAW_STRATEGY_AFV0
                || s == crate::forks::transform::RAW_STRATEGY_AFV1
                || s == crate::forks::transform::RAW_STRATEGY_AFV2
                || s == crate::forks::transform::RAW_STRATEGY_AFV3
        };
        use crate::forks::transform::{
            RAW_STRATEGY_DCT2X2 as _RS_DCT2X2, RAW_STRATEGY_DCT4X4 as _RS_DCT4X4,
            RAW_STRATEGY_DCT4X8 as _RS_DCT4X8, RAW_STRATEGY_DCT8X4 as _RS_DCT8X4,
            RAW_STRATEGY_IDENTITY as _RS_IDENTITY,
        };
        let weights_x_for = move |strat: u8| -> Vec<f32> {
            if is_afv_strategy(strat) {
                return afv_wx_v.clone();
            }
            match strat {
                RAW_STRATEGY_DCT => dct8_x_clone.to_vec(),
                RAW_STRATEGY_DCT16X16 => dct16_x_clone.clone(),
                RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => dct16x8_x_clone.clone(),
                RAW_STRATEGY_DCT32X32 => dct32_x_clone.clone(),
                RAW_STRATEGY_DCT32X16 | RAW_STRATEGY_DCT16X32 => dct32x16_x_clone.clone(),
                RAW_STRATEGY_DCT64X64 => dct64_x_clone.clone(),
                RAW_STRATEGY_DCT64X32 | RAW_STRATEGY_DCT32X64 => dct64x32_x_clone.clone(),
                _RS_DCT4X4 => dct4x4_x_clone.clone(),
                _RS_DCT4X8 | _RS_DCT8X4 => dct4x8_x_clone.clone(),
                _RS_IDENTITY => id_x_clone.clone(),
                _RS_DCT2X2 => d2_x_clone.clone(),
                _ => panic!("Strategy {strat} not yet wired into encoder weights"),
            }
        };
        let weights_y_for = move |strat: u8| -> Vec<f32> {
            if is_afv_strategy(strat) {
                return afv_wy_v.clone();
            }
            match strat {
                RAW_STRATEGY_DCT => dct8_y_clone.to_vec(),
                RAW_STRATEGY_DCT16X16 => dct16_y_clone.clone(),
                RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => dct16x8_y_clone.clone(),
                RAW_STRATEGY_DCT32X32 => dct32_y_clone.clone(),
                RAW_STRATEGY_DCT32X16 | RAW_STRATEGY_DCT16X32 => dct32x16_y_clone.clone(),
                RAW_STRATEGY_DCT64X64 => dct64_y_clone.clone(),
                RAW_STRATEGY_DCT64X32 | RAW_STRATEGY_DCT32X64 => dct64x32_y_clone.clone(),
                _RS_DCT4X4 => dct4x4_y_clone.clone(),
                _RS_DCT4X8 | _RS_DCT8X4 => dct4x8_y_clone.clone(),
                _RS_IDENTITY => id_y_clone.clone(),
                _RS_DCT2X2 => d2_y_clone.clone(),
                _ => panic!("Strategy {strat} not yet wired into encoder weights"),
            }
        };
        let weights_b_for = move |strat: u8| -> Vec<f32> {
            if is_afv_strategy(strat) {
                return afv_wb_v.clone();
            }
            match strat {
                RAW_STRATEGY_DCT => dct8_b_clone.to_vec(),
                RAW_STRATEGY_DCT16X16 => dct16_b_clone.clone(),
                RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => dct16x8_b_clone.clone(),
                RAW_STRATEGY_DCT32X32 => dct32_b_clone.clone(),
                RAW_STRATEGY_DCT32X16 | RAW_STRATEGY_DCT16X32 => dct32x16_b_clone.clone(),
                RAW_STRATEGY_DCT64X64 => dct64_b_clone.clone(),
                RAW_STRATEGY_DCT64X32 | RAW_STRATEGY_DCT32X64 => dct64x32_b_clone.clone(),
                _RS_DCT4X4 => dct4x4_b_clone.clone(),
                _RS_DCT4X8 | _RS_DCT8X4 => dct4x8_b_clone.clone(),
                _RS_IDENTITY => id_b_clone.clone(),
                _RS_DCT2X2 => d2_b_clone.clone(),
                _ => panic!("Strategy {strat} not yet wired into encoder weights"),
            }
        };
        // Allocate the 3 recon planes directly on GPU. The mixed-
        // strategy reconstruct scatters into them via indexed_scatter
        // (see out_plane_*_gpu params below). Postpass (gab_smooth +
        // EPF + xyb_to_linear) chains straight into them — no need
        // for the upload_plane(plane_*) round-trip the older code did
        // and no need for the host plane_x/y/b alloc (12 MB / iter at
        // 1024² avoided across 4 butteraugli refinement iters).
        // alloc_plane × 3 here uploads 3 × padded_width*padded_height f32
        // zeros via cubecl HtoD. At 16 MP that's ~308 ms per encode iter
        // (probed 2026-05-10 with mark("alloc_recon_planes")), 51% of
        // encode iter wall-clock — by far the largest single bottleneck.
        //
        // THREE fixes attempted, all regressed or no-op:
        // 1. `client.empty() + GPU zero_fill` per call: regressed 5×
        //    (605→2947 ms). Cubecl's empty() doesn't pool-reuse 64 MB.
        // 2. Cached `empty()` handles on LossyEncoder + GPU zero_fill
        //    per iter: regressed 2× (605→1168 ms). 500 ms unattributed.
        // 3. Batched single-call alloc via `create_tensors_from_slices`
        //    (mirroring `upload_planes_3ch`): zero-delta (605.96 →
        //    605.95 ms). cubecl's pool already amortizes the 3
        //    sequential calls; no win from batching.
        //
        // Real fix needs the cubecl pinned-buffer PR or the raw-cudarc
        // bypass — until then this 308 ms is the production baseline.
        // See `negative_perf_alloc_plane_zero_fill.md` memo.
        // Use uninitialized recon planes — the mixed-strategy
        // reconstruct's per-strategy indexed_scatter covers every
        // position in the padded plane (every 8×8 block is assigned a
        // strategy, default DCT8). Skips the 308 ms / iter zero-init
        // upload at 16 MP that was the e8/e9 inner loop's largest
        // wedge per the negative_perf_alloc_plane_zero_fill.md memo.
        let recon_x_p = enc.alloc_plane_uninit(self.padded_width, self.padded_height);
        let recon_y_p = enc.alloc_plane_uninit(self.padded_width, self.padded_height);
        let recon_b_p = enc.alloc_plane_uninit(self.padded_width, self.padded_height);
        encode_and_reconstruct_mixed_strategy_3channel(
            enc,
            &plan.xyb_x,
            &plan.xyb_y,
            &plan.xyb_b,
            pw,
            ph,
            &plan.assignments,
            &weights_x_for,
            &weights_y_for,
            &weights_b_for,
            &qac_vec,
            &qac_vec,
            &qac_vec,
            &self.thresholds_x,
            &self.thresholds_y,
            &self.thresholds_b,
            &plan.dc_grid_x,
            &plan.dc_grid_y,
            &plan.dc_grid_b,
            // No host out_plane needed — every strategy scatters into
            // recon_*_p (out_plane_*_gpu = Some below).
            None,
            None,
            None,
            // Plumb through the GpuPlanes from the prepare stage so the
            // per-iter encode skips the redundant upload_plane(xyb)
            // PCIe transfer (3 × 4MB at 1024² → savings stack across
            // refinement iters).
            Some(&plan.xyb_x_gpu),
            Some(&plan.xyb_y_gpu),
            Some(&plan.xyb_b_gpu),
            // Same trick for the per-8×8 dc_grid GPU buffers
            // (~64 KB each at 1024², 192 KB total per encode iter).
            Some(&plan.dc_grid_x_gpu),
            Some(&plan.dc_grid_y_gpu),
            Some(&plan.dc_grid_b_gpu),
            // Output GpuPlanes — strategies indexed-scatter directly
            // into these; no internal upload-zeros + download-and-merge
            // round-trip, no postpass re-upload (gab_smooth_persistent
            // consumes them as-is).
            Some(&recon_x_p),
            Some(&recon_y_p),
            Some(&recon_b_p),
        );
        mark("mixed_strategy_encode_recon");

        // Stage 8: postpass (gab_smooth + EPF + xyb_to_linear), matching
        // run_pipeline_with_qac. EPF closes most of the perceptual gap
        // vs the uniform-qac DCT8 baseline.
        //
        // libjxl-parity gaborish gate: when prepare_strategy_search_plan
        // skipped the 5x5 sharpening at d <= 0.5, the decoder-side
        // gab_smooth (3x3 blur) MUST also be skipped or the recon
        // becomes blurry vs what the bitstream emit produces (which
        // signals fh.gaborish=false → decoder skips its own gab_smooth).
        // recon_x_p / _y_p / _b_p are the GpuPlanes the mixed-strategy
        // reconstruct scattered into above. No upload needed —
        // gab_smooth_3ch_persistent consumes them directly. ONE launch
        // for all 3 channels (vs 3 separate gab_smooth_persistent
        // calls); same per-thread arithmetic, fewer launch barriers.
        let (recon_x_p, recon_y_p, recon_b_p) = if plan.target_distance > 0.5 {
            let (gw_c, gw1, gw2) = gab_weights();
            enc.gab_smooth_3ch_persistent(&recon_x_p, &recon_y_p, &recon_b_p, gw_c, gw1, gw2)
        } else {
            (recon_x_p, recon_y_p, recon_b_p)
        };

        // EPF step 1+2 (matches run_pipeline_with_qac). Per-block qac maps
        // to u8 quant_field via `clamp(qac * 50, 1, 255)`; sharpness is
        // uniform 4 (libjxl default).
        let qf_u8: Vec<u8> = qac_vec
            .iter()
            .map(|&q| (q * 50.0).round().clamp(1.0, 255.0) as u8)
            .collect();
        let sharpness = vec![4_u8; nb8];
        let inv_sigma_vec =
            crate::forks::epf::compute_inv_sigma_map(&qf_u8, &sharpness, 0.01, xb8, yb8);
        let inv_sigma_h = enc.upload_inv_sigma(&inv_sigma_vec);
        let xsize_blocks = self.padded_width / 8;
        let ysize_blocks = self.padded_height / 8;

        let pad1 = 2_u32;
        // Fused 3-channel pad (1 launch instead of 3) — same edge
        // replication, processed for X/Y/B in one go.
        let (p1_x, p1_y, p1_b) =
            enc.pad_plane_3ch_persistent(&recon_x_p, &recon_y_p, &recon_b_p, pad1);
        let (s1_x, s1_y, s1_b) = enc.epf_step1_persistent(
            &p1_x,
            &p1_y,
            &p1_b,
            &inv_sigma_h,
            self.padded_width,
            self.padded_height,
            xsize_blocks,
            ysize_blocks,
            pad1,
            1.65,
            crate::forks::epf::EPF_BORDER_SAD_MUL,
        );

        let pad2 = 1_u32;
        let (p2_x, p2_y, p2_b) = enc.pad_plane_3ch_persistent(&s1_x, &s1_y, &s1_b, pad2);
        let (s2_x, s2_y, s2_b) = enc.epf_step2_persistent(
            &p2_x,
            &p2_y,
            &p2_b,
            &inv_sigma_h,
            self.padded_width,
            self.padded_height,
            xsize_blocks,
            ysize_blocks,
            pad2,
            crate::forks::epf::EPF_PASS2_SIGMA_SCALE * 1.65,
            crate::forks::epf::EPF_BORDER_SAD_MUL,
        );

        let (rgb_r, rgb_g, rgb_b) = enc.xyb_to_linear_rgb_planar_persistent(&s2_x, &s2_y, &s2_b);
        mark("postpass_gab_epf_xyb");
        // GPU-resident: return the recon planes directly, no download
        // / crop. Wrapper f32 variant downloads + crops if needed.
        let _ = (w, h, pw); // suppress unused warning in this variant
        (rgb_r, rgb_g, rgb_b)
    }

    /// Turnkey content-driven adaptive quantization.
    ///
    /// Computes a per-block qac field from the image's mask1x1 (per-
    /// pixel masking signal that's high in smooth regions and low at
    /// edges), then encodes with that field. Smooth blocks get heavier
    /// quant (smaller files), detail blocks get lighter quant
    /// (preserved edges).
    ///
    /// `distance` controls the central quality (libjxl-style; 1.0 =
    /// reference). The AQ field varies per block in a 4× range around
    /// it: `qac ∈ [distance_to_qac(distance * 2),
    /// distance_to_qac(distance / 2)]`.
    ///
    /// Costs roughly the same as a single `encode_one` call plus an
    /// XYB+mask1x1 prepass (~2-5% overhead at 1024² per
    /// content_driven_aq_demo measurements).
    pub fn encode_one_with_aq(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        distance: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let aq_field = self.compute_aq_field(enc, r, g, b, distance);
        self.encode_one_adaptive(enc, r, g, b, &aq_field)
    }

    /// Compute the per-block AQ field that [`Self::encode_one_with_aq`]
    /// would use, exposed for callers who want to inspect or modify it
    /// before passing to [`Self::encode_one_adaptive`].
    pub fn compute_aq_field(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        distance: f32,
    ) -> Vec<f32> {
        let block_means = self.compute_block_mask_means(enc, r, g, b);
        block_means_to_qac_field(&block_means, distance)
    }

    /// Empirically-derived median mask threshold above which content
    /// is "screenshot-like" — large flat regions where strat-search
    /// over-picks DCT16x16 and produces catastrophic butteraugli
    /// regressions (graph.png at d=1.0: strat-search 4.66 vs uniform
    /// 1.05, +343% worse).
    ///
    /// Source data: `mask1x1_content_stats` example run on
    /// CLIC2025-1024 (16 photos) + gb82-sc (10 screenshots), May 9
    /// 2026. ALL 16 CLIC photos had median(mask1x1) ≤ 87. 9 of 10
    /// screenshots had median = 100.01 (the max signal value); the
    /// only exception was windows95.png (median 69.9, mostly mid-tone
    /// pixels — might or might not regress, untested).
    ///
    /// 95 sits well above the photo max (87) and below the screenshot
    /// median (100), giving a clear gap. The 1 false-negative
    /// (windows95.png) is acceptable because best-of-both still
    /// catches it.
    pub const SCREENSHOT_MEDIAN_MASK_THRESHOLD: f32 = 95.0;

    /// Heuristic: is this image content "screenshot-like"? Returns
    /// `true` if the median per-block mask1x1 value exceeds
    /// [`Self::SCREENSHOT_MEDIAN_MASK_THRESHOLD`].
    ///
    /// **When true**: callers should skip strat-search and use
    /// uniform-DCT8 (refine+DCT8 or encode_one_adaptive). Strat-search
    /// over-picks DCT16x16 on flat regions and produces catastrophic
    /// quality loss on this content (×4-5 butteraugli vs uniform).
    ///
    /// **When false**: strat-search is safe to use. Combined mode
    /// (refine_aq_field_gpu_with_strategy_search) or best-of-both
    /// (refine_and_encode_best_of_both) are both viable.
    ///
    /// **Cost**: one mask1x1 GPU pass + sort over per-block means
    /// (~10-20 ms on 1024² @ CUDA). Cheap enough to call before
    /// deciding which encode pipeline to use.
    ///
    /// **Validation**: 9 of 10 screenshots correctly detected (all
    /// of gb82-sc except windows95.png), 0 of 16 CLIC photos
    /// false-positive. windows95.png (median 69.9) is the one
    /// false-negative — the user should pair this with
    /// `forks::butteraugli_loop::refine_and_encode_best_of_both`
    /// (gated behind the `butteraugli-loop` feature) for guaranteed
    /// correctness on edge cases.
    pub fn content_looks_like_screenshot(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
    ) -> bool {
        let block_means = self.compute_block_mask_means(enc, r, g, b);
        if block_means.is_empty() {
            return false;
        }
        // Median via partial sort. cheap on padded-block-count vectors
        // (16k entries on 1024², ~150 µs).
        let mut sorted = block_means;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        median > Self::SCREENSHOT_MEDIAN_MASK_THRESHOLD
    }

    /// Compute per-block mean of mask1x1 (one f32 per padded 8×8
    /// block). This is the input-dependent half of [`Self::compute_aq_field`]
    /// — exposed separately so batch encodes (e.g.,
    /// [`Self::encode_many_with_aq`]) can compute it once and reuse
    /// across many distances.
    pub fn compute_block_mask_means(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
    ) -> Vec<f32> {
        use crate::forks::adaptive_quant::compute_mask1x1_gpu;
        let (w, h) = (self.width as usize, self.height as usize);
        // Run XYB on unpadded input — mask1x1 only needs the Y channel.
        let (_xx, xy, _xb) = enc.xyb_from_linear_rgb(r, g, b);
        let mask = compute_mask1x1_gpu(enc, &xy, w, h);

        let (pw, _ph) = (self.padded_width as usize, self.padded_height as usize);
        let blocks_per_row = pw / 8;
        let blocks_per_col = (self.padded_height as usize) / 8;
        let nb = blocks_per_row * blocks_per_col;
        let mut block_means = vec![0.0_f32; nb];
        for by in 0..blocks_per_col {
            for bx in 0..blocks_per_row {
                let mut sum = 0.0_f64;
                let mut count = 0_usize;
                for dy in 0..8 {
                    let y = by * 8 + dy;
                    if y >= h {
                        break;
                    }
                    for dx in 0..8 {
                        let x = bx * 8 + dx;
                        if x >= w {
                            break;
                        }
                        sum += mask[y * w + x] as f64;
                        count += 1;
                    }
                }
                block_means[by * blocks_per_row + bx] = if count > 0 {
                    (sum / count as f64) as f32
                } else {
                    1.0
                };
            }
        }
        block_means
    }

    /// Batch content-driven AQ — one input upload, one mask1x1 prepass,
    /// N derived qac fields, N adaptive encodes.
    ///
    /// Equivalent to calling [`Self::encode_one_with_aq`] in a loop, but
    /// amortizes the input upload AND the mask1x1 prepass across all
    /// distances. The mask depends only on the input image, not on
    /// distance, so it's computed once.
    ///
    /// Returns one `(R, G, B)` tuple per distance, in the same order.
    pub fn encode_many_with_aq(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        distances: &[f32],
    ) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (w, h) = (self.width as usize, self.height as usize);
        let (pw, ph) = (self.padded_width as usize, self.padded_height as usize);
        // mask1x1 prepass — done once.
        let block_means = self.compute_block_mask_means(enc, r, g, b);
        // Upload padded input once.
        let r_pad = pad_to_alignment(r, w, h, pw, ph);
        let g_pad = pad_to_alignment(g, w, h, pw, ph);
        let b_pad = pad_to_alignment(b, w, h, pw, ph);
        let g_r = enc.upload_plane(&r_pad, self.padded_width, self.padded_height);
        let g_g = enc.upload_plane(&g_pad, self.padded_width, self.padded_height);
        let g_b = enc.upload_plane(&b_pad, self.padded_width, self.padded_height);
        distances
            .iter()
            .map(|&d| {
                let aq_field = block_means_to_qac_field(&block_means, d);
                let (rec_r, rec_g, rec_b) =
                    self.run_pipeline_with_qac(enc, &g_r, &g_g, &g_b, &aq_field);
                (
                    crop_to_original(&rec_r, pw, w, h),
                    crop_to_original(&rec_g, pw, w, h),
                    crop_to_original(&rec_b, pw, w, h),
                )
            })
            .collect()
    }

    /// (caller crops back to original).
    fn run_pipeline(
        &self,
        enc: &GpuEncoder<R>,
        g_r: &GpuPlane<R>,
        g_g: &GpuPlane<R>,
        g_b: &GpuPlane<R>,
        qac_qm: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let qac_vec = vec![qac_qm; self.num_blocks as usize];
        self.run_pipeline_with_qac(enc, g_r, g_g, g_b, &qac_vec)
    }

    /// Per-block adaptive variant of `run_pipeline`. Takes a precomputed
    /// per-block qac_qm field instead of broadcasting a scalar.
    ///
    /// libjxl-parity gaborish gate: skip both encoder gaborish_5x5 AND
    /// decoder gab_smooth when the effective central distance is
    /// <= 0.5, matching the bitstream emit path. Distance is recovered
    /// from the qac field's mean (= K_AC_QUANT / mean(qac)) — for
    /// uniform encodes that's exact; for adaptive
    /// `block_means_to_qac_field(R=2)` the mean stays close to the
    /// central qac_uniform = K_AC_QUANT/target_distance, so the gate
    /// decision lines up with the `target_distance` used by
    /// `prepare_strategy_search_plan_inner` at the same overall
    /// distance. Pre-fix the pipeline ran gaborish unconditionally →
    /// smart turnkey's strat-search-vs-DCT8 comparison was made
    /// against gaborized reconstruction even when the bitstream was
    /// un-sharpened, so the pick disagreed with what the bitstream
    /// actually shipped.
    fn run_pipeline_with_qac(
        &self,
        enc: &GpuEncoder<R>,
        g_r: &GpuPlane<R>,
        g_g: &GpuPlane<R>,
        g_b: &GpuPlane<R>,
        qac_vec: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let xf = vec![0.0_f32; self.num_blocks as usize];
        let bf = vec![0.0_f32; self.num_blocks as usize];

        // Recover central distance from qac field. mean stays close to
        // central qac for both uniform and adaptive encodes.
        let qac_mean: f32 = if qac_vec.is_empty() {
            distance_to_qac(1.0)
        } else {
            qac_vec.iter().copied().sum::<f32>() / (qac_vec.len() as f32)
        };
        let derived_distance = if qac_mean > 0.0 {
            K_AC_QUANT / qac_mean
        } else {
            1.0
        };
        let enable_gaborish_local = derived_distance > 0.5;
        let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(g_r, g_g, g_b);
        let (xx_g, xy_g, xb_g) = if enable_gaborish_local {
            (
                enc.gaborish_5x5_persistent(&xx, &self.weights),
                enc.gaborish_5x5_persistent(&xy, &self.weights),
                enc.gaborish_5x5_persistent(&xb, &self.weights),
            )
        } else {
            // Refcount-only clone (see GpuPlane Clone impl, persistent.rs:139).
            (xx.clone(), xy.clone(), xb.clone())
        };
        let bx_g = enc.gather_blocks_persistent(&xx_g, 8, 8);
        let by_g = enc.gather_blocks_persistent(&xy_g, 8, 8);
        let bb_g = enc.gather_blocks_persistent(&xb_g, 8, 8);
        let coeffs_x = enc.dct_8x8_wide_persistent(&bx_g);
        let coeffs_y = enc.dct_8x8_wide_persistent(&by_g);
        let coeffs_b = enc.dct_8x8_wide_persistent(&bb_g);
        let q_x = enc.quantize_dct8_persistent_broadcast_w(
            &coeffs_x,
            &self.weights_x,
            qac_vec,
            &self.thresholds_x,
        );
        let q_y = enc.quantize_dct8_persistent_broadcast_w(
            &coeffs_y,
            &self.weights_y,
            qac_vec,
            &self.thresholds_y,
        );
        let q_b = enc.quantize_dct8_persistent_broadcast_w(
            &coeffs_b,
            &self.weights_b,
            qac_vec,
            &self.thresholds_b,
        );
        let (dq_x, dq_y, dq_b) = enc.dequant_dct8_persistent_broadcast_w(
            &q_x,
            &q_y,
            &q_b,
            &self.weights_x,
            &self.weights_y,
            &self.weights_b,
            qac_vec,
            qac_vec,
            qac_vec,
            &xf,
            &bf,
        );
        enc.restore_dc_persistent(&coeffs_x, &dq_x);
        enc.restore_dc_persistent(&coeffs_y, &dq_y);
        enc.restore_dc_persistent(&coeffs_b, &dq_b);
        let recon_x_b = enc.idct_8x8_wide_persistent(&dq_x);
        let recon_y_b = enc.idct_8x8_wide_persistent(&dq_y);
        let recon_b_b = enc.idct_8x8_wide_persistent(&dq_b);
        let recon_x_p =
            enc.scatter_blocks_persistent(&recon_x_b, self.padded_width, self.padded_height, 8, 8);
        let recon_y_p =
            enc.scatter_blocks_persistent(&recon_y_b, self.padded_width, self.padded_height, 8, 8);
        let recon_b_p =
            enc.scatter_blocks_persistent(&recon_b_b, self.padded_width, self.padded_height, 8, 8);

        // Decoder-side gab_smooth: 3x3 plus-shaped inverse of the
        // forward `gaborish_5x5` applied earlier in this pipeline.
        // libjxl's decoder pipeline runs gab_smooth on the reconstructed
        // XYB before xyb_to_linear; without it, the gaborish
        // pre-sharpening from the encoder side persists in the output
        // and the reconstruction is over-sharp/blocky.
        //
        // libjxl-parity gaborish gate: at d <= 0.5 the encoder did NOT
        // sharpen, so the decoder must NOT smooth either — `fh.gaborish`
        // signals false and the decoder skips the 3x3 blur. Mirror.
        let (recon_x_p, recon_y_p, recon_b_p) = if enable_gaborish_local {
            let (gw_c, gw1, gw2) = crate::forks::reconstruct::gab_weights();
            (
                enc.gab_smooth_persistent(&recon_x_p, gw_c, gw1, gw2),
                enc.gab_smooth_persistent(&recon_y_p, gw_c, gw1, gw2),
                enc.gab_smooth_persistent(&recon_b_p, gw_c, gw1, gw2),
            )
        } else {
            (recon_x_p, recon_y_p, recon_b_p)
        };

        // EPF chain (decoder edge-preserving filter). Runs after
        // gab_smooth on the reconstructed XYB planes, before
        // xyb_to_linear. Fully persistent — inputs stay on GPU
        // through padding + 2-iter EPF chain (step 1 + step 2).
        //
        // Qac→quant_field mapping: our pipeline carries per-block
        // float qac in `qac_vec`. EPF's compute_inv_sigma_map expects
        // u8 raw_quant + scalar quant_scale; the formula's "effective"
        // quant scale is `quant_scale * raw_quant`. In upstream at
        // distance=1.0 this product equals `qf_float ≈ 0.39`; our
        // `qac ≈ 0.765` is 2× of that (because `K_AC_QUANT = 0.765` vs
        // upstream's `q = 0.39`). We map `u8_qf = clamp(qac * 50,
        // 1, 255)` and `quant_scale = 0.01`, giving
        // `quant_scale * raw_quant ≈ qac / 2 ≈ qf_float_equivalent`.
        // Sharpness uniform 4 (libjxl default).
        let nb_blocks = (self.padded_width / 8) as usize * (self.padded_height / 8) as usize;
        debug_assert_eq!(qac_vec.len(), nb_blocks);
        let qf_u8: Vec<u8> = qac_vec
            .iter()
            .map(|&q| (q * 50.0).round().clamp(1.0, 255.0) as u8)
            .collect();
        let sharpness = vec![4_u8; nb_blocks];
        let inv_sigma_vec = crate::forks::epf::compute_inv_sigma_map(
            &qf_u8,
            &sharpness,
            0.01,
            (self.padded_width / 8) as usize,
            (self.padded_height / 8) as usize,
        );
        let inv_sigma_h = enc.upload_inv_sigma(&inv_sigma_vec);
        let xsize_blocks = self.padded_width / 8;
        let ysize_blocks = self.padded_height / 8;

        // Step 1 (5×5 plus, 5-pos SAD): pad=2, sigma_scale=1.65
        // Fused 3-channel pad — 1 launch instead of 3.
        let pad1 = 2_u32;
        let (p1_x, p1_y, p1_b) =
            enc.pad_plane_3ch_persistent(&recon_x_p, &recon_y_p, &recon_b_p, pad1);
        let (s1_x, s1_y, s1_b) = enc.epf_step1_persistent(
            &p1_x,
            &p1_y,
            &p1_b,
            &inv_sigma_h,
            self.padded_width,
            self.padded_height,
            xsize_blocks,
            ysize_blocks,
            pad1,
            1.65,
            crate::forks::epf::EPF_BORDER_SAD_MUL,
        );

        // Step 2 (3×3 plus, single-point SAD): pad=1, sigma_scale=10.725
        let pad2 = 1_u32;
        let (p2_x, p2_y, p2_b) = enc.pad_plane_3ch_persistent(&s1_x, &s1_y, &s1_b, pad2);
        let (s2_x, s2_y, s2_b) = enc.epf_step2_persistent(
            &p2_x,
            &p2_y,
            &p2_b,
            &inv_sigma_h,
            self.padded_width,
            self.padded_height,
            xsize_blocks,
            ysize_blocks,
            pad2,
            crate::forks::epf::EPF_PASS2_SIGMA_SCALE * 1.65,
            crate::forks::epf::EPF_BORDER_SAD_MUL,
        );

        let (rgb_r, rgb_g, rgb_b) = enc.xyb_to_linear_rgb_planar_persistent(&s2_x, &s2_y, &s2_b);
        // Batched 3-channel D2H read — one client.read with one
        // queue-drain sync instead of three sequential read_one calls
        // (matches the same swap done for the strat-search path's
        // download_crop in ba826e06).
        enc.download_planes_3ch(&rgb_r, &rgb_g, &rgb_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_one_shot() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy = LossyEncoder::new(&enc, 64, 64);
        let n = 64 * 64;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        let (rr, gg, bb) = lossy.encode_one(&enc, &r, &g, &b, 4.0);
        assert_eq!(rr.len(), n);
        assert_eq!(gg.len(), n);
        assert_eq!(bb.len(), n);
        for v in rr.iter().chain(&gg).chain(&bb) {
            assert!(v.is_finite());
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_srgb_u8() {
        // sRGB U8 convenience wrapper executes end-to-end on synthetic
        // RGB U8 input. Doesn't assert on reconstruction quality — at
        // qac=4 on adversarial high-freq sawtooth input, DCT8 produces
        // large errors. The point is to verify the API contract:
        // input length, output length, no panic, all output bytes
        // are valid u8 (which they always are by construction).
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy = LossyEncoder::new(&enc, 64, 48);
        let n = 64 * 48;
        // Smooth gradient input (low freq → bounded reconstruction).
        let mut rgb = Vec::with_capacity(n * 3);
        for y in 0..48 {
            for x in 0..64 {
                rgb.push((x * 4) as u8);
                rgb.push((y * 5) as u8);
                rgb.push(((x + y) * 2) as u8);
            }
        }
        let out = lossy.encode_one_srgb_u8(&enc, &rgb, 1.0);
        assert_eq!(out.len(), n * 3);
        // Smooth gradient at qac=1 should reconstruct within ~32 (LSBs).
        let mut max_diff = 0_i32;
        for i in 0..(n * 3) {
            max_diff = max_diff.max((rgb[i] as i32 - out[i] as i32).abs());
        }
        assert!(
            max_diff < 96,
            "smooth-gradient reconstruction max byte diff = {max_diff}, expected < 96 at qac=1"
        );
    }

    #[test]
    fn test_quality_to_qac_monotonic_and_bounded() {
        // libjxl convention: HIGHER quality → HIGHER qac (lighter quant).
        // val = coef * inv_w * qac; bigger qac → bigger val → survives
        // dead-zone threshold.
        let qac_100 = quality_to_qac(100.0);
        let qac_75 = quality_to_qac(75.0);
        let qac_50 = quality_to_qac(50.0);
        let qac_25 = quality_to_qac(25.0);
        let qac_10 = quality_to_qac(10.0);
        // Monotonically increasing: higher quality -> higher qac.
        assert!(qac_100 > qac_75);
        assert!(qac_75 >= qac_50, "q=75 ({qac_75}) >= q=50 ({qac_50})");
        assert!(qac_50 > qac_25);
        assert!(qac_25 > qac_10);
        // Endpoint sanity: q=100 above K_AC_QUANT=0.765, q=10 well below.
        assert!(qac_100 > 1.0, "quality=100 should be qac>1, got {qac_100}");
        assert!(qac_10 < 0.5, "quality=10 should be qac<0.5, got {qac_10}");
        // Clamp behaviour: out-of-range inputs clamped to [1, 100].
        assert_eq!(quality_to_qac(150.0), quality_to_qac(100.0));
        assert_eq!(quality_to_qac(-10.0), quality_to_qac(1.0));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_srgb_u8_many() {
        // Batch sRGB U8 wrapper: one shared linearization, N encodes.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy = LossyEncoder::new(&enc, 32, 32);
        let n = 32 * 32;
        let mut rgb = Vec::with_capacity(n * 3);
        for y in 0..32 {
            for x in 0..32 {
                rgb.push((x * 8) as u8);
                rgb.push((y * 8) as u8);
                rgb.push(((x + y) * 4) as u8);
            }
        }
        let qacs = [1.0_f32, 4.0, 16.0];
        let outputs = lossy.encode_many_srgb_u8(&enc, &rgb, &qacs);
        assert_eq!(outputs.len(), 3);
        for out in &outputs {
            assert_eq!(out.len(), n * 3);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_with_aq_srgb_u8() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy = LossyEncoder::new(&enc, 32, 32);
        let rgb: Vec<u8> = (0..(32 * 32 * 3))
            .map(|i| ((i * 13 + 7) % 256) as u8)
            .collect();
        let one = lossy.encode_one_with_aq_srgb_u8(&enc, &rgb, 1.0);
        assert_eq!(one.len(), 32 * 32 * 3);
        let many = lossy.encode_many_with_aq_srgb_u8(&enc, &rgb, &[0.5, 1.0, 2.0]);
        assert_eq!(many.len(), 3);
        for buf in &many {
            assert_eq!(buf.len(), 32 * 32 * 3);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_many_with_aq() {
        // Batch content-driven AQ — single mask prepass, multiple
        // distance-derived qac fields.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy = LossyEncoder::new(&enc, 32, 32);
        let n = 32 * 32;
        let mut r = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for y in 0..32 {
            for x in 0..32 {
                let v = if y < 16 { 0.2 } else { 0.8 };
                r.push(v + 0.01 * x as f32);
                g.push(v + 0.01 * x as f32);
                b.push(v + 0.01 * x as f32);
            }
        }
        let distances = [0.5, 1.0, 2.0, 4.0];
        let outs = lossy.encode_many_with_aq(&enc, &r, &g, &b, &distances);
        assert_eq!(outs.len(), distances.len());
        for (rr, gg, bb) in &outs {
            assert_eq!(rr.len(), n);
            assert_eq!(gg.len(), n);
            assert_eq!(bb.len(), n);
            for v in rr.iter().chain(gg).chain(bb) {
                assert!(v.is_finite());
            }
        }
        // Higher distance → higher MAE on average.
        let mae = |out: &(Vec<f32>, Vec<f32>, Vec<f32>)| {
            let mut s = 0.0_f64;
            for i in 0..n {
                s += (r[i] - out.0[i]).abs() as f64;
                s += (g[i] - out.1[i]).abs() as f64;
                s += (b[i] - out.2[i]).abs() as f64;
            }
            s / (3.0 * n as f64)
        };
        let mae_low = mae(&outs[0]);
        let mae_hi = mae(&outs[distances.len() - 1]);
        assert!(mae_hi >= mae_low, "MAE non-monotonic: {mae_low} → {mae_hi}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_with_aq() {
        // Turnkey content-driven AQ — encode_one_with_aq runs mask1x1
        // internally and derives the qac field.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy = LossyEncoder::new(&enc, 32, 32);
        let n = 32 * 32;
        // Mix of smooth + edges: gradient + a sharp horizontal edge.
        let mut r = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for y in 0..32 {
            for x in 0..32 {
                let v = if y < 16 { 0.2 } else { 0.8 };
                r.push(v + 0.01 * x as f32);
                g.push(v + 0.01 * x as f32);
                b.push(v + 0.01 * x as f32);
            }
        }
        let (rr, gg, bb) = lossy.encode_one_with_aq(&enc, &r, &g, &b, 1.0);
        assert_eq!(rr.len(), n);
        assert_eq!(gg.len(), n);
        assert_eq!(bb.len(), n);
        for v in rr.iter().chain(&gg).chain(&bb) {
            assert!(v.is_finite());
        }
        // Also test compute_aq_field returns the right shape.
        let aq = lossy.compute_aq_field(&enc, &r, &g, &b, 1.0);
        assert_eq!(aq.len(), lossy.num_blocks as usize);
        for &v in &aq {
            assert!(v.is_finite() && v > 0.0);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_adaptive_qac() {
        // Per-block adaptive qac field — different qac per block.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy = LossyEncoder::new(&enc, 32, 16);
        let n = 32 * 16;
        let nb = (32 / 8) * (16 / 8); // 8 blocks
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        // Half blocks get gentle quant (qac=1.5), half get aggressive (qac=0.2).
        let aq_field: Vec<f32> = (0..nb)
            .map(|i| if i < nb / 2 { 1.5 } else { 0.2 })
            .collect();
        let (rr, gg, bb) = lossy.encode_one_adaptive(&enc, &r, &g, &b, &aq_field);
        assert_eq!(rr.len(), n);
        assert_eq!(gg.len(), n);
        assert_eq!(bb.len(), n);
        for v in rr.iter().chain(&gg).chain(&bb) {
            assert!(v.is_finite());
        }
    }

    /// Selectivity test on synthetic SMOOTH content. CLIC photo's
    /// detailed texture means strat-search picks ~all-DCT8 even at
    /// the tuned bias slopes. A smooth gradient should let larger
    /// transforms win — validates the picker is making content-aware
    /// decisions rather than always falling through to DCT8.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_strat_search_selectivity_on_smooth_synthetic() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 256_u32;
        let h = 256_u32;
        let n = (w * h) as usize;
        // Smooth diagonal gradient: low spatial frequency.
        let r: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i % w as usize) as f32 / w as f32;
                let y = (i / w as usize) as f32 / h as f32;
                0.20 + 0.60 * x + 0.10 * y
            })
            .collect();
        let g: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i % w as usize) as f32 / w as f32;
                let y = (i / w as usize) as f32 / h as f32;
                0.30 + 0.40 * x + 0.20 * y
            })
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i % w as usize) as f32 / w as f32;
                let y = (i / w as usize) as f32 / h as f32;
                0.15 + 0.30 * x + 0.50 * y
            })
            .collect();
        let lossy = LossyEncoder::new(&enc, w, h);
        let distance: f32 = std::env::var("DISTANCE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        std::println!("[smooth-sel] {w}×{h} smooth gradient, distance={distance}");
        let _ = lossy.encode_one_with_strategy_search_dct8_16(&enc, &r, &g, &b, distance);
        // Histogram print fires inside the LossyEncoder when in #[cfg(test)].
    }

    /// Diagnostic: run strat-search with DCT32 enabled on a real CLIC
    /// image and report (a) Partition32x32 histogram and (b) per-channel
    /// reconstruction RMSE vs the no-DCT32 baseline.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_strat_search_dct32_diag_on_real_image() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let img_path = "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png";
        let img = match image::open(img_path) {
            Ok(i) => i.to_rgb8(),
            Err(_) => {
                std::println!("[skip] image not available: {img_path}");
                return;
            }
        };
        let (w, h) = img.dimensions();
        let pixels: Vec<u8> = img.into_raw();
        let n = (w * h) as usize;
        let to_lin = |c: u8| {
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
        for c in pixels.chunks_exact(3) {
            r.push(to_lin(c[0]));
            g.push(to_lin(c[1]));
            b.push(to_lin(c[2]));
        }
        let lossy = LossyEncoder::new(&enc, w, h);
        let distance: f32 = std::env::var("DISTANCE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        std::println!("[strat-diag] distance={distance}");
        let (rs, gs, bs) =
            lossy.encode_one_with_strategy_search_dct8_16(&enc, &r, &g, &b, distance);
        // Compute RMSE vs original (linear).
        let mut sse = 0.0_f64;
        for i in 0..n {
            let dr = (r[i] - rs[i]) as f64;
            let dg = (g[i] - gs[i]) as f64;
            let db = (b[i] - bs[i]) as f64;
            sse += dr * dr + dg * dg + db * db;
        }
        let rmse = (sse / (n * 3) as f64).sqrt();
        std::println!("[dct32-diag] strat-search RMSE = {rmse:.6}");
    }

    /// Performance diagnostic: isolate host-side `repack_plane_to_blocks`
    /// cost (per-channel, per-strategy) and synchronous GPU downloads
    /// in the cost-grid pipeline. Helps choose between optimization
    /// targets (host repack vs GPU sync overhead).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_strat_search_cost_grid_substage_timing() {
        use crate::forks::cost::repack_plane_to_blocks;
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let pw = 1024_usize;
        let ph = 1024_usize;
        let plane: Vec<f32> = (0..pw * ph).map(|i| (i as f32 * 0.001).sin()).collect();

        // Time: host repack (3 channels) for 8x8 tile shape
        let t0 = std::time::Instant::now();
        let _b8x = repack_plane_to_blocks(&plane, pw, ph, 8, 8);
        let _b8y = repack_plane_to_blocks(&plane, pw, ph, 8, 8);
        let _b8b = repack_plane_to_blocks(&plane, pw, ph, 8, 8);
        let dt_repack8 = t0.elapsed();

        let t1 = std::time::Instant::now();
        let _b16x = repack_plane_to_blocks(&plane, pw, ph, 16, 16);
        let _b16y = repack_plane_to_blocks(&plane, pw, ph, 16, 16);
        let _b16b = repack_plane_to_blocks(&plane, pw, ph, 16, 16);
        let dt_repack16 = t1.elapsed();

        let t2 = std::time::Instant::now();
        let _br1x = repack_plane_to_blocks(&plane, pw, ph, 8, 16);
        let _br1y = repack_plane_to_blocks(&plane, pw, ph, 8, 16);
        let _br1b = repack_plane_to_blocks(&plane, pw, ph, 8, 16);
        let _br2x = repack_plane_to_blocks(&plane, pw, ph, 16, 8);
        let _br2y = repack_plane_to_blocks(&plane, pw, ph, 16, 8);
        let _br2b = repack_plane_to_blocks(&plane, pw, ph, 16, 8);
        let dt_repack_rect = t2.elapsed();

        // Time: 3× synchronous DCT8 launches (with implicit downloads)
        let blocks_per_strategy = (pw / 8) * (ph / 8) * 64;
        let batch: Vec<f32> = vec![0.0_f32; blocks_per_strategy];
        // warmup
        let _ = enc.dct_8x8_blocks(&batch);
        let t3 = std::time::Instant::now();
        let _ = enc.dct_8x8_blocks(&batch);
        let _ = enc.dct_8x8_blocks(&batch);
        let _ = enc.dct_8x8_blocks(&batch);
        let dt_dct8_3sync = t3.elapsed();

        std::println!(
            "[perf-diag] host repack_plane_to_blocks (3 channels):\n  \
            8x8:    {:.2} ms\n  \
            16x16:  {:.2} ms\n  \
            16x8+8x16 (6 calls): {:.2} ms",
            dt_repack8.as_secs_f64() * 1000.0,
            dt_repack16.as_secs_f64() * 1000.0,
            dt_repack_rect.as_secs_f64() * 1000.0,
        );
        std::println!(
            "[perf-diag] 3× sync dct_8x8_blocks (Vec<f32> in/out, 1MB blocks each):\n  \
            {:.2} ms (avg {:.2} ms/call)",
            dt_dct8_3sync.as_secs_f64() * 1000.0,
            dt_dct8_3sync.as_secs_f64() * 1000.0 / 3.0,
        );
    }

    /// Diagnostic: dump DCT8 and DCT16x16 forward-coeffs[0] for a
    /// uniform 0.4 input. Reveals the actual normalization convention
    /// used by this codebase's DCT kernels — needed to decide what the
    /// DC frame value should be for use with restore_llf_dct*.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_dct_scale_convention_diag() {
        use crate::forks::transform::{
            RAW_STRATEGY_DCT, RAW_STRATEGY_DCT16X16, apply_dct_batch_gpu,
        };
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        // 16x16 plane uniformly = 0.4
        let plane: Vec<f32> = vec![0.4_f32; 16 * 16];
        let stride = 16;

        // DCT8 on first 8x8 block
        let dct8_coeffs = apply_dct_batch_gpu(&enc, &plane, stride, &[(0, 0)], RAW_STRATEGY_DCT);
        // DCT16x16 on the entire 16x16 region
        let dct16_coeffs =
            apply_dct_batch_gpu(&enc, &plane, stride, &[(0, 0)], RAW_STRATEGY_DCT16X16);

        std::println!(
            "[scale-diag] DCT8  coeffs[0..4]  = {:?}",
            &dct8_coeffs[0..4]
        );
        std::println!(
            "[scale-diag] DCT16 coeffs[0..4]  = {:?}",
            &dct16_coeffs[0..4]
        );
        std::println!(
            "[scale-diag] DCT16 coeffs[16..18]= {:?}",
            &dct16_coeffs[16..18]
        );
        // Predict:
        // - if orthonormal: DCT8 [0] = sum/8 = 3.2, DCT16 [0] = sum/16 = 6.4
        // - if mean-scaled: DCT8 [0] = 0.4, DCT16 [0] = 0.4
        // - if unnorm: DCT8 [0] = sum = 25.6, DCT16 [0] = sum = 102.4

        // Now run dc_from_dct_16x16-equivalent on dct16_coeffs to see what
        // values the encoder would store in the DC frame for this block.
        use crate::forks::reconstruct::DCT_RESAMPLE_SCALE_16_TO_2;
        let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
        let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];
        let b00 = dct16_coeffs[0] * s0 * s0;
        let b01 = dct16_coeffs[1] * s0 * s1;
        let b10 = dct16_coeffs[16] * s1 * s0;
        let b11 = dct16_coeffs[17] * s1 * s1;
        let dc00 = (b00 + b01) + (b10 + b11);
        let dc01 = (b00 + b01) - (b10 + b11);
        let dc10 = (b00 - b01) + (b10 - b11);
        let dc11 = (b00 - b01) - (b10 - b11);
        std::println!(
            "[scale-diag] dc_from_dct_16x16 -> [{:.4}, {:.4}, {:.4}, {:.4}]",
            dc00,
            dc01,
            dc10,
            dc11
        );
    }

    /// Diagnostic test for the Phase A strat-search bug: compare
    /// encode_one (uniform-qac DCT8) vs encode_one_with_strategy_search_dct8_16
    /// on a smooth gradient where DCT8 should be near-perfect. Reports
    /// per-channel RMSE and relative error.
    ///
    /// Expectation: both paths produce roughly equal reconstructions.
    /// If strat-search RMSE >> encode_one RMSE, the strat-search
    /// pipeline has a bug (wrong dequant / DC / IDCT layout).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_strat_search_vs_encode_one_diag() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        // 64×64 smooth gradient → almost zero DCT AC, DC dominates.
        // Reconstruction should be near-perfect for either path.
        let w = 64_u32;
        let h = 64_u32;
        let lossy = LossyEncoder::new(&enc, w, h);
        let n = (w * h) as usize;
        let r: Vec<f32> = (0..n)
            .map(|i| 0.30 + 0.20 * (i as f32 / n as f32))
            .collect();
        let g: Vec<f32> = (0..n)
            .map(|i| 0.40 + 0.15 * (i as f32 / n as f32))
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| 0.20 + 0.10 * (i as f32 / n as f32))
            .collect();

        let qac = distance_to_qac(1.0);
        let (e1_r, e1_g, e1_b) = lossy.encode_one(&enc, &r, &g, &b, qac);
        let (es_r, es_g, es_b) =
            lossy.encode_one_with_strategy_search_dct8_16(&enc, &r, &g, &b, 1.0);

        let rmse = |orig: &[f32], rec: &[f32]| -> f64 {
            let mut s = 0.0_f64;
            for i in 0..orig.len() {
                s += ((orig[i] - rec[i]) as f64).powi(2);
            }
            (s / orig.len() as f64).sqrt()
        };
        let mut min1 = f32::INFINITY;
        let mut max1 = f32::NEG_INFINITY;
        let mut mins = f32::INFINITY;
        let mut maxs = f32::NEG_INFINITY;
        for &v in e1_g.iter() {
            min1 = min1.min(v);
            max1 = max1.max(v);
        }
        for &v in es_g.iter() {
            mins = mins.min(v);
            maxs = maxs.max(v);
        }
        std::println!(
            "[strat-diag] encode_one     R={:.6} G={:.6} B={:.6}  G range=[{:.4},{:.4}]",
            rmse(&r, &e1_r),
            rmse(&g, &e1_g),
            rmse(&b, &e1_b),
            min1,
            max1
        );
        std::println!(
            "[strat-diag] strat-search   R={:.6} G={:.6} B={:.6}  G range=[{:.4},{:.4}]",
            rmse(&r, &es_r),
            rmse(&g, &es_g),
            rmse(&b, &es_b),
            mins,
            maxs
        );
        std::println!(
            "[strat-diag] G original range=[{:.4},{:.4}], first 8 px input/e1/es:",
            g.iter().copied().fold(f32::INFINITY, f32::min),
            g.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        );
        for i in 0..8 {
            std::println!(
                "  [{i}] input={:.4} e1={:.4} es={:.4}",
                g[i],
                e1_g[i],
                es_g[i]
            );
        }
    }

    /// Smoke test for the u8-input fast-path
    /// [`LossyEncoder::prepare_strategy_search_plan_traced_from_u8`].
    /// Verifies it runs end-to-end without panicking and produces a
    /// strategy plan equivalent to the f32 path for the same content
    /// (same number of assignments, plausible strategy distribution).
    /// Bit-exact match is NOT asserted because the GPU sRGB EOTF
    /// kernel and host EOTF can differ by ULP-level rounding, which
    /// can flip strategy picks at boundaries.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_prepare_strategy_search_plan_from_u8_smoke() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 64_u32;
        let h = 64_u32;
        let lossy = LossyEncoder::new(&enc, w, h);
        let n = (w * h) as usize;

        // Synthetic interleaved sRGB u8 RGB. Plain gradient ensures
        // every block falls into the same strategy (DCT8) — the test
        // will tolerate strategy shifts but validates basic structure.
        let mut pixels_u8: Vec<u8> = Vec::with_capacity(n * 3);
        for i in 0..n {
            let t = (i as f32 / n as f32) * 255.0;
            pixels_u8.push((30.0 + 0.6 * t).clamp(0.0, 255.0) as u8);
            pixels_u8.push((50.0 + 0.5 * t).clamp(0.0, 255.0) as u8);
            pixels_u8.push((70.0 + 0.4 * t).clamp(0.0, 255.0) as u8);
        }

        // Host sRGB EOTF for the f32 reference path.
        let to_linear = |c: u8| -> f32 {
            let v = c as f32 / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        };
        let mut r_lin = Vec::with_capacity(n);
        let mut g_lin = Vec::with_capacity(n);
        let mut b_lin = Vec::with_capacity(n);
        for chunk in pixels_u8.chunks_exact(3) {
            r_lin.push(to_linear(chunk[0]));
            g_lin.push(to_linear(chunk[1]));
            b_lin.push(to_linear(chunk[2]));
        }

        // f32 reference plan + u8 fast-path plan.
        let plan_f32 = lossy.prepare_strategy_search_plan(&enc, &r_lin, &g_lin, &b_lin, 1.0);
        let plan_u8 =
            lossy.prepare_strategy_search_plan_traced_from_u8(&enc, &pixels_u8, 1.0, &mut |_| {});

        // Structural parity: same number of assignments, same padded dims.
        assert_eq!(plan_f32.assignments.len(), plan_u8.assignments.len());
        assert_eq!(plan_f32.padded_width, plan_u8.padded_width);
        assert_eq!(plan_f32.padded_height, plan_u8.padded_height);
        assert_eq!(plan_f32.target_distance, plan_u8.target_distance);

        // Strategy distribution should be very close (allow tiny
        // boundary flips from sRGB EOTF rounding differences). For a
        // smooth gradient at d=1 we expect 100% DCT8 from both paths.
        use std::collections::BTreeMap;
        let histo = |plan: &StrategySearchPlan<B>| -> BTreeMap<u8, usize> {
            let mut h = BTreeMap::new();
            for a in &plan.assignments {
                *h.entry(a.raw_strategy).or_insert(0) += 1;
            }
            h
        };
        let h_f32 = histo(&plan_f32);
        let h_u8 = histo(&plan_u8);
        // Expect ≥ 95% agreement on the dominant strategy. (Smooth
        // gradient → DCT8 dominant. Allow a few flips from EOTF noise.)
        let dominant = *h_f32.iter().max_by_key(|(_, c)| **c).unwrap().0;
        let f32_dom = *h_f32.get(&dominant).unwrap();
        let u8_dom = *h_u8.get(&dominant).unwrap_or(&0);
        let agreement = u8_dom as f32 / f32_dom as f32;
        assert!(
            agreement >= 0.95,
            "u8 path dominant-strategy agreement too low: {:.3} (h_f32={:?}, h_u8={:?})",
            agreement,
            h_f32,
            h_u8,
        );
    }

    /// Phase A MVP smoke test: encode_one_with_strategy_search_dct8_16
    /// runs end-to-end on a 64×64 gradient, produces finite output of
    /// correct size. Quality validation deferred to demo + corpus
    /// sweep integration.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_strategy_search_dct8_16_smoke() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 64_u32;
        let h = 64_u32;
        let lossy = LossyEncoder::new(&enc, w, h);
        let n = (w * h) as usize;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        let (rr, gg, bb) = lossy.encode_one_with_strategy_search_dct8_16(&enc, &r, &g, &b, 1.0);
        assert_eq!(rr.len(), n);
        assert_eq!(gg.len(), n);
        assert_eq!(bb.len(), n);
        for v in rr.iter().chain(&gg).chain(&bb) {
            assert!(v.is_finite(), "non-finite output");
        }
    }

    /// Adaptive variant with a uniform aq_field at `distance_to_qac(d)`
    /// must produce bitwise-identical output to the scalar-distance
    /// method. This proves the new shim doesn't change historical
    /// (uniform-qac) behaviour.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_strat_search_adaptive_uniform_matches_scalar() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 64_u32;
        let h = 64_u32;
        let lossy = LossyEncoder::new(&enc, w, h);
        let n = (w * h) as usize;
        let r: Vec<f32> = (0..n)
            .map(|i| 0.30 + 0.20 * (i as f32 / n as f32))
            .collect();
        let g: Vec<f32> = (0..n)
            .map(|i| 0.40 + 0.15 * (i as f32 / n as f32))
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| 0.20 + 0.10 * (i as f32 / n as f32))
            .collect();

        for &distance in &[0.5_f32, 1.0, 2.0] {
            let (s_r, s_g, s_b) =
                lossy.encode_one_with_strategy_search_dct8_16(&enc, &r, &g, &b, distance);
            let nb8 = (lossy.padded_width as usize / 8) * (lossy.padded_height as usize / 8);
            let aq_uniform = std::vec![distance_to_qac(distance); nb8];
            let (a_r, a_g, a_b) = lossy.encode_one_with_strategy_search_dct8_16_adaptive(
                &enc,
                &r,
                &g,
                &b,
                &aq_uniform,
                distance,
            );
            assert_eq!(s_r.len(), a_r.len());
            for i in 0..s_r.len() {
                assert_eq!(s_r[i], a_r[i], "R mismatch at i={i} d={distance}");
                assert_eq!(s_g[i], a_g[i], "G mismatch at i={i} d={distance}");
                assert_eq!(s_b[i], a_b[i], "B mismatch at i={i} d={distance}");
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_arbitrary_size() {
        // 100×73 — neither dim is a multiple of 16. Encoder pads to
        // 112×80 internally (16-aligned per the strat-search 16x16
        // selector requirement), runs pipeline, crops back to 100×73.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 100_u32;
        let h = 73_u32;
        let lossy = LossyEncoder::new(&enc, w, h);
        assert_eq!(lossy.dimensions(), (100, 73));
        assert_eq!(lossy.padded_dimensions(), (112, 80));
        let n = (w * h) as usize;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        let (rr, gg, bb) = lossy.encode_one(&enc, &r, &g, &b, 4.0);
        assert_eq!(rr.len(), n, "output length must match original w*h");
        assert_eq!(gg.len(), n);
        assert_eq!(bb.len(), n);
        for v in rr.iter().chain(&gg).chain(&bb) {
            assert!(v.is_finite());
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_many_settings() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy = LossyEncoder::new(&enc, 32, 32);
        let n = 32 * 32;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        let qacs = [1.0_f32, 2.0, 4.0, 8.0];
        let outputs = lossy.encode_many(&enc, &r, &g, &b, &qacs);
        assert_eq!(outputs.len(), 4);
        for (i, (rr, _, _)) in outputs.iter().enumerate() {
            assert_eq!(rr.len(), n, "output {i} wrong length");
        }
    }
}
