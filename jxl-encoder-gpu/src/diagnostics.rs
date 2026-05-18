// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Diagnostic thread-local sinks. `#[doc(hidden)]` — instrumentation only,
//! not part of the stable API.
//!
//! ## Why
//!
//! Some encode-pipeline decisions (notably the patches case-1 AC strategy
//! recompute in [`crate::encoder::GpuEncoder::encode_lossy_to_bitstream_via_precomputed`])
//! discard GPU-side state and re-derive it on the host. We want to A/B
//! quantify what was discarded vs preserved without baking the
//! instrumentation into a hot path or threading a `&mut Stats` parameter
//! through 8 layers of function signatures.
//!
//! Pattern mirrors `jxl_encoder::__pre_quantized::take_last_patches_stats`:
//! a thread-local `Cell<Option<Stats>>` is `set_*` from inside the
//! encoder and `take_*` by an external diagnostic example. The release
//! cost is one branchless `Cell::set` per encode — well under any noise
//! floor.

#![allow(dead_code)]

// This module is `#[cfg(feature = "encoder")]`-gated in `lib.rs`; the
// `encoder` feature pulls `jxl-encoder` with `std`, so `std::thread_local`
// is available even though the crate is `no_std`-by-default.
extern crate std;

use core::cell::Cell;
use std::thread_local;

/// Per-encode snapshot of GPU AFV picks vs CPU recompute AFV picks
/// across the patches case-1 path.
///
/// Cleared (consumed) by [`take_last_afv_preservation_stats`]; if the
/// caller never reads it, it stays as the value the last encode wrote.
/// Always reflects the most recent
/// `encode_lossy_to_bitstream_via_precomputed*` invocation on the
/// current thread.
///
/// Captures only **histogram-level deltas** (per-class first-block
/// counts before/after the recompute). Per-block transition counts and
/// position lists would require an `AcStrategyMap` snapshot, which the
/// pre-quantized API does not currently export a way to clone (the
/// `data: Vec<u8>` field is private and there is no `pub fn clone`
/// surface). Adding that to jxl-encoder is a follow-on chunk; see
/// `CHANGELOG.md` under "AFV-across-patches preservation diagnostic
/// (chunk 1)".
#[derive(Clone, Debug)]
pub struct LastAfvPreservationStats {
    /// Whether the patches case-1 recompute fired (patches detected and
    /// admitted). When `false`, `gpu_afv_picks_pre` equals
    /// `cpu_afv_picks_post` and no GPU picks were wiped — the recompute
    /// is the only known site that discards them.
    pub patches_recompute_fired: bool,

    /// Total blocks the strategy map covers (`xsize_blocks * ysize_blocks`).
    pub total_blocks: u32,

    /// Count of AFV0-3 first-blocks the GPU strategy plan produced
    /// BEFORE the patches case-1 recompute. Mirrors
    /// `AcStrategyMap::strategy_histogram()[12..=15].iter().sum()`.
    pub gpu_afv_picks_pre: u32,

    /// AFV count AFTER the CPU `compute_ac_strategy` recompute on
    /// patches-subtracted XYB. When the recompute fires, this is the
    /// CPU's independent AFV count on the post-patches XYB; when it
    /// does not, this equals `gpu_afv_picks_pre`.
    pub cpu_afv_picks_post: u32,

    /// DCT8 first-block count BEFORE the recompute.
    pub gpu_dct8_picks_pre: u32,

    /// DCT8 first-block count AFTER the recompute. The delta
    /// `cpu_dct8_picks_post - gpu_dct8_picks_pre` is an upper bound on
    /// the number of GPU AFV picks that the CPU recompute converted to
    /// DCT8 (the bulk of `afv_to_dct8` transitions when content-class
    /// AFV→DCT4x4/DCT2x2 transitions are negligible — confirmed for
    /// screenshots where AFV ≈ DCT8 are the only two pickable classes
    /// for the touched blocks).
    pub cpu_dct8_picks_post: u32,

    /// Full first-block histogram BEFORE the recompute. Indexed by
    /// `raw_strategy` (0..=18). AFV is 12..=15.
    pub gpu_strategy_histogram_pre: [u32; 19],

    /// Full first-block histogram AFTER the recompute.
    pub cpu_strategy_histogram_post: [u32; 19],
}

thread_local! {
    static LAST_AFV_PRESERVATION_STATS: Cell<Option<LastAfvPreservationStats>> =
        const { Cell::new(None) };
}

/// Set the per-thread AFV-preservation snapshot. Called inside the
/// GPU encoder's patches case-1 path after the recompute completes.
#[doc(hidden)]
pub fn set_last_afv_preservation_stats(stats: LastAfvPreservationStats) {
    LAST_AFV_PRESERVATION_STATS.with(|c| c.set(Some(stats)));
}

/// Consume and return the most recent per-thread AFV-preservation
/// snapshot. Returns `None` if no encode has set the snapshot since
/// the last call (or since thread start).
///
/// Calibration / instrumentation hook only — `#[doc(hidden)]`, not
/// part of the stable API.
#[doc(hidden)]
pub fn take_last_afv_preservation_stats() -> Option<LastAfvPreservationStats> {
    LAST_AFV_PRESERVATION_STATS.with(|c| c.take())
}

/// Compute and stash the AFV-preservation snapshot from a paired GPU
/// pre-recompute / CPU post-recompute strategy map. Called from the
/// patches case-1 path in
/// [`crate::encoder::GpuEncoder::encode_lossy_to_bitstream_via_precomputed`]
/// (and its `_with_butteraugli` / `_from_u8` siblings) right after the
/// `compute_ac_strategy` reassignment.
///
/// `pre`: the GPU strategy plan BEFORE the recompute (still carries
/// any AFV picks the GPU cost-grid produced).
/// `post`: the CPU recompute result on patches-subtracted XYB (the
/// value that flows into [`jxl_encoder::__pre_quantized::EncoderPrecomputed`]).
/// `patches_fired`: `true` when the patches detector returned `Some`
/// AND we ran the recompute (single source of truth — caller only
/// invokes this helper when both are true; the `false` arm is for
/// completeness — caller may stash a no-op snapshot at the top of the
/// encode path so the "no patches" case is observable too).
///
/// AFV raw-strategy codes 12-15 (AFV0-3) are baked in here — matches
/// `jxl_encoder::vardct::ac_strategy::RAW_STRATEGY_AFV0..=RAW_STRATEGY_AFV3`.
/// DCT8 is raw-strategy 0.
#[doc(hidden)]
pub fn record_afv_preservation_diff(
    pre_hist: [u32; 19],
    post_hist: [u32; 19],
    total_blocks: u32,
) {
    let gpu_afv_picks_pre: u32 = pre_hist[12..=15].iter().sum();
    let cpu_afv_picks_post: u32 = post_hist[12..=15].iter().sum();
    let stats = LastAfvPreservationStats {
        patches_recompute_fired: true,
        total_blocks,
        gpu_afv_picks_pre,
        cpu_afv_picks_post,
        gpu_dct8_picks_pre: pre_hist[0],
        cpu_dct8_picks_post: post_hist[0],
        gpu_strategy_histogram_pre: pre_hist,
        cpu_strategy_histogram_post: post_hist,
    };
    set_last_afv_preservation_stats(stats);
}

/// Stash a "no patches" snapshot at encode entry — the caller passes
/// the GPU strategy plan; if the patches recompute never fires, this
/// remains the final snapshot (`patches_recompute_fired = false`,
/// `cpu_afv_picks_post = gpu_afv_picks_pre`).
///
/// Cheap (one `Cell::set`, no allocations) so it can run
/// unconditionally before every encode without measurable cost.
#[doc(hidden)]
pub fn record_afv_preservation_baseline(pre: &jxl_encoder::__pre_quantized::AcStrategyMap) {
    let pre_hist = pre.strategy_histogram();
    let gpu_afv_picks_pre: u32 = pre_hist[12..=15].iter().sum();
    let stats = LastAfvPreservationStats {
        patches_recompute_fired: false,
        total_blocks: (pre.xsize_blocks * pre.ysize_blocks) as u32,
        gpu_afv_picks_pre,
        cpu_afv_picks_post: gpu_afv_picks_pre,
        gpu_dct8_picks_pre: pre_hist[0],
        cpu_dct8_picks_post: pre_hist[0],
        gpu_strategy_histogram_pre: pre_hist,
        cpu_strategy_histogram_post: pre_hist,
    };
    set_last_afv_preservation_stats(stats);
}
