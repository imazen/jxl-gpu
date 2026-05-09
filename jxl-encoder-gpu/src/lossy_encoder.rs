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

/// High-level lossy DCT8 encoder for a fixed image size. Pre-allocates
/// the static inputs (gaborish weights, dead-zone thresholds, unit
/// quant matrix) once at construction so they don't re-upload per call.
///
/// **Arbitrary input sizes are supported.** If `width` or `height` is
/// not a multiple of 8, the encoder pads the input on the right and
/// bottom with edge replication (matching libjxl), runs the pipeline
/// on the padded image, and crops the output back to the original
/// dimensions. The padding cost is per-encode (host-side memcpy at
/// upload time, ~4 KB extra per image at typical sizes).
///
/// Construct one per (width, height) to amortize the static-input
/// upload across many encodes.
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
fn pad_to_alignment(
    src: &[f32],
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
) -> alloc::vec::Vec<f32> {
    debug_assert_eq!(src.len(), width * height);
    if width == padded_width && height == padded_height {
        return src.to_vec();
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
    out
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
    /// Any width and height ≥ 8 are accepted; non-multiple-of-8
    /// dimensions are padded internally with edge replication.
    pub fn new(enc: &GpuEncoder<R>, width: u32, height: u32) -> Self {
        assert!(width >= 8 && height >= 8, "width/height must be at least 8");
        let padded_width = align_up(width, 8);
        let padded_height = align_up(height, 8);
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
        }
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
    #[allow(clippy::too_many_arguments)]
    pub fn encode_one_with_strategy_search_dct8_16_traced(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        distance: f32,
        mark: &mut dyn FnMut(&'static str),
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        use crate::forks::cost::{
            compute_scaled_constants, strategy_search_costs_dct16x8_or_8x16,
            strategy_search_costs_dct32x16_or_16x32, strategy_search_costs_dct32x32,
            strategy_search_costs_dct64x32_or_32x64, strategy_search_costs_dct64x64,
            strategy_search_costs_dct8_16x16, strategy_search_costs_subblock_8x8,
        };
        use crate::forks::reconstruct::{
            compute_dc_grid_per_8x8_block, encode_and_reconstruct_mixed_strategy_3channel,
            gab_weights,
        };
        use crate::forks::transform::{
            RAW_STRATEGY_DCT, RAW_STRATEGY_DCT16X16, RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT16X8,
            RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64,
            RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8, RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64,
            RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT8X4, RAW_STRATEGY_IDENTITY,
        };
        use crate::pipeline::{
            partitions_16x16_to_assignments, partitions_32x32_to_assignments,
            partitions_64x64_to_assignments, select_partitions_16x16_full,
            select_partitions_32x32_with_extras16, select_partitions_64x64_with_extras16,
            CostGrids16x16, CostGrids32x32, CostGrids64x64,
        };
        use crate::quant_weights::{
            dct16x16_weights_per_channel, dct16x32_weights_per_channel,
            dct16x8_weights_per_channel, dct2x2_weights_per_channel,
            dct32x32_weights_per_channel, dct32x64_weights_per_channel,
            dct4x4_weights_per_channel, dct4x8_weights_per_channel,
            dct64x64_weights_per_channel, dct8_weights_per_channel,
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
        let dct32_eligible = (self.padded_width as usize).is_multiple_of(32)
            && (self.padded_height as usize).is_multiple_of(32);
        let dct64_eligible = (self.padded_width as usize).is_multiple_of(64)
            && (self.padded_height as usize).is_multiple_of(64);

        let (w, h) = (self.width as usize, self.height as usize);
        let (pw, ph) = (self.padded_width as usize, self.padded_height as usize);
        let xb8 = pw / 8;
        let yb8 = ph / 8;
        let nb8 = xb8 * yb8;

        // Stage 1: pad + upload
        mark("start");
        let r_pad = pad_to_alignment(r, w, h, pw, ph);
        let g_pad = pad_to_alignment(g, w, h, pw, ph);
        let b_pad = pad_to_alignment(b, w, h, pw, ph);
        let g_r = enc.upload_plane(&r_pad, self.padded_width, self.padded_height);
        let g_g = enc.upload_plane(&g_pad, self.padded_width, self.padded_height);
        let g_b = enc.upload_plane(&b_pad, self.padded_width, self.padded_height);
        mark("pad_upload");

        // Stage 2: XYB + gaborish (GPU)
        let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(&g_r, &g_g, &g_b);
        let xx_g = enc.gaborish_5x5_persistent(&xx, &self.weights);
        let xy_g = enc.gaborish_5x5_persistent(&xy, &self.weights);
        let xb_g = enc.gaborish_5x5_persistent(&xb, &self.weights);

        // Stage 3: mask1x1 from Y channel — keep on GPU (cost grids
        // consume it as a GpuPlane), and download to host so the
        // existing host-repack cost-grid path still has spatial XYB.
        let g_mask = enc.mask1x1_persistent(&xy_g);
        let xyb_x: Vec<f32> = enc.download_plane(&xx_g);
        let xyb_y: Vec<f32> = enc.download_plane(&xy_g);
        let xyb_b: Vec<f32> = enc.download_plane(&xb_g);
        mark("xyb_gab");
        mark("mask1x1");

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

        let (mut cost_dct8, mut cost_dct16) = strategy_search_costs_dct8_16x16(
            enc,
            &xyb_x,
            &xyb_y,
            &xyb_b,
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
        let bias_scale = (distance - 1.0).max(0.0) * 0.3;
        let dist_bias = 1.0 + bias_scale;
        for c in cost_dct16.iter_mut() {
            *c *= dist_bias;
        }

        // 8x8 sub-block strategies (DCT4x4, DCT4x8, DCT8x4, IDENTITY,
        // DCT2x2). All extract 8x8 tiles → 64 coefs. We pre-gather
        // 8x8 blocks ONCE, upload to GPU, then run all 5 cost-grid
        // producers against the same GpuBlocks.
        let bx8_full = crate::forks::cost::repack_plane_to_blocks(&xyb_x, pw, ph, 8, 8);
        let by8_full = crate::forks::cost::repack_plane_to_blocks(&xyb_y, pw, ph, 8, 8);
        let bb8_full = crate::forks::cost::repack_plane_to_blocks(&xyb_b, pw, ph, 8, 8);
        let g_8x = enc.upload_blocks(&bx8_full, (xb8 * yb8) as u32, 64);
        let g_8y = enc.upload_blocks(&by8_full, (xb8 * yb8) as u32, 64);
        let g_8b = enc.upload_blocks(&bb8_full, (xb8 * yb8) as u32, 64);

        let (dct4x4_x, dct4x4_y, dct4x4_b) = dct4x4_weights_per_channel();
        let inv_4x4_x: Vec<f32> = dct4x4_x.iter().map(|w| 1.0 / w).collect();
        let inv_4x4_y: Vec<f32> = dct4x4_y.iter().map(|w| 1.0 / w).collect();
        let inv_4x4_b: Vec<f32> = dct4x4_b.iter().map(|w| 1.0 / w).collect();
        // Empirical anti-bias mul: libjxl reference is 1.08, but feeding
        // that into our cost model picks DCT4x4 too often, causing
        // visible block-edge artifacts that L2 doesn't see. Bump 2× as
        // a starting point — same trick used to make DCT32 work.
        let cost_dct4x4 = strategy_search_costs_subblock_8x8(
            enc, &g_8x, &g_8y, &g_8b, pw, ph, &g_mask, RAW_STRATEGY_DCT4X4,
            &dct4x4_x, &dct4x4_y, &dct4x4_b,
            &inv_4x4_x, &inv_4x4_y, &inv_4x4_b,
            qac, qac, qac, 0, 0, scaled_constants, 2.16,
        );

        let (dct4x8_x, dct4x8_y, dct4x8_b) = dct4x8_weights_per_channel();
        let inv_4x8_x: Vec<f32> = dct4x8_x.iter().map(|w| 1.0 / w).collect();
        let inv_4x8_y: Vec<f32> = dct4x8_y.iter().map(|w| 1.0 / w).collect();
        let inv_4x8_b: Vec<f32> = dct4x8_b.iter().map(|w| 1.0 / w).collect();
        // libjxl reference is 0.859316; bump 2× for the same anti-bias reason.
        let cost_dct4x8 = strategy_search_costs_subblock_8x8(
            enc, &g_8x, &g_8y, &g_8b, pw, ph, &g_mask, RAW_STRATEGY_DCT4X8,
            &dct4x8_x, &dct4x8_y, &dct4x8_b,
            &inv_4x8_x, &inv_4x8_y, &inv_4x8_b,
            qac, qac, qac, 0, 0, scaled_constants, 1.72,
        );
        let cost_dct8x4 = strategy_search_costs_subblock_8x8(
            enc, &g_8x, &g_8y, &g_8b, pw, ph, &g_mask, RAW_STRATEGY_DCT8X4,
            &dct4x8_x, &dct4x8_y, &dct4x8_b,
            &inv_4x8_x, &inv_4x8_y, &inv_4x8_b,
            qac, qac, qac, 0, 0, scaled_constants, 1.72,
        );

        // IDENTITY: libjxl reference 1.0428, bumped 2× for anti-bias
        // (visible blocking on detailed content if under-penalized).
        let (id_x, id_y, id_b) = identity_weights_per_channel();
        let inv_id_x: Vec<f32> = id_x.iter().map(|w| 1.0 / w).collect();
        let inv_id_y: Vec<f32> = id_y.iter().map(|w| 1.0 / w).collect();
        let inv_id_b: Vec<f32> = id_b.iter().map(|w| 1.0 / w).collect();
        let cost_identity = strategy_search_costs_subblock_8x8(
            enc, &g_8x, &g_8y, &g_8b, pw, ph, &g_mask, RAW_STRATEGY_IDENTITY,
            &id_x, &id_y, &id_b,
            &inv_id_x, &inv_id_y, &inv_id_b,
            qac, qac, qac, 0, 0, scaled_constants, 2.09,
        );

        // DCT2X2: libjxl reference 0.95, bumped 2× for anti-bias.
        let (d2_x, d2_y, d2_b) = dct2x2_weights_per_channel();
        let inv_d2_x: Vec<f32> = d2_x.iter().map(|w| 1.0 / w).collect();
        let inv_d2_y: Vec<f32> = d2_y.iter().map(|w| 1.0 / w).collect();
        let inv_d2_b: Vec<f32> = d2_b.iter().map(|w| 1.0 / w).collect();
        let cost_dct2x2 = strategy_search_costs_subblock_8x8(
            enc, &g_8x, &g_8y, &g_8b, pw, ph, &g_mask, RAW_STRATEGY_DCT2X2,
            &d2_x, &d2_y, &d2_b,
            &inv_d2_x, &inv_d2_y, &inv_d2_b,
            qac, qac, qac, 0, 0, scaled_constants, 1.90,
        );
        mark("cost_subblock_8x8");

        // AFV0-3 cost grid: SKIPPED. With the 100× anti-bias needed to
        // keep AFV from over-selecting on detailed content, AFV
        // practically never wins on real photos. Computing the cost
        // grid (~175 ms on 1024×1024) is wasted work for the rare
        // case it would influence picks. Plumbing kept for re-enable
        // once persistent AFV transforms (task #38) cut the cost to
        // ~30 ms — at which point a lower anti-bias may also be tried.
        let cost_afv0: Vec<f32> = Vec::new();
        let cost_afv1: Vec<f32> = Vec::new();
        let cost_afv2: Vec<f32> = Vec::new();
        let cost_afv3: Vec<f32> = Vec::new();
        mark("cost_afv");

        // Distance-scaled anti-bias for sub-blocks (same scale as DCT16).
        let mut cost_dct4x4 = cost_dct4x4;
        let mut cost_dct4x8 = cost_dct4x8;
        let mut cost_dct8x4 = cost_dct8x4;
        let mut cost_identity = cost_identity;
        let mut cost_dct2x2 = cost_dct2x2;
        for c in cost_dct4x4.iter_mut() { *c *= dist_bias; }
        for c in cost_dct4x8.iter_mut() { *c *= dist_bias; }
        for c in cost_dct8x4.iter_mut() { *c *= dist_bias; }
        for c in cost_identity.iter_mut() { *c *= dist_bias; }
        for c in cost_dct2x2.iter_mut() { *c *= dist_bias; }

        let cost_dct16x8 = strategy_search_costs_dct16x8_or_8x16(
            enc,
            &xyb_x,
            &xyb_y,
            &xyb_b,
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
        );
        mark("cost_dct16x8");
        let cost_dct8x16 = strategy_search_costs_dct16x8_or_8x16(
            enc,
            &xyb_x,
            &xyb_y,
            &xyb_b,
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
        );

        mark("cost_dct8x16");
        // Distance-scaled anti-bias (same as DCT16x16).
        let mut cost_dct16x8 = cost_dct16x8;
        let mut cost_dct8x16 = cost_dct8x16;
        for c in cost_dct16x8.iter_mut() { *c *= dist_bias; }
        for c in cost_dct8x16.iter_mut() { *c *= dist_bias; }

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
            strategy_search_costs_dct32x32(
                enc,
                &xyb_x,
                &xyb_y,
                &xyb_b,
                pw,
                ph,
                &g_mask,
                &dct32_x,
                &dct32_y,
                &dct32_b,
                &inv_32x,
                &inv_32y,
                &inv_32b,
                qac,
                qac,
                qac,
                0,
                0,
                scaled_constants,
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
        for c in cost_dct32x32.iter_mut() { *c *= dist_bias_32; }

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
            let c_32x16 = strategy_search_costs_dct32x16_or_16x32(
                enc, &xyb_x, &xyb_y, &xyb_b, pw, ph, &g_mask, RAW_STRATEGY_DCT32X16,
                &dct32x16_x, &dct32x16_y, &dct32x16_b,
                &inv_32x16_x, &inv_32x16_y, &inv_32x16_b,
                qac, qac, qac, 0, 0, scaled_constants,
            );
            let c_16x32 = strategy_search_costs_dct32x16_or_16x32(
                enc, &xyb_x, &xyb_y, &xyb_b, pw, ph, &g_mask, RAW_STRATEGY_DCT16X32,
                &dct32x16_x, &dct32x16_y, &dct32x16_b,
                &inv_32x16_x, &inv_32x16_y, &inv_32x16_b,
                qac, qac, qac, 0, 0, scaled_constants,
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
        for c in cost_dct32x16.iter_mut() { *c *= dist_bias_32; }
        for c in cost_dct16x32.iter_mut() { *c *= dist_bias_32; }

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
            let c64 = strategy_search_costs_dct64x64(
                enc, &xyb_x, &xyb_y, &xyb_b, pw, ph, &g_mask,
                &dct64_x, &dct64_y, &dct64_b,
                &inv_64x, &inv_64y, &inv_64b,
                qac, qac, qac, 0, 0, scaled_constants,
            );
            let c64x32 = strategy_search_costs_dct64x32_or_32x64(
                enc, &xyb_x, &xyb_y, &xyb_b, pw, ph, &g_mask, RAW_STRATEGY_DCT64X32,
                &dct64x32_x, &dct64x32_y, &dct64x32_b,
                &inv_64x32_x, &inv_64x32_y, &inv_64x32_b,
                qac, qac, qac, 0, 0, scaled_constants,
            );
            let c32x64 = strategy_search_costs_dct64x32_or_32x64(
                enc, &xyb_x, &xyb_y, &xyb_b, pw, ph, &g_mask, RAW_STRATEGY_DCT32X64,
                &dct64x32_x, &dct64x32_y, &dct64x32_b,
                &inv_64x32_x, &inv_64x32_y, &inv_64x32_b,
                qac, qac, qac, 0, 0, scaled_constants,
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
        for c in cost_dct64x64.iter_mut() { *c *= dist_bias_64; }
        for c in cost_dct64x32.iter_mut() { *c *= dist_bias_64; }
        for c in cost_dct32x64.iter_mut() { *c *= dist_bias_64; }

        // Stage 5: host-side selector + assignments. All 5 sub-block
        // strategies feed in with anti-bias entropy_muls (2× the libjxl
        // reference) — same trick as DCT32 needed.
        // AFV cost grids skipped (None) — see the cost-grid call site
        // above. Plumbing kept for re-enable post #38.
        let _ = (&cost_afv0, &cost_afv1, &cost_afv2, &cost_afv3);
        let sub_blocks = crate::pipeline::SubBlockCostGrids {
            dct4x4: Some(&cost_dct4x4),
            dct4x8: Some(&cost_dct4x8),
            dct8x4: Some(&cost_dct8x4),
            identity: Some(&cost_identity),
            dct2x2: Some(&cost_dct2x2),
            afv0: None,
            afv1: None,
            afv2: None,
            afv3: None,
        };
        let extra16 = CostGrids16x16 {
            dct_16x8: Some(&cost_dct16x8),
            dct_8x16: Some(&cost_dct8x16),
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

        // Stage 6: per-channel DC grids
        let dc_grid_x = compute_dc_grid_per_8x8_block(&xyb_x, pw, ph);
        let dc_grid_y = compute_dc_grid_per_8x8_block(&xyb_y, pw, ph);
        let dc_grid_b = compute_dc_grid_per_8x8_block(&xyb_b, pw, ph);
        mark("dc_grids");

        // Stage 7: encode + reconstruct via mixed-strategy IDCT.
        // Strategies supported: DCT8, DCT16x16, DCT16x8, DCT8x16, DCT32x32.
        let qac_vec = vec![qac; nb8];
        let dct8_x_clone = dct8_x;
        let dct8_y_clone = dct8_y;
        let dct8_b_clone = dct8_b;
        let dct16_x_clone = dct16_x.clone();
        let dct16_y_clone = dct16_y.clone();
        let dct16_b_clone = dct16_b.clone();
        let dct16x8_x_clone = dct16x8_x.clone();
        let dct16x8_y_clone = dct16x8_y.clone();
        let dct16x8_b_clone = dct16x8_b.clone();
        let dct32_x_clone = dct32_x.clone();
        let dct32_y_clone = dct32_y.clone();
        let dct32_b_clone = dct32_b.clone();
        let dct32x16_x_clone = dct32x16_x.clone();
        let dct32x16_y_clone = dct32x16_y.clone();
        let dct32x16_b_clone = dct32x16_b.clone();
        let dct64_x_clone = dct64_x.clone();
        let dct64_y_clone = dct64_y.clone();
        let dct64_b_clone = dct64_b.clone();
        let dct64x32_x_clone = dct64x32_x.clone();
        let dct64x32_y_clone = dct64x32_y.clone();
        let dct64x32_b_clone = dct64x32_b.clone();
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
                _ => panic!("Phase B strategy {strat} not yet wired into encoder"),
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
                _ => panic!("Phase B strategy {strat} not yet wired into encoder"),
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
                _ => panic!("Phase B strategy {strat} not yet wired into encoder"),
            }
        };
        let mut plane_x = vec![0.0_f32; pw * ph];
        let mut plane_y = vec![0.0_f32; pw * ph];
        let mut plane_b = vec![0.0_f32; pw * ph];
        encode_and_reconstruct_mixed_strategy_3channel(
            enc,
            &xyb_x,
            &xyb_y,
            &xyb_b,
            pw,
            ph,
            &assignments,
            &weights_x_for,
            &weights_y_for,
            &weights_b_for,
            &qac_vec,
            &qac_vec,
            &qac_vec,
            &self.thresholds_x,
            &self.thresholds_y,
            &self.thresholds_b,
            &dc_grid_x,
            &dc_grid_y,
            &dc_grid_b,
            &mut plane_x,
            &mut plane_y,
            &mut plane_b,
        );
        mark("mixed_strategy_encode_recon");


        // Stage 8: postpass (gab_smooth + EPF + xyb_to_linear), matching
        // run_pipeline_with_qac. EPF closes most of the perceptual gap
        // vs the uniform-qac DCT8 baseline.
        let recon_x_p =
            enc.upload_plane(&plane_x, self.padded_width, self.padded_height);
        let recon_y_p =
            enc.upload_plane(&plane_y, self.padded_width, self.padded_height);
        let recon_b_p =
            enc.upload_plane(&plane_b, self.padded_width, self.padded_height);
        let (gw_c, gw1, gw2) = gab_weights();
        let recon_x_p = enc.gab_smooth_persistent(&recon_x_p, gw_c, gw1, gw2);
        let recon_y_p = enc.gab_smooth_persistent(&recon_y_p, gw_c, gw1, gw2);
        let recon_b_p = enc.gab_smooth_persistent(&recon_b_p, gw_c, gw1, gw2);

        // EPF step 1+2 (matches run_pipeline_with_qac). Per-block qac maps
        // to u8 quant_field via `clamp(qac * 50, 1, 255)`; sharpness is
        // uniform 4 (libjxl default).
        let qf_u8: Vec<u8> = qac_vec
            .iter()
            .map(|&q| (q * 50.0).round().clamp(1.0, 255.0) as u8)
            .collect();
        let sharpness = vec![4_u8; nb8];
        let inv_sigma_vec = crate::forks::epf::compute_inv_sigma_map(
            &qf_u8,
            &sharpness,
            0.01,
            xb8,
            yb8,
        );
        let inv_sigma_h = enc.upload_inv_sigma(&inv_sigma_vec);
        let xsize_blocks = self.padded_width / 8;
        let ysize_blocks = self.padded_height / 8;

        let pad1 = 2_u32;
        let p1_x = enc.pad_plane_persistent(&recon_x_p, pad1);
        let p1_y = enc.pad_plane_persistent(&recon_y_p, pad1);
        let p1_b = enc.pad_plane_persistent(&recon_b_p, pad1);
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
        let p2_x = enc.pad_plane_persistent(&s1_x, pad2);
        let p2_y = enc.pad_plane_persistent(&s1_y, pad2);
        let p2_b = enc.pad_plane_persistent(&s1_b, pad2);
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

        let (rgb_r, rgb_g, rgb_b) =
            enc.xyb_to_linear_rgb_planar_persistent(&s2_x, &s2_y, &s2_b);
        mark("postpass_gab_epf_xyb");
        let r_out = enc.download_plane(&rgb_r);
        let g_out = enc.download_plane(&rgb_g);
        let b_out = enc.download_plane(&rgb_b);
        mark("download_crop");

        (
            crop_to_original(&r_out, pw, w, h),
            crop_to_original(&g_out, pw, w, h),
            crop_to_original(&b_out, pw, w, h),
        )
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

        let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(g_r, g_g, g_b);
        let xx_g = enc.gaborish_5x5_persistent(&xx, &self.weights);
        let xy_g = enc.gaborish_5x5_persistent(&xy, &self.weights);
        let xb_g = enc.gaborish_5x5_persistent(&xb, &self.weights);
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
        let (gw_c, gw1, gw2) = crate::forks::reconstruct::gab_weights();
        let recon_x_p = enc.gab_smooth_persistent(&recon_x_p, gw_c, gw1, gw2);
        let recon_y_p = enc.gab_smooth_persistent(&recon_y_p, gw_c, gw1, gw2);
        let recon_b_p = enc.gab_smooth_persistent(&recon_b_p, gw_c, gw1, gw2);

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
        let nb_blocks = (self.padded_width / 8) as usize
            * (self.padded_height / 8) as usize;
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
        let pad1 = 2_u32;
        let p1_x = enc.pad_plane_persistent(&recon_x_p, pad1);
        let p1_y = enc.pad_plane_persistent(&recon_y_p, pad1);
        let p1_b = enc.pad_plane_persistent(&recon_b_p, pad1);
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
        let p2_x = enc.pad_plane_persistent(&s1_x, pad2);
        let p2_y = enc.pad_plane_persistent(&s1_y, pad2);
        let p2_b = enc.pad_plane_persistent(&s1_b, pad2);
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

        let (rgb_r, rgb_g, rgb_b) =
            enc.xyb_to_linear_rgb_planar_persistent(&s2_x, &s2_y, &s2_b);
        (
            enc.download_plane(&rgb_r),
            enc.download_plane(&rgb_g),
            enc.download_plane(&rgb_b),
        )
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
            if f <= 0.04045 { f / 12.92 } else { ((f + 0.055) / 1.055).powf(2.4) }
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
            apply_dct_batch_gpu, RAW_STRATEGY_DCT, RAW_STRATEGY_DCT16X16,
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

        std::println!("[scale-diag] DCT8  coeffs[0..4]  = {:?}", &dct8_coeffs[0..4]);
        std::println!("[scale-diag] DCT16 coeffs[0..4]  = {:?}", &dct16_coeffs[0..4]);
        std::println!("[scale-diag] DCT16 coeffs[16..18]= {:?}", &dct16_coeffs[16..18]);
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
            dc00, dc01, dc10, dc11
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
        let r: Vec<f32> = (0..n).map(|i| 0.30 + 0.20 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.40 + 0.15 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.20 + 0.10 * (i as f32 / n as f32)).collect();

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
            rmse(&r, &e1_r), rmse(&g, &e1_g), rmse(&b, &e1_b), min1, max1
        );
        std::println!(
            "[strat-diag] strat-search   R={:.6} G={:.6} B={:.6}  G range=[{:.4},{:.4}]",
            rmse(&r, &es_r), rmse(&g, &es_g), rmse(&b, &es_b), mins, maxs
        );
        std::println!(
            "[strat-diag] G original range=[{:.4},{:.4}], first 8 px input/e1/es:",
            g.iter().copied().fold(f32::INFINITY, f32::min),
            g.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        );
        for i in 0..8 {
            std::println!("  [{i}] input={:.4} e1={:.4} es={:.4}", g[i], e1_g[i], es_g[i]);
        }
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

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_arbitrary_size() {
        // 100×73 — neither dim is a multiple of 8. Encoder pads to
        // 104×80 internally, runs pipeline, crops back to 100×73.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 100_u32;
        let h = 73_u32;
        let lossy = LossyEncoder::new(&enc, w, h);
        assert_eq!(lossy.dimensions(), (100, 73));
        assert_eq!(lossy.padded_dimensions(), (104, 80));
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
