// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! High-level GPU lossy encoder facade.
//!
//! Wraps the [`crate::persistent`] API into four user-friendly entry
//! points and a JPEG-style quality knob:
//!
//! | Method | Input | Output | Use case |
//! |---|---|---|---|
//! | [`LossyEncoder::encode_one`] | `f32` planar | `(R, G, B)` `f32` | one-shot, already-linear |
//! | [`LossyEncoder::encode_many`] | `f32` planar | `Vec<(R, G, B)>` | quality sweep, already-linear |
//! | [`LossyEncoder::encode_one_srgb_u8`] | sRGB `u8` interleaved | sRGB `u8` interleaved | one-shot, image-crate input |
//! | [`LossyEncoder::encode_many_srgb_u8`] | sRGB `u8` interleaved | `Vec<u8>` per setting | quality sweep, image-crate input |
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
//! - DCT8 only (one strategy). The 13 strategies are available
//!   individually via [`crate::persistent`] but a strategy-search +
//!   per-block dispatch isn't yet wrapped here.
//!
//! For full bitstream encoding today, use
//! [`crate::encoder::GpuEncoder::encode_lossy_via_cpu`] which delegates
//! to jxl-encoder.

use alloc::vec;
use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;
use crate::persistent::{GaborishWeights, GpuBlocks, GpuPlane};

/// libjxl gaborish K_GABORISH constants for mul=1.0. Matches
/// `forks::gaborish::compute_weights(1.0)` bit-for-bit.
const K_GABORISH: [f64; 5] = [
    -0.094_958_15_67,
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
    let m_min = block_means.iter().copied().fold(f32::INFINITY, f32::min);
    let m_max = block_means.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let qac_max = distance_to_qac(distance * 0.5); // detail → light quant
    let qac_min = distance_to_qac(distance * 2.0); // smooth → heavy quant
    block_means
        .iter()
        .map(|&m| {
            let t = if m_max > m_min { (m - m_min) / (m_max - m_min) } else { 0.5 };
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
    [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],   // X channel
    [560.0, 0.0, -0.3, -0.3, -0.3, -0.3],    // Y channel
    [512.0, -2.0, -1.0, 0.0, -1.0, -2.0],    // B channel
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
        // Upload one weights buffer per channel (X, Y, B) using the
        // libjxl DCT8 per-channel quant matrices. Y has the gentlest
        // quant (preserves luma); X/B have steeper quant (chroma).
        let all_weights = generate_dct8_quant_weights();
        let upload_channel = |slice: &[f32]| {
            let mut buf = Vec::with_capacity((num_blocks as usize) * 64);
            for _ in 0..num_blocks {
                buf.extend_from_slice(slice);
            }
            enc.upload_blocks(&buf, num_blocks, 64)
        };
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
    /// reconstructed RGB U8 (linearized output clamped + sRGB-encoded
    /// + rounded). Saves the caller the per-channel sRGB↔linear
    /// boilerplate.
    ///
    /// sRGB transfer function: gamma 2.4 (matches the simple model
    /// used elsewhere in the repo). For the IEC 61966-2-1 piecewise
    /// curve, deinterleave + linearize on the host before calling
    /// `encode_one` directly with f32 planes.
    pub fn encode_one_srgb_u8(
        &self,
        enc: &GpuEncoder<R>,
        rgb: &[u8],
        qac_qm: f32,
    ) -> Vec<u8> {
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
                block_means[by * blocks_per_row + bx] =
                    if count > 0 { (sum / count as f64) as f32 } else { 1.0 };
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
        let q_x =
            enc.quantize_dct8_persistent(&coeffs_x, &self.weights_x, qac_vec, &self.thresholds_x);
        let q_y =
            enc.quantize_dct8_persistent(&coeffs_y, &self.weights_y, qac_vec, &self.thresholds_y);
        let q_b =
            enc.quantize_dct8_persistent(&coeffs_b, &self.weights_b, qac_vec, &self.thresholds_b);
        let (dq_x, dq_y, dq_b) = enc.dequant_dct8_persistent(
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
        let (rgb_r, rgb_g, rgb_b) =
            enc.xyb_to_linear_rgb_planar_persistent(&recon_x_p, &recon_y_p, &recon_b_p);
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
