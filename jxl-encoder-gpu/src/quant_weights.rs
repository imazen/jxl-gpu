// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause via
// jxl-encoder/src/vardct/quant.rs, where the quant module is
// crate-private upstream).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Default DCT8 per-channel quant weights — public re-exposure of the
//! parametric band table that `LossyEncoder` uses internally.
//!
//! Demos and external callers building per-strategy cost grids need
//! these to get realistic per-strategy differentiation. Without real
//! weights (e.g., using `1.0` everywhere) the cost grids zero all but
//! DC under typical qac, making every 8×8-tier strategy look identical.

use alloc::vec;
use alloc::vec::Vec;

/// libjxl DCT8 band parameters from `quant_weights.cc:535-561`.
/// `[X, Y, B]` × 6 bands per channel.
pub const DCT8_PARAMS: [[f64; 6]; 3] = [
    [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],   // X channel
    [560.0, 0.0, -0.3, -0.3, -0.3, -0.3],    // Y channel
    [512.0, -2.0, -1.0, 0.0, -1.0, -2.0],    // B channel
];

/// libjxl DCT16x16 band parameters from `quant_weights.cc:647-676`.
/// `[X, Y, B]` × 7 bands per channel.
pub const DCT16X16_PARAMS: [[f64; 7]; 3] = [
    [
        8996.872_571_181_412,
        -1.300_077_739_335_380_4,
        -0.494_245_298_245_712_25,
        -0.439_093_774_457_103_44,
        -0.635_010_183_269_574_4,
        -0.901_772_640_508_276_1,
        -1.616_209_923_988_741_4,
    ],
    [
        3191.483_662_968_442_3,
        -0.674_245_821_041_943_5,
        -0.807_458_134_284_710_0,
        -0.449_258_374_848_434_4,
        -0.358_654_409_810_334_03,
        -0.313_223_891_118_773_05,
        -0.376_150_253_157_254_83,
    ],
    [
        1157.504_081_454_872,
        -2.053_142_316_580_441_4,
        -1.4,
        -0.506_871_300_333_784_0,
        -0.427_087_306_247_339_03,
        -1.485_683_453_929_624_4,
        -4.920_914_288_440_160_4,
    ],
];

#[inline]
fn band_mult(v: f64) -> f64 {
    if v > 0.0 {
        1.0 + v
    } else {
        1.0 / (1.0 - v)
    }
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
/// channel, X then Y then B). Matches
/// `jxl_encoder::vardct::quant::quant_weights(0, channel)` bit-for-bit.
///
/// Returned values are inverse-dequant weights: `1.0 / dequant_weight`,
/// matching the layout consumed by [`crate::launch::quantize::quantize_dct8`]
/// (which multiplies the coefficient by the weight).
pub fn dct8_weights() -> [f32; 192] {
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

/// Generate per-channel parametric DCT quant weights for an arbitrary
/// `rows × cols` block + per-channel band-parameter table.
///
/// Returns `3 * rows * cols` floats: X channel, then Y, then B.
/// Matches `jxl_encoder::vardct::quant::generate_dct_quant_weights_rect`
/// bit-for-bit.
///
/// Use this if you have band parameters for a strategy not exposed by
/// the named `*_weights()` helpers (e.g., DCT32X32, DCT4 family).
pub fn generate_quant_weights_rect(
    rows: usize,
    cols: usize,
    band_params: &[&[f64]; 3],
    num_bands: usize,
) -> Vec<f32> {
    let num = rows * cols;
    let total = 3 * num;
    let mut out = vec![0.0_f32; total];

    let sqrt2 = core::f64::consts::SQRT_2;
    let scale = (num_bands as f64 - 1.0) / (sqrt2 + 1e-6);
    let rcpcol = scale / (cols as f64 - 1.0);
    let rcprow = scale / (rows as f64 - 1.0);

    for c in 0..3 {
        let params = band_params[c];
        let mut bands = vec![0.0_f64; num_bands];
        bands[0] = params[0];
        for i in 1..num_bands {
            bands[i] = bands[i - 1] * band_mult(params[i]);
        }
        for y in 0..rows {
            let dy = y as f64 * rcprow;
            let dy2 = dy * dy;
            for x in 0..cols {
                let dx = x as f64 * rcpcol;
                let scaled_distance = (dx * dx + dy2).sqrt();
                let dequant_weight = interpolate_band(scaled_distance, &bands);
                out[c * num + y * cols + x] = (1.0 / dequant_weight) as f32;
            }
        }
    }
    out
}

/// Generate the 3-channel DCT16×16 quant weight table (768 floats:
/// 256 per channel, X/Y/B). Matches
/// `jxl_encoder::vardct::quant::quant_weights_dct16x16()` bit-for-bit.
pub fn dct16x16_weights() -> Vec<f32> {
    generate_quant_weights_rect(
        16,
        16,
        &[&DCT16X16_PARAMS[0], &DCT16X16_PARAMS[1], &DCT16X16_PARAMS[2]],
        7,
    )
}

/// Per-channel split: returns three 256-float `Vec<f32>` slices for X/Y/B.
pub fn dct16x16_weights_per_channel() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let all = dct16x16_weights();
    let x = all[..256].to_vec();
    let y = all[256..512].to_vec();
    let b = all[512..768].to_vec();
    (x, y, b)
}

/// Convenience: per-channel DCT8 weights as 3 separate `[f32; 64]`
/// blocks. Equivalent to slicing `dct8_weights()` into thirds.
pub fn dct8_weights_per_channel() -> ([f32; 64], [f32; 64], [f32; 64]) {
    let all = dct8_weights();
    let mut x = [0.0f32; 64];
    let mut y = [0.0f32; 64];
    let mut b = [0.0f32; 64];
    x.copy_from_slice(&all[..64]);
    y.copy_from_slice(&all[64..128]);
    b.copy_from_slice(&all[128..192]);
    (x, y, b)
}

/// Replicate a per-block weight matrix `n_blocks` times into a
/// contiguous `Vec<f32>`. Convenience for cost grid callers.
pub fn replicate_weights(per_block: &[f32], n_blocks: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_blocks * per_block.len()];
    for k in 0..n_blocks {
        out[k * per_block.len()..(k + 1) * per_block.len()].copy_from_slice(per_block);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct8_weights_shape_and_dc() {
        let w = dct8_weights();
        assert_eq!(w.len(), 192);
        // The DC weight (position 0) should be the smallest of each
        // 64-block (heaviest emphasis, smallest divisor in encoder
        // terms). For all 3 channels, w[0] < w[63] (high-freq corner).
        for c in 0..3 {
            let dc = w[c * 64];
            let hf = w[c * 64 + 63];
            assert!(
                dc < hf,
                "channel {c}: DC weight {dc} should be < high-freq corner {hf}"
            );
        }
    }

    #[test]
    fn test_per_channel_split_matches() {
        let all = dct8_weights();
        let (x, y, b) = dct8_weights_per_channel();
        for i in 0..64 {
            assert_eq!(x[i], all[i]);
            assert_eq!(y[i], all[64 + i]);
            assert_eq!(b[i], all[128 + i]);
        }
    }

    #[test]
    fn test_dct16x16_weights_shape_and_dc() {
        let w = dct16x16_weights();
        assert_eq!(w.len(), 768);
        for c in 0..3 {
            let dc = w[c * 256];
            // Last position (15, 15) — high-freq corner.
            let hf = w[c * 256 + 255];
            assert!(
                dc < hf,
                "channel {c}: DCT16 DC weight {dc} should be < high-freq corner {hf}"
            );
        }
    }

    #[test]
    fn test_dct16x16_per_channel_split_matches() {
        let all = dct16x16_weights();
        let (x, y, b) = dct16x16_weights_per_channel();
        for i in 0..256 {
            assert_eq!(x[i], all[i]);
            assert_eq!(y[i], all[256 + i]);
            assert_eq!(b[i], all[512 + i]);
        }
    }

    #[test]
    fn test_generate_rect_matches_dct8() {
        let direct = dct8_weights();
        let via_generic = generate_quant_weights_rect(
            8,
            8,
            &[&DCT8_PARAMS[0], &DCT8_PARAMS[1], &DCT8_PARAMS[2]],
            6,
        );
        assert_eq!(direct.len(), via_generic.len());
        for i in 0..direct.len() {
            assert_eq!(direct[i], via_generic[i]);
        }
    }

    #[test]
    fn test_replicate_weights() {
        let per = [1.0f32, 2.0, 3.0];
        let r = replicate_weights(&per, 4);
        assert_eq!(r.len(), 12);
        for k in 0..4 {
            assert_eq!(r[k * 3], 1.0);
            assert_eq!(r[k * 3 + 1], 2.0);
            assert_eq!(r[k * 3 + 2], 3.0);
        }
    }
}
