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
    [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0], // X channel
    [560.0, 0.0, -0.3, -0.3, -0.3, -0.3],  // Y channel
    [512.0, -2.0, -1.0, 0.0, -1.0, -2.0],  // B channel
];

/// libjxl IDENTITY dequant weights from `quant_weights.cc:80-90, 564-579`.
/// Per-channel: `[DC, AC_pos1_pos8, AC_pos9]`. All other 8×8 positions
/// use the DC weight.
pub const IDENTITY_DEQUANT_WEIGHTS: [[f32; 3]; 3] = [
    [280.0, 3160.0, 3160.0], // X channel
    [60.0, 864.0, 864.0],    // Y channel
    [18.0, 200.0, 200.0],    // B channel
];

/// libjxl DCT2X2 dequant band weights from `quant_weights.cc:48-77, 583-607`.
/// 6 hierarchical band weights per channel.
pub const DCT2_DEQUANT_WEIGHTS: [[f32; 6]; 3] = [
    [3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0], // X channel
    [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],    // Y channel
    [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],       // B channel
];

/// libjxl DCT4X8 band parameters from jxl-oxide `dequant.rs:44-48`.
/// `[X, Y, B]` × 4 bands per channel. Used for BOTH DCT4X8 and DCT8X4.
pub const DCT4X8_PARAMS: [[f64; 4]; 3] = [
    [2198.0505, -0.96269625, -0.7619425, -0.65511405],
    [764.36554, -0.926302, -0.967523, -0.2784529],
    [527.10754, -1.4594386, -1.4500821, -1.5843723],
];

/// libjxl DCT4X4 band parameters from jxl-oxide `dequant.rs:49-53`.
/// `[X, Y, B]` × 4 bands per channel.
pub const DCT4_PARAMS: [[f64; 4]; 3] = [
    [2200.0, 0.0, 0.0, 0.0],
    [392.0, 0.0, 0.0, 0.0],
    [112.0, -0.25, -0.25, -0.5],
];

/// libjxl DCT4X4 LLF multiplier parameters from jxl-oxide `dequant.rs:257-277`.
/// `params[0]` is used for LLF positions 1 and 8, `params[1]` for position 9.
pub const DCT4_LLF_PARAMS: [[f64; 2]; 3] = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];

/// libjxl DCT16x8 band parameters from `quant_weights.cc:716-745`.
/// `[X, Y, B]` × 7 bands per channel. Used for BOTH the DCT16X8 and
/// DCT8X16 strategies (they share the rotated table — the weight
/// generator uses `(rows=8, cols=16)`).
pub const DCT16X8_PARAMS: [[f64; 7]; 3] = [
    [7240.773_439_350_2, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5],
    [1448.154_687_870_04, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
    [506.854_140_754_517, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6],
];

/// libjxl DCT32x32 band parameters from `quant_weights.cc:680-712`.
/// `[X, Y, B]` × 8 bands per channel.
pub const DCT32X32_PARAMS: [[f64; 8]; 3] = [
    [
        15718.408_309_825_19,
        -1.025,
        -0.98,
        -0.9012,
        -0.4,
        -0.488_193_954_64,
        -0.421_064,
        -0.27,
    ],
    [
        7305.763_681_069_598,
        -0.804_195_821_230_640_1,
        -0.763_303_645_748_753_9,
        -0.556_603_799_901_114_64,
        -0.497_853_046_588_576_26,
        -0.436_995_926_835_124_67,
        -0.401_808_665_262_421_09,
        -0.273_216_831_253_580_37,
    ],
    [
        3803.531_737_212_150_5,
        -3.060_733_579_805_728,
        -2.041_327_013_249_034_6,
        -2.023_565_015_972_741_7,
        -0.549_538_950_995_499_3,
        -0.4,
        -0.4,
        -0.3,
    ],
];

/// libjxl DCT16x32 band parameters from jxl-rs `quant_weights.rs:561-590`.
/// `[X, Y, B]` × 8 bands per channel. Used for BOTH DCT32x16 and DCT16x32.
pub const DCT16X32_PARAMS: [[f64; 8]; 3] = [
    [
        13844.970_764_423_006,
        -0.971_138,
        -0.658,
        -0.420_26,
        -0.227_12,
        -0.220_6,
        -0.226,
        -0.6,
    ],
    [
        4798.964_084_220_744_5,
        -0.611_253_089_827_670_57,
        -0.837_707_865_524_913_6,
        -0.790_148_620_794_986_3,
        -0.269_272_745_970_482_9,
        -0.382_727_694_653_885_5,
        -0.229_242_226_530_914_53,
        -0.207_190_988_261_995_78,
    ],
    [
        1807.236_946_760_964_4,
        -1.2,
        -1.2,
        -0.7,
        -0.7,
        -0.7,
        -0.4,
        -0.5,
    ],
];

/// libjxl DCT64x64 band parameters from `quant_weights.cc:899-931`.
/// `[X, Y, B]` × 8 bands per channel.
pub const DCT64X64_PARAMS: [[f64; 8]; 3] = [
    [
        23966.166_529_844_86,
        -1.025,
        -0.78,
        -0.650_12,
        -0.190_415_740_842_864_72,
        -0.208_193_954_64,
        -0.421_064,
        -0.327_338_455_358_486_71,
    ],
    [
        8380.191_483_900_904,
        -0.304_195_821_230_640_1,
        -0.363_303_645_748_753_9,
        -0.356_603_799_901_114_64,
        -0.344_307_445_542_440_3,
        -0.336_995_926_835_124_67,
        -0.301_808_665_262_421_09,
        -0.273_216_831_253_580_37,
    ],
    [
        4493.023_780_098_477,
        -1.2,
        -1.2,
        -0.8,
        -0.7,
        -0.7,
        -0.4,
        -0.5,
    ],
];

/// libjxl DCT32x64 band parameters from `quant_weights.cc:935-968`.
/// `[X, Y, B]` × 8 bands per channel. Used for BOTH DCT64x32 and DCT32x64.
pub const DCT32X64_PARAMS: [[f64; 8]; 3] = [
    [
        15358.898_049_332_4,
        -1.025,
        -0.78,
        -0.650_12,
        -0.190_415_740_842_864_72,
        -0.208_193_954_64,
        -0.421_064,
        -0.327_338_455_358_486_71,
    ],
    [
        5597.360_516_150_653,
        -0.304_195_821_230_640_1,
        -0.363_303_645_748_753_9,
        -0.356_603_799_901_114_64,
        -0.344_307_445_542_440_3,
        -0.336_995_926_835_124_67,
        -0.301_808_665_262_421_09,
        -0.273_216_831_253_580_37,
    ],
    [
        2919.961_618_960_011,
        -1.2,
        -1.2,
        -0.8,
        -0.7,
        -0.7,
        -0.4,
        -0.5,
    ],
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
        &[
            &DCT16X16_PARAMS[0],
            &DCT16X16_PARAMS[1],
            &DCT16X16_PARAMS[2],
        ],
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

/// Generate DCT16x8 quant weights (8 rows × 16 cols = 128 per channel,
/// 384 total). Same table also used for DCT8x16.
pub fn dct16x8_weights() -> Vec<f32> {
    generate_quant_weights_rect(
        8,
        16,
        &[&DCT16X8_PARAMS[0], &DCT16X8_PARAMS[1], &DCT16X8_PARAMS[2]],
        7,
    )
}

/// Per-channel split: returns three 128-float `Vec<f32>` slices for X/Y/B.
/// Used by both DCT16x8 and DCT8x16 strategy paths (the underlying band
/// table is shared).
pub fn dct16x8_weights_per_channel() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let all = dct16x8_weights();
    let x = all[..128].to_vec();
    let y = all[128..256].to_vec();
    let b = all[256..384].to_vec();
    (x, y, b)
}

/// Generate DCT32x32 quant weights (1024 per channel, 3072 total).
pub fn dct32x32_weights() -> Vec<f32> {
    generate_quant_weights_rect(
        32,
        32,
        &[
            &DCT32X32_PARAMS[0],
            &DCT32X32_PARAMS[1],
            &DCT32X32_PARAMS[2],
        ],
        8,
    )
}

/// Per-channel split: returns three 1024-float `Vec<f32>` slices for X/Y/B.
pub fn dct32x32_weights_per_channel() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let all = dct32x32_weights();
    let x = all[..1024].to_vec();
    let y = all[1024..2048].to_vec();
    let b = all[2048..3072].to_vec();
    (x, y, b)
}

/// Per-channel split: returns three 512-float `Vec<f32>` slices for X/Y/B.
/// Used by both DCT16x32 and DCT32x16 paths (shared band table).
pub fn dct16x32_weights_per_channel() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let all = dct16x32_weights();
    let x = all[..512].to_vec();
    let y = all[512..1024].to_vec();
    let b = all[1024..1536].to_vec();
    (x, y, b)
}

/// Generate DCT16x32 quant weights (16 rows × 32 cols = 512 per channel,
/// 1536 total). Same table also used for DCT32x16.
pub fn dct16x32_weights() -> Vec<f32> {
    generate_quant_weights_rect(
        16,
        32,
        &[
            &DCT16X32_PARAMS[0],
            &DCT16X32_PARAMS[1],
            &DCT16X32_PARAMS[2],
        ],
        8,
    )
}

/// Generate DCT64x64 quant weights (4096 per channel, 12288 total).
pub fn dct64x64_weights() -> Vec<f32> {
    generate_quant_weights_rect(
        64,
        64,
        &[
            &DCT64X64_PARAMS[0],
            &DCT64X64_PARAMS[1],
            &DCT64X64_PARAMS[2],
        ],
        8,
    )
}

/// Generate IDENTITY quant weights (64 per channel, 192 total).
/// All 64 positions get the DC weight, then positions 1, 8, 9 are
/// overwritten per `IDENTITY_DEQUANT_WEIGHTS`. Output is `1 / dequant`.
pub fn identity_weights() -> Vec<f32> {
    let mut weights = vec![0.0_f32; 3 * 64];
    for (c, ch) in IDENTITY_DEQUANT_WEIGHTS.iter().enumerate() {
        let start = c * 64;
        let dq0 = ch[0];
        let dq1 = ch[1];
        let dq2 = ch[2];
        for w in &mut weights[start..start + 64] {
            *w = 1.0 / dq0;
        }
        weights[start + 1] = 1.0 / dq1;
        weights[start + 8] = 1.0 / dq1;
        weights[start + 9] = 1.0 / dq2;
    }
    weights
}

/// Generate DCT2X2 quant weights (64 per channel, 192 total). Position 0
/// is filled with the libjxl `0xBAD` sentinel (DC handled separately by
/// the encoder); positions 1/8/9 use band 0/1, and the four hierarchical
/// 2×2/4×4 quadrants use bands 2-5 per `DCT2_DEQUANT_WEIGHTS`.
pub fn dct2x2_weights() -> Vec<f32> {
    let mut weights = vec![0.0_f32; 3 * 64];
    for (c, w) in DCT2_DEQUANT_WEIGHTS.iter().enumerate() {
        let start = c * 64;
        // DC sentinel: 1/0xBAD ≈ 0.0000003. Encoder treats DC specially.
        weights[start] = 1.0 / 0xBAD as f32;
        weights[start + 1] = 1.0 / w[0];
        weights[start + 8] = 1.0 / w[0];
        weights[start + 9] = 1.0 / w[1];
        // 2×2 quadrants at offsets (0..2, 2..4) and (2..4, 0..2) → band 2
        for y in 0..2usize {
            for x in 0..2usize {
                weights[start + y * 8 + x + 2] = 1.0 / w[2];
                weights[start + (y + 2) * 8 + x] = 1.0 / w[2];
            }
        }
        // 2×2 bottom-right quadrant at (2..4, 2..4) → band 3
        for y in 0..2usize {
            for x in 0..2usize {
                weights[start + (y + 2) * 8 + x + 2] = 1.0 / w[3];
            }
        }
        // 4×4 right + bottom band at (0..4, 4..8) and (4..8, 0..4) → band 4
        for y in 0..4usize {
            for x in 0..4usize {
                weights[start + y * 8 + x + 4] = 1.0 / w[4];
                weights[start + (y + 4) * 8 + x] = 1.0 / w[4];
            }
        }
        // 4×4 bottom-right at (4..8, 4..8) → band 5
        for y in 0..4usize {
            for x in 0..4usize {
                weights[start + (y + 4) * 8 + x + 4] = 1.0 / w[5];
            }
        }
    }
    weights
}

/// Generate DCT4X8 quant weights (64 per channel = 8×8 row-duplicated
/// from a 4-tall × 8-wide base, 192 total). Same table used for DCT8X4.
pub fn dct4x8_weights() -> Vec<f32> {
    let mut weights = Vec::with_capacity(192);
    let sqrt2 = core::f64::consts::SQRT_2;

    for params in &DCT4X8_PARAMS {
        let mut bands = vec![params[0]];
        let mut last = params[0];
        for &v in &params[1..] {
            last *= band_mult(v);
            bands.push(last);
        }
        let width = 8usize;
        let height = 4usize;
        let mut mat = vec![0.0_f64; width * height];
        for y in 0..height {
            let dy = y as f64 / (height - 1).max(1) as f64;
            for x in 0..width {
                let dx = x as f64 / (width - 1).max(1) as f64;
                let distance = (dx * dx + dy * dy).sqrt();
                let scaled = distance * (bands.len() - 1) as f64 / (sqrt2 + 1e-6);
                mat[y * width + x] = interpolate_band(scaled, &bands);
            }
        }
        // Duplicate each row to expand 4×8 → 8×8.
        for row in 0..height {
            for x in 0..width {
                weights.push((1.0 / mat[row * width + x]) as f32);
            }
            for x in 0..width {
                weights.push((1.0 / mat[row * width + x]) as f32);
            }
        }
    }
    weights
}

/// Generate DCT4X4 quant weights (64 per channel, 192 total). Builds a
/// 4×4 base via parametric bands then 2×2-replicates each cell into the
/// 8×8 layout, with LLF divisors applied to positions 1, 8, 9.
pub fn dct4x4_weights() -> Vec<f32> {
    let mut weights = Vec::with_capacity(192);
    let sqrt2 = core::f64::consts::SQRT_2;

    for (c, params) in DCT4_PARAMS.iter().enumerate() {
        let mut bands = vec![params[0]];
        let mut last = params[0];
        for &v in &params[1..] {
            last *= band_mult(v);
            bands.push(last);
        }
        let size = 4usize;
        let mut mat = vec![0.0_f64; size * size];
        for y in 0..size {
            let dy = y as f64 / (size - 1).max(1) as f64;
            for x in 0..size {
                let dx = x as f64 / (size - 1).max(1) as f64;
                let distance = (dx * dx + dy * dy).sqrt();
                let scaled = distance * (bands.len() - 1) as f64 / (sqrt2 + 1e-6);
                mat[y * size + x] = interpolate_band(scaled, &bands);
            }
        }
        // 2×2-replicate each weight into the 8×8 layout.
        let mut channel = vec![0.0_f64; 64];
        for y in 0..4 {
            for x in 0..4 {
                let w = mat[y * 4 + x];
                channel[y * 16 + x * 2] = w;
                channel[y * 16 + x * 2 + 1] = w;
                channel[(y * 2 + 1) * 8 + x * 2] = w;
                channel[(y * 2 + 1) * 8 + x * 2 + 1] = w;
            }
        }
        // LLF divisors at positions 1, 8, 9.
        channel[1] /= DCT4_LLF_PARAMS[c][0];
        channel[8] /= DCT4_LLF_PARAMS[c][0];
        channel[9] /= DCT4_LLF_PARAMS[c][1];

        for w in &channel {
            weights.push((1.0 / w) as f32);
        }
    }
    weights
}

/// AFV per-channel parameters: `[afv01, afv10, afv02, afv20, afv22, band0, band1_mult, band2_mult, band3_mult]`.
/// X / Y / B order. Mirrors libjxl `kAfvParams` / our upstream
/// `jxl_encoder::vardct::quant::AFV_WEIGHTS`.
const AFV_WEIGHTS: [[f64; 9]; 3] = [
    // X channel
    [3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0],
    // Y channel
    [1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0],
    // B channel
    [384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25],
];

/// AFV frequency lookup (16 = 4×4). `(0,0)` `(0,1)` `(1,0)` `(1,1)`
/// are unused (DC tendency / corner positions handled separately).
/// From libjxl `kFreqs`.
const AFV_FREQS: [f64; 16] = [
    0.0,
    0.0,
    0.8517778890324296,
    5.37778436506804,
    0.0,
    0.0,
    4.734747904497923,
    5.449245381693219,
    1.6598270267479331,
    4.0,
    7.275749096817861,
    10.423227632456525,
    2.662932286148962,
    7.630657783650829,
    8.962388608184032,
    12.97166202570235,
];

/// Generate AFV quant weights (192 floats: 64 per channel; same table
/// for all four AFV variants AFV0-AFV3).
///
/// Bit-for-bit port of upstream `jxl_encoder::vardct::quant::generate_afv_weights`.
/// Layout per channel (row-major within an 8×8 block):
/// - position (0,0) = DC weight = `1/bands[0]`
/// - positions (0,1), (1,0) = "DC tendency" weights from `afv[0..2]`
/// - positions (0,2), (2,0), (2,2) = corner weights from `afv[2..5]`
/// - other (even, even) cells with `x>=2 || y>=2`: interpolated band weight
/// - odd-row cells: shared with DCT4×8 weights (row-duplicated layout)
/// - (even-row, odd-col) cells: shared with DCT4×4 weights (replicated layout)
pub fn afv_weights() -> Vec<f32> {
    let mut weights = vec![0.0_f32; 192];
    let weights4x8 = dct4x8_weights();
    let weights4x4 = dct4x4_weights();

    const LO: f64 = 0.8517778890324296;
    const HI: f64 = 12.97166202570235 - LO + 1e-6;

    for (c, afv) in AFV_WEIGHTS.iter().enumerate() {
        let start = c * 64;

        let mut bands = [0.0_f64; 4];
        bands[0] = afv[5];
        for i in 1..4 {
            bands[i] = bands[i - 1] * band_mult(afv[5 + i]);
        }

        weights[start] = (1.0 / bands[0]) as f32;
        weights[start + 1] = (1.0 / afv[0]) as f32;
        weights[start + 8] = (1.0 / afv[1]) as f32;
        weights[start + 2] = (1.0 / afv[2]) as f32;
        weights[start + 16] = (1.0 / afv[3]) as f32;
        weights[start + 18] = (1.0 / afv[4]) as f32;

        // Other AFV-corner positions on the (even, even) sublattice with x>=2 or y>=2.
        for y in 0..4_usize {
            for x in 0..4_usize {
                if x < 2 && y < 2 {
                    continue;
                }
                let freq = AFV_FREQS[y * 4 + x];
                let val = interpolate_band((freq - LO) / HI * 3.0, &bands);
                weights[start + (2 * y) * 8 + (2 * x)] = (1.0 / val) as f32;
            }
        }

        // DCT4×8 weights along odd rows (skipping (0,0) which is the DC tendency
        // already populated above).
        for y in 0..4_usize {
            for x in 0..8_usize {
                if x == 0 && y == 0 {
                    continue;
                }
                let idx4x8 = c * 64 + y * 16 + x;
                weights[start + (2 * y + 1) * 8 + x] = weights4x8[idx4x8];
            }
        }

        // DCT4×4 weights at (even-row, odd-col) cells (skipping (0,0)).
        for y in 0..4_usize {
            for x in 0..4_usize {
                if x == 0 && y == 0 {
                    continue;
                }
                let idx4x4 = c * 64 + y * 16 + x * 2;
                weights[start + (2 * y) * 8 + (2 * x + 1)] = weights4x4[idx4x4];
            }
        }
    }

    weights
}

/// Generate DCT32x64 quant weights (32 rows × 64 cols = 2048 per channel,
/// 6144 total). Same table also used for DCT64x32.
pub fn dct32x64_weights() -> Vec<f32> {
    generate_quant_weights_rect(
        32,
        64,
        &[
            &DCT32X64_PARAMS[0],
            &DCT32X64_PARAMS[1],
            &DCT32X64_PARAMS[2],
        ],
        8,
    )
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
    fn test_larger_weights_shapes() {
        assert_eq!(dct16x8_weights().len(), 384);
        assert_eq!(dct32x32_weights().len(), 3072);
        assert_eq!(dct16x32_weights().len(), 1536);
        assert_eq!(dct64x64_weights().len(), 12288);
        assert_eq!(dct32x64_weights().len(), 6144);
    }

    #[test]
    fn test_dct32x32_dc_vs_corner() {
        // Sanity: DC (position 0) weight should be much smaller than the
        // high-freq corner (position 1023) — encoder weights = 1/dequant.
        let w = dct32x32_weights();
        for c in 0..3 {
            let dc = w[c * 1024];
            let hf = w[c * 1024 + 1023];
            assert!(dc < hf, "DCT32 channel {c}: DC={dc} should be < HF={hf}");
        }
    }

    #[test]
    fn test_identity_dct2x2_dct4_shapes() {
        assert_eq!(identity_weights().len(), 192);
        assert_eq!(dct2x2_weights().len(), 192);
        assert_eq!(dct4x8_weights().len(), 192);
        assert_eq!(dct4x4_weights().len(), 192);
    }

    #[test]
    fn test_identity_position_layout() {
        // IDENTITY: positions 1, 8 use ch[1]; position 9 uses ch[2];
        // all other positions use ch[0].
        let w = identity_weights();
        for c in 0..3 {
            let dq = IDENTITY_DEQUANT_WEIGHTS[c];
            let s = c * 64;
            assert_eq!(w[s], 1.0 / dq[0]); // DC = ch[0]
            assert_eq!(w[s + 1], 1.0 / dq[1]); // AC ch[1]
            assert_eq!(w[s + 8], 1.0 / dq[1]); // AC ch[1]
            assert_eq!(w[s + 9], 1.0 / dq[2]); // pos9 ch[2]
            assert_eq!(w[s + 5], 1.0 / dq[0]); // arbitrary other pos = DC
        }
    }

    #[test]
    fn test_dct2x2_dc_sentinel() {
        // DCT2X2: position 0 is the 1/0xBAD sentinel — DC handled separately.
        let w = dct2x2_weights();
        for c in 0..3 {
            assert_eq!(w[c * 64], 1.0 / 0xBAD as f32);
        }
    }

    #[test]
    fn test_afv_weights_shape_and_anchors() {
        let w = afv_weights();
        assert_eq!(w.len(), 192);
        // All weights finite and positive (1/dequant where dequant > 0).
        for v in &w {
            assert!(v.is_finite() && *v > 0.0, "got {v}");
        }
        // Spot-check the explicit positions against the constants.
        // DC at start = 1 / bands[0] = 1 / afv[5]
        for (c, afv) in AFV_WEIGHTS.iter().enumerate() {
            let s = c * 64;
            // Position 0 is DC = 1/bands[0] = 1/afv[5]
            let expected_dc = (1.0 / afv[5]) as f32;
            assert!(
                (w[s] - expected_dc).abs() < 1e-6 * expected_dc.abs().max(1.0),
                "channel {c} DC: got {} expected {}",
                w[s],
                expected_dc
            );
            // (0,1) DC tendency = 1/afv[0]
            assert!((w[s + 1] - (1.0 / afv[0]) as f32).abs() < 1e-6);
            // (1,0) DC tendency = 1/afv[1]
            assert!((w[s + 8] - (1.0 / afv[1]) as f32).abs() < 1e-6);
            // (0,2), (2,0), (2,2) corners
            assert!((w[s + 2] - (1.0 / afv[2]) as f32).abs() < 1e-6);
            assert!((w[s + 16] - (1.0 / afv[3]) as f32).abs() < 1e-6);
            assert!((w[s + 18] - (1.0 / afv[4]) as f32).abs() < 1e-6);
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
