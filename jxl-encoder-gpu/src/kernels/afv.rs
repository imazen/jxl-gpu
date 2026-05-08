// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! AFV 4x4 DCT (forward + inverse) — the unique sub-transform inside
//! the AFV0-3 corner DCT family.
//!
//! The full AFV0-3 transforms (`afv_transform_from_pixels` upstream)
//! also use a regular DCT 4x4 + DCT 4x8 plus mirroring + DC packing.
//! Those are composable from existing kernels (`crate::launch::dct4`)
//! so this module ports just the AFV-specific 4x4 piece. Caller does
//! the corner extraction, mirroring, and DC packing on the host.
//!
//! Strategy: one thread per 4x4 sub-block (16 input pixels → 16
//! coefficients). The 16x16 basis matrix is uploaded once by the
//! caller (as `&Array<f32>` of length 256) and read inline.
//!
//! Forward:  coeffs[j] = sum_i (basis_T[i][j] * pixels[i])
//!           where basis_T is `AFV4X4_BASIS_TRANSPOSE` from libjxl.
//! Inverse:  pixels[i] = sum_j (basis_T[i][j] * coeffs[j])

use cubecl::prelude::*;

/// AFV4x4 basis matrix transpose, flattened row-major (16x16 = 256
/// floats). This is `AFV4X4_BASIS_TRANSPOSE` from
/// `jxl_encoder::vardct::afv` byte-for-byte.
#[rustfmt::skip]
pub const AFV4X4_BASIS_TRANSPOSE: [f32; 256] = [
    0.250_000_00, 0.876_902_9, 0.0, 0.0, 0.0, -0.410_537_75, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.250_000_00, 0.220_651_8, 0.0, 0.0, -0.707_106_77, 0.623_548_56, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.250_000_00, -0.101_400_5, 0.406_700_77, -0.212_557_48, 0.0, -0.064_350_72, -0.451_755_66, -0.304_684_75,
    0.301_792_94, 0.408_248_3, 0.174_786_7, -0.211_056_02, -0.142_660_85, -0.138_135_4, -0.174_376_03, 0.113_549_876,
    0.250_000_00, -0.101_400_5, 0.444_448_15, 0.308_549_7, 0.0, -0.064_350_72, 0.158_545_03, 0.511_261_6,
    0.257_923_63, 0.0, 0.081_261_12, 0.185_671_8, -0.341_644_7, 0.330_228_27, 0.070_279_07, -0.074_175_05,
    0.250_000_00, 0.220_651_8, 0.0, 0.0, 0.707_106_77, 0.623_548_56, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.250_000_00, -0.101_400_5, 0.0, 0.470_670_24, 0.0, -0.064_350_72, -0.040_385_153, 0.0,
    0.162_723_4, 0.0, 0.0, 0.0, 0.736_749_75, 0.087_551_154, -0.292_102_67, 0.194_028_93,
    0.250_000_00, -0.101_400_5, 0.195_744, -0.162_120_52, 0.0, -0.064_350_72, 0.007_418_226_3, -0.290_480_13,
    0.095_200_226, 0.0, -0.367_539_8, 0.492_158_6, 0.246_271_07, -0.079_467_066, 0.362_381_73, -0.435_190_5,
    0.250_000_00, -0.101_400_5, 0.292_910_02, 0.0, 0.0, -0.064_350_72, 0.393_510_34, -0.065_787_017,
    0.0, -0.408_248_3, -0.307_882_22, -0.385_250_14, -0.085_740_19, -0.461_337_5, 0.0, 0.219_186_85,
    0.250_000_00, -0.101_400_5, -0.406_700_77, -0.212_557_48, 0.0, -0.064_350_72, -0.451_755_66, 0.304_684_75,
    0.301_792_94, -0.408_248_3, -0.174_786_7, 0.211_056_02, -0.142_660_85, -0.138_135_4, -0.174_376_03, 0.113_549_876,
    0.250_000_00, -0.101_400_5, -0.195_744, -0.162_120_52, 0.0, -0.064_350_72, 0.007_418_226_3, 0.290_480_13,
    0.095_200_226, 0.0, 0.367_539_8, -0.492_158_6, 0.246_271_07, -0.079_467_066, 0.362_381_73, -0.435_190_5,
    0.250_000_00, -0.101_400_5, 0.0, -0.470_670_24, 0.0, -0.064_350_72, 0.110_741_66, 0.0,
    -0.162_723_4, 0.0, 0.0, 0.0, 0.148_834, 0.497_246_47, 0.292_102_67, 0.555_044_4,
    0.250_000_00, -0.101_400_5, 0.113_790_745, -0.146_429_19, 0.0, -0.064_350_72, 0.082_981_63, -0.238_897_74,
    -0.353_123_85, -0.408_248_3, 0.482_668_92, 0.174_194_13, -0.047_686_804, 0.125_380_6, -0.432_660_8, -0.254_682_77,
    0.250_000_00, -0.101_400_5, -0.444_448_15, 0.308_549_7, 0.0, -0.064_350_72, 0.158_545_03, -0.511_261_6,
    0.257_923_63, 0.0, -0.081_261_12, -0.185_671_8, -0.341_644_7, 0.330_228_27, 0.070_279_07, -0.074_175_05,
    0.250_000_00, -0.101_400_5, -0.292_910_02, 0.0, 0.0, -0.064_350_72, 0.393_510_34, 0.065_787_017,
    0.0, 0.408_248_3, 0.307_882_22, 0.385_250_14, -0.085_740_19, -0.461_337_5, 0.0, 0.219_186_85,
    0.250_000_00, -0.101_400_5, -0.113_790_745, -0.146_429_19, 0.0, -0.064_350_72, 0.082_981_63, 0.238_897_74,
    -0.353_123_85, 0.408_248_3, -0.482_668_92, -0.174_194_13, -0.047_686_804, 0.125_380_6, -0.432_660_8, -0.254_682_77,
    0.250_000_00, -0.101_400_5, 0.0, 0.425_114_96, 0.0, -0.064_350_72, -0.451_755_66, 0.0,
    -0.603_585_9, 0.0, 0.0, 0.0, -0.142_660_85, -0.138_135_4, 0.348_752_05, 0.113_549_876,
];

/// Forward AFV 4x4 DCT on a contiguous batch of 16-pixel sub-blocks.
/// Input/output: `num_blocks * 16` floats. `basis_t` is the 16x16
/// basis-transpose matrix (256 floats), uploaded once by the caller.
#[cube(launch_unchecked)]
pub fn afv_dct_4x4_kernel(pixels: &Array<f32>, basis_t: &Array<f32>, coeffs: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = pixels.len() / 16u32 as usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 16usize;

    // Load all 16 pixels into private scratch.
    let mut p = SharedMemory::<f32>::new(16usize);
    let mut i: u32 = 0u32;
    while i < 16u32 {
        p[i as usize] = pixels[off + i as usize];
        i += 1u32;
    }

    // For each output coeff j, accumulate sum_i (basis_t[i][j] * p[i]).
    let mut j: u32 = 0u32;
    while j < 16u32 {
        let mut sum = 0.0f32;
        let mut k: u32 = 0u32;
        while k < 16u32 {
            sum = sum + basis_t[(k * 16u32 + j) as usize] * p[k as usize];
            k += 1u32;
        }
        coeffs[off + j as usize] = sum;
        j += 1u32;
    }
}

/// Inverse AFV 4x4 DCT. `pixels[i] = sum_j (basis_t[i][j] * coeffs[j])`.
#[cube(launch_unchecked)]
pub fn afv_idct_4x4_kernel(coeffs: &Array<f32>, basis_t: &Array<f32>, pixels: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = coeffs.len() / 16u32 as usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 16usize;

    let mut c = SharedMemory::<f32>::new(16usize);
    let mut j: u32 = 0u32;
    while j < 16u32 {
        c[j as usize] = coeffs[off + j as usize];
        j += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 16u32 {
        let mut sum = 0.0f32;
        let mut k: u32 = 0u32;
        while k < 16u32 {
            sum = sum + basis_t[(i * 16u32 + k) as usize] * c[k as usize];
            k += 1u32;
        }
        pixels[off + i as usize] = sum;
        i += 1u32;
    }
}
