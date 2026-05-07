// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-pixel Wiener denoise filter.
//!
//! Mirrors `jxl_encoder_simd::noise::denoise_channel_scalar`.
//!
//! For each pixel: 8-point noise LUT lookup on `y_channel[idx]` →
//! `noise_var = (lut * denoise_scale)²`. If `noise_var < EPS`, write
//! `orig[idx]` unchanged. Otherwise compute local mean+variance over a
//! 5×5 window (clipped to image bounds), then apply Wiener:
//! `out = mean + (orig - mean) * signal_var / (signal_var + noise_var)`
//! where `signal_var = max(local_var - noise_var, 0)`.

use cubecl::prelude::*;

const NUM_NOISE_POINTS: u32 = 8;
const RADIUS: u32 = 2;
const EPS: f32 = 1e-10;

/// Interpolate the 8-point noise LUT at intensity `x`. Mirrors the
/// scalar `index_and_frac` + `interpolate_noise_lut` exactly.
#[cube]
fn interpolate_lut(noise_lut: &Array<f32>, x: f32) -> f32 {
    let k_scale = (NUM_NOISE_POINTS - 2u32) as f32;
    let scaled = f32::max(x * k_scale, 0.0f32);
    let floored = f32::floor(scaled);
    // Mirror the scalar: if scaled_x >= k_scale + 1.0 (= 7.0), idx = 6
    // and frac = 1.0 → result = lut[7]. Otherwise idx = floor(scaled).
    let mut idx = floored as u32;
    let mut frac = scaled - floored;
    if scaled >= k_scale + 1.0f32 {
        idx = NUM_NOISE_POINTS - 2u32;
        frac = 1.0f32;
    }
    if idx >= NUM_NOISE_POINTS - 1u32 {
        noise_lut[(NUM_NOISE_POINTS - 1u32) as usize]
    } else {
        let lo = noise_lut[idx as usize];
        let hi = noise_lut[(idx + 1u32) as usize];
        lo * (1.0f32 - frac) + hi * frac
    }
}

/// Per-pixel Wiener denoise. One thread per pixel.
#[cube(launch_unchecked)]
pub fn denoise_kernel(
    orig: &Array<f32>,
    y_channel: &Array<f32>,
    noise_lut: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    denoise_scale: f32,
) {
    let idx = ABSOLUTE_POS;
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if idx >= n {
        terminate!();
    }
    let py = idx / w;
    let px = idx - py * w;

    let y_val = y_channel[idx];
    let sigma = interpolate_lut(noise_lut, f32::abs(y_val)) * denoise_scale;
    let noise_var = sigma * sigma;
    if noise_var < EPS {
        output[idx] = orig[idx];
        terminate!();
    }

    let r = RADIUS as usize;
    let y_start = usize::saturating_sub(py, r);
    let y_end = usize::min(py + r + 1usize, h);
    let x_start = usize::saturating_sub(px, r);
    let x_end = usize::min(px + r + 1usize, w);

    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut count = 0.0f32;
    for ny in y_start..y_end {
        for nx in x_start..x_end {
            let v = orig[ny * w + nx];
            sum += v;
            sum_sq += v * v;
            count += 1.0f32;
        }
    }

    let mean = sum / count;
    let local_var = f32::max((sum_sq / count) - mean * mean, 0.0f32);
    let signal_var = f32::max(local_var - noise_var, 0.0f32);
    let wiener = signal_var / (signal_var + noise_var);
    output[idx] = mean + (orig[idx] - mean) * wiener;
}
