// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-block 8th-power-norm of masked pixel errors.
//!
//! Mirrors `jxl_encoder_simd::pixel_loss::pixel_domain_loss_scalar`.
//!
//!   `loss = Σ_pixels ((mask[px] + mask_offset) * error[px])^8`
//!
//! The squaring is done in f64 because m8 = (small)^8 underflows f32
//! for typical perceptual mask values. The CPU AVX2 path explicitly
//! upcasts to f64x4 via `_mm256_cvtps_pd` for the same reason.
//!
//! Output is f64 per block. cubecl 0.10 supports f64 on the CUDA
//! backend (RTX 5070 has hardware DP at reduced throughput); WGPU
//! support depends on the device's `float64-blend` feature.

use cubecl::prelude::*;

/// Per-block 8th-power norm. One cube per block.
///
/// Layout:
/// - `pixel_error`: contiguous error rows for this batch of blocks. Each
///   block occupies `block_width * block_height` floats starting at
///   `block_idx * block_width * block_height`.
/// - `mask`: full-image mask plane, indexed via `mask_row_base +
///   row*mask_stride + col`. `mask_row_base` is per-block — stored as
///   `mask_row_base[block_idx]`.
/// - `mask_offset`: scalar broadcast.
/// - `output`: `num_blocks` f64 per-block losses.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn pixel_loss_kernel(
    pixel_error: &Array<f32>,
    mask: &Array<f32>,
    mask_row_base: &Array<u32>,
    output: &mut Array<f64>,
    mask_stride: u32,
    mask_offset: f32,
    block_width: u32,
    block_height: u32,
) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = output.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let bw = block_width as usize;
    let bh = block_height as usize;
    let ms = mask_stride as usize;
    let mrb = mask_row_base[block_idx] as usize;
    let err_base = block_idx * bw * bh;

    let mut acc = f64::new(0.0);
    let mut py: u32 = 0u32;
    while py < block_height {
        let pyu = py as usize;
        let mask_row_start = mrb + pyu * ms;
        let error_row_start = err_base + pyu * bw;
        let mut px: u32 = 0u32;
        while px < block_width {
            let pxu = px as usize;
            let mask_val = mask[mask_row_start + pxu];
            let err_val = pixel_error[error_row_start + pxu];
            let masked = (mask_val + mask_offset) * err_val;
            // Upcast for the squaring chain (matches CPU avx2 path).
            let m2 = (masked * masked) as f64;
            let m4 = m2 * m2;
            let m8 = m4 * m4;
            acc = acc + m8;
            px += 1u32;
        }
        py += 1u32;
    }
    output[block_idx] = acc;
}
