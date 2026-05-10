// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Host-side launcher for the fused DCT8+quantize kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::fused_dct_quant::{
    dct8_quantize_fused_broadcast_w_wide_kernel, dct8_quantize_fused_wide_kernel,
    dequant_idct8_fused_y_wide_kernel,
};

/// Fused dequant + IDCT8 (Y channel, no CfL).
pub fn dequant_idct8_fused_y_wide<R: Runtime>(
    client: &ComputeClient<R>,
    quant: Handle,
    weights: Handle,
    qac_qm: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n_coef = (num_blocks as usize) * 64;
    let cubes = num_blocks.div_ceil(64).max(1);
    unsafe {
        dequant_idct8_fused_y_wide_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(quant, n_coef),
            ArrayArg::from_raw_parts(weights, n_coef),
            ArrayArg::from_raw_parts(qac_qm, num_blocks as usize),
            ArrayArg::from_raw_parts(output, n_coef),
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dct8_quantize_fused_wide<R: Runtime>(
    client: &ComputeClient<R>,
    pixels: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n_coef = (num_blocks as usize) * 64;
    let cubes = num_blocks.div_ceil(64).max(1);
    unsafe {
        dct8_quantize_fused_wide_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(pixels, n_coef),
            ArrayArg::from_raw_parts(weights, n_coef),
            ArrayArg::from_raw_parts(qac_qm, num_blocks as usize),
            ArrayArg::from_raw_parts(thresholds, 4),
            ArrayArg::from_raw_parts(output, n_coef),
        );
    }
}

/// Broadcast-weights variant — `weights` is exactly 64 floats (one
/// quant matrix), broadcast across all blocks. Saves
/// `(num_blocks - 1) × 64 × 4` bytes of upload traffic compared to
/// the per-block [`dct8_quantize_fused_wide`].
#[allow(clippy::too_many_arguments)]
pub fn dct8_quantize_fused_broadcast_w_wide<R: Runtime>(
    client: &ComputeClient<R>,
    pixels: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n_coef = (num_blocks as usize) * 64;
    let cubes = num_blocks.div_ceil(64).max(1);
    unsafe {
        dct8_quantize_fused_broadcast_w_wide_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(pixels, n_coef),
            ArrayArg::from_raw_parts(weights, 64),
            ArrayArg::from_raw_parts(qac_qm, num_blocks as usize),
            ArrayArg::from_raw_parts(thresholds, 4),
            ArrayArg::from_raw_parts(output, n_coef),
        );
    }
}
