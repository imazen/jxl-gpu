// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for per-block DCT8 quantization with dead-zone.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::quantize::{
    quantize_dct8_kernel, quantize_dct8_kernel_broadcast_w, quantize_large_kernel,
};

#[allow(clippy::too_many_arguments)]
pub fn quantize_large<R: Runtime>(
    client: &ComputeClient<R>,
    coeffs: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle,
    output: Handle,
    num_blocks: u32,
    grid_width: u32,
    grid_height: u32,
    llf_x: u32,
    llf_y: u32,
) {
    let nb = num_blocks as usize;
    let size = (grid_width as usize) * (grid_height as usize);
    let n_coef = nb * size;
    let cubes = num_blocks.max(1);
    unsafe {
        quantize_large_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(coeffs, n_coef),
            ArrayArg::from_raw_parts(weights, n_coef),
            ArrayArg::from_raw_parts(qac_qm, nb),
            ArrayArg::from_raw_parts(thresholds, 4),
            ArrayArg::from_raw_parts(output, n_coef),
            grid_width,
            grid_height,
            llf_x,
            llf_y,
        );
    }
}

pub fn quantize_dct8<R: Runtime>(
    client: &ComputeClient<R>,
    coeffs: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle, // 4 f32
    output: Handle,     // num_blocks * 64 i32
    num_blocks: u32,
) {
    let n_coef = (num_blocks as usize) * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        quantize_dct8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(coeffs, n_coef),
            ArrayArg::from_raw_parts(weights, n_coef),
            ArrayArg::from_raw_parts(qac_qm, num_blocks as usize),
            ArrayArg::from_raw_parts(thresholds, 4),
            ArrayArg::from_raw_parts(output, n_coef),
        );
    }
}

/// Broadcast-weights launcher for [`quantize_dct8_kernel_broadcast_w`].
/// `weights` is exactly 64 f32 (one DCT8 quant matrix); the kernel
/// broadcasts it across all blocks. Saves
/// `(num_blocks - 1) * 64 * 4` bytes of GPU memory and
/// `(num_blocks - 1) * 64 * 4` bytes of upload traffic.
pub fn quantize_dct8_broadcast_w<R: Runtime>(
    client: &ComputeClient<R>,
    coeffs: Handle,
    weights: Handle, // 64 f32
    qac_qm: Handle,
    thresholds: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n_coef = (num_blocks as usize) * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        quantize_dct8_kernel_broadcast_w::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(coeffs, n_coef),
            ArrayArg::from_raw_parts(weights, 64),
            ArrayArg::from_raw_parts(qac_qm, num_blocks as usize),
            ArrayArg::from_raw_parts(thresholds, 4),
            ArrayArg::from_raw_parts(output, n_coef),
        );
    }
}
