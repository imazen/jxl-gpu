// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launchers for entropy_coeffs kernels.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::entropy::{
    entropy_coeffs_coeff_kernel, entropy_coeffs_pixel_kernel,
    entropy_coeffs_pixel_kernel_broadcast_w,
};

#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_pixel<R: Runtime>(
    client: &ComputeClient<R>,
    block_c: Handle,
    block_y: Handle,
    weights: Handle,
    inv_weights: Handle,
    error_coeffs: Handle,
    output: Handle,
    num_blocks: u32,
    n: u32,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
) {
    let n_per = n as usize;
    let n_total = (num_blocks as usize) * n_per;
    let cubes = num_blocks.max(1);
    unsafe {
        entropy_coeffs_pixel_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(block_c, n_total),
            ArrayArg::from_raw_parts(block_y, n_total),
            ArrayArg::from_raw_parts(weights, n_total),
            ArrayArg::from_raw_parts(inv_weights, n_total),
            ArrayArg::from_raw_parts(error_coeffs, n_total),
            ArrayArg::from_raw_parts(output, (num_blocks as usize) * 4),
            n,
            cmap_factor,
            quant,
            k_cost_delta,
        );
    }
}

/// Broadcast-weights launcher for
/// [`entropy_coeffs_pixel_kernel_broadcast_w`]. `weights` and
/// `inv_weights` are each exactly `n` f32 (one quant matrix and its
/// inverse), broadcast across all blocks.
#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_pixel_broadcast_w<R: Runtime>(
    client: &ComputeClient<R>,
    block_c: Handle,
    block_y: Handle,
    weights: Handle,     // n f32
    inv_weights: Handle, // n f32
    error_coeffs: Handle,
    output: Handle,
    num_blocks: u32,
    n: u32,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
) {
    let n_per = n as usize;
    let n_total = (num_blocks as usize) * n_per;
    let cubes = num_blocks.max(1);
    unsafe {
        entropy_coeffs_pixel_kernel_broadcast_w::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(block_c, n_total),
            ArrayArg::from_raw_parts(block_y, n_total),
            ArrayArg::from_raw_parts(weights, n_per),
            ArrayArg::from_raw_parts(inv_weights, n_per),
            ArrayArg::from_raw_parts(error_coeffs, n_total),
            ArrayArg::from_raw_parts(output, (num_blocks as usize) * 4),
            n,
            cmap_factor,
            quant,
            k_cost_delta,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_coeff<R: Runtime>(
    client: &ComputeClient<R>,
    block_c: Handle,
    block_y: Handle,
    inv_weights: Handle,
    output: Handle,
    num_blocks: u32,
    n: u32,
    cmap_factor: f32,
    quant: f32,
    k_cost_delta: f32,
    k_cost2: f32,
) {
    let n_per = n as usize;
    let n_total = (num_blocks as usize) * n_per;
    let cubes = num_blocks.max(1);
    unsafe {
        entropy_coeffs_coeff_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(block_c, n_total),
            ArrayArg::from_raw_parts(block_y, n_total),
            ArrayArg::from_raw_parts(inv_weights, n_total),
            ArrayArg::from_raw_parts(output, (num_blocks as usize) * 4),
            n,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
        );
    }
}
