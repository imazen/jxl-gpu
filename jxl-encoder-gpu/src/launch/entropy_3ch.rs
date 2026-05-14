// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for `entropy_coeffs_pixel_3ch_kernel`.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::entropy_3ch::entropy_coeffs_pixel_3ch_kernel;

const CUBE_DIM: u32 = 64;

#[allow(clippy::too_many_arguments)]
pub fn entropy_coeffs_pixel_3ch<R: Runtime>(
    client: &ComputeClient<R>,
    block_x: Handle,
    block_y: Handle,
    block_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    inv_weights_x: Handle,
    inv_weights_y: Handle,
    inv_weights_b: Handle,
    error_x: Handle,
    error_y: Handle,
    error_b: Handle,
    out_x: Handle,
    out_y: Handle,
    out_b: Handle,
    num_blocks: u32,
    n_per_block: u32,
    cmap_factor_x: f32,
    cmap_factor_b: f32,
    quant_x: f32,
    quant_y: f32,
    quant_b: f32,
    k_cost_delta: f32,
) {
    let cubes = num_blocks.div_ceil(CUBE_DIM).max(1);
    let nb = num_blocks as usize;
    let n_per = n_per_block as usize;
    let total = nb * n_per;
    unsafe {
        entropy_coeffs_pixel_3ch_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(CUBE_DIM),
            ArrayArg::from_raw_parts(block_x, total),
            ArrayArg::from_raw_parts(block_y, total),
            ArrayArg::from_raw_parts(block_b, total),
            ArrayArg::from_raw_parts(weights_x, n_per),
            ArrayArg::from_raw_parts(weights_y, n_per),
            ArrayArg::from_raw_parts(weights_b, n_per),
            ArrayArg::from_raw_parts(inv_weights_x, n_per),
            ArrayArg::from_raw_parts(inv_weights_y, n_per),
            ArrayArg::from_raw_parts(inv_weights_b, n_per),
            ArrayArg::from_raw_parts(error_x, total),
            ArrayArg::from_raw_parts(error_y, total),
            ArrayArg::from_raw_parts(error_b, total),
            ArrayArg::from_raw_parts(out_x, nb * 4),
            ArrayArg::from_raw_parts(out_y, nb * 4),
            ArrayArg::from_raw_parts(out_b, nb * 4),
            n_per_block,
            cmap_factor_x,
            cmap_factor_b,
            quant_x,
            quant_y,
            quant_b,
            k_cost_delta,
        );
    }
}
