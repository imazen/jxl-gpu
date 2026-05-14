// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for `pixel_loss_3ch_kernel`.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::pixel_loss_3ch::pixel_loss_3ch_kernel;

const CUBE_DIM: u32 = 64;

#[allow(clippy::too_many_arguments)]
pub fn pixel_loss_3ch<R: Runtime>(
    client: &ComputeClient<R>,
    pixel_error_x: Handle,
    pixel_error_y: Handle,
    pixel_error_b: Handle,
    mask: Handle,
    mask_row_base: Handle,
    output_x: Handle,
    output_y: Handle,
    output_b: Handle,
    err_floats: usize, // num_blocks * block_width * block_height
    mask_floats: usize,
    n_blocks: u32,
    mask_stride: u32,
    mask_offset_x: f32,
    mask_offset_y: f32,
    mask_offset_b: f32,
    block_width: u32,
    block_height: u32,
) {
    let cubes = n_blocks.div_ceil(CUBE_DIM).max(1);
    let nb = n_blocks as usize;
    unsafe {
        pixel_loss_3ch_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(CUBE_DIM),
            ArrayArg::from_raw_parts(pixel_error_x, err_floats),
            ArrayArg::from_raw_parts(pixel_error_y, err_floats),
            ArrayArg::from_raw_parts(pixel_error_b, err_floats),
            ArrayArg::from_raw_parts(mask, mask_floats),
            ArrayArg::from_raw_parts(mask_row_base, nb),
            ArrayArg::from_raw_parts(output_x, nb),
            ArrayArg::from_raw_parts(output_y, nb),
            ArrayArg::from_raw_parts(output_b, nb),
            mask_stride,
            mask_offset_x,
            mask_offset_y,
            mask_offset_b,
            block_width,
            block_height,
        );
    }
}
