// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for the per-block 8th-power pixel-loss kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::pixel_loss::pixel_loss_kernel;

#[allow(clippy::too_many_arguments)]
pub fn pixel_loss<R: Runtime>(
    client: &ComputeClient<R>,
    pixel_error: Handle,
    mask: Handle,
    mask_row_base: Handle, // u32 per block
    output: Handle,        // f64 per block
    num_blocks: u32,
    mask_len: usize,
    mask_stride: u32,
    mask_offset: f32,
    block_width: u32,
    block_height: u32,
) {
    let nb = num_blocks as usize;
    let n_err = nb * (block_width as usize) * (block_height as usize);
    let cubes = num_blocks.max(1);
    unsafe {
        pixel_loss_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(pixel_error, n_err),
            ArrayArg::from_raw_parts(mask, mask_len),
            ArrayArg::from_raw_parts(mask_row_base, nb),
            ArrayArg::from_raw_parts(output, nb),
            mask_stride,
            mask_offset,
            block_width,
            block_height,
        );
    }
}
