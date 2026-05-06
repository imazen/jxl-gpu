// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for the per-block L2 error reduction.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::block_l2::block_l2_kernel;

#[allow(clippy::too_many_arguments)]
pub fn block_l2<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    recon_x: Handle,
    recon_y: Handle,
    recon_b: Handle,
    mask1x1: Handle,
    output: Handle,
    xsize_blocks: u32,
    ysize_blocks: u32,
    padded_width: u32,
) {
    let n_blocks = (xsize_blocks * ysize_blocks) as usize;
    let n_pixels = (padded_width as usize) * (ysize_blocks as usize) * 8;
    let cubes = (xsize_blocks * ysize_blocks).max(1);
    unsafe {
        block_l2_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(orig_x, n_pixels),
            ArrayArg::from_raw_parts(orig_y, n_pixels),
            ArrayArg::from_raw_parts(orig_b, n_pixels),
            ArrayArg::from_raw_parts(recon_x, n_pixels),
            ArrayArg::from_raw_parts(recon_y, n_pixels),
            ArrayArg::from_raw_parts(recon_b, n_pixels),
            ArrayArg::from_raw_parts(mask1x1, n_pixels),
            ArrayArg::from_raw_parts(output, n_blocks),
            xsize_blocks,
            padded_width,
        );
    }
}
