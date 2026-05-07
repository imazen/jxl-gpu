// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Host-side launchers for the gather/scatter kernels.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::gather::{gather_blocks_kernel, scatter_blocks_kernel};

const TPB: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn gather_blocks<R: Runtime>(
    client: &ComputeClient<R>,
    plane: Handle,
    output: Handle,
    plane_n: usize,
    output_n: usize,
    width: u32,
    blocks_per_row: u32,
    tile_w: u32,
    tile_h: u32,
) {
    let total = output_n as u32;
    let cubes = total.div_ceil(TPB).max(1);
    unsafe {
        gather_blocks_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(plane, plane_n),
            ArrayArg::from_raw_parts(output, output_n),
            width,
            blocks_per_row,
            tile_w,
            tile_h,
            total,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn scatter_blocks<R: Runtime>(
    client: &ComputeClient<R>,
    blocks: Handle,
    plane: Handle,
    blocks_n: usize,
    plane_n: usize,
    width: u32,
    blocks_per_row: u32,
    tile_w: u32,
    tile_h: u32,
) {
    let total = blocks_n as u32;
    let cubes = total.div_ceil(TPB).max(1);
    unsafe {
        scatter_blocks_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(blocks, blocks_n),
            ArrayArg::from_raw_parts(plane, plane_n),
            width,
            blocks_per_row,
            tile_w,
            tile_h,
            total,
        );
    }
}
