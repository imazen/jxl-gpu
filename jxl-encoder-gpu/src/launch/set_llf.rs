// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launcher for the per-strategy LLF restore kernels.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::set_llf::{
    set_llf_dct16x16_indexed_kernel, set_llf_dct16x8_or_8x16_indexed_kernel,
};

const TPB: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn set_llf_dct16x16_indexed<R: Runtime>(
    client: &ComputeClient<R>,
    dc_grid: Handle,
    coords: Handle,
    dst: Handle,
    dc_grid_n: usize,
    dst_n: usize,
    dc_stride: u32,
    coeffs_per_block: u32,
    n_blocks: u32,
) {
    let cubes = n_blocks.div_ceil(TPB).max(1);
    unsafe {
        set_llf_dct16x16_indexed_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(dc_grid, dc_grid_n),
            ArrayArg::from_raw_parts(coords, (n_blocks as usize) * 2),
            ArrayArg::from_raw_parts(dst, dst_n),
            dc_stride,
            coeffs_per_block,
            n_blocks,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn set_llf_dct16x8_or_8x16_indexed<R: Runtime>(
    client: &ComputeClient<R>,
    dc_grid: Handle,
    coords: Handle,
    dst: Handle,
    dc_grid_n: usize,
    dst_n: usize,
    dc_stride: u32,
    dc_step: u32,
    coeffs_per_block: u32,
    n_blocks: u32,
) {
    let cubes = n_blocks.div_ceil(TPB).max(1);
    unsafe {
        set_llf_dct16x8_or_8x16_indexed_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(dc_grid, dc_grid_n),
            ArrayArg::from_raw_parts(coords, (n_blocks as usize) * 2),
            ArrayArg::from_raw_parts(dst, dst_n),
            dc_stride,
            dc_step,
            coeffs_per_block,
            n_blocks,
        );
    }
}
