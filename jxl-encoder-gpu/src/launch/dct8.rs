// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launchers for 8x8 DCT/IDCT kernels.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::dct8::{dct_8x8_kernel, idct_8x8_kernel};

/// Forward 8x8 DCT for `num_blocks` contiguous 8x8 blocks.
///
/// Buffers must each be `num_blocks * 64` f32 elements.
pub fn dct_8x8<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        dct_8x8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}

/// Cooperative forward 8×8 DCT — 8 threads per block, one per row.
/// Same I/O contract as [`dct_8x8`] but uses cube_dim=8 instead of
/// cube_dim=1 for higher parallelism.
pub fn dct_8x8_coop<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        crate::kernels::dct8::dct_8x8_coop_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(8),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}

/// Cooperative inverse 8×8 DCT — 8 threads per block.
pub fn idct_8x8_coop<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        crate::kernels::dct8::idct_8x8_coop_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(8),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}

/// Inverse 8x8 DCT for `num_blocks` contiguous 8x8 blocks.
pub fn idct_8x8<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        idct_8x8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}
