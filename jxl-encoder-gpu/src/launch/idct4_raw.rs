// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launchers for raw inverse 4×4 / 4×8 DCT kernels.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::idct4_raw::{idct_4x4_raw_kernel, idct_4x8_raw_kernel};

pub fn idct_4x4_raw<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 16;
    let cubes = num_blocks.max(1);
    unsafe {
        idct_4x4_raw_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}

pub fn idct_4x8_raw<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 32;
    let cubes = num_blocks.max(1);
    unsafe {
        idct_4x8_raw_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}
