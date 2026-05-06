// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launchers for 16x16 DCT/IDCT kernels.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::dct16::{dct_16x16_kernel, idct_16x16_kernel};

pub fn dct_16x16<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 256;
    let cubes = num_blocks.max(1);
    unsafe {
        dct_16x16_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}

pub fn idct_16x16<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 256;
    let cubes = num_blocks.max(1);
    unsafe {
        idct_16x16_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}
