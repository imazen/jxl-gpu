// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launchers for AFV 4x4 forward + inverse DCT kernels.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::afv::{afv_dct_4x4_kernel, afv_idct_4x4_kernel};

pub fn afv_dct_4x4<R: Runtime>(
    client: &ComputeClient<R>,
    pixels: Handle,
    basis_t: Handle,
    coeffs: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 16;
    let cubes = num_blocks.max(1);
    unsafe {
        afv_dct_4x4_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(pixels, n),
            ArrayArg::from_raw_parts(basis_t, 256),
            ArrayArg::from_raw_parts(coeffs, n),
        );
    }
}

pub fn afv_idct_4x4<R: Runtime>(
    client: &ComputeClient<R>,
    coeffs: Handle,
    basis_t: Handle,
    pixels: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 16;
    let cubes = num_blocks.max(1);
    unsafe {
        afv_idct_4x4_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(coeffs, n),
            ArrayArg::from_raw_parts(basis_t, 256),
            ArrayArg::from_raw_parts(pixels, n),
        );
    }
}
