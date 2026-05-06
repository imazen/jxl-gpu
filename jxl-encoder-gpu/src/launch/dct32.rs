// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launchers for DCT32 family kernels.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::dct32::{
    dct_16x32_kernel, dct_32x16_kernel, dct_32x32_kernel, idct_16x32_kernel, idct_32x16_kernel,
    idct_32x32_kernel,
};

macro_rules! launch_n {
    ($name:ident, $kernel:ident, $per_block:expr) => {
        pub fn $name<R: Runtime>(
            client: &ComputeClient<R>,
            input: Handle,
            output: Handle,
            num_blocks: u32,
        ) {
            let n = (num_blocks as usize) * $per_block;
            let cubes = num_blocks.max(1);
            unsafe {
                $kernel::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(1),
                    ArrayArg::from_raw_parts(input, n),
                    ArrayArg::from_raw_parts(output, n),
                );
            }
        }
    };
}

launch_n!(dct_32x32, dct_32x32_kernel, 1024);
launch_n!(idct_32x32, idct_32x32_kernel, 1024);
launch_n!(dct_32x16, dct_32x16_kernel, 512);
launch_n!(idct_32x16, idct_32x16_kernel, 512);
launch_n!(dct_16x32, dct_16x32_kernel, 512);
launch_n!(idct_16x32, idct_16x32_kernel, 512);
