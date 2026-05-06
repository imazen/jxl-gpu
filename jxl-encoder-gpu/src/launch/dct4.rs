// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launchers for DCT4-based full transforms (4x4, 4x8, 8x4).

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::dct4::{
    dct_4x4_full_kernel, dct_4x8_full_kernel, dct_8x4_full_kernel, idct_4x4_full_kernel,
    idct_4x8_full_kernel, idct_8x4_full_kernel,
};

macro_rules! launch_64 {
    ($name:ident, $kernel:ident) => {
        pub fn $name<R: Runtime>(
            client: &ComputeClient<R>,
            input: Handle,
            output: Handle,
            num_blocks: u32,
        ) {
            let n = (num_blocks as usize) * 64;
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

launch_64!(dct_4x4_full, dct_4x4_full_kernel);
launch_64!(dct_4x8_full, dct_4x8_full_kernel);
launch_64!(dct_8x4_full, dct_8x4_full_kernel);
launch_64!(idct_4x4_full, idct_4x4_full_kernel);
launch_64!(idct_4x8_full, idct_4x8_full_kernel);
launch_64!(idct_8x4_full, idct_8x4_full_kernel);
