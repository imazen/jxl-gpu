// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launcher for the generic per-coefficient dequant kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::dequant_simple::dequant_simple_kernel;

pub fn dequant_simple<R: Runtime>(
    client: &ComputeClient<R>,
    quant: Handle,
    weights: Handle,
    output: Handle,
    num_blocks: u32,
    block_size: u32,
) {
    let n = (num_blocks as usize) * (block_size as usize);
    let cubes = num_blocks.max(1);
    unsafe {
        dequant_simple_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(quant, n),
            ArrayArg::from_raw_parts(weights, n),
            ArrayArg::from_raw_parts(output, n),
            block_size,
        );
    }
}
