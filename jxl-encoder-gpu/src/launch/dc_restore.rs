// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Host-side launcher for the per-block DC restore kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::dc_restore::restore_dc_kernel;

const TPB: u32 = 128;

pub fn restore_dc<R: Runtime>(
    client: &ComputeClient<R>,
    src: Handle,
    dst: Handle,
    coeffs_per_block: u32,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * (coeffs_per_block as usize);
    let cubes = num_blocks.div_ceil(TPB).max(1);
    unsafe {
        restore_dc_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(src, n),
            ArrayArg::from_raw_parts(dst, n),
            coeffs_per_block,
            num_blocks,
        );
    }
}
