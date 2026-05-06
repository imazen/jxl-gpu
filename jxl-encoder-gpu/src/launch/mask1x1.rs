// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launcher for the per-pixel masking-field kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::mask1x1::mask1x1_kernel;

const TPB: u32 = 256;

pub fn mask1x1<R: Runtime>(
    client: &ComputeClient<R>,
    xyb_y: Handle,
    output: Handle,
    width: u32,
    height: u32,
) {
    let n = width * height;
    let nu = n as usize;
    let cubes = n.div_ceil(TPB).max(1);
    unsafe {
        mask1x1_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(xyb_y, nu),
            ArrayArg::from_raw_parts(output, nu),
            width,
            height,
        );
    }
}
