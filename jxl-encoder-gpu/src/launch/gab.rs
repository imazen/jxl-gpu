// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launcher for the 3x3 gab smooth kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::gab::gab_smooth_kernel;

const TPB: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn gab_smooth<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    width: u32,
    height: u32,
    w_center: f32,
    w1: f32,
    w2: f32,
) {
    let n = width * height;
    let nu = n as usize;
    let cubes = n.div_ceil(TPB).max(1);
    unsafe {
        gab_smooth_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(input, nu),
            ArrayArg::from_raw_parts(output, nu),
            width,
            height,
            w_center,
            w1,
            w2,
        );
    }
}
