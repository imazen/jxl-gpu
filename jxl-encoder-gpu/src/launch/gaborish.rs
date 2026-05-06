// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launcher for the 5x5 gaborish inverse kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::gaborish::gaborish_5x5_kernel;

const TPB: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn gaborish_5x5<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    width: u32,
    height: u32,
    wc: f32,
    wr: f32,
    wd: f32,
    w_big_r: f32,
    wl: f32,
    w_big_d: f32,
) {
    let n = width * height;
    let nu = n as usize;
    let cubes = n.div_ceil(TPB).max(1);
    unsafe {
        gaborish_5x5_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(input, nu),
            ArrayArg::from_raw_parts(output, nu),
            width,
            height,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );
    }
}
