// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for `gab_smooth_3ch_kernel` (3-channel fused gab smooth).

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::gab_3ch::gab_smooth_3ch_kernel;

const TPB: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn gab_smooth_3ch<R: Runtime>(
    client: &ComputeClient<R>,
    input_x: Handle,
    input_y: Handle,
    input_b: Handle,
    output_x: Handle,
    output_y: Handle,
    output_b: Handle,
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
        gab_smooth_3ch_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(input_x, nu),
            ArrayArg::from_raw_parts(input_y, nu),
            ArrayArg::from_raw_parts(input_b, nu),
            ArrayArg::from_raw_parts(output_x, nu),
            ArrayArg::from_raw_parts(output_y, nu),
            ArrayArg::from_raw_parts(output_b, nu),
            width,
            height,
            w_center,
            w1,
            w2,
        );
    }
}
