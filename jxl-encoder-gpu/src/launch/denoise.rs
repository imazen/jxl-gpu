// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launcher for the per-pixel Wiener denoise kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::denoise::denoise_kernel;

const TPB: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn denoise<R: Runtime>(
    client: &ComputeClient<R>,
    orig: Handle,
    y_channel: Handle,
    noise_lut: Handle,
    output: Handle,
    width: u32,
    height: u32,
    denoise_scale: f32,
) {
    let n = width * height;
    let nu = n as usize;
    let cubes = n.div_ceil(TPB).max(1);
    unsafe {
        denoise_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(orig, nu),
            ArrayArg::from_raw_parts(y_channel, nu),
            ArrayArg::from_raw_parts(noise_lut, 8),
            ArrayArg::from_raw_parts(output, nu),
            width,
            height,
            denoise_scale,
        );
    }
}
