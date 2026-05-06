// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for per-block DCT8 quantization with dead-zone.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::quantize::quantize_dct8_kernel;

pub fn quantize_dct8<R: Runtime>(
    client: &ComputeClient<R>,
    coeffs: Handle,
    weights: Handle,
    qac_qm: Handle,
    thresholds: Handle, // 4 f32
    output: Handle,     // num_blocks * 64 i32
    num_blocks: u32,
) {
    let n_coef = (num_blocks as usize) * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        quantize_dct8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(coeffs, n_coef),
            ArrayArg::from_raw_parts(weights, n_coef),
            ArrayArg::from_raw_parts(qac_qm, num_blocks as usize),
            ArrayArg::from_raw_parts(thresholds, 4),
            ArrayArg::from_raw_parts(output, n_coef),
        );
    }
}
