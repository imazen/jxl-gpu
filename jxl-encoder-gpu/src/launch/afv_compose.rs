// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for the AFV forward-compose kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::afv_compose::afv_compose_forward_kernel;

/// Launch the AFV forward-compose kernel with one thread per block.
///
/// All 4 buffer handles must already be GPU-resident:
/// - `afv_in` : `n_blocks * 16` f32 (output of AFV 4×4 DCT batch)
/// - `dct4_in`: `n_blocks * 16` f32 (output of raw DCT 4×4 batch)
/// - `dct4x8_in`: `n_blocks * 32` f32 (output of raw DCT 4×8 batch)
/// - `out`    : `n_blocks * 64` f32 (allocated by caller, written
///   here with composed + DC-packed AFV layout matching upstream
///   `afv_transform_from_pixels`)
pub fn afv_compose_forward<R: Runtime>(
    client: &ComputeClient<R>,
    afv_in: Handle,
    dct4_in: Handle,
    dct4x8_in: Handle,
    out: Handle,
    n_blocks: u32,
) {
    if n_blocks == 0 {
        return;
    }
    let cubes = n_blocks.div_ceil(64).max(1);
    unsafe {
        afv_compose_forward_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(afv_in, (n_blocks * 16) as usize),
            ArrayArg::from_raw_parts(dct4_in, (n_blocks * 16) as usize),
            ArrayArg::from_raw_parts(dct4x8_in, (n_blocks * 32) as usize),
            ArrayArg::from_raw_parts(out, (n_blocks * 64) as usize),
        );
    }
}
