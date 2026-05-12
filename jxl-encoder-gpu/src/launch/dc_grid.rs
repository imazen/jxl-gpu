// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launcher for the per-(8×8) block DC-grid kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::dc_grid::dc_grid_8x8_kernel;

const TPB: u32 = 256;

pub fn dc_grid_8x8<R: Runtime>(
    client: &ComputeClient<R>,
    plane: Handle,
    output: Handle,
    width: u32,
    height: u32,
) {
    assert!(
        width.is_multiple_of(8),
        "width {width} must be multiple of 8"
    );
    assert!(
        height.is_multiple_of(8),
        "height {height} must be multiple of 8"
    );
    let blocks_per_row = width / 8;
    let blocks_per_col = height / 8;
    let n_blocks = blocks_per_row * blocks_per_col;
    let n_pixels = (width as usize) * (height as usize);
    let cubes = n_blocks.div_ceil(TPB).max(1);
    unsafe {
        dc_grid_8x8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(plane, n_pixels),
            ArrayArg::from_raw_parts(output, n_blocks as usize),
            width,
            blocks_per_row,
            n_blocks,
        );
    }
}
