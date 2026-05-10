// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launcher for the indexed per-block-buffer → plane scatter.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::indexed_scatter::indexed_scatter_kernel;

const TPB: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn indexed_scatter<R: Runtime>(
    client: &ComputeClient<R>,
    blocks: Handle,
    coords: Handle,
    plane: Handle,
    plane_width: u32,
    plane_n_pixels: usize,
    n_blocks: u32,
    tile_w: u32,
    tile_h: u32,
) {
    let total = n_blocks * tile_w * tile_h;
    let cubes = total.div_ceil(TPB).max(1);
    unsafe {
        indexed_scatter_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(blocks, total as usize),
            ArrayArg::from_raw_parts(coords, (n_blocks as usize) * 2),
            ArrayArg::from_raw_parts(plane, plane_n_pixels),
            plane_width,
            tile_w,
            tile_h,
            total,
        );
    }
}
