// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for `cfl_collect_kernel`.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::cfl_collect::cfl_collect_kernel;

/// 64 threads per cube — one per block slot in tile (TILE_DIM_IN_BLOCKS² = 64).
const CUBE_DIM: u32 = 64;

#[allow(clippy::too_many_arguments)]
pub fn cfl_collect<R: Runtime>(
    client: &ComputeClient<R>,
    dct_y: Handle,
    dct_x: Handle,
    dct_b: Handle,
    inv_qm_x: Handle,
    inv_qm_b: Handle,
    m_yx: Handle,
    s_x: Handle,
    m_yb: Handle,
    s_b: Handle,
    num_tiles: u32,
    in_floats: usize, // gpu_num_blocks * 64
    out_floats: usize, // num_tiles * 4096
    xsize_blocks: u32,
    ysize_blocks: u32,
    gpu_xsize_blocks: u32,
    xsize_tiles: u32,
) {
    unsafe {
        cfl_collect_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(num_tiles.max(1), 1, 1),
            CubeDim::new_1d(CUBE_DIM),
            ArrayArg::from_raw_parts(dct_y, in_floats),
            ArrayArg::from_raw_parts(dct_x, in_floats),
            ArrayArg::from_raw_parts(dct_b, in_floats),
            ArrayArg::from_raw_parts(inv_qm_x, 64),
            ArrayArg::from_raw_parts(inv_qm_b, 64),
            ArrayArg::from_raw_parts(m_yx, out_floats),
            ArrayArg::from_raw_parts(s_x, out_floats),
            ArrayArg::from_raw_parts(m_yb, out_floats),
            ArrayArg::from_raw_parts(s_b, out_floats),
            xsize_blocks,
            ysize_blocks,
            gpu_xsize_blocks,
            xsize_tiles,
        );
    }
}
