// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for `fused_dct8_3ch_kernel`.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::fused_dct8_3ch::fused_dct8_3ch_kernel;

const WIDE_CUBE_DIM: u32 = 64;

#[allow(clippy::too_many_arguments)]
pub fn fused_dct8_3ch<R: Runtime>(
    client: &ComputeClient<R>,
    plane_x: Handle,
    plane_y: Handle,
    plane_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    x_factor_per_block: Handle,
    b_factor_per_block: Handle,
    thresholds_x: Handle,
    thresholds_y: Handle,
    thresholds_b: Handle,
    quant_ac_x: Handle,
    quant_ac_y: Handle,
    quant_ac_b: Handle,
    nzeros_x: Handle,
    nzeros_y: Handle,
    nzeros_b: Handle,
    plane_n: usize,
    n_blocks: u32,
    plane_stride: u32,
    xsize_blocks: u32,
) {
    let n_b = n_blocks as usize;
    let cubes = n_blocks.div_ceil(WIDE_CUBE_DIM).max(1);
    unsafe {
        fused_dct8_3ch_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(WIDE_CUBE_DIM),
            ArrayArg::from_raw_parts(plane_x, plane_n),
            ArrayArg::from_raw_parts(plane_y, plane_n),
            ArrayArg::from_raw_parts(plane_b, plane_n),
            ArrayArg::from_raw_parts(weights_x, 64),
            ArrayArg::from_raw_parts(weights_y, 64),
            ArrayArg::from_raw_parts(weights_b, 64),
            ArrayArg::from_raw_parts(qac_qm_x, n_b),
            ArrayArg::from_raw_parts(qac_qm_y, n_b),
            ArrayArg::from_raw_parts(qac_qm_b, n_b),
            ArrayArg::from_raw_parts(x_factor_per_block, n_b),
            ArrayArg::from_raw_parts(b_factor_per_block, n_b),
            ArrayArg::from_raw_parts(thresholds_x, 4),
            ArrayArg::from_raw_parts(thresholds_y, 4),
            ArrayArg::from_raw_parts(thresholds_b, 4),
            ArrayArg::from_raw_parts(quant_ac_x, n_b * 64),
            ArrayArg::from_raw_parts(quant_ac_y, n_b * 64),
            ArrayArg::from_raw_parts(quant_ac_b, n_b * 64),
            ArrayArg::from_raw_parts(nzeros_x, n_b),
            ArrayArg::from_raw_parts(nzeros_y, n_b),
            ArrayArg::from_raw_parts(nzeros_b, n_b),
            plane_stride,
            xsize_blocks,
        );
    }
}
