// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for `cfl_quantize_dct8_kernel`.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::cfl_quantize::cfl_quantize_dct8_kernel;

const TPB: u32 = 64;

/// Fused CfL-subtract + DCT8 quantize for one chroma channel.
/// All `Handle` arguments live in GPU memory; per-block scalar
/// arrays (`qac_qm_x`, `qac_qm_y`, `x_factor_per_block`) are length
/// `num_blocks` floats each.
#[allow(clippy::too_many_arguments)]
pub fn cfl_quantize_dct8<R: Runtime>(
    client: &ComputeClient<R>,
    x_orig: Handle,
    quant_ac_y: Handle,
    weights_x: Handle,
    weights_y: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    x_factor_per_block: Handle,
    thresholds: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n_blocks = num_blocks as usize;
    let n_coeffs = n_blocks * 64;
    let cubes = num_blocks.div_ceil(TPB).max(1);
    unsafe {
        cfl_quantize_dct8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(x_orig, n_coeffs),
            ArrayArg::from_raw_parts(quant_ac_y, n_coeffs),
            ArrayArg::from_raw_parts(weights_x, 64),
            ArrayArg::from_raw_parts(weights_y, 64),
            ArrayArg::from_raw_parts(qac_qm_x, n_blocks),
            ArrayArg::from_raw_parts(qac_qm_y, n_blocks),
            ArrayArg::from_raw_parts(x_factor_per_block, n_blocks),
            ArrayArg::from_raw_parts(thresholds, 4),
            ArrayArg::from_raw_parts(output, n_coeffs),
        );
    }
}
