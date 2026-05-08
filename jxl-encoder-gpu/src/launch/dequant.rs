// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for per-block DCT8 dequantization with CfL restore.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::dequant::{dequant_dct8_kernel, dequant_dct8_kernel_broadcast_w};

#[allow(clippy::too_many_arguments)]
pub fn dequant_dct8<R: Runtime>(
    client: &ComputeClient<R>,
    quant_x: Handle,
    quant_y: Handle,
    quant_b: Handle,
    weights_x: Handle,
    weights_y: Handle,
    weights_b: Handle,
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    x_factor: Handle,
    b_factor: Handle,
    out_x: Handle,
    out_y: Handle,
    out_b: Handle,
    num_blocks: u32,
) {
    let nb = num_blocks as usize;
    let n_coef = nb * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        dequant_dct8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(quant_x, n_coef),
            ArrayArg::from_raw_parts(quant_y, n_coef),
            ArrayArg::from_raw_parts(quant_b, n_coef),
            ArrayArg::from_raw_parts(weights_x, n_coef),
            ArrayArg::from_raw_parts(weights_y, n_coef),
            ArrayArg::from_raw_parts(weights_b, n_coef),
            ArrayArg::from_raw_parts(qac_qm_x, nb),
            ArrayArg::from_raw_parts(qac_qm_y, nb),
            ArrayArg::from_raw_parts(qac_qm_b, nb),
            ArrayArg::from_raw_parts(x_factor, nb),
            ArrayArg::from_raw_parts(b_factor, nb),
            ArrayArg::from_raw_parts(out_x, n_coef),
            ArrayArg::from_raw_parts(out_y, n_coef),
            ArrayArg::from_raw_parts(out_b, n_coef),
        );
    }
}

/// Broadcast-weights launcher for
/// [`dequant_dct8_kernel_broadcast_w`]. `weights_x/y/b` are each
/// exactly 64 f32 (one DCT8 quant matrix per channel); the kernel
/// broadcasts them across all blocks. Saves
/// `3 * (num_blocks - 1) * 64 * 4` bytes of GPU memory and upload
/// traffic.
#[allow(clippy::too_many_arguments)]
pub fn dequant_dct8_broadcast_w<R: Runtime>(
    client: &ComputeClient<R>,
    quant_x: Handle,
    quant_y: Handle,
    quant_b: Handle,
    weights_x: Handle, // 64 f32
    weights_y: Handle, // 64 f32
    weights_b: Handle, // 64 f32
    qac_qm_x: Handle,
    qac_qm_y: Handle,
    qac_qm_b: Handle,
    x_factor: Handle,
    b_factor: Handle,
    out_x: Handle,
    out_y: Handle,
    out_b: Handle,
    num_blocks: u32,
) {
    let nb = num_blocks as usize;
    let n_coef = nb * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        dequant_dct8_kernel_broadcast_w::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(quant_x, n_coef),
            ArrayArg::from_raw_parts(quant_y, n_coef),
            ArrayArg::from_raw_parts(quant_b, n_coef),
            ArrayArg::from_raw_parts(weights_x, 64),
            ArrayArg::from_raw_parts(weights_y, 64),
            ArrayArg::from_raw_parts(weights_b, 64),
            ArrayArg::from_raw_parts(qac_qm_x, nb),
            ArrayArg::from_raw_parts(qac_qm_y, nb),
            ArrayArg::from_raw_parts(qac_qm_b, nb),
            ArrayArg::from_raw_parts(x_factor, nb),
            ArrayArg::from_raw_parts(b_factor, nb),
            ArrayArg::from_raw_parts(out_x, n_coef),
            ArrayArg::from_raw_parts(out_y, n_coef),
            ArrayArg::from_raw_parts(out_b, n_coef),
        );
    }
}
