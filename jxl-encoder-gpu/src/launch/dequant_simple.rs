// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launcher for the generic per-coefficient dequant kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::dequant_simple::{
    dequant_simple_kernel, dequant_simple_kernel_broadcast_w, dequant_strategy_kernel_broadcast_w,
    dequant_strategy_kernel_broadcast_w_dct8,
};

pub fn dequant_simple<R: Runtime>(
    client: &ComputeClient<R>,
    quant: Handle,
    weights: Handle,
    output: Handle,
    num_blocks: u32,
    block_size: u32,
) {
    let n = (num_blocks as usize) * (block_size as usize);
    let cubes = num_blocks.max(1);
    unsafe {
        dequant_simple_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(quant, n),
            ArrayArg::from_raw_parts(weights, n),
            ArrayArg::from_raw_parts(output, n),
            block_size,
        );
    }
}

/// Broadcast-weights launcher for
/// [`dequant_simple_kernel_broadcast_w`]. `weights` is exactly
/// `block_size` f32 (one quant matrix), broadcast across all blocks.
pub fn dequant_simple_broadcast_w<R: Runtime>(
    client: &ComputeClient<R>,
    quant: Handle,
    weights: Handle, // block_size f32
    output: Handle,
    num_blocks: u32,
    block_size: u32,
) {
    let n = (num_blocks as usize) * (block_size as usize);
    let cubes = num_blocks.max(1);
    unsafe {
        dequant_simple_kernel_broadcast_w::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(quant, n),
            ArrayArg::from_raw_parts(weights, block_size as usize),
            ArrayArg::from_raw_parts(output, n),
            block_size,
        );
    }
}

/// Launcher for [`dequant_strategy_kernel_broadcast_w`]: per-block qac
/// scaled dequant. `qac` is `num_blocks` f32. No bias correction (use
/// the `_dct8` variant for DCT8 channels).
#[allow(clippy::too_many_arguments)]
pub fn dequant_strategy_broadcast_w<R: Runtime>(
    client: &ComputeClient<R>,
    quant: Handle,
    weights: Handle, // block_size f32
    qac: Handle,     // num_blocks f32
    output: Handle,
    num_blocks: u32,
    block_size: u32,
) {
    let n = (num_blocks as usize) * (block_size as usize);
    let cubes = num_blocks.max(1);
    unsafe {
        dequant_strategy_kernel_broadcast_w::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(quant, n),
            ArrayArg::from_raw_parts(weights, block_size as usize),
            ArrayArg::from_raw_parts(qac, num_blocks as usize),
            ArrayArg::from_raw_parts(output, n),
            block_size,
        );
    }
}

/// Launcher for [`dequant_strategy_kernel_broadcast_w_dct8`]: same as
/// [`dequant_strategy_broadcast_w`] but applies `adjust_quant_bias`
/// per coefficient. `channel_bias` is the per-channel constant
/// (BIAS_X = 0.945_349_93, BIAS_Y = 0.929_945_5, BIAS_B = 0.950_064_9).
#[allow(clippy::too_many_arguments)]
pub fn dequant_strategy_broadcast_w_dct8<R: Runtime>(
    client: &ComputeClient<R>,
    quant: Handle,
    weights: Handle,
    qac: Handle,
    output: Handle,
    num_blocks: u32,
    block_size: u32,
    channel_bias: f32,
) {
    let n = (num_blocks as usize) * (block_size as usize);
    let cubes = num_blocks.max(1);
    unsafe {
        dequant_strategy_kernel_broadcast_w_dct8::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(quant, n),
            ArrayArg::from_raw_parts(weights, block_size as usize),
            ArrayArg::from_raw_parts(qac, num_blocks as usize),
            ArrayArg::from_raw_parts(output, n),
            block_size,
            channel_bias,
        );
    }
}
