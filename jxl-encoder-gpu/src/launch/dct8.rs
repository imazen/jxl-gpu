// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launchers for 8x8 DCT/IDCT kernels.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::dct8::{
    dct_8x8_kernel, dequant_idct_dc_scatter_dct8_kernel,
    dequant_idct_dc_scatter_dct8_wide_kernel, idct_8x8_kernel,
    idct_8x8_set_dc_scatter_kernel,
};

/// Forward 8x8 DCT for `num_blocks` contiguous 8x8 blocks.
///
/// Buffers must each be `num_blocks * 64` f32 elements.
pub fn dct_8x8<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        dct_8x8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}

/// Wide-cube forward 8×8 DCT — `cube_dim=32` (one thread per
/// block, 32 blocks per cube). Better warp utilization than cube_dim=1.
pub fn dct_8x8_wide<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 64;
    let cubes = num_blocks.div_ceil(64).max(1);
    unsafe {
        crate::kernels::dct8::dct_8x8_wide_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}

/// Wide-cube inverse 8×8 DCT — `cube_dim=64`, one thread per block.
pub fn idct_8x8_wide<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 64;
    let cubes = num_blocks.div_ceil(64).max(1);
    unsafe {
        crate::kernels::dct8::idct_8x8_wide_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}

/// Cooperative forward 8×8 DCT — 8 threads per block, one per row.
/// Same I/O contract as [`dct_8x8`] but uses cube_dim=8 instead of
/// cube_dim=1 for higher parallelism.
pub fn dct_8x8_coop<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        crate::kernels::dct8::dct_8x8_coop_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(8),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}

/// Cooperative inverse 8×8 DCT — 8 threads per block.
pub fn idct_8x8_coop<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        crate::kernels::dct8::idct_8x8_coop_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(8),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}

/// Inverse 8x8 DCT for `num_blocks` contiguous 8x8 blocks.
pub fn idct_8x8<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    output: Handle,
    num_blocks: u32,
) {
    let n = (num_blocks as usize) * 64;
    let cubes = num_blocks.max(1);
    unsafe {
        idct_8x8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output, n),
        );
    }
}

/// Fused IDCT8 + DC-restore + indexed scatter. One launch instead of
/// three. See [`fn@crate::kernels::dct8::idct_8x8_set_dc_scatter_kernel`]
/// for the layout contract.
#[allow(clippy::too_many_arguments)]
pub fn idct_8x8_set_dc_scatter<R: Runtime>(
    client: &ComputeClient<R>,
    input: Handle,
    dc_grid: Handle,
    coords: Handle,
    plane: Handle,
    plane_n_pixels: usize,
    plane_width: u32,
    dc_grid_len: usize,
    dc_stride: u32,
    n_blocks: u32,
) {
    let n_coef = (n_blocks as usize) * 64;
    let cubes = n_blocks.max(1);
    unsafe {
        idct_8x8_set_dc_scatter_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(input, n_coef),
            ArrayArg::from_raw_parts(dc_grid, dc_grid_len),
            ArrayArg::from_raw_parts(coords, (n_blocks as usize) * 2),
            ArrayArg::from_raw_parts(plane, plane_n_pixels),
            plane_width,
            dc_stride,
            n_blocks,
        );
    }
}

/// 4-way fused: dequant + DC restore + IDCT8 + indexed scatter.
/// Single launch replacing four. See
/// [`fn@crate::kernels::dct8::dequant_idct_dc_scatter_dct8_kernel`]
/// for the full layout contract.
#[allow(clippy::too_many_arguments)]
pub fn dequant_idct_dc_scatter_dct8<R: Runtime>(
    client: &ComputeClient<R>,
    quant: Handle,
    weights: Handle,
    qac: Handle,
    dc_grid: Handle,
    coords: Handle,
    plane: Handle,
    plane_n_pixels: usize,
    plane_width: u32,
    dc_grid_len: usize,
    dc_stride: u32,
    n_blocks: u32,
    channel_bias: f32,
) {
    let n_coef = (n_blocks as usize) * 64;
    let cubes = n_blocks.max(1);
    unsafe {
        dequant_idct_dc_scatter_dct8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(quant, n_coef),
            ArrayArg::from_raw_parts(weights, 64),
            ArrayArg::from_raw_parts(qac, n_blocks as usize),
            ArrayArg::from_raw_parts(dc_grid, dc_grid_len),
            ArrayArg::from_raw_parts(coords, (n_blocks as usize) * 2),
            ArrayArg::from_raw_parts(plane, plane_n_pixels),
            plane_width,
            dc_stride,
            n_blocks,
            channel_bias,
        );
    }
}

/// Wide-cube variant of [`dequant_idct_dc_scatter_dct8`]: cube_dim=64
/// (one thread per block, 64 blocks per cube). Same I/O contract;
/// bit-identical math. See
/// [`fn@crate::kernels::dct8::dequant_idct_dc_scatter_dct8_wide_kernel`]
/// for the rationale.
#[allow(clippy::too_many_arguments)]
pub fn dequant_idct_dc_scatter_dct8_wide<R: Runtime>(
    client: &ComputeClient<R>,
    quant: Handle,
    weights: Handle,
    qac: Handle,
    dc_grid: Handle,
    coords: Handle,
    plane: Handle,
    plane_n_pixels: usize,
    plane_width: u32,
    dc_grid_len: usize,
    dc_stride: u32,
    n_blocks: u32,
    channel_bias: f32,
) {
    let n_coef = (n_blocks as usize) * 64;
    let cubes = n_blocks.div_ceil(64).max(1);
    unsafe {
        dequant_idct_dc_scatter_dct8_wide_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(quant, n_coef),
            ArrayArg::from_raw_parts(weights, 64),
            ArrayArg::from_raw_parts(qac, n_blocks as usize),
            ArrayArg::from_raw_parts(dc_grid, dc_grid_len),
            ArrayArg::from_raw_parts(coords, (n_blocks as usize) * 2),
            ArrayArg::from_raw_parts(plane, plane_n_pixels),
            plane_width,
            dc_stride,
            n_blocks,
            channel_bias,
        );
    }
}
