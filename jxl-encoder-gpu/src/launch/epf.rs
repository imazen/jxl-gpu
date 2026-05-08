// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launchers for pad_plane + EPF kernels.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::epf::{epf_step0_kernel, epf_step1_kernel, epf_step2_kernel, pad_plane_kernel};

const TPB: u32 = 256;

pub fn pad_plane<R: Runtime>(
    client: &ComputeClient<R>,
    src: Handle,
    dst: Handle,
    width: u32,
    height: u32,
    pad: u32,
) {
    let src_n = (width as usize) * (height as usize);
    let dst_w = width + 2 * pad;
    let dst_h = height + 2 * pad;
    let dst_n = (dst_w as usize) * (dst_h as usize);
    let cubes = (dst_n as u32).div_ceil(TPB).max(1);
    unsafe {
        pad_plane_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(src, src_n),
            ArrayArg::from_raw_parts(dst, dst_n),
            width,
            height,
            pad,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn epf_step0<R: Runtime>(
    client: &ComputeClient<R>,
    in_x: Handle,
    in_y: Handle,
    in_b: Handle,
    out_x: Handle,
    out_y: Handle,
    out_b: Handle,
    inv_sigma: Handle,
    width: u32,
    height: u32,
    xsize_blocks: u32,
    ysize_blocks: u32,
    pad: u32,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    let in_stride = width + 2 * pad;
    let in_n = (in_stride as usize) * ((height + 2 * pad) as usize);
    let out_n = (width as usize) * (height as usize);
    let sigma_n = (xsize_blocks as usize) * (ysize_blocks as usize);
    let cubes = (out_n as u32).div_ceil(TPB).max(1);
    unsafe {
        epf_step0_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(in_x, in_n),
            ArrayArg::from_raw_parts(in_y, in_n),
            ArrayArg::from_raw_parts(in_b, in_n),
            ArrayArg::from_raw_parts(out_x, out_n),
            ArrayArg::from_raw_parts(out_y, out_n),
            ArrayArg::from_raw_parts(out_b, out_n),
            ArrayArg::from_raw_parts(inv_sigma, sigma_n),
            width,
            height,
            xsize_blocks,
            in_stride,
            pad,
            sigma_scale,
            border_sigma_mul,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn epf_step1<R: Runtime>(
    client: &ComputeClient<R>,
    in_x: Handle,
    in_y: Handle,
    in_b: Handle,
    out_x: Handle,
    out_y: Handle,
    out_b: Handle,
    inv_sigma: Handle,
    width: u32,
    height: u32,
    xsize_blocks: u32,
    ysize_blocks: u32,
    pad: u32,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    let in_stride = width + 2 * pad;
    let in_n = (in_stride as usize) * ((height + 2 * pad) as usize);
    let out_n = (width as usize) * (height as usize);
    let sigma_n = (xsize_blocks as usize) * (ysize_blocks as usize);
    let cubes = (out_n as u32).div_ceil(TPB).max(1);
    unsafe {
        epf_step1_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(in_x, in_n),
            ArrayArg::from_raw_parts(in_y, in_n),
            ArrayArg::from_raw_parts(in_b, in_n),
            ArrayArg::from_raw_parts(out_x, out_n),
            ArrayArg::from_raw_parts(out_y, out_n),
            ArrayArg::from_raw_parts(out_b, out_n),
            ArrayArg::from_raw_parts(inv_sigma, sigma_n),
            width,
            height,
            xsize_blocks,
            in_stride,
            pad,
            sigma_scale,
            border_sigma_mul,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn epf_step2<R: Runtime>(
    client: &ComputeClient<R>,
    in_x: Handle,
    in_y: Handle,
    in_b: Handle,
    out_x: Handle,
    out_y: Handle,
    out_b: Handle,
    inv_sigma: Handle,
    width: u32,
    height: u32,
    xsize_blocks: u32,
    ysize_blocks: u32,
    pad: u32,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    let in_stride = width + 2 * pad;
    let in_n = (in_stride as usize) * ((height + 2 * pad) as usize);
    let out_n = (width as usize) * (height as usize);
    let sigma_n = (xsize_blocks as usize) * (ysize_blocks as usize);
    let cubes = (out_n as u32).div_ceil(TPB).max(1);
    unsafe {
        epf_step2_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(in_x, in_n),
            ArrayArg::from_raw_parts(in_y, in_n),
            ArrayArg::from_raw_parts(in_b, in_n),
            ArrayArg::from_raw_parts(out_x, out_n),
            ArrayArg::from_raw_parts(out_y, out_n),
            ArrayArg::from_raw_parts(out_b, out_n),
            ArrayArg::from_raw_parts(inv_sigma, sigma_n),
            width,
            height,
            xsize_blocks,
            in_stride,
            pad,
            sigma_scale,
            border_sigma_mul,
        );
    }
}
