// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for per_block_modulations.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::adaptive_quant::{compute_pre_erosion_kernel, per_block_modulations_kernel};

const TPB: u32 = 64;

/// Caller computes pre_erosion_w/h from tile bounds and pre-allocates
/// output. See kernel docs for the formula.
#[allow(clippy::too_many_arguments)]
pub fn compute_pre_erosion<R: Runtime>(
    client: &ComputeClient<R>,
    xyb_y: Handle,
    output: Handle,
    xyb_y_n: usize,
    width: u32,
    height: u32,
    x0: u32,
    y_start: u32,
    pre_erosion_w: u32,
    pre_erosion_h: u32,
) {
    let out_n = (pre_erosion_w as usize) * (pre_erosion_h as usize);
    let cubes = (out_n as u32).div_ceil(TPB).max(1);
    unsafe {
        compute_pre_erosion_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(xyb_y, xyb_y_n),
            ArrayArg::from_raw_parts(output, out_n),
            width,
            height,
            x0,
            y_start,
            pre_erosion_w,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn per_block_modulations<R: Runtime>(
    client: &ComputeClient<R>,
    xyb_x: Handle,
    xyb_y: Handle,
    xyb_b: Handle,
    aq_map: Handle,
    xyb_n: usize,
    aq_map_n: usize,
    stride: u32,
    aq_map_stride: u32,
    rect_x0_blocks: u32,
    rect_y0_blocks: u32,
    rect_w_blocks: u32,
    rect_h_blocks: u32,
    butteraugli_target: f32,
    scale: f32,
) {
    let n_threads = (rect_w_blocks * rect_h_blocks) as usize;
    let _ = aq_map_n;
    let cubes = (n_threads as u32).div_ceil(TPB).max(1);
    unsafe {
        per_block_modulations_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(xyb_x, xyb_n),
            ArrayArg::from_raw_parts(xyb_y, xyb_n),
            ArrayArg::from_raw_parts(xyb_b, xyb_n),
            ArrayArg::from_raw_parts(aq_map, aq_map_n),
            stride,
            aq_map_stride,
            rect_x0_blocks,
            rect_y0_blocks,
            rect_w_blocks,
            butteraugli_target,
            scale,
        );
    }
}
