// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launchers for XYB ↔ Linear RGB kernels.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::xyb::{xyb_forward_kernel, xyb_inverse_kernel};

/// Threads per block. 256 is a safe default across CUDA / WGPU / HIP.
const TPB: u32 = 256;

#[inline]
fn cube_count_1d(n: u32) -> CubeCount {
    let cubes = n.div_ceil(TPB).max(1);
    CubeCount::Static(cubes, 1, 1)
}

/// Launch the forward XYB kernel.
///
/// All six handles must point to f32 buffers of at least `n` elements.
/// Caller is responsible for upload + download.
///
/// # Safety
///
/// Wraps `launch_unchecked` — same safety contract: the kernel must not
/// read out-of-bounds (the kernel itself uses `terminate!()` for the
/// tail thread).
#[allow(clippy::too_many_arguments)]
pub fn xyb_forward<R: Runtime>(
    client: &ComputeClient<R>,
    r: Handle,
    g: Handle,
    b: Handle,
    x_out: Handle,
    y_out: Handle,
    b_out: Handle,
    n: u32,
) {
    let nu = n as usize;
    let count = cube_count_1d(n);
    let dim = CubeDim::new_1d(TPB);
    unsafe {
        xyb_forward_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(r, nu),
            ArrayArg::from_raw_parts(g, nu),
            ArrayArg::from_raw_parts(b, nu),
            ArrayArg::from_raw_parts(x_out, nu),
            ArrayArg::from_raw_parts(y_out, nu),
            ArrayArg::from_raw_parts(b_out, nu),
        );
    }
}

/// Launch the inverse XYB → planar linear RGB kernel.
#[allow(clippy::too_many_arguments)]
pub fn xyb_inverse<R: Runtime>(
    client: &ComputeClient<R>,
    xyb_x: Handle,
    xyb_y: Handle,
    xyb_b: Handle,
    out_r: Handle,
    out_g: Handle,
    out_b: Handle,
    n: u32,
) {
    let nu = n as usize;
    let count = cube_count_1d(n);
    let dim = CubeDim::new_1d(TPB);
    unsafe {
        xyb_inverse_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(xyb_x, nu),
            ArrayArg::from_raw_parts(xyb_y, nu),
            ArrayArg::from_raw_parts(xyb_b, nu),
            ArrayArg::from_raw_parts(out_r, nu),
            ArrayArg::from_raw_parts(out_g, nu),
            ArrayArg::from_raw_parts(out_b, nu),
        );
    }
}
