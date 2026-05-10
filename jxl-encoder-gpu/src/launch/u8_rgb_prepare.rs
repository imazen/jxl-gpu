// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Host-side launcher for the fused u8 sRGB → planar f32 linear +
//! pad kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::u8_rgb_prepare::u8_rgb_to_linear_planar_padded_kernel;

const TPB: u32 = 256;

/// Launch the fused kernel. See
/// [`fn@crate::kernels::u8_rgb_prepare::u8_rgb_to_linear_planar_padded_kernel`]
/// for the layout contract.
///
/// `src_handle` must point at `src_width * src_height * 3` u8 bytes
/// (interleaved RGB). Each `out_*_handle` must point at
/// `padded_width * padded_height * 4` bytes (planar f32).
#[allow(clippy::too_many_arguments)]
pub fn u8_rgb_to_linear_planar_padded<R: Runtime>(
    client: &ComputeClient<R>,
    src: Handle,
    out_r: Handle,
    out_g: Handle,
    out_b: Handle,
    src_width: u32,
    src_height: u32,
    padded_width: u32,
    padded_height: u32,
) {
    let n_src = (src_width as usize) * (src_height as usize) * 3;
    let n_out = (padded_width as usize) * (padded_height as usize);
    let cubes = (n_out as u32).div_ceil(TPB).max(1);
    unsafe {
        u8_rgb_to_linear_planar_padded_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(src, n_src),
            ArrayArg::from_raw_parts(out_r, n_out),
            ArrayArg::from_raw_parts(out_g, n_out),
            ArrayArg::from_raw_parts(out_b, n_out),
            src_width,
            src_height,
            padded_width,
        );
    }
}
