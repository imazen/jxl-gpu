// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for the AFV forward-compose kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::afv_compose::{
    afv_compose_forward_kernel, afv_compose_inverse_kernel, afv_unpack_inverse_kernel,
};

/// Launch the AFV forward-compose kernel with one thread per block.
///
/// All 4 buffer handles must already be GPU-resident:
/// - `afv_in` : `n_blocks * 16` f32 (output of AFV 4×4 DCT batch)
/// - `dct4_in`: `n_blocks * 16` f32 (output of raw DCT 4×4 batch)
/// - `dct4x8_in`: `n_blocks * 32` f32 (output of raw DCT 4×8 batch)
/// - `out`    : `n_blocks * 64` f32 (allocated by caller, written
///   here with composed + DC-packed AFV layout matching upstream
///   `afv_transform_from_pixels`)
pub fn afv_compose_forward<R: Runtime>(
    client: &ComputeClient<R>,
    afv_in: Handle,
    dct4_in: Handle,
    dct4x8_in: Handle,
    out: Handle,
    n_blocks: u32,
) {
    if n_blocks == 0 {
        return;
    }
    let cubes = n_blocks.div_ceil(64).max(1);
    unsafe {
        afv_compose_forward_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(afv_in, (n_blocks * 16) as usize),
            ArrayArg::from_raw_parts(dct4_in, (n_blocks * 16) as usize),
            ArrayArg::from_raw_parts(dct4x8_in, (n_blocks * 32) as usize),
            ArrayArg::from_raw_parts(out, (n_blocks * 64) as usize),
        );
    }
}

/// Launch the inverse-side unpack kernel: takes 64-coef AFV-layout
/// blocks (with packed DCs at [0], [1], [8]) and writes 3 sub-input
/// buffers (afv_out, dct4_out, dct4x8_out) ready for the inverse 4×4
/// / 4×4 / 4×8 DCT kernels.
pub fn afv_unpack_inverse<R: Runtime>(
    client: &ComputeClient<R>,
    coeffs: Handle,
    afv_out: Handle,
    dct4_out: Handle,
    dct4x8_out: Handle,
    n_blocks: u32,
) {
    if n_blocks == 0 {
        return;
    }
    let cubes = n_blocks.div_ceil(64).max(1);
    unsafe {
        afv_unpack_inverse_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(coeffs, (n_blocks * 64) as usize),
            ArrayArg::from_raw_parts(afv_out, (n_blocks * 16) as usize),
            ArrayArg::from_raw_parts(dct4_out, (n_blocks * 16) as usize),
            ArrayArg::from_raw_parts(dct4x8_out, (n_blocks * 32) as usize),
        );
    }
}

/// Launch the inverse-side compose kernel: takes 3 inverse-DCT pixel
/// sub-outputs and writes the 64-pixel block with corner mirroring
/// per `afv_kind` (0..3).
pub fn afv_compose_inverse<R: Runtime>(
    client: &ComputeClient<R>,
    afv_pixels: Handle,
    dct4_pixels: Handle,
    dct4x8_pixels: Handle,
    out: Handle,
    n_blocks: u32,
    afv_kind: u32,
) {
    if n_blocks == 0 {
        return;
    }
    debug_assert!(afv_kind < 4, "afv_kind must be 0..3 (was {afv_kind})");
    let cubes = n_blocks.div_ceil(64).max(1);
    unsafe {
        afv_compose_inverse_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(afv_pixels, (n_blocks * 16) as usize),
            ArrayArg::from_raw_parts(dct4_pixels, (n_blocks * 16) as usize),
            ArrayArg::from_raw_parts(dct4x8_pixels, (n_blocks * 32) as usize),
            ArrayArg::from_raw_parts(out, (n_blocks * 64) as usize),
            afv_kind,
        );
    }
}
