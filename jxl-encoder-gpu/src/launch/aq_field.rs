// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for the 8×8 mask-mean reduction kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::aq_field::block_mask_mean_kernel;

const TPB: u32 = 256;

/// Launch the mask-mean reduction. `mask` is the per-pixel f32
/// masking field at `padded_width × padded_height`. `out` is sized
/// `(padded_width / 8) * (padded_height / 8)` f32 block means.
///
/// `width` / `height` are the *unpadded* image dims; the kernel
/// uses them for edge clipping (matches the CPU loop's
/// `if y >= h { break; }`).
#[allow(clippy::too_many_arguments)]
pub fn block_mask_mean<R: Runtime>(
    client: &ComputeClient<R>,
    mask: Handle,
    out: Handle,
    width: u32,
    height: u32,
    padded_width: u32,
    padded_height: u32,
) {
    let blocks_per_row = padded_width / 8;
    let blocks_per_col = padded_height / 8;
    let n_out = (blocks_per_row as usize) * (blocks_per_col as usize);
    let n_mask = (padded_width as usize) * (padded_height as usize);
    let cubes = (n_out as u32).div_ceil(TPB).max(1);
    unsafe {
        block_mask_mean_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(mask, n_mask),
            ArrayArg::from_raw_parts(out, n_out),
            width,
            height,
            padded_width,
            blocks_per_row,
            blocks_per_col,
        );
    }
}
