// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for the per-block 3-channel masked SSE reduction kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::sse_reduce::sse_reduce_3channel_kernel;

/// Launch the per-block SSE reduction with one thread per block.
/// All 7 input handles must already be GPU-resident (length
/// `n_blocks * 64` for the 6 pixel buffers + mask). `out` is
/// `n_blocks * 4` bytes (one f32 per block, allocated by caller).
#[allow(clippy::too_many_arguments)]
pub fn sse_reduce_3channel<R: Runtime>(
    client: &ComputeClient<R>,
    orig_x: Handle,
    orig_y: Handle,
    orig_b: Handle,
    recon_x: Handle,
    recon_y: Handle,
    recon_b: Handle,
    mask: Handle,
    out: Handle,
    n_blocks: u32,
) {
    if n_blocks == 0 {
        return;
    }
    let cubes = n_blocks.div_ceil(64).max(1);
    let n = n_blocks as usize;
    let pix = n * 64;
    unsafe {
        sse_reduce_3channel_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(orig_x, pix),
            ArrayArg::from_raw_parts(orig_y, pix),
            ArrayArg::from_raw_parts(orig_b, pix),
            ArrayArg::from_raw_parts(recon_x, pix),
            ArrayArg::from_raw_parts(recon_y, pix),
            ArrayArg::from_raw_parts(recon_b, pix),
            ArrayArg::from_raw_parts(mask, pix),
            ArrayArg::from_raw_parts(out, n),
        );
    }
}
