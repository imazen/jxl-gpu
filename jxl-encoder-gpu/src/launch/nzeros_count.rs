// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for `nzeros_count_dct8_kernel`.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::nzeros_count::nzeros_count_dct8_kernel;

const TPB: u32 = 64;

pub fn nzeros_count_dct8<R: Runtime>(
    client: &ComputeClient<R>,
    quant_ac: Handle,
    nzeros: Handle,
    num_blocks: u32,
) {
    let n_blocks = num_blocks as usize;
    let cubes = num_blocks.div_ceil(TPB).max(1);
    unsafe {
        nzeros_count_dct8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(quant_ac, n_blocks * 64),
            ArrayArg::from_raw_parts(nzeros, n_blocks),
        );
    }
}
