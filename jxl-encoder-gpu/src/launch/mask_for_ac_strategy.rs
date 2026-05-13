// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for `mask_for_ac_strategy_kernel`.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::mask_for_ac_strategy::mask_for_ac_strategy_kernel;

const TPB: u32 = 256;

/// `out[i] = 1.0 / (aq_map[i] + 0.001)`. Both buffers are length `n` floats.
pub fn mask_for_ac_strategy<R: Runtime>(
    client: &ComputeClient<R>,
    aq_map: Handle,
    out: Handle,
    n: u32,
) {
    let n_us = n as usize;
    let cubes = n.div_ceil(TPB).max(1);
    unsafe {
        mask_for_ac_strategy_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(aq_map, n_us),
            ArrayArg::from_raw_parts(out, n_us),
            n,
        );
    }
}
