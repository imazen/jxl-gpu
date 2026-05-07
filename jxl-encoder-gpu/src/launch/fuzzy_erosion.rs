// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Host-side launcher for the fuzzy-erosion + 2× downsample kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::fuzzy_erosion::fuzzy_erosion_kernel;

const TPB: u32 = 256;

/// Compute the libjxl `[k0, k1, k2, k3]` weight set for fuzzy_erosion
/// from `butteraugli_target`. Pure scalar — caller passes the result
/// as 4 scalars to the GPU kernel.
pub fn fuzzy_erosion_kmul(butteraugli_target: f32) -> [f32; 4] {
    const K_MUL_BASE: [f32; 4] = [0.125, 0.1, 0.09, 0.06];
    const K_MUL_ADD: [f32; 4] = [0.0, -0.1, -0.09, -0.06];
    const K_TOTAL: f32 = 0.299_597_05;

    let mul = if butteraugli_target < 2.0 {
        (2.0 - butteraugli_target) * 0.5
    } else {
        0.0
    };
    let mut k = [0.0_f32; 4];
    let mut norm_sum = 0.0_f32;
    for i in 0..4 {
        k[i] = K_MUL_BASE[i] + mul * K_MUL_ADD[i];
        norm_sum += k[i];
    }
    let scale = K_TOTAL / norm_sum;
    for kk in &mut k {
        *kk *= scale;
    }
    k
}

#[allow(clippy::too_many_arguments)]
pub fn fuzzy_erosion<R: Runtime>(
    client: &ComputeClient<R>,
    src: Handle,
    output: Handle,
    src_w: u32,
    src_h: u32,
    from_x0: u32,
    from_y0: u32,
    out_w: u32,
    out_h: u32,
    k_mul: [f32; 4],
) {
    let n_out = (out_w * out_h) as usize;
    let n_in = (src_w * src_h) as usize;
    let cubes = (out_w * out_h).div_ceil(TPB).max(1);
    unsafe {
        fuzzy_erosion_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(src, n_in),
            ArrayArg::from_raw_parts(output, n_out),
            src_w,
            src_h,
            from_x0,
            from_y0,
            out_w,
            out_h,
            k_mul[0],
            k_mul[1],
            k_mul[2],
            k_mul[3],
        );
    }
}
