// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launchers for the DCT8 DC quantization kernels.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::quantize_dc::{
    quantize_dc_chroma_dct8_kernel, quantize_dc_y_dct8_kernel,
};

const TPB: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn quantize_dc_y_dct8<R: Runtime>(
    client: &ComputeClient<R>,
    coeffs: Handle,
    quant_dc: Handle,
    float_dc: Handle,
    inv_factor: f32,
    num_blocks: u32,
) {
    let n_blocks = num_blocks as usize;
    let cubes = num_blocks.div_ceil(TPB).max(1);
    unsafe {
        quantize_dc_y_dct8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(coeffs, n_blocks * 64),
            ArrayArg::from_raw_parts(quant_dc, n_blocks),
            ArrayArg::from_raw_parts(float_dc, n_blocks),
            inv_factor,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn quantize_dc_chroma_dct8<R: Runtime>(
    client: &ComputeClient<R>,
    coeffs_chroma: Handle,
    quant_dc_y: Handle,
    quant_dc_chroma: Handle,
    float_dc_chroma: Handle,
    inv_factor: f32,
    dc_cfl_factor: f32,
    num_blocks: u32,
) {
    let n_blocks = num_blocks as usize;
    let cubes = num_blocks.div_ceil(TPB).max(1);
    unsafe {
        quantize_dc_chroma_dct8_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(coeffs_chroma, n_blocks * 64),
            ArrayArg::from_raw_parts(quant_dc_y, n_blocks),
            ArrayArg::from_raw_parts(quant_dc_chroma, n_blocks),
            ArrayArg::from_raw_parts(float_dc_chroma, n_blocks),
            inv_factor,
            dc_cfl_factor,
        );
    }
}
