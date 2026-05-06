// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for CfL find_best_multiplier kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::cfl::{find_best_multiplier_kernel, find_best_multiplier_newton_kernel};

#[allow(clippy::too_many_arguments)]
pub fn find_best_multiplier_newton<R: Runtime>(
    client: &ComputeClient<R>,
    values_m: Handle,
    values_s: Handle,
    bases: Handle,
    output: Handle,
    num_tiles: u32,
    num_per_tile: u32,
    distance_mul: f32,
    eps: f32,
    max_iters: u32,
) {
    let nt = num_tiles as usize;
    let n_total = nt * (num_per_tile as usize);
    let cubes = num_tiles.max(1);
    unsafe {
        find_best_multiplier_newton_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(values_m, n_total),
            ArrayArg::from_raw_parts(values_s, n_total),
            ArrayArg::from_raw_parts(bases, nt),
            ArrayArg::from_raw_parts(output, nt),
            num_per_tile,
            distance_mul,
            eps,
            max_iters,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn find_best_multiplier<R: Runtime>(
    client: &ComputeClient<R>,
    values_m: Handle,
    values_s: Handle,
    bases: Handle,
    output: Handle, // i32 per tile
    num_tiles: u32,
    num_per_tile: u32,
    distance_mul: f32,
) {
    let nt = num_tiles as usize;
    let n_total = nt * (num_per_tile as usize);
    let cubes = num_tiles.max(1);
    unsafe {
        find_best_multiplier_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(values_m, n_total),
            ArrayArg::from_raw_parts(values_s, n_total),
            ArrayArg::from_raw_parts(bases, nt),
            ArrayArg::from_raw_parts(output, nt),
            num_per_tile,
            distance_mul,
        );
    }
}
