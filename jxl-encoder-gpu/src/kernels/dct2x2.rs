// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! DCT2X2 8x8 forward + inverse transform.
//!
//! Mirrors `jxl_encoder::vardct::dct::special::{dct2x2_transform,
//! inverse_dct2x2_transform}` exactly.
//!
//! Hierarchical 2×2 Hadamard at three scales:
//!   - Forward: S=8, then S=4, then S=2. Each pass reads from
//!     interleaved 2×2 positions and writes to quadrant positions
//!     (×0.25). Only the SxS upper-left region is modified per pass.
//!   - Inverse: S=2, then S=4, then S=8. Each pass reverses the
//!     Hadamard (no 0.25 scaling) and writes back to interleaved 2×2
//!     positions in the SxS region.
//!
//! One thread per 8×8 block. Per-pass scratch lives in
//! `SharedMemory<f32>::new(64)`.

// `cube` macro emits `x = x + y` style for parity audits with the CPU
// reference; opt out of clippy's `assign_op_pattern` lint cluster-wide.
#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

const QUARTER: f32 = 0.25;

/// One forward Hadamard pass at scale S.
/// Reads from `data` at interleaved 2×2 positions, writes quadrant
/// layout back to `data` (only the SxS upper-left region is touched).
#[cube]
fn fwd_pass(data: &mut SharedMemory<f32>, temp: &mut SharedMemory<f32>, s: u32) {
    // First, copy the SxS region into temp for safe write-back.
    let su = s as usize;
    let num_2x2 = su / 2usize;
    let mut iy: u32 = 0u32;
    while iy < s {
        let iyu = iy as usize;
        let mut ix: u32 = 0u32;
        while ix < s {
            let ixu = ix as usize;
            temp[iyu * 8 + ixu] = data[iyu * 8 + ixu];
            ix += 1u32;
        }
        iy += 1u32;
    }
    // Now compute quadrants from the interleaved 2x2 positions and
    // write the four quadrants back into `data`.
    let mut y: u32 = 0u32;
    while y < (num_2x2 as u32) {
        let yu = y as usize;
        let mut x: u32 = 0u32;
        while x < (num_2x2 as u32) {
            let xu = x as usize;
            let c00 = temp[yu * 2 * 8 + xu * 2];
            let c01 = temp[yu * 2 * 8 + xu * 2 + 1];
            let c10 = temp[(yu * 2 + 1) * 8 + xu * 2];
            let c11 = temp[(yu * 2 + 1) * 8 + xu * 2 + 1];
            let r00 = (c00 + c01 + c10 + c11) * QUARTER;
            let r01 = (c00 + c01 - c10 - c11) * QUARTER;
            let r10 = (c00 - c01 + c10 - c11) * QUARTER;
            let r11 = (c00 - c01 - c10 + c11) * QUARTER;
            data[yu * 8 + xu] = r00;
            data[yu * 8 + num_2x2 + xu] = r01;
            data[(yu + num_2x2) * 8 + xu] = r10;
            data[(yu + num_2x2) * 8 + num_2x2 + xu] = r11;
            x += 1u32;
        }
        y += 1u32;
    }
}

/// One inverse Hadamard pass at scale S (no 0.25 scaling).
/// Reads from quadrant positions, writes back to interleaved 2×2
/// positions (only the SxS upper-left region is touched).
#[cube]
fn inv_pass(data: &mut SharedMemory<f32>, temp: &mut SharedMemory<f32>, s: u32) {
    let su = s as usize;
    let num_2x2 = su / 2usize;
    // Copy SxS region for safe write-back.
    let mut iy: u32 = 0u32;
    while iy < s {
        let iyu = iy as usize;
        let mut ix: u32 = 0u32;
        while ix < s {
            let ixu = ix as usize;
            temp[iyu * 8 + ixu] = data[iyu * 8 + ixu];
            ix += 1u32;
        }
        iy += 1u32;
    }
    // Inverse Hadamard: read quadrants, write interleaved 2x2.
    let mut y: u32 = 0u32;
    while y < (num_2x2 as u32) {
        let yu = y as usize;
        let mut x: u32 = 0u32;
        while x < (num_2x2 as u32) {
            let xu = x as usize;
            let c00 = temp[yu * 8 + xu];
            let c01 = temp[yu * 8 + num_2x2 + xu];
            let c10 = temp[(yu + num_2x2) * 8 + xu];
            let c11 = temp[(yu + num_2x2) * 8 + num_2x2 + xu];
            let r00 = c00 + c01 + c10 + c11;
            let r01 = c00 + c01 - c10 - c11;
            let r10 = c00 - c01 + c10 - c11;
            let r11 = c00 - c01 - c10 + c11;
            data[yu * 2 * 8 + xu * 2] = r00;
            data[yu * 2 * 8 + xu * 2 + 1] = r01;
            data[(yu * 2 + 1) * 8 + xu * 2] = r10;
            data[(yu * 2 + 1) * 8 + xu * 2 + 1] = r11;
            x += 1u32;
        }
        y += 1u32;
    }
}

/// Forward DCT2X2: hierarchical Hadamard at S=8, S=4, S=2.
#[cube(launch_unchecked)]
pub fn dct2x2_forward_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64u32 as usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;

    let mut data = SharedMemory::<f32>::new(64usize);
    let mut temp = SharedMemory::<f32>::new(64usize);
    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        data[iu] = input[off + iu];
        i += 1u32;
    }

    fwd_pass(&mut data, &mut temp, 8u32);
    fwd_pass(&mut data, &mut temp, 4u32);
    fwd_pass(&mut data, &mut temp, 2u32);

    let mut j: u32 = 0u32;
    while j < 64u32 {
        let ju = j as usize;
        output[off + ju] = data[ju];
        j += 1u32;
    }
}

/// Inverse DCT2X2: inverse Hadamard at S=2, S=4, S=8.
#[cube(launch_unchecked)]
pub fn dct2x2_inverse_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64u32 as usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;

    let mut data = SharedMemory::<f32>::new(64usize);
    let mut temp = SharedMemory::<f32>::new(64usize);
    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        data[iu] = input[off + iu];
        i += 1u32;
    }

    inv_pass(&mut data, &mut temp, 2u32);
    inv_pass(&mut data, &mut temp, 4u32);
    inv_pass(&mut data, &mut temp, 8u32);

    let mut j: u32 = 0u32;
    while j < 64u32 {
        let ju = j as usize;
        output[off + ju] = data[ju];
        j += 1u32;
    }
}
