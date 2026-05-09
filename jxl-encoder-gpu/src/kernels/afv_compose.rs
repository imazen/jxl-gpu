// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! GPU compose kernel for the AFV forward transform.
//!
//! AFV's forward transform produces 3 sub-block outputs from 3 GPU
//! kernel launches:
//!   1. AFV 4×4 DCT     → 16 floats per block (top-left corner)
//!   2. Raw DCT 4×4     → 16 floats per block (top-right corner)
//!   3. Raw DCT 4×8     → 32 floats per block (bottom half)
//!
//! These need to be composed into a single 64-coef block in the libjxl
//! AFV layout, with `pack_afv_dcs` repacking the 3 DC values at
//! positions 0, 1, 8.
//!
//! The host version (`afv_transform_batch_gpu`) does this composition
//! on host AFTER 3 sync `read_one()` downloads — ~9 ms of stalls per
//! call. This kernel keeps everything on GPU, eliminating the syncs.
//!
//! ## Layout
//!
//! Output 8×8 block (row-major):
//! ```text
//!   row 0: A00 D00 A01 D01 A02 D02 A03 D03   (alternating AFV / DCT4)
//!   row 1: H00 H01 H02 H03 H04 H05 H06 H07   (DCT4x8 row 0)
//!   row 2: A10 D10 A11 D11 A12 D12 A13 D13
//!   row 3: H10 H11 H12 H13 H14 H15 H16 H17
//!   ...
//! ```
//! After pack_afv_dcs, positions [0], [1], [8] hold the repacked DCs.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

/// One thread per block. Reads from `afv_in[16]`, `dct4_in[16]`,
/// `dct4x8_in[32]` and writes the composed `out[64]` with packed DCs.
///
/// All inputs are flat per-block: stride 16, 16, 32, 64 respectively.
/// `n_blocks` is derived from `out.len() / 64`.
#[cube(launch_unchecked)]
pub fn afv_compose_forward_kernel(
    afv_in: &Array<f32>,      // n_blocks * 16
    dct4_in: &Array<f32>,     // n_blocks * 16
    dct4x8_in: &Array<f32>,   // n_blocks * 32
    out: &mut Array<f32>,     // n_blocks * 64
) {
    let b = ABSOLUTE_POS;
    let n_blocks = out.len() / 64;
    if b >= n_blocks {
        terminate!();
    }
    let bu = b as usize;
    let afv_base = bu * 16;
    let dct4_base = bu * 16;
    let dct4x8_base = bu * 32;
    let out_base = bu * 64;

    // Step 1: place AFV (16 values) at (even_y, even_x) positions:
    //   out[iy*2*8 + ix*2] = afv_in[iy*4 + ix]
    let mut iy: u32 = 0u32;
    while iy < 4u32 {
        let mut ix: u32 = 0u32;
        while ix < 4u32 {
            let dst = (iy as usize) * 16 + (ix as usize) * 2;
            let src = (iy as usize) * 4 + (ix as usize);
            out[out_base + dst] = afv_in[afv_base + src];
            ix += 1u32;
        }
        iy += 1u32;
    }

    // Step 2: DCT4 (16 values) at (even_y, odd_x): out[iy*2*8 + ix*2 + 1]
    iy = 0u32;
    while iy < 4u32 {
        let mut ix: u32 = 0u32;
        while ix < 4u32 {
            let dst = (iy as usize) * 16 + (ix as usize) * 2 + 1;
            let src = (iy as usize) * 4 + (ix as usize);
            out[out_base + dst] = dct4_in[dct4_base + src];
            ix += 1u32;
        }
        iy += 1u32;
    }

    // Step 3: DCT4x8 (32 values) at odd_y rows: out[(1 + iy*2)*8 + ix]
    //   matches host afv.rs:170-174.
    iy = 0u32;
    while iy < 4u32 {
        let mut ix: u32 = 0u32;
        while ix < 8u32 {
            let dst = (1u32 + iy * 2u32) as usize * 8 + (ix as usize);
            let src = (iy as usize) * 8 + (ix as usize);
            out[out_base + dst] = dct4x8_in[dct4x8_base + src];
            ix += 1u32;
        }
        iy += 1u32;
    }

    // Step 4: pack_afv_dcs in-place at positions 0, 1, 8.
    //   block00 = out[0] * 0.25
    //   block01 = out[1]
    //   block10 = out[8]
    //   out[0] = (block00 + block01 + 2*block10) * 0.25
    //   out[1] = (block00 - block01) * 0.5
    //   out[8] = (block00 + block01 - 2*block10) * 0.25
    let block00 = out[out_base] * 0.25f32;
    let block01 = out[out_base + 1];
    let block10 = out[out_base + 8];
    out[out_base] = (block00 + block01 + 2.0f32 * block10) * 0.25f32;
    out[out_base + 1] = (block00 - block01) * 0.5f32;
    out[out_base + 8] = (block00 + block01 - 2.0f32 * block10) * 0.25f32;
}
