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

/// Inverse of `afv_compose_forward_kernel`'s composition + pack:
/// reads the 64-coef AFV-layout block (with packed DCs at positions
/// [0], [1], [8]), unpacks the 3 sub-block DCs, and writes them
/// alongside the rest of the coefficients into the 3 sub-input
/// buffers ready for the inverse 4×4 / 4×4 / 4×8 DCT kernels.
///
/// One thread per block. DC unpacking math mirrors the host version
/// in `inverse_afv_transform_batch_gpu`:
///   block00 = coefs[0]
///   block01 = coefs[1]
///   block10 = coefs[8]
///   dcs = [
///     (block00 + block10 + block01) * 4.0,
///     block00 + block10 - block01,
///     block00 - block10,
///   ]
///   afv_in[0]    = dcs[0]
///   dct4_in[0]   = dcs[1]
///   dct4x8_in[0] = dcs[2]
///
/// All other positions are direct copies from the matching offsets
/// in the 64-coef source.
#[cube(launch_unchecked)]
pub fn afv_unpack_inverse_kernel(
    coeffs: &Array<f32>,         // n_blocks * 64
    afv_out: &mut Array<f32>,    // n_blocks * 16
    dct4_out: &mut Array<f32>,   // n_blocks * 16
    dct4x8_out: &mut Array<f32>, // n_blocks * 32
) {
    let b = ABSOLUTE_POS;
    let n_blocks = coeffs.len() / 64;
    if b >= n_blocks {
        terminate!();
    }
    let bu = b as usize;
    let coef_base = bu * 64;
    let afv_base = bu * 16;
    let dct4_base = bu * 16;
    let dct4x8_base = bu * 32;

    // Unpack 3 DCs.
    let block00 = coeffs[coef_base];
    let block01 = coeffs[coef_base + 1];
    let block10 = coeffs[coef_base + 8];
    let afv_dc = (block00 + block10 + block01) * 4.0f32;
    let dct4_dc = block00 + block10 - block01;
    let dct4x8_dc = block00 - block10;

    // AFV sub-input (16 floats): coefs at (even_y, even_x) positions,
    // [0] gets the unpacked AFV DC.
    let mut iy: u32 = 0u32;
    while iy < 4u32 {
        let mut ix: u32 = 0u32;
        while ix < 4u32 {
            let dst = (iy as usize) * 4 + (ix as usize);
            let src = (iy as usize) * 16 + (ix as usize) * 2;
            let val = if iy == 0u32 && ix == 0u32 {
                afv_dc
            } else {
                coeffs[coef_base + src]
            };
            afv_out[afv_base + dst] = val;
            ix += 1u32;
        }
        iy += 1u32;
    }

    // DCT4 sub-input (16 floats): coefs at (even_y, odd_x) positions.
    iy = 0u32;
    while iy < 4u32 {
        let mut ix: u32 = 0u32;
        while ix < 4u32 {
            let dst = (iy as usize) * 4 + (ix as usize);
            let src = (iy as usize) * 16 + (ix as usize) * 2 + 1;
            let val = if iy == 0u32 && ix == 0u32 {
                dct4_dc
            } else {
                coeffs[coef_base + src]
            };
            dct4_out[dct4_base + dst] = val;
            ix += 1u32;
        }
        iy += 1u32;
    }

    // DCT4x8 sub-input (32 floats): coefs at odd_y rows.
    iy = 0u32;
    while iy < 4u32 {
        let mut ix: u32 = 0u32;
        while ix < 8u32 {
            let dst = (iy as usize) * 8 + (ix as usize);
            let src = (1u32 + iy * 2u32) as usize * 8 + (ix as usize);
            let val = if iy == 0u32 && ix == 0u32 {
                dct4x8_dc
            } else {
                coeffs[coef_base + src]
            };
            dct4x8_out[dct4x8_base + dst] = val;
            ix += 1u32;
        }
        iy += 1u32;
    }
}

/// Compose the 3 inverse-DCT pixel sub-outputs into a 64-pixel block.
/// Per-block, places AFV pixels (with corner mirroring) + DCT4 pixels
/// + DCT4x8 pixels into the layout the AFV variant expects.
///
/// `afv_kind` is passed as a scalar (0..3); it splits to
/// `afv_x = afv_kind & 1` and `afv_y = afv_kind >> 1` exactly like
/// the host version.
#[cube(launch_unchecked)]
pub fn afv_compose_inverse_kernel(
    afv_pixels: &Array<f32>,    // n_blocks * 16
    dct4_pixels: &Array<f32>,   // n_blocks * 16
    dct4x8_pixels: &Array<f32>, // n_blocks * 32
    out: &mut Array<f32>,       // n_blocks * 64 — output pixels
    afv_kind: u32,
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

    let afv_x = afv_kind & 1u32;
    let afv_y = afv_kind >> 1u32;

    // 1. AFV pixels (16) at the corner indicated by (afv_y, afv_x),
    //    with mirror per kind.
    let mut iy: u32 = 0u32;
    while iy < 4u32 {
        let block_y = if afv_y == 1u32 { 3u32 - iy } else { iy };
        let mut ix: u32 = 0u32;
        while ix < 4u32 {
            let block_x = if afv_x == 1u32 { 3u32 - ix } else { ix };
            let dst = (iy + afv_y * 4u32) as usize * 8
                + (afv_x * 4u32) as usize
                + ix as usize;
            let src = block_y as usize * 4 + block_x as usize;
            out[out_base + dst] = afv_pixels[afv_base + src];
            ix += 1u32;
        }
        iy += 1u32;
    }

    // 2. DCT4 pixels (16) at the OTHER corner ((1-afv_x), same afv_y).
    iy = 0u32;
    while iy < 4u32 {
        let mut ix: u32 = 0u32;
        while ix < 4u32 {
            let dst = (iy + afv_y * 4u32) as usize * 8
                + ((1u32 - afv_x) * 4u32) as usize
                + ix as usize;
            let src = iy as usize * 4 + ix as usize;
            out[out_base + dst] = dct4_pixels[dct4_base + src];
            ix += 1u32;
        }
        iy += 1u32;
    }

    // 3. DCT4x8 pixels (32) at the (1 - afv_y) rows.
    iy = 0u32;
    while iy < 4u32 {
        let mut ix: u32 = 0u32;
        while ix < 8u32 {
            let dst = (iy + (1u32 - afv_y) * 4u32) as usize * 8 + ix as usize;
            let src = iy as usize * 8 + ix as usize;
            out[out_base + dst] = dct4x8_pixels[dct4x8_base + src];
            ix += 1u32;
        }
        iy += 1u32;
    }
}
