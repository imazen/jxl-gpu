// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-block masked sum-squared-error reduction over 3 channels.
//!
//! Used by `afv_cost_grid_xyb_host` to compute the per-block cost
//! `Σ_i ((dx_i² + dy_i² + db_i²) * mask_i)` over the 64 pixels of an
//! 8×8 block, where `dx/dy/db` are per-pixel reconstruction errors
//! and `mask_i` is the per-pixel masking weight. Sums are accumulated
//! per block; the kernel emits one `f32` cost per block.
//!
//! ## Why this kernel
//!
//! The host version of `afv_cost_grid_xyb_host` downloads 3 recon
//! pixel buffers (X/Y/B) per AFV kind (12 downloads total ~36 ms)
//! and computes the per-block error in a host loop (~12 ms). This
//! kernel keeps the recon pixels on GPU and emits only the small
//! per-block cost vector — eliminating the per-kind 3 downloads and
//! the host loop entirely.
//!
//! Combined with the persistent forward + inverse AFV transforms
//! (kernels/afv_compose.rs), the AFV cost grid becomes:
//!
//!   1. host upload of orig_x/y/b + mask (ONCE before the kind loop)
//!   2. per kind: forward → quant → dequant → inverse → reduce
//!      (all GPU, no syncs)
//!   3. ONE small download per kind: per-block costs
//!
//! On 1024² (16k blocks): ~36 ms of downloads → 0; ~12 ms of host
//! loop → 0. Pushes the cost grid from 52 ms toward the original
//! ~35 ms target in task #38.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

/// One thread per block. Each thread sums 64 pixels of
/// `(dx² + dy² + db²) * mask` and writes one `f32` cost.
///
/// All inputs are flat per-block: stride 64 (row-major 8×8 block).
/// `out` is one f32 per block.
#[cube(launch_unchecked)]
pub fn sse_reduce_3channel_kernel(
    orig_x: &Array<f32>,  // n_blocks * 64
    orig_y: &Array<f32>,  // n_blocks * 64
    orig_b: &Array<f32>,  // n_blocks * 64
    recon_x: &Array<f32>, // n_blocks * 64
    recon_y: &Array<f32>, // n_blocks * 64
    recon_b: &Array<f32>, // n_blocks * 64
    mask: &Array<f32>,    // n_blocks * 64
    out: &mut Array<f32>, // n_blocks
) {
    let b = ABSOLUTE_POS;
    let n_blocks = out.len();
    if b >= n_blocks {
        terminate!();
    }
    let bu = b as usize;
    let base = bu * 64;
    let mut acc: f32 = 0.0f32;
    let mut i: u32 = 0u32;
    while i < 64u32 {
        let off = base + i as usize;
        let dx = orig_x[off] - recon_x[off];
        let dy = orig_y[off] - recon_y[off];
        let db = orig_b[off] - recon_b[off];
        let m = mask[off];
        acc = acc + (dx * dx + dy * dy + db * db) * m;
        i += 1u32;
    }
    out[bu] = acc;
}
