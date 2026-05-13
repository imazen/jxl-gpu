// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-element `compute_mask_for_ac_strategy_use` kernel.
//!
//! Mirrors libjxl's `ComputeMaskForAcStrategyUse`:
//! `mask = 1.0 / (aq_map + 0.001)`. One thread per element.
//! Used as Step 2.5 of the adaptive-quant pipeline — runs on the
//! post-`fuzzy_erosion` `aq_map` before `per_block_modulations`
//! mutates it.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn mask_for_ac_strategy_kernel(
    aq_map: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = n as usize;
    if idx >= total {
        terminate!();
    }
    let v = aq_map[idx];
    out[idx] = 1.0f32 / (v + 0.001f32);
}
