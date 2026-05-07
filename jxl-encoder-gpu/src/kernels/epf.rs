// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Edge-preserving filter (EPF) and pad_plane.
//!
//! Mirrors `jxl_encoder_simd::epf::{pad_plane, epf_step2_scalar}`.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

const EPF_CHANNEL_SCALE_X: f32 = 40.0;
const EPF_CHANNEL_SCALE_Y: f32 = 5.0;
const EPF_CHANNEL_SCALE_B: f32 = 3.5;

/// Pad a single channel plane with edge replication. One thread per
/// output pixel. Output shape is `(width + 2*pad) × (height + 2*pad)`,
/// stride = `width + 2*pad`.
#[cube(launch_unchecked)]
pub fn pad_plane_kernel(src: &Array<f32>, dst: &mut Array<f32>, width: u32, height: u32, pad: u32) {
    let idx = ABSOLUTE_POS;
    let w = width as usize;
    let h = height as usize;
    let p = pad as usize;
    let dst_w = w + 2usize * p;
    let dst_h = h + 2usize * p;
    let total = dst_w * dst_h;
    if idx >= total {
        terminate!();
    }
    let oy = idx / dst_w;
    let ox = idx - oy * dst_w;

    // Clamp source coordinates to interior via saturating_sub + min.
    let max_x = w - 1usize;
    let max_y = h - 1usize;
    let sx = usize::min(usize::saturating_sub(ox, p), max_x);
    let sy = usize::min(usize::saturating_sub(oy, p), max_y);
    dst[idx] = src[sy * w + sx];
}

/// 3x3-plus SAD (5 positions: center + 4 cardinals) summed across 3
/// channels, weighted by EPF_CHANNEL_SCALE. Both center and neighbor
/// stencils are evaluated.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sad_3x3_plus(
    in_x: &Array<f32>,
    in_y: &Array<f32>,
    in_b: &Array<f32>,
    cx: u32,
    cy: u32,
    nx: u32,
    ny: u32,
    stride: u32,
) -> f32 {
    let cxu = cx as usize;
    let cyu = cy as usize;
    let nxu = nx as usize;
    let nyu = ny as usize;
    let s = stride as usize;

    // Position (0, 0)
    let cidx = cyu * s + cxu;
    let nidx = nyu * s + nxu;
    let mut sad = f32::abs(in_x[cidx] - in_x[nidx]) * EPF_CHANNEL_SCALE_X
        + f32::abs(in_y[cidx] - in_y[nidx]) * EPF_CHANNEL_SCALE_Y
        + f32::abs(in_b[cidx] - in_b[nidx]) * EPF_CHANNEL_SCALE_B;

    // Position (-1, 0)
    let cidx = cyu * s + cxu - 1usize;
    let nidx = nyu * s + nxu - 1usize;
    sad = sad
        + f32::abs(in_x[cidx] - in_x[nidx]) * EPF_CHANNEL_SCALE_X
        + f32::abs(in_y[cidx] - in_y[nidx]) * EPF_CHANNEL_SCALE_Y
        + f32::abs(in_b[cidx] - in_b[nidx]) * EPF_CHANNEL_SCALE_B;

    // Position (0, -1)
    let cidx = (cyu - 1usize) * s + cxu;
    let nidx = (nyu - 1usize) * s + nxu;
    sad = sad
        + f32::abs(in_x[cidx] - in_x[nidx]) * EPF_CHANNEL_SCALE_X
        + f32::abs(in_y[cidx] - in_y[nidx]) * EPF_CHANNEL_SCALE_Y
        + f32::abs(in_b[cidx] - in_b[nidx]) * EPF_CHANNEL_SCALE_B;

    // Position (+1, 0)
    let cidx = cyu * s + cxu + 1usize;
    let nidx = nyu * s + nxu + 1usize;
    sad = sad
        + f32::abs(in_x[cidx] - in_x[nidx]) * EPF_CHANNEL_SCALE_X
        + f32::abs(in_y[cidx] - in_y[nidx]) * EPF_CHANNEL_SCALE_Y
        + f32::abs(in_b[cidx] - in_b[nidx]) * EPF_CHANNEL_SCALE_B;

    // Position (0, +1)
    let cidx = (cyu + 1usize) * s + cxu;
    let nidx = (nyu + 1usize) * s + nxu;
    sad = sad
        + f32::abs(in_x[cidx] - in_x[nidx]) * EPF_CHANNEL_SCALE_X
        + f32::abs(in_y[cidx] - in_y[nidx]) * EPF_CHANNEL_SCALE_Y
        + f32::abs(in_b[cidx] - in_b[nidx]) * EPF_CHANNEL_SCALE_B;

    sad
}

/// EPF step 1 — 3x3 cross kernel with 3x3-plus SAD weights.
///
/// Mirrors `jxl_encoder_simd::epf::epf_step1_scalar`. One thread per
/// output pixel. Padded inputs guarantee 3x3-plus SAD is in-bounds.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn epf_step1_kernel(
    in_x: &Array<f32>,
    in_y: &Array<f32>,
    in_b: &Array<f32>,
    out_x: &mut Array<f32>,
    out_y: &mut Array<f32>,
    out_b: &mut Array<f32>,
    inv_sigma: &Array<f32>,
    width: u32,
    height: u32,
    xsize_blocks: u32,
    in_stride: u32,
    pad: u32,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    let oidx = ABSOLUTE_POS;
    let w = width as usize;
    let h = height as usize;
    let total = w * h;
    if oidx >= total {
        terminate!();
    }
    let py = oidx / w;
    let px = oidx - py * w;
    let xb = xsize_blocks as usize;
    let stride = in_stride as usize;
    let p = pad as usize;
    let by = py / 8usize;
    let bx = px / 8usize;
    let sigma_idx = by * xb + bx;
    let is = inv_sigma[sigma_idx];

    let ipx = px + p;
    let ipy = py + p;
    let pidx = ipy * stride + ipx;

    let center_x = in_x[pidx];
    let center_y = in_y[pidx];
    let center_b = in_b[pidx];

    if is == 0.0f32 {
        out_x[oidx] = center_x;
        out_y[oidx] = center_y;
        out_b[oidx] = center_b;
    } else {
        let mod_x = px - (px / 8usize) * 8usize;
        let mod_y = py - (py / 8usize) * 8usize;
        let at_border_x = (mod_x == 0usize) || (mod_x == 7usize);
        let at_border_y = (mod_y == 0usize) || (mod_y == 7usize);
        let bm = if at_border_x || at_border_y {
            border_sigma_mul
        } else {
            f32::new(1.0)
        };
        let eff_is = is * sigma_scale * bm;

        let mut total_w = f32::new(1.0);
        let mut sum_x = center_x;
        let mut sum_y = center_y;
        let mut sum_b = center_b;

        // Cross neighbor: left (dx=-1, dy=0)
        let nx_left = ipx as u32 - 1u32;
        let ny_left = ipy as u32;
        let sad = sad_3x3_plus(
            in_x, in_y, in_b, ipx as u32, ipy as u32, nx_left, ny_left, in_stride,
        );
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = ipy * stride + ipx - 1usize;
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Cross neighbor: top (dx=0, dy=-1)
        let nx_top = ipx as u32;
        let ny_top = ipy as u32 - 1u32;
        let sad = sad_3x3_plus(
            in_x, in_y, in_b, ipx as u32, ipy as u32, nx_top, ny_top, in_stride,
        );
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = (ipy - 1usize) * stride + ipx;
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Cross neighbor: bottom (dx=0, dy=+1)
        let nx_bot = ipx as u32;
        let ny_bot = ipy as u32 + 1u32;
        let sad = sad_3x3_plus(
            in_x, in_y, in_b, ipx as u32, ipy as u32, nx_bot, ny_bot, in_stride,
        );
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = (ipy + 1usize) * stride + ipx;
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Cross neighbor: right (dx=+1, dy=0)
        let nx_right = ipx as u32 + 1u32;
        let ny_right = ipy as u32;
        let sad = sad_3x3_plus(
            in_x, in_y, in_b, ipx as u32, ipy as u32, nx_right, ny_right, in_stride,
        );
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = ipy * stride + ipx + 1usize;
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        let inv_tw = 1.0f32 / total_w;
        out_x[oidx] = sum_x * inv_tw;
        out_y[oidx] = sum_y * inv_tw;
        out_b[oidx] = sum_b * inv_tw;
    }
}

/// EPF step 2 — 3x3 cross kernel with single-pixel SAD weights.
/// One thread per OUTPUT pixel. Inputs are padded (in_stride =
/// width + 2*pad, accessed via (py + pad) * in_stride + (px + pad)).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn epf_step2_kernel(
    in_x: &Array<f32>,
    in_y: &Array<f32>,
    in_b: &Array<f32>,
    out_x: &mut Array<f32>,
    out_y: &mut Array<f32>,
    out_b: &mut Array<f32>,
    inv_sigma: &Array<f32>,
    width: u32,
    height: u32,
    xsize_blocks: u32,
    in_stride: u32,
    pad: u32,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    let oidx = ABSOLUTE_POS;
    let w = width as usize;
    let h = height as usize;
    let total = w * h;
    if oidx >= total {
        terminate!();
    }
    let py = oidx / w;
    let px = oidx - py * w;
    let xb = xsize_blocks as usize;
    let stride = in_stride as usize;
    let p = pad as usize;
    let by = py / 8usize;
    let bx = px / 8usize;
    let sigma_idx = by * xb + bx;
    let is = inv_sigma[sigma_idx];

    let pidx = (py + p) * stride + (px + p);
    let center_x = in_x[pidx];
    let center_y = in_y[pidx];
    let center_b = in_b[pidx];

    if is == 0.0f32 {
        out_x[oidx] = center_x;
        out_y[oidx] = center_y;
        out_b[oidx] = center_b;
    } else {
        let mod_x = px - (px / 8usize) * 8usize;
        let mod_y = py - (py / 8usize) * 8usize;
        let at_border_x = (mod_x == 0usize) || (mod_x == 7usize);
        let at_border_y = (mod_y == 0usize) || (mod_y == 7usize);
        let bm = if at_border_x || at_border_y {
            border_sigma_mul
        } else {
            f32::new(1.0)
        };
        let eff_is = is * sigma_scale * bm;

        // 4 cross neighbors via offset indices into padded buffer
        let n_left = pidx - 1usize;
        let n_right = pidx + 1usize;
        let n_top = pidx - stride;
        let n_bot = pidx + stride;

        let mut total_w = f32::new(1.0);
        let mut sum_x = center_x;
        let mut sum_y = center_y;
        let mut sum_b = center_b;

        // Neighbor: left
        let nx = in_x[n_left];
        let ny = in_y[n_left];
        let nb = in_b[n_left];
        let sad = f32::abs(center_x - nx) * EPF_CHANNEL_SCALE_X
            + f32::abs(center_y - ny) * EPF_CHANNEL_SCALE_Y
            + f32::abs(center_b - nb) * EPF_CHANNEL_SCALE_B;
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        total_w = total_w + weight;
        sum_x = sum_x + weight * nx;
        sum_y = sum_y + weight * ny;
        sum_b = sum_b + weight * nb;

        // Neighbor: top
        let nx = in_x[n_top];
        let ny = in_y[n_top];
        let nb = in_b[n_top];
        let sad = f32::abs(center_x - nx) * EPF_CHANNEL_SCALE_X
            + f32::abs(center_y - ny) * EPF_CHANNEL_SCALE_Y
            + f32::abs(center_b - nb) * EPF_CHANNEL_SCALE_B;
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        total_w = total_w + weight;
        sum_x = sum_x + weight * nx;
        sum_y = sum_y + weight * ny;
        sum_b = sum_b + weight * nb;

        // Neighbor: bottom
        let nx = in_x[n_bot];
        let ny = in_y[n_bot];
        let nb = in_b[n_bot];
        let sad = f32::abs(center_x - nx) * EPF_CHANNEL_SCALE_X
            + f32::abs(center_y - ny) * EPF_CHANNEL_SCALE_Y
            + f32::abs(center_b - nb) * EPF_CHANNEL_SCALE_B;
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        total_w = total_w + weight;
        sum_x = sum_x + weight * nx;
        sum_y = sum_y + weight * ny;
        sum_b = sum_b + weight * nb;

        // Neighbor: right
        let nx = in_x[n_right];
        let ny = in_y[n_right];
        let nb = in_b[n_right];
        let sad = f32::abs(center_x - nx) * EPF_CHANNEL_SCALE_X
            + f32::abs(center_y - ny) * EPF_CHANNEL_SCALE_Y
            + f32::abs(center_b - nb) * EPF_CHANNEL_SCALE_B;
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        total_w = total_w + weight;
        sum_x = sum_x + weight * nx;
        sum_y = sum_y + weight * ny;
        sum_b = sum_b + weight * nb;

        let inv_tw = 1.0f32 / total_w;
        out_x[oidx] = sum_x * inv_tw;
        out_y[oidx] = sum_y * inv_tw;
        out_b[oidx] = sum_b * inv_tw;
    }
}

/// EPF step 0 — 5×5 plus kernel with 3×3-plus SAD over 12 neighbor
/// positions. The heaviest filter step. Mirrors
/// `jxl_encoder::vardct::epf::epf_step0_strip` (which itself is the
/// strip-parallel form of upstream's serial step 0).
///
/// One thread per OUTPUT pixel. Inputs must be padded with `pad >= 3`
/// (the 5×5 plus pattern reaches ±2 from center; the 3×3-plus SAD
/// adds another ±1 around each neighbor → ±3 total reach).
///
/// Mirrors the structure of `epf_step1_kernel` (4 cross neighbors)
/// expanded to 12 neighbors per the `EPF0_NEIGHBORS` constant. SAD
/// computation reuses `sad_3x3_plus`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn epf_step0_kernel(
    in_x: &Array<f32>,
    in_y: &Array<f32>,
    in_b: &Array<f32>,
    out_x: &mut Array<f32>,
    out_y: &mut Array<f32>,
    out_b: &mut Array<f32>,
    inv_sigma: &Array<f32>,
    width: u32,
    height: u32,
    xsize_blocks: u32,
    in_stride: u32,
    pad: u32,
    sigma_scale: f32,
    border_sigma_mul: f32,
) {
    let oidx = ABSOLUTE_POS;
    let w = width as usize;
    let h = height as usize;
    let total = w * h;
    if oidx >= total {
        terminate!();
    }
    let py = oidx / w;
    let px = oidx - py * w;
    let xb = xsize_blocks as usize;
    let stride = in_stride as usize;
    let p = pad as usize;
    let by = py / 8usize;
    let bx = px / 8usize;
    let sigma_idx = by * xb + bx;
    let is = inv_sigma[sigma_idx];

    let ipx = px + p;
    let ipy = py + p;
    let pidx = ipy * stride + ipx;

    let center_x = in_x[pidx];
    let center_y = in_y[pidx];
    let center_b = in_b[pidx];

    if is == 0.0f32 {
        out_x[oidx] = center_x;
        out_y[oidx] = center_y;
        out_b[oidx] = center_b;
    } else {
        let mod_x = px - (px / 8usize) * 8usize;
        let mod_y = py - (py / 8usize) * 8usize;
        let at_border_x = (mod_x == 0usize) || (mod_x == 7usize);
        let at_border_y = (mod_y == 0usize) || (mod_y == 7usize);
        let bm = if at_border_x || at_border_y {
            border_sigma_mul
        } else {
            f32::new(1.0)
        };
        let eff_is = is * sigma_scale * bm;

        let cx = ipx as u32;
        let cy = ipy as u32;

        let mut total_w = f32::new(1.0);
        let mut sum_x = center_x;
        let mut sum_y = center_y;
        let mut sum_b = center_b;

        // Neighbor (-2, 0): cy-2, cx
        let nx = cx;
        let ny = cy - 2u32;
        let sad = sad_3x3_plus(in_x, in_y, in_b, cx, cy, nx, ny, in_stride);
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = (ipy - 2usize) * stride + ipx;
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Neighbor (-1, -1): cy-1, cx-1
        let nx = cx - 1u32;
        let ny = cy - 1u32;
        let sad = sad_3x3_plus(in_x, in_y, in_b, cx, cy, nx, ny, in_stride);
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = (ipy - 1usize) * stride + (ipx - 1usize);
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Neighbor (-1, 0): cy-1, cx
        let nx = cx;
        let ny = cy - 1u32;
        let sad = sad_3x3_plus(in_x, in_y, in_b, cx, cy, nx, ny, in_stride);
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = (ipy - 1usize) * stride + ipx;
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Neighbor (-1, 1): cy-1, cx+1
        let nx = cx + 1u32;
        let ny = cy - 1u32;
        let sad = sad_3x3_plus(in_x, in_y, in_b, cx, cy, nx, ny, in_stride);
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = (ipy - 1usize) * stride + (ipx + 1usize);
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Neighbor (0, -2): cy, cx-2
        let nx = cx - 2u32;
        let ny = cy;
        let sad = sad_3x3_plus(in_x, in_y, in_b, cx, cy, nx, ny, in_stride);
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = ipy * stride + (ipx - 2usize);
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Neighbor (0, -1): cy, cx-1
        let nx = cx - 1u32;
        let ny = cy;
        let sad = sad_3x3_plus(in_x, in_y, in_b, cx, cy, nx, ny, in_stride);
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = ipy * stride + (ipx - 1usize);
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Neighbor (0, 1): cy, cx+1
        let nx = cx + 1u32;
        let ny = cy;
        let sad = sad_3x3_plus(in_x, in_y, in_b, cx, cy, nx, ny, in_stride);
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = ipy * stride + (ipx + 1usize);
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Neighbor (0, 2): cy, cx+2
        let nx = cx + 2u32;
        let ny = cy;
        let sad = sad_3x3_plus(in_x, in_y, in_b, cx, cy, nx, ny, in_stride);
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = ipy * stride + (ipx + 2usize);
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Neighbor (1, -1): cy+1, cx-1
        let nx = cx - 1u32;
        let ny = cy + 1u32;
        let sad = sad_3x3_plus(in_x, in_y, in_b, cx, cy, nx, ny, in_stride);
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = (ipy + 1usize) * stride + (ipx - 1usize);
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Neighbor (1, 0): cy+1, cx
        let nx = cx;
        let ny = cy + 1u32;
        let sad = sad_3x3_plus(in_x, in_y, in_b, cx, cy, nx, ny, in_stride);
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = (ipy + 1usize) * stride + ipx;
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Neighbor (1, 1): cy+1, cx+1
        let nx = cx + 1u32;
        let ny = cy + 1u32;
        let sad = sad_3x3_plus(in_x, in_y, in_b, cx, cy, nx, ny, in_stride);
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = (ipy + 1usize) * stride + (ipx + 1usize);
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        // Neighbor (2, 0): cy+2, cx
        let nx = cx;
        let ny = cy + 2u32;
        let sad = sad_3x3_plus(in_x, in_y, in_b, cx, cy, nx, ny, in_stride);
        let weight = f32::max(sad * eff_is + 1.0f32, 0.0f32);
        let nidx = (ipy + 2usize) * stride + ipx;
        total_w = total_w + weight;
        sum_x = sum_x + weight * in_x[nidx];
        sum_y = sum_y + weight * in_y[nidx];
        sum_b = sum_b + weight * in_b[nidx];

        let inv_tw = 1.0f32 / total_w;
        out_x[oidx] = sum_x * inv_tw;
        out_y[oidx] = sum_y * inv_tw;
        out_b[oidx] = sum_b * inv_tw;
    }
}
