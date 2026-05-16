// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Fully-fused 3-channel DCT8 + Y quantize + CfL + chroma quantize +
//! nzeros count, all in shared memory.
//!
//! Replaces what was previously ~10 small cubecl kernel launches in
//! `forks::pre_quantized_ac::compute_pre_quantized_ac_dct8_persistent`:
//!   gather × 3, DCT8 × 3, quantize Y, cfl_quantize X/B (× 2),
//!   nzeros_count × 3
//! with a SINGLE per-block kernel that:
//!   1. Loads X / Y / B 8×8 pixel tiles from the planes.
//!   2. Forward DCT8 each channel into private shared-memory scratch.
//!   3. Quantize Y AC into i32 (write to global quant_ac_y).
//!   4. AdjustQuantBias dequant Y → y_round_coef.
//!   5. CfL X = x_orig - x_factor * y_round_coef; quantize → quant_ac_x.
//!   6. Same for B with b_factor → quant_ac_b.
//!   7. Count non-zero AC per channel → write nzeros_x/y/b (u32).
//!
//! All intermediate float coefs live in shared memory only — no
//! global write+read between stages. Eliminates the per-launch
//! overhead that made the unfused producer slower than CPU
//! transform_and_quantize on cubecl 0.10.

use cubecl::prelude::*;

const ONE_OVER_8: f32 = 0.125;
const WIDE_CUBE_DIM: u32 = 64;
const SQRT2: f32 = core::f32::consts::SQRT_2;
const WC_M4_0: f32 = 0.541_196_1;
const WC_M4_1: f32 = 1.306_563;
const WC_M8_0: f32 = 0.509_795_6;
const WC_M8_1: f32 = 0.601_344_9;
const WC_M8_2: f32 = 0.899_976_2;
const WC_M8_3: f32 = 2.562_915_5;

// Y channel AdjustQuantBias.
const Y_BIAS_PM1: f32 = 0.929_945_5;
const BIAS_RECIP: f32 = 0.145;

#[cube]
fn dct1d_8(mem: &mut SharedMemory<f32>, base: u32) {
    // Mirrors `jxl_encoder_simd::dct8::dct1d_8_scalar` and the
    // standalone `kernels::dct8::dct1d_8` exactly:
    // - Split inputs into 4 sums (t0..t3) + 4 diffs (t4..t7)
    // - DCT-4 on sums (inline butterfly) → even output positions
    // - WC8 multiply on diffs, then DCT-4, **then** the post-DCT4
    //   cascade `[SQRT2*r4+r5, r5+r6, r6+r7, r7]` → odd output positions
    //
    // Two bugs in earlier versions of this kernel — both fixed in
    // jxl-gpu#8:
    //
    // 1. The post-DCT4 cascade on the diff half was missing entirely,
    //    so odd output positions held the raw DCT-4 output instead of
    //    the cascaded `[SQRT2*r4+r5, r5+r6, r6+r7, r7]` pattern.
    //
    // 2. Even output positions [2] and [4] were swapped vs CPU
    //    (`t2 = b1v; t1 = c0_post;` then `mem[2]=t2; mem[4]=t1;` →
    //    `mem[2]=T1, mem[4]=c0_post`, but CPU writes `mem[2]=c0_post,
    //    mem[4]=T1`).
    //
    // Together these meant every odd row+col position and the
    // mem[2]/mem[4] band were wrong; quantization of those positions
    // applied the wrong weight + threshold, and the decoder
    // reconstructed garbage values (max linear pixel ~8.1 vs ~1.5 on
    // the slow path on the gold-glitter image at d=1.0 e7).
    let b0 = base as usize;
    let m0 = mem[b0];
    let m1 = mem[b0 + 1usize];
    let m2 = mem[b0 + 2usize];
    let m3 = mem[b0 + 3usize];
    let m4 = mem[b0 + 4usize];
    let m5 = mem[b0 + 5usize];
    let m6 = mem[b0 + 6usize];
    let m7 = mem[b0 + 7usize];
    let mut t0 = m0 + m7;
    let mut t1 = m1 + m6;
    let mut t2 = m2 + m5;
    let mut t3 = m3 + m4;
    let mut t4 = m0 - m7;
    let mut t5 = m1 - m6;
    let mut t6 = m2 - m5;
    let mut t7 = m3 - m4;
    // ── First half: DCT-4 on the 4 sums (→ even output positions) ──
    {
        let a0 = t0 + t3;
        let a1 = t1 + t2;
        let a2 = t0 - t3;
        let a3 = t1 - t2;
        let b0v = a0 + a1;
        let b1v = a0 - a1;
        let a2s = a2 * WC_M4_0;
        let a3s = a3 * WC_M4_1;
        let c0 = a2s + a3s;
        let c1 = a2s - a3s;
        let c0_post = SQRT2 * c0 + c1;
        // dct1d_4 output ordering: tmp[0..4] = [b0v, c0_post, b1v, c1].
        // Map back into named t-vars so the storage step below matches
        // the standalone `kernels::dct8::dct1d_8` form.
        t0 = b0v;
        t1 = c0_post;
        t2 = b1v;
        t3 = c1;
    }
    // ── Second half: WC8 multiply, DCT-4, then post-DCT4 cascade ──
    t4 *= WC_M8_0;
    t5 *= WC_M8_1;
    t6 *= WC_M8_2;
    t7 *= WC_M8_3;
    {
        let a0 = t4 + t7;
        let a1 = t5 + t6;
        let a2 = t4 - t7;
        let a3 = t5 - t6;
        let b0v = a0 + a1;
        let b1v = a0 - a1;
        let a2s = a2 * WC_M4_0;
        let a3s = a3 * WC_M4_1;
        let c0 = a2s + a3s;
        let c1 = a2s - a3s;
        let c0_post = SQRT2 * c0 + c1;
        // dct1d_4 output ordering on the diff half: same as above
        // (tmp[4..8] = [b0v, c0_post, b1v, c1]).
        t4 = b0v;
        t5 = c0_post;
        t6 = b1v;
        t7 = c1;
    }
    // Post-DCT4 cascade for the diff (odd-position) half — matches
    // CPU scalar `tmp[4]=SQRT2*tmp[4]+tmp[5]; tmp[5]+=tmp[6]; tmp[6]+=tmp[7];`.
    t4 = SQRT2 * t4 + t5;
    t5 = t5 + t6;
    t6 = t6 + t7;
    // Final storage: even positions from sums, odd positions from
    // (cascaded) diffs.
    mem[b0] = t0;
    mem[b0 + 1usize] = t4;
    mem[b0 + 2usize] = t1;
    mem[b0 + 3usize] = t5;
    mem[b0 + 4usize] = t2;
    mem[b0 + 5usize] = t6;
    mem[b0 + 6usize] = t3;
    mem[b0 + 7usize] = t7;
}

#[cube]
fn round_ties_even_to_i32(x: f32) -> i32 {
    let r = f32::round(x);
    let frac = f32::abs(x - f32::floor(x));
    let r_int = r as i32;
    let is_tie = f32::abs(frac - 0.5f32) < 1e-7f32;
    let is_odd = (r_int.abs() & 1i32) == 1i32;
    let mut out = r_int;
    if is_tie && is_odd {
        if r_int > 0i32 {
            out = r_int - 1i32;
        } else {
            out = r_int + 1i32;
        }
    }
    out
}

/// Round-half-AWAY-from-zero for f32 -> i32 (matches Rust's
/// `f32::round() as i32`). Used by DC quantize, NOT by AC quantize
/// (AC uses `round_ties_even_to_i32` to match libjxl's `rintf()`).
#[cube]
fn round_ties_away_to_i32(x: f32) -> i32 {
    f32::round(x) as i32
}

#[cube]
fn dequant_y_with_bias(q: i32) -> f32 {
    let qf = q as f32;
    let abs_q = f32::abs(qf);
    let mut out = f32::new(0.0);
    if q != 0i32 {
        if abs_q < 1.125f32 {
            let s = if qf > 0.0f32 { 1.0f32 } else { -1.0f32 };
            out = s * Y_BIAS_PM1;
        } else {
            out = qf - BIAS_RECIP / qf;
        }
    }
    out
}

#[cube]
fn dct8_block(scratch: &mut SharedMemory<f32>, transposed: &mut SharedMemory<f32>, base: u32) {
    let base_us = base as usize;
    // Row pass.
    let mut r: u32 = 0u32;
    while r < 8u32 {
        let row_off = r * 8u32;
        let row_off_us = row_off as usize;
        dct1d_8(scratch, base + row_off);
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let cu = c as usize;
            scratch[base_us + row_off_us + cu] = scratch[base_us + row_off_us + cu] * ONE_OVER_8;
            c += 1u32;
        }
        r += 1u32;
    }
    // Transpose into `transposed`.
    let mut r: u32 = 0u32;
    while r < 8u32 {
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[base_us + cu * 8usize + ru] = scratch[base_us + ru * 8usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }
    // Column pass (= row pass on transposed).
    let mut r: u32 = 0u32;
    while r < 8u32 {
        let row_off = r * 8u32;
        let row_off_us = row_off as usize;
        dct1d_8(transposed, base + row_off);
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let cu = c as usize;
            transposed[base_us + row_off_us + cu] =
                transposed[base_us + row_off_us + cu] * ONE_OVER_8;
            c += 1u32;
        }
        r += 1u32;
    }
}

/// Fully-fused 3-channel DCT8 + quantize + CfL + nzeros pipeline.
/// One thread per output block.
///
/// Inputs:
/// - `plane_x` / `plane_y` / `plane_b`: padded XYB f32 planes, all
///   sharing stride `plane_stride` and height `plane_h_pixels`.
/// - `xsize_blocks` / `ysize_blocks`: sub-rect of blocks to process
///   (cpu-aligned dims; gpu plane may have extra padding columns
///   past xsize_blocks * 8 — those are simply not touched).
/// - `weights_x` / `weights_y` / `weights_b`: 64-coef quant matrix
///   templates (broadcast across all blocks).
/// - `qac_qm_x` / `qac_qm_y` / `qac_qm_b`: per-block scale (`qac *
///   qm_multiplier`), length `n_blocks`.
/// - `x_factor_per_block` / `b_factor_per_block`: per-block CfL
///   factors (host expanded from per-tile cfl_map).
/// - `thresholds_x` / `thresholds_y` / `thresholds_b`: 4 floats each.
///
/// Outputs:
/// - `quant_ac_x` / `quant_ac_y` / `quant_ac_b`: per-block 64 i32
///   each (DC slot at position 0 set to 0).
/// - `nzeros_x` / `nzeros_y` / `nzeros_b`: per-block u32 non-zero
///   AC count (caller converts to u8 / u16).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn fused_dct8_3ch_kernel(
    plane_x: &Array<f32>,
    plane_y: &Array<f32>,
    plane_b: &Array<f32>,
    weights_x: &Array<f32>,
    weights_y: &Array<f32>,
    weights_b: &Array<f32>,
    qac_qm_x: &Array<f32>,
    qac_qm_y: &Array<f32>,
    qac_qm_b: &Array<f32>,
    x_factor_per_block: &Array<f32>,
    b_factor_per_block: &Array<f32>,
    thresholds_x: &Array<f32>,
    thresholds_y: &Array<f32>,
    thresholds_b: &Array<f32>,
    quant_ac_x: &mut Array<i32>,
    quant_ac_y: &mut Array<i32>,
    quant_ac_b: &mut Array<i32>,
    nzeros_x: &mut Array<u32>,
    nzeros_y: &mut Array<u32>,
    nzeros_b: &mut Array<u32>,
    quant_dc_x: &mut Array<i16>,
    quant_dc_y: &mut Array<i16>,
    quant_dc_b: &mut Array<i16>,
    float_dc_x: &mut Array<f32>,
    float_dc_y: &mut Array<f32>,
    float_dc_b: &mut Array<f32>,
    inv_dc_factor_x: f32,
    inv_dc_factor_y: f32,
    inv_dc_factor_b: f32,
    plane_stride: u32,
    xsize_blocks: u32,
) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = qac_qm_y.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let bpr = xsize_blocks as usize;
    let by = block_idx / bpr;
    let bx = block_idx - by * bpr;
    let stride = plane_stride as usize;
    let off = block_idx * 64usize;
    let unit = UNIT_POS;
    let private_base = unit * 64u32;
    let private_base_us = private_base as usize;

    // Three pairs of (scratch, transposed) shared-memory buffers,
    // one pair per channel. Per-thread private slice via UNIT_POS.
    let mut scratch_x = SharedMemory::<f32>::new((WIDE_CUBE_DIM * 64u32) as usize);
    let mut transposed_x = SharedMemory::<f32>::new((WIDE_CUBE_DIM * 64u32) as usize);
    let mut scratch_y = SharedMemory::<f32>::new((WIDE_CUBE_DIM * 64u32) as usize);
    let mut transposed_y = SharedMemory::<f32>::new((WIDE_CUBE_DIM * 64u32) as usize);
    let mut scratch_b = SharedMemory::<f32>::new((WIDE_CUBE_DIM * 64u32) as usize);
    let mut transposed_b = SharedMemory::<f32>::new((WIDE_CUBE_DIM * 64u32) as usize);

    // Step 1: gather 8×8 pixel tile per channel into scratch.
    let y0 = by * 8usize;
    let x0 = bx * 8usize;
    let mut r: u32 = 0u32;
    while r < 8u32 {
        let row_off = r * 8u32;
        let row_off_us = row_off as usize;
        let src_row_off = (y0 + r as usize) * stride + x0;
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let cu = c as usize;
            scratch_x[private_base_us + row_off_us + cu] = plane_x[src_row_off + cu];
            scratch_y[private_base_us + row_off_us + cu] = plane_y[src_row_off + cu];
            scratch_b[private_base_us + row_off_us + cu] = plane_b[src_row_off + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // Step 2: forward DCT8 per channel.
    dct8_block(&mut scratch_x, &mut transposed_x, private_base);
    dct8_block(&mut scratch_y, &mut transposed_y, private_base);
    dct8_block(&mut scratch_b, &mut transposed_b, private_base);

    // Step 2.5: extract DC + DC quantize, all from shared mem.
    // Eliminates 6 separate kernels (3 gather + 3 dct8 just to reach
    // these float DC values for the DC quantize chain).
    let dc_x = transposed_x[private_base_us];
    let dc_y = transposed_y[private_base_us];
    let dc_b = transposed_b[private_base_us];
    float_dc_x[block_idx] = dc_x;
    float_dc_y[block_idx] = dc_y;
    float_dc_b[block_idx] = dc_b;
    let qdy_i32 = round_ties_away_to_i32(dc_y * inv_dc_factor_y);
    quant_dc_y[block_idx] = qdy_i32 as i16;
    let qdy_f = qdy_i32 as f32;
    // X channel: dc_cfl_factor = 0.0
    quant_dc_x[block_idx] = round_ties_away_to_i32(dc_x * inv_dc_factor_x) as i16;
    // B channel: dc_cfl_factor = 0.5
    quant_dc_b[block_idx] = round_ties_away_to_i32(dc_b * inv_dc_factor_b - qdy_f * 0.5f32) as i16;

    // Step 3: quantize Y AC into i32 + count nzeros.
    let qac_y = qac_qm_y[block_idx];
    let inv_qac_y = 1.0f32 / qac_y;
    let ty0 = thresholds_y[0usize];
    let ty1 = thresholds_y[1usize];
    let ty2 = thresholds_y[2usize];
    let ty3 = thresholds_y[3usize];
    quant_ac_y[off] = 0i32;
    let mut nz_y: u32 = 0u32;
    let mut idx: u32 = 1u32;
    while idx < 64u32 {
        let iu = idx as usize;
        let y = idx / 8u32;
        let x = idx - y * 8u32;
        let row_hi = y >= 4u32;
        let col_hi = x >= 4u32;
        let thr = if row_hi {
            if col_hi { ty3 } else { ty2 }
        } else if col_hi {
            ty1
        } else {
            ty0
        };
        let coef = transposed_y[private_base_us + iu];
        let val = coef * (1.0f32 / weights_y[iu]) * qac_y;
        let absv = f32::abs(val);
        let q = if absv < thr {
            i32::new(0)
        } else {
            round_ties_even_to_i32(val)
        };
        quant_ac_y[off + iu] = q;
        if q != 0i32 {
            nz_y += 1u32;
        }
        idx += 1u32;
    }
    nzeros_y[block_idx] = nz_y;

    // Step 4-5: CfL + quantize X AC.
    let qac_x = qac_qm_x[block_idx];
    let xfac = x_factor_per_block[block_idx];
    let tx0 = thresholds_x[0usize];
    let tx1 = thresholds_x[1usize];
    let tx2 = thresholds_x[2usize];
    let tx3 = thresholds_x[3usize];
    quant_ac_x[off] = 0i32;
    let mut nz_x: u32 = 0u32;
    let mut idx: u32 = 1u32;
    while idx < 64u32 {
        let iu = idx as usize;
        let y = idx / 8u32;
        let x = idx - y * 8u32;
        let row_hi = y >= 4u32;
        let col_hi = x >= 4u32;
        let thr = if row_hi {
            if col_hi { tx3 } else { tx2 }
        } else if col_hi {
            tx1
        } else {
            tx0
        };
        let y_round = dequant_y_with_bias(quant_ac_y[off + iu]);
        let y_round_coef = y_round * weights_y[iu] * inv_qac_y;
        let x_orig = transposed_x[private_base_us + iu];
        let x_cfl = x_orig - xfac * y_round_coef;
        let val = x_cfl * (1.0f32 / weights_x[iu]) * qac_x;
        let absv = f32::abs(val);
        let q = if absv < thr {
            i32::new(0)
        } else {
            round_ties_even_to_i32(val)
        };
        quant_ac_x[off + iu] = q;
        if q != 0i32 {
            nz_x += 1u32;
        }
        idx += 1u32;
    }
    nzeros_x[block_idx] = nz_x;

    // Step 6: CfL + quantize B AC.
    let qac_b = qac_qm_b[block_idx];
    let bfac = b_factor_per_block[block_idx];
    let tb0 = thresholds_b[0usize];
    let tb1 = thresholds_b[1usize];
    let tb2 = thresholds_b[2usize];
    let tb3 = thresholds_b[3usize];
    quant_ac_b[off] = 0i32;
    let mut nz_b: u32 = 0u32;
    let mut idx: u32 = 1u32;
    while idx < 64u32 {
        let iu = idx as usize;
        let y = idx / 8u32;
        let x = idx - y * 8u32;
        let row_hi = y >= 4u32;
        let col_hi = x >= 4u32;
        let thr = if row_hi {
            if col_hi { tb3 } else { tb2 }
        } else if col_hi {
            tb1
        } else {
            tb0
        };
        let y_round = dequant_y_with_bias(quant_ac_y[off + iu]);
        let y_round_coef = y_round * weights_y[iu] * inv_qac_y;
        let b_orig = transposed_b[private_base_us + iu];
        let b_cfl = b_orig - bfac * y_round_coef;
        let val = b_cfl * (1.0f32 / weights_b[iu]) * qac_b;
        let absv = f32::abs(val);
        let q = if absv < thr {
            i32::new(0)
        } else {
            round_ties_even_to_i32(val)
        };
        quant_ac_b[off + iu] = q;
        if q != 0i32 {
            nz_b += 1u32;
        }
        idx += 1u32;
    }
    nzeros_b[block_idx] = nz_b;
}
