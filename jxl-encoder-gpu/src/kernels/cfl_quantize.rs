// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Fused CfL-subtract + DCT8 AC quantization for the chroma channels.
//!
//! Mirrors the CPU encoder's per-block sequence in
//! `jxl_encoder::vardct::transform::transform_blocks_into` for each
//! chroma channel (X or B):
//!
//! 1. Y is already quantized into `quant_ac_y[off + iu]` (i32).
//! 2. Dequantize Y with AdjustQuantBias to get the roundtripped
//!    Y coefficient: `y_round_coef = adjust_quant_bias(quant_ac_y[i]) *
//!                                    weights_y[iu] / qac_y`.
//! 3. CfL-adjust the chroma coefficient:
//!    `x_cfl = x_orig[i] - x_factor * y_round_coef`.
//! 4. Quantize with X's weights/qac/thresholds, dead-zone
//!    matching `quantize_block_dct8` exactly.
//!
//! `x_factor` is per-block (host expands the per-tile `cfl_map`
//! into a per-block array before launch).
//!
//! libjxl reference: `enc_group.cc::QuantizeRoundtripYBlockAC` +
//! `QuantizeBlockAC` for X/B with CfL applied.

use cubecl::prelude::*;

/// libjxl `kDefaultQuantBias` for the Y channel (`channel == 1`).
/// 1.0 − 0.07005449891748593 = 0.929945501082514. See
/// `jxl_encoder::vardct::quantize::adjust_quant_bias`.
const Y_BIAS_PM1: f32 = 0.929_945_5;
/// Reciprocal correction `q − BIAS3 / q` for |q| >= 2.
const BIAS_RECIP: f32 = 0.145;

/// Round-half-to-even (banker's rounding) for f32 → i32. Matches
/// libjxl's `rintf`/`Round()` (used in the scalar quant path).
///
/// cubecl 0.10's `f32::round` is ties-away-from-zero. We adjust
/// the +/-0.5 tie cases manually: if the fractional part is exactly
/// 0.5 and the rounded value is odd, back off by 1 toward even.
#[cube]
fn round_ties_even_to_i32(x: f32) -> i32 {
    let r = f32::round(x);
    let frac = f32::abs(x - f32::floor(x));
    let r_int = r as i32;
    // Detect the tie: |fractional part| exactly 0.5 (within float
    // precision) AND rounded value is odd.
    let is_tie = f32::abs(frac - 0.5f32) < 1e-7f32;
    let is_odd = (r_int.abs() & 1i32) == 1i32;
    let mut out = r_int;
    if is_tie && is_odd {
        // Back off toward even: subtract sign(r).
        if r_int > 0i32 {
            out = r_int - 1i32;
        } else {
            out = r_int + 1i32;
        }
    }
    out
}

/// AdjustQuantBias for Y (channel 1): mirrors the scalar function.
#[cube]
fn dequant_y_with_bias(q: i32) -> f32 {
    let qf = q as f32;
    let abs_q = f32::abs(qf);
    let mut out = f32::new(0.0);
    if q != 0i32 {
        if abs_q < 1.125f32 {
            // ±1 → channel-specific bias with sign
            let s = if qf > 0.0f32 { 1.0f32 } else { -1.0f32 };
            out = s * Y_BIAS_PM1;
        } else {
            out = qf - BIAS_RECIP / qf;
        }
    }
    out
}

/// Fused CfL-subtract + dead-zone quantize for one chroma channel
/// (X or B) on DCT8 blocks. One thread per AC coefficient.
///
/// `x_orig`: pre-CfL, pre-quant chroma DCT8 coefficients
/// (`num_blocks * 64` floats).
/// `quant_ac_y`: already-quantized Y AC coefficients
/// (`num_blocks * 64` i32; position 0 is DC and is ignored).
/// `weights_x`: per-coefficient quant matrix for the chroma channel
/// (length 64, broadcast across all blocks).
/// `weights_y`: per-coefficient quant matrix for Y (length 64,
/// broadcast across all blocks). Used for dequantizing Y.
/// `qac_qm_x`: per-block scale for the chroma channel
/// (`num_blocks` floats, equals `qac * x_qm_mul`).
/// `qac_qm_y`: per-block scale for Y (`num_blocks` floats, equals
/// `qac` since Y has qm_mul = 1.0).
/// `x_factor_per_block`: per-block CfL factor for this chroma channel
/// (`num_blocks` floats, expanded by host from the per-tile cfl_map).
/// `thresholds`: dead-zone thresholds [t_q0, t_q1, t_q2, t_q3] for
/// the four quadrants of an 8×8 block (length 4).
/// `output`: per-block quantized chroma AC coefficients
/// (`num_blocks * 64` i32). Position 0 is set to 0 (DC handled
/// separately by the caller).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cfl_quantize_dct8_kernel(
    x_orig: &Array<f32>,
    quant_ac_y: &Array<i32>,
    weights_x: &Array<f32>,
    weights_y: &Array<f32>,
    qac_qm_x: &Array<f32>,
    qac_qm_y: &Array<f32>,
    x_factor_per_block: &Array<f32>,
    thresholds: &Array<f32>,
    output: &mut Array<i32>,
) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = qac_qm_x.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;
    let qac_x = qac_qm_x[block_idx];
    let qac_y = qac_qm_y[block_idx];
    let x_factor = x_factor_per_block[block_idx];
    let inv_qac_y = 1.0f32 / qac_y;

    let t0 = thresholds[0usize];
    let t1 = thresholds[1usize];
    let t2 = thresholds[2usize];
    let t3 = thresholds[3usize];

    // DC: caller writes the DC slot separately.
    output[off] = 0i32;

    let mut idx: u32 = 1u32;
    while idx < 64u32 {
        let iu = idx as usize;
        let y = idx / 8u32;
        let x = idx - y * 8u32;
        let row_hi = y >= 4u32;
        let col_hi = x >= 4u32;
        let thr = if row_hi {
            if col_hi { t3 } else { t2 }
        } else if col_hi {
            t1
        } else {
            t0
        };

        // Step 1: dequantize Y with AdjustQuantBias.
        let y_round = dequant_y_with_bias(quant_ac_y[off + iu]);
        let y_round_coef = y_round * weights_y[iu] * inv_qac_y;

        // Step 2: CfL subtract.
        let x_cfl_coef = x_orig[off + iu] - x_factor * y_round_coef;

        // Step 3: dead-zone quantize.
        let val = x_cfl_coef * (1.0f32 / weights_x[iu]) * qac_x;
        let absv = f32::abs(val);
        output[off + iu] = if absv < thr {
            i32::new(0)
        } else {
            round_ties_even_to_i32(val)
        };
        idx += 1u32;
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use crate::launch::cfl_quantize::cfl_quantize_dct8;
    use cubecl::Runtime;
    use cubecl::cuda::CudaRuntime;
    use cubecl::prelude::*;
    extern crate alloc;

    /// CPU reference implementation matching the kernel.
    fn cpu_reference(
        x_orig: &[f32],
        quant_ac_y: &[i32],
        weights_x: &[f32; 64],
        weights_y: &[f32; 64],
        qac_qm_x: &[f32],
        qac_qm_y: &[f32],
        x_factor_per_block: &[f32],
        thr: &[f32; 4],
    ) -> Vec<i32> {
        let n_blocks = qac_qm_x.len();
        let mut out = vec![0i32; n_blocks * 64];
        for b in 0..n_blocks {
            let off = b * 64;
            let qx = qac_qm_x[b];
            let qy = qac_qm_y[b];
            let xf = x_factor_per_block[b];
            for idx in 1..64 {
                let y = idx / 8;
                let x = idx - y * 8;
                let row_hi = y >= 4;
                let col_hi = x >= 4;
                let t = if row_hi {
                    if col_hi { thr[3] } else { thr[2] }
                } else if col_hi {
                    thr[1]
                } else {
                    thr[0]
                };

                // dequant Y with bias
                let q = quant_ac_y[off + idx];
                let qf = q as f32;
                let abs_q = qf.abs();
                let y_round = if q == 0 {
                    0.0
                } else if abs_q < 1.125 {
                    qf.signum() * super::Y_BIAS_PM1
                } else {
                    qf - super::BIAS_RECIP / qf
                };
                let y_round_coef = y_round * weights_y[idx] / qy;

                let x_cfl = x_orig[off + idx] - xf * y_round_coef;
                let val = x_cfl * (1.0 / weights_x[idx]) * qx;
                if val.abs() < t {
                    out[off + idx] = 0;
                } else {
                    // Ties-to-even rounding via Rust 1.77+ method.
                    out[off + idx] = (val as f64).round_ties_even() as i32;
                }
            }
        }
        out
    }

    #[test]
    fn cfl_quantize_dct8_matches_cpu_reference() {
        let device = Default::default();
        let client = CudaRuntime::client(&device);

        let n_blocks = 17usize;
        // Pseudo-random non-trivial inputs.
        let x_orig: Vec<f32> = (0..n_blocks * 64)
            .map(|i| ((i as f32) * 0.0317).sin() * 5.5)
            .collect();
        // Y quant_ac in {-3..=3} including 0 and ±1 to exercise the
        // bias branches.
        let quant_ac_y: Vec<i32> = (0..n_blocks * 64)
            .map(|i| ((i as i32 * 7 + 3) % 7) - 3)
            .collect();
        let weights_x: [f32; 64] = core::array::from_fn(|i| 0.5 + (i as f32) * 0.1);
        let weights_y: [f32; 64] = core::array::from_fn(|i| 0.7 + (i as f32) * 0.05);
        let qac_qm_x: Vec<f32> = (0..n_blocks).map(|i| 1.2 + i as f32 * 0.05).collect();
        let qac_qm_y: Vec<f32> = (0..n_blocks).map(|i| 1.5 - i as f32 * 0.03).collect();
        let x_factor: Vec<f32> = (0..n_blocks).map(|i| -0.3 + i as f32 * 0.02).collect();
        let thr: [f32; 4] = [0.58, 0.62, 0.62, 0.62];

        let cpu = cpu_reference(
            &x_orig,
            &quant_ac_y,
            &weights_x,
            &weights_y,
            &qac_qm_x,
            &qac_qm_y,
            &x_factor,
            &thr,
        );

        // Upload everything, launch, download.
        let h_x = client.create_from_slice(f32::as_bytes(&x_orig));
        let h_qy = client.create_from_slice(i32::as_bytes(&quant_ac_y));
        let h_wx = client.create_from_slice(f32::as_bytes(&weights_x[..]));
        let h_wy = client.create_from_slice(f32::as_bytes(&weights_y[..]));
        let h_qmx = client.create_from_slice(f32::as_bytes(&qac_qm_x));
        let h_qmy = client.create_from_slice(f32::as_bytes(&qac_qm_y));
        let h_xf = client.create_from_slice(f32::as_bytes(&x_factor));
        let h_thr = client.create_from_slice(f32::as_bytes(&thr[..]));
        let h_out = client.empty(n_blocks * 64 * core::mem::size_of::<i32>());

        cfl_quantize_dct8::<CudaRuntime>(
            &client,
            h_x,
            h_qy,
            h_wx,
            h_wy,
            h_qmx,
            h_qmy,
            h_xf,
            h_thr,
            h_out.clone(),
            n_blocks as u32,
        );
        let mut bytes = client.read(alloc::vec![h_out]);
        let out_bytes = bytes.pop().unwrap();
        let gpu: Vec<i32> = i32::from_bytes(&out_bytes).to_vec();

        // DC slot is always 0 in the kernel output by contract.
        for b in 0..n_blocks {
            assert_eq!(gpu[b * 64], 0, "DC slot for block {b}");
        }
        let mut diffs = 0;
        for i in 0..gpu.len() {
            if i % 64 == 0 {
                continue;
            } // DC
            if gpu[i] != cpu[i] {
                diffs += 1;
                if diffs <= 5 {
                    eprintln!(
                        "[cfl_quantize parity diff] block={} pos={} gpu={} cpu={}",
                        i / 64,
                        i % 64,
                        gpu[i],
                        cpu[i]
                    );
                }
            }
        }
        // Allow at most 0.05% mismatches due to tie-to-even FP edge
        // cases between GPU and CPU rounding paths. The synthetic
        // input is unlikely to land any.
        let max_allowed = (gpu.len() as f64 * 0.0005) as usize;
        assert!(
            diffs <= max_allowed,
            "{diffs} > {max_allowed} (0.05%) mismatches"
        );
    }
}
