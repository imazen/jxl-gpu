// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-block DC quantization for DCT8 blocks.
//!
//! Mirrors `jxl_encoder::vardct::transform::transform_blocks_into`'s
//! DC extraction step (the `RAW_STRATEGY_DCT == 0` branch in both
//! the Y "Step 2" pre-quant block and the X/B "Step 7" post-CfL
//! block).
//!
//! For DCT8 the per-block DC value is just `dct_coeffs[0]` (no
//! `dc_from_dct_NxN` machinery needed for non-multi-block transforms).
//! `inv_factor = INV_DC_QUANT[c] * params.scale_dc` is a per-channel
//! scalar.
//!
//! Y formula (channel 1):
//!   quant_dc[1] = round(dct_coeffs[1][0] * inv_factor_y)
//!   float_dc[1] = dct_coeffs[1][0]
//!
//! X/B formula (channel 0 or 2, with CfL on DC):
//!   y_dc = quant_dc[1] (already quantized as i16)
//!   quant_dc[c] = round(dct_coeffs[c][0] * inv_factor_c
//!                         - y_dc * dc_cfl_factor[c])
//!   float_dc[c] = dct_coeffs[c][0]
//! where dc_cfl_factor = 0.0 for X (c=0), 0.5 for B (c=2).

use cubecl::prelude::*;

/// Round-half-away-from-zero for f32 → i16 (matches Rust's
/// `f32::round() as i16` exactly). This is what
/// `transform_blocks_into` uses for DC quant on the CPU side
/// (`(dct_coeffs[c][0] * inv_factor).round() as i16` — no
/// `round_ties_even` here, unlike the AC path).
#[cube]
fn round_dc_to_i16(x: f32) -> i16 {
    (f32::round(x)) as i16
}

/// Y channel DC quantize (per block). Reads `coeffs[block_idx * 64]`
/// (the DC coefficient at position 0 of each block); writes
/// `quant_dc[block_idx]` (i16) and `float_dc[block_idx]` (f32).
#[cube(launch_unchecked)]
pub fn quantize_dc_y_dct8_kernel(
    coeffs: &Array<f32>,
    quant_dc: &mut Array<i16>,
    float_dc: &mut Array<f32>,
    inv_factor: f32,
) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = quant_dc.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let dc = coeffs[block_idx * 64usize];
    float_dc[block_idx] = dc;
    quant_dc[block_idx] = round_dc_to_i16(dc * inv_factor);
}

/// X or B channel DC quantize (per block) with DC-side CfL.
/// `dc_cfl_factor` is 0.0 for X (channel 0) and 0.5 for B
/// (channel 2). Y must already be quantized into `quant_dc_y` —
/// run [`quantize_dc_y_dct8_kernel`] first.
#[cube(launch_unchecked)]
pub fn quantize_dc_chroma_dct8_kernel(
    coeffs_chroma: &Array<f32>,
    quant_dc_y: &Array<i16>,
    quant_dc_chroma: &mut Array<i16>,
    float_dc_chroma: &mut Array<f32>,
    inv_factor: f32,
    dc_cfl_factor: f32,
) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = quant_dc_chroma.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let dc = coeffs_chroma[block_idx * 64usize];
    float_dc_chroma[block_idx] = dc;
    let y_dc_f = quant_dc_y[block_idx] as f32;
    quant_dc_chroma[block_idx] = round_dc_to_i16(dc * inv_factor - y_dc_f * dc_cfl_factor);
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use crate::launch::quantize_dc::{quantize_dc_chroma_dct8, quantize_dc_y_dct8};
    use cubecl::Runtime;
    use cubecl::cuda::CudaRuntime;
    use cubecl::prelude::*;
    extern crate alloc;

    #[test]
    fn quantize_dc_y_matches_cpu_reference() {
        let device = Default::default();
        let client = CudaRuntime::client(&device);
        let n_blocks = 23usize;
        let coeffs: Vec<f32> = (0..n_blocks * 64)
            .map(|i| {
                if i % 64 == 0 {
                    // DC slot: pseudo-random non-trivial values
                    ((i / 64) as f32) * 1.7 - 4.5
                } else {
                    0.0
                }
            })
            .collect();
        let inv_factor = 0.2547_f32;

        let cpu: Vec<i16> = (0..n_blocks)
            .map(|b| (coeffs[b * 64] * inv_factor).round() as i16)
            .collect();

        let h_c = client.create_from_slice(f32::as_bytes(&coeffs));
        let h_q = client.empty(n_blocks * 2);
        let h_f = client.empty(n_blocks * 4);
        quantize_dc_y_dct8::<CudaRuntime>(
            &client,
            h_c,
            h_q.clone(),
            h_f,
            inv_factor,
            n_blocks as u32,
        );
        let mut bytes = client.read(alloc::vec![h_q]);
        let qb = bytes.pop().unwrap();
        let gpu: Vec<i16> = i16::from_bytes(&qb).to_vec();
        for b in 0..n_blocks {
            assert_eq!(gpu[b], cpu[b], "block {b}");
        }
    }

    #[test]
    fn quantize_dc_chroma_matches_cpu_reference_x_and_b() {
        let device = Default::default();
        let client = CudaRuntime::client(&device);
        let n_blocks = 23usize;
        let coeffs_x: Vec<f32> = (0..n_blocks * 64)
            .map(|i| {
                if i % 64 == 0 {
                    ((i / 64) as f32) * 0.9 - 2.0
                } else {
                    0.0
                }
            })
            .collect();
        let inv_factor_x = 0.18_f32;
        let dc_cfl_factor_x = 0.0_f32;
        let quant_dc_y: Vec<i16> = (0..n_blocks).map(|b| (b as i16) - 11).collect();

        let cpu_x: Vec<i16> = (0..n_blocks)
            .map(|b| {
                let dc = coeffs_x[b * 64];
                let yd = quant_dc_y[b] as f32;
                (dc * inv_factor_x - yd * dc_cfl_factor_x).round() as i16
            })
            .collect();

        let h_c = client.create_from_slice(f32::as_bytes(&coeffs_x));
        let h_qy = client.create_from_slice(i16::as_bytes(&quant_dc_y));
        let h_q = client.empty(n_blocks * 2);
        let h_f = client.empty(n_blocks * 4);
        quantize_dc_chroma_dct8::<CudaRuntime>(
            &client,
            h_c,
            h_qy,
            h_q.clone(),
            h_f,
            inv_factor_x,
            dc_cfl_factor_x,
            n_blocks as u32,
        );
        let mut bytes = client.read(alloc::vec![h_q]);
        let qb = bytes.pop().unwrap();
        let gpu: Vec<i16> = i16::from_bytes(&qb).to_vec();
        for b in 0..n_blocks {
            assert_eq!(gpu[b], cpu_x[b], "X block {b}");
        }

        // Now B with dc_cfl_factor 0.5
        let coeffs_b: Vec<f32> = (0..n_blocks * 64)
            .map(|i| {
                if i % 64 == 0 {
                    ((i / 64) as f32) * 1.2 - 3.0
                } else {
                    0.0
                }
            })
            .collect();
        let inv_factor_b = 0.21_f32;
        let dc_cfl_factor_b = 0.5_f32;

        let cpu_b: Vec<i16> = (0..n_blocks)
            .map(|b| {
                let dc = coeffs_b[b * 64];
                let yd = quant_dc_y[b] as f32;
                (dc * inv_factor_b - yd * dc_cfl_factor_b).round() as i16
            })
            .collect();

        let h_c2 = client.create_from_slice(f32::as_bytes(&coeffs_b));
        let h_qy2 = client.create_from_slice(i16::as_bytes(&quant_dc_y));
        let h_q2 = client.empty(n_blocks * 2);
        let h_f2 = client.empty(n_blocks * 4);
        quantize_dc_chroma_dct8::<CudaRuntime>(
            &client,
            h_c2,
            h_qy2,
            h_q2.clone(),
            h_f2,
            inv_factor_b,
            dc_cfl_factor_b,
            n_blocks as u32,
        );
        let mut bytes2 = client.read(alloc::vec![h_q2]);
        let qb2 = bytes2.pop().unwrap();
        let gpu_b: Vec<i16> = i16::from_bytes(&qb2).to_vec();
        for b in 0..n_blocks {
            assert_eq!(gpu_b[b], cpu_b[b], "B block {b}");
        }
    }
}
