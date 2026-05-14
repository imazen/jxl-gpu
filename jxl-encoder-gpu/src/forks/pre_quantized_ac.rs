// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! GPU producer for pre-quantized AC coefficients (DCT8 only).
//!
//! Mirrors the per-block sequence in
//! `jxl_encoder::vardct::transform::transform_blocks_into` for the
//! DCT8 strategy:
//!
//! 1. Gather pixel blocks from each XYB plane (8×8 tiles)
//! 2. Forward DCT8 each channel
//! 3. Quantize Y AC (with thresholds)
//! 4. CfL-subtract + quantize X AC and B AC (one fused kernel each;
//!    re-uses Y's already-quantized AC dequant'd via AdjustQuantBias)
//! 5. Quantize DC for Y, then for X / B with DC-side CfL
//! 6. Count non-zero AC coefficients per block
//! 7. Batched download of all per-block buffers
//!
//! All-DCT8 contract: this producer assumes every block uses
//! DCT8. Caller must verify before calling. Mixed-strategy support
//! is incremental future work — DCT8 already covers the "uniform"
//! image case and provides a parity-tested baseline.

use alloc::boxed::Box;
use alloc::vec::Vec;

use cubecl::prelude::*;
use cubecl::Runtime;

use crate::encoder::GpuEncoder;
use crate::persistent::GpuPlane;

/// Per-channel pre-quantized AC + DC outputs in flat layout. Caller
/// reshapes into `[Vec<Vec<...>>; 3]` for the jxl-encoder
/// `encode_from_pre_quantized_ac` consumer (see [`reshape_to_transform_output`]).
pub struct PreQuantizedDct8 {
    /// `quant_dc[c]`: per-block i16 quantized DC (length `n_blocks`).
    pub quant_dc: [Vec<i16>; 3],
    /// `quant_ac[c]`: per-block 64 i32 quantized AC coefficients
    /// (length `n_blocks * 64`). Position 0 of each block is the
    /// DC slot and is set to 0 (DC is in `quant_dc`).
    pub quant_ac: [Vec<i32>; 3],
    /// `nzeros[c]`: per-block u8 non-zero AC count (length `n_blocks`).
    pub nzeros: [Vec<u8>; 3],
    /// `raw_nzeros[c]`: per-block u16 pre-clamp non-zero AC count.
    /// For DCT8 this equals `nzeros[c]` (max 63, no clamping needed).
    pub raw_nzeros: [Vec<u16>; 3],
    /// `float_dc[c]`: per-block raw DC value (length `n_blocks`).
    pub float_dc: [Vec<f32>; 3],
}

/// Per-block CfL factors + per-channel scalars, expanded by the
/// host from the per-tile `CflMap` and `DistanceParams`.
pub struct PreQuantizedDct8Params {
    /// Per-block scale `qac = params.scale * raw_quant`. Same value
    /// for all 3 channels' AC quantizer (the channel-specific
    /// `qm_multiplier` is folded into per-channel `qac_qm_*` below).
    pub qac_per_block: Vec<f32>,
    /// Per-block X-channel CfL factor (`x_factor = ytox * x_factor_scale`),
    /// expanded from the per-tile `CflMap.x_factor` to per-block.
    pub x_factor_per_block: Vec<f32>,
    /// Per-block B-channel CfL factor.
    pub b_factor_per_block: Vec<f32>,
    /// Per-channel `qac * qm_multiplier`:
    ///   `qac_qm_x = qac * x_qm_mul`
    ///   `qac_qm_y = qac` (qm_multiplier == 1)
    ///   `qac_qm_b = qac * b_qm_mul`
    /// Length `n_blocks` each.
    pub qac_qm_x: Vec<f32>,
    pub qac_qm_y: Vec<f32>,
    pub qac_qm_b: Vec<f32>,
    /// DC scale per channel: `INV_DC_QUANT[c] * params.scale_dc`.
    pub inv_dc_factor_x: f32,
    pub inv_dc_factor_y: f32,
    pub inv_dc_factor_b: f32,
    /// Dead-zone thresholds (all 4 quadrants) per channel.
    pub thresholds_x: [f32; 4],
    pub thresholds_y: [f32; 4],
    pub thresholds_b: [f32; 4],
}

/// All-DCT8 producer. See module docs for the per-block algorithm.
///
/// `xx_g` / `xy_g` / `xb_g`: gaborished XYB planes (post-mask1x1,
/// pre-quant). Padded to multiples of 8 — caller verifies dims.
/// `xsize_blocks` × `ysize_blocks`: cpu-aligned per-block grid
/// dimensions. `padded_width` is the gpu plane stride.
pub fn compute_pre_quantized_ac_dct8_persistent<R: Runtime>(
    enc: &GpuEncoder<R>,
    xx_g: &GpuPlane<R>,
    xy_g: &GpuPlane<R>,
    xb_g: &GpuPlane<R>,
    xsize_blocks: usize,
    ysize_blocks: usize,
    weights_x_template: &[f32; 64],
    weights_y_template: &[f32; 64],
    weights_b_template: &[f32; 64],
    params: &PreQuantizedDct8Params,
) -> PreQuantizedDct8 {
    let n_blocks = xsize_blocks * ysize_blocks;
    let gpu_pw = xy_g.width() as usize;
    let gpu_ph = xy_g.height() as usize;
    debug_assert_eq!(params.qac_qm_x.len(), n_blocks);
    debug_assert_eq!(params.qac_qm_y.len(), n_blocks);
    debug_assert_eq!(params.qac_qm_b.len(), n_blocks);
    debug_assert_eq!(params.x_factor_per_block.len(), n_blocks);
    debug_assert_eq!(params.b_factor_per_block.len(), n_blocks);
    let _ = params.qac_per_block; // not directly used; folded into qac_qm_*

    let client = enc.client_ref();

    // FUSED PIPELINE: single mega-kernel does gather + DCT × 3 +
    // Y quantize + CfL X/B + chroma quantize + nzeros × 3 — all in
    // shared memory. Replaces ~10 separate kernel launches with 1.
    // Cubecl 0.10's per-launch overhead is the bottleneck on small
    // kernels; collapsing the chain is what makes the GPU producer
    // faster than CPU transform_and_quantize.
    //
    // DC quant remains separate (3 launches) because DC reads from
    // the DCT float coefs which the fused kernel doesn't expose
    // (they live in shared memory). Future optimization: do DC in
    // the same fused kernel by writing DC to a separate output
    // before the loop — saves another 2 launches.

    // Upload per-channel scalars + weights for the fused kernel.
    let h_qmx = client.create_from_slice(f32::as_bytes(&params.qac_qm_x));
    let h_qmy = client.create_from_slice(f32::as_bytes(&params.qac_qm_y));
    let h_qmb = client.create_from_slice(f32::as_bytes(&params.qac_qm_b));
    let h_xfac = client.create_from_slice(f32::as_bytes(&params.x_factor_per_block));
    let h_bfac = client.create_from_slice(f32::as_bytes(&params.b_factor_per_block));
    let h_wx = client.create_from_slice(f32::as_bytes(weights_x_template));
    let h_wy = client.create_from_slice(f32::as_bytes(weights_y_template));
    let h_wb = client.create_from_slice(f32::as_bytes(weights_b_template));
    let h_thr_x = client.create_from_slice(f32::as_bytes(&params.thresholds_x[..]));
    let h_thr_y = client.create_from_slice(f32::as_bytes(&params.thresholds_y[..]));
    let h_thr_b = client.create_from_slice(f32::as_bytes(&params.thresholds_b[..]));

    let h_q_x = client.empty(n_blocks * 64 * core::mem::size_of::<i32>());
    let h_q_y = client.empty(n_blocks * 64 * core::mem::size_of::<i32>());
    let h_q_b = client.empty(n_blocks * 64 * core::mem::size_of::<i32>());
    let h_nz_x = client.empty(n_blocks * core::mem::size_of::<u32>());
    let h_nz_y = client.empty(n_blocks * core::mem::size_of::<u32>());
    let h_nz_b = client.empty(n_blocks * core::mem::size_of::<u32>());

    let plane_n = gpu_pw * gpu_ph;
    crate::launch::fused_dct8_3ch::fused_dct8_3ch::<R>(
        client,
        xx_g.handle().clone(),
        xy_g.handle().clone(),
        xb_g.handle().clone(),
        h_wx,
        h_wy,
        h_wb,
        h_qmx,
        h_qmy,
        h_qmb,
        h_xfac,
        h_bfac,
        h_thr_x,
        h_thr_y,
        h_thr_b,
        h_q_x.clone(),
        h_q_y.clone(),
        h_q_b.clone(),
        h_nz_x.clone(),
        h_nz_y.clone(),
        h_nz_b.clone(),
        plane_n,
        n_blocks as u32,
        gpu_pw as u32,
        xsize_blocks as u32,
    );

    // DC still needs the float DCT coefs. Run a separate DCT8 for
    // each channel (3 small kernels) then DC quantize × 3. Future
    // chunk: extend the fused kernel to also write DC.
    let g_8y = enc.gather_blocks_persistent(xy_g, 8, 8);
    let g_8x = enc.gather_blocks_persistent(xx_g, 8, 8);
    let g_8b = enc.gather_blocks_persistent(xb_g, 8, 8);
    let dct_x = enc.dct_8x8_persistent(&g_8x);
    let dct_y = enc.dct_8x8_persistent(&g_8y);
    let dct_b = enc.dct_8x8_persistent(&g_8b);

    let h_qdc_y = client.empty(n_blocks * core::mem::size_of::<i16>());
    let h_fdc_y = client.empty(n_blocks * core::mem::size_of::<f32>());
    crate::launch::quantize_dc::quantize_dc_y_dct8::<R>(
        client,
        dct_y.handle().clone(),
        h_qdc_y.clone(),
        h_fdc_y.clone(),
        params.inv_dc_factor_y,
        n_blocks as u32,
    );
    let h_qdc_x = client.empty(n_blocks * core::mem::size_of::<i16>());
    let h_fdc_x = client.empty(n_blocks * core::mem::size_of::<f32>());
    crate::launch::quantize_dc::quantize_dc_chroma_dct8::<R>(
        client,
        dct_x.handle().clone(),
        h_qdc_y.clone(),
        h_qdc_x.clone(),
        h_fdc_x.clone(),
        params.inv_dc_factor_x,
        0.0,
        n_blocks as u32,
    );
    let h_qdc_b = client.empty(n_blocks * core::mem::size_of::<i16>());
    let h_fdc_b = client.empty(n_blocks * core::mem::size_of::<f32>());
    crate::launch::quantize_dc::quantize_dc_chroma_dct8::<R>(
        client,
        dct_b.handle().clone(),
        h_qdc_y.clone(),
        h_qdc_b.clone(),
        h_fdc_b.clone(),
        params.inv_dc_factor_b,
        0.5,
        n_blocks as u32,
    );

    // Step 7: batched download (15 buffers — one sync barrier).
    let mut all_bytes = client.read(alloc::vec![
        h_q_x.clone(), h_q_y.clone(), h_q_b.clone(),       // quant_ac × 3
        h_qdc_x, h_qdc_y, h_qdc_b,                          // quant_dc × 3
        h_fdc_x, h_fdc_y, h_fdc_b,                          // float_dc × 3
        h_nz_x, h_nz_y, h_nz_b,                             // nzeros × 3
    ]);
    // Drain in reverse to match push order.
    let nz_b_b = all_bytes.pop().expect("nz_b");
    let nz_y_b = all_bytes.pop().expect("nz_y");
    let nz_x_b = all_bytes.pop().expect("nz_x");
    let fdc_b_b = all_bytes.pop().expect("fdc_b");
    let fdc_y_b = all_bytes.pop().expect("fdc_y");
    let fdc_x_b = all_bytes.pop().expect("fdc_x");
    let qdc_b_b = all_bytes.pop().expect("qdc_b");
    let qdc_y_b = all_bytes.pop().expect("qdc_y");
    let qdc_x_b = all_bytes.pop().expect("qdc_x");
    let q_b_b = all_bytes.pop().expect("q_b");
    let q_y_b = all_bytes.pop().expect("q_y");
    let q_x_b = all_bytes.pop().expect("q_x");

    let q_x_v: Vec<i32> = i32::from_bytes(&q_x_b).to_vec();
    let q_y_v: Vec<i32> = i32::from_bytes(&q_y_b).to_vec();
    let q_b_v: Vec<i32> = i32::from_bytes(&q_b_b).to_vec();
    let qdc_x_v: Vec<i16> = i16::from_bytes(&qdc_x_b).to_vec();
    let qdc_y_v: Vec<i16> = i16::from_bytes(&qdc_y_b).to_vec();
    let qdc_b_v: Vec<i16> = i16::from_bytes(&qdc_b_b).to_vec();
    let fdc_x_v: Vec<f32> = f32::from_bytes(&fdc_x_b).to_vec();
    let fdc_y_v: Vec<f32> = f32::from_bytes(&fdc_y_b).to_vec();
    let fdc_b_v: Vec<f32> = f32::from_bytes(&fdc_b_b).to_vec();
    let nz_x_u32: Vec<u32> = u32::from_bytes(&nz_x_b).to_vec();
    let nz_y_u32: Vec<u32> = u32::from_bytes(&nz_y_b).to_vec();
    let nz_b_u32: Vec<u32> = u32::from_bytes(&nz_b_b).to_vec();

    // Convert nzeros u32 → u8 (clamp to 255) and u16 (raw).
    let to_u8 = |v: &[u32]| v.iter().map(|&n| n.min(255) as u8).collect::<Vec<u8>>();
    let to_u16 = |v: &[u32]| v.iter().map(|&n| n as u16).collect::<Vec<u16>>();

    PreQuantizedDct8 {
        quant_dc: [qdc_x_v, qdc_y_v, qdc_b_v],
        quant_ac: [q_x_v, q_y_v, q_b_v],
        nzeros: [to_u8(&nz_x_u32), to_u8(&nz_y_u32), to_u8(&nz_b_u32)],
        raw_nzeros: [to_u16(&nz_x_u32), to_u16(&nz_y_u32), to_u16(&nz_b_u32)],
        float_dc: [fdc_x_v, fdc_y_v, fdc_b_v],
    }
}

/// Reshape flat per-channel buffers into the nested `Vec<Vec<...>>`
/// shape `VarDctEncoder::encode_from_pre_quantized_ac` expects.
/// Caller passes `cpu_xsize_blocks` × `cpu_ysize_blocks` (the
/// jxl-encoder's per-block grid; matches `pre_quantized.xsize_blocks
/// × ysize_blocks`).
pub fn reshape_to_transform_output(
    pq: PreQuantizedDct8,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> ReshapedTransformOutput {
    let n_blocks = xsize_blocks * ysize_blocks;
    debug_assert_eq!(pq.quant_dc[0].len(), n_blocks);

    let mut quant_dc: [Vec<Vec<i16>>; 3] = core::array::from_fn(|_| Vec::new());
    let mut quant_ac: [Vec<Vec<[i32; 64]>>; 3] = core::array::from_fn(|_| Vec::new());
    let mut nzeros: [Vec<Vec<u8>>; 3] = core::array::from_fn(|_| Vec::new());
    let mut raw_nzeros: [Vec<Vec<u16>>; 3] = core::array::from_fn(|_| Vec::new());

    for c in 0..3 {
        quant_dc[c].reserve_exact(ysize_blocks);
        quant_ac[c].reserve_exact(ysize_blocks);
        nzeros[c].reserve_exact(ysize_blocks);
        raw_nzeros[c].reserve_exact(ysize_blocks);
        for by in 0..ysize_blocks {
            let row_off = by * xsize_blocks;
            quant_dc[c].push(pq.quant_dc[c][row_off..row_off + xsize_blocks].to_vec());
            nzeros[c].push(pq.nzeros[c][row_off..row_off + xsize_blocks].to_vec());
            raw_nzeros[c].push(pq.raw_nzeros[c][row_off..row_off + xsize_blocks].to_vec());

            let mut ac_row: Vec<[i32; 64]> = Vec::with_capacity(xsize_blocks);
            for bx in 0..xsize_blocks {
                let off = (row_off + bx) * 64;
                let mut blk = [0i32; 64];
                blk.copy_from_slice(&pq.quant_ac[c][off..off + 64]);
                ac_row.push(blk);
            }
            quant_ac[c].push(ac_row);
        }
    }
    ReshapedTransformOutput {
        quant_dc,
        quant_ac,
        nzeros,
        raw_nzeros,
        float_dc: pq.float_dc, // already flat per-block
    }
}

/// Output of [`reshape_to_transform_output`]: shape matches what the
/// jxl-encoder `encode_from_pre_quantized_ac` consumer expects.
pub struct ReshapedTransformOutput {
    pub quant_dc: [Vec<Vec<i16>>; 3],
    pub quant_ac: [Vec<Vec<[i32; 64]>>; 3],
    pub nzeros: [Vec<Vec<u8>>; 3],
    pub raw_nzeros: [Vec<Vec<u16>>; 3],
    pub float_dc: [Vec<f32>; 3],
}

// Suppress an unused-import warning when no tests are compiled.
#[allow(dead_code)]
fn _box_marker(_: Box<u8>) {}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use cubecl::cuda::CudaRuntime;

    /// Smoke test: run the full DCT8 GPU producer on a tiny
    /// synthetic image and verify the output buffer shapes /
    /// non-trivial content. Real bitstream parity vs CPU
    /// transform_and_quantize is the next chunk (needs encode_from_precomputed
    /// + encode_from_pre_quantized_ac comparison).
    #[test]
    fn pre_quantized_dct8_orchestrator_smoke() {
        let enc: GpuEncoder<CudaRuntime> = GpuEncoder::new();
        let xsize_blocks = 4usize;
        let ysize_blocks = 4usize;
        let n_blocks = xsize_blocks * ysize_blocks;
        let pw = xsize_blocks * 8;
        let ph = ysize_blocks * 8;

        // Pseudo-random XYB planes (post-gaborish-equivalent values).
        let make_plane = |seed: usize| -> Vec<f32> {
            (0..pw * ph)
                .map(|i| {
                    let s = ((i + seed * 17) as f32) * 0.0123;
                    0.05 + 0.04 * s.sin()
                })
                .collect()
        };
        let xx = make_plane(1);
        let xy = make_plane(7);
        let xb = make_plane(13);

        let xx_g = enc.upload_plane(&xx, pw as u32, ph as u32);
        let xy_g = enc.upload_plane(&xy, pw as u32, ph as u32);
        let xb_g = enc.upload_plane(&xb, pw as u32, ph as u32);

        // Synthetic non-uniform DCT8 weights / qac / cfl_map.
        let weights_x: [f32; 64] = core::array::from_fn(|i| 0.5 + i as f32 * 0.05);
        let weights_y: [f32; 64] = core::array::from_fn(|i| 0.7 + i as f32 * 0.03);
        let weights_b: [f32; 64] = core::array::from_fn(|i| 0.6 + i as f32 * 0.04);
        let qac_per_block: Vec<f32> = (0..n_blocks).map(|i| 1.5 + i as f32 * 0.02).collect();
        // qm_multiplier: X = 1.05, Y = 1.0, B = 0.95 (synthetic).
        let qac_qm_x: Vec<f32> = qac_per_block.iter().map(|&q| q * 1.05).collect();
        let qac_qm_y: Vec<f32> = qac_per_block.clone();
        let qac_qm_b: Vec<f32> = qac_per_block.iter().map(|&q| q * 0.95).collect();
        let x_factor_per_block: Vec<f32> = (0..n_blocks).map(|i| 0.1 + i as f32 * 0.005).collect();
        let b_factor_per_block: Vec<f32> = (0..n_blocks).map(|i| -0.2 + i as f32 * 0.003).collect();

        let params = PreQuantizedDct8Params {
            qac_per_block,
            x_factor_per_block,
            b_factor_per_block,
            qac_qm_x,
            qac_qm_y,
            qac_qm_b,
            inv_dc_factor_x: 4096.0 / 1024.0,
            inv_dc_factor_y: 512.0 / 1024.0,
            inv_dc_factor_b: 256.0 / 1024.0,
            thresholds_x: [0.58, 0.62, 0.62, 0.62],
            thresholds_y: [0.56, 0.62, 0.62, 0.62],
            thresholds_b: [0.58, 0.62, 0.62, 0.62],
        };

        let out = compute_pre_quantized_ac_dct8_persistent(
            &enc, &xx_g, &xy_g, &xb_g,
            xsize_blocks, ysize_blocks,
            &weights_x, &weights_y, &weights_b,
            &params,
        );

        // Shape checks.
        for c in 0..3 {
            assert_eq!(out.quant_dc[c].len(), n_blocks, "quant_dc[{c}]");
            assert_eq!(out.quant_ac[c].len(), n_blocks * 64, "quant_ac[{c}]");
            assert_eq!(out.nzeros[c].len(), n_blocks, "nzeros[{c}]");
            assert_eq!(out.raw_nzeros[c].len(), n_blocks, "raw_nzeros[{c}]");
            assert_eq!(out.float_dc[c].len(), n_blocks, "float_dc[{c}]");
        }

        // DC slot of every quant_ac block must be 0 (DC is in quant_dc).
        for c in 0..3 {
            for b in 0..n_blocks {
                assert_eq!(out.quant_ac[c][b * 64], 0,
                    "quant_ac[{c}] block {b} DC slot must be 0");
            }
        }
        // float_dc must be finite + non-trivially varied.
        for c in 0..3 {
            let mut all_eq = true;
            let v0 = out.float_dc[c][0];
            for &v in &out.float_dc[c] {
                assert!(v.is_finite(), "float_dc[{c}] non-finite");
                if (v - v0).abs() > 1e-9 { all_eq = false; }
            }
            assert!(!all_eq, "float_dc[{c}] should vary across blocks");
        }
        // nzeros must be in [0, 63] for DCT8.
        for c in 0..3 {
            for &n in &out.nzeros[c] {
                assert!(n <= 63, "nzeros[{c}] exceeded 63");
            }
        }

        // Reshape smoke: dims match.
        let r = reshape_to_transform_output(out, xsize_blocks, ysize_blocks);
        for c in 0..3 {
            assert_eq!(r.quant_dc[c].len(), ysize_blocks);
            assert_eq!(r.quant_dc[c][0].len(), xsize_blocks);
            assert_eq!(r.quant_ac[c].len(), ysize_blocks);
            assert_eq!(r.quant_ac[c][0].len(), xsize_blocks);
            assert_eq!(r.quant_ac[c][0][0].len(), 64);
        }
    }
}
