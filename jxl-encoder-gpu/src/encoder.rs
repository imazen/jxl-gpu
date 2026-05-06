// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! GPU-accelerated JPEG XL encoder facade.
//!
//! `imazen/jxl-gpu` is a sibling project to `jxl-encoder` — not a feature
//! flag of it. This module wraps `jxl-encoder` for the sequential parts
//! (ANS entropy coding, bitstream writer, container muxing, frame headers)
//! and progressively replaces the parallel parts (DCT, quantize, AC search,
//! EPF, masking) with GPU kernels from this crate.
//!
//! ## Status
//!
//! **Pre-alpha skeleton.** Currently delegates the full encode to
//! `jxl-encoder` (CPU path) just to prove the dependency works. Phase 4
//! will progressively swap in GPU kernels for each pipeline stage.
//!
//! ## API design
//!
//! - `GpuEncoder<R: Runtime>` — long-lived encoder holding a cubecl
//!   client + per-(width, height) instance cache. Construct once,
//!   reuse for many encodes of the same dimensions.
//! - `encode_lossy_via_cpu` — minimal proof-of-concept: takes pixels +
//!   `jxl_encoder::api::LossyConfig` and returns JXL bytes via the CPU
//!   path. Future replacements will swap CPU stages for GPU.
//!
//! Gated behind `feature = "encoder"` (default-on).

use alloc::vec;
use alloc::vec::Vec;

use cubecl::Runtime;
use cubecl::prelude::*;

use jxl_encoder::api::{LossyConfig, PixelLayout};

use crate::launch::block_l2::block_l2;
use crate::launch::cfl::find_best_multiplier;
use crate::launch::dct8::{dct_8x8, idct_8x8};
use crate::launch::dequant::dequant_dct8;
use crate::launch::entropy::entropy_coeffs_pixel;
use crate::launch::epf::{epf_step1, epf_step2, pad_plane};
use crate::launch::gab::gab_smooth;
use crate::launch::gaborish::gaborish_5x5;
use crate::launch::mask1x1::mask1x1;
use crate::launch::pixel_loss::pixel_loss;
use crate::launch::quantize::quantize_dct8;
use crate::launch::xyb::xyb_forward;

/// GPU-accelerated JXL encoder. Holds a long-lived cubecl client plus
/// per-(width, height) GPU buffer caches.
///
/// Construct once per process; reuse across many encodes. Per-instance
/// allocation is expensive (hundreds of ms at 1MP); the cache amortizes
/// it. See `kernels::xyb` and friends for the underlying kernels.
pub struct GpuEncoder<R: Runtime> {
    client: ComputeClient<R>,
    // TODO: per-(w,h) buffer cache:
    // instances: HashMap<(u32, u32), GpuInstance<R>>,
    _runtime: core::marker::PhantomData<R>,
}

impl<R: Runtime> GpuEncoder<R> {
    /// Construct a new encoder using the runtime's default device.
    pub fn new() -> Self {
        let device = <R as Runtime>::Device::default();
        let client = <R as Runtime>::client(&device);
        Self::with_client(client)
    }

    /// Construct from an existing cubecl client (e.g., to share devices
    /// across multiple GPU users).
    pub fn with_client(client: ComputeClient<R>) -> Self {
        Self {
            client,
            _runtime: core::marker::PhantomData,
        }
    }

    /// Access the underlying cubecl client.
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// **Pre-alpha proof-of-concept.** Encodes via `jxl-encoder`'s CPU
    /// path, ignoring the GPU client entirely. Verifies the dependency
    /// is wired up and the API shape compiles. Phase 4 work will
    /// progressively replace pipeline stages with GPU calls.
    ///
    /// # Errors
    ///
    /// Forwards any [`jxl_encoder::api::EncodeError`] from the wrapped
    /// CPU encoder.
    /// Convert linear RGB pixels to XYB on the GPU.
    ///
    /// Input: planar linear RGB, three channels of `n` f32 values each.
    /// Output: planar XYB, returned as `(x, y, b)` Vecs.
    ///
    /// Caller is responsible for sRGB→linear conversion. For the
    /// standard sRGB transfer function:
    ///
    /// ```text
    /// fn srgb_to_linear(v: f32) -> f32 {
    ///     if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
    /// }
    /// ```
    ///
    /// This is the first concrete Phase 4 progressive-replacement step:
    /// a public GPU-accelerated XYB transform that downstream encoders
    /// can call directly.
    pub fn xyb_from_linear_rgb(
        &self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = r.len();
        assert_eq!(g.len(), n, "g.len() != r.len()");
        assert_eq!(b.len(), n, "b.len() != r.len()");

        let h_r = self.client.create_from_slice(f32::as_bytes(r));
        let h_g = self.client.create_from_slice(f32::as_bytes(g));
        let h_b = self.client.create_from_slice(f32::as_bytes(b));
        let h_x = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        let h_y = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        let h_b_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

        xyb_forward::<R>(
            &self.client,
            h_r,
            h_g,
            h_b,
            h_x.clone(),
            h_y.clone(),
            h_b_out.clone(),
            n as u32,
        );

        let xb = self.client.read_one(h_x).expect("read x");
        let yb = self.client.read_one(h_y).expect("read y");
        let bb = self.client.read_one(h_b_out).expect("read b");
        (
            f32::from_bytes(&xb).to_vec(),
            f32::from_bytes(&yb).to_vec(),
            f32::from_bytes(&bb).to_vec(),
        )
    }

    /// Compute the per-pixel masking field from an XYB-Y channel.
    ///
    /// Mirrors `jxl_encoder_simd::compute_mask1x1` but on GPU. Output has
    /// the same shape as `xyb_y` (one f32 per pixel).
    pub fn mask1x1_field(&self, xyb_y: &[f32], width: u32, height: u32) -> Vec<f32> {
        let n = (width as usize) * (height as usize);
        assert_eq!(xyb_y.len(), n, "xyb_y length mismatch with width*height");
        let h_in = self.client.create_from_slice(f32::as_bytes(xyb_y));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        mask1x1::<R>(&self.client, h_in, h_out.clone(), width, height);
        let bytes = self.client.read_one(h_out).expect("read mask1x1");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Forward DCT8 on a contiguous batch of 8×8 blocks.
    ///
    /// Input layout: `num_blocks × 64` floats, row-major within each
    /// block. Output same shape, with DCT coefficients (transposed
    /// layout per libjxl convention — see `dct_8x8_scalar` docs).
    pub fn dct_8x8_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        let n = blocks.len();
        assert!(n.is_multiple_of(64), "blocks length must be multiple of 64");
        let num_blocks = (n / 64) as u32;
        let h_in = self.client.create_from_slice(f32::as_bytes(blocks));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        dct_8x8::<R>(&self.client, h_in, h_out.clone(), num_blocks);
        let bytes = self.client.read_one(h_out).expect("read dct");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Inverse DCT8 on a contiguous batch of 8×8 blocks (counterpart to
    /// [`dct_8x8_blocks`](Self::dct_8x8_blocks)).
    pub fn idct_8x8_blocks(&self, dct_coeffs: &[f32]) -> Vec<f32> {
        let n = dct_coeffs.len();
        assert!(n.is_multiple_of(64));
        let num_blocks = (n / 64) as u32;
        let h_in = self.client.create_from_slice(f32::as_bytes(dct_coeffs));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        idct_8x8::<R>(&self.client, h_in, h_out.clone(), num_blocks);
        let bytes = self.client.read_one(h_out).expect("read idct");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Gaborish-inverse 5×5 sharpening on a single channel. Returns a
    /// new buffer with the filter applied.
    #[allow(clippy::too_many_arguments)]
    pub fn gaborish_5x5_channel(
        &self,
        plane: &[f32],
        width: u32,
        height: u32,
        wc: f32,
        wr: f32,
        wd: f32,
        w_big_r: f32,
        wl: f32,
        w_big_d: f32,
    ) -> Vec<f32> {
        let n = (width as usize) * (height as usize);
        assert_eq!(plane.len(), n);
        let h_in = self.client.create_from_slice(f32::as_bytes(plane));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        gaborish_5x5::<R>(
            &self.client,
            h_in,
            h_out.clone(),
            width,
            height,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );
        let bytes = self.client.read_one(h_out).expect("read gaborish");
        f32::from_bytes(&bytes).to_vec()
    }

    /// 3×3 gab smooth on a single channel.
    pub fn gab_smooth_channel(
        &self,
        plane: &[f32],
        width: u32,
        height: u32,
        w_center: f32,
        w1: f32,
        w2: f32,
    ) -> Vec<f32> {
        let n = (width as usize) * (height as usize);
        assert_eq!(plane.len(), n);
        let h_in = self.client.create_from_slice(f32::as_bytes(plane));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        gab_smooth::<R>(
            &self.client,
            h_in,
            h_out.clone(),
            width,
            height,
            w_center,
            w1,
            w2,
        );
        let bytes = self.client.read_one(h_out).expect("read gab");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Per-block DCT8 quantize with dead-zone thresholding.
    ///
    /// `coeffs` and `weights` are `num_blocks * 64` f32 each.
    /// `qac_qm` is `num_blocks` f32 (per-block `qac * qm_mul`).
    /// `thresholds` is exactly 4 f32 (per-quadrant dead zone).
    /// Returns quantized i32 coefficients, same shape as input.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_dct8_blocks(
        &self,
        coeffs: &[f32],
        weights: &[f32],
        qac_qm: &[f32],
        thresholds: &[f32; 4],
    ) -> Vec<i32> {
        let n = coeffs.len();
        assert!(n.is_multiple_of(64));
        assert_eq!(weights.len(), n);
        let num_blocks = (n / 64) as u32;
        assert_eq!(qac_qm.len(), num_blocks as usize);
        let h_c = self.client.create_from_slice(f32::as_bytes(coeffs));
        let h_w = self.client.create_from_slice(f32::as_bytes(weights));
        let h_q = self.client.create_from_slice(f32::as_bytes(qac_qm));
        let h_t = self.client.create_from_slice(f32::as_bytes(&thresholds[..]));
        let h_o = self.client.create_from_slice(i32::as_bytes(&vec![0_i32; n]));
        quantize_dct8::<R>(
            &self.client,
            h_c,
            h_w,
            h_q,
            h_t,
            h_o.clone(),
            num_blocks,
        );
        let bytes = self.client.read_one(h_o).expect("read quant");
        i32::from_bytes(&bytes).to_vec()
    }

    /// Per-block entropy estimation in pixel-domain mode.
    ///
    /// Returns `(out_4xn, error_coeffs)` where `out_4xn` is `num_blocks * 4`
    /// f32 (per-block [entropy_sum, nzeros_sum, info_loss_sum=0,
    /// info_loss2_sum=0]) and `error_coeffs` is `num_blocks * n` f32
    /// (the writeback `weights[i] * (val - rval)` per coefficient).
    #[allow(clippy::too_many_arguments)]
    pub fn entropy_coeffs_pixel_blocks(
        &self,
        block_c: &[f32],
        block_y: &[f32],
        weights: &[f32],
        inv_weights: &[f32],
        n_per_block: u32,
        cmap_factor: f32,
        quant: f32,
        k_cost_delta: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let total = block_c.len();
        let n = n_per_block as usize;
        assert!(total.is_multiple_of(n));
        let num_blocks = (total / n) as u32;
        assert_eq!(block_y.len(), total);
        assert_eq!(weights.len(), total);
        assert_eq!(inv_weights.len(), total);

        let h_c = self.client.create_from_slice(f32::as_bytes(block_c));
        let h_y = self.client.create_from_slice(f32::as_bytes(block_y));
        let h_w = self.client.create_from_slice(f32::as_bytes(weights));
        let h_iw = self.client.create_from_slice(f32::as_bytes(inv_weights));
        let h_err = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; total]));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; (num_blocks as usize) * 4]));
        entropy_coeffs_pixel::<R>(
            &self.client,
            h_c,
            h_y,
            h_w,
            h_iw,
            h_err.clone(),
            h_out.clone(),
            num_blocks,
            n_per_block,
            cmap_factor,
            quant,
            k_cost_delta,
        );
        let out_bytes = self.client.read_one(h_out).expect("read out");
        let err_bytes = self.client.read_one(h_err).expect("read err");
        (
            f32::from_bytes(&out_bytes).to_vec(),
            f32::from_bytes(&err_bytes).to_vec(),
        )
    }

    /// Edge-replicate pad a single channel by `pad` pixels on each side.
    /// Output shape: `(width + 2*pad) × (height + 2*pad)`.
    pub fn pad_plane_channel(&self, plane: &[f32], width: u32, height: u32, pad: u32) -> Vec<f32> {
        let n_in = (width as usize) * (height as usize);
        assert_eq!(plane.len(), n_in);
        let dst_w = (width + 2 * pad) as usize;
        let dst_h = (height + 2 * pad) as usize;
        let n_out = dst_w * dst_h;
        let h_in = self.client.create_from_slice(f32::as_bytes(plane));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_out]));
        pad_plane::<R>(&self.client, h_in, h_out.clone(), width, height, pad);
        let bytes = self.client.read_one(h_out).expect("read pad");
        f32::from_bytes(&bytes).to_vec()
    }

    /// EPF Step 1 — 3×3 cross kernel with 3×3-plus SAD weights.
    /// Inputs are PADDED (`stride = width + 2*pad`); output is unpadded.
    /// Returns `(out_x, out_y, out_b)`.
    #[allow(clippy::too_many_arguments)]
    pub fn epf_step1_channels(
        &self,
        in_x: &[f32],
        in_y: &[f32],
        in_b: &[f32],
        inv_sigma: &[f32],
        width: u32,
        height: u32,
        xsize_blocks: u32,
        ysize_blocks: u32,
        pad: u32,
        sigma_scale: f32,
        border_sigma_mul: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n_out = (width as usize) * (height as usize);
        let h_ix = self.client.create_from_slice(f32::as_bytes(in_x));
        let h_iy = self.client.create_from_slice(f32::as_bytes(in_y));
        let h_ib = self.client.create_from_slice(f32::as_bytes(in_b));
        let h_is = self.client.create_from_slice(f32::as_bytes(inv_sigma));
        let h_ox = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_out]));
        let h_oy = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_out]));
        let h_ob = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_out]));
        epf_step1::<R>(
            &self.client,
            h_ix,
            h_iy,
            h_ib,
            h_ox.clone(),
            h_oy.clone(),
            h_ob.clone(),
            h_is,
            width,
            height,
            xsize_blocks,
            ysize_blocks,
            pad,
            sigma_scale,
            border_sigma_mul,
        );
        let xb = self.client.read_one(h_ox).expect("x");
        let yb = self.client.read_one(h_oy).expect("y");
        let bb = self.client.read_one(h_ob).expect("b");
        (
            f32::from_bytes(&xb).to_vec(),
            f32::from_bytes(&yb).to_vec(),
            f32::from_bytes(&bb).to_vec(),
        )
    }

    /// EPF Step 2 — 3×3 cross kernel with single-pixel SAD weights.
    /// Same I/O shape as `epf_step1_channels`.
    #[allow(clippy::too_many_arguments)]
    pub fn epf_step2_channels(
        &self,
        in_x: &[f32],
        in_y: &[f32],
        in_b: &[f32],
        inv_sigma: &[f32],
        width: u32,
        height: u32,
        xsize_blocks: u32,
        ysize_blocks: u32,
        pad: u32,
        sigma_scale: f32,
        border_sigma_mul: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n_out = (width as usize) * (height as usize);
        let h_ix = self.client.create_from_slice(f32::as_bytes(in_x));
        let h_iy = self.client.create_from_slice(f32::as_bytes(in_y));
        let h_ib = self.client.create_from_slice(f32::as_bytes(in_b));
        let h_is = self.client.create_from_slice(f32::as_bytes(inv_sigma));
        let h_ox = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_out]));
        let h_oy = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_out]));
        let h_ob = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_out]));
        epf_step2::<R>(
            &self.client,
            h_ix,
            h_iy,
            h_ib,
            h_ox.clone(),
            h_oy.clone(),
            h_ob.clone(),
            h_is,
            width,
            height,
            xsize_blocks,
            ysize_blocks,
            pad,
            sigma_scale,
            border_sigma_mul,
        );
        let xb = self.client.read_one(h_ox).expect("x");
        let yb = self.client.read_one(h_oy).expect("y");
        let bb = self.client.read_one(h_ob).expect("b");
        (
            f32::from_bytes(&xb).to_vec(),
            f32::from_bytes(&yb).to_vec(),
            f32::from_bytes(&bb).to_vec(),
        )
    }

    /// Per-block DCT8 dequantize with CfL restore (3 channels at once).
    /// Returns `(out_x, out_y, out_b)`.
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_dct8_blocks(
        &self,
        quant_x: &[i32],
        quant_y: &[i32],
        quant_b: &[i32],
        weights_x: &[f32],
        weights_y: &[f32],
        weights_b: &[f32],
        qac_qm_x: &[f32],
        qac_qm_y: &[f32],
        qac_qm_b: &[f32],
        x_factor: &[f32],
        b_factor: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n_coef = quant_x.len();
        assert!(n_coef.is_multiple_of(64));
        let nb = n_coef / 64;
        let num_blocks = nb as u32;
        let h_qx = self.client.create_from_slice(i32::as_bytes(quant_x));
        let h_qy = self.client.create_from_slice(i32::as_bytes(quant_y));
        let h_qb = self.client.create_from_slice(i32::as_bytes(quant_b));
        let h_wx = self.client.create_from_slice(f32::as_bytes(weights_x));
        let h_wy = self.client.create_from_slice(f32::as_bytes(weights_y));
        let h_wb = self.client.create_from_slice(f32::as_bytes(weights_b));
        let h_qmx = self.client.create_from_slice(f32::as_bytes(qac_qm_x));
        let h_qmy = self.client.create_from_slice(f32::as_bytes(qac_qm_y));
        let h_qmb = self.client.create_from_slice(f32::as_bytes(qac_qm_b));
        let h_xf = self.client.create_from_slice(f32::as_bytes(x_factor));
        let h_bf = self.client.create_from_slice(f32::as_bytes(b_factor));
        let h_ox = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_coef]));
        let h_oy = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_coef]));
        let h_ob = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_coef]));
        dequant_dct8::<R>(
            &self.client,
            h_qx,
            h_qy,
            h_qb,
            h_wx,
            h_wy,
            h_wb,
            h_qmx,
            h_qmy,
            h_qmb,
            h_xf,
            h_bf,
            h_ox.clone(),
            h_oy.clone(),
            h_ob.clone(),
            num_blocks,
        );
        let xb = self.client.read_one(h_ox).expect("x");
        let yb = self.client.read_one(h_oy).expect("y");
        let bb = self.client.read_one(h_ob).expect("b");
        (
            f32::from_bytes(&xb).to_vec(),
            f32::from_bytes(&yb).to_vec(),
            f32::from_bytes(&bb).to_vec(),
        )
    }

    /// Per-8×8-block masked weighted L2 error for 3-channel original/reconstructed.
    /// Returns `xsize_blocks * ysize_blocks` per-block costs.
    #[allow(clippy::too_many_arguments)]
    pub fn block_l2_errors(
        &self,
        orig_x: &[f32],
        orig_y: &[f32],
        orig_b: &[f32],
        recon_x: &[f32],
        recon_y: &[f32],
        recon_b: &[f32],
        mask: &[f32],
        xsize_blocks: u32,
        ysize_blocks: u32,
        padded_width: u32,
    ) -> Vec<f32> {
        let n_blocks = (xsize_blocks * ysize_blocks) as usize;
        let h_ox = self.client.create_from_slice(f32::as_bytes(orig_x));
        let h_oy = self.client.create_from_slice(f32::as_bytes(orig_y));
        let h_ob = self.client.create_from_slice(f32::as_bytes(orig_b));
        let h_rx = self.client.create_from_slice(f32::as_bytes(recon_x));
        let h_ry = self.client.create_from_slice(f32::as_bytes(recon_y));
        let h_rb = self.client.create_from_slice(f32::as_bytes(recon_b));
        let h_m = self.client.create_from_slice(f32::as_bytes(mask));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_blocks]));
        block_l2::<R>(
            &self.client,
            h_ox,
            h_oy,
            h_ob,
            h_rx,
            h_ry,
            h_rb,
            h_m,
            h_out.clone(),
            xsize_blocks,
            ysize_blocks,
            padded_width,
        );
        let bytes = self.client.read_one(h_out).expect("read block_l2");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Per-block 8th-power norm of masked pixel errors. Returns `num_blocks` f64.
    #[allow(clippy::too_many_arguments)]
    pub fn pixel_loss_blocks(
        &self,
        pixel_error: &[f32],
        mask: &[f32],
        mask_row_base: &[u32],
        mask_stride: u32,
        mask_offset: f32,
        block_width: u32,
        block_height: u32,
    ) -> Vec<f64> {
        let num_blocks = mask_row_base.len() as u32;
        let h_err = self.client.create_from_slice(f32::as_bytes(pixel_error));
        let h_mask = self.client.create_from_slice(f32::as_bytes(mask));
        let h_mrb = self.client.create_from_slice(u32::as_bytes(mask_row_base));
        let h_out = self
            .client
            .create_from_slice(f64::as_bytes(&vec![0.0_f64; num_blocks as usize]));
        pixel_loss::<R>(
            &self.client,
            h_err,
            h_mask,
            h_mrb,
            h_out.clone(),
            num_blocks,
            mask.len(),
            mask_stride,
            mask_offset,
            block_width,
            block_height,
        );
        let bytes = self.client.read_one(h_out).expect("read pixel_loss");
        f64::from_bytes(&bytes).to_vec()
    }

    /// Per-tile CfL multiplier search via regularized least-squares.
    /// Returns one i32 per tile (cast to i8 by caller; range [-128, 127]).
    pub fn cfl_multipliers(
        &self,
        values_m: &[f32],
        values_s: &[f32],
        bases: &[f32],
        num_per_tile: u32,
        distance_mul: f32,
    ) -> Vec<i32> {
        let num_tiles = bases.len() as u32;
        let h_m = self.client.create_from_slice(f32::as_bytes(values_m));
        let h_s = self.client.create_from_slice(f32::as_bytes(values_s));
        let h_b = self.client.create_from_slice(f32::as_bytes(bases));
        let h_o = self
            .client
            .create_from_slice(i32::as_bytes(&vec![0_i32; num_tiles as usize]));
        find_best_multiplier::<R>(
            &self.client,
            h_m,
            h_s,
            h_b,
            h_o.clone(),
            num_tiles,
            num_per_tile,
            distance_mul,
        );
        let bytes = self.client.read_one(h_o).expect("read cfl");
        i32::from_bytes(&bytes).to_vec()
    }

    pub fn encode_lossy_via_cpu(
        &self,
        config: &LossyConfig,
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<Vec<u8>, jxl_encoder::api::EncodeError> {
        config
            .encode_request(width, height, layout)
            .encode(pixels)
            .map_err(|e| e.decompose().0)
    }
}

impl<R: Runtime> Default for GpuEncoder<R> {
    fn default() -> Self {
        Self::new()
    }
}
