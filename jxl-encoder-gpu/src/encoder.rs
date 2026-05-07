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

use crate::launch::adaptive_quant::{compute_pre_erosion, per_block_modulations};
use crate::launch::block_l2::block_l2;
use crate::launch::cfl::{find_best_multiplier, find_best_multiplier_newton};
use crate::launch::dct4::{
    dct_4x4_full, dct_4x8_full, dct_8x4_full, idct_4x4_full, idct_4x8_full, idct_8x4_full,
};
use crate::launch::dct8::{dct_8x8, idct_8x8};
use crate::launch::identity::{identity_forward, identity_inverse};
use crate::launch::dct16::{dct_8x16, dct_16x8, dct_16x16, idct_8x16, idct_16x8, idct_16x16};
use crate::launch::dct32::{dct_16x32, dct_32x16, dct_32x32, idct_16x32, idct_32x16, idct_32x32};
use crate::launch::dct64::{dct_32x64, dct_64x32, dct_64x64, idct_32x64, idct_64x32, idct_64x64};
use crate::launch::dct2x2::{dct2x2_forward, dct2x2_inverse};
use crate::launch::denoise::denoise as denoise_launch;
use crate::launch::fuzzy_erosion::{fuzzy_erosion as fuzzy_erosion_launch, fuzzy_erosion_kmul};
use crate::launch::dequant::dequant_dct8;
use crate::launch::entropy::entropy_coeffs_pixel;
use crate::launch::epf::{epf_step1, epf_step2, pad_plane};
use crate::launch::gab::gab_smooth;
use crate::launch::gaborish::gaborish_5x5;
use crate::launch::mask1x1::mask1x1;
use crate::launch::pixel_loss::pixel_loss;
use crate::launch::quantize::quantize_dct8;
use crate::launch::xyb::{xyb_forward, xyb_inverse};

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

    /// Borrow the underlying cubecl client. Used by the persistent-API
    /// methods in [`crate::persistent`] to chain custom launches.
    pub(crate) fn client_ref(&self) -> &ComputeClient<R> {
        &self.client
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

    /// Inverse XYB → planar linear RGB. Mirrors
    /// `jxl_encoder_simd::xyb_to_linear_rgb_planar` but on GPU.
    /// Returns three new buffers `(R, G, B)` each of length `n`.
    pub fn xyb_to_linear_rgb_planar(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = xyb_x.len();
        assert_eq!(xyb_y.len(), n);
        assert_eq!(xyb_b.len(), n);

        let h_x = self.client.create_from_slice(f32::as_bytes(xyb_x));
        let h_y = self.client.create_from_slice(f32::as_bytes(xyb_y));
        let h_b = self.client.create_from_slice(f32::as_bytes(xyb_b));
        let h_r = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        let h_g_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        let h_b_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

        xyb_inverse::<R>(
            &self.client,
            h_x,
            h_y,
            h_b,
            h_r.clone(),
            h_g_out.clone(),
            h_b_out.clone(),
            n as u32,
        );

        let rb = self.client.read_one(h_r).expect("read r");
        let gb = self.client.read_one(h_g_out).expect("read g");
        let bb = self.client.read_one(h_b_out).expect("read b");
        (
            f32::from_bytes(&rb).to_vec(),
            f32::from_bytes(&gb).to_vec(),
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

    /// Per-pixel Wiener denoise (5×5 local statistics) on one channel.
    ///
    /// Mirrors `jxl_encoder_simd::noise::denoise_channel_scalar`. For
    /// each pixel: noise variance is looked up from `y_channel` via the
    /// 8-point `noise_lut` (interpolated, scaled by `denoise_scale²`).
    /// Output is the Wiener-filtered `orig`; pixels where the noise
    /// estimate falls below `EPS` pass through `orig[idx]` unchanged.
    ///
    /// `noise_lut` is the per-frame noise LUT (one of `NoiseParams::lut`
    /// from `jxl_encoder::vardct::noise`); `denoise_scale = denoise_fraction
    /// / (quality_coef * 1.4)` (see `denoise_xyb`).
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_channel(
        &self,
        orig: &[f32],
        y_channel: &[f32],
        noise_lut: &[f32; 8],
        width: u32,
        height: u32,
        denoise_scale: f32,
    ) -> Vec<f32> {
        let n = (width as usize) * (height as usize);
        assert_eq!(orig.len(), n, "orig length mismatch");
        assert_eq!(y_channel.len(), n, "y_channel length mismatch");
        let h_orig = self.client.create_from_slice(f32::as_bytes(orig));
        let h_y = self.client.create_from_slice(f32::as_bytes(y_channel));
        let h_lut = self.client.create_from_slice(f32::as_bytes(noise_lut));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        denoise_launch::<R>(
            &self.client,
            h_orig,
            h_y,
            h_lut,
            h_out.clone(),
            width,
            height,
            denoise_scale,
        );
        let bytes = self.client.read_one(h_out).expect("read denoise");
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

    /// Fuzzy-erosion + 2× downsample on a 2D plane. Mirrors
    /// `jxl_encoder::vardct::adaptive_quant::fuzzy_erosion`.
    ///
    /// For each output pixel (size `out_w × out_h` = `region_w/2 × region_h/2`),
    /// reads four input pixels in the source plane (offset by `from_x0`,
    /// `from_y0`), computes the 3×3 min-of-4 weighted sum at each, and
    /// accumulates the 4 contributions.
    ///
    /// `butteraugli_target` is used to derive `k_mul` weights via
    /// [`fuzzy_erosion_kmul`]; pass directly if you want to override.
    #[allow(clippy::too_many_arguments)]
    pub fn fuzzy_erosion_plane(
        &self,
        src: &[f32],
        src_w: u32,
        src_h: u32,
        from_x0: u32,
        from_y0: u32,
        region_w: u32,
        region_h: u32,
        butteraugli_target: f32,
    ) -> (Vec<f32>, u32, u32) {
        assert_eq!(src.len(), (src_w as usize) * (src_h as usize));
        let out_w = region_w / 2;
        let out_h = region_h / 2;
        let n_out = (out_w as usize) * (out_h as usize);
        let h_src = self.client.create_from_slice(f32::as_bytes(src));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_out]));
        let k_mul = fuzzy_erosion_kmul(butteraugli_target);
        fuzzy_erosion_launch::<R>(
            &self.client,
            h_src,
            h_out.clone(),
            src_w,
            src_h,
            from_x0,
            from_y0,
            out_w,
            out_h,
            k_mul,
        );
        let bytes = self.client.read_one(h_out).expect("read fuzzy_erosion");
        (f32::from_bytes(&bytes).to_vec(), out_w, out_h)
    }

    /// IDENTITY transform on a contiguous batch of 8×8 blocks. Mirrors
    /// `jxl_encoder::vardct::dct::special::identity_transform` exactly
    /// (bit-exact parity).
    pub fn identity_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        let n = blocks.len();
        assert!(n.is_multiple_of(64), "blocks length must be multiple of 64");
        let num_blocks = (n / 64) as u32;
        let h_in = self.client.create_from_slice(f32::as_bytes(blocks));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        identity_forward::<R>(&self.client, h_in, h_out.clone(), num_blocks);
        let bytes = self.client.read_one(h_out).expect("read identity");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Inverse IDENTITY transform (counterpart to
    /// [`identity_blocks`](Self::identity_blocks)).
    pub fn inverse_identity_blocks(&self, coeffs: &[f32]) -> Vec<f32> {
        let n = coeffs.len();
        assert!(n.is_multiple_of(64));
        let num_blocks = (n / 64) as u32;
        let h_in = self.client.create_from_slice(f32::as_bytes(coeffs));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        identity_inverse::<R>(&self.client, h_in, h_out.clone(), num_blocks);
        let bytes = self.client.read_one(h_out).expect("read inv identity");
        f32::from_bytes(&bytes).to_vec()
    }

    /// DCT2X2 transform on a contiguous batch of 8×8 blocks. Mirrors
    /// `jxl_encoder::vardct::dct::special::dct2x2_transform` exactly
    /// (bit-exact parity, 3 hierarchical Hadamard passes at S=8/4/2).
    pub fn dct2x2_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        let n = blocks.len();
        assert!(n.is_multiple_of(64));
        let num_blocks = (n / 64) as u32;
        let h_in = self.client.create_from_slice(f32::as_bytes(blocks));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        dct2x2_forward::<R>(&self.client, h_in, h_out.clone(), num_blocks);
        let bytes = self.client.read_one(h_out).expect("read dct2x2");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Inverse DCT2X2 transform (counterpart to
    /// [`dct2x2_blocks`](Self::dct2x2_blocks)).
    pub fn inverse_dct2x2_blocks(&self, coeffs: &[f32]) -> Vec<f32> {
        let n = coeffs.len();
        assert!(n.is_multiple_of(64));
        let num_blocks = (n / 64) as u32;
        let h_in = self.client.create_from_slice(f32::as_bytes(coeffs));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        dct2x2_inverse::<R>(&self.client, h_in, h_out.clone(), num_blocks);
        let bytes = self.client.read_one(h_out).expect("read inv dct2x2");
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
        let h_t = self
            .client
            .create_from_slice(f32::as_bytes(&thresholds[..]));
        let h_o = self
            .client
            .create_from_slice(i32::as_bytes(&vec![0_i32; n]));
        quantize_dct8::<R>(&self.client, h_c, h_w, h_q, h_t, h_o.clone(), num_blocks);
        let bytes = self.client.read_one(h_o).expect("read quant");
        i32::from_bytes(&bytes).to_vec()
    }

    /// Quantize a contiguous batch of larger-strategy blocks (DCT16,
    /// DCT16x8/8x16, DCT32, DCT32x16/16x32, DCT64, DCT64x32/32x64).
    ///
    /// Same dead-zone semantics as [`Self::quantize_dct8_blocks`] but
    /// the per-block coefficient count and LLF region are
    /// strategy-dependent and supplied by the caller via
    /// `grid_width`/`grid_height` and `llf_x`/`llf_y` (matching the
    /// upstream `quantize_large_scalar` parameters).
    ///
    /// `coeffs.len()` and `weights.len()` must be `num_blocks *
    /// grid_width * grid_height`. Returns the quantized i32 buffer of
    /// the same shape.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_large_blocks(
        &self,
        coeffs: &[f32],
        weights: &[f32],
        qac_qm: &[f32],
        thresholds: &[f32; 4],
        grid_width: u32,
        grid_height: u32,
        llf_x: u32,
        llf_y: u32,
    ) -> Vec<i32> {
        let block_size = (grid_width as usize) * (grid_height as usize);
        let n = coeffs.len();
        assert!(n.is_multiple_of(block_size));
        assert_eq!(weights.len(), n);
        let num_blocks = (n / block_size) as u32;
        assert_eq!(qac_qm.len(), num_blocks as usize);
        let h_c = self.client.create_from_slice(f32::as_bytes(coeffs));
        let h_w = self.client.create_from_slice(f32::as_bytes(weights));
        let h_q = self.client.create_from_slice(f32::as_bytes(qac_qm));
        let h_t = self.client.create_from_slice(f32::as_bytes(&thresholds[..]));
        let h_o = self
            .client
            .create_from_slice(i32::as_bytes(&vec![0_i32; n]));
        crate::launch::quantize::quantize_large::<R>(
            &self.client,
            h_c,
            h_w,
            h_q,
            h_t,
            h_o.clone(),
            num_blocks,
            grid_width,
            grid_height,
            llf_x,
            llf_y,
        );
        let bytes = self.client.read_one(h_o).expect("read quantize_large");
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

    /// Per-tile CfL multiplier search via Newton's method (warm-started
    /// from LS, refines toward smoothed-L1 optimum).
    #[allow(clippy::too_many_arguments)]
    pub fn cfl_multipliers_newton(
        &self,
        values_m: &[f32],
        values_s: &[f32],
        bases: &[f32],
        num_per_tile: u32,
        distance_mul: f32,
        eps: f32,
        max_iters: u32,
    ) -> Vec<i32> {
        let num_tiles = bases.len() as u32;
        let h_m = self.client.create_from_slice(f32::as_bytes(values_m));
        let h_s = self.client.create_from_slice(f32::as_bytes(values_s));
        let h_b = self.client.create_from_slice(f32::as_bytes(bases));
        let h_o = self
            .client
            .create_from_slice(i32::as_bytes(&vec![0_i32; num_tiles as usize]));
        find_best_multiplier_newton::<R>(
            &self.client,
            h_m,
            h_s,
            h_b,
            h_o.clone(),
            num_tiles,
            num_per_tile,
            distance_mul,
            eps,
            max_iters,
        );
        let bytes = self.client.read_one(h_o).expect("read cfl_newton");
        i32::from_bytes(&bytes).to_vec()
    }

    /// Adaptive-quant pre-erosion map. Caller pre-computes the output
    /// dimensions from the tile bounds (see `compute_pre_erosion_kernel`
    /// docs for the formula).
    #[allow(clippy::too_many_arguments)]
    pub fn pre_erosion(
        &self,
        xyb_y: &[f32],
        width: u32,
        height: u32,
        x0: u32,
        y_start: u32,
        pre_erosion_w: u32,
        pre_erosion_h: u32,
    ) -> Vec<f32> {
        let n_in = (width as usize) * (height as usize);
        assert_eq!(xyb_y.len(), n_in);
        let n_out = (pre_erosion_w as usize) * (pre_erosion_h as usize);
        let h_in = self.client.create_from_slice(f32::as_bytes(xyb_y));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_out]));
        compute_pre_erosion::<R>(
            &self.client,
            h_in,
            h_out.clone(),
            n_in,
            width,
            height,
            x0,
            y_start,
            pre_erosion_w,
            pre_erosion_h,
        );
        let bytes = self.client.read_one(h_out).expect("read pre_erosion");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Per-block adaptive-quant modulations (mask + gamma + hf + blue).
    /// `aq_map` is read AND written in-place.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_per_block_modulations(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        aq_map: &mut [f32],
        stride: u32,
        aq_map_stride: u32,
        rect_x0_blocks: u32,
        rect_y0_blocks: u32,
        rect_w_blocks: u32,
        rect_h_blocks: u32,
        butteraugli_target: f32,
        scale: f32,
    ) {
        let xyb_n = xyb_y.len();
        let aq_n = aq_map.len();
        let h_x = self.client.create_from_slice(f32::as_bytes(xyb_x));
        let h_y = self.client.create_from_slice(f32::as_bytes(xyb_y));
        let h_b = self.client.create_from_slice(f32::as_bytes(xyb_b));
        let h_aq = self.client.create_from_slice(f32::as_bytes(aq_map));
        per_block_modulations::<R>(
            &self.client,
            h_x,
            h_y,
            h_b,
            h_aq.clone(),
            xyb_n,
            aq_n,
            stride,
            aq_map_stride,
            rect_x0_blocks,
            rect_y0_blocks,
            rect_w_blocks,
            rect_h_blocks,
            butteraugli_target,
            scale,
        );
        let bytes = self.client.read_one(h_aq).expect("read aq_map");
        let new_aq: &[f32] = f32::from_bytes(&bytes);
        aq_map.copy_from_slice(new_aq);
    }

    /// Forward DCT16×16 on a contiguous batch of 16×16 blocks
    /// (`num_blocks * 256` floats).
    pub fn dct_16x16_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 256, |c, h_in, h_out, n| {
            dct_16x16::<R>(c, h_in, h_out, n)
        })
    }

    /// Inverse DCT16×16.
    pub fn idct_16x16_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 256, |c, h_in, h_out, n| {
            idct_16x16::<R>(c, h_in, h_out, n)
        })
    }

    /// Forward DCT32×32 (`num_blocks * 1024` floats).
    pub fn dct_32x32_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 1024, |c, h_in, h_out, n| {
            dct_32x32::<R>(c, h_in, h_out, n)
        })
    }

    /// Inverse DCT32×32.
    pub fn idct_32x32_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 1024, |c, h_in, h_out, n| {
            idct_32x32::<R>(c, h_in, h_out, n)
        })
    }

    /// Forward DCT64×64 (`num_blocks * 4096` floats).
    pub fn dct_64x64_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 4096, |c, h_in, h_out, n| {
            dct_64x64::<R>(c, h_in, h_out, n)
        })
    }

    /// Inverse DCT64×64.
    pub fn idct_64x64_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 4096, |c, h_in, h_out, n| {
            idct_64x64::<R>(c, h_in, h_out, n)
        })
    }

    /// Forward DCT16×8 (16 tall × 8 wide; `num_blocks * 128` floats).
    pub fn dct_16x8_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 128, |c, h_in, h_out, n| {
            dct_16x8::<R>(c, h_in, h_out, n)
        })
    }

    /// Inverse DCT16×8.
    pub fn idct_16x8_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 128, |c, h_in, h_out, n| {
            idct_16x8::<R>(c, h_in, h_out, n)
        })
    }

    /// Forward DCT8×16 (8 tall × 16 wide; `num_blocks * 128` floats).
    pub fn dct_8x16_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 128, |c, h_in, h_out, n| {
            dct_8x16::<R>(c, h_in, h_out, n)
        })
    }

    /// Inverse DCT8×16.
    pub fn idct_8x16_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 128, |c, h_in, h_out, n| {
            idct_8x16::<R>(c, h_in, h_out, n)
        })
    }

    /// Forward DCT32×16 (`num_blocks * 512` floats).
    pub fn dct_32x16_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 512, |c, h_in, h_out, n| {
            dct_32x16::<R>(c, h_in, h_out, n)
        })
    }

    /// Inverse DCT32×16.
    pub fn idct_32x16_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 512, |c, h_in, h_out, n| {
            idct_32x16::<R>(c, h_in, h_out, n)
        })
    }

    /// Forward DCT16×32 (`num_blocks * 512` floats).
    pub fn dct_16x32_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 512, |c, h_in, h_out, n| {
            dct_16x32::<R>(c, h_in, h_out, n)
        })
    }

    /// Inverse DCT16×32.
    pub fn idct_16x32_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 512, |c, h_in, h_out, n| {
            idct_16x32::<R>(c, h_in, h_out, n)
        })
    }

    /// Forward DCT64×32 (`num_blocks * 2048` floats).
    pub fn dct_64x32_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 2048, |c, h_in, h_out, n| {
            dct_64x32::<R>(c, h_in, h_out, n)
        })
    }

    /// Inverse DCT64×32.
    pub fn idct_64x32_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 2048, |c, h_in, h_out, n| {
            idct_64x32::<R>(c, h_in, h_out, n)
        })
    }

    /// Forward DCT32×64 (`num_blocks * 2048` floats).
    pub fn dct_32x64_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 2048, |c, h_in, h_out, n| {
            dct_32x64::<R>(c, h_in, h_out, n)
        })
    }

    /// Inverse DCT32×64.
    pub fn idct_32x64_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 2048, |c, h_in, h_out, n| {
            idct_32x64::<R>(c, h_in, h_out, n)
        })
    }

    /// Forward DCT4×4 full (sub-block-partitioned 8×8; `num_blocks * 64`).
    pub fn dct_4x4_full_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 64, |c, h_in, h_out, n| {
            dct_4x4_full::<R>(c, h_in, h_out, n)
        })
    }

    /// Inverse DCT4×4 full.
    pub fn idct_4x4_full_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 64, |c, h_in, h_out, n| {
            idct_4x4_full::<R>(c, h_in, h_out, n)
        })
    }

    /// Forward DCT4×8 full.
    pub fn dct_4x8_full_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 64, |c, h_in, h_out, n| {
            dct_4x8_full::<R>(c, h_in, h_out, n)
        })
    }

    /// Inverse DCT4×8 full.
    pub fn idct_4x8_full_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 64, |c, h_in, h_out, n| {
            idct_4x8_full::<R>(c, h_in, h_out, n)
        })
    }

    /// Forward DCT8×4 full.
    pub fn dct_8x4_full_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 64, |c, h_in, h_out, n| {
            dct_8x4_full::<R>(c, h_in, h_out, n)
        })
    }

    /// Inverse DCT8×4 full.
    pub fn idct_8x4_full_blocks(&self, blocks: &[f32]) -> Vec<f32> {
        run_block_inout::<R, _>(&self.client, blocks, 64, |c, h_in, h_out, n| {
            idct_8x4_full::<R>(c, h_in, h_out, n)
        })
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

/// Helper for the common "f32 in, same-size f32 out, num_blocks-driven
/// launch" pattern used by DCT/IDCT methods.
fn run_block_inout<R, F>(
    client: &ComputeClient<R>,
    input: &[f32],
    block_size: usize,
    launcher: F,
) -> Vec<f32>
where
    R: Runtime,
    F: FnOnce(&ComputeClient<R>, cubecl::server::Handle, cubecl::server::Handle, u32),
{
    let n = input.len();
    assert!(
        n.is_multiple_of(block_size),
        "input len {n} not a multiple of block size {block_size}"
    );
    let num_blocks = (n / block_size) as u32;
    let h_in = client.create_from_slice(f32::as_bytes(input));
    let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    launcher(client, h_in, h_out.clone(), num_blocks);
    let bytes = client.read_one(h_out).expect("read");
    f32::from_bytes(&bytes).to_vec()
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    type B = cubecl::cuda::CudaRuntime;

    fn deterministic_blocks(n_blocks: usize) -> alloc::vec::Vec<f32> {
        let mut out = alloc::vec![0.0f32; n_blocks * 64];
        for b in 0..n_blocks {
            for i in 0..64 {
                let v = ((b * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
                out[b * 64 + i] = 0.3 + 0.4 * v;
            }
        }
        out
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn test_identity_blocks_roundtrip() {
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let pixels = deterministic_blocks(8);
        let coeffs = enc.identity_blocks(&pixels);
        let recon = enc.inverse_identity_blocks(&coeffs);
        assert_eq!(coeffs.len(), pixels.len());
        assert_eq!(recon.len(), pixels.len());
        let m = max_abs_diff(&pixels, &recon);
        assert!(m < 1e-5, "IDENTITY roundtrip max|Δ| = {m:.3e} (>1e-5)");
    }

    #[test]
    fn test_dct2x2_blocks_roundtrip() {
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let pixels = deterministic_blocks(8);
        let coeffs = enc.dct2x2_blocks(&pixels);
        let recon = enc.inverse_dct2x2_blocks(&coeffs);
        assert_eq!(coeffs.len(), pixels.len());
        assert_eq!(recon.len(), pixels.len());
        let m = max_abs_diff(&pixels, &recon);
        assert!(m < 1e-5, "DCT2X2 roundtrip max|Δ| = {m:.3e} (>1e-5)");
    }

    #[test]
    fn test_identity_vs_dct2x2_distinct() {
        // Sanity: the two transforms should produce different outputs
        // on the same input — they're different ops. (The DC term
        // post-Hadamard is the same for both, but AC differs.)
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let pixels = deterministic_blocks(2);
        let id_coeffs = enc.identity_blocks(&pixels);
        let dct_coeffs = enc.dct2x2_blocks(&pixels);
        let m = max_abs_diff(&id_coeffs, &dct_coeffs);
        assert!(m > 1e-3, "IDENTITY and DCT2X2 should produce different coeffs (max|Δ|={m:.3e})");
    }
}
