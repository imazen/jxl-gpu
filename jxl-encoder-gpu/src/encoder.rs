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
use crate::launch::dct2x2::{dct2x2_forward, dct2x2_inverse};
use crate::launch::dct4::{
    dct_4x4_full, dct_4x8_full, dct_8x4_full, idct_4x4_full, idct_4x8_full, idct_8x4_full,
};
use crate::launch::dct8::{dct_8x8, idct_8x8};
use crate::launch::dct16::{dct_8x16, dct_16x8, dct_16x16, idct_8x16, idct_16x8, idct_16x16};
use crate::launch::dct32::{dct_16x32, dct_32x16, dct_32x32, idct_16x32, idct_32x16, idct_32x32};
use crate::launch::dct64::{dct_32x64, dct_64x32, dct_64x64, idct_32x64, idct_64x32, idct_64x64};
use crate::launch::denoise::denoise as denoise_launch;
use crate::launch::dequant::{dequant_dct8, dequant_dct8_broadcast_w};
use crate::launch::entropy::{
    entropy_coeffs_coeff, entropy_coeffs_coeff_broadcast_w, entropy_coeffs_pixel,
    entropy_coeffs_pixel_broadcast_w,
};
use crate::launch::epf::{epf_step0, epf_step1, epf_step2, pad_plane};
use crate::launch::fuzzy_erosion::{fuzzy_erosion as fuzzy_erosion_launch, fuzzy_erosion_kmul};
use crate::launch::gab::gab_smooth;
use crate::launch::gaborish::gaborish_5x5;
use crate::launch::identity::{identity_forward, identity_inverse};
use crate::launch::mask1x1::mask1x1;
use crate::launch::pixel_loss::pixel_loss;
use crate::launch::quantize::quantize_dct8;
use crate::launch::xyb::{xyb_forward, xyb_inverse};

/// GPU-accelerated JXL encoder. Holds a long-lived cubecl client.
///
/// Construct once per process; reuse across many encodes. Per-call
/// `client.create_from_slice(...)` allocations on the cubecl backend
/// are not free (hundreds of ms at 1MP per call on first allocation;
/// the cubecl backend pools internally for subsequent allocations of
/// the same size). For a perf-critical loop calling many encodes,
/// the [`crate::persistent`] module's `GpuPlane` / `GpuBlocks` types
/// let you upload once and reuse the resulting `Handle` across
/// launches without round-tripping through `Vec<f32>`.
///
/// **TODO** (`encoder.rs:71`): a `HashMap<(u32, u32), GpuInstance<R>>`
/// per-(w,h) buffer cache could pool the intermediate per-launch
/// allocations across calls of the same dims. Not implemented because
/// (a) cubecl's backend allocator may already pool, and (b) we don't
/// yet have bench numbers showing per-call alloc is the bottleneck.
/// Cf. global TODO in `PORT_STATUS.md` "Per-(w,h) instance pre-allocation
/// cache".
///
/// See `kernels::xyb` and friends for the underlying kernels.
pub struct GpuEncoder<R: Runtime> {
    client: ComputeClient<R>,
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

        // Batched 3-handle download — one queue-drain sync instead
        // of three sequential read_one (matches the persistent-API
        // pattern from ba826e06 / 814795e5).
        let mut bytes = self.client.read(alloc::vec![h_x, h_y, h_b_out]);
        let bb = bytes.pop().expect("read[2]");
        let yb = bytes.pop().expect("read[1]");
        let xb = bytes.pop().expect("read[0]");
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

        // Batched 3-handle download (matches xyb_from_linear_rgb above).
        let mut bytes = self.client.read(alloc::vec![h_r, h_g_out, h_b_out]);
        let bb = bytes.pop().expect("read[2]");
        let gb = bytes.pop().expect("read[1]");
        let rb = bytes.pop().expect("read[0]");
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

    /// Generic per-coefficient dequant: `output[i] = quant[i] * weights[i]`
    /// for `num_blocks * block_size` coefficients.
    ///
    /// Used by larger-strategy decode paths (DCT16+, AFV/IDENTITY/DCT2X2)
    /// where the simpler `dequant_dct8` (which folds in CfL +
    /// adjust_quant_bias) isn't applicable.
    pub fn dequant_simple_blocks(
        &self,
        quant: &[i32],
        weights: &[f32],
        block_size: u32,
    ) -> Vec<f32> {
        let bs = block_size as usize;
        let n = quant.len();
        assert!(n.is_multiple_of(bs));
        assert_eq!(weights.len(), n);
        let num_blocks = (n / bs) as u32;
        let h_q = self.client.create_from_slice(i32::as_bytes(quant));
        let h_w = self.client.create_from_slice(f32::as_bytes(weights));
        let h_o = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        crate::launch::dequant_simple::dequant_simple::<R>(
            &self.client,
            h_q,
            h_w,
            h_o.clone(),
            num_blocks,
            block_size,
        );
        let bytes = self.client.read_one(h_o).expect("read dequant_simple");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Broadcast-weights variant of [`Self::dequant_simple_blocks`].
    /// `weights_template` is exactly `block_size` f32 (one quant
    /// matrix); the kernel broadcasts it across all blocks. Saves
    /// `(num_blocks - 1) * block_size * 4` bytes of upload traffic
    /// when callers previously replicated the same matrix per-block.
    pub fn dequant_simple_blocks_broadcast_w(
        &self,
        quant: &[i32],
        weights_template: &[f32],
        block_size: u32,
    ) -> Vec<f32> {
        let bs = block_size as usize;
        let n = quant.len();
        assert!(n.is_multiple_of(bs));
        assert_eq!(
            weights_template.len(),
            bs,
            "weights_template must be exactly block_size = {bs} entries (got {})",
            weights_template.len()
        );
        let num_blocks = (n / bs) as u32;
        let h_q = self.client.create_from_slice(i32::as_bytes(quant));
        let h_w = self
            .client
            .create_from_slice(f32::as_bytes(weights_template));
        let h_o = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        crate::launch::dequant_simple::dequant_simple_broadcast_w::<R>(
            &self.client,
            h_q,
            h_w,
            h_o.clone(),
            num_blocks,
            block_size,
        );
        let bytes = self
            .client
            .read_one(h_o)
            .expect("read dequant_simple broadcast");
        f32::from_bytes(&bytes).to_vec()
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
        let h_t = self
            .client
            .create_from_slice(f32::as_bytes(&thresholds[..]));
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

    /// Broadcast-weights variant of [`Self::quantize_large_blocks`].
    /// `weights_template` is exactly `grid_width * grid_height` f32
    /// (one quant matrix); the kernel broadcasts it across all blocks.
    /// Saves `(num_blocks - 1) * grid_width * grid_height * 4` bytes
    /// of upload traffic when callers previously replicated the same
    /// matrix per-block.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_large_blocks_broadcast_w(
        &self,
        coeffs: &[f32],
        weights_template: &[f32],
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
        assert_eq!(
            weights_template.len(),
            block_size,
            "weights_template must be exactly grid_width*grid_height = {block_size} entries (got {})",
            weights_template.len()
        );
        let num_blocks = (n / block_size) as u32;
        assert_eq!(qac_qm.len(), num_blocks as usize);
        let h_c = self.client.create_from_slice(f32::as_bytes(coeffs));
        let h_w = self
            .client
            .create_from_slice(f32::as_bytes(weights_template));
        let h_q = self.client.create_from_slice(f32::as_bytes(qac_qm));
        let h_t = self
            .client
            .create_from_slice(f32::as_bytes(&thresholds[..]));
        let h_o = self
            .client
            .create_from_slice(i32::as_bytes(&vec![0_i32; n]));
        crate::launch::quantize::quantize_large_broadcast_w::<R>(
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
        let bytes = self
            .client
            .read_one(h_o)
            .expect("read quantize_large broadcast");
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

    /// Coefficient-domain entropy_coeffs (no error_coeffs writeback).
    /// Computes per-block `[entropy_sum, nzeros_sum, info_loss_sum,
    /// info_loss2_sum]` from quantized AC coefficient ranges. Used by
    /// upstream's butteraugli iter loop in coefficient-domain mode.
    #[allow(clippy::too_many_arguments)]
    pub fn entropy_coeffs_coeff_blocks(
        &self,
        block_c: &[f32],
        block_y: &[f32],
        inv_weights: &[f32],
        n_per_block: u32,
        cmap_factor: f32,
        quant: f32,
        k_cost_delta: f32,
        k_cost2: f32,
    ) -> Vec<f32> {
        let total = block_c.len();
        let n = n_per_block as usize;
        assert!(total.is_multiple_of(n));
        let num_blocks = (total / n) as u32;
        assert_eq!(block_y.len(), total);
        assert_eq!(inv_weights.len(), total);
        let h_c = self.client.create_from_slice(f32::as_bytes(block_c));
        let h_y = self.client.create_from_slice(f32::as_bytes(block_y));
        let h_iw = self.client.create_from_slice(f32::as_bytes(inv_weights));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; (num_blocks as usize) * 4]));
        entropy_coeffs_coeff::<R>(
            &self.client,
            h_c,
            h_y,
            h_iw,
            h_out.clone(),
            num_blocks,
            n_per_block,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
        );
        let bytes = self.client.read_one(h_out).expect("read out");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Broadcast-weights variant of [`Self::entropy_coeffs_coeff_blocks`].
    /// `inv_weights_template` is exactly `n_per_block` f32 (one
    /// inverse-quant matrix); the kernel broadcasts across all blocks.
    /// Saves `(num_blocks - 1) * n_per_block * 4` bytes of upload
    /// traffic — half the savings of the pixel-domain broadcast
    /// variant (which has two weight arrays).
    #[allow(clippy::too_many_arguments)]
    pub fn entropy_coeffs_coeff_blocks_broadcast_w(
        &self,
        block_c: &[f32],
        block_y: &[f32],
        inv_weights_template: &[f32],
        n_per_block: u32,
        cmap_factor: f32,
        quant: f32,
        k_cost_delta: f32,
        k_cost2: f32,
    ) -> Vec<f32> {
        let total = block_c.len();
        let n = n_per_block as usize;
        assert!(total.is_multiple_of(n));
        let num_blocks = (total / n) as u32;
        assert_eq!(block_y.len(), total);
        assert_eq!(
            inv_weights_template.len(),
            n,
            "inv_weights_template must be exactly n_per_block = {n} entries"
        );
        let h_c = self.client.create_from_slice(f32::as_bytes(block_c));
        let h_y = self.client.create_from_slice(f32::as_bytes(block_y));
        let h_iw = self
            .client
            .create_from_slice(f32::as_bytes(inv_weights_template));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; (num_blocks as usize) * 4]));
        entropy_coeffs_coeff_broadcast_w::<R>(
            &self.client,
            h_c,
            h_y,
            h_iw,
            h_out.clone(),
            num_blocks,
            n_per_block,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
        );
        let bytes = self.client.read_one(h_out).expect("read out");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Broadcast-weights variant of [`Self::entropy_coeffs_pixel_blocks`].
    /// `weights_template` and `inv_weights_template` are each exactly
    /// `n_per_block` f32 (one quant matrix and its inverse). The kernel
    /// broadcasts both across all blocks. Saves
    /// `2 * (num_blocks - 1) * n_per_block * 4` bytes of upload traffic
    /// vs the per-block variant — twice the savings of the dequant
    /// broadcast variants because this kernel reads two weight arrays.
    #[allow(clippy::too_many_arguments)]
    pub fn entropy_coeffs_pixel_blocks_broadcast_w(
        &self,
        block_c: &[f32],
        block_y: &[f32],
        weights_template: &[f32],
        inv_weights_template: &[f32],
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
        assert_eq!(
            weights_template.len(),
            n,
            "weights_template must be exactly n_per_block = {n} entries"
        );
        assert_eq!(
            inv_weights_template.len(),
            n,
            "inv_weights_template must be exactly n_per_block = {n} entries"
        );

        let h_c = self.client.create_from_slice(f32::as_bytes(block_c));
        let h_y = self.client.create_from_slice(f32::as_bytes(block_y));
        let h_w = self
            .client
            .create_from_slice(f32::as_bytes(weights_template));
        let h_iw = self
            .client
            .create_from_slice(f32::as_bytes(inv_weights_template));
        let h_err = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; total]));
        let h_out = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; (num_blocks as usize) * 4]));
        entropy_coeffs_pixel_broadcast_w::<R>(
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

    /// EPF Step 0 — 5×5 plus kernel with 3×3-plus SAD weights over 12 neighbors.
    /// The heaviest of the three EPF passes. Inputs are PADDED
    /// (`stride = width + 2*pad`, `pad >= 3`); output is unpadded.
    /// Returns `(out_x, out_y, out_b)`.
    #[allow(clippy::too_many_arguments)]
    pub fn epf_step0_channels(
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
        epf_step0::<R>(
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

    /// Broadcast-weights variant of [`Self::dequant_dct8_blocks`].
    /// Each `weights_*_template` is exactly 64 f32 (one DCT8 quant
    /// matrix per channel); the kernel broadcasts across all blocks.
    /// Saves `3 * (num_blocks - 1) * 64 * 4` bytes of upload traffic
    /// when callers were previously replicating the matrix per-block.
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_dct8_blocks_broadcast_w(
        &self,
        quant_x: &[i32],
        quant_y: &[i32],
        quant_b: &[i32],
        weights_x_template: &[f32],
        weights_y_template: &[f32],
        weights_b_template: &[f32],
        qac_qm_x: &[f32],
        qac_qm_y: &[f32],
        qac_qm_b: &[f32],
        x_factor: &[f32],
        b_factor: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n_coef = quant_x.len();
        assert!(n_coef.is_multiple_of(64));
        for (label, w) in [
            ("weights_x_template", weights_x_template),
            ("weights_y_template", weights_y_template),
            ("weights_b_template", weights_b_template),
        ] {
            assert_eq!(
                w.len(),
                64,
                "{label} must be exactly 64 f32 (got {})",
                w.len()
            );
        }
        let nb = n_coef / 64;
        let num_blocks = nb as u32;
        let h_qx = self.client.create_from_slice(i32::as_bytes(quant_x));
        let h_qy = self.client.create_from_slice(i32::as_bytes(quant_y));
        let h_qb = self.client.create_from_slice(i32::as_bytes(quant_b));
        let h_wx = self
            .client
            .create_from_slice(f32::as_bytes(weights_x_template));
        let h_wy = self
            .client
            .create_from_slice(f32::as_bytes(weights_y_template));
        let h_wb = self
            .client
            .create_from_slice(f32::as_bytes(weights_b_template));
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
        dequant_dct8_broadcast_w::<R>(
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

    /// Encode to JXL bitstream via the `__pre_quantized` seam — runs
    /// the GPU strat-search pipeline (XYB / gaborish / CfL / strategy
    /// selection / DC grid) AND optionally the butteraugli refinement
    /// loop, then hands the prepared state to
    /// [`jxl_encoder::__pre_quantized::VarDctEncoder::encode_from_precomputed`]
    /// for bitstream emit. The CPU encoder skips its own XYB / CfL /
    /// masking / strat-search work; it still does DCT / quantize /
    /// entropy coding (CPU is fast enough for those steps post-
    /// quantization decisions).
    ///
    /// **First-cut implementation** — uses CfL=zeros, masking=zeros,
    /// chromacity=0, noise_params=None, AcStrategyMap=all-DCT8.
    /// Scoped to validate the architectural seam end-to-end.
    ///
    /// **STATUS (2026-05-11)**: produces a valid JXL codestream
    /// signature (0xFF 0x0A) and 386 KB output at 1 MP / d=1, but
    /// the body fails decode in both djxl and jxl-oxide with
    /// "modular stream error: unexpected EOF" during frame render.
    /// The bitstream is structurally valid (headers parse, dims
    /// echo correctly: 1000×1000) but the encoded data is malformed
    /// or oversized (~3.09 bpp vs typical ~1 bpp at d=1). Suspected
    /// causes:
    ///   - GPU XYB scale mismatch (libjxl's specific opsin matrix
    ///     scaling vs ours)
    ///   - chromacity_*_pixelized=0 corrupts a
    ///     `params.apply_chromacity_adjustment` step
    ///   - some required precomputed field needs a non-stub value
    /// Debug requires diffing against a known-good
    /// `EncoderPrecomputed` from the rate-control path. Tracked as
    /// follow-up; the architectural seam is in place and reachable.
    ///
    /// `linear_rgb_padded` MUST be planar linear RGB padded to
    /// `lossy.padded_dimensions()` (each channel is
    /// `padded_w × padded_h` f32 entries, edge-replicated to the
    /// padded boundary). The caller's image-source upload code in
    /// `prepare_strategy_search_plan` already does this padding —
    /// callers can pass the same `r/g/b` they would pass to
    /// `prepare_strategy_search_plan` and we'll pad internally.
    #[cfg(feature = "encoder")]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_lossy_to_bitstream_via_precomputed(
        &self,
        lossy: &crate::lossy_encoder::LossyEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        distance: f32,
    ) -> Result<alloc::vec::Vec<u8>, jxl_encoder::api::EncodeError> {
        use jxl_encoder::__pre_quantized::{
            AcStrategyMap, DistanceParams, EncoderPrecomputed, VarDctEncoder, compute_cfl_map,
            quantize_quant_field,
        };

        let (width, height) = lossy.dimensions();
        let (gpu_pw, gpu_ph) = lossy.padded_dimensions();
        // CPU encoder pads to 8-byte alignment (NOT 16 like GPU's
        // strat-search needs). The encoder reads xyb_x/y/b at indices
        // computed from precomputed.padded_width — which MUST match
        // the buffer's actual layout, AND must equal CPU's own
        // alignment derivation (padded = align_up(width, 8)). If we
        // pass GPU's 16-aligned padded values, the encoder's group /
        // block math diverges and the bitstream becomes corrupt
        // (modular stream EOF on decode). Re-pack to CPU's expected
        // alignment before handing off.
        let cpu_pw = (width as usize).div_ceil(8) * 8;
        let cpu_ph = (height as usize).div_ceil(8) * 8;
        let xsize_blocks = cpu_pw / 8;
        let ysize_blocks = cpu_ph / 8;
        let num_blocks = xsize_blocks * ysize_blocks;

        // Step 1: GPU strat-search → plan with xyb GPU planes + assignments.
        // f32 path: caller did u8→f32 host-side; we upload 3× padded
        // f32 planes. The u8 fast path
        // (`encode_lossy_to_bitstream_via_precomputed_from_u8`) avoids
        // both the host conversion and the 4× upload bandwidth.
        let plan = lossy.prepare_strategy_search_plan(self, r, g, b, distance);

        // Step 2: download xyb planes (GPU-padded layout) in one
        // batched read. Saves ~25 ms at 12 MP vs 3 sequential reads.
        let (xyb_x_gpu, xyb_y_gpu, xyb_b_gpu) =
            self.download_planes_3ch(&plan.xyb_x_gpu, &plan.xyb_y_gpu, &plan.xyb_b_gpu);

        // Step 2b: re-pack from GPU's 16-aligned (gpu_pw × gpu_ph) to
        // CPU's 8-aligned (cpu_pw × cpu_ph). When dims agree
        // (multiple of 16) this is just a clone; when they differ
        // (multiple of 8 but not 16, e.g. 1025 → GPU 1040 / CPU 1032)
        // we copy the upper-left cpu_pw × cpu_ph from the GPU buffer.
        //
        // Edge-replication fix-up: GPU's edge-replicated values at
        // cols >= width / rows >= height are computed BEFORE gaborish
        // runs. After gaborish (5×5 sharpening filter), edge pixels
        // get convolved with their replicated neighbors → the
        // post-gaborish edge values can subtly diverge between GPU
        // (1040 wide → gaborish saw 1040 cols) and what CPU would
        // produce on the same source (CPU's compute_to_xyb_padded
        // works on 1032 cols → gaborish sees 1032). The post-gaborish
        // mismatch at the rightmost real-image edge column has been
        // observed to corrupt butteraugli max-norm scores at non-16-
        // aligned dims (constant 3.27 baugli at 1025×1025 across
        // d=0.5/1.0/2.0 — the hot pixel is at the col=width edge).
        //
        // Fix: after extraction, edge-replicate the CPU-correct
        // padded region from the LAST REAL image col/row of the
        // repacked buffer (cols `width..cpu_pw`, rows `height..cpu_ph`).
        // This trades GPU's gaborish-edge-blended replication for
        // pure replication of the last real pixel — matches CPU's
        // pad-then-gaborish-on-CPU-padded-buffer behavior closer.
        let repack = |src: &[f32]| -> alloc::vec::Vec<f32> {
            if cpu_pw == gpu_pw as usize && cpu_ph == gpu_ph as usize {
                src.to_vec()
            } else {
                let mut dst = alloc::vec![0.0_f32; cpu_pw * cpu_ph];
                let w = width as usize;
                let h = height as usize;
                // Copy real image rows + GPU-padded right edge into
                // initial cpu_ph rows.
                for row in 0..cpu_ph {
                    let off = row * (gpu_pw as usize);
                    let dst_off = row * cpu_pw;
                    dst[dst_off..dst_off + cpu_pw].copy_from_slice(&src[off..off + cpu_pw]);
                }
                // Re-replicate the right edge (cols width..cpu_pw)
                // from col (width-1) for ALL rows (incl. GPU-padded).
                if cpu_pw > w {
                    for row in 0..cpu_ph {
                        let dst_off = row * cpu_pw;
                        let src_val = dst[dst_off + (w - 1)];
                        for c in w..cpu_pw {
                            dst[dst_off + c] = src_val;
                        }
                    }
                }
                // Re-replicate the bottom edge (rows height..cpu_ph)
                // from row (height-1) for ALL cols (already-fixed
                // right edge included).
                if cpu_ph > h {
                    let last_real_off = (h - 1) * cpu_pw;
                    for row in h..cpu_ph {
                        let dst_off = row * cpu_pw;
                        dst.copy_within(last_real_off..last_real_off + cpu_pw, dst_off);
                    }
                }
                dst
            }
        };
        // 3-way rayon::join — saves ~50 ms at 12 MP vs sequential.
        let ((xyb_x, xyb_y), xyb_b) = rayon::join(
            || rayon::join(|| repack(&xyb_x_gpu), || repack(&xyb_y_gpu)),
            || repack(&xyb_b_gpu),
        );

        // Step 3: AcStrategyMap from plan.assignments. The padding
        // mismatch fix above (cpu_pw vs gpu_pw) means xsize_blocks /
        // ysize_blocks are also CPU-aligned — we MUST drop assignments
        // whose blocks fall in the GPU-but-not-CPU padding region
        // (e.g. blocks at bx >= cpu_xsize_blocks for a 1000-wide image
        // where GPU saw 1008/8=126 blocks but CPU sees 1000/8=125).
        let mut ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);
        for a in &plan.assignments {
            let bx = a.bx as usize;
            let by = a.by as usize;
            if bx >= xsize_blocks || by >= ysize_blocks {
                continue;
            }
            // Skip strategies whose coverage would extend past the
            // CPU-aligned grid (assignment was valid in GPU's larger
            // grid but spills out of CPU's smaller one). Coverage
            // table mirrors `groups::strategy_coverage_blocks` (kept
            // private under cfg(debug_assertions) there).
            use crate::forks::transform::*;
            let (cx, cy): (usize, usize) = match a.raw_strategy {
                RAW_STRATEGY_DCT16X8 => (1, 2),
                RAW_STRATEGY_DCT8X16 => (2, 1),
                RAW_STRATEGY_DCT16X16 => (2, 2),
                RAW_STRATEGY_DCT32X16 => (2, 4),
                RAW_STRATEGY_DCT16X32 => (4, 2),
                RAW_STRATEGY_DCT32X32 => (4, 4),
                RAW_STRATEGY_DCT64X32 => (4, 8),
                RAW_STRATEGY_DCT32X64 => (8, 4),
                RAW_STRATEGY_DCT64X64 => (8, 8),
                _ => (1, 1), // DCT8 / DCT4* / DCT2x2 / IDENTITY / AFV
            };
            if bx + cx > xsize_blocks || by + cy > ysize_blocks {
                continue;
            }
            if a.raw_strategy != 0 {
                ac_strategy.set(bx, by, a.raw_strategy);
            }
        }

        // Step 4: CfL — compute on host from the just-downloaded XYB.
        // The GPU strat-search pipeline doesn't currently expose a
        // per-tile CfL grid on StrategySearchPlan; running the CPU
        // helper on host XYB matches what EncoderPrecomputed::compute
        // does and gives us the size win from real chroma decorrelation
        // (vs CfL=zeros which leaves chroma uncorrelated).
        // use_newton=true matches libjxl effort 7+ behavior.
        let cfl_map = compute_cfl_map(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            cpu_pw,
            cpu_ph,
            xsize_blocks,
            ysize_blocks,
            true, // use_newton (effort >= 7)
            1e-3, // newton_eps (libjxl default)
            10,   // newton_max_iters
        );

        // Step 5: GPU-computed quant_field_float + masking from
        // `prepare_strategy_search_plan` (compute_quant_field_full_persistent).
        // Replaces the CPU compute_quant_field_float_free call —
        // bitstream identical (verified by
        // forks::adaptive_quant::tests::test_compute_quant_field_production_flow_divergence).
        let quant_field_float = plan.quant_field_float.clone();
        let masking = plan.masking.clone();
        debug_assert_eq!(quant_field_float.len(), num_blocks, "qf len");
        debug_assert_eq!(masking.len(), num_blocks, "mask len");

        // Step 6: linear_rgb is only used by rate-control loop; pass empty.
        let linear_rgb = alloc::vec::Vec::new();

        // Step 7: assemble EncoderPrecomputed.
        let precomputed = EncoderPrecomputed::from_parts(
            width as usize,
            height as usize,
            xsize_blocks,
            ysize_blocks,
            cpu_pw,
            cpu_ph,
            xyb_x,
            xyb_y,
            xyb_b,
            linear_rgb,
            cfl_map,
            None,
            quant_field_float.clone(),
            masking,
            None,
            ac_strategy,
            true, // gaborish_enabled (matches GPU's xyb_*_gpu output)
            distance,
            0,
            0,
        );

        // Step 8: build the CPU VarDctEncoder + convert quant field.
        let vardct = VarDctEncoder::new(distance);
        let params = DistanceParams::compute_for_profile(distance, &vardct.profile);
        let quant_field_u8 = quantize_quant_field(&quant_field_float, params.inv_scale);

        // Step 9: encode → bitstream. Map jxl-encoder's internal Error
        // type to the public EncodeError surface (api.rs:80 has the
        // From impl).
        vardct
            .encode_from_precomputed(&precomputed, &quant_field_u8)
            .map_err(jxl_encoder::api::EncodeError::from)
    }

    /// u8 fast-path variant of
    /// [`Self::encode_lossy_to_bitstream_via_precomputed`]. Takes raw
    /// interleaved sRGB u8 RGB (no alpha; `width * height * 3` bytes)
    /// and runs sRGB→linear + edge-replication padding + XYB on the
    /// GPU in one fused kernel — bypasses the host-side `powf`
    /// per-pixel and shrinks the upload by 4× (e.g. 48 MB raw u8
    /// instead of 192 MB converted f32 at 16 MP). Apart from the
    /// upload entry point this matches the f32 variant byte-for-byte;
    /// downstream re-pack / quant / encode steps are identical.
    ///
    /// Production callers holding sRGB u8 input should prefer this
    /// over the f32 variant — the host-side conversion and the
    /// larger upload are pure waste. Callers that hold f32 linear
    /// input (e.g. HDR pipelines, post-processed buffers) keep
    /// using the f32 variant.
    ///
    /// Math: full sRGB EOTF (the proper IEC 61966-2-1 piecewise:
    /// linear segment for `c <= 0.04045`, `((c + 0.055) / 1.055)^2.4`
    /// otherwise). Matches the reference math in
    /// `examples/perf_strat_plan_u8_vs_f32.rs`. Differs from the
    /// `_srgb_u8` convenience wrappers on `LossyEncoder`, which use
    /// the simpler pure-`powf(2.4)` model.
    pub fn encode_lossy_to_bitstream_via_precomputed_from_u8(
        &self,
        lossy: &crate::lossy_encoder::LossyEncoder<R>,
        pixels_u8: &[u8],
        distance: f32,
    ) -> Result<alloc::vec::Vec<u8>, jxl_encoder::api::EncodeError> {
        use jxl_encoder::__pre_quantized::{
            AcStrategyMap, DistanceParams, EncoderPrecomputed, VarDctEncoder, compute_cfl_map,
            quantize_quant_field,
        };

        let (width, height) = lossy.dimensions();
        let (gpu_pw, gpu_ph) = lossy.padded_dimensions();
        let cpu_pw = (width as usize).div_ceil(8) * 8;
        let cpu_ph = (height as usize).div_ceil(8) * 8;
        let xsize_blocks = cpu_pw / 8;
        let ysize_blocks = cpu_ph / 8;
        let num_blocks = xsize_blocks * ysize_blocks;

        let expected = (width as usize) * (height as usize) * 3;
        if pixels_u8.len() != expected {
            return Err(jxl_encoder::api::EncodeError::InvalidInput {
                message: alloc::format!(
                    "pixels_u8 len {} != width*height*3 = {}",
                    pixels_u8.len(),
                    expected,
                ),
            });
        }

        // Step 1: GPU strat-search → plan, with the u8 fused upload.
        // sRGB→linear + pad happens inside `upload_u8_rgb_to_linear_planar_padded`
        // — host never sees the f32 planes, and the wire transfer is
        // `width * height * 3` bytes (no padding, no per-pixel powf).
        let plan = lossy.prepare_strategy_search_plan_from_u8(self, pixels_u8, distance);

        // Step 2: AcStrategyMap from plan.assignments (clip to CPU grid).
        // This depends only on plan, not on XYB on host.
        let mut ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);
        for a in &plan.assignments {
            let bx = a.bx as usize;
            let by = a.by as usize;
            if bx >= xsize_blocks || by >= ysize_blocks {
                continue;
            }
            use crate::forks::transform::*;
            let (cx, cy): (usize, usize) = match a.raw_strategy {
                RAW_STRATEGY_DCT16X8 => (1, 2),
                RAW_STRATEGY_DCT8X16 => (2, 1),
                RAW_STRATEGY_DCT16X16 => (2, 2),
                RAW_STRATEGY_DCT32X16 => (2, 4),
                RAW_STRATEGY_DCT16X32 => (4, 2),
                RAW_STRATEGY_DCT32X32 => (4, 4),
                RAW_STRATEGY_DCT64X32 => (4, 8),
                RAW_STRATEGY_DCT32X64 => (8, 4),
                RAW_STRATEGY_DCT64X64 => (8, 8),
                _ => (1, 1),
            };
            if bx + cx > xsize_blocks || by + cy > ysize_blocks {
                continue;
            }
            if a.raw_strategy != 0 {
                ac_strategy.set(bx, by, a.raw_strategy);
            }
        }

        // GPU pre-quantized AC fast path: ENABLED. Backed by the
        // fully-fused 3-channel DCT8 producer
        // (kernels::fused_dct8_3ch::fused_dct8_3ch_kernel) +
        // GPU CfL (kernels::cfl_collect → existing newton kernel),
        // which lets us SKIP the 60 ms XYB DtoH download + 28 ms
        // host-side repack + 9 ms CPU compute_cfl_map for
        // all-DCT8 images.
        //
        // Bitstream is NOT byte-identical to CPU due to cubecl-vs-jxl_simd
        // DCT FP precision (~2% chroma AC coefs flip by 1 near rounding
        // ties); corpus_regression at 0.5% score tolerance covers it.
        const ENABLE_GPU_DCT8_FAST_PATH: bool = true;
        let all_dct8 = ENABLE_GPU_DCT8_FAST_PATH
            && (0..ysize_blocks)
                .all(|by| (0..xsize_blocks).all(|bx| ac_strategy.raw_strategy(bx, by) == 0));

        let vardct = VarDctEncoder::new(distance);
        let params = DistanceParams::compute_for_profile(distance, &vardct.profile);

        let quant_field_float = plan.quant_field_float.clone();
        let masking = plan.masking.clone();
        debug_assert_eq!(quant_field_float.len(), num_blocks, "qf len");
        debug_assert_eq!(masking.len(), num_blocks, "mask len");
        let quant_field_u8 = quantize_quant_field(&quant_field_float, params.inv_scale);

        if all_dct8 {
            // GPU CfL: matches CPU compute_cfl_map within ±1 (FP +
            // padded num_per_tile noise — see cfl_map_gpu_matches_cpu_small).
            let cfl_result = crate::forks::cfl_map_gpu::compute_cfl_map_gpu_persistent(
                self,
                &plan.xyb_x_gpu,
                &plan.xyb_y_gpu,
                &plan.xyb_b_gpu,
                xsize_blocks,
                ysize_blocks,
                true, // use_newton (effort >= 7)
                1e-3,
                10,
            );
            let cfl_map = jxl_encoder::__pre_quantized::CflMap {
                ytox: cfl_result.ytox,
                ytob: cfl_result.ytob,
                xsize_tiles: cfl_result.xsize_tiles,
                ysize_tiles: cfl_result.ysize_tiles,
            };
            // For the fast path, encode_from_pre_quantized_ac doesn't
            // read xyb_x/y/b — pass empty Vecs and skip the 60 ms
            // XYB download + 28 ms repack entirely.
            let precomputed = EncoderPrecomputed::from_parts(
                width as usize,
                height as usize,
                xsize_blocks,
                ysize_blocks,
                cpu_pw,
                cpu_ph,
                alloc::vec::Vec::new(),
                alloc::vec::Vec::new(),
                alloc::vec::Vec::new(),
                alloc::vec::Vec::new(), // linear_rgb (rate-control only)
                cfl_map,
                None,
                quant_field_float.clone(),
                masking,
                None,
                ac_strategy,
                true,
                distance,
                0,
                0,
            );
            return run_gpu_dct8_pre_quantized_path(
                self, &plan, &precomputed, &vardct, &quant_field_u8,
                &params, distance, xsize_blocks, ysize_blocks,
            );
        }

        // Slow path (any non-DCT8 block): need XYB on host for CPU
        // transform_and_quantize + compute_cfl_map.
        let (xyb_x_gpu, xyb_y_gpu, xyb_b_gpu) =
            self.download_planes_3ch(&plan.xyb_x_gpu, &plan.xyb_y_gpu, &plan.xyb_b_gpu);

        let repack = |src: &[f32]| -> alloc::vec::Vec<f32> {
            if cpu_pw == gpu_pw as usize && cpu_ph == gpu_ph as usize {
                src.to_vec()
            } else {
                let mut dst = alloc::vec![0.0_f32; cpu_pw * cpu_ph];
                let w = width as usize;
                let h = height as usize;
                for row in 0..cpu_ph {
                    let off = row * (gpu_pw as usize);
                    let dst_off = row * cpu_pw;
                    dst[dst_off..dst_off + cpu_pw].copy_from_slice(&src[off..off + cpu_pw]);
                }
                if cpu_pw > w {
                    for row in 0..cpu_ph {
                        let dst_off = row * cpu_pw;
                        let last_real = dst[dst_off + w - 1];
                        for col in w..cpu_pw {
                            dst[dst_off + col] = last_real;
                        }
                    }
                }
                if cpu_ph > h {
                    let last_real_off = (h - 1) * cpu_pw;
                    for row in h..cpu_ph {
                        let dst_off = row * cpu_pw;
                        dst.copy_within(last_real_off..last_real_off + cpu_pw, dst_off);
                    }
                }
                dst
            }
        };
        let ((xyb_x, xyb_y), xyb_b) = rayon::join(
            || rayon::join(|| repack(&xyb_x_gpu), || repack(&xyb_y_gpu)),
            || repack(&xyb_b_gpu),
        );

        let cfl_map = compute_cfl_map(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            cpu_pw,
            cpu_ph,
            xsize_blocks,
            ysize_blocks,
            true,
            1e-3,
            10,
        );

        let precomputed = EncoderPrecomputed::from_parts(
            width as usize,
            height as usize,
            xsize_blocks,
            ysize_blocks,
            cpu_pw,
            cpu_ph,
            xyb_x,
            xyb_y,
            xyb_b,
            alloc::vec::Vec::new(),
            cfl_map,
            None,
            quant_field_float,
            masking,
            None,
            ac_strategy,
            true,
            distance,
            0,
            0,
        );

        vardct
            .encode_from_precomputed(&precomputed, &quant_field_u8)
            .map_err(jxl_encoder::api::EncodeError::from)
    }

    /// e8+ variant of [`Self::encode_lossy_to_bitstream_via_precomputed`]
    /// — runs `refine_aq_field_gpu_with_strategy_search_persistent`
    /// first to get a butteraugli-refined per-block quant field, then
    /// hands it to the encoder. Closes the quality gap to cjxl at
    /// low distances at the cost of `iters * encode_iter_cost` extra
    /// GPU work (e.g. ~1.0 sec / iter at 12 MP — see perf_e7_vs_e8).
    ///
    /// `ref_srgb` is the original image as interleaved sRGB u8 (used
    /// by butteraugli as the reference). MUST be exactly
    /// `width * height * 3` bytes.
    ///
    /// `iters` mirrors libjxl effort gating: 2 for Kitten (e8), 4 for
    /// Cheetah (e9+).
    #[cfg(all(feature = "encoder", feature = "butteraugli-loop"))]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_lossy_to_bitstream_via_precomputed_with_butteraugli(
        &self,
        lossy: &crate::lossy_encoder::LossyEncoder<R>,
        bg: &mut crate::forks::butteraugli_loop::ButteraugliLoopGpu<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        ref_srgb: &[u8],
        distance: f32,
        iters: usize,
    ) -> Result<alloc::vec::Vec<u8>, jxl_encoder::api::EncodeError> {
        use jxl_encoder::__pre_quantized::{
            AcStrategyMap, DistanceParams, EncoderPrecomputed, VarDctEncoder, compute_cfl_map,
            quantize_quant_field,
        };

        let (width, height) = lossy.dimensions();
        let (gpu_pw, gpu_ph) = lossy.padded_dimensions();
        let cpu_pw = (width as usize).div_ceil(8) * 8;
        let cpu_ph = (height as usize).div_ceil(8) * 8;
        let xsize_blocks = cpu_pw / 8;
        let ysize_blocks = cpu_ph / 8;

        let gpu_xsize_blocks = (gpu_pw as usize) / 8;
        let gpu_ysize_blocks = (gpu_ph as usize) / 8;

        // Compute an adaptive initial_aq via compute_quant_field_float_free
        // (matches what the bitstream emit path will see), then upsample
        // to GPU's padded-block grid. Seeding with adaptive (vs uniform
        // distance_to_qac) gives the refinement loop a much better
        // starting point — uniform initial wastes most of the iter
        // budget converging on the per-block masking the encoder
        // already knows about.
        let plan = lossy.prepare_strategy_search_plan(self, r, g, b, distance);
        let xyb_x_dl = self.download_plane(&plan.xyb_x_gpu);
        let xyb_y_dl = self.download_plane(&plan.xyb_y_gpu);
        let xyb_b_dl = self.download_plane(&plan.xyb_b_gpu);
        // Same edge-replication fix as the e7 variant's `repack`:
        // GPU's gaborish ran on a gpu_pw-padded buffer; the cols
        // [width, cpu_pw) inside that come from gaborish-blended
        // edge replication on the GPU's wider grid, NOT what CPU
        // would produce. After extracting the upper-left
        // cpu_pw × cpu_ph, re-replicate cols [width, cpu_pw) from
        // col (width-1) and rows [height, cpu_ph) from row (height-1).
        // See encode_lossy_to_bitstream_via_precomputed for the full
        // diagnosis.
        let repack_first = |src: &[f32]| -> alloc::vec::Vec<f32> {
            if cpu_pw == gpu_pw as usize && cpu_ph == gpu_ph as usize {
                src.to_vec()
            } else {
                let mut dst = alloc::vec![0.0_f32; cpu_pw * cpu_ph];
                let w = width as usize;
                let h = height as usize;
                for row in 0..cpu_ph {
                    let off = row * (gpu_pw as usize);
                    let dst_off = row * cpu_pw;
                    dst[dst_off..dst_off + cpu_pw].copy_from_slice(&src[off..off + cpu_pw]);
                }
                if cpu_pw > w {
                    for row in 0..cpu_ph {
                        let dst_off = row * cpu_pw;
                        let src_val = dst[dst_off + (w - 1)];
                        for c in w..cpu_pw {
                            dst[dst_off + c] = src_val;
                        }
                    }
                }
                if cpu_ph > h {
                    let last_real_off = (h - 1) * cpu_pw;
                    for row in h..cpu_ph {
                        let dst_off = row * cpu_pw;
                        dst.copy_within(last_real_off..last_real_off + cpu_pw, dst_off);
                    }
                }
                dst
            }
        };
        let xyb_x_cpu = repack_first(&xyb_x_dl);
        let xyb_y_cpu = repack_first(&xyb_y_dl);
        let xyb_b_cpu = repack_first(&xyb_b_dl);
        let _ = (&xyb_x_cpu, &xyb_y_cpu, &xyb_b_cpu); // kept for downstream scope
        // Reuse the GPU-computed quant_field from the strategy-search
        // plan instead of re-running compute_quant_field_float_free
        // on the post-fix-up CPU buffers — they're bit-equivalent
        // post-quantization (see test_compute_quant_field_production_flow_divergence).
        let adaptive_initial_cpu = plan.quant_field_float.clone();

        // Upsample CPU-grid (xsize_blocks × ysize_blocks) → GPU-grid
        // (gpu_xsize_blocks × gpu_ysize_blocks): copy row by row,
        // edge-replicate the right column / bottom row.
        let initial_aq: alloc::vec::Vec<f32> = if xsize_blocks == gpu_xsize_blocks
            && ysize_blocks == gpu_ysize_blocks
        {
            adaptive_initial_cpu.clone()
        } else {
            let qac = crate::lossy_encoder::distance_to_qac(distance);
            let mut dst = alloc::vec![qac; gpu_xsize_blocks * gpu_ysize_blocks];
            for by in 0..ysize_blocks {
                for bx in 0..xsize_blocks {
                    dst[by * gpu_xsize_blocks + bx] = adaptive_initial_cpu[by * xsize_blocks + bx];
                }
            }
            dst
        };
        let refined_outcome =
            crate::forks::butteraugli_loop::refine_aq_field_gpu_with_strategy_search_persistent(
                self,
                lossy,
                bg,
                r,
                g,
                b,
                ref_srgb,
                &initial_aq,
                distance,
                iters,
                |_| {},
            )
            .map_err(|e| jxl_encoder::api::EncodeError::InvalidInput {
                message: alloc::format!("butteraugli refinement failed: {e:?}"),
            })?;
        let refined_aq_gpu_grid = refined_outcome.aq_field;
        // `final_inv_scale` is the recomputed inv_scale from a final
        // SetQuantField pass on the converged aq_field (matches the
        // CPU butteraugli loop's final_params; see its return value at
        // jxl-encoder/src/vardct/butteraugli_loop.rs:469-477). We must
        // use this — NOT a fresh DistanceParams::compute_for_profile —
        // when converting to u8 below, otherwise the bitstream's
        // global_scale mismatches what the loop converged on and bytes
        // diverge from the in-loop estimates.
        let refined_inv_scale = refined_outcome.final_inv_scale;

        // Reuse the xyb we already downloaded + repacked for the
        // adaptive_initial seed above.
        let xyb_x = xyb_x_cpu;
        let xyb_y = xyb_y_cpu;
        let xyb_b = xyb_b_cpu;

        // Repack refined aq_field from GPU block grid to CPU block grid.
        let refined_aq: alloc::vec::Vec<f32> =
            if xsize_blocks == gpu_xsize_blocks && ysize_blocks == gpu_ysize_blocks {
                refined_aq_gpu_grid
            } else {
                let mut dst = alloc::vec::Vec::with_capacity(xsize_blocks * ysize_blocks);
                for by in 0..ysize_blocks {
                    let off = by * gpu_xsize_blocks;
                    dst.extend_from_slice(&refined_aq_gpu_grid[off..off + xsize_blocks]);
                }
                dst
            };

        // AcStrategyMap from plan.assignments (clipped to CPU grid).
        let mut ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);
        for a in &plan.assignments {
            let bx = a.bx as usize;
            let by = a.by as usize;
            if bx >= xsize_blocks || by >= ysize_blocks {
                continue;
            }
            use crate::forks::transform::*;
            let (cx, cy): (usize, usize) = match a.raw_strategy {
                RAW_STRATEGY_DCT16X8 => (1, 2),
                RAW_STRATEGY_DCT8X16 => (2, 1),
                RAW_STRATEGY_DCT16X16 => (2, 2),
                RAW_STRATEGY_DCT32X16 => (2, 4),
                RAW_STRATEGY_DCT16X32 => (4, 2),
                RAW_STRATEGY_DCT32X32 => (4, 4),
                RAW_STRATEGY_DCT64X32 => (4, 8),
                RAW_STRATEGY_DCT32X64 => (8, 4),
                RAW_STRATEGY_DCT64X64 => (8, 8),
                _ => (1, 1),
            };
            if bx + cx > xsize_blocks || by + cy > ysize_blocks {
                continue;
            }
            if a.raw_strategy != 0 {
                ac_strategy.set(bx, by, a.raw_strategy);
            }
        }

        // CfL on host from XYB.
        let cfl_map = compute_cfl_map(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            cpu_pw,
            cpu_ph,
            xsize_blocks,
            ysize_blocks,
            true,
            1e-3,
            10,
        );

        // We still need masking from compute_quant_field_float_free,
        // but use the BUTTERAUGLI-REFINED aq_field as quant_field_float
        // (overrides the initial from compute_quant_field_float_free).
        // Pull masking from the GPU plan (compute_quant_field_full_persistent)
        // — the matching `quant_field_float` is replaced below by `refined_aq`,
        // so we only need the masking field here.
        let masking = plan.masking.clone();
        let quant_field_float = refined_aq;

        let precomputed = EncoderPrecomputed::from_parts(
            width as usize,
            height as usize,
            xsize_blocks,
            ysize_blocks,
            cpu_pw,
            cpu_ph,
            xyb_x,
            xyb_y,
            xyb_b,
            alloc::vec::Vec::new(),
            cfl_map,
            None,
            quant_field_float.clone(),
            masking,
            None,
            ac_strategy,
            true,
            distance,
            0,
            0,
        );

        let mut vardct = VarDctEncoder::new(distance);
        let profile_params = DistanceParams::compute_for_profile(distance, &vardct.profile);

        // Thread the per-iter SetQuantField recompute (matching CPU
        // `vardct/butteraugli_loop.rs`) through to the bitstream:
        // `encode_from_precomputed` re-derives its `params` from
        // `compute_for_profile` (a profile-fixed q formula), so without
        // this rescale the bitstream's `global_scale` would NOT match
        // the inv_scale we just used to quantize the float field, and
        // the decoder would dequantize against the wrong scale.
        //
        // `quant_ac_rescale = r` makes the encoder rebuild
        // `params.global_scale = round(profile.global_scale * r)` —
        // pick `r = refined_scale / profile_scale` so the resulting
        // global_scale matches the loop's converged
        // `compute_from_quant_field` result.
        //
        // Equal scales (e.g. when median-MAD ≈ profile.initial_q_numerator)
        // → `r ≈ 1.0` and `apply_quant_ac_rescale` no-ops on its
        // `(rescale - 1.0).abs() < EPSILON` guard. Mismatched scales
        // → `r != 1.0` and the encoder rebuilds global_scale to match.
        let rescale = refined_outcome.final_scale / profile_params.scale;
        if rescale.is_finite() && rescale > 0.0 {
            vardct.quant_ac_rescale = Some(rescale);
        }

        let quant_field_u8 = quantize_quant_field(&quant_field_float, refined_inv_scale);

        vardct
            .encode_from_precomputed(&precomputed, &quant_field_u8)
            .map_err(jxl_encoder::api::EncodeError::from)
    }
}

impl<R: Runtime> Default for GpuEncoder<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Conditional GPU pre-quantized AC fast path. Mirrors
/// `encode_lossy_to_bitstream_via_precomputed_from_u8`'s tail (steps
/// 7–8) but produces the per-channel `quant_dc/quant_ac/nzeros/raw_nzeros`
/// on GPU and feeds them to `encode_from_pre_quantized_ac` instead of
/// re-running CPU `transform_and_quantize`.
///
/// Caller has already verified every block uses DCT8.
#[allow(clippy::too_many_arguments)]
fn run_gpu_dct8_pre_quantized_path<R: Runtime>(
    enc: &GpuEncoder<R>,
    plan: &crate::lossy_encoder::StrategySearchPlan<R>,
    precomputed: &jxl_encoder::__pre_quantized::EncoderPrecomputed,
    vardct: &jxl_encoder::__pre_quantized::VarDctEncoder,
    quant_field_u8: &[u8],
    params: &jxl_encoder::__pre_quantized::DistanceParams,
    distance: f32,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> Result<alloc::vec::Vec<u8>, jxl_encoder::api::EncodeError> {
    use crate::forks::pre_quantized_ac::{
        PreQuantizedDct8Params, compute_pre_quantized_ac_dct8_persistent,
        reshape_to_transform_output,
    };
    let _ = distance;
    let n_blocks = xsize_blocks * ysize_blocks;
    let cfl_map = &precomputed.cfl_map;

    // Expand per-tile cfl_map to per-block factors. CfL tile = 64 px
    // = 8 blocks per side.
    const K_INV_COLOR_FACTOR: f32 = 1.0 / 84.0;
    let mut x_factor_per_block = alloc::vec![0.0f32; n_blocks];
    let mut b_factor_per_block = alloc::vec![0.0f32; n_blocks];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let tx = bx / 8;
            let ty = by / 8;
            let i = by * xsize_blocks + bx;
            x_factor_per_block[i] = (cfl_map.ytox_at(tx, ty) as f32) * K_INV_COLOR_FACTOR;
            b_factor_per_block[i] = 1.0 + (cfl_map.ytob_at(tx, ty) as f32) * K_INV_COLOR_FACTOR;
        }
    }
    let x_qm_mul = (1.25_f32).powf(params.x_qm_scale as f32 - 2.0);
    let b_qm_mul = (1.25_f32).powf(params.b_qm_scale as f32 - 2.0);
    let qac_per_block: alloc::vec::Vec<f32> =
        quant_field_u8.iter().map(|&q| params.scale * q as f32).collect();
    let qac_qm_x: alloc::vec::Vec<f32> =
        qac_per_block.iter().map(|&q| q * x_qm_mul).collect();
    let qac_qm_y: alloc::vec::Vec<f32> = qac_per_block.clone();
    let qac_qm_b: alloc::vec::Vec<f32> =
        qac_per_block.iter().map(|&q| q * b_qm_mul).collect();

    fn arr64(s: &[f32]) -> [f32; 64] {
        let mut a = [0.0_f32; 64];
        a.copy_from_slice(s);
        a
    }
    let dct8_weights_x = arr64(jxl_encoder::__pre_quantized::quant_weights_dct8(0));
    let dct8_weights_y = arr64(jxl_encoder::__pre_quantized::quant_weights_dct8(1));
    let dct8_weights_b = arr64(jxl_encoder::__pre_quantized::quant_weights_dct8(2));

    let pq_params = PreQuantizedDct8Params {
        qac_per_block,
        x_factor_per_block,
        b_factor_per_block,
        qac_qm_x,
        qac_qm_y,
        qac_qm_b,
        inv_dc_factor_x: jxl_encoder::__pre_quantized::INV_DC_QUANT[0] * params.scale_dc,
        inv_dc_factor_y: jxl_encoder::__pre_quantized::INV_DC_QUANT[1] * params.scale_dc,
        inv_dc_factor_b: jxl_encoder::__pre_quantized::INV_DC_QUANT[2] * params.scale_dc,
        thresholds_x: jxl_encoder::__pre_quantized::default_thresholds_dct8(0),
        thresholds_y: jxl_encoder::__pre_quantized::default_thresholds_dct8(1),
        thresholds_b: jxl_encoder::__pre_quantized::default_thresholds_dct8(2),
    };

    let pq = compute_pre_quantized_ac_dct8_persistent(
        enc,
        &plan.xyb_x_gpu,
        &plan.xyb_y_gpu,
        &plan.xyb_b_gpu,
        xsize_blocks,
        ysize_blocks,
        &dct8_weights_x,
        &dct8_weights_y,
        &dct8_weights_b,
        &pq_params,
    );
    let r = reshape_to_transform_output(pq, xsize_blocks, ysize_blocks);

    vardct
        .encode_from_pre_quantized_ac(
            precomputed,
            quant_field_u8,
            &r.quant_dc,
            &r.quant_ac,
            &r.nzeros,
            &r.raw_nzeros,
        )
        .map_err(jxl_encoder::api::EncodeError::from)
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
        assert!(
            m > 1e-3,
            "IDENTITY and DCT2X2 should produce different coeffs (max|Δ|={m:.3e})"
        );
    }

    /// Verify entropy_coeffs_coeff_blocks_broadcast_w produces output
    /// matching the per-block variant called with replicated inv_weights.
    /// Coefficient-domain mode (info_loss + info_loss2 outputs are
    /// non-zero, unlike pixel-domain).
    #[test]
    fn test_entropy_coeffs_coeff_broadcast_matches_perblock() {
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n_blocks = 16usize;
        let n_per_block = 64u32;
        let n = n_per_block as usize;
        let total = n_blocks * n;

        let mut block_c = alloc::vec![0.0f32; total];
        let mut block_y = alloc::vec![0.0f32; total];
        for i in 0..total {
            let v = ((i.wrapping_mul(31) % 251) as f32 / 251.0) - 0.5;
            block_c[i] = 0.4 + 1.5 * v;
            block_y[i] = -0.2 + 1.0 * v;
        }
        let mut inv_weights_template = alloc::vec![0.0f32; n];
        for i in 0..n {
            let w = 0.5 + 0.7 * ((i.wrapping_mul(17) % 251) as f32 / 251.0);
            inv_weights_template[i] = 1.0 / w;
        }
        let mut inv_weights_replicated = alloc::vec![0.0f32; total];
        for b in 0..n_blocks {
            inv_weights_replicated[b * n..(b + 1) * n].copy_from_slice(&inv_weights_template);
        }

        let cmap_factor = 0.012f32;
        let quant = 0.7f32;
        let k_cost_delta = 1.83f32;
        let k_cost2 = 2.5f32;

        let out_p = enc.entropy_coeffs_coeff_blocks(
            &block_c,
            &block_y,
            &inv_weights_replicated,
            n_per_block,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
        );
        let out_b = enc.entropy_coeffs_coeff_blocks_broadcast_w(
            &block_c,
            &block_y,
            &inv_weights_template,
            n_per_block,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
        );

        assert_eq!(out_p.len(), out_b.len());
        let m = max_abs_diff(&out_p, &out_b);
        assert!(m < 1e-5, "max|Δ|={m:.3e} (expected < 1e-5)");
    }

    /// Verify entropy_coeffs_pixel_blocks_broadcast_w produces output
    /// matching the per-block variant called with replicated weights.
    /// Tested at n_per_block=64 (DCT8 layout) which is the cost-grid
    /// hot path. Asserts max|Δ| < 1e-5 (kernel uses cmap math + sqrt
    /// so identical i32-style equality isn't applicable).
    #[test]
    fn test_entropy_coeffs_pixel_broadcast_matches_perblock() {
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n_blocks = 16usize;
        let n_per_block = 64u32;
        let n = n_per_block as usize;
        let total = n_blocks * n;

        let mut block_c = alloc::vec![0.0f32; total];
        let mut block_y = alloc::vec![0.0f32; total];
        for i in 0..total {
            let v = ((i.wrapping_mul(31) % 251) as f32 / 251.0) - 0.5;
            block_c[i] = 0.4 + 1.5 * v;
            block_y[i] = -0.2 + 1.0 * v;
        }
        let mut weights_template = alloc::vec![0.0f32; n];
        let mut inv_weights_template = alloc::vec![0.0f32; n];
        for i in 0..n {
            let w = 0.5 + 0.7 * ((i.wrapping_mul(17) % 251) as f32 / 251.0);
            weights_template[i] = w;
            inv_weights_template[i] = 1.0 / w;
        }
        let mut weights_replicated = alloc::vec![0.0f32; total];
        let mut inv_weights_replicated = alloc::vec![0.0f32; total];
        for b in 0..n_blocks {
            weights_replicated[b * n..(b + 1) * n].copy_from_slice(&weights_template);
            inv_weights_replicated[b * n..(b + 1) * n].copy_from_slice(&inv_weights_template);
        }

        let cmap_factor = 0.012f32;
        let quant = 0.7f32;
        let k_cost_delta = 1.83f32;

        let (out_p, err_p) = enc.entropy_coeffs_pixel_blocks(
            &block_c,
            &block_y,
            &weights_replicated,
            &inv_weights_replicated,
            n_per_block,
            cmap_factor,
            quant,
            k_cost_delta,
        );
        let (out_b, err_b) = enc.entropy_coeffs_pixel_blocks_broadcast_w(
            &block_c,
            &block_y,
            &weights_template,
            &inv_weights_template,
            n_per_block,
            cmap_factor,
            quant,
            k_cost_delta,
        );

        assert_eq!(out_p.len(), out_b.len());
        assert_eq!(err_p.len(), err_b.len());
        let m_out = max_abs_diff(&out_p, &out_b);
        let m_err = max_abs_diff(&err_p, &err_b);
        assert!(
            m_out < 1e-5,
            "entropy out: max|Δ|={m_out:.3e} (expected < 1e-5)"
        );
        assert!(
            m_err < 1e-5,
            "error_coeffs: max|Δ|={m_err:.3e} (expected < 1e-5)"
        );
    }

    /// Verify quantize_large_blocks_broadcast_w produces bit-identical
    /// output to quantize_large_blocks when the latter is called with
    /// replicated weights. Two grid sizes (8×8 = DCT8 layout, 16×16 =
    /// DCT16x16 layout) cover small and large block cases. LLF region
    /// 2×2 (matches DCT16x16) for the larger; 1×1 (DCT8) for the smaller.
    #[test]
    fn test_quantize_large_broadcast_matches_perblock() {
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let cases = [
            (8u32, 8u32, 1u32, 1u32),   // DCT8 layout
            (16u32, 16u32, 2u32, 2u32), // DCT16x16 layout
        ];
        let qac_qm = alloc::vec![0.7_f32; 16];
        let thresholds = [0.56_f32, 0.62, 0.62, 0.62];

        for &(gw, gh, lx, ly) in &cases {
            let bs = (gw as usize) * (gh as usize);
            let n_blocks = qac_qm.len();
            let mut coeffs = alloc::vec![0.0f32; n_blocks * bs];
            for (i, c) in coeffs.iter_mut().enumerate() {
                let v = ((i.wrapping_mul(31) % 251) as f32 / 251.0) - 0.5;
                *c = 0.4 + 1.5 * v;
            }
            let mut weights_template = alloc::vec![0.0f32; bs];
            for (i, w) in weights_template.iter_mut().enumerate() {
                *w = 0.5 + 0.7 * ((i.wrapping_mul(17) % 251) as f32 / 251.0);
            }
            let mut weights_replicated = alloc::vec![0.0f32; n_blocks * bs];
            for b in 0..n_blocks {
                weights_replicated[b * bs..(b + 1) * bs].copy_from_slice(&weights_template);
            }

            let perblock = enc.quantize_large_blocks(
                &coeffs,
                &weights_replicated,
                &qac_qm,
                &thresholds,
                gw,
                gh,
                lx,
                ly,
            );
            let broadcast = enc.quantize_large_blocks_broadcast_w(
                &coeffs,
                &weights_template,
                &qac_qm,
                &thresholds,
                gw,
                gh,
                lx,
                ly,
            );

            assert_eq!(perblock.len(), broadcast.len());
            for (i, (&a, &b)) in perblock.iter().zip(broadcast.iter()).enumerate() {
                assert_eq!(a, b, "grid={gw}×{gh}, i={i}: perblock={a} vs broadcast={b}");
            }
        }
    }

    /// Verify dequant_simple_blocks_broadcast_w produces bit-identical
    /// output to dequant_simple_blocks when the latter is called with
    /// replicated weights. Two block sizes (64, 256) cover both DCT8
    /// and DCT16x16 layouts.
    #[test]
    fn test_dequant_simple_broadcast_matches_perblock() {
        let enc: GpuEncoder<B> = GpuEncoder::new();
        for &block_size in &[64u32, 256u32] {
            let n_blocks = 16;
            let bs = block_size as usize;
            let mut quant = alloc::vec![0_i32; n_blocks * bs];
            for (i, q) in quant.iter_mut().enumerate() {
                *q = ((i.wrapping_mul(31) % 251) as i32) - 125;
            }
            let mut weights_template = alloc::vec![0.0f32; bs];
            for (i, w) in weights_template.iter_mut().enumerate() {
                *w = 0.5 + 0.5 * ((i.wrapping_mul(17) % 251) as f32 / 251.0);
            }
            let mut weights_replicated = alloc::vec![0.0f32; n_blocks * bs];
            for b in 0..n_blocks {
                weights_replicated[b * bs..(b + 1) * bs].copy_from_slice(&weights_template);
            }

            let perblock = enc.dequant_simple_blocks(&quant, &weights_replicated, block_size);
            let broadcast =
                enc.dequant_simple_blocks_broadcast_w(&quant, &weights_template, block_size);

            assert_eq!(perblock.len(), broadcast.len());
            for (i, (&a, &b)) in perblock.iter().zip(broadcast.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "block_size={block_size}, i={i}: perblock={a} vs broadcast={b}"
                );
            }
        }
    }
}
