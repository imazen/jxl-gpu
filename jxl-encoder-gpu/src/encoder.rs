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

use crate::launch::dct8::{dct_8x8, idct_8x8};
use crate::launch::gab::gab_smooth;
use crate::launch::gaborish::gaborish_5x5;
use crate::launch::mask1x1::mask1x1;
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
