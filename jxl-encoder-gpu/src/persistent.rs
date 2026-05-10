// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Persistent GPU plane handles + chained-launch encoder API.
//!
//! ## Why this exists
//!
//! The default [`GpuEncoder`] methods (e.g., `xyb_from_linear_rgb`,
//! `gaborish_5x5_channel`) each upload their inputs to GPU, run a
//! kernel, and download the outputs back to host memory. This is
//! convenient for one-off calls but defeats GPU pipelining: every
//! stage pays a full PCIe round-trip.
//!
//! `persistent_buffer_pipeline.rs` benchmark measures the cost: at
//! 1024×1024, the round-trip API takes ~57 ms for XYB + gaborish +
//! mask1x1 vs ~17 ms for a hand-coded persistent pipeline (3.2×
//! slower). At 4096×4096, 1970 ms vs 964 ms (2.0× slower).
//!
//! Most users want [`crate::lossy_encoder::LossyEncoder`] instead;
//! this module is the lower-level building blocks for callers
//! composing custom pipelines.
//!
//! ## Typed handles
//!
//! | Type | Layout | Use |
//! |---|---|---|
//! | [`GpuPlane`] | `width × height` `f32` (row-major) | spatial plane data (XYB channels, RGB, masks) |
//! | [`GpuBlocks`] | `num_blocks × coeffs_per_block` `f32` | per-block coefficient data (DCT output, dequantized) |
//! | [`GpuI32Blocks`] | `num_blocks × coeffs_per_block` `i32` | per-block quantized coefficient data |
//!
//! ## Method naming
//!
//! Persistent-API methods on [`GpuEncoder`] are suffixed `_persistent`
//! and take/return the typed handles above. Caller chains stages
//! without host roundtrips. Boundary methods:
//!
//! - [`GpuEncoder::upload_plane`] / [`GpuEncoder::download_plane`]
//! - [`GpuEncoder::upload_blocks`] / [`GpuEncoder::download_blocks`]
//! - [`GpuEncoder::download_i32_blocks`]
//!
//! ## Lifetime model
//!
//! `GpuPlane<R>` owns a `cubecl::Handle`. Drop the handle when done
//! to free the GPU buffer (cubecl handles are reference-counted on
//! the server side; allocation is recycled).
//!
//! Do NOT mix `GpuPlane`s from different `GpuEncoder` instances —
//! the underlying `ComputeClient` must match. There's no compile-time
//! check; misuse will panic on the next launch.

use alloc::vec::Vec;

use cubecl::Runtime;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::encoder::GpuEncoder;
use crate::launch::dc_restore::restore_dc;
use crate::launch::dct4::{
    dct_4x4_full, dct_4x8_full, dct_8x4_full, idct_4x4_full, idct_4x8_full, idct_8x4_full,
};
use crate::launch::dct2x2::{dct2x2_forward, dct2x2_inverse};
use crate::launch::dct8::{dct_8x8, dct_8x8_wide, idct_8x8, idct_8x8_wide};
use crate::launch::identity::{identity_forward, identity_inverse};
use crate::launch::dct16::{dct_8x16, dct_16x8, dct_16x16, idct_8x16, idct_16x8, idct_16x16};
use crate::launch::dct32::{dct_16x32, dct_32x16, dct_32x32, idct_16x32, idct_32x16, idct_32x32};
use crate::launch::dct64::{dct_32x64, dct_64x32, dct_64x64, idct_32x64, idct_64x32, idct_64x64};
use crate::launch::dequant::{dequant_dct8, dequant_dct8_broadcast_w};
use crate::launch::dequant_simple::{
    dequant_strategy_broadcast_w, dequant_strategy_broadcast_w_dct8,
};
use crate::launch::entropy::entropy_coeffs_pixel_broadcast_w;
use crate::launch::epf::{epf_step1, epf_step2, pad_plane};
use crate::launch::fused_dct_quant::{dct8_quantize_fused_wide, dequant_idct8_fused_y_wide};
use crate::launch::gab::gab_smooth;
use crate::launch::gaborish::gaborish_5x5;
use crate::launch::gather::{gather_blocks, scatter_blocks};
use crate::launch::mask1x1::mask1x1;
use crate::launch::pixel_loss::pixel_loss;
use crate::launch::quantize::{
    quantize_dct8, quantize_dct8_broadcast_w, quantize_large_broadcast_w,
};
use crate::launch::xyb::{xyb_forward, xyb_inverse};

/// Typed handle to a GPU-resident `f32` plane. Owns the underlying
/// `cubecl::Handle`; drop to release the GPU buffer.
///
/// Construct via [`GpuEncoder::upload_plane`] or one of the
/// persistent-API methods that return `GpuPlane`s. Read back via
/// [`GpuEncoder::download_plane`].
pub struct GpuPlane<R: Runtime> {
    handle: Handle,
    width: u32,
    height: u32,
    _r: core::marker::PhantomData<R>,
}

impl<R: Runtime> GpuPlane<R> {
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn n_pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
    /// Borrow the underlying handle for chaining custom launches that
    /// aren't covered by the persistent-API methods. The returned
    /// `Handle` is `Clone`; cubecl reference-counts it.
    pub fn handle(&self) -> &Handle {
        &self.handle
    }
}

/// Typed handle to a GPU-resident contiguous buffer of per-block
/// f32 data — DCT coefficients, quantized values, etc.
///
/// Layout: `num_blocks * coeffs_per_block` floats. The transform that
/// produced these blocks determines `coeffs_per_block`:
/// - DCT8: 64
/// - DCT4×8 / DCT8×4 / DCT4×4: 64 (sub-blocks within an 8×8 region)
/// - DCT16×8 / DCT8×16: 128
/// - DCT16×16: 256
/// - DCT32×16 / DCT16×32: 512
/// - DCT32×32: 1024
/// - DCT64×32 / DCT32×64: 2048
/// - DCT64×64: 4096
///
/// `GpuBlocks` is the per-block analog of [`GpuPlane`] (which holds
/// spatially-laid-out plane data).
pub struct GpuBlocks<R: Runtime> {
    handle: Handle,
    num_blocks: u32,
    coeffs_per_block: u32,
    _r: core::marker::PhantomData<R>,
}

impl<R: Runtime> GpuBlocks<R> {
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }
    pub fn coeffs_per_block(&self) -> u32 {
        self.coeffs_per_block
    }
    pub fn total_floats(&self) -> usize {
        (self.num_blocks as usize) * (self.coeffs_per_block as usize)
    }
    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    /// Wrap an externally-allocated GPU `Handle` as a `GpuBlocks`.
    /// Caller is responsible for ensuring the handle's underlying
    /// buffer holds at least `num_blocks * coeffs_per_block * 4`
    /// bytes of `f32` data.
    ///
    /// Used by sub-modules that produce custom GPU outputs (e.g.,
    /// `forks::afv::afv_transform_batch_persistent`) and want to hand
    /// them back as the standard `GpuBlocks` type for use with the
    /// rest of the persistent pipeline.
    pub fn from_handle(handle: Handle, num_blocks: u32, coeffs_per_block: u32) -> Self {
        Self {
            handle,
            num_blocks,
            coeffs_per_block,
            _r: core::marker::PhantomData,
        }
    }
}

/// Typed handle to GPU-resident per-block `i32` data — quantized
/// coefficients. Same `num_blocks * coeffs_per_block` layout as
/// [`GpuBlocks`], but with `i32` element type (kept as raw bytes
/// in cubecl).
pub struct GpuI32Blocks<R: Runtime> {
    handle: Handle,
    num_blocks: u32,
    coeffs_per_block: u32,
    _r: core::marker::PhantomData<R>,
}

impl<R: Runtime> GpuI32Blocks<R> {
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }
    pub fn coeffs_per_block(&self) -> u32 {
        self.coeffs_per_block
    }
    pub fn total_ints(&self) -> usize {
        (self.num_blocks as usize) * (self.coeffs_per_block as usize)
    }
    pub fn handle(&self) -> &Handle {
        &self.handle
    }
}

/// Per-channel weights for [`GpuEncoder::gaborish_5x5_persistent`].
/// All weights are normalized; see `forks::gaborish::compute_weights`.
#[derive(Clone, Copy, Debug)]
pub struct GaborishWeights {
    pub wc: f32,
    pub wr: f32,
    pub wd: f32,
    pub w_big_r: f32,
    pub wl: f32,
    pub w_big_d: f32,
}

impl<R: Runtime> GpuEncoder<R> {
    /// Upload a host `f32` plane to GPU, returning a `GpuPlane`.
    pub fn upload_plane(&self, data: &[f32], width: u32, height: u32) -> GpuPlane<R> {
        let n = (width as usize) * (height as usize);
        assert_eq!(
            data.len(),
            n,
            "data length {} != width*height {n}",
            data.len()
        );
        let handle = self.client_ref().create_from_slice(f32::as_bytes(data));
        GpuPlane {
            handle,
            width,
            height,
            _r: core::marker::PhantomData,
        }
    }

    /// Allocate a zero-filled GPU plane of the given shape. Useful as
    /// a destination for kernels that take pre-allocated outputs.
    pub fn alloc_plane(&self, width: u32, height: u32) -> GpuPlane<R> {
        let n = (width as usize) * (height as usize);
        let handle = self.client_ref().empty(n * 4);
        GpuPlane {
            handle,
            width,
            height,
            _r: core::marker::PhantomData,
        }
    }

    /// Download a GPU plane back to host memory.
    pub fn download_plane(&self, plane: &GpuPlane<R>) -> Vec<f32> {
        let bytes = self
            .client_ref()
            .read_one(plane.handle.clone())
            .expect("download");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Persistent-API XYB forward. Takes 3 GPU-resident planes,
    /// returns 3 GPU-resident planes. No host transfer.
    pub fn xyb_from_linear_rgb_persistent(
        &self,
        r: &GpuPlane<R>,
        g: &GpuPlane<R>,
        b: &GpuPlane<R>,
    ) -> (GpuPlane<R>, GpuPlane<R>, GpuPlane<R>) {
        assert_eq!(r.width, g.width);
        assert_eq!(r.height, g.height);
        assert_eq!(r.width, b.width);
        assert_eq!(r.height, b.height);
        let w = r.width;
        let h = r.height;
        let n = r.n_pixels();
        let h_x = self.client_ref().empty(n * 4);
        let h_y = self.client_ref().empty(n * 4);
        let h_b_out = self.client_ref().empty(n * 4);
        xyb_forward::<R>(
            self.client_ref(),
            r.handle.clone(),
            g.handle.clone(),
            b.handle.clone(),
            h_x.clone(),
            h_y.clone(),
            h_b_out.clone(),
            n as u32,
        );
        (
            GpuPlane {
                handle: h_x,
                width: w,
                height: h,
                _r: core::marker::PhantomData,
            },
            GpuPlane {
                handle: h_y,
                width: w,
                height: h,
                _r: core::marker::PhantomData,
            },
            GpuPlane {
                handle: h_b_out,
                width: w,
                height: h,
                _r: core::marker::PhantomData,
            },
        )
    }

    /// Persistent-API XYB inverse (XYB → planar linear RGB).
    pub fn xyb_to_linear_rgb_planar_persistent(
        &self,
        x: &GpuPlane<R>,
        y: &GpuPlane<R>,
        b: &GpuPlane<R>,
    ) -> (GpuPlane<R>, GpuPlane<R>, GpuPlane<R>) {
        let w = x.width;
        let h = x.height;
        let n = x.n_pixels();
        let h_r = self.client_ref().empty(n * 4);
        let h_g = self.client_ref().empty(n * 4);
        let h_b = self.client_ref().empty(n * 4);
        xyb_inverse::<R>(
            self.client_ref(),
            x.handle.clone(),
            y.handle.clone(),
            b.handle.clone(),
            h_r.clone(),
            h_g.clone(),
            h_b.clone(),
            n as u32,
        );
        (
            GpuPlane {
                handle: h_r,
                width: w,
                height: h,
                _r: core::marker::PhantomData,
            },
            GpuPlane {
                handle: h_g,
                width: w,
                height: h,
                _r: core::marker::PhantomData,
            },
            GpuPlane {
                handle: h_b,
                width: w,
                height: h,
                _r: core::marker::PhantomData,
            },
        )
    }

    /// Persistent-API gaborish 5×5 sharpen on one channel.
    /// Returns a new `GpuPlane`; input plane unmodified.
    pub fn gaborish_5x5_persistent(
        &self,
        plane: &GpuPlane<R>,
        weights: &GaborishWeights,
    ) -> GpuPlane<R> {
        let n = plane.n_pixels();
        let h_out = self.client_ref().empty(n * 4);
        gaborish_5x5::<R>(
            self.client_ref(),
            plane.handle.clone(),
            h_out.clone(),
            plane.width,
            plane.height,
            weights.wc,
            weights.wr,
            weights.wd,
            weights.w_big_r,
            weights.wl,
            weights.w_big_d,
        );
        GpuPlane {
            handle: h_out,
            width: plane.width,
            height: plane.height,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent-API decoder gab smoothing (3×3 plus, used in
    /// reconstruct.rs). Returns a new `GpuPlane`.
    pub fn gab_smooth_persistent(
        &self,
        plane: &GpuPlane<R>,
        w_center: f32,
        w1: f32,
        w2: f32,
    ) -> GpuPlane<R> {
        let n = plane.n_pixels();
        let h_out = self.client_ref().empty(n * 4);
        gab_smooth::<R>(
            self.client_ref(),
            plane.handle.clone(),
            h_out.clone(),
            plane.width,
            plane.height,
            w_center,
            w1,
            w2,
        );
        GpuPlane {
            handle: h_out,
            width: plane.width,
            height: plane.height,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent-API edge-replicate plane padding. Returns a new
    /// `GpuPlane` of size `(width + 2*pad) × (height + 2*pad)`.
    pub fn pad_plane_persistent(&self, plane: &GpuPlane<R>, pad: u32) -> GpuPlane<R> {
        let dst_w = plane.width + 2 * pad;
        let dst_h = plane.height + 2 * pad;
        let dst_n = (dst_w as usize) * (dst_h as usize);
        let h_out = self.client_ref().empty(dst_n * 4);
        pad_plane::<R>(
            self.client_ref(),
            plane.handle.clone(),
            h_out.clone(),
            plane.width,
            plane.height,
            pad,
        );
        GpuPlane {
            handle: h_out,
            width: dst_w,
            height: dst_h,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent-API EPF step 1 (3×3 plus, 5-pos SAD). Inputs are
    /// PADDED `GpuPlane`s (caller is responsible for padding via
    /// [`Self::pad_plane_persistent`] with `pad = 2`). `inv_sigma` is
    /// a per-(8×8)-block `Handle` of length `xsize_blocks *
    /// ysize_blocks`. `width` / `height` are the UNPADDED output
    /// dimensions. Returns three new unpadded `GpuPlane`s.
    ///
    /// Mirrors the shape of [`crate::forks::epf::apply_epf_step1_gpu`]
    /// but skips upload/download — all I/O stays on GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn epf_step1_persistent(
        &self,
        in_x: &GpuPlane<R>,
        in_y: &GpuPlane<R>,
        in_b: &GpuPlane<R>,
        inv_sigma: &cubecl::server::Handle,
        width: u32,
        height: u32,
        xsize_blocks: u32,
        ysize_blocks: u32,
        pad: u32,
        sigma_scale: f32,
        border_sigma_mul: f32,
    ) -> (GpuPlane<R>, GpuPlane<R>, GpuPlane<R>) {
        let n_out = (width as usize) * (height as usize);
        let h_ox = self.client_ref().empty(n_out * 4);
        let h_oy = self.client_ref().empty(n_out * 4);
        let h_ob = self.client_ref().empty(n_out * 4);
        epf_step1::<R>(
            self.client_ref(),
            in_x.handle.clone(),
            in_y.handle.clone(),
            in_b.handle.clone(),
            h_ox.clone(),
            h_oy.clone(),
            h_ob.clone(),
            inv_sigma.clone(),
            width,
            height,
            xsize_blocks,
            ysize_blocks,
            pad,
            sigma_scale,
            border_sigma_mul,
        );
        (
            GpuPlane {
                handle: h_ox,
                width,
                height,
                _r: core::marker::PhantomData,
            },
            GpuPlane {
                handle: h_oy,
                width,
                height,
                _r: core::marker::PhantomData,
            },
            GpuPlane {
                handle: h_ob,
                width,
                height,
                _r: core::marker::PhantomData,
            },
        )
    }

    /// Persistent-API EPF step 2 (3×3 plus, single-point SAD). Same
    /// I/O shape as [`Self::epf_step1_persistent`] except `pad = 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn epf_step2_persistent(
        &self,
        in_x: &GpuPlane<R>,
        in_y: &GpuPlane<R>,
        in_b: &GpuPlane<R>,
        inv_sigma: &cubecl::server::Handle,
        width: u32,
        height: u32,
        xsize_blocks: u32,
        ysize_blocks: u32,
        pad: u32,
        sigma_scale: f32,
        border_sigma_mul: f32,
    ) -> (GpuPlane<R>, GpuPlane<R>, GpuPlane<R>) {
        let n_out = (width as usize) * (height as usize);
        let h_ox = self.client_ref().empty(n_out * 4);
        let h_oy = self.client_ref().empty(n_out * 4);
        let h_ob = self.client_ref().empty(n_out * 4);
        epf_step2::<R>(
            self.client_ref(),
            in_x.handle.clone(),
            in_y.handle.clone(),
            in_b.handle.clone(),
            h_ox.clone(),
            h_oy.clone(),
            h_ob.clone(),
            inv_sigma.clone(),
            width,
            height,
            xsize_blocks,
            ysize_blocks,
            pad,
            sigma_scale,
            border_sigma_mul,
        );
        (
            GpuPlane {
                handle: h_ox,
                width,
                height,
                _r: core::marker::PhantomData,
            },
            GpuPlane {
                handle: h_oy,
                width,
                height,
                _r: core::marker::PhantomData,
            },
            GpuPlane {
                handle: h_ob,
                width,
                height,
                _r: core::marker::PhantomData,
            },
        )
    }

    /// Upload an `inv_sigma` map as a raw `Handle` (no GpuPlane wrapper
    /// because per-block layout differs from per-pixel `GpuPlane`).
    /// Caller-managed lifetime — drop to release.
    pub fn upload_inv_sigma(&self, inv_sigma: &[f32]) -> cubecl::server::Handle {
        self.client_ref()
            .create_from_slice(f32::as_bytes(inv_sigma))
    }

    /// Upload per-block coefficient data (e.g., a contiguous batch of
    /// DCT8 blocks) to GPU. `data.len()` must equal
    /// `num_blocks * coeffs_per_block`.
    pub fn upload_blocks(
        &self,
        data: &[f32],
        num_blocks: u32,
        coeffs_per_block: u32,
    ) -> GpuBlocks<R> {
        let expected = (num_blocks as usize) * (coeffs_per_block as usize);
        assert_eq!(
            data.len(),
            expected,
            "data length {} != num_blocks*coeffs_per_block {expected}",
            data.len()
        );
        let handle = self.client_ref().create_from_slice(f32::as_bytes(data));
        GpuBlocks {
            handle,
            num_blocks,
            coeffs_per_block,
            _r: core::marker::PhantomData,
        }
    }

    /// Allocate zero-filled per-block GPU buffer.
    pub fn alloc_blocks(&self, num_blocks: u32, coeffs_per_block: u32) -> GpuBlocks<R> {
        let n = (num_blocks as usize) * (coeffs_per_block as usize);
        let handle = self.client_ref().empty(n * 4);
        GpuBlocks {
            handle,
            num_blocks,
            coeffs_per_block,
            _r: core::marker::PhantomData,
        }
    }

    /// Download a `GpuBlocks` back to host memory.
    pub fn download_blocks(&self, blocks: &GpuBlocks<R>) -> Vec<f32> {
        let bytes = self
            .client_ref()
            .read_one(blocks.handle.clone())
            .expect("download");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Persistent-API forward DCT8 on a batch of 8×8 blocks.
    /// Input `coeffs_per_block` must be 64; output is 64-per-block.
    pub fn dct_8x8_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        assert_eq!(
            blocks.coeffs_per_block, 64,
            "DCT8 expects 64 floats per block, got {}",
            blocks.coeffs_per_block
        );
        let n = blocks.total_floats();
        let h_out = self.client_ref().empty(n * 4);
        dct_8x8::<R>(
            self.client_ref(),
            blocks.handle.clone(),
            h_out.clone(),
            blocks.num_blocks,
        );
        GpuBlocks {
            handle: h_out,
            num_blocks: blocks.num_blocks,
            coeffs_per_block: 64,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent-API inverse DCT8 on a batch of 8×8 coefficient blocks.
    pub fn idct_8x8_persistent(&self, coeffs: &GpuBlocks<R>) -> GpuBlocks<R> {
        assert_eq!(
            coeffs.coeffs_per_block, 64,
            "IDCT8 expects 64 floats per block, got {}",
            coeffs.coeffs_per_block
        );
        let n = coeffs.total_floats();
        let h_out = self.client_ref().empty(n * 4);
        idct_8x8::<R>(
            self.client_ref(),
            coeffs.handle.clone(),
            h_out.clone(),
            coeffs.num_blocks,
        );
        GpuBlocks {
            handle: h_out,
            num_blocks: coeffs.num_blocks,
            coeffs_per_block: 64,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent-API wide-cube forward DCT8. Same I/O contract as
    /// [`Self::dct_8x8_persistent`] but uses cube_dim=64 (~3× faster
    /// at 1024² per dct8_coop_bench results).
    pub fn dct_8x8_wide_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        assert_eq!(blocks.coeffs_per_block, 64);
        let n = blocks.total_floats();
        let h_out = self.client_ref().empty(n * 4);
        dct_8x8_wide::<R>(
            self.client_ref(),
            blocks.handle.clone(),
            h_out.clone(),
            blocks.num_blocks,
        );
        GpuBlocks {
            handle: h_out,
            num_blocks: blocks.num_blocks,
            coeffs_per_block: 64,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent-API wide-cube inverse DCT8.
    pub fn idct_8x8_wide_persistent(&self, coeffs: &GpuBlocks<R>) -> GpuBlocks<R> {
        assert_eq!(coeffs.coeffs_per_block, 64);
        let n = coeffs.total_floats();
        let h_out = self.client_ref().empty(n * 4);
        idct_8x8_wide::<R>(
            self.client_ref(),
            coeffs.handle.clone(),
            h_out.clone(),
            coeffs.num_blocks,
        );
        GpuBlocks {
            handle: h_out,
            num_blocks: coeffs.num_blocks,
            coeffs_per_block: 64,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent-API entropy_coeffs + error-coef writeback (broadcast
    /// weights variant). Same algorithmic semantics as
    /// [`Self::entropy_coeffs_pixel_blocks_broadcast_w`] but keeps
    /// inputs/outputs on GPU (no synchronous downloads).
    ///
    /// Inputs:
    /// - `block_c`: per-block channel coefficients (`coeffs_per_block ==
    ///   n_per_block`).
    /// - `block_y`: per-block Y-channel coefficients (same shape; used
    ///   for CfL prediction via `cmap_factor`).
    /// - `weights_template`: exactly `n_per_block` f32 (one quant
    ///   matrix), broadcast across all blocks.
    /// - `inv_weights_template`: exactly `n_per_block` f32 (1/weights),
    ///   broadcast across all blocks.
    ///
    /// Returns `(stats, error_coeffs)` as `(GpuBlocks 4-per-block,
    /// GpuBlocks n-per-block)`. Stats layout: per-block
    /// `[entropy, nzeros, info_loss, info_loss2]`.
    ///
    /// Saves the per-call host download of stats + error_coeffs that
    /// the non-persistent variant pays — critical for cost-grid pipelines
    /// that chain DCT → entropy → IDCT → pixel_loss across multiple
    /// channels and strategies.
    #[allow(clippy::too_many_arguments)]
    pub fn entropy_coeffs_pixel_blocks_broadcast_w_persistent(
        &self,
        block_c: &GpuBlocks<R>,
        block_y: &GpuBlocks<R>,
        weights_template: &[f32],
        inv_weights_template: &[f32],
        cmap_factor: f32,
        quant: f32,
        k_cost_delta: f32,
    ) -> (GpuBlocks<R>, GpuBlocks<R>) {
        assert_eq!(block_c.num_blocks, block_y.num_blocks);
        assert_eq!(block_c.coeffs_per_block, block_y.coeffs_per_block);
        let n = block_c.coeffs_per_block;
        assert_eq!(weights_template.len() as u32, n);
        assert_eq!(inv_weights_template.len() as u32, n);
        let total = block_c.total_floats();
        let num_blocks = block_c.num_blocks;
        let h_w = self
            .client_ref()
            .create_from_slice(f32::as_bytes(weights_template));
        let h_iw = self
            .client_ref()
            .create_from_slice(f32::as_bytes(inv_weights_template));
        let h_err = self.client_ref().empty(total * 4);
        let h_out = self.client_ref().empty((num_blocks as usize) * 4 * 4);
        entropy_coeffs_pixel_broadcast_w::<R>(
            self.client_ref(),
            block_c.handle.clone(),
            block_y.handle.clone(),
            h_w,
            h_iw,
            h_err.clone(),
            h_out.clone(),
            num_blocks,
            n,
            cmap_factor,
            quant,
            k_cost_delta,
        );
        (
            GpuBlocks {
                handle: h_out,
                num_blocks,
                coeffs_per_block: 4,
                _r: core::marker::PhantomData,
            },
            GpuBlocks {
                handle: h_err,
                num_blocks,
                coeffs_per_block: n,
                _r: core::marker::PhantomData,
            },
        )
    }

    /// Persistent-API per-block 8th-power masked pixel loss. Same
    /// semantics as [`Self::pixel_loss_blocks`] but keeps inputs/outputs
    /// on GPU.
    ///
    /// Inputs:
    /// - `pixel_error`: per-block pixel-domain error values
    ///   (`block_width * block_height` per block).
    /// - `mask_plane`: GPU plane containing the masking image (mask1x1).
    /// - `mask_row_base`: per-block start offsets into the mask plane
    ///   (host slice; uploaded internally).
    ///
    /// Returns `num_blocks` f64 values as a `GpuBlocks` with
    /// `coeffs_per_block = 1` (caller can `download_blocks_f64` if they
    /// need them on host; otherwise feed straight into a combiner kernel).
    ///
    /// **Note**: kernel writes f64 not f32. The returned GpuBlocks has
    /// `coeffs_per_block = 1` and the underlying buffer is `num_blocks
    /// * 8` bytes. Callers must use a dedicated f64 reader to pull it
    /// back (or compose with another kernel that consumes f64).
    #[allow(clippy::too_many_arguments)]
    pub fn pixel_loss_blocks_persistent(
        &self,
        pixel_error: &GpuBlocks<R>,
        mask_plane: &GpuPlane<R>,
        mask_row_base: &[u32],
        mask_offset: f32,
        block_width: u32,
        block_height: u32,
    ) -> GpuBlocks<R> {
        let num_blocks = pixel_error.num_blocks;
        assert_eq!(
            pixel_error.coeffs_per_block,
            block_width * block_height,
            "pixel_error coeffs_per_block must equal block_width*block_height"
        );
        assert_eq!(mask_row_base.len() as u32, num_blocks);
        let h_mrb = self
            .client_ref()
            .create_from_slice(u32::as_bytes(mask_row_base));
        // Output: num_blocks × f64 = num_blocks × 8 bytes.
        let h_out = self.client_ref().empty((num_blocks as usize) * 8);
        let mask_len = (mask_plane.width as usize) * (mask_plane.height as usize);
        pixel_loss::<R>(
            self.client_ref(),
            pixel_error.handle.clone(),
            mask_plane.handle.clone(),
            h_mrb,
            h_out.clone(),
            num_blocks,
            mask_len,
            mask_plane.width,
            mask_offset,
            block_width,
            block_height,
        );
        GpuBlocks {
            handle: h_out,
            num_blocks,
            // coeffs_per_block=1 (logical), but underlying buffer is f64
            // sized — caller must use a matching f64 reader.
            coeffs_per_block: 1,
            _r: core::marker::PhantomData,
        }
    }

    /// Download a `GpuBlocks` whose underlying buffer is f64-typed
    /// (e.g. the result of [`Self::pixel_loss_blocks_persistent`]).
    /// Length returned: `num_blocks` f64 values.
    pub fn download_blocks_f64(&self, blocks: &GpuBlocks<R>) -> Vec<f64> {
        let bytes = self
            .client_ref()
            .read_one(blocks.handle.clone())
            .expect("download_blocks_f64");
        f64::from_bytes(&bytes).to_vec()
    }

    /// Persistent-API fused DCT8 + quantize. One kernel launch instead
    /// of two (DCT then quantize); ~2.84× faster at 1024² per the
    /// fused_dct_quant_bench results. Bit-exact with the split chain.
    ///
    /// Inputs:
    /// - `pixels`: per-block pixel-domain blocks (`coeffs_per_block == 64`)
    /// - `weights`: per-coefficient inverse quant matrix entries
    ///   (same shape as `pixels`)
    /// - `qac_qm`: per-block scale slice (host); future revision can
    ///   take a `GpuPlane`-style handle for repeated calls.
    /// - `thresholds`: 4-quadrant dead-zone thresholds.
    ///
    /// Returns quantized i32 blocks (`GpuI32Blocks` with same num_blocks).
    pub fn dct8_quantize_fused_persistent(
        &self,
        pixels: &GpuBlocks<R>,
        weights: &GpuBlocks<R>,
        qac_qm: &[f32],
        thresholds: &[f32; 4],
    ) -> GpuI32Blocks<R> {
        assert_eq!(pixels.coeffs_per_block, 64);
        assert_eq!(weights.coeffs_per_block, 64);
        assert_eq!(pixels.num_blocks, weights.num_blocks);
        assert_eq!(qac_qm.len() as u32, pixels.num_blocks);
        let n = pixels.total_floats();
        let h_qac = self.client_ref().create_from_slice(f32::as_bytes(qac_qm));
        let h_thr = self
            .client_ref()
            .create_from_slice(f32::as_bytes(thresholds));
        let h_out = self.client_ref().empty(n * 4);
        dct8_quantize_fused_wide::<R>(
            self.client_ref(),
            pixels.handle.clone(),
            weights.handle.clone(),
            h_qac,
            h_thr,
            h_out.clone(),
            pixels.num_blocks,
        );
        GpuI32Blocks {
            handle: h_out,
            num_blocks: pixels.num_blocks,
            coeffs_per_block: 64,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent-API fused dequant + IDCT8 for the Y channel. One
    /// launch instead of two; ~3.07× faster at 1024² per the
    /// fused_dequant_idct_bench results. Bit-exact.
    ///
    /// Single-channel only (no CfL). For X/B channels with CfL,
    /// use [`Self::dequant_dct8_persistent`] + per-channel
    /// [`Self::idct_8x8_persistent`].
    ///
    /// DC slot is forced to 0 (caller restores via dc_coding or
    /// [`Self::restore_dc_persistent`]).
    pub fn dequant_idct8_fused_y_persistent(
        &self,
        quant: &GpuI32Blocks<R>,
        weights: &GpuBlocks<R>,
        qac_qm: &[f32],
    ) -> GpuBlocks<R> {
        assert_eq!(quant.coeffs_per_block, 64);
        assert_eq!(weights.coeffs_per_block, 64);
        assert_eq!(quant.num_blocks, weights.num_blocks);
        assert_eq!(qac_qm.len() as u32, quant.num_blocks);
        let n = (quant.num_blocks as usize) * 64;
        let h_qac = self.client_ref().create_from_slice(f32::as_bytes(qac_qm));
        let h_out = self.client_ref().empty(n * 4);
        dequant_idct8_fused_y_wide::<R>(
            self.client_ref(),
            quant.handle.clone(),
            weights.handle.clone(),
            h_qac,
            h_out.clone(),
            quant.num_blocks,
        );
        GpuBlocks {
            handle: h_out,
            num_blocks: quant.num_blocks,
            coeffs_per_block: 64,
            _r: core::marker::PhantomData,
        }
    }

    /// Internal helper: launch a per-block kernel that takes the same
    /// `(client, in, out, num_blocks)` shape as all DCT/IDCT launchers.
    fn run_per_block_kernel<F>(
        &self,
        blocks: &GpuBlocks<R>,
        out_coeffs_per_block: u32,
        expected_in_coeffs: u32,
        op_name: &str,
        launch: F,
    ) -> GpuBlocks<R>
    where
        F: FnOnce(&cubecl::prelude::ComputeClient<R>, Handle, Handle, u32),
    {
        assert_eq!(
            blocks.coeffs_per_block, expected_in_coeffs,
            "{op_name} expects {expected_in_coeffs} floats/block, got {}",
            blocks.coeffs_per_block
        );
        let n = (blocks.num_blocks as usize) * (out_coeffs_per_block as usize);
        let h_out = self.client_ref().empty(n * 4);
        launch(
            self.client_ref(),
            blocks.handle.clone(),
            h_out.clone(),
            blocks.num_blocks,
        );
        GpuBlocks {
            handle: h_out,
            num_blocks: blocks.num_blocks,
            coeffs_per_block: out_coeffs_per_block,
            _r: core::marker::PhantomData,
        }
    }

    // ── IDENTITY + DCT2x2 (8×8 input, 64-float output, 8×8-class
    //    sub-block strategies) ─────────────────────────────────────────
    pub fn identity_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 64, 64, "Identity", identity_forward::<R>)
    }
    pub fn inverse_identity_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 64, 64, "InverseIdentity", identity_inverse::<R>)
    }
    pub fn dct2x2_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 64, 64, "DCT2x2", dct2x2_forward::<R>)
    }
    pub fn inverse_dct2x2_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 64, 64, "InverseDCT2x2", dct2x2_inverse::<R>)
    }

    // ── DCT/IDCT 4-family (sub-block DCTs operating on 8×8 input,
    //    producing 64-float output) ──────────────────────────────────
    pub fn dct_4x4_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 64, 64, "DCT4x4", dct_4x4_full::<R>)
    }
    pub fn idct_4x4_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 64, 64, "IDCT4x4", idct_4x4_full::<R>)
    }
    pub fn dct_4x8_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 64, 64, "DCT4x8", dct_4x8_full::<R>)
    }
    pub fn idct_4x8_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 64, 64, "IDCT4x8", idct_4x8_full::<R>)
    }
    pub fn dct_8x4_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 64, 64, "DCT8x4", dct_8x4_full::<R>)
    }
    pub fn idct_8x4_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 64, 64, "IDCT8x4", idct_8x4_full::<R>)
    }

    // ── DCT/IDCT 16 family (16×8, 8×16, 16×16) ─────────────────────
    pub fn dct_16x8_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 128, 128, "DCT16x8", dct_16x8::<R>)
    }
    pub fn idct_16x8_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 128, 128, "IDCT16x8", idct_16x8::<R>)
    }
    pub fn dct_8x16_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 128, 128, "DCT8x16", dct_8x16::<R>)
    }
    pub fn idct_8x16_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 128, 128, "IDCT8x16", idct_8x16::<R>)
    }
    pub fn dct_16x16_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 256, 256, "DCT16x16", dct_16x16::<R>)
    }
    pub fn idct_16x16_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 256, 256, "IDCT16x16", idct_16x16::<R>)
    }

    // ── DCT/IDCT 32 family (32×16, 16×32, 32×32) ───────────────────
    pub fn dct_32x16_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 512, 512, "DCT32x16", dct_32x16::<R>)
    }
    pub fn idct_32x16_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 512, 512, "IDCT32x16", idct_32x16::<R>)
    }
    pub fn dct_16x32_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 512, 512, "DCT16x32", dct_16x32::<R>)
    }
    pub fn idct_16x32_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 512, 512, "IDCT16x32", idct_16x32::<R>)
    }
    pub fn dct_32x32_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 1024, 1024, "DCT32x32", dct_32x32::<R>)
    }
    pub fn idct_32x32_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 1024, 1024, "IDCT32x32", idct_32x32::<R>)
    }

    // ── DCT/IDCT 64 family (64×32, 32×64, 64×64) ───────────────────
    pub fn dct_64x32_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 2048, 2048, "DCT64x32", dct_64x32::<R>)
    }
    pub fn idct_64x32_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 2048, 2048, "IDCT64x32", idct_64x32::<R>)
    }
    pub fn dct_32x64_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 2048, 2048, "DCT32x64", dct_32x64::<R>)
    }
    pub fn idct_32x64_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 2048, 2048, "IDCT32x64", idct_32x64::<R>)
    }
    pub fn dct_64x64_persistent(&self, blocks: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(blocks, 4096, 4096, "DCT64x64", dct_64x64::<R>)
    }
    pub fn idct_64x64_persistent(&self, c: &GpuBlocks<R>) -> GpuBlocks<R> {
        self.run_per_block_kernel(c, 4096, 4096, "IDCT64x64", idct_64x64::<R>)
    }

    /// Allocate zero-filled per-block `i32` GPU buffer.
    pub fn alloc_i32_blocks(&self, num_blocks: u32, coeffs_per_block: u32) -> GpuI32Blocks<R> {
        let n = (num_blocks as usize) * (coeffs_per_block as usize);
        let handle = self.client_ref().empty(n * 4);
        GpuI32Blocks {
            handle,
            num_blocks,
            coeffs_per_block,
            _r: core::marker::PhantomData,
        }
    }

    /// Upload a host `i32` per-block buffer to GPU. `data.len()` must
    /// equal `num_blocks * coeffs_per_block`.
    pub fn upload_i32_blocks(
        &self,
        data: &[i32],
        num_blocks: u32,
        coeffs_per_block: u32,
    ) -> GpuI32Blocks<R> {
        let expected = (num_blocks as usize) * (coeffs_per_block as usize);
        assert_eq!(
            data.len(),
            expected,
            "data length {} != num_blocks*coeffs_per_block {expected}",
            data.len()
        );
        let handle = self.client_ref().create_from_slice(i32::as_bytes(data));
        GpuI32Blocks {
            handle,
            num_blocks,
            coeffs_per_block,
            _r: core::marker::PhantomData,
        }
    }

    /// Download a `GpuI32Blocks` back to host memory.
    pub fn download_i32_blocks(&self, blocks: &GpuI32Blocks<R>) -> Vec<i32> {
        let bytes = self
            .client_ref()
            .read_one(blocks.handle.clone())
            .expect("download");
        i32::from_bytes(&bytes).to_vec()
    }

    /// Count occurrences of each token bucket in a host-supplied
    /// `&[u32]` token stream. `bucket_count` MUST be a power of 2;
    /// tokens are masked with `bucket_count - 1` (so values larger
    /// than the bucket range get folded back, matching the standard
    /// hybrid-uint encoding pattern).
    ///
    /// Returns a `Vec<u32>` of length `bucket_count` with the per-
    /// bucket counts.
    ///
    /// **Cost** (CUDA, RTX 5070): ~1 µs per 1k tokens up to ~1M
    /// tokens, then memory-bound on the atomic-add output. The
    /// alternative CPU loop is ~1-2 ns per token; for typical token
    /// stream sizes (1M-10M for a 1024² JXL frame) this is roughly
    /// 5x faster than CPU once the upload is amortized across
    /// multiple calls (use [`Self::histogram_count_pow2_persistent`]
    /// to skip re-uploading).
    pub fn histogram_count_pow2(&self, tokens: &[u32], bucket_count: u32) -> Vec<u32> {
        assert!(
            bucket_count.is_power_of_two(),
            "bucket_count must be a power of 2 (was {bucket_count})"
        );
        let n = tokens.len();
        let client = self.client_ref();
        let h_tokens = client.create_from_slice(u32::as_bytes(tokens));
        let zeros = alloc::vec![0u32; bucket_count as usize];
        let h_hist = client.create_from_slice(u32::as_bytes(&zeros));
        crate::launch::histogram::histogram_count_pow2::<R>(
            client,
            h_tokens,
            h_hist.clone(),
            n as u32,
            bucket_count,
        );
        let bytes = client.read_one(h_hist).expect("histogram download");
        u32::from_bytes(&bytes).to_vec()
    }

    /// GPU-resident variant of [`Self::histogram_count_pow2`]. Takes
    /// a pre-uploaded `GpuI32Blocks` (treats the i32 buffer as u32 by
    /// reinterpretation — caller responsible for ensuring tokens are
    /// non-negative or the masked low bits are still meaningful) and
    /// returns the histogram as host `Vec<u32>`. The token buffer
    /// stays on GPU; only the small histogram comes back.
    ///
    /// Use when feeding many histogram queries against the same
    /// pre-tokenized stream (e.g., evaluating multiple bucket-count
    /// alphabets to pick the best ANS distribution).
    pub fn histogram_count_pow2_persistent(
        &self,
        tokens: &GpuI32Blocks<R>,
        bucket_count: u32,
    ) -> Vec<u32> {
        assert!(
            bucket_count.is_power_of_two(),
            "bucket_count must be a power of 2 (was {bucket_count})"
        );
        let n = (tokens.num_blocks * tokens.coeffs_per_block) as usize;
        let client = self.client_ref();
        let zeros = alloc::vec![0u32; bucket_count as usize];
        let h_hist = client.create_from_slice(u32::as_bytes(&zeros));
        crate::launch::histogram::histogram_count_pow2::<R>(
            client,
            tokens.handle.clone(),
            h_hist.clone(),
            n as u32,
            bucket_count,
        );
        let bytes = client.read_one(h_hist).expect("histogram download");
        u32::from_bytes(&bytes).to_vec()
    }

    /// Persistent-API DCT8 quantize. Takes f32 coeffs + f32 weights +
    /// f32 per-block qac_qm + 4-quadrant thresholds, returns i32
    /// quantized blocks. All inputs live on GPU; output is also GPU-
    /// resident as a [`GpuI32Blocks`].
    ///
    /// `weights` and `coeffs` must have `coeffs_per_block == 64`.
    /// `qac_qm` is a host slice (small enough that uploading per-call
    /// is cheap; future revision can take a `GpuPlane` for repeated
    /// use).
    pub fn quantize_dct8_persistent(
        &self,
        coeffs: &GpuBlocks<R>,
        weights: &GpuBlocks<R>,
        qac_qm: &[f32],
        thresholds: &[f32; 4],
    ) -> GpuI32Blocks<R> {
        assert_eq!(coeffs.coeffs_per_block, 64);
        assert_eq!(weights.coeffs_per_block, 64);
        assert_eq!(coeffs.num_blocks, weights.num_blocks);
        assert_eq!(qac_qm.len() as u32, coeffs.num_blocks);
        let n = coeffs.total_floats();
        let h_qac = self.client_ref().create_from_slice(f32::as_bytes(qac_qm));
        let h_thr = self
            .client_ref()
            .create_from_slice(f32::as_bytes(thresholds));
        let h_out = self.client_ref().empty(n * 4);
        quantize_dct8::<R>(
            self.client_ref(),
            coeffs.handle.clone(),
            weights.handle.clone(),
            h_qac,
            h_thr,
            h_out.clone(),
            coeffs.num_blocks,
        );
        GpuI32Blocks {
            handle: h_out,
            num_blocks: coeffs.num_blocks,
            coeffs_per_block: 64,
            _r: core::marker::PhantomData,
        }
    }

    /// Broadcast-weights variant of [`Self::quantize_dct8_persistent`].
    /// `weights` must be a `GpuBlocks` with `num_blocks == 1` and
    /// `coeffs_per_block == 64` (one DCT8 quant matrix, kernel
    /// broadcasts across all input blocks). Algorithmically identical
    /// to the per-block variant when called with replicated weights;
    /// saves the per-block weight-buffer storage and replication.
    pub fn quantize_dct8_persistent_broadcast_w(
        &self,
        coeffs: &GpuBlocks<R>,
        weights: &GpuBlocks<R>,
        qac_qm: &[f32],
        thresholds: &[f32; 4],
    ) -> GpuI32Blocks<R> {
        assert_eq!(coeffs.coeffs_per_block, 64);
        assert_eq!(
            weights.num_blocks, 1,
            "broadcast_w expects a single 64-coeff weights template"
        );
        assert_eq!(weights.coeffs_per_block, 64);
        assert_eq!(qac_qm.len() as u32, coeffs.num_blocks);
        let n = coeffs.total_floats();
        let h_qac = self.client_ref().create_from_slice(f32::as_bytes(qac_qm));
        let h_thr = self
            .client_ref()
            .create_from_slice(f32::as_bytes(thresholds));
        let h_out = self.client_ref().empty(n * 4);
        quantize_dct8_broadcast_w::<R>(
            self.client_ref(),
            coeffs.handle.clone(),
            weights.handle.clone(),
            h_qac,
            h_thr,
            h_out.clone(),
            coeffs.num_blocks,
        );
        GpuI32Blocks {
            handle: h_out,
            num_blocks: coeffs.num_blocks,
            coeffs_per_block: 64,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent-API generic large-block quantize (broadcast weights).
    /// Same as [`Self::quantize_dct8_persistent_broadcast_w`] but for
    /// strategies whose `coeffs_per_block` is `grid_w * grid_h` and that
    /// have an LLF rectangle of `(llf_x, llf_y)` to be forced to 0.
    ///
    /// `weights_template` is a small host slice (length `grid_w *
    /// grid_h`) — uploaded internally; the kernel broadcasts it across
    /// all blocks. Returns quantized `GpuI32Blocks` (no host roundtrip).
    ///
    /// Algorithmically identical to
    /// [`GpuEncoder::quantize_large_blocks_broadcast_w`] but eliminates
    /// the synchronous output download.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_large_blocks_broadcast_w_persistent(
        &self,
        coeffs: &GpuBlocks<R>,
        weights_template: &[f32],
        qac_qm: &[f32],
        thresholds: &[f32; 4],
        grid_width: u32,
        grid_height: u32,
        llf_x: u32,
        llf_y: u32,
    ) -> GpuI32Blocks<R> {
        let block_size = (grid_width * grid_height) as usize;
        assert_eq!(coeffs.coeffs_per_block as usize, block_size);
        assert_eq!(weights_template.len(), block_size);
        assert_eq!(qac_qm.len() as u32, coeffs.num_blocks);
        let n = coeffs.total_floats();
        let h_w = self
            .client_ref()
            .create_from_slice(f32::as_bytes(weights_template));
        let h_q = self.client_ref().create_from_slice(f32::as_bytes(qac_qm));
        let h_t = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&thresholds[..]));
        let h_o = self.client_ref().empty(n * 4);
        quantize_large_broadcast_w::<R>(
            self.client_ref(),
            coeffs.handle.clone(),
            h_w,
            h_q,
            h_t,
            h_o.clone(),
            coeffs.num_blocks,
            grid_width,
            grid_height,
            llf_x,
            llf_y,
        );
        GpuI32Blocks {
            handle: h_o,
            num_blocks: coeffs.num_blocks,
            coeffs_per_block: block_size as u32,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent strategy dequant (no bias). Same semantics as the
    /// host-side dequant loop in
    /// `forks/reconstruct.rs::encode_and_reconstruct_mixed_strategy_single_channel`
    /// for non-DCT8 strategies: `output[i] = quant[i] * weight[i] / qac[block]`
    /// with per-block qac and broadcast weights.
    ///
    /// LLF positions are NOT zeroed — caller is expected to overwrite
    /// them via `dispatch_restore_llf` (or its GPU variant) on the
    /// downloaded buffer.
    pub fn dequant_strategy_persistent(
        &self,
        quant: &GpuI32Blocks<R>,
        weights_template: &[f32],
        qac_per_block: &[f32],
    ) -> GpuBlocks<R> {
        let bs = quant.coeffs_per_block as usize;
        assert_eq!(weights_template.len(), bs);
        assert_eq!(qac_per_block.len() as u32, quant.num_blocks);
        let n = quant.total_ints();
        let h_w = self
            .client_ref()
            .create_from_slice(f32::as_bytes(weights_template));
        let h_q = self
            .client_ref()
            .create_from_slice(f32::as_bytes(qac_per_block));
        let h_o = self.client_ref().empty(n * 4);
        dequant_strategy_broadcast_w::<R>(
            self.client_ref(),
            quant.handle.clone(),
            h_w,
            h_q,
            h_o.clone(),
            quant.num_blocks,
            quant.coeffs_per_block,
        );
        GpuBlocks {
            handle: h_o,
            num_blocks: quant.num_blocks,
            coeffs_per_block: quant.coeffs_per_block,
            _r: core::marker::PhantomData,
        }
    }

    /// DCT8 variant of [`Self::dequant_strategy_persistent`] that
    /// applies `adjust_quant_bias` for the given channel. `channel`
    /// is `0` (X), `1` (Y), or `2` (B).
    pub fn dequant_strategy_dct8_persistent(
        &self,
        quant: &GpuI32Blocks<R>,
        weights_template: &[f32],
        qac_per_block: &[f32],
        channel: usize,
    ) -> GpuBlocks<R> {
        const BIAS_X: f32 = 0.945_349_93;
        const BIAS_Y: f32 = 0.929_945_5;
        const BIAS_B: f32 = 0.950_064_9;
        let channel_bias = match channel {
            0 => BIAS_X,
            1 => BIAS_Y,
            2 => BIAS_B,
            _ => panic!("dequant_strategy_dct8_persistent: channel must be 0, 1, or 2; got {channel}"),
        };
        let bs = quant.coeffs_per_block as usize;
        assert_eq!(weights_template.len(), bs);
        assert_eq!(qac_per_block.len() as u32, quant.num_blocks);
        let n = quant.total_ints();
        let h_w = self
            .client_ref()
            .create_from_slice(f32::as_bytes(weights_template));
        let h_q = self
            .client_ref()
            .create_from_slice(f32::as_bytes(qac_per_block));
        let h_o = self.client_ref().empty(n * 4);
        dequant_strategy_broadcast_w_dct8::<R>(
            self.client_ref(),
            quant.handle.clone(),
            h_w,
            h_q,
            h_o.clone(),
            quant.num_blocks,
            quant.coeffs_per_block,
            channel_bias,
        );
        GpuBlocks {
            handle: h_o,
            num_blocks: quant.num_blocks,
            coeffs_per_block: quant.coeffs_per_block,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent-API 3-channel DCT8 dequant. Takes quantized i32
    /// blocks for each channel + per-coefficient weights + per-block
    /// scale + CfL factors. Returns 3 `GpuBlocks` (X, Y, B) of
    /// dequantized f32 coefficients.
    ///
    /// All inputs live on GPU; outputs also stay GPU-resident.
    /// `x_factor` / `b_factor` are host slices (per-block CfL factors,
    /// length `num_blocks`).
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_dct8_persistent(
        &self,
        quant_x: &GpuI32Blocks<R>,
        quant_y: &GpuI32Blocks<R>,
        quant_b: &GpuI32Blocks<R>,
        weights_x: &GpuBlocks<R>,
        weights_y: &GpuBlocks<R>,
        weights_b: &GpuBlocks<R>,
        qac_qm_x: &[f32],
        qac_qm_y: &[f32],
        qac_qm_b: &[f32],
        x_factor: &[f32],
        b_factor: &[f32],
    ) -> (GpuBlocks<R>, GpuBlocks<R>, GpuBlocks<R>) {
        let nb = quant_x.num_blocks;
        assert_eq!(quant_y.num_blocks, nb);
        assert_eq!(quant_b.num_blocks, nb);
        for q in [quant_x, quant_y, quant_b] {
            assert_eq!(q.coeffs_per_block, 64);
        }
        for w in [weights_x, weights_y, weights_b] {
            assert_eq!(w.coeffs_per_block, 64);
            assert_eq!(w.num_blocks, nb);
        }
        for s in [qac_qm_x, qac_qm_y, qac_qm_b, x_factor, b_factor] {
            assert_eq!(s.len() as u32, nb);
        }
        let n = (nb as usize) * 64;
        let h_qmx = self.client_ref().create_from_slice(f32::as_bytes(qac_qm_x));
        let h_qmy = self.client_ref().create_from_slice(f32::as_bytes(qac_qm_y));
        let h_qmb = self.client_ref().create_from_slice(f32::as_bytes(qac_qm_b));
        let h_xf = self.client_ref().create_from_slice(f32::as_bytes(x_factor));
        let h_bf = self.client_ref().create_from_slice(f32::as_bytes(b_factor));
        let h_ox = self.client_ref().empty(n * 4);
        let h_oy = self.client_ref().empty(n * 4);
        let h_ob = self.client_ref().empty(n * 4);
        dequant_dct8::<R>(
            self.client_ref(),
            quant_x.handle.clone(),
            quant_y.handle.clone(),
            quant_b.handle.clone(),
            weights_x.handle.clone(),
            weights_y.handle.clone(),
            weights_b.handle.clone(),
            h_qmx,
            h_qmy,
            h_qmb,
            h_xf,
            h_bf,
            h_ox.clone(),
            h_oy.clone(),
            h_ob.clone(),
            nb,
        );
        let mk = |h| GpuBlocks {
            handle: h,
            num_blocks: nb,
            coeffs_per_block: 64,
            _r: core::marker::PhantomData,
        };
        (mk(h_ox), mk(h_oy), mk(h_ob))
    }

    /// Broadcast-weights variant of [`Self::dequant_dct8_persistent`].
    /// Each `weights_*` must be a `GpuBlocks` with `num_blocks == 1` and
    /// `coeffs_per_block == 64`. Algorithmically identical to the
    /// per-block variant when called with replicated weights; saves the
    /// per-block weight-buffer storage and replication.
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_dct8_persistent_broadcast_w(
        &self,
        quant_x: &GpuI32Blocks<R>,
        quant_y: &GpuI32Blocks<R>,
        quant_b: &GpuI32Blocks<R>,
        weights_x: &GpuBlocks<R>,
        weights_y: &GpuBlocks<R>,
        weights_b: &GpuBlocks<R>,
        qac_qm_x: &[f32],
        qac_qm_y: &[f32],
        qac_qm_b: &[f32],
        x_factor: &[f32],
        b_factor: &[f32],
    ) -> (GpuBlocks<R>, GpuBlocks<R>, GpuBlocks<R>) {
        let nb = quant_x.num_blocks;
        assert_eq!(quant_y.num_blocks, nb);
        assert_eq!(quant_b.num_blocks, nb);
        for q in [quant_x, quant_y, quant_b] {
            assert_eq!(q.coeffs_per_block, 64);
        }
        for w in [weights_x, weights_y, weights_b] {
            assert_eq!(w.coeffs_per_block, 64);
            assert_eq!(w.num_blocks, 1, "broadcast_w expects 1-block templates");
        }
        for s in [qac_qm_x, qac_qm_y, qac_qm_b, x_factor, b_factor] {
            assert_eq!(s.len() as u32, nb);
        }
        let n = (nb as usize) * 64;
        let h_qmx = self.client_ref().create_from_slice(f32::as_bytes(qac_qm_x));
        let h_qmy = self.client_ref().create_from_slice(f32::as_bytes(qac_qm_y));
        let h_qmb = self.client_ref().create_from_slice(f32::as_bytes(qac_qm_b));
        let h_xf = self.client_ref().create_from_slice(f32::as_bytes(x_factor));
        let h_bf = self.client_ref().create_from_slice(f32::as_bytes(b_factor));
        let h_ox = self.client_ref().empty(n * 4);
        let h_oy = self.client_ref().empty(n * 4);
        let h_ob = self.client_ref().empty(n * 4);
        dequant_dct8_broadcast_w::<R>(
            self.client_ref(),
            quant_x.handle.clone(),
            quant_y.handle.clone(),
            quant_b.handle.clone(),
            weights_x.handle.clone(),
            weights_y.handle.clone(),
            weights_b.handle.clone(),
            h_qmx,
            h_qmy,
            h_qmb,
            h_xf,
            h_bf,
            h_ox.clone(),
            h_oy.clone(),
            h_ob.clone(),
            nb,
        );
        let mk = |h| GpuBlocks {
            handle: h,
            num_blocks: nb,
            coeffs_per_block: 64,
            _r: core::marker::PhantomData,
        };
        (mk(h_ox), mk(h_oy), mk(h_ob))
    }

    /// GPU spatial-plane → per-block gather. Reshapes a `GpuPlane`
    /// (W×H spatial layout) into a `GpuBlocks` of `num_blocks × tile_w
    /// × tile_h` floats, with raster-grid block ordering.
    ///
    /// Plane dimensions must be exact multiples of `tile_w` × `tile_h`
    /// (no fractional/edge blocks). For non-multiple sizes, pad the
    /// plane first via [`Self::pad_plane_persistent`].
    ///
    /// Replaces the host-side `download_plane` + manual gather +
    /// `upload_blocks` round-trip used by the lossy_roundtrip_persistent
    /// example, keeping all data on-GPU through the spatial→per-block
    /// boundary.
    pub fn gather_blocks_persistent(
        &self,
        plane: &GpuPlane<R>,
        tile_w: u32,
        tile_h: u32,
    ) -> GpuBlocks<R> {
        assert!(
            plane.width.is_multiple_of(tile_w),
            "plane width {} not multiple of tile_w {tile_w}",
            plane.width
        );
        assert!(
            plane.height.is_multiple_of(tile_h),
            "plane height {} not multiple of tile_h {tile_h}",
            plane.height
        );
        let blocks_per_row = plane.width / tile_w;
        let blocks_per_col = plane.height / tile_h;
        let num_blocks = blocks_per_row * blocks_per_col;
        let coeffs_per_block = tile_w * tile_h;
        let n_out = (num_blocks as usize) * (coeffs_per_block as usize);
        let h_out = self.client_ref().empty(n_out * 4);
        gather_blocks::<R>(
            self.client_ref(),
            plane.handle.clone(),
            h_out.clone(),
            plane.n_pixels(),
            n_out,
            plane.width,
            blocks_per_row,
            tile_w,
            tile_h,
        );
        GpuBlocks {
            handle: h_out,
            num_blocks,
            coeffs_per_block,
            _r: core::marker::PhantomData,
        }
    }

    /// Inverse of [`Self::gather_blocks_persistent`]: scatter a per-block
    /// buffer back into a spatial `GpuPlane`. Reverses the layout
    /// transformation; useful for putting reconstructed IDCT output
    /// back into a plane shape.
    pub fn scatter_blocks_persistent(
        &self,
        blocks: &GpuBlocks<R>,
        width: u32,
        height: u32,
        tile_w: u32,
        tile_h: u32,
    ) -> GpuPlane<R> {
        assert!(width.is_multiple_of(tile_w));
        assert!(height.is_multiple_of(tile_h));
        assert_eq!(blocks.coeffs_per_block, tile_w * tile_h);
        let blocks_per_row = width / tile_w;
        let blocks_per_col = height / tile_h;
        assert_eq!(blocks.num_blocks, blocks_per_row * blocks_per_col);
        let n_plane = (width as usize) * (height as usize);
        let h_out = self.client_ref().empty(n_plane * 4);
        scatter_blocks::<R>(
            self.client_ref(),
            blocks.handle.clone(),
            h_out.clone(),
            blocks.total_floats(),
            n_plane,
            width,
            blocks_per_row,
            tile_w,
            tile_h,
        );
        GpuPlane {
            handle: h_out,
            width,
            height,
            _r: core::marker::PhantomData,
        }
    }

    /// Restore DC values (slot 0 of each block) from `src` into `dst`,
    /// leaving all AC coefficients in `dst` untouched. Both inputs
    /// must have the same shape.
    ///
    /// Use case: the GPU `quantize_dct8` kernel always zeros DC (the
    /// real encoder uses dc_coding for DC). After dequant, callers
    /// can bit-exact-restore DC from the forward DCT output by
    /// calling this method with `src = forward_dct_output, dst =
    /// dequantized_output`. Replaces the host-side download +
    /// fixup + re-upload pattern.
    ///
    /// `dst` is mutated in place via the existing GPU buffer; the
    /// caller's `&GpuBlocks` reference is unchanged after this call.
    pub fn restore_dc_persistent(&self, src: &GpuBlocks<R>, dst: &GpuBlocks<R>) {
        assert_eq!(src.coeffs_per_block, dst.coeffs_per_block);
        assert_eq!(src.num_blocks, dst.num_blocks);
        restore_dc::<R>(
            self.client_ref(),
            src.handle.clone(),
            dst.handle.clone(),
            src.coeffs_per_block,
            src.num_blocks,
        );
    }

    /// Persistent-API indexed LLF restore for DCT64×64 (8×8 DC-grid →
    /// 64 LLF positions per block at coeffs[iy * 64 + ix] for iy, ix
    /// in 0..8). GPU equivalent of `dispatch_restore_llf` for DCT64×64.
    ///
    /// `dst.coeffs_per_block` must be 4096 (DCT64x64 standard size).
    pub fn set_llf_dct64x64_indexed_persistent(
        &self,
        dc_grid: &GpuBlocks<R>,
        coords: &[(u32, u32)],
        dst: &GpuBlocks<R>,
        dc_stride: u32,
    ) {
        assert_eq!(dc_grid.coeffs_per_block, 1);
        assert_eq!(
            dst.coeffs_per_block, 4096,
            "dst.coeffs_per_block must be 4096 for DCT64x64; got {}",
            dst.coeffs_per_block
        );
        assert_eq!(coords.len() as u32, dst.num_blocks);
        let mut flat: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(coords.len() * 2);
        for &(bx, by) in coords {
            flat.push(bx);
            flat.push(by);
        }
        let h_coords = self.client_ref().create_from_slice(u32::as_bytes(&flat));
        crate::launch::set_llf::set_llf_dct64x64_indexed::<R>(
            self.client_ref(),
            dc_grid.handle.clone(),
            h_coords,
            dst.handle.clone(),
            dc_grid.num_blocks as usize,
            dst.total_floats(),
            dc_stride,
            dst.coeffs_per_block,
            dst.num_blocks,
        );
    }

    /// Persistent-API indexed LLF restore for DCT64×32 (8×4 DC-grid →
    /// 32 LLF positions per block at coeffs[iy * 64 + ix] for iy in 0..4,
    /// ix in 0..8). GPU equivalent of `dispatch_restore_llf` for DCT64×32.
    ///
    /// `dst.coeffs_per_block` must be 2048 (DCT64x32 standard size).
    pub fn set_llf_dct64x32_indexed_persistent(
        &self,
        dc_grid: &GpuBlocks<R>,
        coords: &[(u32, u32)],
        dst: &GpuBlocks<R>,
        dc_stride: u32,
    ) {
        assert_eq!(dc_grid.coeffs_per_block, 1);
        assert_eq!(
            dst.coeffs_per_block, 2048,
            "dst.coeffs_per_block must be 2048 for DCT64x32; got {}",
            dst.coeffs_per_block
        );
        assert_eq!(coords.len() as u32, dst.num_blocks);
        let mut flat: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(coords.len() * 2);
        for &(bx, by) in coords {
            flat.push(bx);
            flat.push(by);
        }
        let h_coords = self.client_ref().create_from_slice(u32::as_bytes(&flat));
        crate::launch::set_llf::set_llf_dct64x32_indexed::<R>(
            self.client_ref(),
            dc_grid.handle.clone(),
            h_coords,
            dst.handle.clone(),
            dc_grid.num_blocks as usize,
            dst.total_floats(),
            dc_stride,
            dst.coeffs_per_block,
            dst.num_blocks,
        );
    }

    /// Persistent-API indexed LLF restore for DCT32×64 (4×8 DC-grid →
    /// 32 LLF positions per block at coeffs[iy * 64 + ix] for iy in 0..4,
    /// ix in 0..8). GPU equivalent of `dispatch_restore_llf` for DCT32×64.
    ///
    /// `dst.coeffs_per_block` must be 2048 (DCT32x64 standard size).
    pub fn set_llf_dct32x64_indexed_persistent(
        &self,
        dc_grid: &GpuBlocks<R>,
        coords: &[(u32, u32)],
        dst: &GpuBlocks<R>,
        dc_stride: u32,
    ) {
        assert_eq!(dc_grid.coeffs_per_block, 1);
        assert_eq!(
            dst.coeffs_per_block, 2048,
            "dst.coeffs_per_block must be 2048 for DCT32x64; got {}",
            dst.coeffs_per_block
        );
        assert_eq!(coords.len() as u32, dst.num_blocks);
        let mut flat: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(coords.len() * 2);
        for &(bx, by) in coords {
            flat.push(bx);
            flat.push(by);
        }
        let h_coords = self.client_ref().create_from_slice(u32::as_bytes(&flat));
        crate::launch::set_llf::set_llf_dct32x64_indexed::<R>(
            self.client_ref(),
            dc_grid.handle.clone(),
            h_coords,
            dst.handle.clone(),
            dc_grid.num_blocks as usize,
            dst.total_floats(),
            dc_stride,
            dst.coeffs_per_block,
            dst.num_blocks,
        );
    }

    /// Persistent-API indexed LLF restore for DCT32×16 (4×2 DC-grid →
    /// 8 LLF positions per block at coeffs[iy * 32 + ix] for iy in 0..2,
    /// ix in 0..4). GPU equivalent of `dispatch_restore_llf` for DCT32×16.
    ///
    /// `dst.coeffs_per_block` must be 512 (DCT32x16 standard size).
    pub fn set_llf_dct32x16_indexed_persistent(
        &self,
        dc_grid: &GpuBlocks<R>,
        coords: &[(u32, u32)],
        dst: &GpuBlocks<R>,
        dc_stride: u32,
    ) {
        assert_eq!(dc_grid.coeffs_per_block, 1);
        assert_eq!(
            dst.coeffs_per_block, 512,
            "dst.coeffs_per_block must be 512 for DCT32x16; got {}",
            dst.coeffs_per_block
        );
        assert_eq!(coords.len() as u32, dst.num_blocks);
        let mut flat: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(coords.len() * 2);
        for &(bx, by) in coords {
            flat.push(bx);
            flat.push(by);
        }
        let h_coords = self.client_ref().create_from_slice(u32::as_bytes(&flat));
        crate::launch::set_llf::set_llf_dct32x16_indexed::<R>(
            self.client_ref(),
            dc_grid.handle.clone(),
            h_coords,
            dst.handle.clone(),
            dc_grid.num_blocks as usize,
            dst.total_floats(),
            dc_stride,
            dst.coeffs_per_block,
            dst.num_blocks,
        );
    }

    /// Persistent-API indexed LLF restore for DCT16×32 (2×4 DC-grid →
    /// 8 LLF positions per block at coeffs[iy * 32 + ix] for iy in 0..2,
    /// ix in 0..4). GPU equivalent of `dispatch_restore_llf` for DCT16×32.
    ///
    /// `dst.coeffs_per_block` must be 512 (DCT16x32 standard size).
    pub fn set_llf_dct16x32_indexed_persistent(
        &self,
        dc_grid: &GpuBlocks<R>,
        coords: &[(u32, u32)],
        dst: &GpuBlocks<R>,
        dc_stride: u32,
    ) {
        assert_eq!(dc_grid.coeffs_per_block, 1);
        assert_eq!(
            dst.coeffs_per_block, 512,
            "dst.coeffs_per_block must be 512 for DCT16x32; got {}",
            dst.coeffs_per_block
        );
        assert_eq!(coords.len() as u32, dst.num_blocks);
        let mut flat: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(coords.len() * 2);
        for &(bx, by) in coords {
            flat.push(bx);
            flat.push(by);
        }
        let h_coords = self.client_ref().create_from_slice(u32::as_bytes(&flat));
        crate::launch::set_llf::set_llf_dct16x32_indexed::<R>(
            self.client_ref(),
            dc_grid.handle.clone(),
            h_coords,
            dst.handle.clone(),
            dc_grid.num_blocks as usize,
            dst.total_floats(),
            dc_stride,
            dst.coeffs_per_block,
            dst.num_blocks,
        );
    }

    /// Persistent-API indexed LLF restore for DCT32×32 (4×4 DC-grid →
    /// 16 LLF positions per block at coeffs[iy * 32 + ix] for iy, ix
    /// in 0..4). GPU equivalent of the `RAW_STRATEGY_DCT32X32` arm of
    /// `forks::reconstruct::dispatch_restore_llf`.
    ///
    /// `dst.coeffs_per_block` must be 1024 (DCT32x32 standard size).
    /// AC coefficients (the 1008 positions outside the 4×4 LLF region)
    /// untouched.
    pub fn set_llf_dct32x32_indexed_persistent(
        &self,
        dc_grid: &GpuBlocks<R>,
        coords: &[(u32, u32)],
        dst: &GpuBlocks<R>,
        dc_stride: u32,
    ) {
        assert_eq!(
            dc_grid.coeffs_per_block, 1,
            "dc_grid must have coeffs_per_block=1; got {}",
            dc_grid.coeffs_per_block
        );
        assert_eq!(
            dst.coeffs_per_block, 1024,
            "dst.coeffs_per_block must be 1024 for DCT32x32; got {}",
            dst.coeffs_per_block
        );
        assert_eq!(coords.len() as u32, dst.num_blocks);
        let mut flat: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(coords.len() * 2);
        for &(bx, by) in coords {
            flat.push(bx);
            flat.push(by);
        }
        let h_coords = self.client_ref().create_from_slice(u32::as_bytes(&flat));
        crate::launch::set_llf::set_llf_dct32x32_indexed::<R>(
            self.client_ref(),
            dc_grid.handle.clone(),
            h_coords,
            dst.handle.clone(),
            dc_grid.num_blocks as usize,
            dst.total_floats(),
            dc_stride,
            dst.coeffs_per_block,
            dst.num_blocks,
        );
    }

    /// Persistent-API indexed LLF restore for DCT16×16 (2×2 DC-grid →
    /// 4 LLF positions per block at coeffs[0], [1], [16], [17]). GPU
    /// equivalent of the `RAW_STRATEGY_DCT16X16` arm of
    /// `forks::reconstruct::dispatch_restore_llf`.
    ///
    /// `dst.coeffs_per_block` must be 256 (DCT16x16 standard size).
    /// AC coefficients (positions other than [0], [1], [16], [17])
    /// untouched.
    pub fn set_llf_dct16x16_indexed_persistent(
        &self,
        dc_grid: &GpuBlocks<R>,
        coords: &[(u32, u32)],
        dst: &GpuBlocks<R>,
        dc_stride: u32,
    ) {
        assert_eq!(
            dc_grid.coeffs_per_block, 1,
            "dc_grid must have coeffs_per_block=1; got {}",
            dc_grid.coeffs_per_block
        );
        assert_eq!(
            dst.coeffs_per_block, 256,
            "dst.coeffs_per_block must be 256 for DCT16x16; got {}",
            dst.coeffs_per_block
        );
        assert_eq!(coords.len() as u32, dst.num_blocks);
        let mut flat: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(coords.len() * 2);
        for &(bx, by) in coords {
            flat.push(bx);
            flat.push(by);
        }
        let h_coords = self.client_ref().create_from_slice(u32::as_bytes(&flat));
        crate::launch::set_llf::set_llf_dct16x16_indexed::<R>(
            self.client_ref(),
            dc_grid.handle.clone(),
            h_coords,
            dst.handle.clone(),
            dc_grid.num_blocks as usize,
            dst.total_floats(),
            dc_stride,
            dst.coeffs_per_block,
            dst.num_blocks,
        );
    }

    /// Persistent-API indexed LLF restore for DCT16×8 / DCT8×16
    /// strategies (1×2 or 2×1 DC-grid → 2 LLF positions per block).
    /// GPU equivalent of the `RAW_STRATEGY_DCT16X8 | DCT8X16` arm of
    /// `forks::reconstruct::dispatch_restore_llf`.
    ///
    /// `dc_step` differentiates the two strategies:
    /// - DCT16×8 (vertical pair, dc_grid[(by, bx), (by+1, bx)]):
    ///   `dc_step = dc_stride`.
    /// - DCT8×16 (horizontal pair, dc_grid[(by, bx), (by, bx+1)]):
    ///   `dc_step = 1`.
    ///
    /// Writes only positions [0] and [1] of each block; AC coefficients
    /// untouched. `dst.coeffs_per_block` must be ≥ 2 (typically 128 for
    /// these strategies).
    pub fn set_llf_dct16x8_or_8x16_indexed_persistent(
        &self,
        dc_grid: &GpuBlocks<R>,
        coords: &[(u32, u32)],
        dst: &GpuBlocks<R>,
        dc_stride: u32,
        dc_step: u32,
    ) {
        assert_eq!(
            dc_grid.coeffs_per_block, 1,
            "dc_grid must have coeffs_per_block=1; got {}",
            dc_grid.coeffs_per_block
        );
        assert!(
            dst.coeffs_per_block >= 2,
            "dst.coeffs_per_block must be >= 2; got {}",
            dst.coeffs_per_block
        );
        assert_eq!(coords.len() as u32, dst.num_blocks);
        let mut flat: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(coords.len() * 2);
        for &(bx, by) in coords {
            flat.push(bx);
            flat.push(by);
        }
        let h_coords = self.client_ref().create_from_slice(u32::as_bytes(&flat));
        crate::launch::set_llf::set_llf_dct16x8_or_8x16_indexed::<R>(
            self.client_ref(),
            dc_grid.handle.clone(),
            h_coords,
            dst.handle.clone(),
            dc_grid.num_blocks as usize,
            dst.total_floats(),
            dc_stride,
            dc_step,
            dst.coeffs_per_block,
            dst.num_blocks,
        );
    }

    /// Persistent-API indexed DC-from-grid set: writes one DC value
    /// per block into position 0 of each output coefficient block,
    /// looking the value up from a per-(8×8)-block scalar plane.
    ///
    /// Mirrors the host loop inside
    /// `forks::reconstruct::dispatch_restore_llf` for the 1×1 LLF
    /// case (DCT8 / DCT4x4 / DCT4x8 / DCT8x4 / IDENTITY / DCT2x2 +
    /// AFV0-3 — every strategy whose LLF rectangle is a single value).
    ///
    /// `dc_grid` is the per-(8×8) block scalar plane (e.g. output of
    /// [`Self::dc_grid_8x8_persistent`]); `dc_stride` is its xsize
    /// (`padded_width / 8`). `coords` lists the (bx, by) of each
    /// block in the strategy's batch (raster order). `dst` is the
    /// per-block coefficient buffer to mutate (only position 0 of
    /// each block is touched; AC values untouched).
    ///
    /// `dst.coeffs_per_block` must be ≥ 1; the strategy's natural
    /// coefficient count works (64 for DCT8/DCT4-family/IDENTITY/DCT2x2,
    /// also 64 for AFV).
    pub fn set_dc_from_grid_indexed_persistent(
        &self,
        dc_grid: &GpuBlocks<R>,
        coords: &[(u32, u32)],
        dst: &GpuBlocks<R>,
        dc_stride: u32,
    ) {
        assert_eq!(
            dc_grid.coeffs_per_block, 1,
            "dc_grid must have coeffs_per_block=1; got {}",
            dc_grid.coeffs_per_block
        );
        assert_eq!(coords.len() as u32, dst.num_blocks);
        let mut flat: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(coords.len() * 2);
        for &(bx, by) in coords {
            flat.push(bx);
            flat.push(by);
        }
        let h_coords = self.client_ref().create_from_slice(u32::as_bytes(&flat));
        crate::launch::set_dc::set_dc_from_grid_indexed::<R>(
            self.client_ref(),
            dc_grid.handle.clone(),
            h_coords,
            dst.handle.clone(),
            dc_grid.num_blocks as usize,
            dst.total_floats(),
            dc_stride,
            dst.coeffs_per_block,
            dst.num_blocks,
        );
    }

    /// Persistent-API indexed plane-to-blocks gather. Pulls a sparse
    /// subset of `(bx, by)` 8×8-block-grid positions (each spanning
    /// `tile_w × tile_h` pixels) from `plane` into a contiguous
    /// `GpuBlocks` of `n_blocks * (tile_w * tile_h)` floats.
    ///
    /// Differs from [`Self::gather_blocks_persistent`] (which gathers
    /// the FULL raster grid) — this variant lets the caller batch only
    /// the blocks one AC strategy was assigned to.
    ///
    /// `coords` is a flat host slice in `[bx_0, by_0, bx_1, by_1, …]`
    /// order. Each `(bx, by)` is in 8×8-block-grid units; the kernel
    /// multiplies by 8 to compute the pixel-space top-left of the
    /// strategy's tile. Plane access is bounds-unchecked — caller MUST
    /// ensure every `(bx*8 + tile_w, by*8 + tile_h)` falls inside
    /// `(plane.width, plane.height)`.
    ///
    /// Used by the mixed-strategy reconstruct path to skip the host
    /// `extend_from_slice` per-strategy gather + the subsequent
    /// per-strategy `upload_blocks` PCIe round-trip.
    pub fn indexed_gather_blocks_persistent(
        &self,
        plane: &GpuPlane<R>,
        coords: &[(u32, u32)],
        tile_w: u32,
        tile_h: u32,
    ) -> GpuBlocks<R> {
        let n_blocks = coords.len() as u32;
        let coeffs_per_block = tile_w * tile_h;
        let n_out = (n_blocks as usize) * (coeffs_per_block as usize);
        // Flatten coords to interleaved [bx, by, bx, by, …] u32 layout.
        let mut flat: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(coords.len() * 2);
        for &(bx, by) in coords {
            flat.push(bx);
            flat.push(by);
        }
        let h_coords = self.client_ref().create_from_slice(u32::as_bytes(&flat));
        let h_out = self.client_ref().empty(n_out * 4);
        crate::launch::indexed_gather::indexed_gather::<R>(
            self.client_ref(),
            plane.handle.clone(),
            h_coords,
            h_out.clone(),
            plane.width,
            plane.n_pixels(),
            n_blocks,
            tile_w,
            tile_h,
        );
        GpuBlocks {
            handle: h_out,
            num_blocks: n_blocks,
            coeffs_per_block,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent-API indexed per-block-buffer → spatial-plane scatter.
    /// Inverse of [`Self::indexed_gather_blocks_persistent`]: takes
    /// batched per-block pixels and writes them back into `plane` at
    /// each `(bx, by)`'s tile rectangle.
    ///
    /// `coords` is the same `[(bx, by)]` shape used by indexed_gather
    /// (each in 8×8-block-grid units). Tiles outside any covered
    /// `(bx, by)` are untouched in `plane`.
    ///
    /// `blocks.coeffs_per_block` must equal `tile_w * tile_h`. Plane
    /// access is bounds-unchecked — caller MUST ensure every
    /// `(bx*8 + tile_w, by*8 + tile_h)` falls inside
    /// `(plane.width, plane.height)`.
    ///
    /// Used by the GPU-resident mixed-strategy reconstruct path to
    /// scatter per-strategy IDCT outputs back into a destination
    /// GpuPlane without the host-side `scatter_block_to_plane` loop +
    /// `apply_idct_batch_gpu`'s implicit download.
    pub fn indexed_scatter_blocks_persistent(
        &self,
        blocks: &GpuBlocks<R>,
        coords: &[(u32, u32)],
        plane: &GpuPlane<R>,
        tile_w: u32,
        tile_h: u32,
    ) {
        assert_eq!(blocks.num_blocks, coords.len() as u32);
        assert_eq!(blocks.coeffs_per_block, tile_w * tile_h);
        let mut flat: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(coords.len() * 2);
        for &(bx, by) in coords {
            flat.push(bx);
            flat.push(by);
        }
        let h_coords = self.client_ref().create_from_slice(u32::as_bytes(&flat));
        crate::launch::indexed_scatter::indexed_scatter::<R>(
            self.client_ref(),
            blocks.handle.clone(),
            h_coords,
            plane.handle.clone(),
            plane.width,
            plane.n_pixels(),
            blocks.num_blocks,
            tile_w,
            tile_h,
        );
    }

    /// Persistent-API per-(8×8)-block DC grid: one f32 per block =
    /// mean of the 64 covered pixels. Mirrors
    /// `crate::forks::reconstruct::compute_dc_grid_per_8x8_block`
    /// (host scalar) on GPU.
    ///
    /// Plane dimensions must be multiples of 8. Returns a `GpuBlocks`
    /// with `coeffs_per_block = 1` and `num_blocks = (W/8) * (H/8)`.
    pub fn dc_grid_8x8_persistent(&self, plane: &GpuPlane<R>) -> GpuBlocks<R> {
        assert!(
            plane.width.is_multiple_of(8),
            "dc_grid_8x8_persistent: width {} not multiple of 8",
            plane.width
        );
        assert!(
            plane.height.is_multiple_of(8),
            "dc_grid_8x8_persistent: height {} not multiple of 8",
            plane.height
        );
        let blocks_per_row = plane.width / 8;
        let blocks_per_col = plane.height / 8;
        let num_blocks = blocks_per_row * blocks_per_col;
        let h_out = self.client_ref().empty((num_blocks as usize) * 4);
        crate::launch::dc_grid::dc_grid_8x8::<R>(
            self.client_ref(),
            plane.handle.clone(),
            h_out.clone(),
            plane.width,
            plane.height,
        );
        GpuBlocks {
            handle: h_out,
            num_blocks,
            coeffs_per_block: 1,
            _r: core::marker::PhantomData,
        }
    }

    /// Persistent-API mask1x1 field on the Y channel.
    pub fn mask1x1_persistent(&self, y: &GpuPlane<R>) -> GpuPlane<R> {
        let n = y.n_pixels();
        let h_out = self.client_ref().empty(n * 4);
        mask1x1::<R>(
            self.client_ref(),
            y.handle.clone(),
            h_out.clone(),
            y.width,
            y.height,
        );
        GpuPlane {
            handle: h_out,
            width: y.width,
            height: y.height,
            _r: core::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `histogram_count_pow2` must match the trivial host-side
    /// `for tok in tokens { hist[(tok & mask) as usize] += 1 }`
    /// over a non-trivial token stream including out-of-range values
    /// (which the kernel masks back into range).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_histogram_count_pow2_matches_host() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();

        for &bucket_count in &[16u32, 64, 256, 1024] {
            let mask = bucket_count - 1;
            // Mix of small in-range tokens, repeated tokens, and OOB
            // tokens that exercise the mask.
            let n: usize = 10_000;
            let tokens: Vec<u32> = (0..n)
                .map(|i| {
                    let raw = (i as u32).wrapping_mul(2654435761).wrapping_add(13);
                    // Sometimes pass an in-range value, sometimes OOB
                    // to exercise the mask.
                    if i % 3 == 0 {
                        raw % bucket_count
                    } else {
                        raw
                    }
                })
                .collect();

            let mut host_hist = vec![0u32; bucket_count as usize];
            for &t in &tokens {
                host_hist[(t & mask) as usize] += 1;
            }
            let gpu_hist = enc.histogram_count_pow2(&tokens, bucket_count);

            assert_eq!(
                gpu_hist.len(),
                bucket_count as usize,
                "GPU histogram length"
            );
            assert_eq!(
                gpu_hist, host_hist,
                "histogram mismatch at bucket_count={bucket_count}"
            );

            let total_gpu: u32 = gpu_hist.iter().sum();
            assert_eq!(
                total_gpu as usize, n,
                "total count must equal n_tokens at bucket_count={bucket_count}"
            );
        }
    }

    /// Empty-input edge case: zero tokens → all-zero histogram, no panic.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_histogram_count_pow2_empty() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let tokens: Vec<u32> = Vec::new();
        let hist = enc.histogram_count_pow2(&tokens, 64);
        assert_eq!(hist.len(), 64);
        assert!(hist.iter().all(|&c| c == 0));
    }

    /// Non-power-of-2 bucket_count must panic at the launcher (not
    /// silently produce wrong results).
    #[cfg(feature = "cuda")]
    #[test]
    #[should_panic(expected = "must be a power of 2")]
    fn test_histogram_count_pow2_rejects_non_pow2() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let tokens = vec![0u32; 10];
        let _ = enc.histogram_count_pow2(&tokens, 100);
    }

    /// `sse_reduce_3channel` GPU kernel must match the trivial host
    /// `for i in 0..64 { sum += (dx² + dy² + db²) * mask }` reduction
    /// per block. Bit-exact within fp32 rounding (1e-4 tolerance).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_sse_reduce_3channel_matches_host() {
        use cubecl::prelude::*;
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n_blocks = 23_usize; // odd, exercises tail thread
        let n = n_blocks * 64;
        // Synthesize per-pixel data with non-trivial pattern.
        let orig_x: Vec<f32> = (0..n).map(|i| ((i * 7) as f32).sin() * 16.0 + 64.0).collect();
        let orig_y: Vec<f32> = (0..n).map(|i| ((i * 11) as f32).sin() * 24.0 + 80.0).collect();
        let orig_b: Vec<f32> = (0..n).map(|i| ((i * 13) as f32).sin() * 12.0 + 48.0).collect();
        let recon_x: Vec<f32> = orig_x.iter().enumerate().map(|(i, &v)| v + ((i * 3) as f32 * 0.1).sin()).collect();
        let recon_y: Vec<f32> = orig_y.iter().enumerate().map(|(i, &v)| v + ((i * 5) as f32 * 0.1).sin()).collect();
        let recon_b: Vec<f32> = orig_b.iter().enumerate().map(|(i, &v)| v + ((i * 17) as f32 * 0.1).sin()).collect();
        let mask: Vec<f32> = (0..n).map(|i| 0.5 + 0.4 * ((i as f32 * 0.07).cos())).collect();

        // Host reference.
        let mut host_costs = vec![0.0_f32; n_blocks];
        for b in 0..n_blocks {
            let mut s = 0.0_f32;
            let r0 = b * 64;
            for i in 0..64 {
                let dx = orig_x[r0 + i] - recon_x[r0 + i];
                let dy = orig_y[r0 + i] - recon_y[r0 + i];
                let db = orig_b[r0 + i] - recon_b[r0 + i];
                let m = mask[r0 + i];
                s += (dx * dx + dy * dy + db * db) * m;
            }
            host_costs[b] = s;
        }

        // GPU run.
        let client = enc.client_ref();
        let h_ox = client.create_from_slice(f32::as_bytes(&orig_x));
        let h_oy = client.create_from_slice(f32::as_bytes(&orig_y));
        let h_ob = client.create_from_slice(f32::as_bytes(&orig_b));
        let h_rx = client.create_from_slice(f32::as_bytes(&recon_x));
        let h_ry = client.create_from_slice(f32::as_bytes(&recon_y));
        let h_rb = client.create_from_slice(f32::as_bytes(&recon_b));
        let h_m = client.create_from_slice(f32::as_bytes(&mask));
        let h_out = client.empty(n_blocks * 4);
        crate::launch::sse_reduce::sse_reduce_3channel::<B>(
            client, h_ox, h_oy, h_ob, h_rx, h_ry, h_rb, h_m, h_out.clone(), n_blocks as u32,
        );
        let bytes = client.read_one(h_out).expect("sse download");
        let gpu_costs: &[f32] = f32::from_bytes(&bytes);
        assert_eq!(gpu_costs.len(), n_blocks);
        for (b, (&h, &g)) in host_costs.iter().zip(gpu_costs.iter()).enumerate() {
            let rel = ((h - g) / h.max(1e-9)).abs();
            assert!(
                rel < 1e-4 || (h - g).abs() < 1e-4,
                "block {b} host={h:.6} gpu={g:.6} rel={rel:.6}"
            );
        }
    }

    /// Persistent dequant_strategy (no bias) must match the host-side
    /// `quant * weight / qac` formula bit-exactly within fp32 rounding.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_dequant_strategy_persistent_matches_host_formula() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        for &block_size in &[64u32, 128, 256, 1024] {
            let nb = 8u32;
            let total = (nb * block_size) as usize;
            let quant: Vec<i32> = (0..total).map(|i| (i as i32 % 13) - 6).collect();
            let weights: Vec<f32> = (0..block_size as usize).map(|i| 0.5 + 0.07 * i as f32).collect();
            let qac: Vec<f32> = (0..nb).map(|i| 0.5 + 0.1 * i as f32).collect();

            let g_q = GpuI32Blocks {
                handle: enc.client_ref().create_from_slice(i32::as_bytes(&quant)),
                num_blocks: nb,
                coeffs_per_block: block_size,
                _r: core::marker::PhantomData,
            };
            let g_d = enc.dequant_strategy_persistent(&g_q, &weights, &qac);
            let got = enc.download_blocks(&g_d);

            for b in 0..nb as usize {
                let inv_qac = 1.0_f32 / qac[b];
                let off = b * block_size as usize;
                for i in 0..block_size as usize {
                    let expected = (quant[off + i] as f32) * weights[i] * inv_qac;
                    let actual = got[off + i];
                    assert!(
                        (expected - actual).abs() < 1e-5,
                        "block_size={block_size} b={b} i={i}: got {actual}, want {expected}"
                    );
                }
            }
        }
    }

    /// Persistent dequant_strategy_dct8 must match the host
    /// `adjust_quant_bias(q, channel) * weight / qac` formula.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_dequant_strategy_dct8_persistent_matches_host() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let block_size = 64u32;
        let nb = 4u32;
        let total = (nb * block_size) as usize;
        let quant: Vec<i32> = (0..total).map(|i| (i as i32 % 11) - 5).collect();
        let weights: Vec<f32> = (0..block_size as usize).map(|i| 0.7 + 0.05 * i as f32).collect();
        let qac: Vec<f32> = (0..nb).map(|i| 0.6 + 0.15 * i as f32).collect();

        for channel in 0..3 {
            let g_q = GpuI32Blocks {
                handle: enc.client_ref().create_from_slice(i32::as_bytes(&quant)),
                num_blocks: nb,
                coeffs_per_block: block_size,
                _r: core::marker::PhantomData,
            };
            let g_d = enc.dequant_strategy_dct8_persistent(&g_q, &weights, &qac, channel);
            let got = enc.download_blocks(&g_d);

            for b in 0..nb as usize {
                let inv_qac = 1.0_f32 / qac[b];
                let off = b * block_size as usize;
                for i in 0..block_size as usize {
                    let biased = crate::forks::dequant::adjust_quant_bias(quant[off + i], channel);
                    let expected = biased * weights[i] * inv_qac;
                    let actual = got[off + i];
                    assert!(
                        (expected - actual).abs() < 1e-5,
                        "channel={channel} b={b} i={i}: got {actual}, want {expected}"
                    );
                }
            }
        }
    }

    /// Persistent quantize_large_blocks_broadcast_w must produce
    /// identical i32 output to the non-persistent variant on both
    /// DCT8 (8×8 grid, 1×1 LLF) and DCT16x16 (16×16 grid, 2×2 LLF)
    /// configurations.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_quantize_large_persistent_matches_non_persistent() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();

        for &(gw, gh, lx, ly) in &[(8u32, 8, 1u32, 1), (16, 16, 2, 2), (8, 16, 1, 2)] {
            let bs = (gw * gh) as usize;
            let nb = 8u32;
            let total = (nb as usize) * bs;
            let coeffs: Vec<f32> = (0..total).map(|i| 0.05 + 0.13 * (i as f32 * 0.07).sin()).collect();
            let weights: Vec<f32> = (0..bs).map(|i| 0.5 + 0.1 * i as f32).collect();
            let qac: Vec<f32> = (0..nb).map(|i| 0.7 + 0.1 * i as f32).collect();
            let thresholds = [0.62_f32, 0.62, 0.62, 0.62];

            let q_a = enc.quantize_large_blocks_broadcast_w(
                &coeffs, &weights, &qac, &thresholds, gw, gh, lx, ly,
            );

            let g_c = enc.upload_blocks(&coeffs, nb, (gw * gh) as u32);
            let g_q = enc.quantize_large_blocks_broadcast_w_persistent(
                &g_c, &weights, &qac, &thresholds, gw, gh, lx, ly,
            );
            // Download GpuI32Blocks; reuse client.
            let bytes = enc
                .client_ref()
                .read_one(g_q.handle.clone())
                .expect("read q persistent");
            let q_b: Vec<i32> = i32::from_bytes(&bytes).to_vec();

            assert_eq!(q_a.len(), q_b.len(), "(gw={gw}, gh={gh}) length mismatch");
            for (i, (a, b)) in q_a.iter().zip(&q_b).enumerate() {
                assert_eq!(
                    a, b,
                    "(gw={gw}, gh={gh}) q[{i}] differs: persistent={b} vs non={a}"
                );
            }
        }
    }

    /// Persistent entropy_coeffs_pixel_blocks_broadcast_w must produce
    /// stats and error_coeffs identical to the non-persistent variant.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_entropy_coeffs_persistent_matches_non_persistent() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 16_u32;
        let n = 64_u32;
        let nf = (nb as usize) * (n as usize);
        let block_c: Vec<f32> = (0..nf).map(|i| (i as f32 * 0.013).sin()).collect();
        let block_y: Vec<f32> = (0..nf).map(|i| (i as f32 * 0.017).cos() + 0.1).collect();
        let weights: Vec<f32> = (0..n).map(|i| 0.5 + 0.1 * i as f32).collect();
        let inv_w: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();
        let cmap_factor = 0.25;
        let quant = 0.7;
        let k_cost = 10.0;

        let (stats_a, err_a) = enc.entropy_coeffs_pixel_blocks_broadcast_w(
            &block_c, &block_y, &weights, &inv_w, n, cmap_factor, quant, k_cost,
        );

        let gc = enc.upload_blocks(&block_c, nb, n);
        let gy = enc.upload_blocks(&block_y, nb, n);
        let (gstats, gerr) = enc.entropy_coeffs_pixel_blocks_broadcast_w_persistent(
            &gc, &gy, &weights, &inv_w, cmap_factor, quant, k_cost,
        );
        let stats_b = enc.download_blocks(&gstats);
        let err_b = enc.download_blocks(&gerr);

        assert_eq!(stats_a.len(), stats_b.len(), "stats len mismatch");
        for (i, (a, b)) in stats_a.iter().zip(&stats_b).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "stats[{i}] differs: persistent={b} vs non={a}"
            );
        }
        for (i, (a, b)) in err_a.iter().zip(&err_b).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "err[{i}] differs: persistent={b} vs non={a}"
            );
        }
    }

    /// Persistent pixel_loss_blocks must match the non-persistent
    /// variant's f64 output.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_pixel_loss_persistent_matches_non_persistent() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let bw = 8_u32;
        let bh = 8_u32;
        let nb = 4_u32;
        let coeffs = (bw * bh) as usize;
        let pixel_err: Vec<f32> = (0..(nb as usize) * coeffs)
            .map(|i| 0.001 * (i as f32 * 0.07).sin())
            .collect();
        let mask_w = 64_u32;
        let mask_h = 32_u32;
        let mask: Vec<f32> = (0..(mask_w * mask_h) as usize)
            .map(|i| 0.5 + 0.1 * (i as f32 * 0.013).sin())
            .collect();
        // Each block reads mask[base..base+bw] for bh rows. Keep 4 starts
        // strictly inside the 64×32 mask (bw=8, bh=8 → end < 32).
        let mask_row_base: Vec<u32> = vec![0, 8, 8 * mask_w, 8 * mask_w + 16];
        let mask_offset = 0.05;

        let loss_a = enc.pixel_loss_blocks(
            &pixel_err,
            &mask,
            &mask_row_base,
            mask_w,
            mask_offset,
            bw,
            bh,
        );

        let gerr = enc.upload_blocks(&pixel_err, nb, bw * bh);
        let gplane = enc.upload_plane(&mask, mask_w, mask_h);
        let gloss = enc.pixel_loss_blocks_persistent(
            &gerr,
            &gplane,
            &mask_row_base,
            mask_offset,
            bw,
            bh,
        );
        let loss_b = enc.download_blocks_f64(&gloss);

        assert_eq!(loss_a.len(), loss_b.len());
        for (i, (a, b)) in loss_a.iter().zip(&loss_b).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "loss[{i}] differs: persistent={b} vs non={a}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_upload_download_roundtrip() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let plane = enc.upload_plane(&data, 16, 16);
        assert_eq!(plane.width(), 16);
        assert_eq!(plane.height(), 16);
        assert_eq!(plane.n_pixels(), 256);
        let back = enc.download_plane(&plane);
        assert_eq!(back, data);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_persistent_xyb_chain() {
        // Persistent XYB forward + gaborish on Y + mask1x1 on Y, with
        // only one upload (RGB) and one download (mask).
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n = 64 * 64;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();

        // Upload (no download yet).
        let g_r = enc.upload_plane(&r, 64, 64);
        let g_g = enc.upload_plane(&g, 64, 64);
        let g_b = enc.upload_plane(&b, 64, 64);

        // Pipeline.
        let (_xx, xy, _xb) = enc.xyb_from_linear_rgb_persistent(&g_r, &g_g, &g_b);
        let weights = GaborishWeights {
            wc: 1.5,
            wr: -0.1,
            wd: -0.05,
            w_big_r: 0.02,
            wl: 0.01,
            w_big_d: -0.005,
        };
        let xy_g = enc.gaborish_5x5_persistent(&xy, &weights);
        let mask = enc.mask1x1_persistent(&xy_g);

        // Download only the mask.
        let mask_host = enc.download_plane(&mask);
        assert_eq!(mask_host.len(), n);
        for &v in &mask_host {
            assert!(v.is_finite() && v > 0.0);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gab_smooth_persistent_uniform() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n = 16 * 16;
        let plane = enc.upload_plane(&vec![0.5_f32; n], 16, 16);
        // Symmetric 3×3 plus weights: center + 4*w1 + 4*w2 = 1 → uniform
        // input stays uniform.
        let w1 = 0.1;
        let w2 = 0.05;
        let wc = 1.0 - 4.0 * w1 - 4.0 * w2;
        let out = enc.gab_smooth_persistent(&plane, wc, w1, w2);
        let host = enc.download_plane(&out);
        for &v in &host {
            assert!((v - 0.5).abs() < 1e-5);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_pad_plane_persistent_dims() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let plane = enc.upload_plane(&vec![0.7_f32; 8 * 8], 8, 8);
        let padded = enc.pad_plane_persistent(&plane, 4);
        assert_eq!(padded.width(), 16);
        assert_eq!(padded.height(), 16);
        let host = enc.download_plane(&padded);
        for &v in &host {
            assert!((v - 0.7).abs() < 1e-5);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fused_dct8_quantize_persistent() {
        // Fused DCT+quantize via persistent API matches split chain
        // (DCT8 → quantize_dct8) bit-exactly.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 8_u32;
        let n = (nb as usize) * 64;
        let pixels: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
        let weights = vec![1.0_f32; n];
        let qac = vec![4.0_f32; nb as usize];
        let thr = [0.56_f32, 0.62, 0.62, 0.62];

        let p_blocks = enc.upload_blocks(&pixels, nb, 64);
        let w_blocks = enc.upload_blocks(&weights, nb, 64);
        let q_fused = enc.dct8_quantize_fused_persistent(&p_blocks, &w_blocks, &qac, &thr);
        let q_fused_host = enc.download_i32_blocks(&q_fused);

        // Split: DCT first, then quantize.
        let coeffs = enc.dct_8x8_wide_persistent(&p_blocks);
        let q_split = enc.quantize_dct8_persistent(&coeffs, &w_blocks, &qac, &thr);
        let q_split_host = enc.download_i32_blocks(&q_split);

        assert_eq!(
            q_fused_host, q_split_host,
            "fused DCT+quant must match split"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fused_dequant_idct8_y_persistent() {
        // Fused dequant+IDCT (Y) produces finite output. The bench
        // (fused_dequant_idct_bench) already verifies bit-exact
        // match vs split chain (3-channel dequant_dct8 + wide IDCT
        // with CfL=0); here we just confirm the persistent wrapper
        // executes end-to-end on synthetic input.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 8_u32;
        let n = (nb as usize) * 64;

        let quant: Vec<i32> = (0..n).map(|i| ((i as i32 * 7) % 11) - 5).collect();
        let weights = vec![1.0_f32; n];
        let qac = vec![4.0_f32; nb as usize];

        let q_buf = enc.upload_i32_blocks(&quant, nb, 64);
        let w_buf = enc.upload_blocks(&weights, nb, 64);
        let recon = enc.dequant_idct8_fused_y_persistent(&q_buf, &w_buf, &qac);
        assert_eq!(recon.coeffs_per_block(), 64);
        let recon_host = enc.download_blocks(&recon);
        for v in &recon_host {
            assert!(v.is_finite());
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_dct_8x8_wide_persistent_roundtrip() {
        // Wide DCT8 + wide IDCT8 round-trip via persistent API.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 16;
        let n = nb * 64;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
        let blocks = enc.upload_blocks(&input, nb as u32, 64);
        let coeffs = enc.dct_8x8_wide_persistent(&blocks);
        let recon = enc.idct_8x8_wide_persistent(&coeffs);
        let recon_host = enc.download_blocks(&recon);
        let mut max_err = 0.0_f32;
        for i in 0..n {
            max_err = max_err.max((input[i] - recon_host[i]).abs());
        }
        assert!(
            max_err < 5e-5,
            "wide DCT8 persistent roundtrip drift: {max_err:.3e}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_dct_16x16_persistent_roundtrip() {
        // 16x16 forward + inverse should round-trip.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 4;
        let n = nb * 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.007).cos()).collect();
        let blocks = enc.upload_blocks(&input, nb as u32, 256);
        let coeffs = enc.dct_16x16_persistent(&blocks);
        assert_eq!(coeffs.coeffs_per_block(), 256);
        let recon = enc.idct_16x16_persistent(&coeffs);
        let recon_host = enc.download_blocks(&recon);
        let mut max_err = 0.0_f32;
        for i in 0..n {
            max_err = max_err.max((input[i] - recon_host[i]).abs());
        }
        assert!(
            max_err < 1e-4,
            "DCT16x16 persistent roundtrip drift: {max_err:.3e}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_dct_32x32_persistent_roundtrip() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 2;
        let n = nb * 1024;
        let input: Vec<f32> = (0..n)
            .map(|i| 0.5 + 0.3 * (i as f32 * 0.003).sin())
            .collect();
        let blocks = enc.upload_blocks(&input, nb as u32, 1024);
        let coeffs = enc.dct_32x32_persistent(&blocks);
        assert_eq!(coeffs.coeffs_per_block(), 1024);
        let recon = enc.idct_32x32_persistent(&coeffs);
        let recon_host = enc.download_blocks(&recon);
        let mut max_err = 0.0_f32;
        for i in 0..n {
            max_err = max_err.max((input[i] - recon_host[i]).abs());
        }
        assert!(
            max_err < 5e-4,
            "DCT32x32 persistent roundtrip drift: {max_err:.3e}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gather_scatter_roundtrip() {
        // Gather a 16×16 plane into per-8×8 blocks then scatter back
        // → bit-exact roundtrip.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 16_u32;
        let h = 16_u32;
        let n = (w * h) as usize;
        let plane_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.013).collect();
        let plane = enc.upload_plane(&plane_data, w, h);
        let blocks = enc.gather_blocks_persistent(&plane, 8, 8);
        // 16x16 = 4 blocks of 8x8.
        assert_eq!(blocks.num_blocks(), 4);
        assert_eq!(blocks.coeffs_per_block(), 64);
        let plane2 = enc.scatter_blocks_persistent(&blocks, w, h, 8, 8);
        let back = enc.download_plane(&plane2);
        assert_eq!(back, plane_data, "gather→scatter must be bit-exact");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gather_layout_correctness() {
        // Verify the gather places pixels in the expected per-block order.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        // Small 8x8 single-block test: gather should produce identical
        // per-block buffer because there's only one 8×8 block.
        let plane_data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let plane = enc.upload_plane(&plane_data, 8, 8);
        let blocks = enc.gather_blocks_persistent(&plane, 8, 8);
        let host = enc.download_blocks(&blocks);
        assert_eq!(host, plane_data);

        // Now 16×8 (2 blocks of 8×8 in a row): block 0 should be the
        // first 8 columns, block 1 the second 8 columns.
        let mut p2 = vec![0.0_f32; 16 * 8];
        for y in 0..8 {
            for x in 0..16 {
                p2[y * 16 + x] = (y * 16 + x) as f32;
            }
        }
        let plane2 = enc.upload_plane(&p2, 16, 8);
        let blocks2 = enc.gather_blocks_persistent(&plane2, 8, 8);
        assert_eq!(blocks2.num_blocks(), 2);
        let host2 = enc.download_blocks(&blocks2);
        // Block 0 row 0: pixels p2[0..8] = 0..8.
        for (x, &v) in host2[..8].iter().enumerate() {
            assert_eq!(v, x as f32);
        }
        // Block 1 row 0: pixels p2[8..16] = 8..16.
        for (x, &v) in host2[64..72].iter().enumerate() {
            assert_eq!(v, (8 + x) as f32);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_quantize_dct8_persistent_zero() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 4_u32;
        let n = (nb as usize) * 64;
        let coeffs = enc.upload_blocks(&vec![0.0_f32; n], nb, 64);
        let weights = enc.upload_blocks(&vec![1.0_f32; n], nb, 64);
        let qac = vec![1.0_f32; nb as usize];
        let thr = [0.56_f32, 0.62, 0.62, 0.62];
        let q = enc.quantize_dct8_persistent(&coeffs, &weights, &qac, &thr);
        assert_eq!(q.num_blocks(), nb);
        assert_eq!(q.coeffs_per_block(), 64);
        let host = enc.download_i32_blocks(&q);
        assert!(host.iter().all(|&v| v == 0));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_restore_dc_persistent() {
        // src has DC=42, AC=anything. dst has DC=0, AC=anything.
        // After restore, dst[block*64] should equal src[block*64];
        // dst's AC slots should remain unchanged.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 4_u32;
        let cpb = 64_u32;
        let n = (nb * cpb) as usize;
        let mut src_data = vec![0.0_f32; n];
        let mut dst_data = vec![0.0_f32; n];
        for b in 0..nb as usize {
            src_data[b * 64] = 42.0; // DC
            for k in 1..64 {
                src_data[b * 64 + k] = 99.0; // AC (should NOT propagate)
                dst_data[b * 64 + k] = 7.0; // dst AC (must remain)
            }
        }
        let src = enc.upload_blocks(&src_data, nb, cpb);
        let dst = enc.upload_blocks(&dst_data, nb, cpb);
        enc.restore_dc_persistent(&src, &dst);
        let dst_after = enc.download_blocks(&dst);
        for b in 0..nb as usize {
            assert_eq!(dst_after[b * 64], 42.0, "DC not restored at block {b}");
            for k in 1..64 {
                assert_eq!(
                    dst_after[b * 64 + k],
                    7.0,
                    "AC drifted at block {b} slot {k}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_quantize_dequant_chain_persistent() {
        // Quantize all 3 channels then dequant — verifies the typed
        // GpuI32Blocks → GpuBlocks transition works end-to-end.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 8_u32;
        let n = (nb as usize) * 64;
        let coeffs_x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin()).collect();
        let coeffs_y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos()).collect();
        let coeffs_b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin()).collect();
        let cx = enc.upload_blocks(&coeffs_x, nb, 64);
        let cy = enc.upload_blocks(&coeffs_y, nb, 64);
        let cb = enc.upload_blocks(&coeffs_b, nb, 64);
        let weights = vec![1.0_f32; n];
        let wx = enc.upload_blocks(&weights, nb, 64);
        let wy = enc.upload_blocks(&weights, nb, 64);
        let wb = enc.upload_blocks(&weights, nb, 64);
        let qac = vec![4.0_f32; nb as usize];
        let thr = [0.56_f32, 0.62, 0.62, 0.62];
        let qx = enc.quantize_dct8_persistent(&cx, &wx, &qac, &thr);
        let qy = enc.quantize_dct8_persistent(&cy, &wy, &qac, &thr);
        let qb = enc.quantize_dct8_persistent(&cb, &wb, &qac, &thr);

        let xf = vec![0.0_f32; nb as usize];
        let bf = vec![0.0_f32; nb as usize];
        let (dx, dy, db) =
            enc.dequant_dct8_persistent(&qx, &qy, &qb, &wx, &wy, &wb, &qac, &qac, &qac, &xf, &bf);
        assert_eq!(dx.coeffs_per_block(), 64);
        let dx_host = enc.download_blocks(&dx);
        let dy_host = enc.download_blocks(&dy);
        let db_host = enc.download_blocks(&db);
        assert_eq!(dx_host.len(), n);
        for v in dx_host.iter().chain(&dy_host).chain(&db_host) {
            assert!(v.is_finite());
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_dct_idct_persistent_roundtrip() {
        // Forward + inverse DCT8 on persistent blocks should round-trip.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 8;
        let n = nb * 64;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
        let blocks = enc.upload_blocks(&input, nb as u32, 64);
        assert_eq!(blocks.num_blocks(), 8);
        assert_eq!(blocks.coeffs_per_block(), 64);
        assert_eq!(blocks.total_floats(), n);
        let coeffs = enc.dct_8x8_persistent(&blocks);
        let recon = enc.idct_8x8_persistent(&coeffs);
        let recon_host = enc.download_blocks(&recon);
        let mut max_err = 0.0_f32;
        for i in 0..n {
            max_err = max_err.max((input[i] - recon_host[i]).abs());
        }
        assert!(
            max_err < 5e-5,
            "DCT/IDCT roundtrip via persistent API drift: {max_err:.3e}"
        );
    }

    /// `indexed_scatter_blocks_persistent` must round-trip with
    /// `indexed_gather_blocks_persistent`: gathering a sparse subset
    /// of (bx, by) tiles from a plane and then scattering them back to
    /// a fresh plane at the same coords yields a plane with those
    /// tiles bit-equal to the original and untouched pixels at zero
    /// (since the destination is allocated zero-filled).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_indexed_scatter_roundtrip_with_gather() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        for &(pw, ph, tile_w, tile_h) in &[
            (32u32, 32u32, 8u32, 8u32),
            (64, 32, 16, 8),
            (32, 64, 8, 16),
            (64, 64, 16, 16),
        ] {
            let n = (pw * ph) as usize;
            let plane: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011 + 0.3).cos()).collect();
            let g_plane = enc.upload_plane(&plane, pw, ph);
            // Sparse coords: every other valid (bx, by).
            let max_bx = (pw - tile_w) / 8 + 1;
            let max_by = (ph - tile_h) / 8 + 1;
            let mut coords: Vec<(u32, u32)> = Vec::new();
            for by in (0..max_by).step_by(2) {
                for bx in (0..max_bx).step_by(2) {
                    coords.push((bx, by));
                }
            }
            let g_blocks =
                enc.indexed_gather_blocks_persistent(&g_plane, &coords, tile_w, tile_h);
            // Upload an explicitly-zero-initialized destination — alloc_plane
            // returns uninitialized memory (cubecl's empty(); see persistent.rs:231).
            let zero = vec![0.0_f32; n];
            let g_dst = enc.upload_plane(&zero, pw, ph);
            enc.indexed_scatter_blocks_persistent(&g_blocks, &coords, &g_dst, tile_w, tile_h);
            let dst = enc.download_plane(&g_dst);
            // Build expected: 0 everywhere, original pixels in covered tiles.
            let mut expected = vec![0.0_f32; n];
            for &(bx, by) in &coords {
                let x0 = (bx * 8) as usize;
                let y0 = (by * 8) as usize;
                for dy in 0..tile_h as usize {
                    for dx in 0..tile_w as usize {
                        let off = (y0 + dy) * pw as usize + (x0 + dx);
                        expected[off] = plane[off];
                    }
                }
            }
            for (i, (g, e)) in dst.iter().zip(expected.iter()).enumerate() {
                assert_eq!(g, e, "{pw}×{ph} tile {tile_w}×{tile_h} pixel {i}");
            }
        }
    }

    /// `set_llf_dct64x{64,32}_indexed_persistent` +
    /// `set_llf_dct32x64_indexed_persistent` must match the host
    /// scalar `restore_llf_dct64x64`/`restore_llf_dct64x32`/
    /// `restore_llf_dct32x64` per-block within fp32 rounding. All
    /// LLF positions written, AC untouched.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_set_llf_dct64_family_indexed_persistent_matches_host() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        // Need a DC grid large enough to fit DCT64x64's 8×8 subgrid.
        let xsize_blocks_8 = 16u32;
        let ysize_blocks_8 = 16u32;
        let n_total = (xsize_blocks_8 * ysize_blocks_8) as usize;
        let dc_host: Vec<f32> = (0..n_total)
            .map(|i| (i as f32 * 0.041).sin() * 0.6 + 0.25)
            .collect();
        let g_dc = enc.upload_blocks(&dc_host, n_total as u32, 1);

        // ----- DCT64x64 (8×8 subgrid) -----
        let mut coords_64x64: Vec<(u32, u32)> = Vec::new();
        for by in 0..(ysize_blocks_8 - 7) {
            for bx in 0..(xsize_blocks_8 - 7) {
                coords_64x64.push((bx, by));
            }
        }
        let cpb_64x64 = 4096u32;
        let init_64x64: Vec<f32> = (0..(coords_64x64.len() * cpb_64x64 as usize))
            .map(|i| -((i as f32) * 0.0007 + 1.0))
            .collect();
        let g_dst = enc.upload_blocks(&init_64x64, coords_64x64.len() as u32, cpb_64x64);
        enc.set_llf_dct64x64_indexed_persistent(&g_dc, &coords_64x64, &g_dst, xsize_blocks_8);
        let got = enc.download_blocks(&g_dst);
        for (i, &(bx, by)) in coords_64x64.iter().enumerate() {
            let mut sub = [0.0_f32; 64];
            for dy in 0..8u32 {
                for dx in 0..8u32 {
                    let idx = ((by + dy) * xsize_blocks_8 + (bx + dx)) as usize;
                    sub[(dy * 8 + dx) as usize] = dc_host[idx];
                }
            }
            let host_llf = crate::forks::reconstruct::restore_llf_dct64x64(sub);
            let off = i * cpb_64x64 as usize;
            for iy in 0..8 {
                for ix in 0..8 {
                    let pos = iy * 64 + ix;
                    let expected = host_llf[iy * 8 + ix];
                    assert!(
                        (got[off + pos] - expected).abs() < 1e-3,
                        "DCT64x64 block {i} pos ({iy},{ix}): gpu={} host={expected}",
                        got[off + pos]
                    );
                }
            }
            // AC sample untouched.
            for &k in &[8usize, 9, 63, 72, 4095] {
                assert_eq!(got[off + k], init_64x64[off + k]);
            }
        }

        // ----- DCT64x32 (8×4 subgrid) -----
        let mut coords_64x32: Vec<(u32, u32)> = Vec::new();
        for by in 0..(ysize_blocks_8 - 7) {
            for bx in 0..(xsize_blocks_8 - 3) {
                coords_64x32.push((bx, by));
            }
        }
        let cpb_64x32 = 2048u32;
        let init_64x32: Vec<f32> = (0..(coords_64x32.len() * cpb_64x32 as usize))
            .map(|i| -((i as f32) * 0.00029 + 1.0))
            .collect();
        let g_dst_64x32 = enc.upload_blocks(&init_64x32, coords_64x32.len() as u32, cpb_64x32);
        enc.set_llf_dct64x32_indexed_persistent(&g_dc, &coords_64x32, &g_dst_64x32, xsize_blocks_8);
        let got_64x32 = enc.download_blocks(&g_dst_64x32);
        for (i, &(bx, by)) in coords_64x32.iter().enumerate() {
            // Build 8×4 host subgrid: dc[iy*4+ix] for iy in 0..8, ix in 0..4.
            let mut sub = [0.0_f32; 32];
            for dy in 0..8u32 {
                for dx in 0..4u32 {
                    let idx = ((by + dy) * xsize_blocks_8 + (bx + dx)) as usize;
                    sub[(dy * 4 + dx) as usize] = dc_host[idx];
                }
            }
            let host_llf = crate::forks::reconstruct::restore_llf_dct64x32(sub);
            let off = i * cpb_64x32 as usize;
            for iy in 0..4 {
                for ix in 0..8 {
                    let pos = iy * 64 + ix;
                    let expected = host_llf[iy * 8 + ix];
                    assert!(
                        (got_64x32[off + pos] - expected).abs() < 1e-3,
                        "DCT64x32 block {i} pos ({iy},{ix}): gpu={} host={expected}",
                        got_64x32[off + pos]
                    );
                }
            }
        }

        // ----- DCT32x64 (4×8 subgrid) -----
        let mut coords_32x64: Vec<(u32, u32)> = Vec::new();
        for by in 0..(ysize_blocks_8 - 3) {
            for bx in 0..(xsize_blocks_8 - 7) {
                coords_32x64.push((bx, by));
            }
        }
        let init_32x64: Vec<f32> = (0..(coords_32x64.len() * cpb_64x32 as usize))
            .map(|i| -((i as f32) * 0.00037 + 1.0))
            .collect();
        let g_dst_32x64 = enc.upload_blocks(&init_32x64, coords_32x64.len() as u32, cpb_64x32);
        enc.set_llf_dct32x64_indexed_persistent(&g_dc, &coords_32x64, &g_dst_32x64, xsize_blocks_8);
        let got_32x64 = enc.download_blocks(&g_dst_32x64);
        for (i, &(bx, by)) in coords_32x64.iter().enumerate() {
            // Build 4×8 host subgrid: dc[iy*8+ix] for iy in 0..4, ix in 0..8.
            let mut sub = [0.0_f32; 32];
            for dy in 0..4u32 {
                for dx in 0..8u32 {
                    let idx = ((by + dy) * xsize_blocks_8 + (bx + dx)) as usize;
                    sub[(dy * 8 + dx) as usize] = dc_host[idx];
                }
            }
            let host_llf = crate::forks::reconstruct::restore_llf_dct32x64(sub);
            let off = i * cpb_64x32 as usize;
            for iy in 0..4 {
                for ix in 0..8 {
                    let pos = iy * 64 + ix;
                    let expected = host_llf[iy * 8 + ix];
                    assert!(
                        (got_32x64[off + pos] - expected).abs() < 1e-3,
                        "DCT32x64 block {i} pos ({iy},{ix}): gpu={} host={expected}",
                        got_32x64[off + pos]
                    );
                }
            }
        }
    }

    /// `set_llf_dct32x16_indexed_persistent` and
    /// `set_llf_dct16x32_indexed_persistent` must match the host scalar
    /// `restore_llf_dct32x16` / `restore_llf_dct16x32` per-block within
    /// fp32 rounding. 8 LLF positions written, AC untouched.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_set_llf_dct32x16_and_16x32_indexed_persistent_matches_host() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let xsize_blocks_8 = 16u32;
        let ysize_blocks_8 = 8u32;
        let n_total = (xsize_blocks_8 * ysize_blocks_8) as usize;
        let dc_host: Vec<f32> = (0..n_total)
            .map(|i| (i as f32 * 0.23).cos() * 0.4 + 0.15)
            .collect();
        let g_dc = enc.upload_blocks(&dc_host, n_total as u32, 1);

        // DCT32x16: 4×2 DC subgrid, coords need bx+1 ≤ stride-1, by+3 ≤ ysize-1.
        let mut coords_32x16: Vec<(u32, u32)> = Vec::new();
        for by in 0..(ysize_blocks_8 - 3) {
            for bx in 0..(xsize_blocks_8 - 1) {
                coords_32x16.push((bx, by));
            }
        }
        let cpb = 512u32;
        // Size init for the LARGER of the two test runs below
        // (coords_16x32 has more entries than coords_32x16 because
        // its (max_bx_offset, max_by_offset) shape has more valid
        // origins in this 16×8 grid).
        let max_count = (((ysize_blocks_8 - 1) * (xsize_blocks_8 - 3))
            .max((ysize_blocks_8 - 3) * (xsize_blocks_8 - 1))) as usize;
        let init: Vec<f32> = (0..(max_count * cpb as usize))
            .map(|i| -((i as f32) * 0.0029 + 1.0))
            .collect();
        let g_dst_32x16 = enc.upload_blocks(
            &init[..coords_32x16.len() * cpb as usize],
            coords_32x16.len() as u32,
            cpb,
        );
        enc.set_llf_dct32x16_indexed_persistent(&g_dc, &coords_32x16, &g_dst_32x16, xsize_blocks_8);
        let got_32x16 = enc.download_blocks(&g_dst_32x16);
        for (i, &(bx, by)) in coords_32x16.iter().enumerate() {
            // Build host 4×2 DC subgrid.
            let mut sub = [0.0_f32; 8];
            for dy in 0..4u32 {
                for dx in 0..2u32 {
                    let idx = ((by + dy) * xsize_blocks_8 + (bx + dx)) as usize;
                    sub[(dy * 2 + dx) as usize] = dc_host[idx];
                }
            }
            let host_llf = crate::forks::reconstruct::restore_llf_dct32x16(sub);
            let off = i * cpb as usize;
            for iy in 0..2 {
                for ix in 0..4 {
                    let pos = iy * 32 + ix;
                    let expected = host_llf[iy * 4 + ix];
                    assert!(
                        (got_32x16[off + pos] - expected).abs() < 1e-4,
                        "DCT32x16 block {i} pos ({iy},{ix}): gpu={} host={expected}",
                        got_32x16[off + pos]
                    );
                }
            }
            // AC sample.
            for &k in &[4usize, 5, 31, 36, 100, 511] {
                assert_eq!(got_32x16[off + k], init[off + k]);
            }
        }

        // DCT16x32: 2×4 DC subgrid, coords need bx+3 ≤ stride-1, by+1 ≤ ysize-1.
        let mut coords_16x32: Vec<(u32, u32)> = Vec::new();
        for by in 0..(ysize_blocks_8 - 1) {
            for bx in 0..(xsize_blocks_8 - 3) {
                coords_16x32.push((bx, by));
            }
        }
        let g_dst_16x32 = enc.upload_blocks(&init[..coords_16x32.len() * cpb as usize], coords_16x32.len() as u32, cpb);
        enc.set_llf_dct16x32_indexed_persistent(&g_dc, &coords_16x32, &g_dst_16x32, xsize_blocks_8);
        let got_16x32 = enc.download_blocks(&g_dst_16x32);
        for (i, &(bx, by)) in coords_16x32.iter().enumerate() {
            // Build host 2×4 DC subgrid.
            let mut sub = [0.0_f32; 8];
            for dy in 0..2u32 {
                for dx in 0..4u32 {
                    let idx = ((by + dy) * xsize_blocks_8 + (bx + dx)) as usize;
                    sub[(dy * 4 + dx) as usize] = dc_host[idx];
                }
            }
            let host_llf = crate::forks::reconstruct::restore_llf_dct16x32(sub);
            let off = i * cpb as usize;
            for iy in 0..2 {
                for ix in 0..4 {
                    let pos = iy * 32 + ix;
                    let expected = host_llf[iy * 4 + ix];
                    assert!(
                        (got_16x32[off + pos] - expected).abs() < 1e-4,
                        "DCT16x32 block {i} pos ({iy},{ix}): gpu={} host={expected}",
                        got_16x32[off + pos]
                    );
                }
            }
        }
    }

    /// `set_llf_dct32x32_indexed_persistent` must match the host scalar
    /// `forks::reconstruct::restore_llf_dct32x32` per-block within fp32
    /// rounding. 4×4 LLF positions written, AC untouched.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_set_llf_dct32x32_indexed_persistent_matches_host() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let xsize_blocks_8 = 16u32;
        let ysize_blocks_8 = 8u32;
        let n_total = (xsize_blocks_8 * ysize_blocks_8) as usize;
        let dc_host: Vec<f32> = (0..n_total)
            .map(|i| (i as f32 * 0.19).sin() * 0.5 + 0.2)
            .collect();
        let g_dc = enc.upload_blocks(&dc_host, n_total as u32, 1);
        // Sparse coords: pick blocks where (bx+3, by+3) stays in-range.
        let mut coords: Vec<(u32, u32)> = Vec::new();
        for by in 0..(ysize_blocks_8 - 3) {
            for bx in 0..(xsize_blocks_8 - 3) {
                coords.push((bx, by));
            }
        }
        let cpb = 1024u32;
        let init: Vec<f32> = (0..(coords.len() * cpb as usize))
            .map(|i| -((i as f32) * 0.0017 + 1.0))
            .collect();
        let g_dst = enc.upload_blocks(&init, coords.len() as u32, cpb);
        enc.set_llf_dct32x32_indexed_persistent(&g_dc, &coords, &g_dst, xsize_blocks_8);
        let got = enc.download_blocks(&g_dst);
        for (i, &(bx, by)) in coords.iter().enumerate() {
            // Build host 4×4 DC subgrid for this block.
            let mut sub = [0.0_f32; 16];
            for dy in 0..4u32 {
                for dx in 0..4u32 {
                    let idx = ((by + dy) * xsize_blocks_8 + (bx + dx)) as usize;
                    sub[(dy * 4 + dx) as usize] = dc_host[idx];
                }
            }
            let host_llf = crate::forks::reconstruct::restore_llf_dct32x32(sub);
            let off = i * cpb as usize;
            // 16 LLF positions: iy * 32 + ix for iy, ix in 0..4.
            for iy in 0..4 {
                for ix in 0..4 {
                    let pos = iy * 32 + ix;
                    let expected = host_llf[iy * 4 + ix];
                    assert!(
                        (got[off + pos] - expected).abs() < 1e-4,
                        "block {i} pos ({iy},{ix}) idx {pos}: gpu={} host={expected}",
                        got[off + pos]
                    );
                }
            }
            // AC positions untouched (sample a few in the LLF row band).
            for &k in &[4usize, 5, 30, 31, 100, 500, 1023] {
                assert_eq!(
                    got[off + k],
                    init[off + k],
                    "block {i} AC drift at slot {k}"
                );
            }
        }
    }

    /// `set_llf_dct16x16_indexed_persistent` must match the host scalar
    /// `forks::reconstruct::restore_llf_dct16x16` applied per-block
    /// within fp32 rounding, with all 4 LLF positions written and AC
    /// coefficients untouched.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_set_llf_dct16x16_indexed_persistent_matches_host() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let xsize_blocks_8 = 16u32;
        let ysize_blocks_8 = 8u32;
        let n_total = (xsize_blocks_8 * ysize_blocks_8) as usize;
        let dc_host: Vec<f32> = (0..n_total)
            .map(|i| (i as f32 * 0.27).cos() * 0.4 + 0.1)
            .collect();
        let g_dc = enc.upload_blocks(&dc_host, n_total as u32, 1);
        // Sparse coords: pick blocks where (bx+1, by+1) stays in-range.
        let mut coords: Vec<(u32, u32)> = Vec::new();
        for by in 0..(ysize_blocks_8 - 1) {
            for bx in 0..(xsize_blocks_8 - 1) {
                coords.push((bx, by));
            }
        }
        let cpb = 256u32;
        let init: Vec<f32> = (0..(coords.len() * cpb as usize))
            .map(|i| -((i as f32) * 0.011 + 1.0))
            .collect();
        let g_dst = enc.upload_blocks(&init, coords.len() as u32, cpb);
        enc.set_llf_dct16x16_indexed_persistent(&g_dc, &coords, &g_dst, xsize_blocks_8);
        let got = enc.download_blocks(&g_dst);
        for (i, &(bx, by)) in coords.iter().enumerate() {
            let row0 = (by * xsize_blocks_8 + bx) as usize;
            let row1 = ((by + 1) * xsize_blocks_8 + bx) as usize;
            let dc_subgrid = [
                dc_host[row0],
                dc_host[row0 + 1],
                dc_host[row1],
                dc_host[row1 + 1],
            ];
            let host_llf = crate::forks::reconstruct::restore_llf_dct16x16(dc_subgrid);
            let off = i * cpb as usize;
            // 4 LLF positions: 0, 1, 16, 17.
            let positions = [(0, host_llf[0]), (1, host_llf[1]), (16, host_llf[2]), (17, host_llf[3])];
            for (pos, expected) in positions {
                assert!(
                    (got[off + pos] - expected).abs() < 1e-5,
                    "block {i} pos {pos}: gpu={} host={expected}",
                    got[off + pos]
                );
            }
            // AC positions untouched (sample a few).
            for &k in &[2usize, 5, 15, 18, 32, 100, 255] {
                assert_eq!(
                    got[off + k],
                    init[off + k],
                    "block {i} AC drift at slot {k}"
                );
            }
        }
    }

    /// `set_llf_dct16x8_or_8x16_indexed_persistent` must match the
    /// host scalar `forks::reconstruct::restore_llf_dct16x8_or_8x16`
    /// applied per-block within fp32 rounding.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_set_llf_dct16x8_or_8x16_indexed_persistent_matches_host() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        // 16×8 DC grid (in 8×8-block units) covering a 128×64 image.
        let xsize_blocks_8 = 16u32;
        let ysize_blocks_8 = 8u32;
        let n_total = (xsize_blocks_8 * ysize_blocks_8) as usize;
        let dc_host: Vec<f32> = (0..n_total).map(|i| (i as f32 * 0.31).sin() * 0.7).collect();
        let g_dc = enc.upload_blocks(&dc_host, n_total as u32, 1);
        // Test both DCT16x8 (vertical pair, dc_step = stride)
        // and DCT8x16 (horizontal pair, dc_step = 1).
        for &(label, dc_step) in &[("DCT16x8", xsize_blocks_8), ("DCT8x16", 1u32)] {
            // Sparse coords: pick blocks where both dc_grid reads stay in-range.
            let mut coords: Vec<(u32, u32)> = Vec::new();
            for by in 0..(ysize_blocks_8 - 1) {
                for bx in 0..(xsize_blocks_8 - 1) {
                    coords.push((bx, by));
                }
            }
            let cpb = 128u32;
            // Init dst with sentinels to verify only positions 0/1 get written.
            let init: Vec<f32> = (0..(coords.len() * cpb as usize))
                .map(|i| -((i as f32) + 1.0))
                .collect();
            let g_dst = enc.upload_blocks(&init, coords.len() as u32, cpb);
            enc.set_llf_dct16x8_or_8x16_indexed_persistent(
                &g_dc, &coords, &g_dst, xsize_blocks_8, dc_step,
            );
            let got = enc.download_blocks(&g_dst);
            for (i, &(bx, by)) in coords.iter().enumerate() {
                let dc0_idx = (by * xsize_blocks_8 + bx) as usize;
                let dc1_idx = dc0_idx + dc_step as usize;
                let dc0 = dc_host[dc0_idx];
                let dc1 = dc_host[dc1_idx];
                let host_llf = crate::forks::reconstruct::restore_llf_dct16x8_or_8x16(dc0, dc1);
                let g0 = got[i * cpb as usize];
                let g1 = got[i * cpb as usize + 1];
                assert!(
                    (g0 - host_llf[0]).abs() < 1e-5,
                    "{label} block {i} llf0: gpu={g0} host={}",
                    host_llf[0]
                );
                assert!(
                    (g1 - host_llf[1]).abs() < 1e-5,
                    "{label} block {i} llf1: gpu={g1} host={}",
                    host_llf[1]
                );
                // AC positions untouched.
                for k in 2..cpb as usize {
                    assert_eq!(
                        got[i * cpb as usize + k],
                        init[i * cpb as usize + k],
                        "{label} block {i} AC drift at slot {k}"
                    );
                }
            }
        }
    }

    /// `set_dc_from_grid_indexed_persistent` must write
    /// `dc_grid[by * stride + bx]` into `dst[i * cpb + 0]` for each
    /// block in `coords`, leaving AC positions untouched. Bit-exact
    /// vs the equivalent host loop.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_set_dc_from_grid_indexed_persistent_matches_host() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        // 8×4 block grid → 32 dc values. Sparse coords picking 5 blocks
        // out of 32; cpb=64 (DCT8 shape).
        let xsize_blocks_8 = 8u32;
        let ysize_blocks_8 = 4u32;
        let n_total = (xsize_blocks_8 * ysize_blocks_8) as usize;
        let dc_host: Vec<f32> = (0..n_total).map(|i| i as f32 * 1.5 + 0.7).collect();
        let g_dc = enc.upload_blocks(&dc_host, n_total as u32, 1);
        let coords: Vec<(u32, u32)> = vec![
            (0, 0),
            (3, 1),
            (7, 2),
            (5, 3),
            (1, 0),
        ];
        let cpb = 64u32;
        let n_strat_blocks = coords.len();
        // Initialize dst with sentinel values to verify only DC is written.
        let init: Vec<f32> = (0..(n_strat_blocks * cpb as usize))
            .map(|i| -((i as f32) + 1.0))
            .collect();
        let g_dst = enc.upload_blocks(&init, n_strat_blocks as u32, cpb);
        enc.set_dc_from_grid_indexed_persistent(&g_dc, &coords, &g_dst, xsize_blocks_8);
        let got = enc.download_blocks(&g_dst);
        for (i, &(bx, by)) in coords.iter().enumerate() {
            let dc_idx = (by * xsize_blocks_8 + bx) as usize;
            // Position 0 of each strategy block.
            assert_eq!(
                got[i * cpb as usize],
                dc_host[dc_idx],
                "DC mismatch at block {i} (bx={bx}, by={by})"
            );
            // AC positions untouched.
            for k in 1..cpb as usize {
                assert_eq!(
                    got[i * cpb as usize + k],
                    init[i * cpb as usize + k],
                    "AC drifted at block {i} slot {k}"
                );
            }
        }
    }

    /// `indexed_gather_blocks_persistent` must produce bit-identical
    /// output to the host pattern `for (bx, by) in coords {
    /// extend_from_slice(plane[(by*8+row)*W + bx*8 .. + tile_w]) }`
    /// — same layout as the per-strategy `extend_from_slice` loop in
    /// `forks::reconstruct::encode_and_reconstruct_mixed_strategy_single_channel`.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_indexed_gather_blocks_persistent_matches_host() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        for &(pw, ph, tile_w, tile_h) in &[
            (32u32, 32u32, 8u32, 8u32),
            (64, 32, 16, 8),  // DCT8x16 tile
            (32, 64, 8, 16),  // DCT16x8 tile
            (64, 64, 16, 16), // DCT16x16
            (128, 128, 32, 32), // DCT32x32
        ] {
            let n = (pw * ph) as usize;
            let plane: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
            let g_plane = enc.upload_plane(&plane, pw, ph);
            // Sparse coords: every other valid (bx, by) starting at (1, 0).
            let max_bx = (pw - tile_w) / 8 + 1;
            let max_by = (ph - tile_h) / 8 + 1;
            let mut coords: Vec<(u32, u32)> = Vec::new();
            for by in 0..max_by {
                for bx in (1..max_bx).step_by(2) {
                    coords.push((bx, by));
                }
            }
            // Host reference: replicate the extend_from_slice loop.
            let mut host_batch: Vec<f32> =
                Vec::with_capacity(coords.len() * (tile_w as usize) * (tile_h as usize));
            for &(bx, by) in &coords {
                let x0 = (bx * 8) as usize;
                let y0 = (by * 8) as usize;
                for dy in 0..tile_h as usize {
                    let src_off = (y0 + dy) * (pw as usize) + x0;
                    host_batch
                        .extend_from_slice(&plane[src_off..src_off + tile_w as usize]);
                }
            }
            let g_blocks = enc.indexed_gather_blocks_persistent(&g_plane, &coords, tile_w, tile_h);
            let gpu_batch = enc.download_blocks(&g_blocks);
            assert_eq!(
                gpu_batch.len(),
                host_batch.len(),
                "len mismatch at {pw}×{ph} tile {tile_w}×{tile_h}"
            );
            for (i, (g, h_)) in gpu_batch.iter().zip(host_batch.iter()).enumerate() {
                assert_eq!(g, h_, "{pw}×{ph} tile {tile_w}×{tile_h} pixel {i}");
            }
        }
    }

    /// `dc_grid_8x8_persistent` GPU kernel must match the host scalar
    /// `compute_dc_grid_per_8x8_block` reference exactly within fp32
    /// rounding (1e-5 tolerance — the host computes via f32 sum, kernel
    /// uses fp32 sum too, so order-of-summation may differ by a few
    /// ULPs but never more than 1e-5 on bounded synthetic input).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_dc_grid_8x8_persistent_matches_host() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        for &(w, h) in &[(16u32, 16u32), (32, 32), (64, 48), (128, 256)] {
            let n = (w * h) as usize;
            let plane: Vec<f32> = (0..n).map(|i| 0.05 * ((i as f32 * 0.013).sin() + 0.5)).collect();
            let g_plane = enc.upload_plane(&plane, w, h);
            let g_dc = enc.dc_grid_8x8_persistent(&g_plane);
            let gpu_dc = enc.download_blocks(&g_dc);
            let host_dc = crate::forks::reconstruct::compute_dc_grid_per_8x8_block(
                &plane, w as usize, h as usize,
            );
            assert_eq!(gpu_dc.len(), host_dc.len(), "len mismatch at {w}×{h}");
            for (i, (g, h_)) in gpu_dc.iter().zip(host_dc.iter()).enumerate() {
                assert!(
                    (g - h_).abs() < 1e-5,
                    "{w}×{h} block {i}: gpu={g} host={h_}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_xyb_inverse_persistent_roundtrip() {
        // Persistent XYB forward + inverse should round-trip linear RGB.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n = 256;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();

        let g_r = enc.upload_plane(&r, 16, 16);
        let g_g = enc.upload_plane(&g, 16, 16);
        let g_b = enc.upload_plane(&b, 16, 16);

        let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(&g_r, &g_g, &g_b);
        let (rx, gy, bz) = enc.xyb_to_linear_rgb_planar_persistent(&xx, &xy, &xb);

        let r2 = enc.download_plane(&rx);
        let g2 = enc.download_plane(&gy);
        let b2 = enc.download_plane(&bz);

        let mut max_err = 0.0_f32;
        for i in 0..n {
            max_err = max_err.max((r[i] - r2[i]).abs());
            max_err = max_err.max((g[i] - g2[i]).abs());
            max_err = max_err.max((b[i] - b2[i]).abs());
        }
        assert!(
            max_err < 5e-4,
            "XYB roundtrip via persistent API drift: {max_err:.3e}"
        );
    }
}
