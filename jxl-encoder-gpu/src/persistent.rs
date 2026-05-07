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
//! This module provides:
//! - [`GpuPlane`] — typed handle to a GPU-resident `f32` plane
//!   (width, height, GPU buffer). Holds an opaque `cubecl::Handle`.
//! - Persistent-API methods on [`GpuEncoder`] that return `GpuPlane`s
//!   (suffixed `_persistent`). Caller chains stages without
//!   round-tripping data through host memory.
//! - [`GpuEncoder::download_plane`] / `upload_plane` for explicit
//!   transfer at the pipeline boundaries.
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

use alloc::vec;
use alloc::vec::Vec;

use cubecl::Runtime;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::encoder::GpuEncoder;
use crate::launch::dct16::{dct_8x16, dct_16x8, dct_16x16, idct_8x16, idct_16x8, idct_16x16};
use crate::launch::dct32::{
    dct_16x32, dct_32x16, dct_32x32, idct_16x32, idct_32x16, idct_32x32,
};
use crate::launch::dct4::{
    dct_4x4_full, dct_4x8_full, dct_8x4_full, idct_4x4_full, idct_4x8_full, idct_8x4_full,
};
use crate::launch::dct64::{
    dct_32x64, dct_64x32, dct_64x64, idct_32x64, idct_64x32, idct_64x64,
};
use crate::launch::dct8::{dct_8x8, idct_8x8};
use crate::launch::epf::pad_plane;
use crate::launch::gab::gab_smooth;
use crate::launch::gaborish::gaborish_5x5;
use crate::launch::mask1x1::mask1x1;
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
        assert_eq!(data.len(), n, "data length {} != width*height {n}", data.len());
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
        let handle = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        GpuPlane {
            handle,
            width,
            height,
            _r: core::marker::PhantomData,
        }
    }

    /// Download a GPU plane back to host memory.
    pub fn download_plane(&self, plane: &GpuPlane<R>) -> Vec<f32> {
        let bytes = self.client_ref().read_one(plane.handle.clone()).expect("download");
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
        let h_x = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        let h_y = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        let h_b_out = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
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
        let h_r = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        let h_g = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        let h_b = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
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
        let h_out = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
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
        let h_out = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
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
        let h_out = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; dst_n]));
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
        let handle = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        GpuBlocks {
            handle,
            num_blocks,
            coeffs_per_block,
            _r: core::marker::PhantomData,
        }
    }

    /// Download a `GpuBlocks` back to host memory.
    pub fn download_blocks(&self, blocks: &GpuBlocks<R>) -> Vec<f32> {
        let bytes = self.client_ref().read_one(blocks.handle.clone()).expect("download");
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
        let h_out = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
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
        let h_out = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
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
        let h_out = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
        launch(self.client_ref(), blocks.handle.clone(), h_out.clone(), blocks.num_blocks);
        GpuBlocks {
            handle: h_out,
            num_blocks: blocks.num_blocks,
            coeffs_per_block: out_coeffs_per_block,
            _r: core::marker::PhantomData,
        }
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

    /// Persistent-API mask1x1 field on the Y channel.
    pub fn mask1x1_persistent(&self, y: &GpuPlane<R>) -> GpuPlane<R> {
        let n = y.n_pixels();
        let h_out = self
            .client_ref()
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
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
        assert!(max_err < 1e-4, "DCT16x16 persistent roundtrip drift: {max_err:.3e}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_dct_32x32_persistent_roundtrip() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let nb = 2;
        let n = nb * 1024;
        let input: Vec<f32> = (0..n).map(|i| 0.5 + 0.3 * (i as f32 * 0.003).sin()).collect();
        let blocks = enc.upload_blocks(&input, nb as u32, 1024);
        let coeffs = enc.dct_32x32_persistent(&blocks);
        assert_eq!(coeffs.coeffs_per_block(), 1024);
        let recon = enc.idct_32x32_persistent(&coeffs);
        let recon_host = enc.download_blocks(&recon);
        let mut max_err = 0.0_f32;
        for i in 0..n {
            max_err = max_err.max((input[i] - recon_host[i]).abs());
        }
        assert!(max_err < 5e-4, "DCT32x32 persistent roundtrip drift: {max_err:.3e}");
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
        assert!(max_err < 5e-4, "XYB roundtrip via persistent API drift: {max_err:.3e}");
    }
}
