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
