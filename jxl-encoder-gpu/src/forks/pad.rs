// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/epf.rs::pad_plane_into (BSD-3-Clause
// via libjxl + AGPL/commercial), with the SIMD edge-replication
// substituted for a GPU launch.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted edge-replicate plane padding.
//!
//! Used by EPF and any other pipeline stage that needs a stride-padded
//! input buffer with replicated boundary pixels.
//!
//! ## Reshape vs upstream `pad_plane_into`
//!
//! Upstream's `pad_plane_into` copies the interior region row-by-row,
//! replicates left/right edges per row, then copies the top and bottom
//! padded rows verbatim. CPU-friendly tiny work.
//!
//! On GPU: one launch handles the whole image. The kernel reads from
//! the unpadded input and writes to the padded output, computing
//! per-output-pixel which source pixel to read (clamping coordinates
//! to `[0, width-1]` × `[0, height-1]`).
//!
//! Slightly more compute on GPU (compute the clamp per pixel instead
//! of memcpy), but a single launch instead of `2*pad + 1` calls.
//!
//! Output shape: `(width + 2*pad) × (height + 2*pad)`.

use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

/// GPU `pad_plane`. Mirrors upstream `pad_plane_into` but returns a
/// fresh `Vec<f32>` (the GPU kernel manages its own output buffer).
///
/// `pad` is the number of pixels to add on each side; output dimensions
/// are `(width + 2*pad) × (height + 2*pad)`.
///
/// Boundary semantics: edge replication (clamp coordinates).
pub fn pad_plane_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    plane: &[f32],
    width: usize,
    height: usize,
    pad: usize,
) -> Vec<f32> {
    assert_eq!(plane.len(), width * height);
    enc.pad_plane_channel(plane, width as u32, height as u32, pad as u32)
}

/// Convenience: pad all 3 XYB channels in 3 sequential launches. Mirrors
/// upstream's "pad all three then EPF" pattern (`epf.rs:672-674`).
///
/// Returns `(padded_x, padded_y, padded_b)`, each of shape
/// `(width + 2*pad) × (height + 2*pad)`.
pub fn pad_xyb_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    pad: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        pad_plane_gpu(enc, xyb_x, width, height, pad),
        pad_plane_gpu(enc, xyb_y, width, height, pad),
        pad_plane_gpu(enc, xyb_b, width, height, pad),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_pad_uniform_gpu() {
        // Uniform plane → uniform padded plane.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 16;
        let h = 8;
        let pad = 4;
        let plane = vec![0.5_f32; w * h];
        let out = pad_plane_gpu(&enc, &plane, w, h, pad);
        let pw = w + 2 * pad;
        let ph = h + 2 * pad;
        assert_eq!(out.len(), pw * ph);
        for &v in &out {
            assert_eq!(v, 0.5);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_pad_interior_preserved_gpu() {
        // Interior pixels should match the input exactly.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 8;
        let h = 8;
        let pad = 2;
        let plane: Vec<f32> = (0..w * h).map(|i| i as f32 * 0.01).collect();
        let out = pad_plane_gpu(&enc, &plane, w, h, pad);
        let pw = w + 2 * pad;
        for y in 0..h {
            for x in 0..w {
                let src = y * w + x;
                let dst = (y + pad) * pw + (x + pad);
                assert_eq!(
                    out[dst], plane[src],
                    "interior pixel ({x},{y}) drifted: {} vs {}",
                    out[dst], plane[src]
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_pad_edge_replication_gpu() {
        // Top-left padded corner should equal the (0, 0) pixel of input;
        // bottom-right padded corner should equal the (w-1, h-1) pixel.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 4;
        let h = 4;
        let pad = 2;
        let plane: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
        let out = pad_plane_gpu(&enc, &plane, w, h, pad);
        let pw = w + 2 * pad;
        // Top-left padded corner (0,0) → should be input (0,0) = 0.0
        assert_eq!(out[0], plane[0]);
        // Bottom-right padded corner → should be input (w-1, h-1) = 15.0
        let last = out.len() - 1;
        assert_eq!(out[last], plane[w * h - 1]);
        // Middle of top padded row (col 3) → should be input (col 1, row 0) = 1.0
        assert_eq!(out[3], plane[1]);
        // Middle of left padded col (row 3) → should be input (col 0, row 1) = 4.0
        assert_eq!(out[3 * pw], plane[w]);
    }
}
