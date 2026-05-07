// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder (BSD-3-Clause via libjxl + AGPL/commercial),
// reshaped for GPU-friendly whole-image batching.
// Licensed under AGPL-3.0-or-later or commercial; see LICENSE-AGPL3 /
// LICENSE-COMMERCIAL.

//! GPU-substituted XYB conversion for the encoder pipeline.
//!
//! Reshape vs upstream `jxl_encoder::vardct::xyb::convert_strip`:
//! - **Whole-image batch instead of per-row strips.** Original CPU code
//!   parallelizes via `rayon::par_chunks_mut(strip_len)` because per-row
//!   work is cheap. On GPU, kernel launch overhead dominates at small
//!   sizes — batching the whole image into ONE launch is ~50× faster
//!   than per-row launches at 1 MP.
//! - **Deinterleave done on host before upload.** The XYB kernel takes
//!   planar inputs (separate R, G, B arrays). Source is interleaved RGB.
//!   We deinterleave once then upload three contiguous arrays; the
//!   alternative (an interleaved-input kernel) would force GPU memory
//!   accesses with stride 3, which on CUDA causes uncoalesced loads
//!   (~3× slower).
//! - **Edge replication after XYB rather than before.** Upstream pads
//!   each row's right edge with the last in-row pixel BEFORE XYB
//!   conversion. We do XYB on the unpadded plane, then pad in a
//!   separate small CPU loop. Total work is identical; the GPU kernel
//!   doesn't need to know about padded_width vs width.
//!
//! Currently does NOT support `primaries_matrix` (non-sRGB primaries).
//! Caller must apply the matrix on host before calling. Future: add a
//! GPU `apply_matrix_3x3` kernel and chain it.

use alloc::vec;
use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

/// GPU XYB for a full linear-RGB image. Reshape of upstream
/// `convert_strip` over the whole `height × width` image as one launch.
///
/// Inputs:
/// - `linear_rgb`: interleaved RGB, length `width * height * 3`. Already
///   linearized (caller handles sRGB→linear).
/// - `primaries_matrix`: optional 3x3 RGB primaries transform. Currently
///   panics if Some — TODO add GPU primaries kernel.
///
/// Outputs (each `padded_width * height` floats; padded rows are
/// edge-replicated from the last in-row pixel):
/// - `xyb_x`, `xyb_y`, `xyb_b`
///
/// Output length matches upstream's contract: rows `[0, height)` are
/// written; rows `[height, padded_height)` are NOT written here (caller
/// is responsible for bottom padding).
#[allow(clippy::too_many_arguments)]
pub fn convert_image_to_xyb_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    width: usize,
    height: usize,
    padded_width: usize,
    linear_rgb: &[f32],
    primaries_matrix: Option<&[[f32; 3]; 3]>,
    xyb_x: &mut [f32],
    xyb_y: &mut [f32],
    xyb_b: &mut [f32],
) {
    assert_eq!(linear_rgb.len(), width * height * 3);
    assert!(xyb_x.len() >= padded_width * height);
    assert!(xyb_y.len() >= padded_width * height);
    assert!(xyb_b.len() >= padded_width * height);

    // Deinterleave the whole image into three planar buffers (host side).
    let n = width * height;
    let mut row_r = vec![0.0_f32; n];
    let mut row_g = vec![0.0_f32; n];
    let mut row_b = vec![0.0_f32; n];
    for i in 0..n {
        let si = i * 3;
        row_r[i] = linear_rgb[si];
        row_g[i] = linear_rgb[si + 1];
        row_b[i] = linear_rgb[si + 2];
    }

    // Apply primaries matrix if needed (host-side for now).
    if let Some(m) = primaries_matrix {
        apply_matrix_3x3_inplace(&mut row_r, &mut row_g, &mut row_b, m);
    }

    // Single GPU XYB launch over the whole image.
    let (gpu_x, gpu_y, gpu_b) = enc.xyb_from_linear_rgb(&row_r, &row_g, &row_b);

    // Scatter back into the padded output buffers, replicating the
    // right edge for each row's `[width, padded_width)` columns.
    let pad_cols = padded_width - width;
    if pad_cols == 0 {
        // Hot path: no padding, just copy.
        let dst_len = width * height;
        xyb_x[..dst_len].copy_from_slice(&gpu_x);
        xyb_y[..dst_len].copy_from_slice(&gpu_y);
        xyb_b[..dst_len].copy_from_slice(&gpu_b);
    } else {
        for y in 0..height {
            let src_off = y * width;
            let dst_off = y * padded_width;
            xyb_x[dst_off..dst_off + width].copy_from_slice(&gpu_x[src_off..src_off + width]);
            xyb_y[dst_off..dst_off + width].copy_from_slice(&gpu_y[src_off..src_off + width]);
            xyb_b[dst_off..dst_off + width].copy_from_slice(&gpu_b[src_off..src_off + width]);
            // Replicate right edge
            let last_x = xyb_x[dst_off + width - 1];
            let last_y = xyb_y[dst_off + width - 1];
            let last_b = xyb_b[dst_off + width - 1];
            for px in width..padded_width {
                xyb_x[dst_off + px] = last_x;
                xyb_y[dst_off + px] = last_y;
                xyb_b[dst_off + px] = last_b;
            }
        }
    }
}

/// Host-side 3x3 primaries matrix multiply, in-place on the three planes.
/// Mirrors upstream `apply_matrix_3x3`.
fn apply_matrix_3x3_inplace(
    row_r: &mut [f32],
    row_g: &mut [f32],
    row_b: &mut [f32],
    m: &[[f32; 3]; 3],
) {
    assert_eq!(row_r.len(), row_g.len());
    assert_eq!(row_r.len(), row_b.len());
    for i in 0..row_r.len() {
        let r = row_r[i];
        let g = row_g[i];
        let b = row_b[i];
        row_r[i] = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        row_g[i] = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        row_b[i] = m[2][0] * r + m[2][1] * g + m[2][2] * b;
    }
}

/// Convenience: helper Vec returns (no caller-allocated outputs needed).
/// Returns `(xyb_x, xyb_y, xyb_b)` each of length `padded_width * height`.
pub fn convert_image_to_xyb_gpu_alloc<R: Runtime>(
    enc: &GpuEncoder<R>,
    width: usize,
    height: usize,
    padded_width: usize,
    linear_rgb: &[f32],
    primaries_matrix: Option<&[[f32; 3]; 3]>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = padded_width * height;
    let mut xyb_x = vec![0.0_f32; n];
    let mut xyb_y = vec![0.0_f32; n];
    let mut xyb_b = vec![0.0_f32; n];
    convert_image_to_xyb_gpu(
        enc,
        width,
        height,
        padded_width,
        linear_rgb,
        primaries_matrix,
        &mut xyb_x,
        &mut xyb_y,
        &mut xyb_b,
    );
    (xyb_x, xyb_y, xyb_b)
}
