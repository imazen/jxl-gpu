// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/gaborish.rs (BSD-3-Clause via libjxl
// + AGPL/commercial), with the per-channel SIMD call substituted for a
// GPU launch.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted gaborish-inverse 5×5 sharpening.
//!
//! Reshape vs upstream `jxl_encoder::vardct::gaborish::gaborish_inverse`:
//! - **Three sequential GPU launches instead of rayon::join across CPU
//!   threads.** On GPU the bottleneck is kernel occupancy, not host
//!   thread count — three back-to-back launches keep the SMs busy
//!   without competing CPU↔GPU memcpy bandwidth.
//! - **Returned Vec instead of in-place mutation.** The GPU encoder
//!   returns a new buffer per call; we copy back into the caller's
//!   slice for API compatibility with the upstream signature shape.
//!   (Future: have the GPU kernel write into a caller-provided buffer
//!   to skip the final copy.)
//!
//! The kernel constants (`K_GABORISH`) are duplicated bit-for-bit from
//! upstream so weight computation matches exactly.

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

/// Butteraugli-optimized 5x5 symmetric kernel weights — duplicated from
/// upstream so weight derivation matches exactly.
const K_GABORISH: [f64; 5] = [
    -0.09495815671340026,
    -0.041031725066768575,
    0.013710004822696948,
    0.006510206083837737,
    -0.0014789063378272242,
];

/// Compute normalized weights for one channel — bit-for-bit copy of
/// upstream `compute_weights`.
fn compute_weights(mul: f64) -> (f32, f32, f32, f32, f32, f32) {
    let sum = 1.0
        + mul
            * 4.0
            * (K_GABORISH[0] + K_GABORISH[1] + K_GABORISH[2] + K_GABORISH[4] + 2.0 * K_GABORISH[3]);
    let sum = if sum < 1e-5 { 1e-5 } else { sum };
    let normalize = 1.0 / sum;
    let normalize_mul = mul * normalize;
    (
        normalize as f32,
        (normalize_mul * K_GABORISH[0]) as f32,
        (normalize_mul * K_GABORISH[1]) as f32,
        (normalize_mul * K_GABORISH[2]) as f32,
        (normalize_mul * K_GABORISH[3]) as f32,
        (normalize_mul * K_GABORISH[4]) as f32,
    )
}

/// Apply gaborish inverse to one channel via GPU. In-place semantics
/// (matches upstream `apply_channel`); ignores the scratch buffer
/// argument since the GPU kernel manages its own output.
pub fn apply_channel_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    data: &mut [f32],
    width: usize,
    height: usize,
    mul: f64,
) {
    assert_eq!(data.len(), width * height);
    let (wc, wr, wd, w_big_r, wl, w_big_d) = compute_weights(mul);
    let out = enc.gaborish_5x5_channel(
        data, width as u32, height as u32, wc, wr, wd, w_big_r, wl, w_big_d,
    );
    data.copy_from_slice(&out);
}

/// Apply gaborish inverse to all three XYB channels via GPU.
///
/// Mirrors upstream `gaborish_inverse` API. mul=1.0 for all channels,
/// matching libjxl VarDCT default. Three sequential GPU launches.
pub fn gaborish_inverse_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &mut [f32],
    xyb_y: &mut [f32],
    xyb_b: &mut [f32],
    width: usize,
    height: usize,
) {
    apply_channel_gpu(enc, xyb_x, width, height, 1.0);
    apply_channel_gpu(enc, xyb_y, width, height, 1.0);
    apply_channel_gpu(enc, xyb_b, width, height, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_normalization() {
        let (wc, wr, wd, w_big_r, wl, w_big_d) = compute_weights(1.0);
        let sum = wc + 4.0 * wr + 4.0 * wd + 4.0 * w_big_r + 8.0 * wl + 4.0 * w_big_d;
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_uniform_image_preserved_gpu() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let width = 16;
        let height = 16;
        let value = 0.5_f32;
        let mut data = vec![value; width * height];
        apply_channel_gpu(&enc, &mut data, width, height, 1.0);
        for v in &data {
            assert!((v - value).abs() < 1e-5);
        }
    }
}
