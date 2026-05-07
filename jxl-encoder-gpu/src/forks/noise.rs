// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/noise.rs (BSD-3-Clause via libjxl +
// AGPL/commercial), with the SIMD calls substituted for GPU launches.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted Wiener denoise on XYB planes.
//!
//! Mirrors `jxl_encoder::vardct::noise::denoise_xyb`. Three sequential
//! GPU launches (one per channel) replace the upstream `rayon::join`
//! over the three CPU SIMD calls. All three reads are from the
//! caller-supplied `orig_*` snapshots (the encoder snapshots them up
//! front before any channel is mutated, exactly as upstream does).
//!
//! Unlike upstream's mutate-in-place signature, the GPU variant
//! returns three new `Vec<f32>` planes — keeping ownership clean and
//! letting the GPU side allocate output buffers itself. Caller can
//! `swap` them back into the XYB struct.

use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

/// Matches upstream's `DENOISE_FRACTION` constant.
pub const DENOISE_FRACTION: f32 = 0.25;

/// Compute the `denoise_scale` parameter from the encoder's
/// `quality_coef` (from `noise_quality_coef(distance)`).
///
/// Mirrors the formula at the top of upstream `denoise_xyb`:
/// `denoise_scale = DENOISE_FRACTION / (quality_coef * 1.4)`.
pub fn denoise_scale(quality_coef: f32) -> f32 {
    DENOISE_FRACTION / (quality_coef * 1.4)
}

/// GPU `denoise_xyb`. Mirrors upstream
/// `jxl_encoder::vardct::noise::denoise_xyb`.
///
/// Returns `(out_x, out_y, out_b)`. Each channel is filtered against
/// the same `xyb_y` snapshot for noise-LUT lookup, matching upstream
/// (the Y channel determines noise variance for ALL three channels).
pub fn denoise_xyb_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    noise_lut: &[f32; 8],
    quality_coef: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = width * height;
    assert_eq!(xyb_x.len(), n);
    assert_eq!(xyb_y.len(), n);
    assert_eq!(xyb_b.len(), n);

    let scale = denoise_scale(quality_coef);
    let w = width as u32;
    let h = height as u32;

    // Three sequential GPU launches. All read xyb_y for noise lookup;
    // each channel writes its own output. Sequential rather than
    // overlapped — cubecl 0.10 doesn't expose stream-level overlap,
    // and the per-launch cost is small enough that pipelining isn't
    // worth the complexity. Profile if needed.
    let out_x = enc.denoise_channel(xyb_x, xyb_y, noise_lut, w, h, scale);
    let out_y = enc.denoise_channel(xyb_y, xyb_y, noise_lut, w, h, scale);
    let out_b = enc.denoise_channel(xyb_b, xyb_y, noise_lut, w, h, scale);

    (out_x, out_y, out_b)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::encoder::GpuEncoder;

    type B = cubecl::cuda::CudaRuntime;

    #[test]
    fn test_denoise_xyb_gpu_shape() {
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 32_usize;
        let h = 24_usize;
        let n = w * h;
        let mut x = vec![0.0_f32; n];
        let mut y = vec![0.0_f32; n];
        let mut b = vec![0.0_f32; n];
        for i in 0..n {
            let u = (i % w) as f32 / (w - 1) as f32;
            let v = (i / w) as f32 / (h - 1) as f32;
            x[i] = (u - 0.5) * 0.4;
            y[i] = 0.1 + 0.6 * u + 0.2 * v;
            b[i] = (v - 0.5) * 0.5;
        }
        let lut: [f32; 8] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.20, 0.10];
        let (ox, oy, ob) = denoise_xyb_gpu(&enc, &x, &y, &b, w, h, &lut, 1.0);
        assert_eq!(ox.len(), n);
        assert_eq!(oy.len(), n);
        assert_eq!(ob.len(), n);
        for v in ox.iter().chain(oy.iter()).chain(ob.iter()) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_denoise_scale_quality_coef() {
        // d=1.0 → quality_coef=0.25 → scale = 0.25 / (0.25 * 1.4) ≈ 0.714
        let s = denoise_scale(0.25);
        assert!((s - 0.7142857).abs() < 1e-5, "scale={s}");
    }
}
