// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! GPU-backed JPEG XL encoder kernels via [CubeCL](https://github.com/tracel-ai/cubecl).
//!
//! ## Status
//!
//! GPU lossy DCT8 pipeline is **1.05–3.95× faster than CPU AVX2** at sizes
//! 256² → 2048² (RTX 5070 + Ryzen 9 7950X). Parity verified at 4e-6 max
//! abs delta vs `jxl-encoder-simd` scalar reference.
//!
//! ## Three layers of API
//!
//! 1. **[`lossy_encoder::LossyEncoder`]** — high-level, recommended for
//!    most users. One-shot + batch encode, `f32` + `u8` (sRGB) entry
//!    points, arbitrary image sizes, JPEG-style quality knob via
//!    [`lossy_encoder::quality_to_qac`].
//!
//!    ```no_run
//!    # #[cfg(feature = "cuda")] {
//!    use jxl_encoder_gpu::encoder::GpuEncoder;
//!    use jxl_encoder_gpu::lossy_encoder::{LossyEncoder, quality_to_qac};
//!
//!    type Backend = cubecl::cuda::CudaRuntime;
//!    let enc: GpuEncoder<Backend> = GpuEncoder::new();
//!    let lossy = LossyEncoder::new(&enc, 1024, 768);
//!
//!    // Get RGB U8 from any image source (e.g., the `image` crate).
//!    let rgb_in: Vec<u8> = vec![128; 1024 * 768 * 3];
//!    let qac = quality_to_qac(85.0); // JPEG-quality 85 → ~qac 2.0
//!    let rgb_out: Vec<u8> = lossy.encode_one_srgb_u8(&enc, &rgb_in, qac);
//!    # }
//!    ```
//!
//! 2. **[`persistent`]** — typed-handle GPU buffer API for callers
//!    who need fine-grained pipeline control. Use for chained custom
//!    pipelines, scientific/research use, etc.
//!
//! 3. **[`launch`] / [`kernels`]** — raw kernel launchers + cubecl
//!    `#[cube]` source. Use only when neither high-level layer fits.
//!
//! ## Reference parity
//!
//! Every public kernel here mirrors a `_scalar` reference function in
//! `jxl-encoder-simd`. The scalar reference is the parity target for the
//! corresponding kernel (sub-ulp for pointwise math; documented tolerance
//! otherwise).
//!
//! ## Backends
//!
//! Selectable via cargo features: `cuda` (default), `wgpu`, `hip`, `cpu`.
//! Multiple may be enabled; the caller chooses one at runtime by passing
//! the matching `Runtime` type to the launcher functions.
//!
//! ## Forks of `jxl-encoder` pipeline stages
//!
//! [`forks`] contains GPU-substituted variants of `jxl-encoder`'s
//! pipeline stages (XYB, gaborish, DCT/IDCT, quantize, dequant, EPF,
//! CfL multiplier search, etc.). Use these when building a hybrid
//! GPU/CPU encoder that delegates the parallel stages to GPU but
//! keeps the entropy coding / bitstream serialization on CPU
//! (`jxl-encoder` integration path).

#![cfg_attr(not(test), no_std)]

extern crate alloc;

#[cfg(feature = "encoder")]
pub mod encoder;
#[cfg(feature = "encoder")]
pub mod forks;
pub mod kernels;
pub mod launch;
#[cfg(feature = "encoder")]
pub mod lossy_encoder;
#[cfg(feature = "encoder")]
pub mod persistent;
pub mod pipeline;
pub mod quant_weights;

pub use cubecl;
