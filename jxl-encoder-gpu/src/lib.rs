// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! GPU-backed kernels for `jxl-encoder` via [CubeCL](https://github.com/tracel-ai/cubecl).
//!
//! Each public function here mirrors a `_scalar` reference function in
//! `jxl-encoder-simd`. The scalar reference is the parity target for the
//! corresponding kernel (sub-ulp for pointwise math; documented tolerance
//! otherwise).
//!
//! Backend selection is via cargo features: `cuda` (default), `wgpu`, `hip`,
//! `cpu`. Multiple may be enabled; the caller chooses one at runtime by
//! passing the matching `Runtime` type to the launcher functions.

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

pub use cubecl;
