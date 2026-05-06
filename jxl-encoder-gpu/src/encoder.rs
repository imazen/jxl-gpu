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

use alloc::vec::Vec;

use cubecl::Runtime;
use cubecl::prelude::*;

use jxl_encoder::api::{LossyConfig, PixelLayout};

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
