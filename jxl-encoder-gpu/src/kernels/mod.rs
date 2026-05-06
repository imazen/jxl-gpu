//! `#[cube]` kernel definitions.
//!
//! Every public kernel here has a host-side launcher in `crate::launch::*`
//! and a parity example under `examples/` proving it matches the
//! corresponding `jxl_encoder_simd::*_scalar` function.

pub mod dct8;
pub mod gab;
pub mod gaborish;
pub mod mask1x1;
pub mod xyb;
