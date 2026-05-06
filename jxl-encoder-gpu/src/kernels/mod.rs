//! `#[cube]` kernel definitions.
//!
//! Every public kernel here has a host-side launcher in `crate::launch::*`
//! and a parity example under `examples/` proving it matches the
//! corresponding `jxl_encoder_simd::*_scalar` function.

pub mod adaptive_quant;
pub mod block_l2;
pub mod cfl;
pub mod dct16;
pub mod dct32;
pub mod dct4;
pub mod dct64;
pub mod dct8;
pub mod dequant;
pub mod entropy;
pub mod epf;
pub mod gab;
pub mod gaborish;
pub mod mask1x1;
pub mod pixel_loss;
pub mod quantize;
pub mod xyb;
