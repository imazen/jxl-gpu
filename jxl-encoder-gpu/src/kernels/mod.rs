//! `#[cube]` kernel definitions.
//!
//! Every public kernel here has a host-side launcher in `crate::launch::*`
//! and a parity example under `examples/` proving it matches the
//! corresponding `jxl_encoder_simd::*_scalar` function.

pub mod adaptive_quant;
pub mod afv;
pub mod afv_compose;
pub mod block_l2;
pub mod cfl;
pub mod dc_grid;
pub mod dc_restore;
pub mod dct16;
pub mod dct2x2;
pub mod dct32;
pub mod dct4;
pub mod dct4_raw;
pub mod dct64;
pub mod dct8;
pub mod denoise;
pub mod dequant;
pub mod dequant_simple;
pub mod entropy;
pub mod epf;
pub mod fused_dct_quant;
pub mod fuzzy_erosion;
pub mod gab;
pub mod gaborish;
pub mod gather;
pub mod histogram;
pub mod idct4_raw;
pub mod indexed_gather;
pub mod identity;
pub mod mask1x1;
pub mod pixel_loss;
pub mod quantize;
pub mod sse_reduce;
pub mod xyb;
