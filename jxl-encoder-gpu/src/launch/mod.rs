//! Host-side launchers for kernels in `crate::kernels::*`.
//!
//! Each launcher is generic over `Runtime: cubecl::Runtime` so callers
//! select the backend at construction time (e.g.
//! `jxl_encoder_gpu::launch::xyb_forward::<cubecl::cuda::CudaRuntime>(...)`).

pub mod adaptive_quant;
pub mod afv;
pub mod block_l2;
pub mod cfl;
pub mod dc_restore;
pub mod denoise;
pub mod dct16;
pub mod dct2x2;
pub mod dct32;
pub mod dct4;
pub mod dct4_raw;
pub mod dct64;
pub mod dct8;
pub mod dequant;
pub mod entropy;
pub mod epf;
pub mod fused_dct_quant;
pub mod gab;
pub mod gaborish;
pub mod fuzzy_erosion;
pub mod gather;
pub mod identity;
pub mod mask1x1;
pub mod pixel_loss;
pub mod quantize;
pub mod xyb;
