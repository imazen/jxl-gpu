//! Host-side launchers for kernels in `crate::kernels::*`.
//!
//! Each launcher is generic over `Runtime: cubecl::Runtime` so callers
//! select the backend at construction time (e.g.
//! `jxl_encoder_gpu::launch::xyb_forward::<cubecl::cuda::CudaRuntime>(...)`).

pub mod adaptive_quant;
pub mod afv;
pub mod aq_field;
pub mod afv_compose;
pub mod block_l2;
pub mod cfl;
pub mod cfl_collect;
pub mod cfl_quantize;
pub mod nzeros_count;
pub mod quantize_dc;
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
pub mod entropy_3ch;
pub mod epf;
pub mod fused_dct8_3ch;
pub mod fused_dct_quant;
pub mod fuzzy_erosion;
pub mod gab;
pub mod gab_3ch;
pub mod gaborish;
pub mod gather;
pub mod histogram;
pub mod idct4_raw;
pub mod identity;
pub mod indexed_gather;
pub mod indexed_scatter;
pub mod mask1x1;
pub mod mask_for_ac_strategy;
pub mod pixel_loss;
pub mod pixel_loss_3ch;
pub mod quantize;
pub mod set_dc;
pub mod set_llf;
pub mod sse_reduce;
pub mod u8_rgb_prepare;
pub mod xyb;
