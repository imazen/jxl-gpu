//! Host-side launchers for kernels in `crate::kernels::*`.
//!
//! Each launcher is generic over `Runtime: cubecl::Runtime` so callers
//! select the backend at construction time (e.g.
//! `jxl_encoder_gpu::launch::xyb_forward::<cubecl::cuda::CudaRuntime>(...)`).

pub mod block_l2;
pub mod dct16;
pub mod dct8;
pub mod dequant;
pub mod gab;
pub mod gaborish;
pub mod mask1x1;
pub mod pixel_loss;
pub mod quantize;
pub mod xyb;
