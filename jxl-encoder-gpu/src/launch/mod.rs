//! Host-side launchers for kernels in `crate::kernels::*`.
//!
//! Each launcher is generic over `Runtime: cubecl::Runtime` so callers
//! select the backend at construction time (e.g.
//! `jxl_encoder_gpu::launch::xyb_forward::<cubecl::cuda::CudaRuntime>(...)`).

pub mod gab;
pub mod gaborish;
pub mod mask1x1;
pub mod xyb;
