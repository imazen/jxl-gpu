//! Forked-and-modified copies of jxl-encoder pipeline functions.
//!
//! Per the project plan: jxl-gpu may freely duplicate jxl-encoder source
//! code (per user authorization). Each module here mirrors the structure
//! of the upstream file it derives from, with the parallel work replaced
//! by GPU calls and the algorithm reshaped for GPU-friendly batching
//! (whole-image launches instead of per-row strips, etc.).
//!
//! ## Naming convention
//!
//! `forks::xyb` corresponds to `jxl_encoder::vardct::xyb`.
//! `forks::transform` corresponds to `jxl_encoder::vardct::transform`.
//! Etc.
//!
//! ## License
//!
//! Forked code retains its origin attribution (BSD-3-Clause via libjxl,
//! AGPL/commercial via jxl-encoder dual-license). Modifications are
//! AGPL-3.0-or-later or commercial per `LICENSE-AGPL3` /
//! `LICENSE-COMMERCIAL` at the repo root.

pub mod adaptive_quant;
pub mod cfl;
pub mod dequant;
pub mod epf;
pub mod gaborish;
pub mod reconstruct;
pub mod transform;
pub mod xyb;
