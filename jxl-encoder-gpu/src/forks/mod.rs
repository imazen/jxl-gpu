//! Forked-and-modified copies of `jxl-encoder` pipeline functions.
//!
//! Each module here mirrors the structure of an upstream file in
//! `jxl_encoder::vardct::*`, with the parallel work replaced by GPU
//! calls and the algorithm reshaped for GPU-friendly batching
//! (whole-image launches instead of per-row strips, batched per-
//! strategy DCT instead of per-block, etc.).
//!
//! ## Naming convention
//!
//! `forks::FOO` corresponds to `jxl_encoder::vardct::FOO`. So
//! `forks::xyb::convert_image_to_xyb_gpu` is the GPU substitute for
//! `jxl_encoder::vardct::xyb::convert_strip`, `forks::transform::
//! apply_dct_batch_gpu` is the GPU substitute for
//! `jxl_encoder::vardct::transform::Transform::apply_dct`, etc.
//!
//! ## When to use which fork module
//!
//! | Module | Substitutes | Reshape |
//! |---|---|---|
//! | [`xyb`] | `vardct::xyb` | whole-image batch instead of per-row strips |
//! | [`gaborish`] | `vardct::gaborish` | 3 sequential GPU launches instead of `rayon::join` |
//! | [`noise`] | `vardct::noise::denoise_xyb` | 3 sequential GPU launches; same Y-snapshot read pattern |
//! | [`adaptive_quant`] | `vardct::adaptive_quant` (`compute_mask1x1` + `pre_erosion` + `per_block_modulations`) | direct GPU substitution; `fuzzy_erosion` stays CPU |
//! | [`reconstruct`] | `vardct::reconstruct::{gab_smooth, xyb_to_linear_rgb_planar}` | 3 sequential launches |
//! | [`transform`] | `vardct::transform::Transform::apply_dct` | per-strategy batched: gather all blocks of one strategy, single launch covers all |
//! | [`cfl`] | `vardct::chroma_from_luma::find_best_multiplier` | single-tile + multi-tile batched (one launch covers all tiles) |
//! | [`epf`] | `vardct::epf::{compute_inv_sigma_map, apply_epf step1+step2}` | direct; step 0 (12-tap) stays CPU |
//! | [`dequant`] | `vardct::quantize::adjust_quant_bias` + `vardct::reconstruct` DequantBlock for DCT8 | 3-channel batched DCT8 dequant |
//! | [`quantize`] | `vardct::quantize::{default_thresholds, quantize_ac_block (DCT8)}` | one launch per channel |
//! | [`cost`] | `vardct::ac_strategy::estimate_entropy_full` leaves | batched per-strategy entropy + L2 + 8th-power norm |
//! | [`pad`] | `vardct::epf::pad_plane_into` | single GPU launch instead of 4 separate copy phases |
//!
//! Most users will go through [`crate::lossy_encoder::LossyEncoder`]
//! which composes these directly. The fork modules are exposed for
//! callers building their own pipeline (e.g., a hybrid CPU/GPU
//! encoder that uses jxl-encoder's bitstream layer but offloads the
//! parallel pipeline stages here).
//!
//! ## License
//!
//! Forked code retains its origin attribution (BSD-3-Clause via libjxl,
//! AGPL/commercial via jxl-encoder dual-license). Modifications are
//! AGPL-3.0-or-later or commercial per `LICENSE-AGPL3` /
//! `LICENSE-COMMERCIAL` at the repo root.

pub mod adaptive_quant;
pub mod cfl;
pub mod cost;
pub mod dequant;
pub mod epf;
pub mod gaborish;
pub mod noise;
pub mod pad;
pub mod quantize;
pub mod reconstruct;
pub mod transform;
pub mod xyb;
