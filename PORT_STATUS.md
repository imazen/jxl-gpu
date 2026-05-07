# Port Status

Living status report. Every kernel claim must be verified by a parity example
(see `examples/`) before it lands here as ✓.

## Status legend

- ✓ — Implemented + parity verified vs `jxl-encoder-simd` scalar reference (sub-ulp / documented tolerance)
- ⚙ — Kernel exists, parity not yet verified
- ✗ — Implementation in progress / known broken
- ⏸ — Deferred (see notes)
- ❌ — Not started

## Phase 1 — Pointwise + separable kernels

Verified on RTX 5070 + CUDA 13.2 (cubecl-cuda 0.10.0-pre.4).

| Kernel | Parity ref (jxl-encoder-simd) | Status | Tolerance achieved |
|---|---|---|---|
| `xyb_forward` | `forward_xyb_scalar` | ✓ | 1.19e-7 abs (forward), 1.30e-6 (inverse) |
| `xyb_inverse` | `inverse_xyb_planar_scalar` | ✓ | 1.30e-6 abs |
| `gab_smooth` (3x3) | `gab_smooth_scalar` | ✓ | 5.96e-8 abs |
| `gaborish_5x5` | `gaborish_5x5_scalar` | ✓ | 1.19e-7 abs |
| `mask1x1` | `compute_mask1x1_scalar` | ✓ | 6.79e-4 abs (FMA-contraction noise on `fast_log2f`; ~7e-6 relative on output range) |
| `denoise` | `noise::denoise_channel_scalar` | ✓ | 5.96e-8 abs (257×191, FMA-contraction noise) |
| `pad_plane` | `epf::pad_plane` | ✓ | 0.0 abs (bit-exact, 128x96 with pad=4) |

## Phase 2 — Per-block kernels

| Kernel | Parity ref | Status | Tolerance |
|---|---|---|---|
| `dct_8x8` | `dct8::dct_8x8_scalar` | ✓ | 2.98e-8 abs (256 blocks) |
| `idct_8x8` | `dct8::idct_8x8_scalar` | ✓ | 1.79e-7 abs; roundtrip 2.24e-7 |
| `dct_16x16` | `dct_16x16_scalar` | ✓ | 5.96e-8 abs (64 blocks) |
| `idct_16x16` | `idct_16x16_scalar` | ✓ | 5.36e-7 abs; roundtrip 4.47e-7 |
| `dct_16x8` | `dct_16x8_scalar` | ✓ | 3.73e-8 abs (64 blocks) |
| `dct_8x16` | `dct_8x16_scalar` | ✓ | 4.01e-8 abs (64 blocks) |
| `idct_16x8` | `idct_16x8_scalar` | ✓ | 3.58e-7 abs |
| `idct_8x16` | `idct_8x16_scalar` | ✓ | 3.13e-7 abs; roundtrip 3.87e-7 |
| `dct_16x16` | `dct_16x16_scalar` | ✓ | 5.96e-8 abs |
| `idct_16x16` | `idct_16x16_scalar` | ✓ | 5.36e-7 abs; rt 4.47e-7 |
| `dct_16x8` / `dct_8x16` + IDCTs | `dct16::*_scalar` | ✓ | 3.7e-8 to 3.6e-7 abs |
| `dct_32x32` / `32x16` / `16x32` + IDCTs | `dct32::*_scalar` | ✓ | fwd 4.5-6e-8, inv 6e-7 to 1e-6, rt 8e-7 to 1.1e-6 |
| `dct_64x64` / `64x32` / `32x64` + IDCTs | `dct64::*_scalar` | ✓ | fwd 4.5-6e-8, inv 1.9-2.4e-6, rt 1.2-1.9e-6 |
| `dct_4x4_full` / `4x8_full` / `8x4_full` + IDCTs | `dct4::*_scalar` | ✓ | 3.4e-8 to 1.8e-7; rt 1.2-1.8e-7 |
| `quantize_dct8` | `quantize::quantize_dct8_scalar` | ✓ | 0 diffs (bit-exact, 64 blocks) |
| `quantize_large` | `quantize::quantize_large_scalar` | ✓ | 0 diffs (bit-exact, 32 blocks 16x16 with LLF 2x2) |
| `dequant_dct8` | `dequant_dct8_scalar` | ✓ | Y bit-exact; X/B 1.53e-5 abs (FMA noise) |
| `block_l2` | `compute_block_l2_errors_scalar` | ✓ | 1.40e-9 abs (16x12 blocks) |
| `pixel_loss` | `pixel_domain_loss_scalar` | ✓ | 3.75e-16 rel (f64, 256 blocks) |
| `cfl_find_best_multiplier` | `cfl_find_best_multiplier_scalar` | ✓ | 0 diffs (bit-exact, 32 tiles of 256 elements) |
| `cfl_find_best_multiplier_newton` | `cfl_find_best_multiplier_newton_scalar` | ✓ | 0 diffs (bit-exact, 32 tiles × 256 elements, 10 iters) |
| `compute_pre_erosion` | `compute_pre_erosion_scalar` | ✓ | 3.43e-7 rel; 1.53e-5 abs (256x192 → 50x38, caller pre-allocates output) |
| `per_block_modulations` | `per_block_modulations_scalar` | ✓ | 2.71e-6 rel (16x12 blocks; abs 1.76e-2 because output spans 0.3-6500 from fast_pow2f) |
| `epf_step2` | `epf_step2_scalar` | ✓ | 3.7e-9 to 4.5e-8 abs (3 channels) |
| `epf_step1` | `epf_step1_scalar` | ✓ | 4.7e-9 to 6.0e-8 abs (3 channels, 5-pos SAD) |
| `fused_dct8_entropy` | `fused_dct8::fused_dct8_entropy_fallback` | ⏸ | use dct8 + entropy_coeffs chain instead (no register-residency win on GPU) |
| `entropy_coeffs_pixel` | `entropy_coeffs_scalar` (pixel_domain=true) | ✓ | out bit-exact, err 3.8e-6 abs (64 blocks of 64) |
| `entropy_coeffs_coeff` | `entropy_coeffs_scalar` (pixel_domain=false) | ✓ | 3.8e-6 abs |

## Phase 3 — Whole-image AC strategy search

| Component | Status |
|---|---|
| Per-strategy whole-image cost grid kernels | ⚙ | Single-channel cost grids for DCT8, DCT16x8, DCT8x16, DCT16x16, DCT32x16, DCT16x32, DCT32x32, DCT64x64 (`pipeline.rs::compute_cost_grid_*`); needs 3-channel + CfL + DCT64x32/32x64 + DCT4 family |
| Host-side partition selector | ✓ | Full coverage of standard rectangular family across 16/32/64 region tiers (12 variants). 12 unit tests passing. End-to-end integration demo verifies algorithmic correctness on synthetic smooth+noisy input |
| Refactor `ac_strategy_search.rs` to consume cost grids | ⛔ | requires touching `jxl-encoder` crate (separate repo) — needs user permission |

## Phase 4 — encoder facade (in THIS repo)

Per project plan: `imazen/jxl-gpu` is a **separate** project from
`jxl-encoder` — not a fork or feature flag. The integration work below
all happens IN THIS REPO; downstream encoders depend on it.

| Component | Status |
|---|---|
| `GpuEncoder<R: Runtime>` facade | ✓ | 42+ public methods covering xyb, gaborish, mask1x1, gab_smooth, all 13 DCT/IDCT pairs, quantize, dequant, EPF step1/step2, pre_erosion, per_block_modulations, cfl LS + Newton, block_l2, pixel_loss, entropy_coeffs_pixel, encode_lossy_via_cpu |
| Per-(width, height) instance pre-allocation cache | ❌ | (currently per-call upload) |
| End-to-end roundtrip test through `jxl-rs` decoder | ❌ | (waiting on full GPU bitstream path) |
| Optional dep on `jxl-encoder` / `jxl-encoder-simd` for sequential parts | ✓ | `feature = "encoder"` (default-on); used for ANS, bitstream, container muxing |

## Phase 5 — `forks::*` substitutions of jxl-encoder pipeline stages

Per-user authorization to fork-and-modify jxl-encoder source code.
Each `forks::*` module mirrors a `jxl_encoder::vardct::*` file with
the SIMD work substituted for GPU launches and the algorithm reshaped
for GPU-friendly batching.

| Fork module | Mirrors upstream | Status | Reshape |
|---|---|---|---|
| `forks::xyb` | `vardct::xyb::convert_strip` | ✓ | Whole-image batch instead of per-row strips; planar deinterleave on host; right-edge pad after XYB |
| `forks::gaborish` | `vardct::gaborish::gaborish_inverse` | ✓ | 3 sequential GPU launches instead of rayon::join; bit-for-bit `K_GABORISH` constants |
| `forks::adaptive_quant` | `vardct::adaptive_quant::{compute_mask1x1, compute_pre_erosion, per_block_modulations}` | ✓ | mask1x1 = mask1x1_field + Symmetric5 blur via gaborish_5x5 weights; fuzzy_erosion stays CPU |
| `forks::reconstruct` | `vardct::reconstruct::{gab_smooth, xyb_to_linear_rgb_planar, xyb_to_linear_rgb}` | ✓ | 3 sequential GPU launches; planar GPU output then host re-interleave |
| `forks::transform` | `vardct::transform::Transform::apply_dct` | ✓ | Per-strategy batched: gather all blocks of one strategy into one Vec<f32>, single GPU launch covers all of them. 13 strategies (DCT8/4/16/32/64 family + rectangulars). Inverse symmetric. |
| `forks::cfl` | `vardct::chroma_from_luma::find_best_multiplier` | ✓ | Single-tile drop-in API + multi-tile batched (one launch covers all tiles); LS + Newton variants |
| `forks::epf` | `vardct::epf::{compute_inv_sigma_map, apply_epf step1+step2}` | ✓ | Step 1 + Step 2 GPU; Step 0 (12-tap) and sharpness selection stay CPU |
| `forks::dequant` | `vardct::quantize::adjust_quant_bias` + `vardct::reconstruct` DequantBlock for DCT8 | ✓ | 3-channel batched DCT8 dequant in one launch; scalar `adjust_quant_bias` bit-for-bit |
| `forks::quantize` | `vardct::quantize::{default_thresholds, quantize_ac_block (DCT8 path)}` | ✓ | One launch per channel; 3-channel convenience helper computes channel-specific thresholds; non-DCT8 strategies stay CPU |

**Composition demos:**
- `examples/forks_pipeline_demo.rs` — chains XYB + gaborish + mask1x1 + DCT8 forward+inverse on 64×64 (DCT8 roundtrip 2.4e-7 abs)
- `examples/lossy_roundtrip_demo.rs` — full GPU per-block lossy path through 6 fork modules end-to-end

**Test coverage:** 31 unit tests pass on RTX 5070 + CUDA 13.2 (5 scalar + 26 GPU).

**Not yet covered (CPU path stays):** EPF Step 0 (12-tap), `compute_epf_sharpness`, fuzzy_erosion, AdjustQuantBlockAC heuristics, non-DCT8 strategies for quantize/dequant, `estimate_entropy_full` orchestration, `apply_dct` for IDENTITY/DCT2X2/AFV0-3.

## Coverage summary

- Phase 1: 7 of 7 ✓ (xyb fwd/inv, gab, gaborish_5x5, mask1x1, denoise, pad_plane)
- Phase 2 DCT/IDCT: 26 of 26 ✓ (all 8/16/32/64 squares, rectangulars,
  and 4-family sub-block variants)
- Phase 2 other: 13 of ~13 ✓ (quantize_dct8, quantize_large,
  dequant_dct8, block_l2, pixel_loss, cfl_find_best_multiplier +
  Newton, compute_pre_erosion, per_block_modulations, epf_step1,
  epf_step2, entropy_coeffs_pixel + coeff)
- Phase 3: 1.5 of 3 (cost grids partial; partition selector ✓)
- Phase 4: 2 of 4 (encoder facade ✓, jxl-encoder dep ✓)
- Phase 5: 9 of ~12 fork modules ✓ (xyb, gaborish, adaptive_quant,
  reconstruct, transform, cfl, epf, dequant, quantize; remaining:
  fuzzy_erosion, EPF Step 0, full estimate_entropy_full)

**Grand total: 56 of ~65 deliverables verified (~86%)**

DCT/IDCT family complete. CfL complete. EPF Steps 1+2 complete.
9 fork modules verified composing through GpuEncoder. Remaining
work concentrates in (a) fuzzy_erosion + EPF Step 0 GPU kernels,
(b) full estimate_entropy_full orchestration, (c) GPU kernels for
non-DCT8 strategies' quantize/dequant.

### Note on AC strategy search (Phase 3)

The Phase 2 kernels listed are the "naive" port (one cube per block,
single-threaded). For the whole-image-per-strategy AC search planned in
Phase 3, the per-block kernels should be re-emitted as cooperative
(cube_dim = 8 or 64, intra-block parallelism) for higher throughput.
The naive versions stay as the parity baseline.
