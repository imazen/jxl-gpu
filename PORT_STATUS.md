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
| `denoise` | `noise::denoise_channel` (no `_scalar` exported) | ❌ | — |
| `pad_plane` | `epf::pad_plane` | ❌ | — |

## Phase 2 — Per-block kernels

| Kernel | Parity ref | Status | Tolerance |
|---|---|---|---|
| `dct_8x8` | `dct8::dct_8x8_scalar` | ✓ | 2.98e-8 abs (256 blocks) |
| `idct_8x8` | `dct8::idct_8x8_scalar` | ✓ | 1.79e-7 abs; roundtrip 2.24e-7 |
| `dct_16x16` | `dct_16x16_scalar` | ✓ | 5.96e-8 abs (64 blocks) |
| `idct_16x16` | `idct_16x16_scalar` | ✓ | 5.36e-7 abs; roundtrip 4.47e-7 |
| `dct_8x16` / `dct_16x8` | `dct16::*_scalar` | ❌ | — |
| `dct_16x32` / `dct_32x16` / `dct_32x32` | `dct32::*_scalar` | ❌ | — |
| `dct_32x64` / `dct_64x32` / `dct_64x64` | `dct64::*_scalar` | ❌ | — |
| `dct_4x4` / `dct_4x8` / `dct_8x4` | `dct4::*_scalar` | ❌ | — |
| `idct_*` (matched inverses) | `idct{8,16,32,64}::*_scalar` | ❌ | — |
| `quantize_dct8` | `quantize::quantize_dct8_scalar` | ✓ | 0 diffs (bit-exact, 64 blocks) |
| `quantize_large` | `quantize::quantize_large_scalar` | ❌ | — |
| `dequant_dct8` | `dequant_dct8_scalar` | ✓ | Y bit-exact; X/B 1.53e-5 abs (FMA noise on CfL add) |
| `block_l2` | `compute_block_l2_errors_scalar` | ✓ | 1.40e-9 abs (16x12 blocks) |
| `pixel_loss` | `pixel_domain_loss_scalar` | ✓ | 3.75e-16 rel (f64, 256 blocks) |
| `compute_pre_erosion` | `adaptive_quant::compute_pre_erosion_scalar` | ❌ | — |
| `per_block_modulations` | `adaptive_quant::per_block_modulations_scalar` | ❌ | — |
| `cfl_find_best_multiplier` | `cfl::find_best_multiplier_scalar` | ❌ | — |
| `cfl_find_best_multiplier_newton` | `cfl::find_best_multiplier_newton_scalar` | ❌ | — |
| `epf_step1` / `epf_step2` | `epf::epf_step{1,2}_scalar` | ❌ | — |
| `fused_dct8_entropy` | `fused_dct8::fused_dct8_entropy_scalar` | ❌ | — |
| `entropy_*` (estimate) | `entropy::*_scalar` | ❌ | — |

## Phase 3 — Whole-image AC strategy search

| Component | Status |
|---|---|
| Per-strategy whole-image cost grid kernels | ❌ |
| Host-side partition selector | ❌ |
| Refactor `ac_strategy_search.rs` to consume cost grids | ❌ |

## Phase 4 — jxl-encoder integration

| Component | Status |
|---|---|
| `gpu` feature in `jxl-encoder` Cargo.toml | ❌ |
| `Pipeline<R>` swap-in for SIMD entry points | ❌ |
| Per-(width, height) instance pre-allocation | ❌ |
| End-to-end roundtrip test (jxl-rs decoder) | ❌ |

## Coverage summary

- Phase 1: 5 of 7 kernels with parity (71%)
- Phase 2: 8 of 23 kernels with parity (35%)
- Phase 3: 0 of 3 components done (0%)
- Phase 4: 0 of 4 integration points done (0%)

**Grand total: 13 of 37 deliverables verified (35%)**

### Note on AC strategy search (Phase 3)

The Phase 2 kernels listed are the "naive" port (one cube per block,
single-threaded). For the whole-image-per-strategy AC search planned in
Phase 3, the per-block kernels should be re-emitted as cooperative
(cube_dim = 8 or 64, intra-block parallelism) for higher throughput.
The naive versions stay as the parity baseline.
