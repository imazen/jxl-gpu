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
| `epf_step0` | inline CPU port of upstream `epf_step0_strip` | ✓ | 4.7e-10 to 4.5e-8 abs (3 channels, 12 neighbors × 5-pos SAD) — verified `examples/epf_step0_parity.rs` on CUDA |
| `fused_dct8_entropy` | `fused_dct8::fused_dct8_entropy_fallback` | ⏸ | use dct8 + entropy_coeffs chain instead (no register-residency win on GPU) |
| `entropy_coeffs_pixel` | `entropy_coeffs_scalar` (pixel_domain=true) | ✓ | out bit-exact, err 3.8e-6 abs (64 blocks of 64) |
| `entropy_coeffs_coeff` | `entropy_coeffs_scalar` (pixel_domain=false) | ✓ | 3.8e-6 abs |

## Phase 3 — Whole-image AC strategy search

| Component | Status |
|---|---|
| Per-strategy whole-image cost grid kernels | ✓ | 15 GPU-resident strategies × 2 flavors = 30 cost-grid functions: full DCT4/8/16/32/64 family (squares + rects) + IDENTITY + DCT2X2, single-channel proxy + 3-channel XYB-weighted + mask1x1-modulated. Validated end-to-end on real CLIC2025 photos (`phase3_xyb_real_image_demo`, `corpus_subblock_picks_demo`). **AFV0-3 cost grid integration done as of 2026-05-07** via host-side `forks::afv::afv_cost_grid_single_channel` + `afv_cost_grid_xyb_host` (host-orchestrated because the AFV transform composition itself stays on the host: 3 GPU launches per direction with per-block DC pack/unpack + corner mirroring). Remaining: CfL-aware variants (chroma-from-luma decorrelation in coefficient space) |
| Host-side partition selector | ✓ | Full 7-strategy coverage at 16×16 tier: DCT16×16 + 2× DCT16×8 + 2× DCT8×16 + Four DCT8×8 + per-cell `FourSubBlocks([SubStrategy; 4])` choosing from {DCT8, DCT4×4, DCT4×8, DCT8×4, IDENTITY, DCT2X2}. Recursive composition through 32×32 + 64×64 tiers. Corpus validation: pure 4-DCT8 essentially dies (0.1% of 32k regions) when sub-block alternatives offered |
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
| `forks::adaptive_quant` | `vardct::adaptive_quant::{compute_mask1x1, compute_pre_erosion, fuzzy_erosion, per_block_modulations}` | ✓ | full chain on GPU: mask1x1 = mask1x1_field + Symmetric5 blur via gaborish_5x5 weights; fuzzy_erosion = 3×3 min-of-4 weighted sum + 2× downsample (one thread per output pixel, gathers 4 input contributions to avoid the unsynchronized += race) |
| `forks::reconstruct` | `vardct::reconstruct::{gab_smooth, xyb_to_linear_rgb_planar, xyb_to_linear_rgb}` | ✓ | 3 sequential GPU launches; planar GPU output then host re-interleave |
| `forks::transform` | `vardct::transform::Transform::apply_dct` | ✓ | Per-strategy batched: gather all blocks of one strategy into one Vec<f32>, single GPU launch covers all of them. 13 strategies (DCT8/4/16/32/64 family + rectangulars). Inverse symmetric. |
| `forks::cfl` | `vardct::chroma_from_luma::find_best_multiplier` | ✓ | Single-tile drop-in API + multi-tile batched (one launch covers all tiles); LS + Newton variants |
| `forks::epf` | `vardct::epf::{compute_inv_sigma_map, apply_epf step1+step2}` | ✓ | Step 1 + Step 2 GPU; Step 0 (12-tap) and sharpness selection stay CPU |
| `forks::quantize` | `vardct::quantize::{default_thresholds, quantize_ac_block (full strategy family)}` | ✓ | One launch per channel; 3-channel convenience helper computes channel-specific thresholds; `quantize_blocks_gpu` dispatches DCT8 fast path or `quantize_large` for any (grid_w, grid_h, llf_x, llf_y) tuple |
| `forks::dequant` | `vardct::quantize::adjust_quant_bias` + `vardct::reconstruct` DequantBlock + generic per-coef dequant | ✓ | DCT8 path with CfL + adjust_quant_bias (`dequant_dct8_blocks_gpu`); generic `dequant_blocks_gpu` for arbitrary block size (DCT16+, AFV/IDENTITY/DCT2X2) |

**Composition demos:**
- `examples/forks_pipeline_demo.rs` — chains XYB + gaborish + mask1x1 + DCT8 forward+inverse on 64×64 (DCT8 roundtrip 2.4e-7 abs)
- `examples/lossy_roundtrip_demo.rs` — full GPU per-block lossy path through 6 fork modules end-to-end

**Test coverage:** 31 unit tests pass on RTX 5070 + CUDA 13.2 (5 scalar + 26 GPU).

### Validation status (G5.1 from zenmetrics CUBECL_GOTCHAS)

The G5.1 "validate against the published CPU crate, not a hand-rolled
re-derivation" rule says parity tests must call the upstream
function directly, not a hand-roll in the test file.

**G5.1-compliant (validated against `pub` upstream symbols):**

| Helper | Upstream symbol | Test |
|---|---|---|
| `forks::reconstruct::dc_from_dct_*` (8 helpers covering all DCT16+ strategies) | `jxl_encoder::vardct::dct::dc_from_dct_*` | `test_dc_from_dct_*_matches_upstream` (3-4 trials each, FP32 floor tolerance) |
| `forks::reconstruct::restore_llf_dct_*` (9 helpers) | (private upstream) | **transitively validated** via roundtrip: `restore_llf(dc_from_dct(llf)) == llf`. Single-position-on trials at all coefficient slots catch sign/scale/transpose bugs |
| `forks::cost::EntropyMulTable::{reference, experimental}` | `jxl_encoder::effort::EntropyMulTable` | `test_entropy_mul_table_*_matches_upstream` field-by-field |

**In-tree validated only (upstream symbol is private — needs
cross-repo work for G5.1 compliance):**

| Helper | Upstream symbol | Visibility | Current test |
|---|---|---|---|
| `examples/epf_step0_parity.rs` | `vardct::epf::epf_step0_strip` | `fn` private | inline copy-paste — risk of shared bug between GPU and hand-rolled CPU references |
| `forks::quantize::adjust_quant_block_ac_host` (+ 6 heuristics) | `VarDctEncoder::adjust_quant_block_ac` | `pub(crate)` impl method | property tests (skip-vs-fire, clamp arms) |
| `forks::cost::compute_scaled_constants` | `vardct::ac_strategy::compute_scaled_constants` | `pub(super) fn` | property test (`ratio == 1.0` echoes bases) |
| `forks::cfl::ytox_ratio` / `ytob_ratio` | `vardct::chroma_from_luma::*` | `pub fn` in `pub(crate) mod` | property tests (linearity, 0/1 baseline) — *would* be parity-testable if `vardct::chroma_from_luma` were `pub mod` |
| `forks::reconstruct::INV_DC_QUANT` | `vardct::quant::INV_DC_QUANT` | `pub const` in private mod | literal spot-check |

**Constants ported from private modules with literal-spot-check
tests** (acceptable because they're tiny `[f32; N]` arrays where
each value matches the upstream literal exactly): `MASK_CHANNEL_OFFSET`,
`CHANNEL_MUL`, `DCT_RESAMPLE_SCALE_*`, `K_INV_COLOR_FACTOR`.

**To resolve the in-tree-only entries**: bump visibility on the 4
private upstream symbols (`epf_step0_strip`, `adjust_quant_block_ac`,
`compute_scaled_constants`) plus re-export the `chroma_from_luma`
and `quant` modules at the upstream crate root. Cross-repo change
in `jxl-encoder`; gated on user approval.



**Not yet covered (CPU path stays):** `estimate_entropy_full` orchestration. **Mixed-strategy reconstruct on GPU as of 2026-05-07** — `forks::reconstruct::reconstruct_mixed_strategy_gpu` accepts a heterogeneous `&[BlockRecipe]` (each carrying `bx, by, raw_strategy, coeffs`), groups by strategy, emits ≤ 15 GPU launches per image (one per supported strategy that appears in the recipes). AFV0-3 still route through `forks::afv` separately. **`compute_epf_sharpness_dct8_gpu` fully composed as of 2026-05-07** — runs reconstruct → gaborish (opt) → per-candidate EPF + L2 → two-pass selection on GPU end-to-end for the DCT8-only path. **All per-strategy LLF restoration helpers ported as of 2026-05-07** — `forks::reconstruct::restore_llf_*` covers DCT16×8, DCT8×16, DCT16×16, DCT32×32, DCT32×16, DCT16×32, DCT64×64, DCT64×32, DCT32×64 (15 unit tests), each as a pure-scalar host helper. The 1×1-LLF strategies (IDENTITY/DCT2X2/DCT4×*/AFV0-3) reuse `restore_dct8_dc_override`'s simple DC formula. **AdjustQuantBlockAC fully ported as host helpers as of 2026-05-07** — pre-scan + all 6 heuristics A-F + orchestrator (`forks::quantize::adjust_quant_block_ac_host`) match upstream. A future `#[cube]` kernel can transcribe the now-standalone heuristics for per-block-parallel execution without further reverse-engineering. **EPF Step 0 (12-tap) ported and parity-verified at FP32 floor as of 2026-05-07** — closes the heaviest of the three EPF passes; all three are now on GPU. **DCT8-only reconstruct path on GPU as of 2026-05-07** — `forks::reconstruct::reconstruct_xyb_dct8_only_gpu` composes dequant + DC override + IDCT + scatter into 4 GPU launches per image. Sufficient for the all-blocks-are-DCT8 case (common for straightforward distance values). **All standard JXL AC strategy forward + inverse transforms are now on GPU** (DCT4/8/16/32/64 family, IDENTITY, DCT2X2, AFV0-3) as of 2026-05-07. **Quantize + dequant kernels cover the full strategy family** (DCT8 fast path + generic `quantize_large` / `dequant_simple` for any block size) as of 2026-05-07.

## Coverage summary

- Phase 1: 7 of 7 ✓ (xyb fwd/inv, gab, gaborish_5x5, mask1x1, denoise, pad_plane)
- Phase 2 DCT/IDCT: 32/32 ✓ (all 8/16/32/64 squares + rectangulars;
  4-family sub-block variants; raw 4×4 + 4×8 forward + inverse DCTs
  as primitives for AFV; AFV4×4 forward + inverse; **AFV0-3 forward
  + inverse composition done in `forks::afv`**)
- Phase 2 other: 14 of ~14 ✓ (quantize_dct8, quantize_large,
  dequant_dct8, block_l2, pixel_loss, cfl_find_best_multiplier +
  Newton, compute_pre_erosion, per_block_modulations, epf_step0,
  epf_step1, epf_step2, entropy_coeffs_pixel + coeff)
- Phase 3: 2.5 of 3 ✓ cost grids (15 strategies × 2 flavors), ✓
  partition selector (7-strategy 16×16 + recursive 32×32/64×64);
  ⛔ refactor of jxl-encoder ac_strategy_search.rs awaits permission
- Phase 4: 2 of 4 (encoder facade ✓, jxl-encoder dep ✓)
- Phase 5: 10 of ~12 fork modules ✓ (xyb, gaborish, adaptive_quant
  [full chain incl. fuzzy_erosion], reconstruct, transform [incl.
  IDENTITY+DCT2X2], cfl, epf, dequant, quantize, noise; remaining
  CPU bits inside existing forks: EPF Step 0, full
  estimate_entropy_full, AdjustQuantBlockAC heuristics)

**Grand total: 56 of ~65 deliverables verified (~86%)**

DCT/IDCT family complete. CfL complete. **All three EPF passes
complete** (Step 0 ported and parity-verified at FP32 floor on
2026-05-07; Steps 1 + 2 done in earlier phases). AFV cost grid
integration complete (host-side wrappers in `forks::afv` consuming
`afv_weights()` + the existing batched AFV transforms). 11 fork
modules verified composing through GpuEncoder. Sharpness picker
two-pass selection logic ported as pure-CPU helper.
**AdjustQuantBlockAC fully ported** (pre-scan + 6 heuristics A-F
+ orchestrator). **DCT8-only reconstruct path on GPU** (4 launches
per image — dequant + IDCT×3 + host DC override + host scatter).
**`compute_epf_sharpness_dct8_gpu` end-to-end** (reconstruct →
gaborish → per-candidate EPF + L2 → two-pass picker, all on GPU
plus the small host-side picker logic). **Mixed-strategy reconstruct
landed** (`reconstruct_mixed_strategy_gpu`: groups
`&[BlockRecipe]` by strategy, ≤ 15 GPU launches/image, AFV0-3
routed through `forks::afv` separately). Remaining work
concentrates in (a) full estimate_entropy_full orchestration,
(b) optional GPU kernels: per-block AdjustQuantBlockAC batch
parallelism, sharpness candidate fan-out.

### Note on AC strategy search (Phase 3)

The Phase 2 kernels listed are the "naive" port (one cube per block,
single-threaded). For the whole-image-per-strategy AC search planned in
Phase 3, the per-block kernels should be re-emitted as cooperative
(cube_dim = 8 or 64, intra-block parallelism) for higher throughput.
The naive versions stay as the parity baseline.
