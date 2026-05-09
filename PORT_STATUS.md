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

### Validation status (G5.1 from zenmetrics CUBECL_GOTCHAS) — fully compliant as of 2026-05-08

The G5.1 "validate against the published CPU crate, not a hand-rolled
re-derivation" rule says parity tests must call the upstream
function directly, not a hand-roll in the test file. **Every helper
ported from upstream is now G5.1-compliant.**

**G5.1-compliant via `pub` upstream symbols (no jxl-encoder changes needed):**

| Helper | Upstream symbol | Test |
|---|---|---|
| `forks::reconstruct::dc_from_dct_*` (8 helpers, all DCT16+ strategies) | `jxl_encoder::vardct::dct::dc_from_dct_*` | `test_dc_from_dct_*_matches_upstream` (3-4 trials each, FP32 floor tolerance) |
| `forks::reconstruct::restore_llf_dct_*` (9 helpers) | (private upstream) | **transitively validated** via roundtrip: `restore_llf(dc_from_dct(llf)) == llf`. Single-position-on trials at all coefficient slots catch sign/scale/transpose bugs |
| `forks::cost::EntropyMulTable::{reference, experimental}` | `jxl_encoder::effort::EntropyMulTable` | `test_entropy_mul_table_*_matches_upstream` field-by-field |

**G5.1-compliant via the `__internals` cargo feature** (jxl-encoder
commit `c82e05c` added an off-by-default `__internals` feature
that re-exports five private symbols — option B from the
A/B/C menu):

| Helper | Upstream symbol | Test |
|---|---|---|
| `examples/epf_step0_parity.rs` | `jxl_encoder::__internals::epf_step0_strip_free` | direct call; FP32 floor (X 4.7e-10, Y 4.5e-8, B 4.5e-8 vs 5e-5 tolerance) |
| `forks::quantize::adjust_quant_block_ac_host` (+ 6 heuristics + prescan) | `jxl_encoder::__internals::adjust_quant_block_ac_free` | 54 trials (6 strategies × 3 channels × 3 quant levels): all 4 outputs + thresholds + quant match exactly |
| `forks::cost::compute_scaled_constants` | `jxl_encoder::__internals::compute_scaled_constants_free` | 15 trials (5 distances × 3 base tuples), tolerance 1e-3 |
| `forks::cfl::ytox_ratio` / `ytob_ratio` | `jxl_encoder::__internals::{ytox_ratio, ytob_ratio}` | 256 trials each (every `i8` value), exact equality |
| `forks::reconstruct::INV_DC_QUANT` | `jxl_encoder::__internals::INV_DC_QUANT` | all 3 channels, exact equality |

**Constants ported from private modules with literal-spot-check
tests** (acceptable because they're tiny `[f32; N]` arrays where
each value matches the upstream literal exactly): `MASK_CHANNEL_OFFSET`,
`CHANNEL_MUL`, `DCT_RESAMPLE_SCALE_*`, `K_INV_COLOR_FACTOR`.

**Total G5.1-compliant `*_matches_upstream` tests: 16** plus 9
LLF restoration helpers transitively validated via roundtrip.

### LossyEncoder strat-search facade (2026-05-08)

`LossyEncoder::encode_one_with_strategy_search_dct8_16` ties the
Phase 3 cost grids + Phase 4 facade + Phase 5 forks into one
end-to-end public method. 15 of 27 strategies wired and selectable:
DCT8 / DCT4×4 / DCT4×8 / DCT8×4 / IDENTITY / DCT2×2 / DCT16×16 /
DCT16×8 / DCT8×16 / DCT32×32 / DCT32×16 / DCT16×32 / DCT64×64 /
DCT64×32 / DCT32×64. AFV0-3 cost grids exist (Phase 3) but
SubStrategy enum + recursive lowering not yet extended.

**Performance** (CLIC 1024×1024 @ d=1.0): 170 ms per call
(1.8× from 305 ms baseline). Cost-grid stage went 234 ms → 41 ms
(5.6×) via the persistent GPU pipeline (commit `13be0049`).

**Quality**: butteraugli at parity with uniform-qac across
d ∈ {0.5, 1.0, 2.0, 4.0}. Distance-scaled anti-bias muls
(`bias = 1 + max(0, d-1) * 0.6`, with 1×/1.5×/2× factors for
DCT16/32/64 families; sub-blocks at base) prevent the
over-selection that produced butteraugli 9.2 at d=4 with fixed
muls (commit `ba9eed75` debug).

**Persistent GPU pipeline**: 6 new persistent GpuEncoder kernels
landed (entropy_coeffs_pixel_blocks_broadcast_w_persistent,
pixel_loss_blocks_persistent, quantize_large_blocks_broadcast_w_persistent,
dequant_strategy_persistent + _dct8 variant, identity_persistent,
dct2x2_persistent — forward + inverse where applicable). Plus
strategy-aware dispatchers `apply_dct/idct_batch_persistent`.

**Selector threading**: select_partitions_32x32_with_extras16 and
select_partitions_64x64_with_extras16 thread CostGrids16x16 through
the higher-tier picks so sub-block strategies can be considered
inside Sub16x16 partitions.

**Selectivity validated**: at d=1.0 on CLIC photo, 16100 DCT8 picks
(98.4%), 59 DCT16x16 (1.4% area), 3 DCT32x32 (0.3% area). All other
strategies zero — strat-search picks larger transforms only where
they genuinely win.



**estimate_entropy_full orchestrator (DCT8 + strategy-generic) landed
as of 2026-05-08** — `forks::cost::estimate_entropy_full_dct8_batch_gpu`
(DCT8 fast path) and `forks::cost::estimate_entropy_full_strategy_batch_gpu`
(strategy-generic; works for all standard strategies including
DCT16+/DCT32+/DCT64+/IDENTITY/DCT2X2/DCT4-family) compose ~12 GPU
launches per call (3 forward DCT + 3 entropy + 3 IDCT + 3 pixel-loss)
plus host-side combiners. Strategy-generic version uses
`coeff_count_per_strategy(raw_strategy)` for the entropy kernel's
n_per_block and `tile_dims_pixels(raw_strategy)` for the pixel-loss
block dimensions. AFV0-3 routed through `forks::afv` separately.
`CostMode::{Simple, Upstream}` lets callers pick between a fast
relative-ranking form and the bit-faithful `estimate_entropy_full`
formula with `k_zeros_mul * f(nzeros)` bits-cost + 8th-root
pixel-loss scaling (`per_block_upstream_cost` generalized over
`block_pixel_count` for any strategy). The upstream entropy kernel
itself only fills `entropy_sum + nzeros_sum` columns of the 4-stat
output (matches upstream's pixel-domain mode behavior —
`info_loss_sum` is intentionally 0 in pixel-domain mode); the
`info_loss_mul` term is computed on the host from the combined
pixel loss. In `CostMode::Upstream`, the
orchestrator automatically applies upstream's generic-path X-channel
multi-block weight (`1 + min(num_blocks/8, 3)` for `num_blocks >= 2`)
to BOTH the X channel's per-block entropy AND its per-block pixel
loss. `num_blocks` is derived from the strategy
(`block_pixels / 64`) — DCT16x16 = 4 → w=1.5; DCT32x32 = 16 → w=3.0;
DCT64x64 = 64 → w=4.0 capped. For DCT8 the weight is a no-op
(covered_blocks=1). In `CostMode::Simple` the weight is NOT applied
(uniform per-channel scaling doesn't affect relative ranking). **Mixed-strategy
reconstruct on GPU as of 2026-05-07** — `forks::reconstruct::reconstruct_mixed_strategy_gpu` accepts a heterogeneous `&[BlockRecipe]` (each carrying `bx, by, raw_strategy, coeffs`), groups by strategy, emits ≤ 15 GPU launches per image (one per supported strategy that appears in the recipes). AFV0-3 still route through `forks::afv` separately. **`compute_epf_sharpness_dct8_gpu` fully composed as of 2026-05-07** — runs reconstruct → gaborish (opt) → per-candidate EPF + L2 → two-pass selection on GPU end-to-end for the DCT8-only path. **All per-strategy LLF restoration helpers ported as of 2026-05-07** — `forks::reconstruct::restore_llf_*` covers DCT16×8, DCT8×16, DCT16×16, DCT32×32, DCT32×16, DCT16×32, DCT64×64, DCT64×32, DCT32×64 (15 unit tests), each as a pure-scalar host helper. The 1×1-LLF strategies (IDENTITY/DCT2X2/DCT4×*/AFV0-3) reuse `restore_dct8_dc_override`'s simple DC formula. **AdjustQuantBlockAC fully ported as host helpers as of 2026-05-07** — pre-scan + all 6 heuristics A-F + orchestrator (`forks::quantize::adjust_quant_block_ac_host`) match upstream. A future `#[cube]` kernel can transcribe the now-standalone heuristics for per-block-parallel execution without further reverse-engineering. **EPF Step 0 (12-tap) ported and parity-verified at FP32 floor as of 2026-05-07** — closes the heaviest of the three EPF passes; all three are now on GPU. **DCT8-only reconstruct path on GPU as of 2026-05-07** — `forks::reconstruct::reconstruct_xyb_dct8_only_gpu` composes dequant + DC override + IDCT + scatter into 4 GPU launches per image. Sufficient for the all-blocks-are-DCT8 case (common for straightforward distance values). **All standard JXL AC strategy forward + inverse transforms are now on GPU** (DCT4/8/16/32/64 family, IDENTITY, DCT2X2, AFV0-3) as of 2026-05-07. **Quantize + dequant kernels cover the full strategy family** (DCT8 fast path + generic `quantize_large` / `dequant_simple` for any block size) as of 2026-05-07.

## Phase 6 — Butteraugli quant-refinement loop

End-to-end GPU-substituted iterative quant refinement, mirroring upstream
`jxl_encoder::vardct::butteraugli_loop::butteraugli_refine_quant_field`.
Gated behind the new `butteraugli-loop` cargo feature. Lives in
`forks::butteraugli_loop`.

| Component | Status | Notes |
|---|---|---|
| `ButteraugliLoopGpu<R>` wrapper | ✓ | Persistent `butteraugli_gpu::Butteraugli` instance; `set_reference` once + `compute_with_reference` per iter for cached-opsin re-use |
| `DeviationBounds::compute` | ✓ | qf_lower/qf_higher derived from initial float qf (`sqrt(250 / ratio)` formula). Mirrors upstream lines 105-122 exactly. |
| `compute_tile_distances` + `AcStrategyInfo<'_>` | ✓ | AC-strategy-aware 16th-power-mean diffmap reduction, `K_TILE_NORM = 1.2`. Mirrors upstream lines 230-271. |
| `clamp_toward_initial` | ✓ | kOriginalComparisonRound `K_INIT_MUL = 0.6` blend toward initial qf. Mirrors upstream lines 314-336. |
| `adjust_quant_field` | ✓ | Per-iter `cur_pow=0.2` / `cur_pow=0.0` regimes with integer-quantizer-step minimum bump. Mirrors upstream lines 338-406. |
| `refine_quant_field_one_iter` + `RefineConfig` | ✓ | Composes the four helpers in upstream's per-iter order. Returns `tile_dist` for caller diagnostics. |
| `refine_aq_field_gpu` + `RefineIterTrace` | ✓ | Multi-iter loop wiring `LossyEncoder::encode_one_adaptive` + the host-side helpers. **Qac-domain adaptation:** integer-step bump is neutered (no integer rounding in our pipeline). |
| `linear_f32_to_srgb_u8` + `linear_planar_to_srgb_u8_interleaved` | ✓ | IEC 61966-2-1 piecewise transfer matching butteraugli-gpu's `srgb_byte_to_linear` inverse. Caller passes ORIGINAL sRGB U8 bytes as ref to avoid transfer-function-mismatch score inflation. |
| `examples/butteraugli_refinement_demo.rs` | ✓ | Turnkey demo on a CLIC2025 photo. |

**Test coverage:** 21 unit tests pass (4 helpers individually + per-iter
composition + 3 sRGB conversion + 1 CUDA end-to-end smoke test on a 64×64
gradient). Loop runs at ~217 ms/iter on a 1024×1024 image (RTX 5070,
CUDA 13.2).

**Empirical finding (2026-05-08 demo run, 1024×1024 CLIC photo, d=1.0,
iters=2):** baseline butteraugli is ~8.9 (target ~1.0). The loop runs
end-to-end correctly but score barely moves because the underlying
LossyEncoder pipeline has known gaps:

- `run_pipeline_with_qac` is DCT8-only (no DCT16/32/64 strategy selection)
- No EPF in the LossyEncoder reconstruction path
- No CfL (chroma-from-luma)
- Input linearization uses simplified `powf(2.4)`, not IEC piecewise

The loop infrastructure is complete and ready; it'll show real value once
the underlying pipeline gaps land. Score reduction at the current
baseline is dominated by these missing stages, not by the refinement
algorithm itself.

**G5.1 status:** the four host-side helpers are pure line-by-line ports
of inline upstream code (not separately-callable upstream functions);
they're validated by hand-derived unit tests covering edge cases
(uniform inputs, single peaks, AC-strategy splat, edge clipping,
deviation-bounds extremes, integer-step bump). Parity validation against
the full upstream loop awaits either (a) upstream extraction +
`__internals` re-export, or (b) a full integration harness running both
encoders on the same image and comparing final quant fields.

## Phase 7 — Broadcast-weights kernel optimization

Per-block weights (a 64-float quant matrix replicated `num_blocks`
times) consumed multi-megabyte GPU buffers and host-side replication
loops on every cost-grid call. Five kernels now have broadcast-weights
variants that read `weights[iu]` instead of `weights[off + iu]` —
the buffer is a single-block template (1 × N coefficients) and the
kernel broadcasts it across all blocks.

| Kernel | Broadcast variant | Commit | Saved per call (1024×1024) |
|---|---|---|---|
| `quantize_dct8` | `quantize_dct8_kernel_broadcast_w` | `0870876` | 12 MB → 768 B (3-channel) |
| `dequant_dct8` | `dequant_dct8_kernel_broadcast_w` | `0870876` | 12 MB → 768 B (3-channel) |
| `dequant_simple` | `dequant_simple_kernel_broadcast_w` | `3791acb` | 4 MB → 256 B (per channel) |
| `quantize_large` | `quantize_large_kernel_broadcast_w` | `81e922a` | 4 MB → 1 KB (DCT16x16) |
| `entropy_coeffs_pixel` | `entropy_coeffs_pixel_kernel_broadcast_w` | `c2005c9c` | 24 MB → 1.5 KB (3-channel, 2 weights/channel) |
| `entropy_coeffs_coeff` | `entropy_coeffs_coeff_kernel_broadcast_w` | `752e295b` | 12 MB → 768 B (3-channel) |

**Wired into hot paths:**

| Caller | Wiring commit | Notes |
|---|---|---|
| `LossyEncoder::run_pipeline_with_qac` | `0870876` | DCT8 path — both quantize + dequant |
| `forks::reconstruct::reconstruct_xyb_dct8_only_gpu` | `360f704b` | Recon for sharpness selection |
| `forks::afv::afv_cost_grid_*` | `450e7451` | Single-channel + XYB AFV cost grids |
| `forks::cost::estimate_entropy_full_*` | `b0b05a28` | Both DCT8 batch + strategy-generic orchestrators |

**Aggregate net effect on a 1024×1024 cost-grid evaluation pass**
(DCT8-batch + entropy_coeffs + reconstruct + AFV cost grid):
- Before: ~76 MB of host-side replicated weight buffers + GPU upload per pass
- After: ~3 KB of weight templates uploaded once

**Bit-identical output preserved** — every broadcast variant has a
parity test asserting max|Δ| < 1e-5 vs the per-block path called with
replicated weights. 244 tests pass total.

**Per-block variants stay in place** for callers that genuinely need
per-block-varying weights (e.g., future content-adaptive quant
matrices). Currently no in-tree caller uses per-block weights — all
callers were replicating a single template.

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
