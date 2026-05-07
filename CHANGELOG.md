# Changelog

## [Unreleased]

### EPF sharpness picker — selection logic ported (`4f49174e`, `a270f002`)

Two host-side additions in `forks::epf`:

1. `apply_epf_step0_gpu` — symmetric wrapper around the new
   `GpuEncoder::epf_step0_channels` method, matching the shape of
   `apply_epf_step1_gpu` / `apply_epf_step2_gpu`. The pad ≥ 3
   contract is documented inline.

2. `select_sharpness_two_pass` — pure-CPU port of the per-block
   sharpness picker from upstream `compute_epf_sharpness`. Pass 1
   is greedy with `K_FAVOR_NO_SMOOTHING = 0.99` and a neighbor-
   preference fallback; Pass 2 is the context-reweighted re-scan
   with libjxl's exact `size_t / size_t` integer division (which
   makes the entropy term a no-op for most contexts; only the c3
   bias on sharpness=0 has real effect).

Once a `forks::reconstruct::reconstruct_xyb_gpu` lands, these two
plus the existing `apply_epf_step{0,1,2}_gpu` + `block_l2_errors`
will form the full `compute_epf_sharpness_gpu` orchestrator.

### EPF Step 0 — all three EPF passes now on GPU (`147e6ffb`, `094ea81f`, `3ec161d1`)

The heaviest EPF pass (5×5 plus pattern, 12 neighbors × 5-position
SAD) is now ported and parity-verified at FP32 floor:

1. `kernels::epf::epf_step0_kernel` — one thread per output pixel,
   reuses the existing `sad_3x3_plus` helper, mirrors the structure
   of `epf_step1_kernel` expanded to 12 neighbors.
2. `launch::epf::epf_step0` + `GpuEncoder::epf_step0_channels` —
   slice-in / slice-out wrapper following the step 1 / step 2
   pattern.
3. `examples/epf_step0_parity.rs` — line-by-line inline CPU port of
   upstream `jxl_encoder::vardct::epf::epf_step0_strip` (which is
   private inside the upstream crate; not exposed via
   jxl_encoder_simd). On CUDA: max|Δ| 4.7e-10 / 4.5e-8 / 4.5e-8 vs
   5e-5 tolerance.

Closes the largest "CPU bits inside existing forks" gap from
PORT_STATUS. All three EPF passes are now GPU kernels with parity
tests at FP32 floor.

### AFV0-3 cost grid integration — last per-strategy gap closed (`1f25c7c3`, `f6c3e45e`, `2ef97a2e`)

The AFV0-3 corner-DCT family is now wired through the cost-grid path
the AC strategy search consumes. Three pieces:

1. `quant_weights::afv_weights() -> Vec<f32>` (192 floats: 64 per
   channel; same table for AFV0-AFV3). Bit-for-bit port of upstream
   `jxl_encoder::vardct::quant::generate_afv_weights`.
2. `forks::afv::afv_cost_grid_single_channel` — host-side, all 4
   afv_kinds in one call. Per kind: forward AFV batch → quantize
   (DCT8-shape with afv_weights) → dequant (generic) → inverse AFV
   batch → host L2 reduction. Output `Vec<f32>` of length
   `4 * n_blocks`, indexed by `[kind * n_blocks + b]`.
3. `forks::afv::afv_cost_grid_xyb_host` — 3-channel mask1x1-modulated
   counterpart, matching the shape of the GPU-resident
   `compute_cost_grid_*_xyb` family in pipeline.rs. Stays host-side
   because the AFV transform composition itself is host-orchestrated
   (3 GPU launches per direction with per-block DC pack/unpack and
   corner mirroring on the host).

This closes the last per-strategy cost grid gap. The remaining
items in PORT_STATUS are the three CPU-bits-inside-existing-forks:
EPF Step 0, full `estimate_entropy_full` orchestration, and
AdjustQuantBlockAC heuristics.

### Quantize + dequant cover the full strategy family (`768e5763`, `c79c3c5d`)

`forks::quantize::quantize_blocks_gpu` dispatches to either the
DCT8 fast path or the generic `quantize_large` kernel based on the
`(grid_w, grid_h, llf_x, llf_y)` strategy descriptor. Mirrors the
shape of `quantize_large_scalar` upstream and works for every block
size DCT8/16/32/64 family produces.

`forks::dequant::dequant_blocks_gpu` is the symmetric generic
dequant: `output[i] = quant[i] * weights[i]` for arbitrary
`block_size`. Promoted from a previously-internal pipeline.rs
helper to a public module (`kernels::dequant_simple` +
`launch::dequant_simple`).

The "non-DCT8 strategies for quantize/dequant" item is removed from
the "CPU path stays" list — production-encoder paths can now
quantize and dequant any AC strategy on GPU.

### AFV batched APIs (`248dff39`, `15b309eb`)

`forks::afv::{afv_transform_batch_gpu, inverse_afv_transform_batch_gpu}`
process N same-kind 8×8 blocks in **3 GPU launches per direction**
(one per sub-transform) instead of 3 launches per block. Per-block
extraction/composition/DC packing happen on the host. Roundtrip
test covers all 4 afv_kinds × 8 synthetic blocks at FP32 floor.

### Phase 5 — AFV0-3 inverse transform — full standard JXL AC strategy family on GPU (`de952f60`, `1d098cbe`)

Closes the AFV port arc. Two new GPU kernels:
- **Raw 4×4 inverse DCT** (`idct_4x4_raw_kernel`): 16-coeff primitive,
  bit-near-perfect 5.96e-8 vs upstream `idct_4x4`.
- **Raw 4×8 inverse DCT** (`idct_4x8_raw_kernel`): 32-coeff primitive
  consuming the transposed forward output, 5.96e-8.

Plus host-side composition `forks::afv::inverse_afv_transform_gpu`.

Forward + inverse AFV roundtrip on the same synthetic input for all
4 corner variants (`afv_transform_parity`):

| afv_kind | roundtrip max\|Δ\| |
|---|---|
| 0 | 8.94e-8 ✓ |
| 1 | 5.96e-8 ✓ |
| 2 | 1.19e-7 ✓ |
| 3 | 1.19e-7 ✓ |

All FP32 noise floor. **The standard JXL AC strategy family is now
fully ported on GPU**: every forward + inverse transform in the spec
(DCT4/8/16/32/64 family squares + rects, IDENTITY, DCT2X2, AFV0-3)
is available as a GPU kernel and exposed through the encoder facade.

### Phase 5 — AFV0-3 forward transform on GPU (`faf658a6`, `e8ebb52f`, `0d9514e5`, `4a5fca9b`, `de77c788`, `be8b6206`)

Completes the AFV (Adaptive Frequency Variable) corner DCT family,
the last remaining unported transform in the standard JXL AC strategy
set. AFV is used for 8×8 blocks at the corners of larger transform
regions and provides better frequency localization than DCT8.

Three new GPU kernels:
- **AFV 4×4 DCT** (`afv_dct_4x4_kernel` + inverse): the unique
  16×16 matmul against the libjxl `AFV4X4_BASIS_TRANSPOSE` matrix.
  Forward + inverse both bit-exact (8.85e-9 / 5.96e-8).
- **Raw 4×4 forward DCT** (`dct_4x4_raw_kernel`): the 16-coeff
  primitive (NOT the 64-coeff dct_4x4_full). 8.85e-9 parity.
- **Raw 4×8 forward DCT** (`dct_4x8_raw_kernel`): 32-coeff primitive,
  transposed output. 2.24e-8 parity after fixing dct1d_8 output
  ordering bug (was misreading upstream `dct1d_8_val` array layout —
  positions 2/3/4/5 were swapped).

Plus host-side composition (`forks::afv`):
- `extract_afv_corner` (with mirror per `afv_kind`)
- `extract_dct4_corner`, `extract_dct4x8_half`
- `pack_afv_dcs` (DC repacking)
- `afv_transform_gpu(enc, basis_t, pixels, afv_kind) -> [f32; 64]`

End-to-end parity (`afv_transform_parity` example) for all 4
corner variants:

| afv_kind | max\|Δ\| |
|---|---|
| 0 | 3.07e-8 ✓ |
| 1 | 7.45e-8 ✓ |
| 2 | 3.70e-8 ✓ |
| 3 | 3.91e-8 ✓ |

3 GPU launches per block (un-batched). Future optimization: batch
by `afv_kind` for higher throughput. **Inverse AFV transform port
is still TODO** (decoder side; needs idct_4x4_raw + idct_4x8_raw +
the inverse AFV-DCT kernel which already exists).

### Phase 5 — fuzzy_erosion ported, forks::adaptive_quant fully on GPU (`feb0ec97`, `67c00680`, `53efc207`, `bf89275c`)

GPU-port of `jxl_encoder::vardct::adaptive_quant::fuzzy_erosion` —
3×3 min-of-4 weighted sum + 2× downsample. The "find smallest 4 of
9" partial sort runs per-thread (per output pixel); each thread
gathers 4 input contributions directly, avoiding the unsynchronized
`+=` write race the CPU sequential loop sidesteps.

Parity: **2.98e-8 abs** vs inline CPU reference (FP32 noise floor)
on a 73×51 source plane with non-aligned region offset.

API:
- `pub fn launch::fuzzy_erosion::fuzzy_erosion_kmul(butteraugli_target) -> [f32; 4]`
- `pub fn GpuEncoder::fuzzy_erosion_plane(src, src_w, src_h,
  from_x0, from_y0, region_w, region_h, butteraugli_target) -> (Vec<f32>, u32, u32)`
- `pub fn forks::adaptive_quant::fuzzy_erosion_gpu(enc, ...)` — usize
  signature matching upstream

With this, the **full `forks::adaptive_quant` chain runs end-to-end
on GPU**: `mask1x1 → pre_erosion → fuzzy_erosion →
per_block_modulations`. The "fuzzy_erosion stays CPU" caveat is
removed from the fork module docs and PORT_STATUS.

### Quant weights — full 15-strategy coverage in `crate::quant_weights`

`pub mod quant_weights` now exposes real libjxl quant weight
tables for all 15 GPU-supported AC strategies, not just DCT8:

```text
DCT8                  dct8_weights()           192 floats (3 × 64)
DCT16x16              dct16x16_weights()       768 floats (3 × 256)
DCT16x8 / DCT8x16     dct16x8_weights()        384 floats (3 × 128)
DCT32x32              dct32x32_weights()       3072 floats (3 × 1024)
DCT16x32 / DCT32x16   dct16x32_weights()       1536 floats (3 × 512)
DCT64x64              dct64x64_weights()       12288 floats (3 × 4096)
DCT32x64 / DCT64x32   dct32x64_weights()       6144 floats (3 × 2048)
DCT4X4                dct4x4_weights()         192 floats
DCT4X8 / DCT8X4       dct4x8_weights()         192 floats
IDENTITY              identity_weights()       192 floats
DCT2X2                dct2x2_weights()         192 floats
```

Generic `pub fn generate_quant_weights_rect(rows, cols, band_params,
num_bands)` mirrors `jxl_encoder::vardct::quant::generate_dct_quant_weights_rect`
bit-for-bit. All five Phase 3 partition demos retrofitted to use
real weights — no mock matrices remain.

### forks::transform — IDENTITY + DCT2X2 dispatch entries (`806c3871`, `a03b3a28`)

`apply_dct_batch_gpu` and `apply_idct_batch_gpu` now dispatch
IDENTITY (`RAW_STRATEGY_IDENTITY=15`) and DCT2X2
(`RAW_STRATEGY_DCT2X2=16`) alongside the existing 13 DCT family
strategies. Two new dispatcher roundtrip tests verify the path.
Local dispatcher codes 15/16 (vs upstream wire codes 8/9) —
documented inline.

### Validated — Sub-block selector across CLIC2025 corpus (`6b7411cb`)

`corpus_subblock_picks_demo` runs the full 7-strategy 16×16 partition
selector across 8 CLIC2025-1024 images (32,768 regions, 45,272
sub-block cells). The single-image finding from
`phase3_subblock_real_image_demo` generalizes:

| Strategy | Aggregate pick % |
|---|---|
| DCT16×16 | 25.5% |
| 2×DCT16×8 horiz | 13.7% |
| 2×DCT8×16 vert | 26.2% |
| **Four DCT8×8** | **0.1%** (23 of 32,768) |
| **Four SubBlocks** | **34.5%** |

Pure 4-DCT8 picks 0.0–0.1% on **every image** in the corpus. Within
the 45,272 sub-block cells, the selector chooses not-DCT8 in 74% of
cells (DCT4 family + IDENTITY + DCT2X2 win the rest). Stable across
content — per-cell sub-block ranges are 29-43% and per-cell
DCT8-share ranges 24-28% across the corpus.

Robust, content-agnostic finding that per-cell strategy choice is
substantively useful for natural-image VarDCT encoding.

### Phase 3 — per-cell sub-block strategy choice in select_partitions_16x16

The 16×16 partition selector now supports per-cell sub-block
strategy choice via a new `Partition16x16::FourSubBlocks([SubStrategy; 4])`
variant. Each of the 4 8×8 cells independently picks the lowest-
cost strategy from `{DCT8, DCT4×4, DCT4×8, DCT8×4, IDENTITY, DCT2X2}`,
based on optional per-cell cost grids passed via the new
`SubBlockCostGrids` field on `CostGrids16x16`.

API additions (`6c61103d`, `6345f558`):
- `pub enum SubStrategy { Dct8, Dct4x4, Dct4x8, Dct8x4, Identity, Dct2x2 }`
- `pub struct SubBlockCostGrids<'a>` — optional per-cell grids
- `pub fn pick_subblock_strategies(...) -> (f32, [SubStrategy; 4])`
- `Partition16x16::FourSubBlocks([SubStrategy; 4])` (new variant)
- `CostGrids16x16::sub_blocks` (new field, defaults to all-None)

Backwards compatible — existing callers that don't set `sub_blocks`
continue picking from the original 4-strategy set.

`phase3_subblock_real_image_demo` validation on 1024² CLIC photo
with real DCT8 quant weights (`dcd1af2c`):

| Strategy           | 16×16-tier picks |
|---|---|
| DCT16×16           | 28.2% |
| Two DCT16×8 horiz  | 10.1% |
| Two DCT8×16 vert   | 28.2% |
| **Four DCT8×8**    | **0.1%** (3 regions of 4096) |
| **Four SubBlocks** | **33.4%** (1367 regions) |

Within the 5468 sub-block cells:
- DCT8: 27.9%
- DCT4×4: 14.2%, DCT4×8: 20.2%, DCT8×4: 21.0%
- IDENTITY: 6.0%, DCT2X2: 10.8%

Pure 4-DCT8 nearly vanishes when sub-block alternatives are
offered — clear signal that per-cell strategy choice is
substantively useful for natural-image encoding.

### Real-weights helper module (`939b400b`)

New `pub mod quant_weights` exposes the libjxl default DCT8 quant
weights (previously private to `LossyEncoder`):

- `pub fn dct8_weights() -> [f32; 192]`
- `pub fn dct8_weights_per_channel() -> ([f32; 64], [f32; 64], [f32; 64])`
- `pub fn replicate_weights(per_block, n_blocks) -> Vec<f32>`
- `pub const DCT8_PARAMS: [[f64; 6]; 3]` — band parameter table

Bit-for-bit match with `jxl_encoder::vardct::quant::quant_weights(0, c)`.
Cost grid demos using these weights produce content-driven strategy
distributions; with mock unit weights all strategies produce
degenerate identical costs.

### Phase 3 — IDENTITY + DCT2X2 ported, 15-strategy cost grid coverage

Two new GPU AC-strategy kernels with **bit-exact parity** vs the
upstream scalar reference (`8550f846` IDENTITY, `fc8b8603` DCT2X2):

- IDENTITY 8×8 forward + inverse: per-sub-block DC + residual layout
  (4 4×4 sub-blocks) + 2×2 Hadamard merge of the 4 DCs. Pure scalar
  arithmetic, no butterflies. Forward + inverse both bit-exact;
  roundtrip 1.04e-7.
- DCT2X2 8×8 forward + inverse: hierarchical 2×2 Hadamard at scales
  S=8/4/2 (forward) and S=2/4/8 (inverse). Forward + inverse both
  bit-exact; roundtrip 1.19e-7.

Exposed on `GpuEncoder` (`identity_blocks`, `inverse_identity_blocks`,
`dct2x2_blocks`, `inverse_dct2x2_blocks`) and wired into Phase 3
cost grids in both single-channel and 3-channel flavors
(`compute_cost_grid_{identity,dct2x2}_{single_channel,xyb}`)
(`36342558`, `4734522b`).

**Total Phase 3 cost-grid coverage now 15 strategies × 2 flavors =
30 functions**: full DCT4/8/16/32/64 family + IDENTITY + DCT2X2.
All standard JXL AC strategies except AFV0-3 are wired.

### Phase 3 — full 13-strategy 3-channel cost grids + rect-aware selector validated

The Phase 3 strategy-selection stack now exercises the full
DCT4/8/16/32/64 family with proper 3-channel XYB-weighted +
mask1x1-modulated cost grids — the same cost the VarDCT encoder
pays for in production.

**Cost grids added** (sessions 5–6):

- 13 single-channel grids (`compute_cost_grid_*_single_channel`):
  DCT4x4, DCT4x8, DCT8x4, DCT8, DCT16x8, DCT8x16, DCT16x16,
  DCT32x16, DCT16x32, DCT32x32, DCT64x32, DCT32x64, DCT64x64.
- 13 3-channel grids (`compute_cost_grid_*_xyb`): same coverage,
  with XYB weighting + per-pixel mask1x1 perceptual modulation.

**Validation** (`corpus_rect_picks_demo`, 8 CLIC2025-1024 images,
32,768 16×16 regions, full 4-strategy 16×16 selector with proper
3-channel cost grids):

| Strategy           | Aggregate pick % |
|---|---|
| DCT16×16           | 21.2% |
| Two DCT16×8 horiz  | 33.2% |
| Two DCT8×16 vert   | 26.9% |
| Four DCT8×8        | 18.8% |
| **rect total**     | **60.1%** |

Per-image rect% range: 56.7%–65.6% (σ ≈ 2.5pp). Rect strategies
win the majority of picks on every image in the corpus —
content-driven selection across the full strategy family is
robust and generalizes.

**Key commits:** `48a1a207` (3-channel DCT8), `4ddd27b9` (DCT16),
`e93fcc46` (DCT32), `13d94fb3` (DCT64), `ee44b868` (rect family),
`7d27b18d` (DCT4 family), `5552e611` (real-image rect demo),
`3562ac26` (corpus rect picks).

### Phase 1 — denoise port complete (`b0131b8b`, `dc853da3`)

Ported the missing Wiener 5×5 denoise filter
(`jxl_encoder_simd::noise::denoise_channel_scalar`) to GPU with
**5.96e-8 abs parity** (FMA-contraction noise floor). Wired into the
fork pipeline as `forks::noise::denoise_xyb_gpu(enc, x, y, b, w, h,
lut, quality_coef)` — three sequential GPU launches replacing the
upstream `rayon::join`. Phase 1 is now **7/7 ✓** (full coverage).

### Validated — Content-driven AQ wins across CLIC2025 corpus (`a1617849`)

Added fast-ssim2 + imgref dev-deps and SSIMULACRA2 measurement to
`quality_sweep_with_aq_demo` and `corpus_aq_sweep_demo`. AQ wins on
8/8 CLIC2025-1024 images at every distance from d=1.0 onwards,
with **zero perceptual losses** across the corpus:

| dist | uniform µ | AQ µ | Δssim2 µ | wins | losses |
|---|---|---|---|---|---|
| 0.5 | 73.45 | 73.91 | +0.46 | 7 | 0 |
| 1.0 | 71.51 | 72.45 | +0.94 | 8 | 0 |
| 2.0 | 67.03 | 69.23 | +2.19 | 8 | 0 |
| 4.0 | 58.21 | 62.46 | +4.25 | 8 | 0 |
| 8.0 | 43.80 | 50.33 | +6.52 | 8 | 0 |

(8-image subset; win threshold |Δssim2| > 0.05.) The benefit grows
with distance — exactly as expected, since heavy-quant headroom for
bit redistribution is what AQ trades on. Generalizes across content,
not image-specific.

`corpus_aq_sweep_demo`: corpus-level harness (CORPUS_DIR + MAX_IMAGES
env vars) reporting per-distance Δssim2 mean, win/loss counts, and
worst-case regression.

### Added — Turnkey content-driven AQ + batch sweep (`3cfb4ad9`, `e0634ae4`, `0ebbb22d`)

Top-of-stack one-call interface for content-driven adaptive
quantization, plus the batch-amortized variant for distance sweeps:

```rust
// Turnkey: single image, single distance.
let (r, g, b) = lossy.encode_one_with_aq(&enc, &r, &g, &b, distance);

// Batch: single mask prepass, N distance points.
let outs = lossy.encode_many_with_aq(&enc, &r, &g, &b, &distances);
```

Internally runs `XYB → mask1x1 → per-block reduce → derived qac
field → adaptive encode`. The mask depends only on the image —
not distance — so the batch variant computes it once and reuses
across the whole sweep.

Composable building blocks for callers who want a custom prepass:

- `LossyEncoder::compute_block_mask_means(...)` — GPU prepass,
  one f32 per padded 8×8 block.
- `block_means_to_qac_field(means, distance)` — pure-CPU mapping,
  centers on `distance` and varies in a 4× range.
- `LossyEncoder::compute_aq_field(...)` — composition of the two,
  for callers who want to inspect/modify the field before passing
  to `encode_one_adaptive`.

`content_driven_aq_demo` rewritten to use the turnkey API directly
(was previously inlining the chain).

### Added — Adaptive quantization API + content-driven AQ demo (`02d80bef`, `c4225cee`, `4400081a`)

New `LossyEncoder::encode_one_adaptive` accepts a per-block `aq_field:
&[f32]` (one qac scalar per padded block) instead of broadcasting a
single value. Enables real adaptive quantization where smooth
regions get heavier quant and detail regions get lighter quant.

Two demos:
- `adaptive_quant_demo` — synthetic split (left half qac=1.530,
  right half qac=0.096); confirms per-half MAE differs by 1.9-3×.
- `content_driven_aq_demo` — chains XYB → `compute_mask1x1_gpu`
  (forks::adaptive_quant) → per-block reduce → derived qac field →
  encode_one_adaptive. Beats uniform encode at midpoint qac by
  **15-35% lower MAE per channel** on a 1024×1024 CLIC photo.

This demonstrates the natural composition of the three layers:
`crate::forks` (mask1x1) + `crate::persistent` (chained launches) +
`crate::lossy_encoder::LossyEncoder::encode_one_adaptive` produce
end-to-end content-driven AQ.

### Added — Per-channel dead-zone thresholds + `upload_i32_blocks` (`3fdadf2d`, `3c654bfd`)

Two cleanup additions matching libjxl semantics:

- `LossyEncoder` now uses per-channel `thresholds_x/y/b` arrays
  matching `jxl_encoder::vardct::quantize::default_thresholds`:
  Y has tightest TL (0.56), X/B share 0.58. Combined with the
  per-channel quant weights, LossyEncoder is now bit-equivalent
  to libjxl's DCT8 lossy path at single-block coverage.
- `GpuEncoder::upload_i32_blocks` — public API for uploading
  per-block i32 data (matches `upload_blocks` for f32). Removes
  the test-only `client_ref_for_test` accessor that was a hack
  workaround when this method didn't exist.

### Added — `distance_to_qac` + `K_AC_QUANT` public (`c77a41a7`)

Direct libjxl-style distance interface:
  - `pub const K_AC_QUANT: f32 = 0.765` (libjxl AC scale at d=1)
  - `pub fn distance_to_qac(distance) -> f32` = `K_AC_QUANT/distance`

`quality_to_qac` now delegates to `distance_to_qac(50/q)`.
README quick-start + lossy_encoder demo updated to use the new
distance interface.

### Fixed — `quality_to_qac` direction was inverted vs libjxl (`1fa03c76`)

The kernel uses `val = coef * inv_weight * qac`, so SMALLER qac
means MORE aggressive quant (val falls below dead-zone). My
`quality_to_qac` had higher quality → smaller qac, which produced
the opposite of what the docstring claimed (q=100 gave heavy
quant, q=10 gave light). Caught when `quality_sweep_demo` showed
non-monotonic MAE.

Fix: `distance = 50/q; qac = K_AC_QUANT/distance` (libjxl
convention, K_AC_QUANT=0.765). Now monotonic — q=100 → qac=1.530,
q=50 → qac=0.765, q=10 → qac=0.153.

### Improved — Real DCT8 quant matrices in LossyEncoder (`0a85b6f6`, `7eaac931`)

Replaced placeholder unit weights (all-1.0s) with libjxl's per-
channel DCT8 quant matrices derived from `DCT8_PARAMS` via the
parametric band formula. Constants + math duplicated bit-for-bit
from `jxl_encoder::vardct::quant` (the upstream module is
crate-private).

Per-channel: X / Y / B each get their own `weights_g` handle.
B-channel error at high q is now visibly larger than Y (B has
narrowest libjxl quant matrix), matching real JPEG XL behavior.

Measured impact on quality_sweep_demo (1024×1024 photo) after
both this fix and the inverted-direction fix:

  qual    qac    R MAE   G MAE   B MAE
   95   1.453   5.12    4.66    8.94
   80   1.224   5.20    4.74    9.62
   60   0.918   5.36    4.92   10.98
   40   0.612   5.70    5.26   13.31
   20   0.306   6.60    6.11   18.31

Monotonic + sensible scale. (Previous unit-weights output had
MAE values in the 18+ range across all qualities.)

### Added — `LossyEncoder` high-level API (`b4eb3619`, `c5c312fb`)

Productionizes the persistent pipeline as a single user-facing API:

```rust
let lossy = LossyEncoder::new(&enc, width, height);

// One-shot encode (input uploaded once, output downloaded).
let (r_out, g_out, b_out) = lossy.encode_one(&enc, &r, &g, &b, qac);

// Batch encode at multiple settings on the same input.
// Input uploaded ONCE; ~5× faster than encode_one in a loop.
let outputs = lossy.encode_many(&enc, &r, &g, &b, &qac_settings);
```

**Arbitrary image sizes are supported.** Non-multiples-of-8 are
padded internally with right+bottom edge replication (matching
libjxl), pipeline runs on padded dims, output cropped back to
caller dims. `LossyEncoder::dimensions()` and `padded_dimensions()`
expose both.

`lossy_encoder_demo` and `lossy_encoder_real_image` examples
demonstrate one-shot + batch on synthetic and real CLIC2025 photo
data (cropped to 1017×1013 to exercise the non-aligned path).

### Performance — GPU pipeline now FASTER than CPU AVX2 (`d6ef26c7`)

The breakthrough: replacing
  `client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))`
with
  `client.empty(n * 4)`
in the persistent API gave a **7.8× pipeline speedup**. Old pattern
allocated host vec + did real PCIe upload of zeros to a buffer the
kernel was about to overwrite. empty() reserves GPU memory without
touching the bus.

End-to-end CPU vs GPU lossy DCT8 (RTX 5070 + Ryzen 9 7950X):

```
side    CPU ms    GPU ms    ratio        throughput
 256     1.61     1.49     1.08× GPU   44 MP/s
 512     9.39     4.28     2.19× GPU   61 MP/s
1024    39.46    17.66     2.23× GPU   59 MP/s
2048   150.55    41.33     3.64× GPU   101 MP/s vs 28 MP/s
```

Parity preserved at 4e-6 max abs delta. Win grows with size.

Also: per-kernel fusion wins documented in fused_dct_quant_bench
(2.84× DCT+quant fused) and fused_dequant_idct_bench (3.07× inverse).

### Added — Fused DCT+quant kernels (`b1bc500c`, `53bbae07`)

Two new kernels that combine consecutive stages into one launch:
- dct8_quantize_fused_wide_kernel (forward: DCT → quant in one pass,
  intermediate coeffs stay in shared memory)
- dequant_idct8_fused_y_wide_kernel (inverse: dequant → IDCT, Y
  channel only, no CfL)

Both bit-exact vs split chains. 2-3× speedup at sweet spot (1024²)
for the per-kernel measurements; pipeline-level use is gated on
having a real DC handling path (real encoders use dc_coding).

### Added — Wide-cube DCT8 (`a9ecc28b`, `32377a60`)

cube_dim=64 variant of DCT8 (one block per thread, per-thread
private slice in shared memory). 2.9× faster than naive cube_dim=1
at 1024² sweet spot. Tuned via sweep: cube_dim ∈ {16, 32, 64} →
64 wins by small margin at 1024² and 4096².

### Added — Persistent GPU buffer API + full-GPU lossy roundtrip

The biggest single architectural addition since the kernel library
landed. Closes the round-trip-API perf gap by exposing typed handles
that keep data on-GPU across many pipeline stages.

- `crate::persistent` module with `GpuPlane<R>`, `GpuBlocks<R>`,
  `GpuI32Blocks<R>` typed handles. (`a146601f`, `2a596b40`, `a1f99672`)
- ~25 persistent-API methods on `GpuEncoder<R>` covering full
  encoder front+back: XYB fwd/inv, gaborish, gab smooth, mask1x1,
  pad_plane, all 13 DCT/IDCT strategies, DCT8 quantize+dequant,
  spatial↔per-block gather/scatter, DC restore.
- 3 new GPU kernels closing mid-pipeline host hops:
  - `gather_blocks_kernel` / `scatter_blocks_kernel` (`7d4fef19`)
  - `restore_dc_kernel` (`75298a62`)
- Cooperative DCT8 (`dct_8x8_coop_kernel`, cube_dim=8). Sub-ulp
  parity with naive; no significant speedup. (`1c882573`)

### Added — fork-and-modify of jxl-encoder pipeline (`forks::*`)

Per user authorization to fork-and-modify upstream source. 11 fork
modules + `forks_pipeline_demo` + `lossy_roundtrip_demo`:
xyb, gaborish, adaptive_quant, reconstruct, transform (13
DCT strategies), cfl (LS+Newton, batched), epf (Step 1+2),
dequant, quantize, cost (entropy/block_l2/pixel_loss), pad.

### Added — examples + benchmarks

- `forks_pipeline_demo` (`f4bf2be2`), `lossy_roundtrip_demo` (`a5640970`)
- `lossy_roundtrip_persistent` (`6ef942f7`, `75298a62`)
- `real_image_encode` (1024×1024 CLIC, djxl-verified) (`89a8bf4c`)
- `jxl_rs_roundtrip` (pure-Rust roundtrip via jxl-rs) (`a6bf966a`)
- `xyb_throughput_bench` / `xyb_scaling_bench` (`66f1a6ea`, `0da14544`)
- `persistent_buffer_pipeline` (1.7-3.4× faster than round-trip API) (`532ba90c`)
- `lossy_pipeline_throughput` (full pipeline parity 4e-6) (`7af4c3f9`)
- `dct8_coop_bench` (cooperative vs naive vs CPU) (`1c882573`)

### Added
- Initial repo scaffold: workspace, `jxl-encoder-gpu` crate skeleton with
  cubecl 0.10.0-pre.4 dependency, feature flags for cuda/wgpu/hip/cpu backends.
- `CLAUDE.md`, `README.md`, `PORT_STATUS.md` documenting phase plan and
  per-kernel parity grid (37 deliverables tracked, 5 complete).
- Phase 1 kernels with parity verified on RTX 5070 + CUDA 13.2:
  - `kernels::xyb` (forward + inverse) — 1.19e-7 / 1.30e-6 abs
  - `kernels::gab` (3x3 gab smooth) — 5.96e-8 abs
  - `kernels::gaborish` (5x5 gaborish inverse) — 1.19e-7 abs
  - `kernels::mask1x1` (fast_log2f + reciprocal) — 6.79e-4 abs
    (within FMA-contraction noise vs CPU `_scalar`)
- Examples: `examples/xyb_parity.rs`, `examples/phase1_parity.rs`
- Phase 2 kernels with parity verified on RTX 5070 + CUDA 13.2:
  - `kernels::dct8::dct_8x8_kernel` — 2.98e-8 abs vs scalar reference
  - `kernels::dct8::idct_8x8_kernel` — 1.79e-7 abs; roundtrip 2.24e-7
  - One-thread-per-block strategy (cube_dim=1, cube_count=num_blocks).
    Cooperative (intra-block parallelism) variant deferred — naive port
    is the parity baseline.
- Example: `examples/dct8_parity.rs`
- Phase 2 small kernels:
  - `kernels::block_l2::block_l2_kernel` — 1.40e-9 abs vs scalar
  - `kernels::quantize::quantize_dct8_kernel` — bit-exact (0 diffs, 64 blocks)
  - User-implemented `round_ties_even_to_i32` in cubecl since 0.10 has
    no ties-to-even cast. Required for AdjustQuantBlockAC parity.
- Example: `examples/phase2_small_parity.rs`
- `kernels::pixel_loss::pixel_loss_kernel` — per-block 8th-power norm
  with f64 accumulation. Confirms cubecl 0.10 supports f64 ops on the
  CUDA backend. 3.75e-16 relative parity vs scalar reference.
- `kernels::dequant::dequant_dct8_kernel` — per-block AC dequant + CfL
  restore. Y bit-exact; X/B 1.53e-5 abs (FMA contraction on the CfL add).
- Examples: `examples/pixel_loss_parity.rs`, `examples/dequant_parity.rs`
- `kernels::dct16` — 16x16 forward + inverse DCT (1-cube-per-block).
  Forward 5.96e-8 abs, inverse 5.36e-7, roundtrip 4.47e-7 (64 blocks).
  Recursive butterfly: dct1d_16 → dct1d_8 → dct1d_4 → dct1d_2 (forward);
  inv_idct1d_16 → inv_idct1d_8_core → inv_idct1d_4 (inverse).
- Example: `examples/dct16_parity.rs`
- DCT16 rectangular family (forward + inverse for 16x8 and 8x16). All
  four kernels parity-verified vs jxl-encoder-simd scalars:
  - `dct_16x8_kernel` 3.73e-8, `dct_8x16_kernel` 4.01e-8
  - `idct_16x8_kernel` 3.58e-7, `idct_8x16_kernel` 3.13e-7
  - DCT8x16 roundtrip 3.87e-7. DCT16x8 roundtrip skipped — CPU API
    has asymmetric forward/inverse storage layouts (CPU also fails,
    1.28 abs); the encoder pipeline transposes between them.
- Example: `examples/dct16_rect_parity.rs`
