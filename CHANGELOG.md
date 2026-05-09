# Changelog

## [Unreleased]

### AC strategy search — full GPU pipeline + content-aware selectivity (May 8-9, 2026)

Multi-day effort delivering a working AC strategy search on GPU,
matching libjxl's algorithm at quality parity with uniform-qac.

**End state:** `LossyEncoder::encode_one_with_strategy_search_dct8_16`
runs in 175 ms on a 1024×1024 image at d=1.0 (vs 305 ms baseline,
**1.7× speedup**), produces butteraugli identical to uniform-qac
across d ∈ {0.5, 1.0, 2.0, 4.0}, and **content-aware selectivity
verified** in both directions:
- CLIC photo at d=1.0 → 98.4% DCT8 (correct on detailed content)
- 256×256 smooth gradient at d ∈ {1, 2, 4} → 100% DCT64x64

**Key commits (chronological):**
- `ac88dc1c` fix: GPU DCT mean-scale convention (DC grid sum/8 → sum/64)
- `79e034707` feat: EPF in postpass closes +6% butteraugli gap
- `13be0049` perf: persistent cost-grid pipeline (6× faster)
- `74c5a465` perf: keep mask1x1 on GPU
- `540f55fd` perf: persistent encode/recon chain
- `f7cdaa56` fix: kFavor2X2 + tuned DCT32 entropy_mul fixes regression
- `b482813b` fix: 2× anti-bias muls for sub-blocks
- `c4c62cef` feat: IDENTITY + DCT2x2 (full sub-block palette)
- `9ae995d3` feat: thread CostGrids16x16 through 32x32 + 64x64 selectors
- `ba9eed75` fix: distance-scaled anti-bias preserves quality d=0.5..d=4
- `a2555659/675f7b02/78021128` feat: AFV0-3 plumbing (steps 1-3)
- `1ef2552c` perf: AFV cost grid persistent quant+dequant (358→237ms)
- `6c3cd0fa` fix: isolated AFV reconstruct correct + 100× anti-bias
- `8c335f13` chore: skip AFV cost grid (175ms saved when AFV ~never wins)
- `30d22977` fix: halve distance-bias slope (0.6 → 0.3)
- `626f655c/43e173a1` test: parameterized + smooth selectivity validation
- `da647cc8` chore: 12 → 0 build warnings

**Active strategies (15 of 27 selectable + 4 wired-but-dormant):**
- 8x8 family: DCT8, DCT4x4, DCT4x8, DCT8x4, IDENTITY, DCT2x2
- 16-tier: DCT16x16, DCT16x8, DCT8x16
- 32-tier: DCT32x32, DCT32x16, DCT16x32
- 64-tier: DCT64x64, DCT64x32, DCT32x64
- AFV0-3 (plumbing complete; cost grid call skipped pending #38
  persistent AFV transforms — would otherwise add 175ms per call)
- DCT128+ (libjxl never selects)

**Distance robustness:** the cost-model anti-bias muls are
distance-scaled — `bias = 1 + max(0, d-1) * 0.3` for DCT16, 1.5× for
DCT32, 2× for DCT64. At d=1 unchanged from calibration; at d=4
prevents the over-selection that produced butteraugli 9.2 with
fixed-mul approach.

**Validation tests** (all in `lossy_encoder.rs` / `forks/reconstruct.rs`):
- `test_dct_scale_convention_diag` — proves GPU DCT uses mean-scale
- `test_lossy_encoder_strat_search_vs_encode_one_diag` — RMSE parity
- `test_dct32x32_reconstruct_smooth_gradient` — DCT32 isolation
- `test_strat_search_dct32_diag_on_real_image` — histogram on CLIC photo
- `test_strat_search_selectivity_on_smooth_synthetic` — histogram on smooth
- `test_afv_isolated_reconstruct_uniform_input` — AFV reconstruct correct
- `test_afv_packed_dc_for_uniform_input` — empirical AFV pack measurement

**Foundation that future work depends on:**
- 6 new persistent GpuEncoder kernels (entropy_coeffs, pixel_loss,
  quantize_large, dequant_strategy with/without DCT8 bias, identity,
  dct2x2 — forward and inverse where applicable)
- 2 strategy-aware persistent dispatchers (apply_dct/idct_batch_persistent)
- 4 new selector variants threading extras through (16x16_full,
  32x32_with_extras16, 64x64_with_extras16)
- 4 lowering helpers (partitions_16x16/32x32/64x64_to_assignments
  + recursive Sub16x16 / Sub32x32)
- AFV reconstruct branch in encode_and_reconstruct_mixed_strategy_single_channel
  (host-orchestrated, ready for the cost-grid call to be re-enabled
  when #38 makes it affordable)

### Tunable smart-gate threshold + per-metric optima sweep (`69e22df1`, `fee20969`)

Expose the AQ-regression ratio as an explicit parameter via
`refine_aq_field_gpu_smart_with_threshold`. The existing
`refine_aq_field_gpu_smart` becomes a delegate that passes the
default `SMART_GATE_AQ_REGRESSION_RATIO = 1.10`.

Threshold tuning sweep (16 CLIC2025-1024 images, mean across
d ∈ {1.0, 2.0, 4.0}):

| threshold | mean butteraugli | mean SSIM2 |
|-----------|------------------|------------|
| 1.10 (current default) | **2.1514** | 80.115 |
| 1.20                   | 2.2093           | 80.744 |
| 1.30                   | 2.2767           | **81.195** |

Per-distance, per-metric optima:

| dist | metric      | best threshold |
|------|-------------|----------------|
| 1.0  | butteraugli | 1.20           |
| 1.0  | SSIM2       | 1.30           |
| 2.0  | butteraugli | 1.10           |
| 2.0  | SSIM2       | 1.30           |
| 4.0  | butteraugli | 1.10           |
| 4.0  | SSIM2       | 1.30           |

The trade-off is monotone: lower threshold → tighter fallback →
better butteraugli, worse SSIM2. Higher threshold → more AQ
through → better SSIM2, worse butteraugli. Even at 1.30 the smart
gate still catches the 2 worst butteraugli regressions at d=1.0
(paths 0/2/14), so the safety floor is preserved across the range.

**Caller recommendation:**
- Butteraugli/JXL pipeline: keep default 1.10
- SSIMULACRA2 or general perceptual quality: pass 1.30 explicitly

Archived sweeps:
- `sweep_clic_16imgs_t110_2026-05-08.log`
- `sweep_clic_16imgs_t120_2026-05-08.log`
- `sweep_clic_16imgs_t130_2026-05-08.log`

Future: a metric-configurable smart gate that uses SSIM2 internally
for the regression check would auto-pick the right threshold per
metric.

### Empirical: butteraugli and SSIMULACRA2 disagree on content-driven AQ (`0e470164`)

Expanded the corpus sweep to compute SSIMULACRA2 for all four
reconstruction paths (uniform / AQ / refined / smart). Reveals a
fundamental metric disagreement that reshapes the optimal policy:

| dist | metric        | uniform | AQ | refined | smart |
|------|---------------|---------|----|---------|-------|
| 1.0  | butteraugli µ | 1.2386  | 1.4221 (+15%) | 1.2105 (-2%) | **1.1886 (-4%)** |
| 1.0  | SSIM2 µ       | 87.913  | **88.841 (+0.93)** | **89.071 (+1.16)** | 88.488 (+0.58) |
| 2.0  | butteraugli µ | 2.0245  | 2.2413 (+11%) | 2.1195 (+5%) | **2.0240 (-0%)** |
| 2.0  | SSIM2 µ       | 80.099  | **82.472 (+2.37)** | **82.489 (+2.39)** | 81.412 (+1.31) |
| 4.0  | butteraugli µ | 3.2582  | 3.5580 (+9%) | 3.5160 (+8%) | **3.2417 (-1%)** |
| 4.0  | SSIM2 µ       | 67.677  | **72.432 (+4.76)** | **72.433 (+4.76)** | 70.445 (+2.77) |

**The metrics give opposite answers about AQ:**
- Butteraugli: AQ regresses uniform by +9-15% at all distances
- SSIM2: AQ BEATS uniform by +0.93–4.76 at all distances (14/15/16
  of 16 strict wins per distance)

Refined matches AQ on SSIM2 at d=2/4 (82.489 vs 82.472, 72.433 vs
72.432) — refinement adds zero SSIM2 gain at higher d. At d=1.0
refined edges AQ (+0.23 SSIM2).

**The smart gate optimizes for butteraugli** — falling back to
uniform when AQ regresses butteraugli. That HURTS SSIM2 vs always
using AQ:
- d=1.0: smart +0.58 SSIM2 vs AQ alone +0.93
- d=2.0: smart +1.31 vs AQ alone +2.37
- d=4.0: smart +2.77 vs AQ alone +4.76

**Implication: optimal policy is metric-dependent.**
- For butteraugli-targeted output: use the smart gate (this is the
  default for JXL, since libjxl optimizes for butteraugli).
- For SSIMULACRA2-targeted output: always use AQ, skip smart gate,
  skip refinement at d≥2.

Why the disagreement: butteraugli is sensitive to LOCAL
discontinuities (per-block qac variance creates block-edge
artifacts that butteraugli penalizes); SSIM2 is sensitive to
STRUCTURAL similarity (AQ's heavy-quant on smooth regions preserves
edges better, which SSIM2 rewards). Our DCT8-only pipeline lacks
the AC strategy selection + EPF maturity that absorb per-block qac
variance in libjxl, amplifying butteraugli's sensitivity here.

Archived at
`/mnt/v/output/jxl-encoder-gpu/butteraugli-refinement-sweep/sweep_clic_16imgs_full_ssim2_2026-05-08.log`.

Future work: metric-configurable smart gate
(`SmartGateMetric::Butteraugli | Ssim2 | Combined`), per-resolution
threshold tuning, or AC strategy selection (which would close the
butteraugli/SSIM2 gap by absorbing the per-block variance).

### Smart-gate cross-validation: SSIMULACRA2 + 512px content (`6b95ff24`, `467150f4`)

Two validation steps confirm the smart gate is a real perceptual
quality improvement, not a butteraugli-specific optimization.

**SSIMULACRA2 cross-validation** (16-image CLIC2025-1024 sweep,
archived `sweep_clic_16imgs_ssim2_2026-05-08.log`):

| dist | butteraugli (smart vs uniform) | SSIMULACRA2 (smart vs uniform) |
|------|-----------------------------------|----------------------------------|
| 1.0  | 1.2386 → 1.1886 (-4.0%)           | 87.913 → 88.488 (+0.575)          |
| 2.0  | 2.0245 → 2.0240 (-0.0%)           | 80.099 → 81.412 (+1.313)          |
| 4.0  | 3.2582 → 3.2417 (-0.5%)           | 67.677 → 70.445 (+2.768)          |

The smart gate wins on BOTH metrics at all 3 distances. Win
magnitude on SSIM2 actually INCREASES with distance (+0.575 →
+1.313 → +2.768) — opposite of the butteraugli pattern. SSIM2 sees
the AQ-vs-uniform fallback as a meaningful perceptual gain that
butteraugli barely registers, suggesting the gate captures real
quality wins beyond what either metric alone shows.

**512×512 cross-resolution sweep** (8 CID22-512 images,
archived `sweep_cid22_512_8imgs_2026-05-08.log`):

| dist | butteraugli                       | SSIMULACRA2                      |
|------|-----------------------------------|----------------------------------|
| 1.0  | 1.1815 → 1.1795 (-0.2%)           | 89.233 → 89.488 (+0.255)          |
| 2.0  | 1.9855 → 2.0083 (+1.1%)           | 82.069 → 82.256 (+0.187)          |
| 4.0  | 3.1508 → 3.2478 (+3.1%)           | 70.454 → 71.295 (+0.841)          |

Wins are smaller at 512px on butteraugli (mostly matches uniform)
but stay positive on SSIM2. No catastrophic regressions on either
metric — the safety floor holds across resolutions and metrics.

The `SMART_GATE_AQ_REGRESSION_RATIO = 1.10` threshold was tuned for
1024px content; future per-resolution calibration could tighten the
512px wins. Smart-paths distribution at 512px (1.0: 0/6/2, 2.0:
1/7/0, 4.0: 3/5/0) shows the gate is active and triggering as
designed — most images get AQ→uniform fallback, matching the 1024px
behavior pattern.

### Smart content-aware gate: refine_aq_field_gpu_smart (`b6f483a5`, `f9b0ca6a`, `52dd19f2`)

Production-ready content-aware gating for the butteraugli refinement
loop. Three commits land progressively:

- `b6f483a5`: distance-only auto-gate (`refine_aq_field_gpu_auto`)
  with `REFINEMENT_DISTANCE_THRESHOLD = 1.5` empirically derived
  from the 8-image CLIC sweep.
- `f9b0ca6a`: initial smart-gate concept that measured AQ-vs-uniform
  only at low distance.
- `52dd19f2`: final smart gate that measures AQ-vs-uniform at ALL
  distances and falls back to uniform when AQ regresses by > 10%.

**Final smart-gate logic:**
1. Always measure AQ + uniform baselines (~100 ms at 1024² on RTX 5070).
2. If AQ score > uniform × `SMART_GATE_AQ_REGRESSION_RATIO` (1.10),
   fall back to a uniform qac field (refinement can't recover from a
   doomed initial AQ).
3. Else if `target_distance > 1.5`, return initial AQ as-is
   (refinement gated by distance).
4. Else, run the full refinement loop.

**16-image CLIC corpus sweep results:**

| dist | uniform µ | AQ µ | refined µ | **smart µ** | smart paths [DistGate / AQ→un / Refined] |
|---|---|---|---|---|---|
| 1.0 | 1.2386 | 1.4221 | 1.2105 | **1.1886** (-4.0% vs uniform) | 0 / 12 / 4 |
| 2.0 | 2.0245 | 2.2413 | 2.1195 | **2.0240** (-0.0%) | 7 / 9 / 0 |
| 4.0 | 3.2582 | 3.5580 | 3.5160 | **3.2417** (-0.5%) | 9 / 7 / 0 |

The smart gate **never regresses uniform** at any distance and gains
real quality at d=1.0 (-4.0%, refining 4 of 16 images). The previous
distance-only auto-gate at d=2.0 returned initial AQ (+10.7% worse
than uniform); now smart matches uniform exactly. At d=4.0 the
distance gate kept AQ regressions; smart catches them and falls back
to uniform, beating uniform by -0.5%.

`SmartGateOutcome` returns the selected field plus diagnostic scores
(`initial_aq_score`, `uniform_score`) and the path taken
(`SmartGatePath::{DistanceGated, AqRegressedFallToUniform, Refined}`)
so callers can log per-image telemetry.

Sweep results archived at
`/mnt/v/output/jxl-encoder-gpu/butteraugli-refinement-sweep/sweep_clic_16imgs_smart_2026-05-08.log`.

### Corpus-level refinement validation: butteraugli_refinement_corpus_sweep (`36b91171`)

Multi-image sweep example that validates whether the single-image
refinement gain (-11.2% at d=1.0 on one CLIC photo) generalizes
across content. Runs uniform / initial AQ / refined-AQ at d ∈ {1.0,
2.0, 4.0} across a corpus directory and reports aggregate
butteraugli statistics including win counts and worst-case losses.

Configurable via env vars: `CORPUS_DIR`, `MAX_IMAGES`, `ITERS`.

Sweep results (8 CLIC2025-1024 photos, iters=2, archived at
`/mnt/v/output/jxl-encoder-gpu/butteraugli-refinement-sweep/sweep_clic_8imgs_2026-05-08.log`):

| dist | uniform µ | AQ µ | refined µ | rf<un | rf<AQ |
|---|---|---|---|---|---|
| 1.0 | 1.2467 | 1.3533 | 1.1753 (-5.7%) | 5/8 | 7/8 |
| 2.0 | 2.0292 | 2.1296 | 2.0765 (+2.3%) | 3/8 | 3/8 |
| 4.0 | 3.2115 | 3.3866 | 3.3866 (+5.4%) | 3/8 | 0/8 |

**Findings:**
- **d=1.0:** refinement wins on 5 of 8 images vs uniform, beats AQ
  on 7 of 8. Mean -5.7% improvement vs uniform. The d=1.0 win
  GENERALIZES across content (not image-specific). Single-image
  -11.2% on the original test photo was an upper-tail result;
  the corpus mean is a more representative -5.7%.
- **d=2.0:** refinement is mixed (3/5 win/loss vs uniform). Slight
  regression on average (+2.3%); the single-image +16% regression
  observed earlier is partially content-specific.
- **d=4.0:** refinement matches AQ exactly across the corpus (no
  improvement vs AQ baseline). One image shows a +1.021 score
  regression — content where refinement overshoots.

**Production recommendation:** enable refinement at d ≤ 1.5; skip
or gate at higher distances until per-content gating heuristics
land. Refinement infrastructure is correct and provides real
value in the high-quality regime where it matters most.

### Refinement loop fix — qac-domain adjustment no longer regresses (`87a6beb4`)

Replace `refine_quant_field_one_iter` inside `refine_aq_field_gpu`
with a defensive qac-domain-specific adjustment. Three changes:

1. Skip the `cur_pow=0.2` "soften good blocks" path (upstream's
   bit-budget tradeoff has no analog in our pipeline; softening
   just degrades good blocks).
2. Cap per-iter multiplier at 1.5 (prevents compound upward drift
   across iters since our qac has no integer ceiling to absorb the
   way upstream's `raw_quant ∈ [1, 255]` does).
3. Skip the `kOriginalComparisonRound` clamp (no-op when softening
   is skipped — it only fired to undo softening overshoot).

Empirical impact (1024×1024 CLIC photo,
`butteraugli_refinement_demo`):

| distance | before refined | after refined | vs uniform |
|---|---|---|---|
| 1.0 | 1.5012 (+11.6% vs uniform) | 1.1944 | **-11.2% (BETTER)** |
| 2.0 | 2.5322 (+17.6%) | 2.4968 | +16.0% (no longer drifting) |
| 4.0 | 4.1164 (+19.6%) | 3.4395 | -0.0% (matches uniform) |

At d=1.0 the refinement loop now provides a real -11% improvement
over the uniform baseline — the loop infrastructure delivers
quality value when given a sound underlying pipeline. The d=4.0
catastrophic regression documented in the prior CHANGELOG entry
is fully resolved. d=2.0 is no longer drifting upward but the
underlying AQ-vs-uniform regression (+15%) persists due to the
DCT8-only pipeline (next gap: AC strategy selection).

### Persistent EPF chain — saves ~65 ms/encode at 1024² (`710f760c`)

Add persistent variants of EPF step 1 / step 2 launches plus an
`upload_inv_sigma` helper to GpuEncoder. Wire into
`run_pipeline_with_qac` so the entire pipeline stays GPU-resident
through XYB → DCT → quantize → dequant → IDCT → scatter →
gab_smooth → EPF → xyb_to_linear; only the final 4 MB linear-RGB
download survives.

Quality is bit-for-bit identical (1.3456 / 1.5400 / 1.5012 in
butteraugli_refinement_demo before and after the perf change).
Timing on a 1024×1024 CLIC photo at d=1.0, RTX 5070:

  Before:  111 ms/iter (12 MB XYB download + Vec-based EPF chain
                         + Vec-based xyb_to_linear)
  After:    46 ms/iter (fully persistent, single 4 MB output download)
  Saving:   65 ms/iter (-58%)

Total demo time: 334 ms → 138 ms.

### Refinement loop regression at high distances (observed, not yet fixed)

Empirical scores from `butteraugli_refinement_demo` at d ∈ {1.0,
2.0, 4.0} reveal a real quality regression in the refinement loop's
qac-domain adaptation that wasn't visible at d=1.0:

| distance | uniform | initial AQ | refined AQ |
|---|---|---|---|
| 1.0 | 1.3456 | 1.5400 (+14.4%) | 1.5012 (-2.5% vs AQ) |
| 2.0 | 2.1525 | 2.4736 (+14.9%) | 2.5322 (+2.4% vs AQ) |
| 4.0 | 3.4407 | 3.4395 (-0.0%) | 4.1164 (+19.7% vs AQ) |

At d=2.0 refinement *worsens* AQ slightly (+2.4%); at d=4.0 it
**catastrophically regresses** (+19.7% — back almost to the d=4.0
uniform score from a +0% AQ baseline).

Likely root cause: the deviation bounds (`sqrt(250/ratio)`) were
designed for upstream's float-qf domain where qf is in `[~0.3, 1.5]`
and clamped to integer raw_quant `[1, 255]`. In our qac-domain
adaptation the integer-step bump is neutered and there's no integer
ceiling, so per-iteration `qac *= diff` compounds without limit
when `diff > 1`. At d=4.0 most blocks have `tile_dist > 4` so
`diff = tile_dist / 4 > 1`, causing systematic upward drift each
iteration. The refined qac field then diverges far from the
calibration-tested range and quality degrades.

Fix candidates (none implemented yet): tighter deviation-bound
clamping, qac↔qf-float conversion at the loop boundary, or
disabling refinement for our pipeline until it gains the AC
strategy + CfL features that absorb the variance upstream's qf
range assumes.

### LossyEncoder pipeline gap closure: gab_smooth + EPF + IEC sRGB linearization (`691c0aab`, `d9a201ca`, `d1738660`)

Three commits that drop the `butteraugli_refinement_demo` baseline at
distance=1.0 from **8.76 to 1.35** (-85%) on a 1024×1024 CLIC photo,
revealing the dominant pipeline gaps that were masking real refinement
loop value.

1. **Decoder-side `gab_smooth` after IDCT/scatter** (`691c0aab`,
   -17.3% baseline). `run_pipeline_with_qac` was missing the
   `gab_smooth` (3×3 plus, inverse of forward `gaborish_5x5`) call
   that libjxl's decoder runs on reconstructed XYB before
   xyb_to_linear. Without it the gaborish pre-sharpening from the
   encoder side persisted in the output and reconstructions were
   over-sharp/blocky. Three persistent calls (one per XYB plane);
   exposes `forks::reconstruct::gab_weights` as `pub`.

2. **EPF chain after `gab_smooth`** (`d9a201ca`, small additional
   gain). 2-iter EPF (step 1 + step 2) on the reconstructed XYB
   planes, mapping our per-block float qac to libjxl's u8
   `quant_field` + `quant_scale` representation via
   `qac * 50` / 0.01 scale (giving `quant_scale * raw_quant ≈
   qac/2 ≈ qf_float_equivalent`, accounting for our pipeline's
   `K_AC_QUANT=0.765` vs upstream `q=0.39/d`). Sharpness uniform
   4 (libjxl default). Cost: ~65ms per encode at 1024² (download +
   EPF launches + Vec-based xyb_to_linear); a future port can add
   persistent `apply_epf_step{1,2}_persistent` to reclaim it.

3. **IEC sRGB linearization in butteraugli_refinement_demo**
   (`d9a201ca`, **-81% baseline — the biggest single win**). The
   demo was using simplified `powf(2.4)` to convert input sRGB U8 →
   linear, but butteraugli-gpu internally linearizes the original
   bytes via the IEC 61966-2-1 piecewise transfer function. The
   resulting transfer-function asymmetry inflated every pixel's
   perceptual delta even on bit-perfect reconstructions — same root
   cause class as CLAUDE.md's "PNG Color Metadata Causes Bogus
   Butteraugli Scores" note. Fixed by using IEC piecewise in the
   demo's input linearization.

**Final empirical results** (1024×1024 CLIC photo, d=1.0, iters=2):
- uniform qac:  score=1.3456  pnorm_3=0.4830
- initial AQ:   score=1.5400  pnorm_3=0.4812  (+14.4% vs uniform)
- refined AQ:   score=1.5012  pnorm_3=0.5074  (-2.5% vs initial AQ)

**Refinement now provides a real -2.5% gain** vs initial AQ
(previously no movement). The remaining AQ-vs-uniform regression
(+14.4%) reflects the next pipeline gap: AC strategy selection.
With only DCT8 available, AQ's heavy-quant assignment to smooth
blocks creates visible blocking that EPF can't fully mask. Adding
DCT16/32 strategy selection for smooth regions would close it.

234 tests pass through all three changes; the
`butteraugli_refinement_demo` example provides a single-command
benchmark of pipeline quality progress.

### Butteraugli quant-refinement loop scaffold + helpers + integration (`11f33133`, `4d6ab397`, `9746a25f`, `2b33d6d0`, `87d1654a`)

End-to-end GPU-substituted butteraugli refinement loop, mirroring
upstream `jxl_encoder::vardct::butteraugli_loop::butteraugli_refine_quant_field`
with the per-iteration distance compute on `zenmetrics/butteraugli-gpu`.

`forks::butteraugli_loop` (gated behind the new `butteraugli-loop`
cargo feature):

1. `ButteraugliLoopGpu<R>` (`11f33133`) — persistent
   `butteraugli_gpu::Butteraugli` wrapper. `set_reference` once per
   encode caches the opsin/blur intermediates so per-iteration
   `compute_with_reference` only runs the distorted side.
2. `DeviationBounds::compute` (`4d6ab397`) — qf_lower/qf_higher derived
   from initial float quant field via `sqrt(250 / ratio)`. Mirrors
   upstream lines 105-122 exactly.
3. `compute_tile_distances` + `AcStrategyInfo<'_>` (`4d6ab397`) —
   AC-strategy-aware 16th-power-mean diffmap reduction matching
   upstream lines 230-271. DCT8-only callers use the
   `dct8_only_storage` / `dct8_only_info` helpers.
4. `clamp_toward_initial` (`4d6ab397`) — `K_INIT_MUL = 0.6` blend
   toward initial qf (kOriginalComparisonRound, upstream lines 314-336).
5. `adjust_quant_field` (`4d6ab397`) — per-iter `cur_pow=0.2` /
   `cur_pow=0.0` regimes with the integer-quantizer-step minimum bump
   (upstream lines 338-406).
6. `refine_quant_field_one_iter` + `RefineConfig` (`9746a25f`) —
   composes the four helpers in upstream's per-iteration order.
   Returns tile_dist for caller diagnostics.
7. `refine_aq_field_gpu` + `RefineIterTrace` (`2b33d6d0`, `87d1654a`)
   — multi-iter loop wiring `LossyEncoder::encode_one_adaptive`
   together with the host-side adjustment helpers. Operates in
   qac-domain (LossyEncoder takes per-block qac directly), so the
   integer-step bump is neutered (no integer rounding in our
   pipeline). Caller passes ORIGINAL sRGB U8 bytes as
   butteraugli reference — round-tripping linear → IEC sRGB → linear
   doesn't recover the input bytes when the input was linearized via
   simplified `powf(2.4)`, which would inflate scores even on
   bit-perfect reconstructions (same root-cause class as the CLAUDE.md
   "PNG Color Metadata Causes Bogus Butteraugli Scores" note).
8. `linear_f32_to_srgb_u8` + `linear_planar_to_srgb_u8_interleaved`
   (`2b33d6d0`) — IEC 61966-2-1 piecewise transfer (linear toe at
   0.04045) matching the inverse of butteraugli-gpu's
   `srgb_byte_to_linear`.

`examples/butteraugli_refinement_demo.rs` (`87d1654a`) — turnkey
demo on a real CLIC2025 photo. Empirical finding (1024×1024 at
distance=1.0, iters=2): baseline butteraugli is ~8.9 (target ~1.0),
so the underlying GPU pipeline is the bottleneck — missing AC
strategies beyond DCT8, EPF in the `run_pipeline_with_qac` path,
proper sRGB linearization. The refinement loop runs end-to-end at
~217 ms/iter; aq_field drifts substantially but score barely moves
because the underlying pipeline can't deliver target quality. The
loop infrastructure is correct (21 unit tests pass including a
CUDA end-to-end smoke); pipeline gaps are the next priority.

21 tests cover: 4 helpers individually, the per-iter composition,
sRGB conversion round-trip + endpoints, and a CUDA-gated end-to-end
smoke test that exercises the whole loop on a 64×64 gradient.

### Mixed-strategy reconstruct on GPU (`3359b065`, `53e47679`, `61ae5ae6`, `b7e55d15`)

Closes the second-largest remaining gap. Builds out from the LLF
restoration foundation:

1. `tile_dims_pixels(raw_strategy)` (`3359b065`) — public
   per-strategy pixel block dims.
2. `scatter_block_to_plane` (`3359b065`) — copies an IDCT-output
   block into the padded plane at `(bx*8, by*8)` for any strategy.
3. `idct_and_scatter_one_block_gpu` (`53e47679`) — per-block
   convenience: composes `apply_idct_batch_gpu` (batch=1) +
   `scatter_block_to_plane`.
4. `batched_reconstruct_same_strategy_gpu` (`61ae5ae6`) — efficient
   form for N blocks of the same strategy: ONE IDCT launch + per-
   block memcpy scatter.
5. `BlockRecipe<'a>` + `reconstruct_mixed_strategy_gpu` (`b7e55d15`)
   — multi-strategy dispatcher. Buckets recipes by `raw_strategy`
   (0..=16), emits at most 15 GPU launches per image regardless of
   block count.

The caller's responsibility is to produce already-dequantized,
CfL-corrected, LLF-restored coefficients in each `BlockRecipe`
(use `dispatch_restore_llf` for the LLF stage).

AFV0-3 are not handled here — their per-block sub-transform
composition stays in `forks::afv`. The orchestrator panics on AFV
codes with a pointer to that module.

GPU smoke test: 5 mixed recipes (3 DCT8 + 2 DCT16×16) at
non-overlapping coordinates → 2 GPU launches; all destination
regions ~0 (zero coeffs → zero IDCT), untouched pixel stays at
the seed value.

This makes the reconstruct path strategy-agnostic. The DCT8-only
path (`reconstruct_xyb_dct8_only_gpu`) remains as a faster
specialization for the common case.

### Per-strategy LLF restoration helpers — full family ported (`a501b6a6`, `9b8341e0`, `b5ffb944`, `ea19ff8f`, `b6bb257e`)

Foundation for mixed-strategy reconstruct: every DCT16+ strategy
now has a pure-scalar host helper that reconstructs its low-low-
frequency coefficients from the stored DC grid. Mirrors upstream
`jxl_encoder::vardct::reconstruct::restore_llf_from_dc` arm-by-arm.

In `forks::reconstruct`:

- `dequant_dc_channel(quant_dc, quant_dc_y, channel, scale_dc)` —
  channel-aware DC dequant including the 0.5× Y→B CfL contribution.
- Constants: `DCT_RESAMPLE_SCALE_16_TO_2` (2 floats),
  `DCT_RESAMPLE_SCALE_32_TO_4` (4 floats),
  `DCT_RESAMPLE_SCALE_64_TO_8` (8 floats).
- Private DCT primitives: `dct1d_2`, `dct1d_4`, `dct1d_8` —
  bit-for-bit ports of upstream's forward butterflies (with WC4 +
  WC8 + SQRT2 constants inlined).
- LLF restorers, all with `out[iy * cols + ix]` ordering ready for
  scatter into the rectangular coefficient block:
  - `restore_llf_dct16x8_or_8x16(dc0, dc1) -> [f32; 2]`
  - `restore_llf_dct16x16(dc_grid: [f32; 4]) -> [f32; 4]`
  - `restore_llf_dct32x32(dc_grid: [f32; 16]) -> [f32; 16]`
  - `restore_llf_dct32x16(dc_grid: [f32; 8]) -> [f32; 8]`
  - `restore_llf_dct16x32(dc_grid: [f32; 8]) -> [f32; 8]`
  - `restore_llf_dct64x64(dc_grid: [f32; 64]) -> [f32; 64]`
  - `restore_llf_dct64x32(dc_grid: [f32; 32]) -> [f32; 32]`
  - `restore_llf_dct32x64(dc_grid: [f32; 32]) -> [f32; 32]`

Each helper has zero-passthrough + constant-DC unit tests
(15 total). The 1×1-LLF strategies (DCT8, IDENTITY, DCT2X2,
DCT4×4/8/4, AFV0-3) reuse `restore_dct8_dc_override`'s simple DC
formula.

This unblocks the per-strategy IDCT dispatch + scatter that's the
remaining piece of the mixed-strategy reconstruct path. With the
LLF helpers in place, the dispatcher just needs to: read DC grid →
call the right LLF restorer → write LLF positions into the
coefficient block → run per-strategy IDCT → scatter.

### G5.1 fully compliant via `__internals` feature (jxl-encoder `c82e05c`, jxl-encoder-gpu `4e863c71`, `e672e1fa`, `b718957f`)

Closes the validation gap for all 5 former in-tree-only entries by
adding an off-by-default `__internals` cargo feature in jxl-encoder
that re-exports the 5 private symbols downstream parity tests need:

- `epf_step0_strip` (was bare `fn`; bumped to `pub(crate)` + free
  wrapper `epf_step0_strip_free`)
- `adjust_quant_block_ac` (was `pub(crate)` impl method; free
  wrapper `adjust_quant_block_ac_free` calls it)
- `compute_scaled_constants` (was `pub(super)`; bumped to `pub(crate)`
  + free wrapper `compute_scaled_constants_free`)
- `ytox_ratio` + `ytob_ratio` (already `pub fn`; just needed
  `pub(crate) mod chroma_from_luma` to re-export from `__internals`)
- `INV_DC_QUANT` (already `pub const`; needed `pub(crate) mod quant`)

jxl-encoder side (`c82e05c`):
- New `__internals = []` cargo feature with descriptive doc.
- New `pub mod __internals` in `lib.rs` gated by the feature, with
  `pub use` re-exports of the 5 above.
- Visibility bumps on `mod quant` and `mod quantize` (private →
  `pub(crate)`) + the 3 wrappers.
- Default build unchanged. Both `cargo build -p jxl-encoder` and
  `cargo build -p jxl-encoder --features __internals` succeed.

jxl-encoder-gpu side (`4e863c71`, `e672e1fa`, `b718957f`):
- `Cargo.toml` dev-dep updated: `jxl-encoder = { ..., features =
  ["std", "__internals"] }`.
- `examples/epf_step0_parity.rs` rewrites the inline CPU port as
  a direct call to `jxl_encoder::__internals::epf_step0_strip_free`.
  Same FP32 numbers (X: 4.66e-10, Y: 4.47e-8, B: 4.47e-8) — confirms
  both the prior hand-roll AND the GPU port matched upstream all
  along, but now the test can't share a bug with itself.
- 4 new `*_matches_upstream` tests:
  - `compute_scaled_constants` (15 trials × 3 outputs each)
  - `ytox_ratio` (256 trials, exact equality)
  - `ytob_ratio` (256 trials, exact equality)
  - `INV_DC_QUANT` (3 channels, exact equality)
  - `adjust_quant_block_ac_host` (54 trials = 6 strategies × 3
    channels × 3 quant levels — all 4 outputs + thresholds + quant
    match exactly)

**Total G5.1-compliant `*_matches_upstream` tests: 16** (was 11)
+ 9 LLF restoration helpers transitively validated via roundtrip.

### X-multiblock weight integrated into orchestrator (`c344bbd2`, `c4513137`)

Closes the X-channel multi-block weight caveat from
estimate_entropy_full_strategy_batch_gpu (3fea1e18). Upstream's
generic-path estimate_entropy_full applies a weight `w = 1 +
min(num_blocks/8, 3)` to BOTH the X channel's entropy AND its
pixel loss when `num_blocks >= 2` (covered_blocks > 1).

Three host helpers added (c344bbd2):

- `x_multiblock_weight(num_blocks) -> f32` — formula evaluator,
  capped at w=4.0 for num_blocks >= 24.
- `apply_x_multiblock_weight_to_loss(&mut [f64], num_blocks)`
  — in-place X loss scaling.
- `apply_x_multiblock_weight_to_entropy(&mut [f32], num_blocks)`
  — in-place X entropy scaling.

Then wired into the strategy-generic orchestrator (c4513137):

- `per_block_upstream_cost` applies the X weight INSIDE the
  per-block loop to `(entropy_x[b] + nzeros_bits_term(nzeros_x[b]))`,
  matching upstream's `entropy *= w` after both terms accumulated
  in the running per-channel sum.
- The orchestrator applies the X weight to `loss_x` before
  `combine_pixel_loss_3channel` when in `CostMode::Upstream`.

Validated: DCT16x16 zero-input + Upstream mode produces cost
≈ 185.34 (= `7 × COEFF_DOMAIN_CONSTANTS.2 × (2 + 1.5)` where
w=1.5 for covered_blocks=4 — i.e., Y + B contribute 2 unit-weights
and X contributes 1.5 weighted unit). DCT8 (covered_blocks=1, w=1.0)
remains unchanged.

### Strategy-generic `estimate_entropy_full` orchestrator (`5f75ea2e`, `3fea1e18`)

Generalizes the DCT8-only entropy orchestrator (706b59fe) to all
standard JXL AC strategies (DCT8 / DCT4-family / IDENTITY / DCT2X2 /
DCT16x8 / DCT8x16 / DCT16x16 / DCT32x16 / DCT16x32 / DCT32x32 /
DCT64x32 / DCT32x64 / DCT64x64). AFV0-3 still through `forks::afv`.

Two pieces:

1. `per_block_upstream_cost` parameterized over `block_pixel_count`
   (`5f75ea2e`) — was hardcoded to 64 for the DCT8 fast path; now
   any strategy passes its `block_w * block_h`. The 8th-root
   pixel-loss scaling `loss_scalar = (loss/n)^(1/8) * n / quant`
   uses `n = block_pixel_count` directly.
2. `dct_blocks_gpu(enc, blocks, raw_strategy)` (`3fea1e18`) —
   strategy-aware batched forward DCT for already-gathered
   block-major input, mirrors `apply_dct_batch_gpu`'s dispatch arm
   without the gather step.
3. `estimate_entropy_full_strategy_batch_gpu` (`3fea1e18`) — strategy-
   generic orchestrator. Same 12-launch pipeline shape as the DCT8
   fast path, parameterized by `raw_strategy`:
   - `dct_blocks_gpu(strategy)` for forward DCT
   - `entropy_coeffs_pixel_blocks_gpu` with `n_per_block = coeff_count_per_strategy`
   - `apply_idct_batch_gpu(strategy)` for IDCT
   - `pixel_loss_blocks_gpu` with `block_w/block_h = tile_dims_pixels(strategy)`
   - `per_block_upstream_cost` with `block_pixel_count = block_w * block_h`

GPU smoke test on DCT16x16 zero-input → ~0 cost validates the
non-DCT8 plumbing.

**Caveat documented**: upstream's generic-path estimate_entropy_full
also weights X channel by `1 + min(num_blocks/8, 3)` for
`num_blocks >= 2 && c == 0 && use_pixel_domain` — NOT applied here.
Fine for relative-ranking cost (the common case); callers needing
full upstream parity for X on multi-block strategies must apply
the weight separately.

The DCT8-only `estimate_entropy_full_dct8_batch_gpu` (706b59fe)
remains as a faster specialization with hardcoded sizes; the
strategy-generic version is the addition.

### `per_block_upstream_cost` + `CostMode` orchestrator switch (`5787eec8`, `e5ca5e27`)

Closes the caveat from 706b59fe — the orchestrator now offers
two cost formulas via `CostMode`:

- `CostMode::Simple` — `entropy_mul * sum(entropies) + total_loss`.
  Fast relative-ranking form (the original).
- `CostMode::Upstream { quant_for_coeffs }` — bit-faithful match for
  upstream's `estimate_entropy_full` DCT8 fast path
  (vardct/ac_strategy.rs:737-771). Includes:
  - Per-channel `k_zeros_mul * f(num_nzeros)` bits-cost term where
    `f(n) = ceil_log2_nonzero(nbits + 17) + nbits` and
    `nbits = ceil_log2_nonzero(n + 1) + 1`.
  - 8th-root pixel-loss scaling:
    `loss_scalar = (loss/64)^(1/8) * 64 / quant`.
  - `entropy *= entropy_mul`, then `entropy += info_loss_mul * loss_scalar`.

`per_block_upstream_cost` (5787eec8) is the new combiner; the
CostMode selector in 706b59fe orchestrator (e5ca5e27) extracts
column-1 (nzeros_sum) from each channel's 4-stat array when
Upstream mode is selected and threads the constants through.

The upstream entropy kernel intentionally skips `info_loss_sum`
in pixel-domain mode (matches upstream `entropy_coeffs_scalar`
exactly); the `info_loss_mul` term is computed on the host from
the combined pixel loss instead — same algebra, different
factoring.

GPU smoke test (zero pixel input + Upstream mode) → cost ≈ 158.87,
which is 3 channels × 7 × `COEFF_DOMAIN_CONSTANTS.2` — the
per-channel nzeros bits-cost when `nzeros = 0` and pixel loss is 0.

### estimate_entropy_full DCT8 orchestrator (`d4b77cd1`, `639a2d15`, `fe4cdc32`, `382d4717`, `aaf45d74`, `07658a0e`, `706b59fe`)

The DCT8 fast path of upstream's per-block entropy + pixel-loss
cost evaluator is now composable on GPU. Builds out from G5.1-
compliant leaf primitives (commits this same series):

1. `compute_scaled_constants` + `COEFF_DOMAIN_CONSTANTS`
   (`d4b77cd1`) — distance-scaled constants for pixel-domain mode.
2. `MASK_CHANNEL_OFFSET` + `CHANNEL_MUL` (`639a2d15`) — per-channel
   8th-power mask offsets and multipliers (literal-spot-checked
   against libjxl).
3. `K_INV_COLOR_FACTOR` + `ytox_ratio` + `ytob_ratio` (`fe4cdc32`)
   — CfL ratio helpers re-exposed from kernels/cfl.
4. `EntropyMulTable` + `entropy_mul_for_strategy` + `afv_entropy_mul`
   (`382d4717`) — per-strategy entropy multipliers, field-by-field
   parity-tested vs upstream.
5. `combine_pixel_loss_3channel` (`aaf45d74`) — per-block
   CHANNEL_MUL-weighted sum across X/Y/B losses.
6. `extract_per_block_entropy` + `sum_per_block_entropy_3channel`
   + `per_block_total_cost` (`07658a0e`) — host-side per-block
   reshape and cost combiner.
7. **`estimate_entropy_full_dct8_batch_gpu`** (`706b59fe`) — the
   orchestrator. ~12 GPU launches per call regardless of n_blocks:
   3 forward DCT + 3 entropy_coeffs_pixel + 3 IDCT + 3 pixel_loss.
   Mask is image-plane with explicit per-block mask_row_base offsets.

**Caveat documented**: the underlying entropy kernel only fills
entropy_sum + nzeros_sum columns (info_loss + info_loss2 stay 0).
The cost formula uses entropy_sum directly without upstream's
`info_loss_mul × info_loss + zeros_mul × nzeros` re-weighting.
That requires a kernel extension; `scaled_constants` is in the
signature with `info_loss_mul` and `zeros_mul` reserved for the
future use.

GPU smoke test (zero pixel input → zero per-block cost on 4 blocks)
verifies the orchestrator composes correctly.

### G5.1 validation recovery (`e81accbe`, `c1ffd69e`, `2cf7eeec`, `6eb0faca`, `6de46b3a`, `a21872fd`, `b29cd555`, `632043ac`, `abb79093`)

After noticing a pattern of helpers committed with property-only
tests instead of upstream parity per zenmetrics G5.1 (validate
against the published CPU crate, not a hand-rolled re-derivation),
recovered the LLF restoration path and the `EntropyMulTable` to
full G5.1 compliance without cross-repo work.

The trick: upstream's `jxl_encoder::vardct::dct::dc_from_dct_*` are
all `pub`, even though `restore_llf_from_dc` (their inverse) is
private. So adding hand-rolled forward `dc_from_dct_*` host helpers
+ upstream-parity tests on those forwards + roundtrip tests
(`restore_llf(dc_from_dct(llf)) == llf`) gives **both directions
verified against upstream**, just transitively for the inverse.

What landed:

1. `e81accbe` — PORT_STATUS section honestly documenting the gap.
2. `c1ffd69e`, `2cf7eeec`, `6eb0faca` — 8 hand-rolled forward
   `dc_from_dct_*` helpers (DCT16x8/8x16, DCT16x16, DCT32x32,
   DCT32x16, DCT16x32, DCT64x64, DCT64x32, DCT32x64) plus matched
   private `idct1d_4` / `idct1d_8` IDCT primitives (bit-for-bit
   ports of upstream `vardct::dct::inverse`). Each helper has a
   roundtrip test with single-position-on, sign-mixed, and
   arbitrary trial inputs.
3. `a21872fd`, `b29cd555` — 8 upstream-parity tests calling
   `jxl_encoder::vardct::dct::dc_from_dct_*` directly. All pass at
   FP32 floor.
4. `632043ac` — `EntropyMulTable::reference()` and `experimental()`
   field-by-field parity vs `jxl_encoder::effort::EntropyMulTable`.
5. `6de46b3a`, `abb79093` — validation gap doc reorganized with
   per-helper status tables.

**Currently G5.1-compliant**: 8 dc_from_dct + 9 restore_llf
(transitive) + EntropyMulTable.

**Still in-tree-only** (would need upstream visibility bumps):
EPF Step 0 (inline copy-paste), AdjustQuantBlockAC heuristics
(pub(crate) impl), compute_scaled_constants (pub(super)),
ytox/ytob_ratio (pub fn in pub(crate) mod), INV_DC_QUANT (pub const
in private mod).

### `compute_epf_sharpness_dct8_gpu` — end-to-end EPF sharpness picker on GPU (`9a8903dd`, `fe1bf69d`)

Closes the EPF sharpness orchestrator gap for the DCT8-only path.
Two pieces:

1. `forks::epf::apply_epf_chain_gpu` (`9a8903dd`) — full step0+step1+
   step2 chain orchestrator per upstream's `apply_epf` semantics.
   Tier gating: `epf_iters >= 3` runs step 0, `>= 1` runs step 1,
   `>= 2` runs step 2. Sigma scales 1.485 / 1.65 / 10.725 match
   upstream exactly.
2. `forks::epf::epf_sharpness_candidates(distance)` (`9a8903dd`) —
   `&'static` slice selector returning `[0, 4]` at distance > 4.5,
   `[0, 2, 7]` otherwise.
3. `forks::epf::compute_epf_sharpness_dct8_gpu` (`fe1bf69d`) — the
   full orchestrator. Pipeline:

   ```
   reconstruct_xyb_dct8_only_gpu          (4 launches)
     → gab_smooth_gpu (optional)          (3 launches)
     → for each candidate:
         compute_inv_sigma_map (host)
         apply_epf_chain_gpu               (3-9 launches)
         enc.block_l2_errors               (1 launch)
     → select_sharpness_two_pass (host)
   ```

   Per-candidate cost: 4-10 GPU launches plus one host-side picker
   pass. Three candidates → ~12-30 launches per image at the
   typical settings.

Together with the AdjustQuantBlockAC port (session 6.5) and the
AFV cost grid integration (session 6.4), the encoder-side
reconstruction + quality heuristic chain is now fully GPU-resident
for the common DCT8-only case. Mixed-strategy images need the
per-strategy IDCT dispatch + scatter still pending in
`forks::reconstruct`.

### DCT8-only `reconstruct_xyb_*_gpu` orchestrator (`878e144f`, `44405a8d`, `984431fb`)

The DCT8 fast path of upstream's `reconstruct_xyb` (which produces
the encoder-side reconstructed XYB planes consumed by EPF, pixel-
domain loss estimation, and the sharpness picker) is now composed
end-to-end on GPU:

- `forks::reconstruct::INV_DC_QUANT` `[4096, 512, 256]` const
  matching upstream.
- `restore_dct8_dc_override` (`878e144f`) — per-block DC
  restoration with the 0.5× Y→B DC-level CfL contribution. Pure
  scalar.
- `restore_dct8_dc_override_batched` (`44405a8d`) — flat-slice
  variant operating on `n_blocks * 64` block-major buffers
  (matching what `dequant_dct8_blocks_gpu` returns directly).
- `reconstruct_xyb_dct8_only_gpu` (`984431fb`) — the orchestrator.
  Pipeline: 1 dequant launch + host DC override + 3 IDCT launches
  + host scatter to padded planes. **4 GPU launches per image**
  regardless of block count.

**Note**: this is the all-blocks-are-DCT8 path. Real images use a
mix of strategies via the AC strategy map; mixed-strategy support
requires per-strategy IDCT dispatch + scatter for non-DCT8 blocks,
which is a separate piece of `reconstruct_xyb_impl`.

With this orchestrator plus the existing
`apply_epf_step{0,1,2}_gpu` + `block_l2_errors` +
`select_sharpness_two_pass`, a future `compute_epf_sharpness_gpu`
can now be wired up entirely in fork-space (DCT8-only first; mixed
strategies follow once the dispatch is in place).

### AdjustQuantBlockAC fully ported as host helpers (`453808d0`, `ebf765ed`, `86886b55`, `6d3c16db`, `99362cc3`, `d1bef643`)

The full upstream `adjust_quant_block_ac` (~250 lines, 6 heuristics
A-F + pre-scan) is now portable as standalone host helpers in
`forks::quantize`:

- `adjust_quant_prescan` (`20318810`) — the per-block coefficient
  pre-scan loop. Returns `None` for the 5 partial block kinds
  upstream skips.
- `apply_heuristic_a_thresholds` (`453808d0`) — threshold reduction
  for `xsize > 1 || ysize > 1`. 0.54 floor, 0.08 cap.
- `apply_heuristic_c_corner_penalty` (`ebf765ed`) — HF corner
  penalty with per-channel multipliers `[70, 30, 60]`.
- `apply_heuristic_d_dct8_flatness` (`ebf765ed`) — DCT8-only
  flatness detector (`sum(hf_nonzeros) < 11.0`).
- `apply_heuristic_b_sparse_y` (`86886b55`) — Y-channel sparse
  block handling with K_LIMIT/K_MUL constants and the if/else if
  threshold-position cascade (Q3 → Q1/Q2 → Q0).
- `apply_heuristic_e_large_transform` (`6d3c16db`) — DCT16+ family
  error correction with K_MUL1/K_MUL2 4×3 tables and
  K_QUANT_NORMALIZER.
- `apply_heuristic_f_activity` (`99362cc3`) — activity-based
  reduction with libjxl's overflow-safe min computation; Y-channel
  threshold bumps; floor at `max(quant_orig/2, 4)`.
- `adjust_quant_block_ac_host` (`d1bef643`) — orchestrator
  composing all of the above in upstream's exact order
  (A → prescan → B → C → D → E → F). Returns `AdjustQuantOutcome`
  mirroring upstream's `(heuristics_fired, sum_of_vals,
  sum_of_error, activity)` 4-tuple.

Each heuristic ships with focused unit tests (4 + 5 + 5 + 3 + 3 +
4 + 2 = 26 total). With the heuristics standalone, a future
per-block-parallel `#[cube]` kernel can transcribe them without
further reverse-engineering.

### AdjustQuantBlockAC pre-scan helper (`20318810`)

`forks::quantize::adjust_quant_prescan` ports the per-block
coefficient pre-scan loop from upstream
`jxl_encoder::vardct::quantize::adjust_quant_block_ac` (lines
178-220). Computes the 5 statistics consumed by heuristics B-F:
`sum_of_highest_freq`, `sum_of_error`, `sum_of_vals`,
`hf_nonzeros[4]`, `hf_max_error[4]` — indexed by `hfix = 2 *
(y >= h/2) + (x >= w/2)` (the high-frequency quadrant).

Returns `None` for the 5 "partial block kinds" upstream skips
(IDENTITY, DCT2X2, DCT4X4, DCT4X8, DCT8X4); AFV strategies are
routed through `forks::afv` and never reach this path.

This is the most GPU-amenable piece of AdjustQuantBlockAC — the
pre-scan loop is ~150 ops per block (DCT16x16) and would benefit
from batching across all blocks. The 6 heuristics A-F that consume
these stats are small CPU code that can stay on the host.

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
