# jxl-encoder-gpu — Claude Code instructions

## What this crate is

Multi-vendor GPU kernels for `jxl-encoder` via [CubeCL](https://github.com/tracel-ai/cubecl).
Mirrors `jxl-encoder-simd` 1:1 — every public function in `jxl-encoder-simd`
(scalar reference path) is the parity target for one CubeCL kernel here.

We dispatch across CUDA (NVIDIA), WGPU (Vulkan/Metal/DX12/WebGPU), HIP (AMD),
and CPU (cubecl-cpu) from a single `#[cube]`-annotated kernel source.

## Architectural position vs jxl-encoder CPU

This crate is a **GPU acceleration library**, NOT a competing JXL bitstream
encoder. The structure:

```
~/work/zen/jxl-encoder/jxl-encoder       (live CPU encoder, sibling workspace)
   ↑
   │ pulled in via path dep:
   │   jxl-encoder = { path = "../jxl-encoder/jxl-encoder", ... }
   │
~/work/zen/jxl-encoder-gpu/jxl-encoder-gpu (this crate)
   ├── src/forks/        ← parallel GPU-flavored copies of CPU modules that
   │                       traffic in GpuBlocks/GpuPlane handles instead of
   │                       host Vec<f32>. Header comment in each fork says
   │                       "Forked from jxl-encoder X — Why a fork instead
   │                       of editing jxl-encoder: zero edits to upstream."
   ├── src/persistent.rs ← GPU-only state machinery (handle types, batch ops)
   ├── src/kernels/      ← #[cube] kernel sources
   ├── src/launch/       ← kernel launchers (CubeDim sizing, etc.)
   └── src/encoder.rs    ← thin facade. encode_lossy_via_cpu is currently a
                            passthrough to jxl-encoder's CPU path; none of
                            our GPU work reaches the bitstream stage.
```

### Why this shape

1. **GPU is great at the numerics, terrible at the bitstream.** XYB /
   gaborish / EPF / mask1x1 / butteraugli / DCT / IDCT / quant are dense
   FP arithmetic with regular access patterns — perfect SIMT fit.
   ANS/Huffman entropy coding, context modeling, bit packing, container
   assembly are bit-serial, branch-heavy, dependency-chained — the
   worst fit for GPU. Trying to GPU the entropy coder loses to CPU SIMD.

2. **CPU encoder is mature, complex, and not ours to break.** Forking
   the whole thing means owning a permanent fork and merging upstream
   forever. Forking only the numeric modules we GPU-port keeps the
   CPU encoder shippable as-is.

3. **Persistent GPU buffers need API control.** Stages need to hand
   each other `GpuBlocks` / `GpuPlane` handles — not host buffers
   round-tripped through PCIe per stage. The CPU encoder's APIs only
   speak host buffers, so we need parallel module copies that traffic
   in GPU handles. That's what `forks/*` is.

### The handoff gap (open, blocked on jxl-encoder API)

Today the GPU pipeline ends at "reconstructed pixels + refined AQ
field + strategy assignments + per-strategy quantized AC coefficients."
The CPU entropy coder never sees any of that — `encode_lossy_via_cpu`
makes the CPU encoder redo XYB / AQ / strat-search / DCT / quant
from scratch. So calling our GPU encoder for an actual JXL file is
currently *slower* than calling the CPU encoder directly.

To close the loop, `jxl-encoder` needs a "pre-quantized input" entry
point that ACCEPTS:
- per-strategy quantized AC coefficients (we have as `GpuI32Blocks`)
- `Vec<StrategyAssignment>` (we have)
- DC grid per channel (we have via `compute_dc_grid_per_8x8_block`)
- refined per-block qac field (we have)
- CfL ytox/ytob maps (we have)
- EPF per-block sharpness (we have)

…and SKIPS its own XYB / AQ / strat-search / DCT / quant /
butteraugli loop — just runs tokenize → ANS → bitstream assemble.
That seam is already gated behind `features = ["__internals"]` on
the dep, but the actual API isn't built yet. **Building it requires
editing jxl-encoder upstream**, which breaks the "zero edits" rule.
That trade-off is intentional and pending until the upstream encoder
quiesces.

### What's bounded to this crate (no upstream blocker)

These can be done locally without touching jxl-encoder:

- **Per-stage GPU primitives that the future handoff will consume**:
  - Group-level streaming output (256×256 groups; emit each group's
    coefficients + per-block metadata as soon as it's GPU-ready, not
    after whole-image completion). Lets the future CPU consumer
    pipeline tokenize/ANS work concurrent with later GPU groups.
  - Histogram counting on GPU (atomic-add per token bucket — well-
    known SIMT pattern, way faster than CPU on large tensors).
  - Histogram clustering on GPU (pair-merge / k-means-style — also
    SIMT-friendly). The actual ANS table build + bit-pack stays CPU,
    but the histogram-prep stages are GPU-friendly.
- **Polish on existing GPU work**:
  - Persistent AFV transforms (#38 — the only in_progress task).
    Currently the AFV cost grid is skipped (175 ms saved) because
    its host-orchestrated transforms cost too much; persistent
    versions would cut that to ~30 ms and let us re-enable AFV
    cost-grid evaluation.
  - More dimension-flexibility tests (sub-block strategies on
    arbitrary dimensions now that align-16 fix landed).
  - Per-stage benchmarks comparing GPU vs `jxl-encoder-simd` CPU
    timings (baseline measurement, not handoff-gated).
- **API hygiene**:
  - Document the smart turnkey as the recommended entry point
  - Examples + roundtrip tests against the upstream CPU encoder
    (compare GPU reconstruction butteraugli to CPU encoder's
    reconstruction)

### What's NOT in scope for this crate

- GPU entropy coder. Wrong tool. Will always lose to CPU SIMD.
- Bitstream assembly / container muxing. Same.
- Replacing the CPU encoder. The structure is "accelerate the
  numerics, leave the rest alone."

## Autonomous mandate

**Drive this port forever, until done. Do NOT stop. Do NOT ask
permission. Do NOT leave half-finished work in the tree. Do NOT
schedule "bounded pause points."**

There are exactly two states for the project:
- `in-progress` (loop continues, work continues)
- `done` (defined below; loop terminates and tells the user)

There is no third state. "Good enough for this session," "natural
stopping point," "let me check back in" — all forbidden. Token usage,
context size, time of day are NOT stop conditions.

### Loop forever via `/loop`

Use the `/loop` skill (or `ScheduleWakeup` in dynamic mode) to keep
firing the same continuation prompt indefinitely. After every commit,
schedule the next wake-up. Pattern:

```
Skill: loop
args:  continue jxl-encoder-gpu port work
```

— or directly via `ScheduleWakeup(delaySeconds=60, prompt="continue
jxl-encoder-gpu port work", reason="next kernel")`.

Each firing's job:
1. Re-read this `CLAUDE.md` (in case it changed)
2. Run the smoke test (verify nothing regressed since last firing)
3. Find the next undone item in `PORT_STATUS.md`
4. Port it. Parity-verify it. Commit it. Update `PORT_STATUS.md`.
5. **Schedule the next firing before this one ends.**

Step 5 is mandatory. If you finish a turn without scheduling the next
wake-up, the loop dies and the project stalls. Always re-arm.

### After context compaction — re-read everything

Compaction is a transparent runtime mechanism, NOT a stop signal. The
post-compaction Claude has a lossy summary of the prior history. Do
NOT rely on the summary. Treat post-compaction wake-ups as a fresh
session that knows nothing.

**Mandatory re-read list at the start of every post-compact firing:**

1. **The full message history** of this conversation, scrolled back to
   the original user request. Use whatever browsing/scrollback the
   runtime provides. Look for: the original "create jxl-encoder-gpu
   and complete 100%" mandate, all subsequent user clarifications,
   and any in-flight discussion that the summary may have collapsed.

2. **Every load-bearing markdown file:**
   - `CLAUDE.md` (this file — the rules)
   - `CONTEXT-HANDOFF.md` (last contract)
   - `PORT_STATUS.md` (live deliverable grid)
   - `CHANGELOG.md` (history of what landed)
   - `~/work/zen/zenmetrics/docs/CUBECL_PORTING_GUIDE.md`
   - `~/work/zen/zenmetrics/docs/CUBECL_GOTCHAS.md`
   - `~/work/zen/jxl-encoder/CLAUDE.md` (algorithm-level decisions)
   - `~/.claude/CLAUDE.md` (global rules; project rules layer over this)

3. **Every `.rs` source file in this crate** (so the gotchas already
   worked around in code carry forward):
   - `src/lib.rs`, `src/kernels/*.rs`, `src/launch/*.rs`
   - All `examples/*_parity.rs` (parity invariants and tolerances)

4. **The CPU reference for the next undone kernel.** Read the matching
   `_scalar` function in `~/work/zen/jxl-encoder/jxl-encoder-simd/src/`
   in full before writing the GPU port.

5. **Run the smoke test** — every existing parity example must print ✓
   on `cargo run --release --features cuda --example <X>`. If anything
   regressed, that's the first thing to fix before adding new kernels.

Don't trust your "memory" from before compaction. Re-read the source of
truth. Files are cheap to read; bugs from stale assumptions are not.

### "Done" — the only stop condition

The loop terminates ONLY when ALL of the following hold simultaneously:

1. All 37 deliverables in `PORT_STATUS.md` are `✓` (kernel landed,
   parity example passes its documented tolerance, commit referenced).
2. `jxl-encoder` builds with `--features gpu` and produces lossy output
   that matches CPU-encoded reference within documented tolerance on a
   real photo from `~/work/codec-corpus/`.
3. A jxl-rs roundtrip test passes on the GPU-encoded output of a real
   photo.
4. CI is set up and green on `windows-11-arm`, `macos-15-intel`, and
   `i686-unknown-linux-gnu` (per global CLAUDE.md mandate). If no
   GitHub remote exists yet, create the repo and push first.

Until all four hold, **keep the loop firing**. There are no
intermediate stop conditions.

When all four hold: stop the loop, write a final completion message to
the user with the verified evidence (commit hashes, CI URLs, parity
numbers), and idle.

### Hard rules (still apply, unchanged)

1. **No incomplete work in the tree.** Every kernel either lands fully
   (parity-verified, committed, ✓ in `PORT_STATUS.md`) or is
   **absent**. If a firing can't finish a kernel before its turn ends,
   `jj abandon` the change before scheduling the next wake-up.
   Half-written `#[cube]` functions in the tree poison the next firing.

2. **No incorrect work claimed correct.** Parity gate non-negotiable.
   `✓` requires a runnable example that exits 0 with documented
   tolerance. Tolerance loosening only with documented physical
   reason (FMA contraction, f64-vs-f32, etc.) — never to mask a bug.

3. **No false completion claims.** Per global `~/.claude/CLAUDE.md`:
   fractional reporting always (`X of 37`), list MISSING before
   PRESENT. "Done" means all four conditions above; nothing else
   qualifies.

## Workflow rules (project-specific overlay; see ~/.claude/CLAUDE.md for global)

- jj on top of colocated git, work directly on `main` (no feature branches).
- `.workongoing` marker required before any work — see global CLAUDE.md.
- Every kernel lands as one commit: `feat(kernel): <name> + parity test (commit hash)`.
- Pre-commit: `cargo fmt && cargo clippy --all-features -- -D warnings && cargo test --features cuda`.
- Cold compile is 5-9 min on first build of any cubecl-cuda crate; plan iteration
  in 1-2h batches and reuse the cached build (G6.1 in zenmetrics CUBECL_GOTCHAS.md).

## Reference docs (READ FIRST)

- `~/work/zen/zenmetrics/docs/CUBECL_PORTING_GUIDE.md` — 559-line per-pattern guide.
- `~/work/zen/zenmetrics/docs/CUBECL_GOTCHAS.md` — 30-entry trap catalogue.
- `~/work/zen/zenmetrics/crates/butteraugli-gpu/` — worked example.
- `~/work/zen/jxl-encoder/jxl-encoder-simd/src/` — the CPU reference being mirrored.
- `~/work/zen/jxl-encoder/CLAUDE.md` — algorithm-level decisions for the encoder.

## Validation discipline (mandatory)

Three layers, in order:
1. **Per-kernel parity** — `examples/<name>_parity.rs` runs the GPU kernel and
   the corresponding `jxl_encoder_simd::*_scalar` function on the same input.
   Targets: max_abs_diff < 1e-6 for unit-range, < 1e-5 for opsin-scale.
2. **Pipeline parity** — once N kernels are wired, `examples/parity_real_image.rs`
   runs the full GPU pipeline against a real PNG; Δ ≤ 0.1%.
3. **Bit-exact lock test** — hash the output, freeze in a unit test under `tests/`.

NEVER hand-roll the CPU reference. ALWAYS call `jxl_encoder_simd::*_scalar` directly.

## Active gotchas to internalise (from zenmetrics)

1. `f32::exp` not registered → use `f32::powf(2.0, x * LOG2_E)`.
2. `Atomic<f32>::fetch_max` doesn't lower on CUDA → use `Atomic<u32>` on bit-reinterpreted f32.
3. `0.0` literal in if/else with cube f32 → use `f32::new(0.0)`.
4. `u32::abs_diff` not registered → `saturating_sub` both ways and add.
5. `SharedMemory::new(N)` is `usize`; index by `usize` (cast).
6. `f32::log` is base-2, not natural — use `f32::ln` for natural log.
7. Comptime generics on `bool` not supported — split into separate kernels.
8. `CubeCount` and `CubeDim` are not `Copy` — `.clone()` per launch.

## DCT scale convention (CRITICAL — different from libjxl CPU)

The GPU DCT kernels in this codebase use **mean-scale** convention, NOT the
orthonormal `sum/sqrt(N²)` convention from libjxl's CPU code:

- `forward_dct8(uniform_X)[0]   = X` (= mean), not `8*X` (= sum/8)
- `forward_dct16x16(uniform_X)[0] = X`, not `16*X`
- `dc_from_dct_16x16(forward_dct16x16(uniform_X)) = [X, X, X, X]`

So the DC frame stores **mean values** (`sum/64` per 8x8 block), NOT `sum/8`.
Verified by `test_dct_scale_convention_diag` in lossy_encoder.rs.

This bit Phase A hard (commit `ac88dc1c`): `compute_dc_grid_per_8x8_block`
returned `sum/8`, making `restore_llf_dct16x16` produce LLF[0] 8× too large
and reconstruction blow up to ~165 butteraugli (vs target ~1.3). Fix was
`out[i] = sum / 64` and a regression test `test_lossy_encoder_strat_search_vs_encode_one_diag`.

**When porting CPU code to GPU, double-check the normalization convention
before trusting any constant from libjxl reconstruct.cc / dct_scales.h.**

## Combining strat-search with butteraugli AQ refinement (May 9, 2026)

**Infrastructure landed.** `encode_one_with_strategy_search_dct8_16_adaptive`
takes `&[f32]` per-block qac instead of scalar distance. The cost-grid
stage still scales with `target_distance` (so strategy assignments
stay stable when only `aq_field` varies — required for the loop to
converge). Only the encode/recon stage consumes `aq_field`.
`refine_aq_field_gpu_with_strategy_search` is the matching loop entry.

**Empirical results** (CLIC 1024×1024 @ 4-iter refinement,
`combined_strat_search_aq_demo`):

| d   | uniform | strat alone | refine+DCT8 | refine+strat | combined Δ vs DCT8 |
|-----|---------|-------------|-------------|--------------|---------------------|
| 0.5 | 0.7006  | 0.7006      | 0.6641      | 0.6641       | -0.0000 (-0.00%)   |
| 1.0 | 1.3456  | 1.3456      | 1.1475      | 1.1475       | +0.0000 (-0.00%)   |
| 1.5 | 1.6745  | 1.7491*     | 1.6299      | **2.0604**   | +0.4235 (+25.9%)   |
| 2.0 | 2.1525  | 2.2506*     | 2.1526      | 2.1566       | +0.0040 (+0.19%)   |
| 3.0 | —       | —           | 2.9187      | 3.1508       | +0.2321 (+7.95%)   |

*at parity at d≤1, regresses at d≥1.5 per the strat-search distance-
limitation entry below.

**Combined mode is at parity through d≤1.0**, slightly better on
pnorm_3 (0.4696 vs 0.4703 at d=1.0). At d≥1.5 it INHERITS strat-
search's cost-model regression, producing strictly worse results
than refine+DCT8 alone. **Combined mode is NOT a quality win until
the strat-search distance-scaling cost model is fixed** (kAvoidEntropyOfTransforms
and X-channel multi-block weight ports). The plumbing is correct;
the limitation is upstream.

**Cost: 4.6× more wall-clock per iter** than refine+DCT8 (202ms vs
44ms/iter on CLIC 1024² at d=1). Each iter re-runs cost grids +
selector + mixed encode-recon. **Future optimization**: cache the
cost-grid output before entering the refinement loop — assignments
are invariant when only `aq_field` varies, so cost-grid recompute is
wasted work. Could drop combined-mode iter cost to ~50ms (one DCT/quant
batch + recon). At that point combined mode becomes free relative to
refine+DCT8 even when quality wins are absent.

**Conclusion**: the integration is shipped and correct; it's gated
behind explicit caller choice (no auto-promotion in turnkey APIs)
until the cost-model regression at d≥1.5 is fixed.

## Strat-search corpus quality (May 9, 2026, post DCT32/DCT64 retune)

The previous "Strat-search distance limitation" entry below claimed
parity at d≤1 — that was WRONG. It was based on cherry-picked single-
image testing. A 16-image CLIC sweep at d=1.0 (combined_strat_search_aq_demo
+ histogram print) revealed strat-search was producing CATASTROPHIC
quality regressions on >50% of CLIC images:

  Pre-fix (DCT64 mul=3.5, DCT32 mul=2.5):
    16 CLIC images @ d=1.0:
    - 0 wins, 7 parity, 9 LOSSES (3 catastrophic +50% to +94%)
    - Pattern: cost grids over-pick large transforms on detailed content

The root cause: cost model lacks libjxl's pixel-loss penalty for
large transforms. DCT64x64's "few-coefs" entropy advantage wins even
when reconstruction is catastrophically blurry on detailed content.

Quick fix (commit 73dab065 May 9 2026):
- DCT64x64 entropy_mul: 3.5 → 8.0
- DCT64x32 / DCT32x64 entropy_mul: 3.5 → 8.0
- DCT32x32 entropy_mul: 2.5 → 4.0

Post-fix:
  16 CLIC images @ d=1.0:
  - strat-search alone: ALL within ±0.1% of uniform (parity)
  - refine+strat (combined mode) vs refine+DCT8:
    * 3 WINS (-0.4% to -2.4%): 07b9f93f, 22ea12c9, 2684452d
    * 11 parity
    * 2 slight losses (+0.8%, +3.0%): 0c49a5cc, 11f2b039

Combined mode is now a net quality win on diverse content.
The proper long-term fix is porting libjxl's missing pixel-loss term
for large transforms (kAvoidEntropyOfTransforms is the d>4 piece, but
the d≤4 piece is via the per-strategy entropy_mul calibration). The
band-aid muls work but suppress correct DCT64 picks on truly smooth
content — re-tune to libjxl reference values once the missing
counterweights land.

## Strat-search distance limitation (May 8, 2026 — DEPRECATED)

LossyEncoder::encode_one_with_strategy_search_dct8_16 produces butteraugli
at uniform-qac parity for d≤1, but degrades at higher distances:

| d   | uniform | strat-search | Δ      |
|-----|---------|--------------|--------|
| 0.5 | 0.7006  | 0.7006       | 0%     |
| 1.0 | 1.3456  | 1.3456       | 0%     |
| 2.0 | 2.1525  | 2.2506       | +4.6%  |
| 4.0 | 3.4407  | 9.1984       | +167%  |

Investigation (commits this session, especially the cost-model tuning
sequence in f7cdaa56 / b482813b / c4c62cef): the fixed 2× anti-bias
muls (DCT32x32=2.5, DCT64=3.5, sub-blocks=2.16/1.72/2.09/1.90) were
calibrated at d=1.0. At higher distances:
- More aggressive quantization → more zeros in larger transforms →
  larger transforms become artificially cheap in the cost model
- Doubling the muls (5.0/7.0) had **no measurable effect** on the
  d=4 regression — picks are stable across mul changes
- Disabling sub-blocks at d=4 also didn't help — score stayed 9.1984

Root cause appears structural: our cost grid formula has a different
distance-scaling profile than libjxl's full cost model (kAvoidEntropyOfTransforms,
X-channel multi-block weight, AdjustQuantBlockAC). Fixed muls work at
the calibration point but the curves diverge at d≥2.

Fix paths (deferred):
1. **Distance-scaled muls**: `entropy_mul = base + scale * distance`
   per strategy. Empirical fit needed.
2. **Implement the missing libjxl counterweights** (X-channel weight,
   kAvoidEntropyOfTransforms penalty). Then drop the 2× anti-bias and
   use libjxl reference muls directly.

For now, strat-search is best used at d≤1 where it matches uniform-qac
exactly. The infrastructure (15/27 strategies, persistent GPU pipeline)
generalizes; only the cost-model calibration is distance-fragile.

## Phase plan

| Phase | Scope | Effort | Status |
|---|---|---|---|
| 0 | Wire zenmetrics into jxl-encoder's butteraugli/ssim2/zensim refinement loops | 1 day | not started |
| 1 | Pointwise + separable kernels (xyb, gaborish, mask1x1, noise) | 3-5 days | in progress |
| 2 | Per-block kernels (DCT/IDCT/quantize/dequant/block_l2/pixel_loss/adaptive_quant/cfl) | 2 weeks | not started |
| 3 | Whole-image-per-strategy AC search (replaces per-region trial) | 2-3 weeks | not started |
| 4 | jxl-encoder integration behind `gpu` feature | 1 week | not started |

See `PORT_STATUS.md` for the per-kernel grid.

## Architectural decision: whole-image-per-strategy AC search

Per-region trial encoding produces severe GPU control-flow divergence (each region
in a warp picks a different candidate strategy). Replace with: for each candidate
strategy S, one fully-coherent whole-image kernel computes per-tile costs at S's
natural alignment. Then a host-side (or trivial GPU) pass enumerates legal
partitions per 32×32/64×64 region and picks the min-cost partition.

This is algebraically equivalent to libjxl's algorithm (per-region cost decouples
given a fixed quant field — `AdjustQuantBlockAC` runs after selection, so trial
cost has no neighbour coupling). It also makes pruning gates (DCT64x64 only at
d≥3, etc.) free to drop, since whole-image trial is cheap on GPU.

## Don't fake completion

- State coverage as a fraction in PORT_STATUS.md (e.g. "5 of 23 kernels with parity").
- List MISSING before PRESENT in any status report.
- Tests passing ≠ feature working — run `parity_real_image.rs` end-to-end.
- See ~/.claude/CLAUDE.md "NEVER CLAIM FALSE COMPLETION" for the full discipline.
