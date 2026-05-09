# jxl-encoder-gpu — Claude Code instructions

## What this crate is

Multi-vendor GPU kernels for `jxl-encoder` via [CubeCL](https://github.com/tracel-ai/cubecl).
Mirrors `jxl-encoder-simd` 1:1 — every public function in `jxl-encoder-simd`
(scalar reference path) is the parity target for one CubeCL kernel here.

We dispatch across CUDA (NVIDIA), WGPU (Vulkan/Metal/DX12/WebGPU), HIP (AMD),
and CPU (cubecl-cpu) from a single `#[cube]`-annotated kernel source.

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

## Combining strat-search with butteraugli AQ refinement (open)

`LossyEncoder::encode_one_with_strategy_search_dct8_16` (strat-search)
and `forks::butteraugli_loop::refine_aq_field_gpu` (per-iter qac
refinement) are independently working. On CLIC 1024×1024 @ d=1.0 with
the current implementations:

| Pipeline | Score (butteraugli) |
|----------|---------------------|
| uniform-qac (encode_one) | 1.3456 |
| strat-search (uniform qac) | 1.3456 |
| butteraugli refine on uniform-qac (4 iters) | **1.1475** |

The refinement loop wins (-15% vs strat-search) because it tunes
per-block qac based on perceptual feedback, while strat-search picks
transforms but keeps qac uniform. Combining them — strat-search picks
transforms AND butteraugli refinement tunes per-block qac — would
likely produce the best quality.

The integration: replace the per-iter `encode_one_adaptive` call
inside `refine_aq_field_gpu` with a strat-search call that takes a
per-block qac field. Currently `encode_one_with_strategy_search_dct8_16`
takes a scalar `distance` (broadcasts to qac); a new
`encode_one_with_strategy_search_dct8_16_adaptive` would take
`&[f32]` per-block qac. This is the natural next step for max quality.

## Strat-search distance limitation (May 8, 2026)

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
