# jxl-encoder-gpu — Claude Code instructions

## What this crate is

Multi-vendor GPU kernels for `jxl-encoder` via [CubeCL](https://github.com/tracel-ai/cubecl).
Mirrors `jxl-encoder-simd` 1:1 — every public function in `jxl-encoder-simd`
(scalar reference path) is the parity target for one CubeCL kernel here.

We dispatch across CUDA (NVIDIA), WGPU (Vulkan/Metal/DX12/WebGPU), HIP (AMD),
and CPU (cubecl-cpu) from a single `#[cube]`-annotated kernel source.

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
