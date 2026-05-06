# jxl-encoder-gpu / imazen/jxl-gpu — context handoff

**Last updated:** 2026-05-06 (session 2 end — autonomous plateau)
**Repo:** https://github.com/imazen/jxl-gpu (live, public)
**Local:** ~/work/zen/jxl-encoder-gpu/ (37 commits on main, in sync with origin)
**Hardware verified:** RTX 5070, CUDA 13.2, jj 0.40, rustc 1.95

## Bottom line

This session built a substantial GPU JXL encoder kernel library + facade:
- **45 of ~52 kernel deliverables** verified with parity tests (~87%)
- **Phase 3 components 1+2** prototyped end-to-end (cost-grid composition + 4-strategy partition selector across 3 region tiers, 12 unit tests)
- **42-method `GpuEncoder<R>` facade** covering the entire DCT/IDCT family + all supporting kernels
- **Equivalence validated** at 256×256 random scale: GPU XYB matches jxl-encoder-simd CPU XYB at 1.79e-7 abs (sub-ulp)
- **End-to-end pipeline demo** chains 8 GpuEncoder methods on real image-shaped data
- **CI workflow** ready (multi-platform build matrix, fmt + clippy gates)
- **Public GitHub repo** at imazen/jxl-gpu

Nothing in the repo is half-finished. Every kernel ships with a parity test. Every GpuEncoder method is callable today. The Phase 3 prototype runs end-to-end with sensible decisions on synthetic input.

## Why the loop is paused here

The remaining work falls into three categories:

### 1. Mechanical incremental wrappers (~5 small methods)
- `entropy_coeffs_coeff_blocks` (coefficient-domain entropy mode)
- `quantize_large_blocks` (DCT16+ quantize wrapper)
- A few rectangular cost-grid compositions
Each is ~30 lines following the established pattern. Low marginal value.

### 2. Substantial architectural extensions
- **3-channel + CfL cost grids** in `pipeline.rs` — extends `compute_cost_grid_*_single_channel` to handle X/Y/B with CfL decorrelation. ~200 lines + parity test against full cost.
- **Multi-channel partition selector wiring** — feed the 3-channel cost grids into the existing partition selector with proper aggregation.
- **Performance benchmark** — measure GPU vs CPU throughput for each pipeline stage.

### 3. Actual jxl-encoder integration (the genuine Phase 4 endpoint)
The progressive-replacement pattern needs human design judgment about:
- **Where to fork** vs. add a hook to upstream jxl-encoder. Per-call-site forking risks divergence; upstream hooks need maintainership coordination.
- **What pipeline-stage granularity** to substitute (whole encoder, per-frame, per-tile, per-stage). The sweet spot is probably per-stage with downloaded intermediates, but may be wasteful vs. keeping data on GPU across many stages.
- **Fallback behavior** when GPU isn't available — silently CPU? feature-gated? error?

The best XYB substitution target is `jxl-encoder/src/vardct/xyb.rs::convert_strip` (line 387). Replace the per-row `jxl_simd::linear_rgb_to_xyb_batch` call with a batched GpuEncoder call. ~50-line change but needs care around the strip parallelism (rayon par_chunks_mut competes with GPU async if every strip launches its own kernel).

## What's done in this session (cumulative)

### Phase 1 — pointwise + separable kernels (6 of 7)
- xyb forward + inverse, gab, gaborish_5x5, mask1x1, pad_plane (parity-verified)
- Missing: denoise (no `_scalar` exported in upstream)

### Phase 2 — per-block kernels (39 of ~40)
- **DCT/IDCT family — COMPLETE (26)**: 8/16/32/64 squares + rectangulars + DCT4 sub-block variants
- **Other**: quantize_dct8, quantize_large, dequant_dct8, block_l2, pixel_loss (f64), cfl LS + Newton, epf_step1 + step2, entropy_coeffs (pixel + coeff), per_block_modulations, compute_pre_erosion
- ⏸ Deferred: fused_dct8_entropy (no GPU fusion benefit when cube_dim=1)

### Phase 3 — whole-image AC strategy search
- C1 cost grid composer: DCT8/16/32/64 single-channel strategies + DCT16x16 with proper aggregation
- C2 host-side partition selector: 4 strategies × 3 region tiers (16/32/64), 12 unit tests
- End-to-end demo: smooth half → DCT32x32, noisy half → 4×Sub16x16+FourDct8x8 (algorithmically correct)

### Phase 4 — encoder facade (`encoder.rs`)
42 GPU-backed methods on `GpuEncoder<R: Runtime>`:
- xyb_from_linear_rgb, mask1x1_field, pad_plane_channel
- gaborish_5x5_channel, gab_smooth_channel
- 7 DCT pairs (8x8, 16x16, 32x32, 64x64, 16x8, 8x16, 32x16, 16x32, 64x32, 32x64, 4x4_full, 4x8_full, 8x4_full) + matching IDCTs (= 26 methods)
- quantize_dct8_blocks, dequant_dct8_blocks
- entropy_coeffs_pixel_blocks
- block_l2_errors, pixel_loss_blocks
- epf_step1_channels, epf_step2_channels
- pre_erosion, apply_per_block_modulations
- cfl_multipliers (LS), cfl_multipliers_newton
- encode_lossy_via_cpu (full delegation to jxl-encoder)

### Documentation + infra
- README.md with status, quick-start, backend matrix, architecture rationale
- CLAUDE.md with autonomous-mandate section
- PORT_STATUS.md per-kernel grid + Phase 3 / Phase 4 status
- CHANGELOG.md keep-a-changelog format
- .github/workflows/ci.yml multi-platform build matrix (cuda commented out, needs self-hosted runner)
- LICENSE-AGPL3 + LICENSE-COMMERCIAL

### Verification
- 12 partition selector unit tests pass
- 18 parity examples pass on RTX 5070 + CUDA 13.2
- `cargo clippy --release --features cuda --lib --tests -- -D warnings` clean
- GPU XYB ≡ CPU XYB at 256×256 random (1.79e-7 abs, sub-ulp)
- End-to-end 8-method pipeline demo runs cleanly on 32x32 input

## Smoke test (next session, first thing)

```sh
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd ~/work/zen/jxl-encoder-gpu
cargo build --release --features cuda --examples 2>&1 | tail -3
for ex in xyb_parity phase1_parity dct8_parity phase2_small_parity \
          pixel_loss_parity dequant_parity dct16_parity dct16_rect_parity \
          dct4_parity dct32_parity dct64_parity quantize_cfl_parity \
          epf_parity entropy_parity per_block_modulations_parity \
          pre_erosion_parity cost_grid_demo phase3_integration_demo \
          encoder_smoketest gpu_encoder_pipeline_demo; do
    echo "=== $ex ==="
    ./target/release/examples/$ex 2>&1 | tail -1
done
cargo test --release --no-default-features --features cpu --tests 2>&1 | tail -5
```

All 20 examples should print ✓ and 12 partition selector tests should pass.

## Resume options

### A. Mechanical incremental wrappers
Continue adding the ~5 remaining method bindings. ~30 minutes per kernel. Tightens API completeness but doesn't unlock new use cases.

### B. Multi-channel + CfL cost grid extension
Most architecturally substantial unblocked work. ~200 lines + parity test. Genuine Phase 3 advancement that makes the cost-grid composer usable for real strategy decisions.

### C. Begin actual jxl-encoder integration
Pick a small XYB or DCT call site in jxl-encoder, fork the file into our crate, modify to use GpuEncoder, verify byte-equivalent output on a real photo. The genuine Phase 4 endpoint.

### D. Performance benchmarking
Measure GPU vs CPU throughput for each pipeline stage. Quantify the speedup (or expose where the GPU is slower than CPU at small sizes due to launch overhead).

### E. Stop and accept current state
The repo is genuinely useful as a kernel library + Phase 3 architecture proof. Anyone wanting to build a GPU JXL encoder has the primitives.

## Cubecl 0.10 gotchas catalog (13 unique)

Internalized during this port; documented in commit history of dct16.rs, dct32.rs, adaptive_quant.rs, cfl.rs, mask1x1.rs:

1. `f32::exp` not registered → use `f32::powf(2.0, x * LOG2_E)`
2. `Atomic<f32>::fetch_max` doesn't lower on CUDA → bit-cast to `Atomic<u32>`
3. `0.0` literal in if/else with cube f32 → use `f32::new(0.0)`
4. `u32::abs_diff` not registered → use `saturating_sub` both ways + add
5. `SharedMemory::new(N)` is `usize`; index by `usize` (cast)
6. `f32::log` is base-2, not natural — use `f32::ln` for ln
7. Comptime generics on `bool` not supported — split into separate kernels
8. `CubeCount` and `CubeDim` not `Copy` — `.clone()` per launch
9. `u32::wrapping_sub` not registered — use plain `i32` raw subtract
10. `return` forbidden in `#[cube]` bodies — single-expression form
11. Strict u32/usize index typing — cast scalars to usize at function entry
12. Function-call result into `let` may type-mismatch — inline call into if/else arm
13. cubecl-cuda supports f64 (no special config) — used by pixel_loss
14. No stack arrays inside `#[cube]` — use SharedMemory as recursive arg buffer
15. No round-ties-even cast — implemented manually for AdjustQuantBlockAC parity
16. Scalars pass directly to `launch_unchecked` (NOT via `ScalarArg::new`)
17. Cross-module `#[cube]` imports work — `pub(crate)` helpers in dct16.rs callable from dct32.rs and dct64.rs
