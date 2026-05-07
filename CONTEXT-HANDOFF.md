# jxl-encoder-gpu / imazen/jxl-gpu — context handoff

**Last updated:** 2026-05-07 (session 4 — LossyEncoder polish)
**Repo:** https://github.com/imazen/jxl-gpu (live, public)
**Local:** ~/work/zen/jxl-encoder-gpu/ (111 commits on main, in sync with origin)
**Hardware verified:** RTX 5070, CUDA 13.2, jj 0.40, rustc 1.95

## Bottom line — GPU now BEATS CPU AVX2 at every measured size

End-to-end CPU vs GPU lossy DCT8 pipeline (RTX 5070 + Ryzen 9 7950X):

```
side    CPU ms    GPU ms    ratio        throughput
 256     1.61     1.49     1.08× GPU    44 MP/s
 512     9.39     4.28     2.19× GPU    61 MP/s
1024    39.46    17.66     2.23× GPU    59 MP/s
2048   150.55    41.33     3.64× GPU   101 MP/s vs 28 MP/s
4096   604.21   506.34     1.19× GPU    33 MP/s
```

Parity 4e-6 max abs delta. **The breakthrough was switching
`client.create_from_slice(&vec![0.0; n])` → `client.empty(n*4)` in
the persistent API** (commit `d6ef26c7`) — eliminated a ~200MB host-
to-GPU memcpy of zeros per pipeline run that the kernel was about to
overwrite anyway. Pipeline went from 375ms → 48ms at 2048² (7.8×).

Without that fix, GPU was 1.7-3.3× SLOWER than CPU. With it, the
GPU is 1.05-3.95× FASTER. Same code, one allocation API switch.

## What's been built

### High-level `LossyEncoder` API (`crate::lossy_encoder`)
- `LossyEncoder::new(enc, w, h)` — accepts arbitrary sizes (pads
  internally with right+bottom edge replication)
- `encode_one(rgb_f32, qac)` / `encode_many(rgb_f32, &qacs)`
- `encode_one_srgb_u8(rgb_u8, qac)` / `encode_many_srgb_u8(rgb_u8, &qacs)`
- `quality_to_qac(quality_1_to_100)` — JPEG-style quality knob

This is the user-facing entry point. Construct one per `(width, height)`
to amortize static-input upload; pass RGB U8 (interleaved) or planar
f32. Batch variants amortize input upload across N encodes (5× speedup
per `lossy_pipeline_repeated_input`).

### Persistent GPU buffer API (`crate::persistent`)
- `GpuPlane<R>`, `GpuBlocks<R>`, `GpuI32Blocks<R>` typed handles
- ~30 persistent methods covering full encoder front+back
- 3 new GPU kernels: `gather_blocks`, `scatter_blocks`, `restore_dc`
- Cooperative DCT8 (cube_dim=8) + Wide DCT8/IDCT8 (cube_dim=64)
- Fused DCT+quant + dequant+IDCT-Y (2-3× per-kernel)

This is the lower-level building blocks. Use directly when you need
fine-grained control over which kernels run, or to chain custom
pipelines that LossyEncoder doesn't cover.

### 11 fork modules of `jxl-encoder` pipeline stages (`crate::forks`)
xyb, gaborish, adaptive_quant, reconstruct, transform (13 DCT
strategies), cfl, epf, dequant, quantize, cost, pad

### 14+ example demos
- API: `lossy_encoder_demo`, `lossy_encoder_real_image`
- Composition: `forks_pipeline_demo`, `lossy_roundtrip_demo`,
  `lossy_roundtrip_persistent`
- Real-image: `real_image_encode` (djxl-verified),
  `jxl_rs_roundtrip` (jxl-rs-verified)
- Throughput: `xyb_throughput_bench`, `xyb_scaling_bench`,
  `persistent_buffer_pipeline`, `lossy_pipeline_throughput`,
  `lossy_pipeline_fused_throughput`, `lossy_pipeline_breakdown`,
  `lossy_pipeline_no_io`, `lossy_pipeline_repeated_input`,
  `dct8_coop_bench`, `fused_dct_quant_bench`,
  `fused_dequant_idct_bench`

### Test coverage
- 59 unit tests passing (cuda)
- 12 partition selector tests (cpu)
- Real 1024×1024 CLIC photo encode + djxl + jxl-rs roundtrip
- 1017×1013 cropped (non-aligned) photo via LossyEncoder
- Full pipeline parity 4e-6 max abs delta CPU vs GPU

## Next perf frontier (in priority order)

1. **Buffer reuse for batch workloads.** `lossy_pipeline_no_io`
   shows headroom: at 4096² no-IO is 19ms vs 592ms with-IO (30×
   faster, the entire delta is host↔GPU memcpy). For video encoders
   or batch image processors that re-use input buffers, a
   `LossyContext<R>` API that pre-allocates per-shape buffers and
   reuses them across encode calls would push GPU to 10-60× CPU.

2. **Pinned host memory for upload/download.** cubecl currently uses
   pageable host memory which caps PCIe transfers at ~3-5 GB/s.
   Pinned memory hits ~25 GB/s. Requires cubecl-internal change.

3. **Larger fusions.** XYB+DCT+quantize as one kernel would skip
   the intermediate XYB plane entirely. Requires re-architecting
   the persistent API to handle planar→per-block transitions
   inside a kernel.

4. **GPU dc_coding.** Currently DC restore is a roundtrip-demo
   shortcut. Real encoders need a separate GPU DC quant + entropy
   path that integrates with the fused DCT+quant kernel (which
   currently zeros DC).

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
