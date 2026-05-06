# jxl-encoder-gpu — context handoff

**Last updated:** 2026-05-06 (session 2)
**Repo head:** main, 6 jj changes
**Hardware verified:** RTX 5070, CUDA 13.2, jj 0.40, rustc 1.95
**Coverage when this was written:** **13 of 37 deliverables verified (35%)**

## What's done

Phase 1 — pointwise + separable kernels (5 of 7):

| Kernel | Tolerance achieved (vs `jxl_encoder_simd::*_scalar`) |
|---|---|
| `kernels::xyb::xyb_forward_kernel` | forward 1.19e-7, inverse 1.30e-6 abs |
| `kernels::xyb::xyb_inverse_kernel` | (same) |
| `kernels::gab::gab_smooth_kernel` (3x3) | 5.96e-8 abs |
| `kernels::gaborish::gaborish_5x5_kernel` | 1.19e-7 abs |
| `kernels::mask1x1::mask1x1_kernel` | 6.79e-4 abs (FMA-noise; doc'd 1.5e-3 tol) |

Phase 2 — per-block kernels (8 of 23):

| Kernel | Tolerance |
|---|---|
| `kernels::dct8::dct_8x8_kernel` | 2.98e-8 abs (256 blocks) |
| `kernels::dct8::idct_8x8_kernel` | 1.79e-7 abs; roundtrip 2.24e-7 |
| `kernels::block_l2::block_l2_kernel` | 1.40e-9 abs (16x12 blocks) |
| `kernels::quantize::quantize_dct8_kernel` | bit-exact (0 diffs, 64 blocks) |
| `kernels::pixel_loss::pixel_loss_kernel` | 3.75e-16 rel (f64, 256 blocks) |
| `kernels::dequant::dequant_dct8_kernel` | Y bit-exact; X/B 1.53e-5 (FMA noise) |
| `kernels::dct16::dct_16x16_kernel` | 5.96e-8 abs (64 blocks) |
| `kernels::dct16::idct_16x16_kernel` | 5.36e-7 abs; roundtrip 4.47e-7 |

Strategy used so far: **one cube per block, single-threaded** (`cube_dim = 1`,
`cube_count = num_blocks`). Per-block scratch in `SharedMemory::<f32>::new(N)`.
Cooperative intra-block parallelism is a future optimization.

## What's NOT done — 24 of 37 missing

**Phase 1 holdovers (2 kernels):**
- `kernels::denoise` — `noise.rs` has no `_scalar` exported.
- `kernels::pad_plane` — trivial; ~30 LOC.

**Phase 2 (15 kernels):**
- DCT16 rectangular: `dct_8x16` / `dct_16x8` + matching IDCT (4 kernels).
  Uses `fwd_dct1d_8` and `fwd_dct1d_16` already in `dct16.rs`. Estimated
  ~1 hour each pair.
- DCT32 (16x32, 32x16, 32x32) + matching IDCT — same shape, 32-pt
  butterflies recursively wrap dct1d_16. Each ~2 hours.
- DCT64 (32x64, 64x32, 64x64) + matching IDCT — wrap dct1d_32. Each ~2-3 hours.
- DCT4 (4x4, 4x8, 8x4) + matching IDCT — smaller, ~1 hour each pair.
- `quantize_large` (DCT16+ blocks, 128-4096 coeffs).
- `compute_pre_erosion`, `per_block_modulations` (adaptive_quant).
- `cfl_find_best_multiplier`, `cfl_find_best_multiplier_newton`.
- `epf_step1`, `epf_step2`.
- `fused_dct8_entropy`.
- `entropy::estimate_entropy_full`.

**Phase 3** — whole-image-per-strategy AC search (3 components):
- Per-strategy whole-image cost grid kernels
- Host-side partition selector
- Refactor `ac_strategy_search.rs` to consume cost grids

**Phase 4** — `jxl-encoder` integration (4 components):
- `gpu` cargo feature, pipeline swap-in, instance cache, jxl-rs roundtrip

## Environment quirks (unchanged from session 1)

CUDA 13.2 at `/usr/local/cuda-13.2/`, symlinked to `/usr/local/cuda`.
Required env every session:
```sh
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```

## Gotchas accumulated

From session 1:
- G1.4-bis `u32::wrapping_sub` not registered → use `i32` raw subtract.
- G1.8 `return` forbidden in `#[cube]` bodies → single-expression form.
- G1.9 strict u32/usize index typing → cast width/height to usize at function entry.
- G2.5 scalars pass directly (no `ScalarArg::new`).
- G6.4 FMA contraction divergence vs CPU non-FMA scalars.

New in session 2:
- **G1.10 — cube can't return into a `let`-bound `i32` from a function call
  in some contexts.** Workaround: inline the function call into the if-arm
  rather than `let rounded = func(); if .. { 0 } else { rounded }`. Concrete
  example: `quantize_dct8_kernel` uses `if absv < thr { i32::new(0) } else
  { round_ties_even_to_i32(val) }` directly.
- **G1.11 — cubecl-cuda supports f64.** No special configuration needed.
  Used by `pixel_loss_kernel` for the 8th-power chain. Sub-ulp f64 parity
  achieved (3.75e-16 relative).
- **G1.12 — no stack arrays inside `#[cube]`.** Use
  `SharedMemory::<f32>::new(N)` as the recursive-arg buffer for nested
  butterflies. With `cube_dim=1` it codegens to register/local memory.
  See `inv_idct1d_16` in `kernels/dct16.rs` for the pattern (passes 8-element
  scratches into `inv_idct1d_8_core`).
- **G1.13 — round-ties-even cast not exposed.** Implemented manually in
  `kernels/quantize.rs::round_ties_even_to_i32` because Rust's `f32::round`
  is round-half-away-from-zero and would diverge from libjxl's `rintf`-based
  CPU reference (matters for AdjustQuantBlockAC).

## Smoke test — first thing to run next session

```sh
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd ~/work/zen/jxl-encoder-gpu
cargo build --release --features cuda --examples 2>&1 | tail -3
for ex in xyb_parity phase1_parity dct8_parity phase2_small_parity \
          pixel_loss_parity dequant_parity dct16_parity; do
    echo "=== $ex ==="; ./target/release/examples/$ex 2>&1 | tail -3
done
```
All seven should print a `✓` line.

## Recommended next-session priority

1. **DCT16x8 + DCT8x16 (forward + IDCT)** — quickest ROI. Helpers already
   exist in `kernels/dct16.rs` (just needs new kernel entry points + launchers
   + parity tests). +4 to deliverables in ~2 hours.
2. **DCT4 family (4x4, 4x8, 8x4 forward + IDCT)** — mirrors DCT8 structure
   with smaller blocks. New `kernels/dct4.rs`. +6 deliverables.
3. **DCT32 family (32x32, 32x16, 16x32 forward + IDCT)** — wraps DCT16.
   New `kernels/dct32.rs`. +6 deliverables. Pattern is identical to DCT16x16
   but with 32-pt butterflies.
4. **`compute_pre_erosion` and `per_block_modulations`** (adaptive_quant) —
   needed for the encoder's quant-field generation. Look at
   `jxl-encoder-simd/src/adaptive_quant.rs::*_scalar`.
5. **`cfl_find_best_multiplier`** — per-tile Newton iteration, can stay
   host-side initially.

After ~25 of 37 deliverables, start Phase 3 / Phase 4 architecture work
(see initial CONTEXT-HANDOFF rationale on whole-image-per-strategy AC search).

## Honest scope reminder

This is a 6-8 week port for a focused human. Two autonomous sessions have
delivered the foundation + 35% of the kernels. Per-kernel pace is now
~5 minutes for build + ~10 minutes for parity test (most build is
incremental). Architecture decisions (crate layout, parity discipline,
whole-image AC search approach) are locked. Do NOT claim "100% complete"
until `jxl-encoder` produces tolerance-bounded lossy output through the
GPU pipeline AND a roundtrip through `jxl-rs` passes on a real photo.
