# Changelog

## [Unreleased]

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
