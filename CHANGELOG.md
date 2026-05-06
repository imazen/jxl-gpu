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
