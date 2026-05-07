# Changelog

## [Unreleased]

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
