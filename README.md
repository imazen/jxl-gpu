# jxl-encoder-gpu

[![CI](https://github.com/imazen/jxl-encoder-gpu/actions/workflows/ci.yml/badge.svg?style=flat-square)](https://github.com/imazen/jxl-encoder-gpu/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--commercial-blue?style=flat-square)](LICENSE-AGPL3)

Multi-vendor GPU kernels for [`jxl-encoder`](https://crates.io/crates/jxl-encoder)
via [CubeCL](https://github.com/tracel-ai/cubecl).

A single `#[cube]`-annotated Rust source dispatches across CUDA (NVIDIA),
WGPU (Vulkan / Metal / DX12 / WebGPU), HIP (AMD ROCm), and a build-time CPU
fallback. Mirrors [`jxl-encoder-simd`](https://crates.io/crates/jxl-encoder-simd)
1:1 — every kernel here has a sub-ulp parity target against the scalar function
in `jxl-encoder-simd`.

## Status — GPU pipeline now FASTER than CPU AVX2

End-to-end CPU vs GPU lossy DCT8 throughput (RTX 5070 + Ryzen 9 7950X):

```
side    CPU ms    GPU ms    ratio        throughput
 256     1.61     1.49     1.08× GPU    44 MP/s
 512     9.39     4.28     2.19× GPU    61 MP/s
1024    39.46    17.66     2.23× GPU    59 MP/s
2048   150.55    41.33     3.64× GPU   101 MP/s vs 28 MP/s
```

Parity 4e-6 max abs delta. **For batch workloads** (encoding the
same image at multiple settings), input-handle reuse delivers an
additional **5.28× speedup** at 2048² × 5 settings (366 MP/s
aggregate vs 69 MP/s naive).

**Implemented:**
- 11 fork modules covering the full lossy DCT8 pipeline
- Persistent GPU buffer API (`crate::persistent`) — typed `GpuPlane`
  / `GpuBlocks` / `GpuI32Blocks` handles with chained-launch methods
- High-level `LossyEncoder` API (`crate::lossy_encoder`) —
  `encode_one` + `encode_many` for one-shot and batch use cases
- 26 GPU kernels: 13 DCT/IDCT strategies + supporting (XYB,
  gaborish, mask1x1, gather/scatter, DC restore, fused DCT+quant,
  fused dequant+IDCT-Y, etc.)
- 55 unit tests passing (cuda) + 12 partition selector tests (cpu)
- 13 example demos covering composition, real-image roundtrips
  (djxl + jxl-rs verified), and throughput benchmarks at 64²-4096²

**What it does NOT yet do:**
- Produce JXL bitstream bytes directly (use
  `GpuEncoder::encode_lossy_via_cpu` which delegates to jxl-encoder
  for the full encode; GPU path is for the parallel pipeline stages,
  not entropy coding / container muxing).
- DC quant + entropy coding (DC restore is a passthrough).
- Strategy-search dispatch above DCT8 in the high-level encoder.

## Optional: GPU butteraugli quant-refinement loop

Behind the `butteraugli-loop` cargo feature: a content-aware
adaptive-quantization refinement loop driven by GPU butteraugli
(`zenmetrics/butteraugli-gpu`). Mirrors
`jxl_encoder::vardct::butteraugli_loop::butteraugli_refine_quant_field`
with the per-iteration distance compute on GPU instead of CPU
(~46 ms/iter at 1024² on RTX 5070).

**Smart content-aware gate** (`refine_aq_field_gpu_smart`): always
measures AQ vs uniform baselines first, then chooses one of three
paths per image:
- AQ regresses uniform by > 10% → fall back to a uniform qac field
- High distance (> 1.5) → return initial AQ as-is
- Else → run the full refinement loop

Validated on a 16-image CLIC2025-1024 sweep across two metrics:

| dist | uniform → smart (butteraugli) | uniform → smart (SSIMULACRA2) |
|------|-------------------------------|--------------------------------|
| 1.0  | 1.2386 → 1.1886 (-4.0%)       | 87.913 → 88.488 (+0.575)       |
| 2.0  | 2.0245 → 2.0240 (-0.0%)       | 80.099 → 81.412 (+1.313)       |
| 4.0  | 3.2582 → 3.2417 (-0.5%)       | 67.677 → 70.445 (+2.768)       |

Smart gate **never regresses uniform** on either metric. The
threshold is tunable via `refine_aq_field_gpu_smart_with_threshold`
(default 1.10 for butteraugli optimum; 1.30 for SSIM2 optimum). See
`examples/butteraugli_refinement_demo` for usage and
`examples/butteraugli_refinement_corpus_sweep` for the validation
harness.

See [`PORT_STATUS.md`](PORT_STATUS.md) for the per-kernel grid +
[`CONTEXT-HANDOFF.md`](CONTEXT-HANDOFF.md) for the perf breakthrough
write-up.

## Quick start (high-level `LossyEncoder` API)

```rust
use jxl_encoder_gpu::encoder::GpuEncoder;
use jxl_encoder_gpu::lossy_encoder::{LossyEncoder, distance_to_qac};

type Backend = cubecl::cuda::CudaRuntime;

let enc: GpuEncoder<Backend> = GpuEncoder::new();

// Construct one LossyEncoder per (width, height) — amortizes
// per-channel quant matrix uploads across all encodes that follow.
// Arbitrary dimensions are supported (padded internally).
let lossy = LossyEncoder::new(&enc, 1024, 768);

// sRGB U8 input from any image source (e.g., the `image` crate).
let rgb_in: Vec<u8> = vec![128; 1024 * 768 * 3];

// libjxl-style distance: 1.0 = visually transparent, higher = lossier.
let qac = distance_to_qac(1.0);
let rgb_out: Vec<u8> = lossy.encode_one_srgb_u8(&enc, &rgb_in, qac);

// Batch: encode the same input at multiple settings (one upload).
let qacs = [distance_to_qac(0.5), distance_to_qac(1.0), distance_to_qac(2.0)];
let outputs: Vec<Vec<u8>> = lossy.encode_many_srgb_u8(&enc, &rgb_in, &qacs);
```

For lower-level access (custom pipelines, individual GPU kernels),
see [`crate::persistent`] and [`crate::launch`].

## Backends

Cargo features select the cubecl backend(s):

```toml
[dependencies]
jxl-encoder-gpu = { version = "0.0.1", default-features = false, features = ["cuda"] }
```

| Feature | Backend | Notes |
|---|---|---|
| `cuda` (default) | NVIDIA CUDA via cubecl-cuda | Needs CUDA 13+ on host (see [zenmetrics CUBECL_GOTCHAS.md](https://github.com/imazen/zenmetrics/blob/main/docs/CUBECL_GOTCHAS.md) G3.1) |
| `wgpu` | Vulkan / Metal / DX12 / WebGPU via cubecl-wgpu | Won't work in WSL2 without Vulkan ICD |
| `hip` | AMD ROCm via cubecl-hip | Untested in this repo |
| `cpu` | cubecl-cpu LLVM JIT | Useful for CI without GPU; no atomics support |

Multiple may be enabled. The caller selects at construction time by passing
the runtime type to launcher functions.

## Architecture: whole-image-per-strategy AC search

Per-region trial encoding (libjxl's approach) produces severe GPU control-flow
divergence — adjacent regions in a warp pick different candidate strategies.
This crate uses **whole-image-per-strategy** instead: for each candidate
strategy S, one fully-coherent whole-image kernel computes per-tile cost.
A host-side partition selector then enumerates legal partitions per
32×32/64×64 region and picks the min-cost partition.

This is algebraically equivalent to libjxl's algorithm — `AdjustQuantBlockAC`
runs after selection, so trial cost has no neighbor coupling. Eliminates GPU
control-flow divergence; pruning gates (DCT64×64 only at d≥3, etc.) become
free to drop. See [`CLAUDE.md`](CLAUDE.md) for the full rationale.

## Parity discipline

Every kernel ships with a runnable parity example under `examples/` that
validates against the corresponding `jxl_encoder_simd::*_scalar` function on
synthetic data. Tolerances are documented per-kernel:

- Most kernels: sub-ulp (3e-9 to 6e-7 abs)
- Bit-exact (0 diffs): `quantize_dct8`, `quantize_large`, `cfl::find_best_multiplier`,
  `cfl::find_best_multiplier_newton`, `pad_plane`
- f64 ulp (3.75e-16 rel): `pixel_loss` (uses cubecl's f64 ops on CUDA)
- FMA-noise tolerance: `mask1x1` (1.5e-3 abs), `dequant_dct8` X/B (1.5e-5),
  `per_block_modulations` (5e-4 rel)

## Reference docs

- [`PORT_STATUS.md`](PORT_STATUS.md) — per-kernel verification grid
- [`CONTEXT-HANDOFF.md`](CONTEXT-HANDOFF.md) — current state-of-the-port for next session
- [`CLAUDE.md`](CLAUDE.md) — autonomous-mandate rules + architectural decisions
- [`CHANGELOG.md`](CHANGELOG.md) — what landed when

External:
- [zenmetrics CUBECL_PORTING_GUIDE.md](https://github.com/imazen/zenmetrics/blob/main/docs/CUBECL_PORTING_GUIDE.md)
- [zenmetrics CUBECL_GOTCHAS.md](https://github.com/imazen/zenmetrics/blob/main/docs/CUBECL_GOTCHAS.md)

## License

Dual-licensed: AGPL-3.0-only ([LICENSE-AGPL3](LICENSE-AGPL3)) or commercial
(Imazen Site-wide Subscription License v1.1; see
[LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) and https://www.imazen.io/pricing).

Algorithms and constants derived from [libjxl](https://github.com/libjxl/libjxl)
(BSD-3-Clause).
