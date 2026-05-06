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

## Status

**Pre-alpha**, work in progress. **45 of ~52 deliverables verified (~87%)**.
See [`PORT_STATUS.md`](PORT_STATUS.md) for the per-kernel grid.

The DCT/IDCT family is complete (26 sub-kernels: DCT8/16/32/64 squares +
rectangulars + DCT4 sub-block variants). Phase 2 non-DCT primitives done
(quantize, dequant, pixel_loss, block_l2, cfl, epf, entropy, adaptive_quant).
Phase 3 (whole-image AC strategy search) prototyped: cost-grid composition
+ host-side partition selector with 12 unit tests covering 3 region tiers ×
4 strategies each.

Phase 4 (`jxl-encoder` integration) requires modifying the sibling
`jxl-encoder` repository — pending user authorization.

## Quick start

```rust
use cubecl::prelude::*;
use jxl_encoder_gpu::launch::xyb::{xyb_forward, xyb_inverse};

type Backend = cubecl::cuda::CudaRuntime;

let device = <Backend as cubecl::Runtime>::Device::default();
let client = <Backend as cubecl::Runtime>::client(&device);

// Linear-RGB input, 256×256 pixels, planar (R, G, B separate).
let n = 256 * 256;
let r = vec![0.5_f32; n];
let g = vec![0.5_f32; n];
let b = vec![0.5_f32; n];

let h_r = client.create_from_slice(f32::as_bytes(&r));
let h_g = client.create_from_slice(f32::as_bytes(&g));
let h_b = client.create_from_slice(f32::as_bytes(&b));
let h_x = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
let h_y = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
let h_b_out = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

xyb_forward::<Backend>(&client, h_r, h_g, h_b, h_x, h_y, h_b_out, n as u32);

// Read result back to host
let xyb_y_bytes = client.read_one(h_y).expect("read y");
let xyb_y: &[f32] = f32::from_bytes(&xyb_y_bytes);
println!("XYB Y[0] = {}", xyb_y[0]);
```

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
