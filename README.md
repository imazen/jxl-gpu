# jxl-encoder-gpu

Multi-vendor GPU kernels for [`jxl-encoder`](https://crates.io/crates/jxl-encoder)
via [CubeCL](https://github.com/tracel-ai/cubecl).

A single `#[cube]`-annotated Rust source dispatches across CUDA (NVIDIA),
WGPU (Vulkan / Metal / DX12 / WebGPU), HIP (AMD ROCm), and a build-time CPU
fallback. Mirrors [`jxl-encoder-simd`](https://crates.io/crates/jxl-encoder-simd)
1:1 — every kernel here has a sub-ulp parity target against the scalar function
in `jxl-encoder-simd`.

## Status

**Pre-alpha**, work in progress. See [`PORT_STATUS.md`](PORT_STATUS.md) for the
per-kernel grid.

## License

Dual-licensed: AGPL-3.0-only ([LICENSE-AGPL3](LICENSE-AGPL3)) or commercial
(Imazen Site-wide Subscription License v1.1; see
[LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) and https://www.imazen.io/pricing).

Algorithms and constants derived from [libjxl](https://github.com/libjxl/libjxl)
(BSD-3-Clause).
