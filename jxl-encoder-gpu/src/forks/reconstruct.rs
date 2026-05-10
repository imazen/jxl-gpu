// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Forked from jxl-encoder vardct/reconstruct.rs (BSD-3-Clause via libjxl
// + AGPL/commercial), with the SIMD calls substituted for GPU launches.
// Licensed under AGPL-3.0-or-later or commercial.

//! GPU-substituted final-stage reconstruction utilities.
//!
//! Currently covers the cleanly substitutable, kernel-bound pieces of
//! `jxl_encoder::vardct::reconstruct`:
//! - `gab_smooth_gpu` — 3-channel decoder gab smoothing
//! - `xyb_to_linear_rgb_planar_gpu` — XYB → planar linear RGB
//! - `xyb_to_linear_rgb_gpu` — XYB → interleaved linear RGB
//!   (re-interleaves on host after GPU planar inverse)
//!
//! Not yet covered (algorithm-heavy, not pure SIMD):
//! - `restore_llf_from_dc` (Hadamard inverse for large transforms)
//! - `idct_for_strategy` (per-strategy IDCT dispatch)
//! - `reconstruct_xyb_impl` (full pipeline orchestration)
//!
//! These can be progressively forked once we have GPU kernels for the
//! per-strategy IDCT dispatch logic (we have all the IDCT math; we just
//! need a host-side strategy selector that picks the right kernel).
//!
//! Reshape vs upstream:
//! - `gab_smooth`: original CPU code reuses one scratch buffer across
//!   all 3 channels. GPU kernel manages its own buffer; we just call it
//!   3 times sequentially. Future fusion: a 3-channel GPU kernel that
//!   does X+Y+B in one launch.
//! - `xyb_to_linear_rgb` (interleaved): GPU kernel returns planar
//!   buffers; we re-interleave on host. The alternative (interleaved
//!   GPU output) would force stride-3 stores on CUDA — slower than
//!   planar + host re-interleave at the sizes we care about.

use alloc::vec;
use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;

/// `INV_DC_QUANT[c]` constants — channel-specific inverse DC quantizers
/// from upstream `jxl_encoder::vardct::quant::INV_DC_QUANT`. Used by
/// the DC override step of `reconstruct_xyb_impl`.
pub const INV_DC_QUANT: [f32; 3] = [4096.0, 512.0, 256.0];

/// `DCT_RESAMPLE_SCALE_16_TO_2[i]` — scale factors for the 2-point
/// resample used by the DC-from-DCT16 forward operation. Bit-for-bit
/// from upstream `jxl_encoder::vardct::dct::constants::DCT_RESAMPLE_SCALE_16_TO_2`.
pub const DCT_RESAMPLE_SCALE_16_TO_2: [f32; 2] = [1.000_000_000_0, 0.901_764_2];

/// Dequantize a single channel's DC value with the channel-specific
/// CfL contribution from Y. Mirrors the inline `dequant_dc` closure in
/// upstream's `restore_llf_from_dc`.
///
/// - `channel == 0` (X) or `channel == 1` (Y): no CfL → returns
///   `quant_dc / inv_factor`.
/// - `channel == 2` (B): adds `quant_dc_y * 0.5 / inv_factor` (the
///   fixed B-channel DC-level CfL contribution).
///
/// `inv_factor = INV_DC_QUANT[channel] * scale_dc`.
#[inline]
pub fn dequant_dc_channel(quant_dc: f32, quant_dc_y: f32, channel: usize, scale_dc: f32) -> f32 {
    let dc_cfl_factor: f32 = if channel == 2 { 0.5 } else { 0.0 };
    let inv_factor = INV_DC_QUANT[channel] * scale_dc;
    (quant_dc + quant_dc_y * dc_cfl_factor) / inv_factor
}

/// `DCT_RESAMPLE_SCALE_32_TO_4[i]` — scale factors for the 4-point
/// resample used by the DC-from-DCT32 forward operation.
/// Bit-for-bit from upstream.
pub const DCT_RESAMPLE_SCALE_32_TO_4: [f32; 4] = [1.0, 0.974_886_8, 0.901_764_2, 0.787_054_9];

/// In-place 4-point DCT (libjxl `dct1d_4`). Pure scalar, used by the
/// DCT32 LLF restoration.
#[allow(clippy::approx_constant)] // upstream literal SQRT2 = 1.41421356237
fn dct1d_4(mem: &mut [f32]) {
    const SQRT2: f32 = 1.414_213_5;
    const WC4: [f32; 2] = [0.541_196_1, 1.306_563_0];
    let (a, b, c, d) = (mem[0], mem[1], mem[2], mem[3]);
    let t0 = a + d;
    let t1 = b + c;
    let t2 = a - d;
    let t3 = b - c;
    let u0 = t0 + t1;
    let u1 = t0 - t1;
    let v0 = t2 * WC4[0];
    let v1 = t3 * WC4[1];
    let w0 = v0 + v1;
    let w1 = v0 - v1;
    let b0 = SQRT2 * w0 + w1;
    mem[0] = u0;
    mem[1] = b0;
    mem[2] = u1;
    mem[3] = w1;
}

/// In-place 2-point DCT (libjxl `dct1d_2`). Pure scalar:
/// `[a, b] -> [a + b, a - b]`. Used by the rectangular DCT32×16 /
/// DCT16×32 LLF restoration.
fn dct1d_2(mem: &mut [f32]) {
    let a = mem[0];
    let b = mem[1];
    mem[0] = a + b;
    mem[1] = a - b;
}

/// In-place 8-point DCT (libjxl `dct1d_8`). Pure scalar — bit-for-bit
/// port of upstream's `dct1d_8_val` butterfly. Used by the DCT64×64 /
/// DCT64×32 / DCT32×64 LLF restoration.
#[allow(clippy::approx_constant)] // upstream literal SQRT2 = 1.41421356237
fn dct1d_8(mem: &mut [f32]) {
    const SQRT2: f32 = 1.414_213_5;
    const WC4: [f32; 2] = [0.541_196_1, 1.306_563_0];
    const WC8: [f32; 4] = [0.509_795_6, 0.601_344_9, 0.899_976_2, 2.562_915_4];
    let m = [
        mem[0], mem[1], mem[2], mem[3], mem[4], mem[5], mem[6], mem[7],
    ];
    let t0 = m[0] + m[7];
    let t1 = m[1] + m[6];
    let t2 = m[2] + m[5];
    let t3 = m[3] + m[4];
    let t4 = m[0] - m[7];
    let t5 = m[1] - m[6];
    let t6 = m[2] - m[5];
    let t7 = m[3] - m[4];
    // dct1d_4_val on first half (t0..t3).
    let dct4 = |a: f32, b: f32, c: f32, d: f32| -> [f32; 4] {
        let u0 = a + d;
        let u1 = b + c;
        let u2 = a - d;
        let u3 = b - c;
        let v0 = u0 + u1;
        let v1 = u0 - u1;
        let w0 = u2 * WC4[0];
        let w1 = u3 * WC4[1];
        let s0 = w0 + w1;
        let s1 = w0 - w1;
        [v0, SQRT2 * s0 + s1, v1, s1]
    };
    let r0 = dct4(t0, t1, t2, t3);
    // Wc multiply on second half (t4..t7).
    let w4 = t4 * WC8[0];
    let w5 = t5 * WC8[1];
    let w6 = t6 * WC8[2];
    let w7 = t7 * WC8[3];
    // dct1d_4_val on second half.
    let r1 = dct4(w4, w5, w6, w7);
    // B transform.
    let b0 = SQRT2 * r1[0] + r1[1];
    let b1 = r1[1] + r1[2];
    let b2 = r1[2] + r1[3];
    let b3 = r1[3];
    // InverseEvenOdd: interleave dct4 results with B transform.
    mem[0] = r0[0];
    mem[1] = b0;
    mem[2] = r0[1];
    mem[3] = b1;
    mem[4] = r0[2];
    mem[5] = b2;
    mem[6] = r0[3];
    mem[7] = b3;
}

/// Compute the per-(8×8) block DC value for a padded spatial plane.
/// Returns one f32 per block in raster order (row-major over 8×8
/// block grid).
///
/// DC value matches this codebase's DCT forward convention: the
/// **mean** of the 64 pixels in the block (`sum / 64`). Verified
/// against the kernels: a uniform-0.4 input gives `forward_dct8(...)
/// .coeffs[0] = 0.4`, and `dc_from_dct_16x16(forward_dct16x16(...))
/// = [0.4, 0.4, 0.4, 0.4]`. (libjxl's CPU code uses an orthonormal
/// `sum / 8` convention; the GPU kernels here normalize by N² instead,
/// so the DC frame stores mean values.)
///
/// Used by [`encode_and_reconstruct_mixed_strategy_single_channel`]
/// to build the `dc_grid_per_8x8_block` argument that
/// [`dispatch_restore_llf`] consumes.
pub fn compute_dc_grid_per_8x8_block(
    plane: &[f32],
    padded_width: usize,
    padded_height: usize,
) -> Vec<f32> {
    debug_assert_eq!(plane.len(), padded_width * padded_height);
    debug_assert!(padded_width.is_multiple_of(8));
    debug_assert!(padded_height.is_multiple_of(8));
    let xsize_blocks = padded_width / 8;
    let ysize_blocks = padded_height / 8;
    let mut out = alloc::vec![0.0_f32; xsize_blocks * ysize_blocks];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let mut sum = 0.0_f32;
            let y0 = by * 8;
            let x0 = bx * 8;
            for dy in 0..8 {
                let row_off = (y0 + dy) * padded_width + x0;
                for dx in 0..8 {
                    sum += plane[row_off + dx];
                }
            }
            // mean = sum / 64
            out[by * xsize_blocks + bx] = sum * (1.0 / 64.0);
        }
    }
    out
}

/// `DCT_RESAMPLE_SCALE_64_TO_8[i]` — scale factors for the 8-point
/// resample used by the DC-from-DCT64 forward operation.
/// Bit-for-bit from upstream.
pub const DCT_RESAMPLE_SCALE_64_TO_8: [f32; 8] = [
    1.0,
    0.993_686_6,
    0.974_886_8,
    0.944_018_1,
    0.901_764_2,
    0.849_057_5,
    0.787_054_9,
    0.717_108_1,
];

/// One block's reconstruct recipe — the inputs to the multi-strategy
/// reconstruct orchestrator.
///
/// `coeffs.len()` must equal
/// `forks::transform::coeff_count_per_strategy(raw_strategy)` —
/// 64 for DCT8/DCT4×/IDENTITY/DCT2X2, 128 for DCT16×8/DCT8×16,
/// 256 for DCT16×16, 512 for DCT32×16/DCT16×32, 1024 for DCT32×32,
/// 2048 for DCT64×32/DCT32×64, 4096 for DCT64×64.
///
/// The caller is responsible for producing already-dequantized,
/// CfL-corrected, LLF-restored coefficients (use `dispatch_restore_llf`
/// for the LLF stage). AFV0-3 strategies route through `forks::afv`
/// instead — they panic if passed to `reconstruct_mixed_strategy_gpu`.
#[derive(Debug, Clone)]
pub struct BlockRecipe<'a> {
    pub bx: usize,
    pub by: usize,
    pub raw_strategy: u8,
    pub coeffs: &'a [f32],
}

/// Multi-strategy reconstruct orchestrator. Groups recipes by AC
/// strategy, emits one batched IDCT launch per group, then scatters
/// each block individually into the padded plane.
///
/// This is the efficient form for mixed-strategy reconstruct: at
/// most 15 GPU launches per image (one per supported strategy that
/// appears in `recipes`), regardless of block count.
///
/// **AFV constraint**: AFV0-3 are not supported here; they require
/// the per-block sub-transform composition handled by
/// `forks::afv::afv_transform_batch_gpu` (forward) /
/// `inverse_afv_transform_batch_gpu` (inverse). Caller must filter
/// AFV blocks out and process them separately.
///
/// **Per-strategy LLF region dims** (in coefficient-grid units):
/// matches the upstream `AcStrategy::ll_dim_x/y()` pair. DC values
/// from the underlying 8×8 sub-blocks live in this rectangle; the
/// rest of the LLF block is restored by the per-strategy
/// `restore_llf_*` helpers.
fn llf_dims_for_strategy(raw_strategy: u8) -> (u32, u32) {
    use crate::forks::transform::{
        RAW_STRATEGY_DCT, RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8,
        RAW_STRATEGY_DCT8X4, RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16,
        RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64,
        RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64, RAW_STRATEGY_IDENTITY,
    };
    use crate::forks::transform::{
        RAW_STRATEGY_AFV0, RAW_STRATEGY_AFV1, RAW_STRATEGY_AFV2, RAW_STRATEGY_AFV3,
    };
    match raw_strategy {
        RAW_STRATEGY_DCT
        | RAW_STRATEGY_DCT4X4
        | RAW_STRATEGY_DCT4X8
        | RAW_STRATEGY_DCT8X4
        | RAW_STRATEGY_IDENTITY
        | RAW_STRATEGY_DCT2X2
        | RAW_STRATEGY_AFV0
        | RAW_STRATEGY_AFV1
        | RAW_STRATEGY_AFV2
        | RAW_STRATEGY_AFV3 => (1, 1),
        RAW_STRATEGY_DCT16X8 => (1, 2),
        RAW_STRATEGY_DCT8X16 => (2, 1),
        RAW_STRATEGY_DCT16X16 => (2, 2),
        RAW_STRATEGY_DCT32X16 => (2, 4),
        RAW_STRATEGY_DCT16X32 => (4, 2),
        RAW_STRATEGY_DCT32X32 => (4, 4),
        RAW_STRATEGY_DCT64X32 => (4, 8),
        RAW_STRATEGY_DCT32X64 => (8, 4),
        RAW_STRATEGY_DCT64X64 => (8, 8),
        _ => panic!(
            "llf_dims_for_strategy: unsupported strategy {raw_strategy} \
             (use forks::afv for AFV0-3)"
        ),
    }
}

/// Encode a single channel under a heterogeneous strategy assignment
/// and reconstruct the result into a padded plane.
///
/// **Phase A MVP** of the AC strategy search orchestrator. Closes the
/// gap between `pipeline::partitions_16x16_to_assignments` (host
/// partition selector output) and
/// `reconstruct_mixed_strategy_gpu` (mixed-strategy IDCT). The
/// algorithm per strategy in the assignments:
///
/// 1. Gather pixel blocks at the strategy's tile size
///    (`apply_dct_batch_gpu` does the gather + forward DCT in one).
/// 2. Quantize via the broadcast-weights kernel (one weights template
///    per strategy; per-block qac vector covers all blocks of that
///    strategy).
/// 3. Dequant via `dequant_simple_blocks_broadcast_w` (no CfL — that
///    happens in the encoder-side dequant_dct8 path; for the
///    strategy-search reconstruct we use the simpler per-coefficient
///    dequant so the broadcast pattern works for all block sizes).
/// 4. Per-block LLF restore via `dispatch_restore_llf` — pulls
///    `llf_dim_x × llf_dim_y` DC values from `dc_grid` (which is in
///    8×8-block units, raster order).
/// 5. Build `BlockRecipe` slices into the dequantized buffer.
/// 6. `reconstruct_mixed_strategy_gpu(...)` runs the mixed-strategy
///    IDCT batch + scatter into `out_plane`.
///
/// **Single-channel only.** 3-channel orchestration is a thin
/// wrapper that calls this 3 times (once per X / Y / B), reusing
/// the same assignments + DC grid shape per channel. Each channel
/// has its own quant matrix templates + qac scaling + dc_grid.
///
/// **Strategy support**: every standard JXL AC strategy except
/// AFV0-3. The 8×8-class small strategies (DCT4×4, DCT4×8, DCT8×4,
/// IDENTITY, DCT2X2) are handled identically to DCT8 via
/// `apply_dct_batch_gpu`'s dispatch.
#[allow(clippy::too_many_arguments)]
pub fn encode_and_reconstruct_mixed_strategy_single_channel<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_channel: &[f32],
    padded_width: usize,
    padded_height: usize,
    assignments: &[crate::pipeline::StrategyAssignment],
    weights_template_for_strategy: &dyn Fn(u8) -> Vec<f32>,
    qac_per_8x8_block: &[f32],
    thresholds: &[f32; 4],
    dc_grid_per_8x8_block: &[f32],
    // Channel index for adjust_quant_bias: 0=X, 1=Y, 2=B. Used only
    // for DCT8 (RAW_STRATEGY_DCT) — non-DCT8 strategies skip the bias
    // correction in upstream too.
    channel: usize,
    // Host output plane. Required when `out_plane_gpu` is `None`
    // (legacy path: GPU-resident strategies scatter into an internal
    // GpuPlane which is downloaded + added into this buffer at exit;
    // AFV strategies host-scatter directly into it). When
    // `out_plane_gpu` is `Some`, `out_plane` is left untouched and
    // may safely be `None` (caller can skip the dead allocation).
    mut out_plane: Option<&mut [f32]>,
    // Optional pre-uploaded GPU plane mirroring `xyb_channel`. When
    // `Some`, the function reuses the caller's GpuPlane and skips the
    // internal `upload_plane(xyb_channel)` PCIe transfer — useful when
    // the caller already produced this plane on GPU (e.g., from
    // gaborish in `prepare_strategy_search_plan_traced`) and stored it
    // in a `StrategySearchPlan`. The `xyb_channel` host slice is still
    // required for the AFV strategy path (host extend_from_slice).
    xyb_channel_gpu: Option<&crate::persistent::GpuPlane<R>>,
    // Optional pre-uploaded GPU dc_grid mirroring
    // `dc_grid_per_8x8_block`. When `Some`, the function reuses the
    // caller's GpuBlocks and skips the per-call
    // `upload_blocks(dc_grid_per_8x8_block)` PCIe transfer that the
    // 1×1-LLF / DCT16/32/64 GPU LLF kernels would otherwise pay each
    // time. `dc_grid_per_8x8_block` host slice is still required for
    // the AFV branch's `coeffs[0] = mean` host loop.
    dc_grid_gpu: Option<&crate::persistent::GpuBlocks<R>>,
    // Optional GPU output plane. When `Some`, the per-strategy IDCT
    // output is scattered directly into this plane (caller-supplied
    // zero-init expected — same coverage-disjoint invariant) and the
    // function skips the internal upload-zeros + download-and-merge
    // pair. When `None`, the function allocates an internal
    // zero-filled GpuPlane, scatters into it, and at function exit
    // downloads it and adds to `out_plane` (the host buffer).
    //
    // Use `Some` when the caller's downstream pipeline can consume
    // GpuPlane directly (e.g., the postpass gab_smooth +
    // EPF + xyb_to_linear chain in lossy_encoder). The host
    // `out_plane` still must be supplied for the AFV branch (which
    // host-scatters there) and as a fallback target — but when
    // `out_plane_gpu` is `Some`, the function ALSO routes AFV writes
    // to the GpuPlane via indexed_scatter so `out_plane` is left
    // untouched. (AFV is currently never selected in production cost
    // grids.)
    out_plane_gpu: Option<&crate::persistent::GpuPlane<R>>,
) {
    use crate::forks::transform::{
        apply_dct_batch_persistent, coeff_count_per_strategy, tile_dims_pixels, RAW_STRATEGY_DCT,
    };
    use crate::pipeline::group_assignments_by_strategy;

    debug_assert_eq!(xyb_channel.len(), padded_width * padded_height);
    // `out_plane` must be supplied unless `out_plane_gpu` covers all
    // strategies (which it does when Some — both the GPU strategies'
    // scatter target and the AFV branch's GPU scatter target). When
    // both are None, GPU strategies write into an internal GpuPlane
    // that we'd later need to download into out_plane — so the
    // caller's `out_plane = None` is only valid alongside Some
    // out_plane_gpu.
    if let Some(o) = out_plane.as_deref() {
        debug_assert_eq!(o.len(), padded_width * padded_height);
    } else {
        debug_assert!(
            out_plane_gpu.is_some(),
            "encode_and_reconstruct_mixed_strategy_single_channel: at least one of \
             out_plane / out_plane_gpu must be Some"
        );
    }

    let xsize_blocks_8 = padded_width / 8;
    let ysize_blocks_8 = padded_height / 8;
    debug_assert_eq!(qac_per_8x8_block.len(), xsize_blocks_8 * ysize_blocks_8);
    debug_assert_eq!(dc_grid_per_8x8_block.len(), xsize_blocks_8 * ysize_blocks_8);

    let groups = group_assignments_by_strategy(assignments);

    // Upload the spatial channel ONCE — every non-AFV strategy below
    // gathers its per-strategy block subset from this GpuPlane via
    // indexed_gather_blocks_persistent (one GPU kernel launch per
    // strategy), replacing the host extend_from_slice loop +
    // per-strategy upload_blocks PCIe round-trip the older code did.
    // AFV strategies (currently disabled in production cost grids
    // but the code path is kept) still build a host batch because
    // forks::afv::afv_transform_batch_gpu takes a host slice.
    //
    // When the caller already has the spatial channel on GPU (e.g., the
    // gaborished plane from prepare_strategy_search_plan_traced), they
    // can skip this upload by passing `xyb_channel_gpu = Some(&plane)`.
    let g_plane_owned;
    let g_plane: &crate::persistent::GpuPlane<R> = if let Some(p) = xyb_channel_gpu {
        debug_assert_eq!(p.width(), padded_width as u32);
        debug_assert_eq!(p.height(), padded_height as u32);
        p
    } else {
        g_plane_owned =
            enc.upload_plane(xyb_channel, padded_width as u32, padded_height as u32);
        &g_plane_owned
    };

    // GPU output plane: when the caller provides one, scatter directly
    // into it (no internal upload-zeros, no download+merge at exit).
    // When None, allocate one internally — disjoint per-pixel coverage
    // (AC strategy assignments partition the image's 8×8 grid) means
    // adding the GPU and host out planes at the end equals the union
    // with no overlap.
    let g_out_plane_owned;
    let g_out_plane: &crate::persistent::GpuPlane<R> = if let Some(p) = out_plane_gpu {
        debug_assert_eq!(p.width(), padded_width as u32);
        debug_assert_eq!(p.height(), padded_height as u32);
        p
    } else {
        // alloc_plane (commit 0a04ed89) returns an actually-zeroed
        // buffer — no need for an explicit upload_plane(&zeros).
        g_out_plane_owned = enc.alloc_plane(padded_width as u32, padded_height as u32);
        &g_out_plane_owned
    };
    let mut used_gpu_for_any = false;

    // Upload the per-(8×8)-block DC grid ONCE (caller-supplied). Used
    // by the 1×1-LLF GPU LLF restore (set_dc_from_grid_indexed_persistent)
    // for every 1×1-LLF strategy that the loop sees. When the caller
    // provides `dc_grid_gpu = Some(&g_dc)` (e.g., from
    // StrategySearchPlan), this upload is skipped — saves
    // n_padded_blocks * 4 bytes per call (≈ 64 KB at 1024² × 3 channels
    // = 192 KB of PCIe per encode iter).
    let n_blocks_8x8 = (xsize_blocks_8 * ysize_blocks_8) as u32;
    let g_dc_grid_owned;
    let g_dc_grid: &crate::persistent::GpuBlocks<R> = if let Some(g) = dc_grid_gpu {
        debug_assert_eq!(g.num_blocks(), n_blocks_8x8);
        debug_assert_eq!(g.coeffs_per_block(), 1);
        g
    } else {
        g_dc_grid_owned = enc.upload_blocks(dc_grid_per_8x8_block, n_blocks_8x8, 1);
        &g_dc_grid_owned
    };

    // Per-strategy: encode and store dequant result. Buffers live in
    // `coeff_buffers` for the lifetime of the recipe build + reconstruct
    // call below.
    let mut coeff_buffers: Vec<(u8, Vec<(usize, usize)>, Vec<f32>)> =
        Vec::with_capacity(groups.len());

    use crate::forks::transform::{
        RAW_STRATEGY_AFV0, RAW_STRATEGY_AFV1, RAW_STRATEGY_AFV2, RAW_STRATEGY_AFV3,
    };
    let is_afv = |s: u8| {
        s == RAW_STRATEGY_AFV0
            || s == RAW_STRATEGY_AFV1
            || s == RAW_STRATEGY_AFV2
            || s == RAW_STRATEGY_AFV3
    };

    for (raw_strategy, coords) in groups {
        let coeff_count = coeff_count_per_strategy(raw_strategy);
        let (tile_w, tile_h) = tile_dims_pixels(raw_strategy);
        // (grid_width, grid_height) in coefficient-grid units; same as tile pixels here.
        let grid_w = tile_w as u32;
        let grid_h = tile_h as u32;
        let (llf_x, llf_y) = llf_dims_for_strategy(raw_strategy);

        // Per-region qac. For multi-block transforms (DCT16x16, DCT32x32,
        // DCT64x64, rectangular variants, etc.), take the MAX qac across
        // all covered 8x8 sub-blocks of the region — not just the
        // first-block coord. Higher qac = lighter quant = better quality;
        // the butteraugli refinement loop bumps qac UP on perceptually-
        // bad blocks. If we ignored sub-block qac for multi-block
        // transforms, the loop's bumps would be discarded for 3/4 of the
        // sub-blocks of a DCT16x16, 15/16 of a DCT32x32, etc., and the
        // refined-strat pipeline would plateau (verified at d=1.5 in
        // combined_strat_search_aq_demo before this fix).
        //
        // Single-block strategies (DCT8, DCT2x2, IDENTITY, DCT4x4,
        // sub-block 8x8 family) have covered_blocks_xy = (1, 1) so the
        // MAX over a single coord equals the first-block read — no
        // behaviour change. AFV is single-block too (8x8 grid).
        let cov_x = (tile_w as usize / 8).max(1);
        let cov_y = (tile_h as usize / 8).max(1);
        let qac_for_strategy: Vec<f32> = coords
            .iter()
            .map(|&(bx, by)| {
                let mut q = qac_per_8x8_block[by * xsize_blocks_8 + bx];
                for dy in 0..cov_y {
                    for dx in 0..cov_x {
                        let bxx = bx + dx;
                        let byy = by + dy;
                        if bxx < xsize_blocks_8 && byy < ysize_blocks_8 {
                            let qq = qac_per_8x8_block[byy * xsize_blocks_8 + bxx];
                            if qq > q {
                                q = qq;
                            }
                        }
                    }
                }
                q
            })
            .collect();

        // Per-strategy quant matrix template (caller-supplied).
        let weights_template = weights_template_for_strategy(raw_strategy);
        debug_assert_eq!(
            weights_template.len(),
            coeff_count,
            "weights template for strategy {raw_strategy} must have {coeff_count} entries"
        );

        let n_blocks = coords.len();

        // AFV path: forward + inverse use forks::afv (per-kind), not
        // apply_dct/idct_batch_persistent. Quantize + dequant still use
        // the persistent variants. Scatter directly into out_plane;
        // skip pushing to coeff_buffers (reconstruct_mixed_strategy_gpu
        // can't handle AFV anyway).
        //
        // AFV needs a host batch because afv_transform_batch_gpu takes
        // a host slice — build it here on demand. Non-AFV strategies
        // skip this and use the GPU indexed_gather below.
        if is_afv(raw_strategy) {
            let block_pixels = (tile_w * tile_h) as usize;
            let mut batch = Vec::with_capacity(n_blocks * block_pixels);
            for &(bx, by) in &coords {
                let x0 = bx * 8;
                let y0 = by * 8;
                for dy in 0..tile_h as usize {
                    let src_off = (y0 + dy) * padded_width + x0;
                    batch.extend_from_slice(&xyb_channel[src_off..src_off + tile_w as usize]);
                }
            }
            let kind = (raw_strategy - RAW_STRATEGY_AFV0) as usize;
            // Forward AFV (host-orchestrated, returns Vec<f32>).
            let coeffs_host = crate::forks::afv::afv_transform_batch_gpu(
                enc,
                &crate::kernels::afv::AFV4X4_BASIS_TRANSPOSE,
                &batch,
                kind,
            );
            // Persistent quant + dequant (8×8 grid, 1×1 LLF — same as DCT8 shape).
            let g_coeffs = enc.upload_blocks(&coeffs_host, n_blocks as u32, 64);
            let g_quant = enc.quantize_large_blocks_broadcast_w_persistent(
                &g_coeffs,
                &weights_template,
                &qac_for_strategy,
                thresholds,
                8, 8, 1, 1,
            );
            let g_dequant = enc.dequant_strategy_persistent(
                &g_quant,
                &weights_template,
                &qac_for_strategy,
            );
            let mut dequant = enc.download_blocks(&g_dequant);
            // LLF restore for AFV: empirically (test_afv_packed_dc_for_uniform_input)
            // the forward AFV transform on uniform M input produces
            // packed coeffs[0] = M, coeffs[1] = 0, coeffs[8] = 0.
            // (The AFV 4×4 sub-DCT is orthonormal but the DCT4-corner
            // and DCT4×8 sub-DCTs are mean-scaled, and pack_afv_dcs's
            // 3-way mix produces M at coeffs[0] for uniform input.)
            //
            // So the LLF restore for AFV is simply coeffs[0] = mean —
            // exactly like DCT8. Positions [1] and [8] keep their
            // dequantized AC values from the forward pass.
            const AFV_PACKED_DC_MEAN_FACTOR: f32 = 1.0;
            for (i, &(bx, by)) in coords.iter().enumerate() {
                let mean = dc_grid_per_8x8_block[by * xsize_blocks_8 + bx];
                dequant[i * 64] = AFV_PACKED_DC_MEAN_FACTOR * mean;
            }
            // Inverse AFV (host-orchestrated).
            let pixels_host = crate::forks::afv::inverse_afv_transform_batch_gpu(
                enc,
                &crate::kernels::afv::AFV4X4_BASIS_TRANSPOSE,
                &dequant,
                kind,
            );
            // Scatter: 8×8 region per block (AFV uses 8×8 tiles).
            // When the caller routed output through a GpuPlane, upload
            // the AFV pixels and indexed_scatter on GPU so we don't
            // touch the host out_plane (which the caller may not
            // intend to consume); when None, host-scatter as before.
            if out_plane_gpu.is_some() {
                let g_pixels = enc.upload_blocks(&pixels_host, n_blocks as u32, 64);
                let coords_u32: Vec<(u32, u32)> = coords
                    .iter()
                    .map(|&(bx, by)| (bx as u32, by as u32))
                    .collect();
                enc.indexed_scatter_blocks_persistent(
                    &g_pixels, &coords_u32, g_out_plane, 8u32, 8u32,
                );
                used_gpu_for_any = true;
            } else {
                let out_host = out_plane
                    .as_deref_mut()
                    .expect("AFV needs out_plane (host) when out_plane_gpu is None");
                for (i, &(bx, by)) in coords.iter().enumerate() {
                    let src = &pixels_host[i * 64..i * 64 + 64];
                    scatter_block_to_plane(out_host, src, bx, by, raw_strategy, padded_width);
                }
            }
            continue;
        }
        // GPU gather: pull this strategy's per-block tiles from the
        // pre-uploaded plane via indexed_gather_blocks_persistent.
        // Replaces the host extend_from_slice loop + per-strategy
        // upload_blocks pair that the older code did. The GpuBlocks
        // layout matches the host batch byte-for-byte (verified by
        // test_indexed_gather_blocks_persistent_matches_host across 5
        // tile shapes).
        let coords_u32: Vec<(u32, u32)> =
            coords.iter().map(|&(bx, by)| (bx as u32, by as u32)).collect();
        let g_pixels = enc.indexed_gather_blocks_persistent(
            g_plane,
            &coords_u32,
            tile_w as u32,
            tile_h as u32,
        );

        // Step 2: persistent forward DCT → quantize → dequant chain.
        // For DCT8 (raw_strategy == RAW_STRATEGY_DCT) the forward DCT
        // and quantize collapse into one fused kernel — saves the
        // intermediate g_coeffs global-memory roundtrip (per-block
        // 256 bytes never written/read at the SM↔HBM boundary).
        // Bit-identical to the split chain — verified by
        // test_fused_dct8_quantize_persistent (per-block) and
        // test_fused_dct8_quantize_broadcast_w_matches_perblock
        // (broadcast-W bridge to the per-block test).
        let g_quant = if raw_strategy == RAW_STRATEGY_DCT {
            enc.dct8_quantize_fused_broadcast_w_persistent(
                &g_pixels,
                &weights_template,
                &qac_for_strategy,
                thresholds,
            )
        } else {
            let g_coeffs = apply_dct_batch_persistent(enc, &g_pixels, raw_strategy);
            enc.quantize_large_blocks_broadcast_w_persistent(
                &g_coeffs,
                &weights_template,
                &qac_for_strategy,
                thresholds,
                grid_w,
                grid_h,
                llf_x,
                llf_y,
            )
        };
        let g_dequant = if raw_strategy == RAW_STRATEGY_DCT {
            enc.dequant_strategy_dct8_persistent(
                &g_quant,
                &weights_template,
                &qac_for_strategy,
                channel,
            )
        } else {
            enc.dequant_strategy_persistent(&g_quant, &weights_template, &qac_for_strategy)
        };

        // 1×1-LLF non-AFV fast path: GPU LLF restore + GPU IDCT + GPU
        // scatter, fully bypassing the host download/LLF-loop/re-upload
        // round-trip the slower path below pays.
        //
        // Eligible strategies: DCT8, DCT4×4, DCT4×8, DCT8×4, IDENTITY,
        // DCT2X2 (every strategy whose llf_dim_x = llf_dim_y = 1, except
        // AFV which is handled by the dedicated branch above).
        if llf_x == 1 && llf_y == 1 {
            // DCT8 (the dominant strategy on real photos — ~99% of
            // blocks on most CLIC content) gets a single fused launch
            // for set_dc + IDCT8 + scatter, skipping the intermediate
            // g_recon GpuBlocks roundtrip (~256 bytes/block of HBM
            // traffic). Bit-identical to the split chain — proven by
            // test_idct_8x8_set_dc_scatter_matches_split.
            //
            // NOTE: A 4-way fused variant exists
            // (dequant_idct_dc_scatter_dct8_persistent) that ALSO
            // collapses the dequant step. Bit-identical and
            // unit-tested across all 3 channels, but paired A/B on
            // CLIC photo (10 runs each) showed +3.4% mixed_strategy_
            // encode_recon vs the 3-way version — likely register
            // pressure / occupancy from doing dequant + IDCT + scatter
            // in a single cube_dim=1 cube. Available as
            // `dequant_idct_dc_scatter_dct8_persistent` if a wide-cube
            // variant or a different occupancy-balanced design proves
            // out later.
            if raw_strategy == RAW_STRATEGY_DCT {
                enc.idct_8x8_set_dc_scatter_persistent(
                    &g_dequant,
                    g_dc_grid,
                    &coords_u32,
                    g_out_plane,
                    xsize_blocks_8 as u32,
                );
                used_gpu_for_any = true;
                continue;
            }

            // Other 1×1-LLF strategies (DCT4×4, DCT4×8, DCT8×4,
            // IDENTITY, DCT2x2) keep the split chain — each has its
            // own IDCT kernel, and at <1% of block share on photos
            // the fusion ROI doesn't justify per-strategy fused
            // variants yet.
            //
            // GPU LLF restore: writes dc_grid[by * stride + bx] into
            // position 0 of each block. AC positions untouched.
            enc.set_dc_from_grid_indexed_persistent(
                g_dc_grid,
                &coords_u32,
                &g_dequant,
                xsize_blocks_8 as u32,
            );
            // Persistent IDCT for this strategy + GPU scatter back into
            // g_out_plane at the strategy's tile positions.
            let g_recon = crate::forks::transform::apply_idct_batch_persistent(
                enc,
                &g_dequant,
                raw_strategy,
            );
            enc.indexed_scatter_blocks_persistent(
                &g_recon,
                &coords_u32,
                g_out_plane,
                tile_w as u32,
                tile_h as u32,
            );
            used_gpu_for_any = true;
            continue;
        }

        // 8×8 / 8×4 / 4×8-LLF GPU fast path for DCT64 family.
        use crate::forks::transform::{
            RAW_STRATEGY_DCT32X64, RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64,
        };
        if raw_strategy == RAW_STRATEGY_DCT64X64
            || raw_strategy == RAW_STRATEGY_DCT64X32
            || raw_strategy == RAW_STRATEGY_DCT32X64
        {
            if raw_strategy == RAW_STRATEGY_DCT64X64 {
                enc.set_llf_dct64x64_indexed_persistent(
                    g_dc_grid, &coords_u32, &g_dequant, xsize_blocks_8 as u32,
                );
            } else if raw_strategy == RAW_STRATEGY_DCT64X32 {
                enc.set_llf_dct64x32_indexed_persistent(
                    g_dc_grid, &coords_u32, &g_dequant, xsize_blocks_8 as u32,
                );
            } else {
                enc.set_llf_dct32x64_indexed_persistent(
                    g_dc_grid, &coords_u32, &g_dequant, xsize_blocks_8 as u32,
                );
            }
            let g_recon = crate::forks::transform::apply_idct_batch_persistent(
                enc,
                &g_dequant,
                raw_strategy,
            );
            enc.indexed_scatter_blocks_persistent(
                &g_recon,
                &coords_u32,
                g_out_plane,
                tile_w as u32,
                tile_h as u32,
            );
            used_gpu_for_any = true;
            continue;
        }

        // 4×2 / 2×4-LLF GPU fast path for DCT32×16 / DCT16×32.
        use crate::forks::transform::{RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT32X16};
        if raw_strategy == RAW_STRATEGY_DCT32X16 {
            enc.set_llf_dct32x16_indexed_persistent(
                g_dc_grid,
                &coords_u32,
                &g_dequant,
                xsize_blocks_8 as u32,
            );
            let g_recon = crate::forks::transform::apply_idct_batch_persistent(
                enc,
                &g_dequant,
                raw_strategy,
            );
            enc.indexed_scatter_blocks_persistent(
                &g_recon,
                &coords_u32,
                g_out_plane,
                tile_w as u32,
                tile_h as u32,
            );
            used_gpu_for_any = true;
            continue;
        }
        if raw_strategy == RAW_STRATEGY_DCT16X32 {
            enc.set_llf_dct16x32_indexed_persistent(
                g_dc_grid,
                &coords_u32,
                &g_dequant,
                xsize_blocks_8 as u32,
            );
            let g_recon = crate::forks::transform::apply_idct_batch_persistent(
                enc,
                &g_dequant,
                raw_strategy,
            );
            enc.indexed_scatter_blocks_persistent(
                &g_recon,
                &coords_u32,
                g_out_plane,
                tile_w as u32,
                tile_h as u32,
            );
            used_gpu_for_any = true;
            continue;
        }

        // 4×4-LLF GPU fast path for DCT32×32.
        use crate::forks::transform::RAW_STRATEGY_DCT32X32;
        if raw_strategy == RAW_STRATEGY_DCT32X32 {
            enc.set_llf_dct32x32_indexed_persistent(
                g_dc_grid,
                &coords_u32,
                &g_dequant,
                xsize_blocks_8 as u32,
            );
            let g_recon = crate::forks::transform::apply_idct_batch_persistent(
                enc,
                &g_dequant,
                raw_strategy,
            );
            enc.indexed_scatter_blocks_persistent(
                &g_recon,
                &coords_u32,
                g_out_plane,
                tile_w as u32,
                tile_h as u32,
            );
            used_gpu_for_any = true;
            continue;
        }

        // 2×2-LLF GPU fast path for DCT16×16.
        use crate::forks::transform::RAW_STRATEGY_DCT16X16;
        if raw_strategy == RAW_STRATEGY_DCT16X16 {
            enc.set_llf_dct16x16_indexed_persistent(
                g_dc_grid,
                &coords_u32,
                &g_dequant,
                xsize_blocks_8 as u32,
            );
            let g_recon = crate::forks::transform::apply_idct_batch_persistent(
                enc,
                &g_dequant,
                raw_strategy,
            );
            enc.indexed_scatter_blocks_persistent(
                &g_recon,
                &coords_u32,
                g_out_plane,
                tile_w as u32,
                tile_h as u32,
            );
            used_gpu_for_any = true;
            continue;
        }

        // 1×2 / 2×1-LLF GPU fast path for DCT16×8 / DCT8×16 — same chain
        // shape as the 1×1-LLF path above but with the matching
        // 2-position LLF kernel.
        use crate::forks::transform::{RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT8X16};
        if raw_strategy == RAW_STRATEGY_DCT16X8 || raw_strategy == RAW_STRATEGY_DCT8X16 {
            // dc_step = stride for vertical pair (DCT16x8), 1 for
            // horizontal pair (DCT8x16). dc_stride is xsize_blocks_8.
            let dc_step = if raw_strategy == RAW_STRATEGY_DCT16X8 {
                xsize_blocks_8 as u32
            } else {
                1u32
            };
            enc.set_llf_dct16x8_or_8x16_indexed_persistent(
                g_dc_grid,
                &coords_u32,
                &g_dequant,
                xsize_blocks_8 as u32,
                dc_step,
            );
            let g_recon = crate::forks::transform::apply_idct_batch_persistent(
                enc,
                &g_dequant,
                raw_strategy,
            );
            enc.indexed_scatter_blocks_persistent(
                &g_recon,
                &coords_u32,
                g_out_plane,
                tile_w as u32,
                tile_h as u32,
            );
            used_gpu_for_any = true;
            continue;
        }

        // Step 3: download dequant result for host LLF restore.
        // (LLF restore is per-block scalar work; deferring to host
        // avoids a separate GPU kernel per strategy. Future work could
        // fuse this into a single launch.)
        let mut dequant = enc.download_blocks(&g_dequant);
        debug_assert_eq!(dequant.len(), n_blocks * coeff_count);

        // Step 4: per-block LLF restore. Pull this strategy's
        // (llf_x × llf_y) DC values from dc_grid_per_8x8_block at the
        // first-block position and pass to dispatch_restore_llf.
        let llf_count = (llf_x as usize) * (llf_y as usize);
        let mut dc_subgrid = vec![0.0_f32; llf_count];
        for (i, &(bx, by)) in coords.iter().enumerate() {
            // Gather 8×8 DC values for this strategy's footprint.
            for dy in 0..llf_y as usize {
                for dx in 0..llf_x as usize {
                    dc_subgrid[dy * (llf_x as usize) + dx] =
                        dc_grid_per_8x8_block[(by + dy) * xsize_blocks_8 + (bx + dx)];
                }
            }
            let block_off = i * coeff_count;
            dispatch_restore_llf(
                &mut dequant[block_off..block_off + coeff_count],
                &dc_subgrid,
                raw_strategy,
            );
        }

        coeff_buffers.push((raw_strategy, coords, dequant));
    }

    // Step 5: build BlockRecipe references into the per-strategy buffers.
    let mut recipes: Vec<BlockRecipe<'_>> = Vec::with_capacity(assignments.len());
    for (raw_strategy, coords, buf) in coeff_buffers.iter() {
        let coeff_count = coeff_count_per_strategy(*raw_strategy);
        for (i, &(bx, by)) in coords.iter().enumerate() {
            recipes.push(BlockRecipe {
                bx,
                by,
                raw_strategy: *raw_strategy,
                coeffs: &buf[i * coeff_count..(i + 1) * coeff_count],
            });
        }
    }

    // Step 6: dispatch mixed-strategy reconstruct (host LLF + AFV path).
    // In production, every strategy goes through one of the GPU fast
    // paths above and `recipes` is empty. We keep this call for any
    // future strategy that's added to the dispatcher but doesn't yet
    // have a GPU LLF kernel — it would land in coeff_buffers and need
    // the host fallback. The unwrap is safe because, by the entry
    // assert, out_plane must be Some when out_plane_gpu is None;
    // recipes is non-empty only when the host LLF fallback ran, which
    // implies neither GPU path nor AFV's host scatter took it (so
    // out_plane was never used) — but the caller still needs to have
    // supplied it for those hypothetical strategies.
    if !recipes.is_empty() {
        let out_host = out_plane.as_deref_mut().expect(
            "encode_and_reconstruct: out_plane must be Some when host LLF fallback fires",
        );
        reconstruct_mixed_strategy_gpu(enc, &recipes, out_host, padded_width);
    }

    // Step 7: merge GPU-resident scatter results back into out_plane.
    // The 1×1-LLF non-AFV fast path scatters its IDCT outputs into
    // g_out_plane on GPU; here we download once and add into the host
    // out_plane. Disjoint per-pixel coverage (AC strategy partitioning)
    // means addition equals union with no overlap. Skip the download
    // entirely if no GPU-fast-path strategy fired this call OR if the
    // caller supplied an out_plane_gpu (in which case the strategies
    // already scattered directly into the caller's buffer — no merge
    // needed and the host out_plane was untouched).
    if used_gpu_for_any && out_plane_gpu.is_none() {
        // Caller is on the legacy host path — out_plane must be Some
        // here (the entry-side debug_assert above guarantees it).
        let out_host = out_plane
            .as_deref_mut()
            .expect("merge-back requires out_plane (host) when out_plane_gpu is None");
        let gpu_out = enc.download_plane(g_out_plane);
        debug_assert_eq!(gpu_out.len(), out_host.len());
        for (o, g) in out_host.iter_mut().zip(gpu_out.iter()) {
            *o += *g;
        }
    }
}

/// 3-channel wrapper around [`encode_and_reconstruct_mixed_strategy_single_channel`].
///
/// All three channels share the same `assignments` and `dc_grid`
/// shape (matches upstream JXL: AC strategy is picked per-Y-channel
/// then applied to all 3 channels). Each channel has its own
/// `weights_template_for_strategy` closure (per-channel quant
/// matrix), qac field, thresholds, and DC grid.
///
/// Calls the single-channel helper 3 times serially. A future perf
/// optimization could batch the per-channel forward DCT calls into
/// a single launch per strategy (one DCT+quant+dequant per channel
/// per strategy → one fused 3-channel kernel per strategy), but
/// that's a separate kernel-level change deferred to perf phase.
#[allow(clippy::too_many_arguments)]
pub fn encode_and_reconstruct_mixed_strategy_3channel<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    padded_width: usize,
    padded_height: usize,
    assignments: &[crate::pipeline::StrategyAssignment],
    weights_x_for_strategy: &dyn Fn(u8) -> Vec<f32>,
    weights_y_for_strategy: &dyn Fn(u8) -> Vec<f32>,
    weights_b_for_strategy: &dyn Fn(u8) -> Vec<f32>,
    qac_x_per_8x8_block: &[f32],
    qac_y_per_8x8_block: &[f32],
    qac_b_per_8x8_block: &[f32],
    thresholds_x: &[f32; 4],
    thresholds_y: &[f32; 4],
    thresholds_b: &[f32; 4],
    dc_grid_x_per_8x8_block: &[f32],
    dc_grid_y_per_8x8_block: &[f32],
    dc_grid_b_per_8x8_block: &[f32],
    out_plane_x: Option<&mut [f32]>,
    out_plane_y: Option<&mut [f32]>,
    out_plane_b: Option<&mut [f32]>,
    // Optional pre-uploaded GPU planes mirroring xyb_x / xyb_y / xyb_b.
    // See `encode_and_reconstruct_mixed_strategy_single_channel`'s
    // `xyb_channel_gpu` parameter for the rationale.
    xyb_x_gpu: Option<&crate::persistent::GpuPlane<R>>,
    xyb_y_gpu: Option<&crate::persistent::GpuPlane<R>>,
    xyb_b_gpu: Option<&crate::persistent::GpuPlane<R>>,
    // Optional pre-uploaded GPU dc_grids mirroring
    // dc_grid_x/y/b_per_8x8_block. See `_single_channel`'s
    // `dc_grid_gpu` for the rationale.
    dc_grid_x_gpu: Option<&crate::persistent::GpuBlocks<R>>,
    dc_grid_y_gpu: Option<&crate::persistent::GpuBlocks<R>>,
    dc_grid_b_gpu: Option<&crate::persistent::GpuBlocks<R>>,
    // Optional GPU output planes mirroring out_plane_x / out_plane_y /
    // out_plane_b. See `_single_channel`'s `out_plane_gpu` for the
    // rationale (skip merge-back, scatter directly into caller's
    // GpuPlane so downstream postpass can chain without re-uploading).
    out_plane_x_gpu: Option<&crate::persistent::GpuPlane<R>>,
    out_plane_y_gpu: Option<&crate::persistent::GpuPlane<R>>,
    out_plane_b_gpu: Option<&crate::persistent::GpuPlane<R>>,
) {
    encode_and_reconstruct_mixed_strategy_single_channel(
        enc,
        xyb_x,
        padded_width,
        padded_height,
        assignments,
        weights_x_for_strategy,
        qac_x_per_8x8_block,
        thresholds_x,
        dc_grid_x_per_8x8_block,
        0,
        out_plane_x,
        xyb_x_gpu,
        dc_grid_x_gpu,
        out_plane_x_gpu,
    );
    encode_and_reconstruct_mixed_strategy_single_channel(
        enc,
        xyb_y,
        padded_width,
        padded_height,
        assignments,
        weights_y_for_strategy,
        qac_y_per_8x8_block,
        thresholds_y,
        dc_grid_y_per_8x8_block,
        1,
        out_plane_y,
        xyb_y_gpu,
        dc_grid_y_gpu,
        out_plane_y_gpu,
    );
    encode_and_reconstruct_mixed_strategy_single_channel(
        enc,
        xyb_b,
        padded_width,
        padded_height,
        assignments,
        weights_b_for_strategy,
        qac_b_per_8x8_block,
        thresholds_b,
        dc_grid_b_per_8x8_block,
        2,
        out_plane_b,
        xyb_b_gpu,
        dc_grid_b_gpu,
        out_plane_b_gpu,
    );
}

pub fn reconstruct_mixed_strategy_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    recipes: &[BlockRecipe<'_>],
    plane: &mut [f32],
    padded_width: usize,
) {
    use crate::forks::transform::coeff_count_per_strategy;

    if recipes.is_empty() {
        return;
    }

    // Group recipe indices by raw_strategy. We use a fixed-size
    // lookup array sized to cover all strategy codes the dispatcher
    // recognizes (0..=16). AFV codes aren't in this range.
    const NUM_STRATEGY_CODES: usize = 17;
    let mut groups: [Vec<usize>; NUM_STRATEGY_CODES] = core::array::from_fn(|_| Vec::new());
    for (i, r) in recipes.iter().enumerate() {
        let s = r.raw_strategy as usize;
        if s >= NUM_STRATEGY_CODES {
            panic!(
                "reconstruct_mixed_strategy_gpu: strategy {s} out of range \
                 (use forks::afv for AFV0-3)"
            );
        }
        groups[s].push(i);
    }

    // For each populated group, batch the coeff buffers and dispatch.
    for (strategy_code, indices) in groups.iter().enumerate() {
        if indices.is_empty() {
            continue;
        }
        let raw_strategy = strategy_code as u8;
        let coeff_count = coeff_count_per_strategy(raw_strategy);
        let mut batched = Vec::with_capacity(indices.len() * coeff_count);
        let mut coords = Vec::with_capacity(indices.len());
        for &i in indices {
            let r = &recipes[i];
            debug_assert_eq!(
                r.coeffs.len(),
                coeff_count,
                "recipe {i}: coeffs.len() = {} but strategy {raw_strategy} expects {coeff_count}",
                r.coeffs.len()
            );
            batched.extend_from_slice(r.coeffs);
            coords.push((r.bx, r.by));
        }
        batched_reconstruct_same_strategy_gpu(
            enc,
            &batched,
            &coords,
            raw_strategy,
            plane,
            padded_width,
        );
    }
}

/// Batched IDCT + scatter for many blocks of the SAME AC strategy.
/// One GPU launch for the IDCT regardless of `n_blocks`, then a per-
/// block host-side scatter.
///
/// `coeffs_batched.len()` must equal `n_blocks * coeff_count`, where
/// `coeff_count = forks::transform::coeff_count_per_strategy(raw_strategy)`
/// (DCT8 → 64, DCT16×16 → 256, DCT32×32 → 1024, DCT64×64 → 4096, etc.).
/// `block_coords[i] = (bx, by)` is where the i-th block's coefficients
/// (slice `[i * coeff_count .. (i+1) * coeff_count]`) should land in
/// the padded plane.
///
/// This is the efficient form for mixed-strategy reconstruct: the
/// caller groups blocks by strategy and calls this once per group.
/// Per-block GPU dispatch overhead is amortized across N blocks.
pub fn batched_reconstruct_same_strategy_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    coeffs_batched: &[f32],
    block_coords: &[(usize, usize)],
    raw_strategy: u8,
    plane: &mut [f32],
    padded_width: usize,
) {
    use crate::forks::transform::{apply_idct_batch_gpu, coeff_count_per_strategy};

    if block_coords.is_empty() {
        return;
    }
    let coeff_count = coeff_count_per_strategy(raw_strategy);
    debug_assert_eq!(coeffs_batched.len(), block_coords.len() * coeff_count);

    // One GPU launch covers ALL blocks of this strategy.
    let pixels_batched = apply_idct_batch_gpu(enc, coeffs_batched, raw_strategy);

    let (block_w, block_h) = crate::forks::transform::tile_dims_pixels(raw_strategy);
    let block_pixels = block_w * block_h;
    debug_assert_eq!(pixels_batched.len(), block_coords.len() * block_pixels);

    // Per-block host-side scatter (cheap memcpy).
    for (i, &(bx, by)) in block_coords.iter().enumerate() {
        let src = &pixels_batched[i * block_pixels..(i + 1) * block_pixels];
        scatter_block_to_plane(plane, src, bx, by, raw_strategy, padded_width);
    }
}

/// IDCT a single block's coefficient buffer and scatter the resulting
/// pixels into the padded plane at `(bx * 8, by * 8)`. Composes
/// [`crate::forks::transform::apply_idct_batch_gpu`] (with batch
/// size 1) and [`scatter_block_to_plane`].
///
/// `coeffs.len()` must match the strategy's full coefficient block
/// size (DCT8 → 64, DCT16×16 → 256, etc.). The caller is responsible
/// for filling `coeffs` with already-dequantized + CfL-corrected +
/// LLF-restored coefficients before calling this.
///
/// Note: per-block GPU IDCT is wasteful when many blocks share a
/// strategy. Use [`crate::forks::transform::apply_idct_batch_gpu`]
/// directly with a batched coefficient buffer + multiple
/// `scatter_block_to_plane` calls when batching is possible. This
/// per-block form is the simplest building block for a future
/// strategy-grouping orchestrator.
pub fn idct_and_scatter_one_block_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    coeffs: &[f32],
    plane: &mut [f32],
    bx: usize,
    by: usize,
    raw_strategy: u8,
    padded_width: usize,
) {
    use crate::forks::transform::apply_idct_batch_gpu;
    let pixels = apply_idct_batch_gpu(enc, coeffs, raw_strategy);
    scatter_block_to_plane(plane, &pixels, bx, by, raw_strategy, padded_width);
}

/// Scatter a single block's IDCT output (in pixel layout, row-major,
/// stride = block_width) into the padded plane at the block's pixel
/// position `(bx * 8, by * 8)`.
///
/// `block_pixels` must contain `block_w * block_h` floats where
/// `(block_w, block_h) = forks::transform::tile_dims_pixels(raw_strategy)`.
///
/// Mirrors the per-strategy scatter loop in upstream's
/// `reconstruct_xyb_impl` (reconstruct.rs lines 459-469): writes
/// `block_w` floats per row for `block_h` rows into the destination
/// plane, advancing `padded_width` per row.
///
/// `plane.len()` must be at least `padded_width * (by * 8 + block_h)`
/// (the plane should be the full padded XYB plane for that channel).
pub fn scatter_block_to_plane(
    plane: &mut [f32],
    block_pixels: &[f32],
    bx: usize,
    by: usize,
    raw_strategy: u8,
    padded_width: usize,
) {
    let (block_w, block_h) = crate::forks::transform::tile_dims_pixels(raw_strategy);
    debug_assert_eq!(block_pixels.len(), block_w * block_h);
    let pixel_x = bx * 8;
    let pixel_y = by * 8;
    for row in 0..block_h {
        let dst_off = (pixel_y + row) * padded_width + pixel_x;
        let src_off = row * block_w;
        plane[dst_off..dst_off + block_w]
            .copy_from_slice(&block_pixels[src_off..src_off + block_w]);
    }
}

/// Fast 1D IDCT for N=8 — bit-for-bit port of upstream
/// `vardct::dct::inverse::idct1d_8`. Pure scalar; matched inverse
/// of our private `dct1d_8` (i.e., `idct1d_8(dct1d_8(x)) == x`).
///
/// Includes the `*= 8` scaling to compensate for the `1/8` scaling
/// applied by upstream's `dct_8x8` wrapper.
#[allow(clippy::approx_constant)] // upstream literal SQRT2 = 1.41421356237
fn idct1d_8(mem: &mut [f32]) {
    const INV_SQRT2: f32 = 1.0 / 1.414_213_5;
    const INV_WC4: [f32; 2] = [1.0 / 0.541_196_1, 1.0 / 1.306_563_0];
    const INV_WC8: [f32; 4] = [
        1.0 / 0.509_795_6,
        1.0 / 0.601_344_9,
        1.0 / 0.899_976_2,
        1.0 / 2.562_915_4,
    ];

    // Scale by 8 to compensate for forward's 1/8.
    let m = [
        mem[0] * 8.0,
        mem[1] * 8.0,
        mem[2] * 8.0,
        mem[3] * 8.0,
        mem[4] * 8.0,
        mem[5] * 8.0,
        mem[6] * 8.0,
        mem[7] * 8.0,
    ];

    // De-interleave: even = [m[0],m[2],m[4],m[6]], odd = [m[1],m[3],m[5],m[7]].
    let (e0, e1, e2, e3) = (m[0], m[2], m[4], m[6]);
    let (mut o0, mut o1, mut o2, o3) = (m[1], m[3], m[5], m[7]);

    // Reverse B transform on odd half.
    o2 -= o3;
    o1 -= o2;
    o0 = (o0 - o1) * INV_SQRT2;

    // Reverse idct1d_4_val on odd half (uses idct1d_4_val signature
    // from upstream — caller pre-orders args so internal de-interleave
    // does the right thing).
    let idct4_val = |a: f32, b: f32, c: f32, d: f32| -> [f32; 4] {
        let odd0 = (c - d) * INV_SQRT2;
        let o0p = (odd0 + d) * 0.5;
        let o1p = (odd0 - d) * 0.5;
        let o0 = o0p * INV_WC4[0];
        let o1 = o1p * INV_WC4[1];
        let e0_l = (a + b) * 0.5;
        let e1_l = (a - b) * 0.5;
        [
            (e0_l + o0) * 0.5,
            (e1_l + o1) * 0.5,
            (e1_l - o1) * 0.5,
            (e0_l - o0) * 0.5,
        ]
    };
    let odd = idct4_val(o0, o2, o1, o3);
    let o = [
        odd[0] * INV_WC8[0],
        odd[1] * INV_WC8[1],
        odd[2] * INV_WC8[2],
        odd[3] * INV_WC8[3],
    ];

    // Reverse idct1d_4 on even half.
    let e = idct4_val(e0, e2, e1, e3);

    // Combine even/odd.
    mem[0] = (e[0] + o[0]) * 0.5;
    mem[1] = (e[1] + o[1]) * 0.5;
    mem[2] = (e[2] + o[2]) * 0.5;
    mem[3] = (e[3] + o[3]) * 0.5;
    mem[4] = (e[3] - o[3]) * 0.5;
    mem[5] = (e[2] - o[2]) * 0.5;
    mem[6] = (e[1] - o[1]) * 0.5;
    mem[7] = (e[0] - o[0]) * 0.5;
}

/// Forward DC extraction for DCT64×64 — inverse of
/// [`restore_llf_dct64x64`]. Mirrors upstream's
/// `dc_from_dct_64x64` (forward_large.rs:315).
///
/// Takes the 8×8 LLF coefficient grid (positions
/// `coeffs[iy * 64 + ix]` for `iy, ix in 0..8`) and returns the
/// 8×8 DC grid. Applies SCALE_64_TO_8 in both dims, then 8×8 IDCT
/// (idct1d_8 rows × 2 with transpose).
pub fn dc_from_dct_64x64(llf_grid: [f32; 64]) -> [f32; 64] {
    let mut block = [0.0_f32; 64];
    for iy in 0..8 {
        for ix in 0..8 {
            block[iy * 8 + ix] = llf_grid[iy * 8 + ix]
                * DCT_RESAMPLE_SCALE_64_TO_8[iy]
                * DCT_RESAMPLE_SCALE_64_TO_8[ix];
        }
    }
    for iy in 0..8 {
        idct1d_8(&mut block[iy * 8..(iy + 1) * 8]);
    }
    let mut transposed = [0.0_f32; 64];
    for iy in 0..8 {
        for ix in 0..8 {
            transposed[ix * 8 + iy] = block[iy * 8 + ix];
        }
    }
    for iy in 0..8 {
        idct1d_8(&mut transposed[iy * 8..(iy + 1) * 8]);
    }
    transposed
}

/// Forward DC extraction for DCT64×32 — inverse of
/// [`restore_llf_dct64x32`]. Mirrors upstream
/// `dc_from_dct_64x32` (forward_large.rs:364).
pub fn dc_from_dct_64x32(llf_grid: [f32; 32]) -> [f32; 32] {
    // llf_grid[iy*8+ix] for iy in 0..4, ix in 0..8 (matches restore_llf output).
    let mut block = [0.0_f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            block[iy * 8 + ix] = llf_grid[iy * 8 + ix]
                * DCT_RESAMPLE_SCALE_32_TO_4[iy]
                * DCT_RESAMPLE_SCALE_64_TO_8[ix]
                * 4.0;
        }
    }
    // IDCT on 8-element rows (4 rows).
    for iy in 0..4 {
        idct1d_8(&mut block[iy * 8..(iy + 1) * 8]);
    }
    // Transpose 4×8 → 8×4.
    let mut t = [0.0_f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            t[ix * 4 + iy] = block[iy * 8 + ix];
        }
    }
    // IDCT on 4-element rows (8 rows).
    for iy in 0..8 {
        idct1d_4(&mut t[iy * 4..(iy + 1) * 4]);
    }
    t
}

/// Forward DC extraction for DCT32×64 — inverse of
/// [`restore_llf_dct32x64`]. Mirrors upstream
/// `dc_from_dct_32x64` (forward_large.rs:410).
pub fn dc_from_dct_32x64(llf_grid: [f32; 32]) -> [f32; 32] {
    let mut block = [0.0_f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            block[iy * 8 + ix] = llf_grid[iy * 8 + ix]
                * DCT_RESAMPLE_SCALE_32_TO_4[iy]
                * DCT_RESAMPLE_SCALE_64_TO_8[ix]
                * 4.0;
        }
    }
    // IDCT on 8-element rows (4 rows).
    for iy in 0..4 {
        idct1d_8(&mut block[iy * 8..(iy + 1) * 8]);
    }
    // Transpose 4×8 → 8×4.
    let mut t = [0.0_f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            t[ix * 4 + iy] = block[iy * 8 + ix];
        }
    }
    // IDCT on 4-element rows (8 rows).
    for iy in 0..8 {
        idct1d_4(&mut t[iy * 4..(iy + 1) * 4]);
    }
    // Transpose back 8×4 → 4×8.
    let mut result = [0.0_f32; 32];
    for iy in 0..8 {
        for ix in 0..4 {
            result[ix * 8 + iy] = t[iy * 4 + ix];
        }
    }
    result
}

/// Fast 1D IDCT for N=4 — bit-for-bit port of upstream
/// `vardct::dct::inverse::idct1d_4`. Pure scalar; matched inverse of
/// our private `dct1d_4` (i.e., `idct1d_4(dct1d_4(x)) == x`).
#[allow(clippy::approx_constant)] // upstream literal INV_SQRT2 = 1/1.41421356237
fn idct1d_4(mem: &mut [f32]) {
    const INV_SQRT2: f32 = 1.0 / 1.414_213_5;
    const INV_WC4: [f32; 2] = [1.0 / 0.541_196_1, 1.0 / 1.306_563_0];
    let (a, b, c, d) = (mem[0], mem[2], mem[1], mem[3]); // de-interleave
    // Reverse B transform on odd half.
    let odd0 = (c - d) * INV_SQRT2;
    // Reverse idct1d_2 on odd half.
    let o0_pre = (odd0 + d) * 0.5;
    let o1_pre = (odd0 - d) * 0.5;
    // Reverse Wc multipliers (multiply by reciprocal).
    let o0 = o0_pre * INV_WC4[0];
    let o1 = o1_pre * INV_WC4[1];
    // Reverse idct1d_2 on even half.
    let e0 = (a + b) * 0.5;
    let e1 = (a - b) * 0.5;
    mem[0] = (e0 + o0) * 0.5;
    mem[1] = (e1 + o1) * 0.5;
    mem[2] = (e1 - o1) * 0.5;
    mem[3] = (e0 - o0) * 0.5;
}

/// Forward DC extraction for DCT32×32 — inverse of
/// [`restore_llf_dct32x32`]. Mirrors upstream's
/// `dc_from_dct_32x32` (forward_large.rs:68).
///
/// Takes the 16 LLF coefficient values from a fully-populated 32×32
/// coefficient block (positions `coeffs[iy * 32 + ix]` for
/// `iy, ix in 0..4`) and returns the 4×4 DC grid that the encoder
/// would store. Inputs are in the same order as
/// `restore_llf_dct32x32`'s output.
///
/// Math (forward direction):
/// ```text
///   block[iy*4+ix] = llf[iy*4+ix] * SCALE_32_TO_4[iy] * SCALE_32_TO_4[ix] * 16.0
///   4x4 IDCT: idct1d_4(rows), transpose, idct1d_4(rows)
/// ```
pub fn dc_from_dct_32x32(llf_grid: [f32; 16]) -> [f32; 16] {
    let mut block = [0.0_f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            block[iy * 4 + ix] = llf_grid[iy * 4 + ix]
                * DCT_RESAMPLE_SCALE_32_TO_4[iy]
                * DCT_RESAMPLE_SCALE_32_TO_4[ix]
                * 16.0;
        }
    }
    // 4x4 IDCT: idct1d_4 on rows, transpose, idct1d_4 on rows.
    for iy in 0..4 {
        idct1d_4(&mut block[iy * 4..(iy + 1) * 4]);
    }
    let mut transposed = [0.0_f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            transposed[ix * 4 + iy] = block[iy * 4 + ix];
        }
    }
    for iy in 0..4 {
        idct1d_4(&mut transposed[iy * 4..(iy + 1) * 4]);
    }
    transposed
}

/// Forward DC extraction for DCT32×16 — inverse of
/// [`restore_llf_dct32x16`]. Mirrors upstream
/// `dc_from_dct_32x16` (forward_large.rs:139).
///
/// Takes 8 LLF values (the 2×4 LLF block at positions
/// `coeffs[iy * 32 + ix]` for `iy in 0..2, ix in 0..4`) and returns
/// the 4×2 DC grid in row-major (`dc[iy * 2 + ix]` for
/// `iy in 0..4, ix in 0..2`).
///
/// Math: scale + compensation × 4, IDCT 4-pt on rows, transpose,
/// 2-pt IDCT (unnormalized; the ×4 compensation factor absorbs the
/// `*0.5` that idct1d_2 would otherwise apply).
pub fn dc_from_dct_32x16(llf_grid: [f32; 8]) -> [f32; 8] {
    // llf_grid[iy*4+ix] for iy in 0..2, ix in 0..4 (matches restore_llf_dct32x16 output).
    let mut block = [0.0_f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            block[iy * 4 + ix] = llf_grid[iy * 4 + ix]
                * DCT_RESAMPLE_SCALE_16_TO_2[iy]
                * DCT_RESAMPLE_SCALE_32_TO_4[ix]
                * 4.0;
        }
    }
    // IDCT on 4-element rows (2 rows).
    idct1d_4(&mut block[0..4]);
    idct1d_4(&mut block[4..8]);
    // Transpose 2×4 → 4×2.
    let mut t = [0.0_f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            t[ix * 2 + iy] = block[iy * 4 + ix];
        }
    }
    // Unnormalized 2-pt IDCT on each of 4 rows of 2 (no *0.5).
    for iy in 0..4 {
        let a = t[iy * 2];
        let b = t[iy * 2 + 1];
        t[iy * 2] = a + b;
        t[iy * 2 + 1] = a - b;
    }
    t
}

/// Forward DC extraction for DCT16×32 — inverse of
/// [`restore_llf_dct16x32`]. Mirrors upstream
/// `dc_from_dct_16x32` (the ROWS<COLS branch, transpose).
///
/// Takes 8 LLF values (the 2×4 LLF block at positions
/// `coeffs[iy * 32 + ix]` for `iy in 0..2, ix in 0..4`) and returns
/// the 2×4 DC grid in row-major (`dc[iy * 4 + ix]`).
///
/// Math: scale + compensation × 4, 2-pt IDCT (unnormalized) on rows
/// of 2 after transpose, then IDCT 4-pt on rows of 4. Transposes
/// match the inverse of `restore_llf_dct16x32`'s row-major path.
pub fn dc_from_dct_16x32(llf_grid: [f32; 8]) -> [f32; 8] {
    let mut block = [0.0_f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            block[iy * 4 + ix] = llf_grid[iy * 4 + ix]
                * DCT_RESAMPLE_SCALE_16_TO_2[iy]
                * DCT_RESAMPLE_SCALE_32_TO_4[ix]
                * 4.0;
        }
    }
    // Transpose 2x4 → 4x2.
    let mut t = [0.0_f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            t[ix * 2 + iy] = block[iy * 4 + ix];
        }
    }
    // Unnormalized 2-pt IDCT on each of 4 rows of 2.
    for iy in 0..4 {
        let a = t[iy * 2];
        let b = t[iy * 2 + 1];
        t[iy * 2] = a + b;
        t[iy * 2 + 1] = a - b;
    }
    // Transpose back 4x2 → 2x4.
    let mut result = [0.0_f32; 8];
    for iy in 0..4 {
        for ix in 0..2 {
            result[ix * 4 + iy] = t[iy * 2 + ix];
        }
    }
    // IDCT on 4-element rows (2 rows).
    idct1d_4(&mut result[0..4]);
    idct1d_4(&mut result[4..8]);
    result
}

/// Forward DC extraction for DCT16×16 — bit-for-bit inverse of
/// [`restore_llf_dct16x16`]. Mirrors upstream
/// `vardct::dct::forward_large::dc_from_dct_16x16`.
///
/// Takes the 4 LLF coefficient values from a fully-populated 16×16
/// coefficient block (positions `[0, 1, 16, 17]`) and returns the
/// 2×2 DC grid that the encoder would store.
///
/// Used here for the roundtrip parity test
/// `forward(restore(dc)) == dc` and `restore(forward(llf)) == llf`,
/// which validates `restore_llf_dct16x16` more strongly than the
/// constant-DC property test alone (this catches sign / scale /
/// transpose inversions that constant-DC misses).
///
/// Math (forward direction; per upstream comments in
/// `restore_llf_from_dc`):
/// ```text
///   temp[iy,ix] = llf[iy,ix] * s_iy * s_ix * 4.0
///   dc_grid = 2x2_IDCT(temp) where 2x2_IDCT == H/4
/// ```
/// `H` is the unnormalized 2×2 Hadamard (H*H = 4*I).
pub fn dc_from_dct_16x16(llf_grid: [f32; 4]) -> [f32; 4] {
    let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
    let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];
    // Apply per-position scale + 4.0 factor.
    let t00 = llf_grid[0] * s0 * s0 * 4.0;
    let t01 = llf_grid[1] * s0 * s1 * 4.0;
    let t10 = llf_grid[2] * s1 * s0 * 4.0;
    let t11 = llf_grid[3] * s1 * s1 * 4.0;
    // 2x2 IDCT = H/4.
    [
        (t00 + t01 + t10 + t11) / 4.0,
        (t00 + t01 - t10 - t11) / 4.0,
        (t00 - t01 + t10 - t11) / 4.0,
        (t00 - t01 - t10 + t11) / 4.0,
    ]
}

/// Forward DC extraction for DCT16×8 / DCT8×16 — inverse of
/// [`restore_llf_dct16x8_or_8x16`]. Mirrors upstream's
/// `dc_from_dct_16x8` / `dc_from_dct_8x16` LLF extraction step.
///
/// Takes 2 LLF coefficient values (positions `[0, 1]`) and returns
/// the pair of DC values (vertical for DCT16×8, horizontal for
/// DCT8×16).
///
/// Math:
/// ```text
///   dc[0] = llf[0] * s0 + llf[1] * s1
///   dc[1] = llf[0] * s0 - llf[1] * s1
/// ```
pub fn dc_from_dct_16x8_or_8x16(llf0: f32, llf1: f32) -> (f32, f32) {
    let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
    let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];
    let t0 = llf0 * s0;
    let t1 = llf1 * s1;
    (t0 + t1, t0 - t1)
}

/// Per-strategy LLF dispatcher. Given the dequantized DC grid for a
/// single block of the given AC strategy, calls the matching
/// `restore_llf_*` helper and writes the resulting LLF coefficients
/// into the right positions of `coeffs`.
///
/// Mirrors the body of upstream's `restore_llf_from_dc` per arm
/// — but factored so the per-strategy DCT math lives in the
/// individual `restore_llf_*` helpers.
///
/// `dc_grid` layout per strategy (row-major within the sub-block grid):
/// - DCT8 / DCT4×4 / DCT4×8 / DCT8×4 / IDENTITY / DCT2X2 / AFV0-3:
///   `[dc]` (single value).
/// - DCT16×8: `[dc(by, bx), dc(by+1, bx)]` (vertical pair).
/// - DCT8×16: `[dc(by, bx), dc(by, bx+1)]` (horizontal pair).
/// - DCT16×16: 2×2 grid `[(0,0), (0,1), (1,0), (1,1)]`.
/// - DCT32×16: 4×2 grid (`iy * 2 + ix`).
/// - DCT16×32: 2×4 grid (`iy * 4 + ix`).
/// - DCT32×32: 4×4 grid (`iy * 4 + ix`).
/// - DCT64×32: 8×4 grid (`iy * 4 + ix`).
/// - DCT32×64: 4×8 grid (`iy * 8 + ix`).
/// - DCT64×64: 8×8 grid (`iy * 8 + ix`).
///
/// `coeffs.len()` must match the strategy's full coefficient block
/// size (DCT8 → 64, DCT16×16 → 256, DCT32×32 → 1024, DCT64×64 →
/// 4096, etc.) — the dispatcher writes only the LLF positions and
/// leaves AC positions untouched.
///
/// Returns the LLF coefficient stride (= columns of the coefficient
/// block — 8 for square DCT8, 16 for DCT16×16, 32 for DCT32×*, 64
/// for DCT64×*) so callers can walk the LLF positions if they need
/// to (the function itself already has).
pub fn dispatch_restore_llf(coeffs: &mut [f32], dc_grid: &[f32], raw_strategy: u8) -> usize {
    use crate::forks::transform::{
        RAW_STRATEGY_DCT, RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8,
        RAW_STRATEGY_DCT8X4, RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16,
        RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64,
        RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64, RAW_STRATEGY_IDENTITY,
    };

    match raw_strategy {
        RAW_STRATEGY_DCT
        | RAW_STRATEGY_DCT4X4
        | RAW_STRATEGY_DCT4X8
        | RAW_STRATEGY_DCT8X4
        | RAW_STRATEGY_IDENTITY
        | RAW_STRATEGY_DCT2X2 => {
            // 1×1 LLF: just the single DC at position [0].
            coeffs[0] = dc_grid[0];
            8
        }
        RAW_STRATEGY_DCT16X8 | RAW_STRATEGY_DCT8X16 => {
            let llf = restore_llf_dct16x8_or_8x16(dc_grid[0], dc_grid[1]);
            // Coefficient layout: 16×8 = 128 coeffs, stride = 8 cols
            // (DCT16×8) or 16 (DCT8×16). LLF goes at [0] and [1] for both.
            coeffs[0] = llf[0];
            coeffs[1] = llf[1];
            // For DCT16×8 the post-swap layout is (cy=2, cx=1) → stride 8.
            // For DCT8×16 the post-swap layout is (cy=1, cx=2) → stride 16.
            if raw_strategy == RAW_STRATEGY_DCT8X16 {
                16
            } else {
                8
            }
        }
        RAW_STRATEGY_DCT16X16 => {
            let llf = restore_llf_dct16x16([dc_grid[0], dc_grid[1], dc_grid[2], dc_grid[3]]);
            // 16×16 layout: stride 16. LLF positions [0, 1, 16, 17].
            coeffs[0] = llf[0];
            coeffs[1] = llf[1];
            coeffs[16] = llf[2];
            coeffs[17] = llf[3];
            16
        }
        RAW_STRATEGY_DCT32X32 => {
            let mut grid = [0.0_f32; 16];
            grid.copy_from_slice(&dc_grid[..16]);
            let llf = restore_llf_dct32x32(grid);
            // 32×32 layout: stride 32. LLF positions [iy * 32 + ix] for iy, ix in 0..4.
            for iy in 0..4 {
                for ix in 0..4 {
                    coeffs[iy * 32 + ix] = llf[iy * 4 + ix];
                }
            }
            32
        }
        RAW_STRATEGY_DCT32X16 => {
            let mut grid = [0.0_f32; 8];
            grid.copy_from_slice(&dc_grid[..8]);
            let llf = restore_llf_dct32x16(grid);
            // 32×16 post-swap layout: stride 32. LLF at [iy*32 + ix] for
            // iy in 0..2, ix in 0..4 (2×4 LLF positions).
            for iy in 0..2 {
                for ix in 0..4 {
                    coeffs[iy * 32 + ix] = llf[iy * 4 + ix];
                }
            }
            32
        }
        RAW_STRATEGY_DCT16X32 => {
            let mut grid = [0.0_f32; 8];
            grid.copy_from_slice(&dc_grid[..8]);
            let llf = restore_llf_dct16x32(grid);
            for iy in 0..2 {
                for ix in 0..4 {
                    coeffs[iy * 32 + ix] = llf[iy * 4 + ix];
                }
            }
            32
        }
        RAW_STRATEGY_DCT64X64 => {
            let mut grid = [0.0_f32; 64];
            grid.copy_from_slice(&dc_grid[..64]);
            let llf = restore_llf_dct64x64(grid);
            // 64×64 layout: stride 64. LLF at [iy*64 + ix] for iy, ix in 0..8.
            for iy in 0..8 {
                for ix in 0..8 {
                    coeffs[iy * 64 + ix] = llf[iy * 8 + ix];
                }
            }
            64
        }
        RAW_STRATEGY_DCT64X32 => {
            let mut grid = [0.0_f32; 32];
            grid.copy_from_slice(&dc_grid[..32]);
            let llf = restore_llf_dct64x32(grid);
            // 64×32 post-swap layout: stride 64. LLF at [iy*64 + ix] for
            // iy in 0..4, ix in 0..8.
            for iy in 0..4 {
                for ix in 0..8 {
                    coeffs[iy * 64 + ix] = llf[iy * 8 + ix];
                }
            }
            64
        }
        RAW_STRATEGY_DCT32X64 => {
            let mut grid = [0.0_f32; 32];
            grid.copy_from_slice(&dc_grid[..32]);
            let llf = restore_llf_dct32x64(grid);
            for iy in 0..4 {
                for ix in 0..8 {
                    coeffs[iy * 64 + ix] = llf[iy * 8 + ix];
                }
            }
            64
        }
        _ => panic!(
            "dispatch_restore_llf: unsupported strategy {raw_strategy} \
             (AFV0-3 routed through forks::afv; other codes unmapped)"
        ),
    }
}

/// Restore the 4×8 LLF coefficients of a DCT64×32 block from the 8×4
/// stored DC grid. Mirrors upstream `restore_llf_from_dc` for
/// `RAW_STRATEGY_DCT64X32` (reconstruct.rs lines 758-805).
///
/// `dc_grid[iy * 4 + ix]` is the dequantized DC value at sub-block
/// `(iy, ix)` within the 8×4 region (8 rows × 4 cols, post-swap).
///
/// Returns `[f32; 32]` ordered as `out[iy * 8 + ix]` for `iy in 0..4,
/// ix in 0..8` — to be written at coefficient positions
/// `coeffs[iy * 64 + ix]` in the 64×32 coefficient block.
///
/// Math: forward 4-pt DCT on each of 8 rows → transpose 8×4 → 4×8 →
/// forward 8-pt DCT (with 1/8 compensation) on each of 4 rows →
/// divide by `(scale_iy * scale_ix * 4)`. The combined forward gain
/// is `dct1d_4(4) × dct1d_8/8(1) = 4`.
pub fn restore_llf_dct64x32(dc_grid: [f32; 32]) -> [f32; 32] {
    let mut block = dc_grid;
    // Forward 4-pt DCT on rows (8 rows of 4).
    for iy in 0..8 {
        dct1d_4(&mut block[iy * 4..(iy + 1) * 4]);
    }
    // Transpose 8×4 → 4×8.
    let mut t = [0.0_f32; 32];
    for iy in 0..8 {
        for ix in 0..4 {
            t[ix * 8 + iy] = block[iy * 4 + ix];
        }
    }
    // Forward 8-pt DCT on rows (4 rows of 8) with 1/8 compensation.
    for iy in 0..4 {
        let s = iy * 8;
        dct1d_8(&mut t[s..s + 8]);
        for v in &mut t[s..s + 8] {
            *v *= 1.0 / 8.0;
        }
    }
    // Apply per-position scale + 1/4 normalization.
    let mut out = [0.0_f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            let scale = DCT_RESAMPLE_SCALE_32_TO_4[iy] * DCT_RESAMPLE_SCALE_64_TO_8[ix];
            out[iy * 8 + ix] = t[iy * 8 + ix] / (scale * 4.0);
        }
    }
    out
}

/// Restore the 4×8 LLF coefficients of a DCT32×64 block from the 4×8
/// stored DC grid. Mirrors upstream `restore_llf_from_dc` for
/// `RAW_STRATEGY_DCT32X64` (reconstruct.rs lines 807-852).
///
/// `dc_grid[iy * 8 + ix]` is the dequantized DC value at sub-block
/// `(iy, ix)` within the 4×8 region (4 rows × 8 cols, post-swap).
///
/// Returns `[f32; 32]` ordered as `out[iy * 8 + ix]` for `iy in 0..4,
/// ix in 0..8` — to be written at coefficient positions
/// `coeffs[iy * 64 + ix]` in the 32×64 coefficient block.
///
/// Math: forward 8-pt DCT (with 1/8 compensation) on each of 4 rows →
/// transpose 4×8 → 8×4 → forward 4-pt DCT on each of 8 rows →
/// transpose back 8×4 → 4×8 → divide by `(scale_iy * scale_ix * 4)`.
pub fn restore_llf_dct32x64(dc_grid: [f32; 32]) -> [f32; 32] {
    let mut block = dc_grid;
    // Forward 8-pt DCT on rows (4 rows of 8) with 1/8 compensation.
    for iy in 0..4 {
        let s = iy * 8;
        dct1d_8(&mut block[s..s + 8]);
        for v in &mut block[s..s + 8] {
            *v *= 1.0 / 8.0;
        }
    }
    // Transpose 4×8 → 8×4.
    let mut t = [0.0_f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            t[ix * 4 + iy] = block[iy * 8 + ix];
        }
    }
    // Forward 4-pt DCT on rows (8 rows of 4).
    for iy in 0..8 {
        dct1d_4(&mut t[iy * 4..(iy + 1) * 4]);
    }
    // Transpose back 8×4 → 4×8.
    let mut result = [0.0_f32; 32];
    for iy in 0..8 {
        for ix in 0..4 {
            result[ix * 8 + iy] = t[iy * 4 + ix];
        }
    }
    // Apply per-position scale + 1/4 normalization.
    let mut out = [0.0_f32; 32];
    for iy in 0..4 {
        for ix in 0..8 {
            let scale = DCT_RESAMPLE_SCALE_32_TO_4[iy] * DCT_RESAMPLE_SCALE_64_TO_8[ix];
            out[iy * 8 + ix] = result[iy * 8 + ix] / (scale * 4.0);
        }
    }
    out
}

/// Restore the 8×8 LLF coefficients of a DCT64×64 block from the 8×8
/// stored DC grid. Mirrors upstream `restore_llf_from_dc` for
/// `RAW_STRATEGY_DCT64X64` (reconstruct.rs lines 732-757).
///
/// `dc_grid[iy * 8 + ix]` is the dequantized DC value at sub-block
/// `(iy, ix)` within the 8×8 region the DCT64×64 covers.
///
/// Returns `[f32; 64]` ordered as `out[iy * 8 + ix]` for
/// `iy, ix in 0..8` — to be written at coefficient positions
/// `coeffs[iy * 64 + ix]` in the 64×64 coefficient block.
///
/// Math: row-DCT (8-pt) → 1/8 scale → transpose → row-DCT (8-pt) →
/// 1/8 scale → divide by `(scale_iy * scale_ix)`. Combined forward
/// gain of 64 is split as two 1/8 scalings; the per-position
/// resample scale completes the normalization.
pub fn restore_llf_dct64x64(dc_grid: [f32; 64]) -> [f32; 64] {
    let mut block = dc_grid;
    // Forward 8-pt DCT on rows (8 rows of 8) with 1/8 scale.
    for iy in 0..8 {
        let s = iy * 8;
        dct1d_8(&mut block[s..s + 8]);
        for v in &mut block[s..s + 8] {
            *v *= 1.0 / 8.0;
        }
    }
    // Transpose 8×8.
    let mut t = [0.0_f32; 64];
    for iy in 0..8 {
        for ix in 0..8 {
            t[ix * 8 + iy] = block[iy * 8 + ix];
        }
    }
    // Forward 8-pt DCT on rows again with 1/8 scale.
    for iy in 0..8 {
        let s = iy * 8;
        dct1d_8(&mut t[s..s + 8]);
        for v in &mut t[s..s + 8] {
            *v *= 1.0 / 8.0;
        }
    }
    // Square-block convention: do NOT transpose back (matches the
    // libjxl `output[cx * 8 + cy] = ...` coefficient layout that
    // dc_from_dct_64x64 inverts). Apply per-position scale.
    let mut out = [0.0_f32; 64];
    for iy in 0..8 {
        for ix in 0..8 {
            let scale = DCT_RESAMPLE_SCALE_64_TO_8[iy] * DCT_RESAMPLE_SCALE_64_TO_8[ix];
            out[iy * 8 + ix] = t[iy * 8 + ix] / scale;
        }
    }
    out
}

/// Restore the 2×4 LLF coefficients of a DCT32×16 block from the 4×2
/// stored DC grid. Mirrors upstream `restore_llf_from_dc` for
/// `RAW_STRATEGY_DCT32X16` (reconstruct.rs lines 643-684).
///
/// `dc_grid[iy * 2 + ix]` is the dequantized DC value at sub-block
/// `(iy, ix)` within the 4×2 region (4 rows × 2 cols, post-swap layout).
///
/// Returns `[f32; 8]` ordered as `out[iy * 4 + ix]` for `iy in 0..2,
/// ix in 0..4` — to be written at coefficient positions
/// `coeffs[iy * 32 + ix]` in the 32×16 coefficient block.
///
/// Math: forward 2-pt DCT on each of 4 rows → transpose 4×2 → 2×4 →
/// forward 4-pt DCT on each of 2 rows → divide by `(scale * 8)`.
/// The 8 = `dct1d_2(2) * dct1d_4(4)` forward gain.
pub fn restore_llf_dct32x16(dc_grid: [f32; 8]) -> [f32; 8] {
    let mut block = dc_grid;
    // Forward 2-pt DCT on rows (4 rows of 2).
    for iy in 0..4 {
        dct1d_2(&mut block[iy * 2..(iy + 1) * 2]);
    }
    // Transpose 4×2 → 2×4.
    let mut t = [0.0_f32; 8];
    for iy in 0..4 {
        for ix in 0..2 {
            t[ix * 4 + iy] = block[iy * 2 + ix];
        }
    }
    // Forward 4-pt DCT on rows (2 rows of 4).
    dct1d_4(&mut t[0..4]);
    dct1d_4(&mut t[4..8]);
    // Apply per-position scale + 1/8 normalization.
    let mut out = [0.0_f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            let scale = DCT_RESAMPLE_SCALE_16_TO_2[iy] * DCT_RESAMPLE_SCALE_32_TO_4[ix];
            out[iy * 4 + ix] = t[iy * 4 + ix] / (scale * 8.0);
        }
    }
    out
}

/// Restore the 2×4 LLF coefficients of a DCT16×32 block from the 2×4
/// stored DC grid. Mirrors upstream `restore_llf_from_dc` for
/// `RAW_STRATEGY_DCT16X32` (reconstruct.rs lines 686-734).
///
/// `dc_grid[iy * 4 + ix]` is the dequantized DC value at sub-block
/// `(iy, ix)` within the 2×4 region (2 rows × 4 cols, post-swap layout).
///
/// Returns `[f32; 8]` ordered as `out[iy * 4 + ix]` for `iy in 0..2,
/// ix in 0..4` — to be written at coefficient positions
/// `coeffs[iy * 32 + ix]` in the 16×32 coefficient block.
///
/// Math: forward 4-pt DCT on each of 2 rows → transpose 2×4 → 4×2 →
/// forward 2-pt DCT on each of 4 rows → transpose 4×2 → 2×4 →
/// divide by `(scale * 8)`.
pub fn restore_llf_dct16x32(dc_grid: [f32; 8]) -> [f32; 8] {
    let mut block = dc_grid;
    // Forward 4-pt DCT on rows (2 rows of 4).
    dct1d_4(&mut block[0..4]);
    dct1d_4(&mut block[4..8]);
    // Transpose 2×4 → 4×2.
    let mut t = [0.0_f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            t[ix * 2 + iy] = block[iy * 4 + ix];
        }
    }
    // Forward 2-pt DCT on rows (4 rows of 2).
    for iy in 0..4 {
        dct1d_2(&mut t[iy * 2..(iy + 1) * 2]);
    }
    // Transpose back 4×2 → 2×4.
    let mut result = [0.0_f32; 8];
    for iy in 0..4 {
        for ix in 0..2 {
            result[ix * 4 + iy] = t[iy * 2 + ix];
        }
    }
    // Apply per-position scale + 1/8 normalization.
    let mut out = [0.0_f32; 8];
    for iy in 0..2 {
        for ix in 0..4 {
            let scale = DCT_RESAMPLE_SCALE_16_TO_2[iy] * DCT_RESAMPLE_SCALE_32_TO_4[ix];
            out[iy * 4 + ix] = result[iy * 4 + ix] / (scale * 8.0);
        }
    }
    out
}

/// Restore the 4×4 LLF coefficients of a DCT32×32 block from the 4×4
/// stored DC grid. Mirrors upstream `restore_llf_from_dc` for
/// `RAW_STRATEGY_DCT32X32` (reconstruct.rs lines 600-641).
///
/// `dc_grid[iy * 4 + ix]` is the dequantized DC value at sub-block
/// `(iy, ix)` within the 4×4 region the DCT32×32 covers (already
/// produced by [`dequant_dc_channel`] for each of `(by..by+4, bx..bx+4)`).
///
/// Output layout: returns `[f32; 16]` ordered to be written at
/// coefficient positions `coeffs[iy * 32 + ix]` for `iy, ix in 0..4`.
/// The caller is responsible for placing the 16 values at the right
/// positions in the larger 32×32 coefficient buffer.
///
/// Math (inverse of `dc_from_dct_32x32`):
/// ```text
///   Forward: scale + 4×4 IDCT (idct1d_4 on rows, transpose,
///            idct1d_4 on rows). The 4×4 IDCT is the inverse of
///            our 4-point DCT divided by 4 (libjxl IDCT
///            normalization).
///   Inverse: forward 4-point DCT on rows, transpose, forward
///            4-point DCT on rows, then divide by (scale * 16).
/// ```
/// where `scale = DCT_RESAMPLE_SCALE_32_TO_4[iy] *
///                DCT_RESAMPLE_SCALE_32_TO_4[ix]`.
pub fn restore_llf_dct32x32(dc_grid: [f32; 16]) -> [f32; 16] {
    let mut block = dc_grid;
    // Forward 4pt DCT on rows.
    dct1d_4(&mut block[0..4]);
    dct1d_4(&mut block[4..8]);
    dct1d_4(&mut block[8..12]);
    dct1d_4(&mut block[12..16]);
    // Transpose 4×4.
    let mut transposed = [0.0_f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            transposed[ix * 4 + iy] = block[iy * 4 + ix];
        }
    }
    // Forward 4pt DCT on rows.
    dct1d_4(&mut transposed[0..4]);
    dct1d_4(&mut transposed[4..8]);
    dct1d_4(&mut transposed[8..12]);
    dct1d_4(&mut transposed[12..16]);
    // Apply per-position scale + 1/16 normalization.
    let mut out = [0.0_f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            let scale = DCT_RESAMPLE_SCALE_32_TO_4[iy] * DCT_RESAMPLE_SCALE_32_TO_4[ix];
            out[iy * 4 + ix] = transposed[iy * 4 + ix] / (scale * 16.0);
        }
    }
    out
}

/// Restore the 2 LLF coefficients of a DCT16×8 or DCT8×16 block from
/// the 2 stored DC values. Mirrors upstream
/// `restore_llf_from_dc` for `RAW_STRATEGY_DCT16X8` /
/// `RAW_STRATEGY_DCT8X16` (reconstruct.rs lines 546-570).
///
/// Inputs:
/// - `dc0`, `dc1`: the two stored DC values (already dequantized via
///   [`dequant_dc_channel`]). For DCT16×8 these come from the
///   vertically-adjacent pair `(by, by+1)`; for DCT8×16 the
///   horizontally-adjacent pair `(bx, bx+1)`.
///
/// Returns `[llf0, llf1]` as a `[f32; 2]` ready to be written into
/// `coeffs[0]` and `coeffs[1]` of the rectangular coefficient block.
///
/// Math (inverse of `dc_from_dct_16x8` / `dc_from_dct_8x16`):
/// ```text
///   Forward: dc0 = llf0 * s0 + llf1 * s1
///            dc1 = llf0 * s0 - llf1 * s1
///   Inverse: llf0 = (dc0 + dc1) / (2 * s0)
///            llf1 = (dc0 - dc1) / (2 * s1)
/// ```
/// where `s0 = DCT_RESAMPLE_SCALE_16_TO_2[0] = 1.0` and
/// `s1 = DCT_RESAMPLE_SCALE_16_TO_2[1] ≈ 0.9018`. The factor 2 comes
/// from the 2-point Hadamard's `H * H = 2 * I` self-product.
#[inline]
pub fn restore_llf_dct16x8_or_8x16(dc0: f32, dc1: f32) -> [f32; 2] {
    let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
    let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];
    [(dc0 + dc1) / (2.0 * s0), (dc0 - dc1) / (2.0 * s1)]
}

/// Restore the 2×2 LLF coefficients of a DCT16×16 block from the
/// 2×2 stored DC grid. Mirrors upstream `restore_llf_from_dc` for
/// `RAW_STRATEGY_DCT16X16` (reconstruct.rs lines 572-598).
///
/// `dc_grid[iy * 2 + ix]` is the dequantized DC value at sub-block
/// `(iy, ix)` within the 2×2 region the DCT16×16 covers (already
/// produced by [`dequant_dc_channel`] for each of `(by..by+2, bx..bx+2)`).
///
/// Returns `[llf00, llf01, llf10, llf11]` to be written at coefficient
/// positions `[0, 1, 16, 17]` of the 16×16 coefficient block.
///
/// Math (inverse of `dc_from_dct_16x16` — 2-point row+column DCT
/// followed by SCALE_16_TO_2 scaling, where the 2-point DCT is
/// Hadamard with `H * H = 4 * I` for the 2×2):
/// ```text
///   h00 = dc00 + dc01 + dc10 + dc11
///   h01 = dc00 + dc01 - dc10 - dc11
///   h10 = dc00 - dc01 + dc10 - dc11
///   h11 = dc00 - dc01 - dc10 + dc11
///   llf00 = h00 / (4 * s0 * s0)
///   llf01 = h01 / (4 * s0 * s1)
///   llf10 = h10 / (4 * s1 * s0)
///   llf11 = h11 / (4 * s1 * s1)
/// ```
#[inline]
pub fn restore_llf_dct16x16(dc_grid: [f32; 4]) -> [f32; 4] {
    let h00 = dc_grid[0] + dc_grid[1] + dc_grid[2] + dc_grid[3];
    let h01 = dc_grid[0] + dc_grid[1] - dc_grid[2] - dc_grid[3];
    let h10 = dc_grid[0] - dc_grid[1] + dc_grid[2] - dc_grid[3];
    let h11 = dc_grid[0] - dc_grid[1] - dc_grid[2] + dc_grid[3];
    let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
    let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];
    [
        h00 / (4.0 * s0 * s0),
        h01 / (4.0 * s0 * s1),
        h10 / (4.0 * s1 * s0),
        h11 / (4.0 * s1 * s1),
    ]
}

/// DC restoration for the DCT8 fast path of upstream's
/// `reconstruct_xyb`. Pure scalar — bit-for-bit copy of upstream
/// (reconstruct.rs lines 297-317).
///
/// Inputs:
/// - `dq_x`/`dq_y`/`dq_b`: 64-element dequantized coefficient arrays
///   for the block (output of dequant_dct8 — positions 1..64 are AC).
/// - `quant_dc_x`/`quant_dc_y`/`quant_dc_b`: stored DC values
///   (typically `i16`, cast to `f32` here).
/// - `scale_dc`: from upstream `params.scale_dc`.
///
/// Behavior (matches upstream):
/// 1. Compute per-channel `inv_factor[c] = INV_DC_QUANT[c] * scale_dc`.
/// 2. Override the DC slot:
///    - `dq_y[0] = quant_dc_y / inv_factor[1]`
///    - `dq_x[0] = quant_dc_x / inv_factor[0]`
///    - `dq_b[0] = (quant_dc_b + quant_dc_y * dc_cfl_factor_b) / inv_factor[2]`
///      where `dc_cfl_factor_b = 0.5` (B-channel DC-level CfL).
///
/// Note: the AC-level CfL (per-tile `ytox_ratio` / `ytob_ratio`) is
/// already applied during dequant. This function applies *only* the
/// DC-level CfL — a separate fixed 0.5× contribution from Y to B at
/// position 0.
pub fn restore_dct8_dc_override(
    dq_x: &mut [f32; 64],
    dq_y: &mut [f32; 64],
    dq_b: &mut [f32; 64],
    quant_dc_x: f32,
    quant_dc_y: f32,
    quant_dc_b: f32,
    scale_dc: f32,
) {
    let inv_factor = [
        INV_DC_QUANT[0] * scale_dc,
        INV_DC_QUANT[1] * scale_dc,
        INV_DC_QUANT[2] * scale_dc,
    ];
    const DC_CFL_FACTOR_B: f32 = 0.5;
    dq_y[0] = quant_dc_y / inv_factor[1];
    dq_x[0] = quant_dc_x / inv_factor[0];
    dq_b[0] = (quant_dc_b + quant_dc_y * DC_CFL_FACTOR_B) / inv_factor[2];
}

/// Batched form of [`restore_dct8_dc_override`] for `n_blocks` 64-coef
/// DCT8 blocks. Operates on flat slices in block-major layout —
/// matches what `dequant_dct8_blocks_gpu` returns.
///
/// `quant_dc_*` are per-block DC values (length `n_blocks`, typically
/// `i16`-stored, passed as `f32` via `as f32` cast). `dq_*` are
/// dequantized coefficient blocks (length `n_blocks * 64`); only the
/// `[b * 64]` slot of each block is mutated.
///
/// Bit-for-bit equivalent to running [`restore_dct8_dc_override`]
/// in a per-block loop. Pure scalar — kept on host because the
/// per-block work is just three scalar divides and an FMA, dwarfed
/// by GPU-launch overhead at typical batch sizes.
pub fn restore_dct8_dc_override_batched(
    dq_x: &mut [f32],
    dq_y: &mut [f32],
    dq_b: &mut [f32],
    quant_dc_x: &[f32],
    quant_dc_y: &[f32],
    quant_dc_b: &[f32],
    scale_dc: f32,
) {
    let n_blocks = quant_dc_y.len();
    debug_assert_eq!(quant_dc_x.len(), n_blocks);
    debug_assert_eq!(quant_dc_b.len(), n_blocks);
    debug_assert_eq!(dq_x.len(), n_blocks * 64);
    debug_assert_eq!(dq_y.len(), n_blocks * 64);
    debug_assert_eq!(dq_b.len(), n_blocks * 64);
    let inv_factor = [
        INV_DC_QUANT[0] * scale_dc,
        INV_DC_QUANT[1] * scale_dc,
        INV_DC_QUANT[2] * scale_dc,
    ];
    const DC_CFL_FACTOR_B: f32 = 0.5;
    for b in 0..n_blocks {
        let dy = quant_dc_y[b];
        dq_y[b * 64] = dy / inv_factor[1];
        dq_x[b * 64] = quant_dc_x[b] / inv_factor[0];
        dq_b[b * 64] = (quant_dc_b[b] + dy * DC_CFL_FACTOR_B) / inv_factor[2];
    }
}

/// Reconstruct XYB pixel planes from quantized DC + AC coefficients,
/// for an image where every block is DCT8. Mirrors the DCT8 fast path
/// of upstream `reconstruct_xyb_impl` (reconstruct.rs lines 268-344)
/// composed end-to-end on GPU.
///
/// Pipeline (host-orchestrated, 4 GPU launches per image):
/// 1. `dequant_dct8_blocks_gpu` — batched 3-channel dequant + AC-level
///    CfL fold (one launch).
/// 2. host: `restore_dct8_dc_override_batched` — DC override with the
///    fixed 0.5× Y→B DC-level CfL.
/// 3. `apply_idct_batch_gpu(DCT8)` per channel — three launches.
/// 4. host: scatter the per-block 8×8 outputs into padded
///    `(xsize_blocks * 8) × (ysize_blocks * 8)` planes.
///
/// Inputs are all flat block-major slices (`n_blocks * 64` for AC,
/// `n_blocks` for per-block scalars). Per-block CfL factors must
/// already be resolved from the per-tile CfL map by the caller.
///
/// Returns `[plane_x, plane_y, plane_b]` of length
/// `xsize_blocks * ysize_blocks * 64` (= padded width × padded height).
///
/// **Note**: this is the all-blocks-are-DCT8 path. Real images use a
/// mix of strategies via the AC strategy map; supporting that
/// requires the per-strategy IDCT dispatch + scatter for non-DCT8
/// blocks, which is a separate piece of `reconstruct_xyb_impl`.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_xyb_dct8_only_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    quant_dc_x: &[f32],
    quant_dc_y: &[f32],
    quant_dc_b: &[f32],
    quant_ac_x: &[i32],
    quant_ac_y: &[i32],
    quant_ac_b: &[i32],
    weights_x_per_block: &[f32; 64],
    weights_y_per_block: &[f32; 64],
    weights_b_per_block: &[f32; 64],
    qac_qm_x: &[f32],
    qac_qm_y: &[f32],
    qac_qm_b: &[f32],
    x_factor: &[f32],
    b_factor: &[f32],
    scale_dc: f32,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> [Vec<f32>; 3] {
    use crate::forks::dequant::dequant_dct8_blocks_gpu_broadcast_w;
    use crate::forks::transform::{RAW_STRATEGY_DCT, apply_idct_batch_gpu};

    let n_blocks = xsize_blocks * ysize_blocks;
    debug_assert_eq!(quant_dc_x.len(), n_blocks);
    debug_assert_eq!(quant_dc_y.len(), n_blocks);
    debug_assert_eq!(quant_dc_b.len(), n_blocks);
    debug_assert_eq!(quant_ac_x.len(), n_blocks * 64);
    debug_assert_eq!(quant_ac_y.len(), n_blocks * 64);
    debug_assert_eq!(quant_ac_b.len(), n_blocks * 64);
    debug_assert_eq!(qac_qm_x.len(), n_blocks);
    debug_assert_eq!(qac_qm_y.len(), n_blocks);
    debug_assert_eq!(qac_qm_b.len(), n_blocks);
    debug_assert_eq!(x_factor.len(), n_blocks);
    debug_assert_eq!(b_factor.len(), n_blocks);

    // Step 1: GPU dequant (one launch, three channels) using
    // broadcast-weights variant — saves the n_blocks-replication of
    // the 3 × 64-float weight templates that the per-block variant
    // required (4 MB per channel saved at 16384 blocks).
    let (mut dq_x, mut dq_y, mut dq_b) = dequant_dct8_blocks_gpu_broadcast_w(
        enc,
        quant_ac_x,
        quant_ac_y,
        quant_ac_b,
        weights_x_per_block.as_slice(),
        weights_y_per_block.as_slice(),
        weights_b_per_block.as_slice(),
        qac_qm_x,
        qac_qm_y,
        qac_qm_b,
        x_factor,
        b_factor,
    );

    // Step 2: host DC override (overwrites position [b * 64] of each plane).
    restore_dct8_dc_override_batched(
        &mut dq_x, &mut dq_y, &mut dq_b, quant_dc_x, quant_dc_y, quant_dc_b, scale_dc,
    );

    // Step 3: per-channel IDCT 8x8 (three launches).
    let pix_x = apply_idct_batch_gpu(enc, &dq_x, RAW_STRATEGY_DCT);
    let pix_y = apply_idct_batch_gpu(enc, &dq_y, RAW_STRATEGY_DCT);
    let pix_b = apply_idct_batch_gpu(enc, &dq_b, RAW_STRATEGY_DCT);

    // Step 4: scatter block-major pixels into padded planes.
    let padded_w = xsize_blocks * 8;
    let padded_h = ysize_blocks * 8;
    let n_pix = padded_w * padded_h;
    let mut plane_x = vec![0.0_f32; n_pix];
    let mut plane_y = vec![0.0_f32; n_pix];
    let mut plane_b = vec![0.0_f32; n_pix];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let b = by * xsize_blocks + bx;
            let src = b * 64;
            let dst_y0 = by * 8;
            let dst_x0 = bx * 8;
            for row in 0..8 {
                let s = src + row * 8;
                let d = (dst_y0 + row) * padded_w + dst_x0;
                plane_x[d..d + 8].copy_from_slice(&pix_x[s..s + 8]);
                plane_y[d..d + 8].copy_from_slice(&pix_y[s..s + 8]);
                plane_b[d..d + 8].copy_from_slice(&pix_b[s..s + 8]);
            }
        }
    }

    [plane_x, plane_y, plane_b]
}

/// Decoder-side gab smoothing weights from libjxl epf.cc / loop_filter.h.
/// Duplicated bit-for-bit from upstream `gab_smooth`.
///
/// Returns `(w_center, w1, w2)` — center weight, edge-neighbor weight,
/// corner-neighbor weight. All four edges share `w1`, all four corners
/// share `w2`. Sum is 1.0 (energy-preserving).
///
/// Public so callers chaining `GpuEncoder::gab_smooth_persistent` (e.g.,
/// `LossyEncoder`) can pass the same constants the upstream decoder
/// would produce.
pub fn gab_weights() -> (f32, f32, f32) {
    let w1_base = 0.104_699_57_f32 * 1.1;
    let w2_base = 0.055_680_54_f32 * 1.1;
    let div = 1.0 + 4.0 * (w1_base + w2_base);
    let w_center = 1.0 / div;
    let w1 = w1_base / div;
    let w2 = w2_base / div;
    (w_center, w1, w2)
}

/// GPU `gab_smooth`. Mirrors upstream
/// `jxl_encoder::vardct::reconstruct::gab_smooth`.
///
/// Three sequential GPU launches over the X/Y/B planes (planes order
/// matches upstream: `planes[0]=X`, `planes[1]=Y`, `planes[2]=B`). Each
/// channel is mutated in place via copy-from-Vec on the GPU return.
pub fn gab_smooth_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    planes: &mut [Vec<f32>; 3],
    width: usize,
    height: usize,
) {
    let (w_center, w1, w2) = gab_weights();
    for plane in planes.iter_mut() {
        assert_eq!(plane.len(), width * height);
        let out = enc.gab_smooth_channel(plane, width as u32, height as u32, w_center, w1, w2);
        plane.copy_from_slice(&out);
    }
}

/// GPU `xyb_to_linear_rgb_planar`. Mirrors upstream signature exactly.
#[allow(clippy::too_many_arguments)]
pub fn xyb_to_linear_rgb_planar_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    out_r: &mut [f32],
    out_g: &mut [f32],
    out_b: &mut [f32],
    num_pixels: usize,
) {
    assert_eq!(xyb_x.len(), num_pixels);
    assert_eq!(xyb_y.len(), num_pixels);
    assert_eq!(xyb_b.len(), num_pixels);
    assert_eq!(out_r.len(), num_pixels);
    assert_eq!(out_g.len(), num_pixels);
    assert_eq!(out_b.len(), num_pixels);
    let (r, g, b) = enc.xyb_to_linear_rgb_planar(xyb_x, xyb_y, xyb_b);
    out_r.copy_from_slice(&r);
    out_g.copy_from_slice(&g);
    out_b.copy_from_slice(&b);
}

/// GPU `xyb_to_linear_rgb` (interleaved). Mirrors upstream return shape:
/// a `Vec<f32>` of length `num_pixels * 3` with `[R, G, B, R, G, B, ...]`.
pub fn xyb_to_linear_rgb_gpu<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let num_pixels = width * height;
    assert_eq!(xyb_x.len(), num_pixels);
    let (r, g, b) = enc.xyb_to_linear_rgb_planar(xyb_x, xyb_y, xyb_b);
    let mut interleaved = vec![0.0_f32; num_pixels * 3];
    for i in 0..num_pixels {
        interleaved[i * 3] = r[i];
        interleaved[i * 3 + 1] = g[i];
        interleaved[i * 3 + 2] = b[i];
    }
    interleaved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inv_dc_quant_matches_upstream() {
        // G5.1 parity: each of the 3 channel values must match upstream's
        // jxl_encoder::__internals::INV_DC_QUANT exactly.
        for c in 0..3 {
            assert_eq!(
                INV_DC_QUANT[c],
                jxl_encoder::__internals::INV_DC_QUANT[c],
                "channel {c}"
            );
        }
    }

    #[test]
    fn test_dequant_dc_channel_x_y_no_cfl() {
        // X / Y channels: no CfL contribution. Output = quant_dc / inv_factor.
        // X: inv_factor = 4096 * 0.5 = 2048; 100 / 2048 = 0.04883
        let v = dequant_dc_channel(100.0, 999.0, 0, 0.5);
        assert!((v - (100.0 / 2048.0)).abs() < 1e-6);
        // Y: inv_factor = 512 * 1.0 = 512; 50 / 512 = 0.09766
        let v = dequant_dc_channel(50.0, 999.0, 1, 1.0);
        assert!((v - (50.0 / 512.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dequant_dc_channel_b_includes_y_cfl() {
        // B: dc_cfl_factor = 0.5. inv_factor = 256.
        // (10 + 100 * 0.5) / 256 = 60 / 256 = 0.234375
        let v = dequant_dc_channel(10.0, 100.0, 2, 1.0);
        assert!((v - 0.234_375).abs() < 1e-6);
    }

    #[test]
    fn test_restore_llf_dct16x8_roundtrip() {
        // Forward dc_from_dct_16x8 followed by inverse should be identity.
        let s0 = DCT_RESAMPLE_SCALE_16_TO_2[0];
        let s1 = DCT_RESAMPLE_SCALE_16_TO_2[1];
        // Pick arbitrary llf0/llf1 values, project forward to dc0/dc1, then invert.
        let llf0 = 1.7_f32;
        let llf1 = -0.4_f32;
        let dc0 = llf0 * s0 + llf1 * s1;
        let dc1 = llf0 * s0 - llf1 * s1;
        let [r0, r1] = restore_llf_dct16x8_or_8x16(dc0, dc1);
        assert!((r0 - llf0).abs() < 1e-5, "got {r0} expected {llf0}");
        assert!((r1 - llf1).abs() < 1e-5, "got {r1} expected {llf1}");
    }

    /// DC grid mean invariant: uniform plane → all blocks same DC =
    /// mean of the block. (Was previously `sum/8`; corrected in commit
    /// ac88dc1c to match this codebase's mean-scale DCT convention.)
    #[test]
    fn test_compute_dc_grid_uniform() {
        let plane = alloc::vec![0.5_f32; 16 * 16];
        let dc = compute_dc_grid_per_8x8_block(&plane, 16, 16);
        // 4 blocks (2×2 grid). Each block: mean = 0.5.
        assert_eq!(dc.len(), 4);
        for v in &dc {
            assert!((v - 0.5).abs() < 1e-5, "got {v}");
        }
    }

    /// DC grid varies with content: gradient → distinct per-block DCs
    /// (each = block mean).
    #[test]
    fn test_compute_dc_grid_gradient() {
        // 16x8 plane with gradient: pixel value = x.
        let mut plane = alloc::vec![0.0_f32; 16 * 8];
        for y in 0..8 {
            for x in 0..16 {
                plane[y * 16 + x] = x as f32;
            }
        }
        let dc = compute_dc_grid_per_8x8_block(&plane, 16, 8);
        // Two blocks.
        // Block (0,0): pixels x=0..7, mean = (0+1+..+7)/8 = 28/8 = 3.5
        //   (every row identical, so block mean = row mean).
        // Block (1,0): pixels x=8..15, mean = (8+9+..+15)/8 = 92/8 = 11.5.
        assert_eq!(dc.len(), 2);
        assert!((dc[0] - 3.5).abs() < 1e-4, "block0: got {} expected 3.5", dc[0]);
        assert!((dc[1] - 11.5).abs() < 1e-4, "block1: got {} expected 11.5", dc[1]);
    }

    /// Smoke test: encode_and_reconstruct_mixed_strategy_single_channel
    /// with a heterogeneous DCT8/DCT16x16 assignment over a synthetic
    /// uniform-DC plane. Verifies it composes end-to-end without panic
    /// and produces finite output. Quality validation deferred to
    /// integration with LossyEncoder where real cost grids drive the
    /// assignments.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_encode_and_reconstruct_mixed_strategy_single_channel_smoke() {
        use crate::forks::transform::{RAW_STRATEGY_DCT, RAW_STRATEGY_DCT16X16};
        use crate::pipeline::StrategyAssignment;
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();

        let padded_w = 32_usize;
        let padded_h = 32_usize;
        let xsize_blocks_8 = padded_w / 8;
        let ysize_blocks_8 = padded_h / 8;
        let n_blocks = xsize_blocks_8 * ysize_blocks_8;

        // Synthetic plane: uniform 0.5 → block DC = 4.0 (= 0.5 * 8 for
        // mean-preserving 8x8 DCT) so non-DC coeffs are zero.
        let xyb_channel = alloc::vec![0.5_f32; padded_w * padded_h];

        // Mixed: top-left 16x16 region = DCT16x16; rest = DCT8.
        let mut assignments = vec![StrategyAssignment {
            bx: 0,
            by: 0,
            raw_strategy: RAW_STRATEGY_DCT16X16,
        }];
        for by in 0..ysize_blocks_8 {
            for bx in 0..xsize_blocks_8 {
                if bx < 2 && by < 2 {
                    continue; // covered by the DCT16 block
                }
                assignments.push(StrategyAssignment {
                    bx,
                    by,
                    raw_strategy: RAW_STRATEGY_DCT,
                });
            }
        }

        // Per-strategy weights: identity (all 1.0).
        let weights_for = |raw_strategy: u8| -> Vec<f32> {
            use crate::forks::transform::coeff_count_per_strategy;
            alloc::vec![1.0_f32; coeff_count_per_strategy(raw_strategy)]
        };

        let qac = alloc::vec![1.0_f32; n_blocks];
        let thresholds = [0.0_f32; 4]; // no dead-zone for this smoke test
        // DC grid: from uniform 0.5 plane, every 8x8 block has DC = 4.0.
        let dc_grid = alloc::vec![4.0_f32; n_blocks];

        let mut out_plane = alloc::vec![0.0_f32; padded_w * padded_h];
        encode_and_reconstruct_mixed_strategy_single_channel(
            &enc,
            &xyb_channel,
            padded_w,
            padded_h,
            &assignments,
            &weights_for,
            &qac,
            &thresholds,
            &dc_grid,
            1, // Y channel for the test
            Some(&mut out_plane),
            None,
            None,
            None,
        );

        // All output pixels finite, roughly close to original (~0.5).
        for (i, &v) in out_plane.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at i={i}: {v}");
        }
    }

    /// Isolated AFV reconstruct diagnostic. Forces all blocks to AFV0
    /// (or AFV1/2/3) on uniform M=1.0 input and measures RMSE. If the
    /// RMSE is high, the AFV reconstruct branch in
    /// encode_and_reconstruct_mixed_strategy_single_channel has a bug
    /// (kind, basis, dequant qac, scatter, etc.). For uniform input
    /// the DC reconstruction should be near-perfect.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_afv_isolated_reconstruct_uniform_input() {
        use crate::forks::transform::{
            coeff_count_per_strategy, RAW_STRATEGY_AFV0, RAW_STRATEGY_AFV1, RAW_STRATEGY_AFV2,
            RAW_STRATEGY_AFV3,
        };
        use crate::pipeline::StrategyAssignment;
        use crate::quant_weights::afv_weights_per_channel;
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();

        let pw = 32_usize;
        let ph = 32_usize;
        let xb8 = pw / 8;
        let yb8 = ph / 8;
        let nb8 = xb8 * yb8;

        // Smooth gradient input — reveals reconstruction bugs that
        // uniform input hides (uniform DCT-transforms-back-to-DC means
        // all AC coeffs are 0 and the output is just DC).
        let xyb: Vec<f32> = (0..pw * ph)
            .map(|i| {
                let x = (i % pw) as f32 / pw as f32;
                let y = (i / pw) as f32 / ph as f32;
                0.4 + 0.15 * x + 0.15 * y
            })
            .collect();
        let dc_grid = compute_dc_grid_per_8x8_block(&xyb, pw, ph);
        let qac = alloc::vec![1.0_f32; nb8];
        let thresholds = [0.0_f32; 4];

        let (_afv_x, afv_y, _afv_b) = afv_weights_per_channel();
        let weights_for = move |s: u8| -> Vec<f32> {
            match s {
                RAW_STRATEGY_AFV0
                | RAW_STRATEGY_AFV1
                | RAW_STRATEGY_AFV2
                | RAW_STRATEGY_AFV3 => afv_y.to_vec(),
                _ => alloc::vec![1.0_f32; coeff_count_per_strategy(s)],
            }
        };

        // Reference: all-DCT8 on the same gradient (known good).
        {
            let assignments: Vec<StrategyAssignment> = (0..yb8)
                .flat_map(|by| {
                    (0..xb8).map(move |bx| StrategyAssignment {
                        bx,
                        by,
                        raw_strategy: crate::forks::transform::RAW_STRATEGY_DCT,
                    })
                })
                .collect();
            let mut out = alloc::vec![0.0_f32; pw * ph];
            encode_and_reconstruct_mixed_strategy_single_channel(
                &enc, &xyb, pw, ph, &assignments, &weights_for, &qac, &thresholds,
                &dc_grid, 1, Some(&mut out), None, None, None,
            );
            let mut sumsq = 0.0_f64;
            for i in 0..pw * ph {
                sumsq += ((out[i] - xyb[i]) as f64).powi(2);
            }
            let rmse = (sumsq / (pw * ph) as f64).sqrt();
            std::println!("[afv-recon-iso] DCT8 ref:  RMSE={rmse:.6e}");
        }

        for (kind_name, kind_strat) in [
            ("AFV0", RAW_STRATEGY_AFV0),
            ("AFV1", RAW_STRATEGY_AFV1),
            ("AFV2", RAW_STRATEGY_AFV2),
            ("AFV3", RAW_STRATEGY_AFV3),
        ] {
            let assignments: Vec<StrategyAssignment> = (0..yb8)
                .flat_map(|by| {
                    (0..xb8).map(move |bx| StrategyAssignment {
                        bx,
                        by,
                        raw_strategy: kind_strat,
                    })
                })
                .collect();
            let mut out = alloc::vec![0.0_f32; pw * ph];
            encode_and_reconstruct_mixed_strategy_single_channel(
                &enc, &xyb, pw, ph, &assignments, &weights_for, &qac, &thresholds,
                &dc_grid, 1, Some(&mut out), None, None, None,
            );
            let mut sumsq = 0.0_f64;
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for i in 0..pw * ph {
                sumsq += ((out[i] - xyb[i]) as f64).powi(2);
                min_v = min_v.min(out[i]);
                max_v = max_v.max(out[i]);
            }
            let rmse = (sumsq / (pw * ph) as f64).sqrt();
            std::println!(
                "[afv-recon-iso] {kind_name}: RMSE={rmse:.6e} out range=[{min_v:.4},{max_v:.4}]"
            );
        }
    }

    /// Diagnostic for the DCT32x32 quality regression: force ALL blocks
    /// to DCT32x32 on a smooth gradient, reconstruct, and compare RMSE
    /// against an all-DCT8 reconstruction. Both should give similar
    /// (small) RMSE on smooth content. If DCT32x32 is much higher,
    /// the encode_and_reconstruct path has a bug for that strategy.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_dct32x32_reconstruct_smooth_gradient() {
        use crate::forks::transform::{
            coeff_count_per_strategy, RAW_STRATEGY_DCT, RAW_STRATEGY_DCT32X32,
        };
        use crate::pipeline::StrategyAssignment;
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();

        // 64×64 plane (= 2×2 grid of DCT32x32 blocks, or 8×8 grid of
        // DCT8 blocks). Smooth gradient in [0.4, 0.7].
        let pw = 64_usize;
        let ph = 64_usize;
        let xb8 = pw / 8;
        let yb8 = ph / 8;
        let nb8 = xb8 * yb8;
        let xyb: Vec<f32> = (0..pw * ph)
            .map(|i| {
                let x = (i % pw) as f32 / pw as f32;
                let y = (i / pw) as f32 / ph as f32;
                0.4 + 0.15 * x + 0.15 * y
            })
            .collect();

        let dc_grid = compute_dc_grid_per_8x8_block(&xyb, pw, ph);
        let qac = alloc::vec![1.0_f32; nb8];
        let thresholds = [0.0_f32; 4]; // no dead-zone

        // All-DCT8 reconstruction (baseline).
        let assignments_dct8: Vec<StrategyAssignment> = (0..yb8)
            .flat_map(|by| {
                (0..xb8).map(move |bx| StrategyAssignment {
                    bx,
                    by,
                    raw_strategy: RAW_STRATEGY_DCT,
                })
            })
            .collect();
        let weights_for = |s: u8| -> Vec<f32> {
            alloc::vec![1.0_f32; coeff_count_per_strategy(s)]
        };
        let mut out_dct8 = alloc::vec![0.0_f32; pw * ph];
        encode_and_reconstruct_mixed_strategy_single_channel(
            &enc, &xyb, pw, ph, &assignments_dct8, &weights_for, &qac, &thresholds,
            &dc_grid, 1, Some(&mut out_dct8), None, None, None,
        );
        let mut sse_dct8 = 0.0_f64;
        for i in 0..pw * ph {
            let d = (xyb[i] - out_dct8[i]) as f64;
            sse_dct8 += d * d;
        }
        let rmse_dct8 = (sse_dct8 / (pw * ph) as f64).sqrt();

        // All-DCT32x32 reconstruction (4 blocks at (0,0), (4,0), (0,4), (4,4)).
        let assignments_dct32: Vec<StrategyAssignment> = (0..2)
            .flat_map(|ry| {
                (0..2).map(move |rx| StrategyAssignment {
                    bx: rx * 4,
                    by: ry * 4,
                    raw_strategy: RAW_STRATEGY_DCT32X32,
                })
            })
            .collect();
        let mut out_dct32 = alloc::vec![0.0_f32; pw * ph];
        encode_and_reconstruct_mixed_strategy_single_channel(
            &enc, &xyb, pw, ph, &assignments_dct32, &weights_for, &qac, &thresholds,
            &dc_grid, 1, Some(&mut out_dct32), None, None, None,
        );
        let mut sse_dct32 = 0.0_f64;
        for i in 0..pw * ph {
            let d = (xyb[i] - out_dct32[i]) as f64;
            sse_dct32 += d * d;
        }
        let rmse_dct32 = (sse_dct32 / (pw * ph) as f64).sqrt();

        std::println!("[dct32-diag] all-DCT8  RMSE = {rmse_dct8:.6}");
        std::println!("[dct32-diag] all-DCT32 RMSE = {rmse_dct32:.6}");
        std::println!("[dct32-diag] sample pixels (ref / dct8 / dct32):");
        for &i in &[0_usize, 33, 1024, 2047, 4095] {
            std::println!("  [{i}] ref={:.4} dct8={:.4} dct32={:.4}",
                xyb[i], out_dct8[i], out_dct32[i]);
        }
    }

    #[test]
    fn test_reconstruct_mixed_strategy_gpu_dct8_and_dct16x16() {
        // Mix two strategies: 3 DCT8 blocks + 2 DCT16x16 blocks at
        // non-overlapping positions.
        use crate::forks::transform::{RAW_STRATEGY_DCT, RAW_STRATEGY_DCT16X16};
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();

        let padded_w = 64_usize;
        let padded_h = 32_usize;
        let mut plane = alloc::vec![0.42_f32; padded_w * padded_h];

        let coeffs_dct8 = alloc::vec![0.0_f32; 64];
        let coeffs_dct16 = alloc::vec![0.0_f32; 256];
        let recipes = [
            BlockRecipe {
                bx: 0,
                by: 0,
                raw_strategy: RAW_STRATEGY_DCT,
                coeffs: &coeffs_dct8,
            },
            BlockRecipe {
                bx: 1,
                by: 1,
                raw_strategy: RAW_STRATEGY_DCT,
                coeffs: &coeffs_dct8,
            },
            BlockRecipe {
                bx: 7,
                by: 0,
                raw_strategy: RAW_STRATEGY_DCT,
                coeffs: &coeffs_dct8,
            },
            BlockRecipe {
                bx: 2,
                by: 2,
                raw_strategy: RAW_STRATEGY_DCT16X16,
                coeffs: &coeffs_dct16,
            },
            BlockRecipe {
                bx: 5,
                by: 0,
                raw_strategy: RAW_STRATEGY_DCT16X16,
                coeffs: &coeffs_dct16,
            },
        ];

        reconstruct_mixed_strategy_gpu(&enc, &recipes, &mut plane, padded_w);

        // DCT8 blocks: each is 8×8 at (bx*8, by*8).
        for (bx, by) in [(0_usize, 0_usize), (1, 1), (7, 0)] {
            for row in 0..8 {
                for col in 0..8 {
                    let v = plane[(by * 8 + row) * padded_w + bx * 8 + col];
                    assert!(v.abs() < 1e-5, "DCT8 ({bx},{by}) [{row},{col}] = {v}");
                }
            }
        }
        // DCT16x16 blocks: each is 16×16 at (bx*8, by*8).
        for (bx, by) in [(2_usize, 2_usize), (5, 0)] {
            for row in 0..16 {
                for col in 0..16 {
                    let v = plane[(by * 8 + row) * padded_w + bx * 8 + col];
                    assert!(v.abs() < 1e-5, "DCT16 ({bx},{by}) [{row},{col}] = {v}");
                }
            }
        }
        // An untouched pixel stays seeded.
        // (3, 0) is empty (DCT8 covers (0,0)..(8,8); (1,1) covers
        // (8,8)..(16,16); (7,0) covers (56,0)..(64,8); (5,0) DCT16
        // covers (40,0)..(56,16); (2,2) DCT16 covers (16,16)..(32,32).
        // So (24, 0) is far from all of these.
        assert_eq!(plane[0 * padded_w + 24], 0.42);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_batched_reconstruct_same_strategy_gpu_dct8_zero() {
        use crate::forks::transform::RAW_STRATEGY_DCT;
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();

        // 4 blocks placed at non-trivial positions on a 64×32 plane.
        let padded_w = 64_usize;
        let padded_h = 32_usize;
        let mut plane = alloc::vec![0.42_f32; padded_w * padded_h]; // seeded
        let coords = [(0_usize, 0_usize), (3, 1), (5, 2), (7, 0)];
        let coeffs = alloc::vec![0.0_f32; coords.len() * 64];

        batched_reconstruct_same_strategy_gpu(
            &enc,
            &coeffs,
            &coords,
            RAW_STRATEGY_DCT,
            &mut plane,
            padded_w,
        );

        // For each placed block, the destination region should be ~0.
        for &(bx, by) in &coords {
            for row in 0..8 {
                for col in 0..8 {
                    let v = plane[(by * 8 + row) * padded_w + bx * 8 + col];
                    assert!(
                        v.abs() < 1e-5,
                        "block ({bx},{by}) row {row} col {col} = {v}"
                    );
                }
            }
        }
        // A block coordinate that wasn't in the recipe stays seeded
        // (e.g., (1, 0) was not in coords).
        let untouched = plane[0 * padded_w + 1 * 8 + 3];
        assert_eq!(untouched, 0.42, "untouched pixel was modified");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_batched_reconstruct_same_strategy_gpu_empty_no_op() {
        use crate::forks::transform::RAW_STRATEGY_DCT;
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let mut plane = alloc::vec![0.5_f32; 64];
        batched_reconstruct_same_strategy_gpu(&enc, &[], &[], RAW_STRATEGY_DCT, &mut plane, 8);
        // Plane unchanged.
        for &v in &plane {
            assert_eq!(v, 0.5);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_idct_and_scatter_one_block_gpu_dct8_zero() {
        // All-zero coefficients → all-zero block → padded plane stays zero
        // at the destination region.
        use crate::forks::transform::RAW_STRATEGY_DCT;
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let padded_w = 32_usize;
        let padded_h = 16_usize;
        let mut plane = alloc::vec![1.0_f32; padded_w * padded_h]; // seeded
        let coeffs = alloc::vec![0.0_f32; 64];
        idct_and_scatter_one_block_gpu(&enc, &coeffs, &mut plane, 2, 1, RAW_STRATEGY_DCT, padded_w);
        // Destination region (16..24, 8..16) should now be ~0.
        for row in 0..8 {
            for col in 0..8 {
                let v = plane[(8 + row) * padded_w + 16 + col];
                assert!(v.abs() < 1e-5, "row {row} col {col} = {v}");
            }
        }
        // A pixel just outside the destination region stays seeded.
        assert_eq!(plane[8 * padded_w + 15], 1.0);
    }

    #[test]
    fn test_scatter_block_to_plane_dct8() {
        use crate::forks::transform::RAW_STRATEGY_DCT;
        // 32×16 padded plane (4 wide × 2 tall blocks).
        let padded_w = 32_usize;
        let padded_h = 16_usize;
        let mut plane = vec![0.0_f32; padded_w * padded_h];
        // Place a constant block at (bx=2, by=1).
        let block: Vec<f32> = (0..64).map(|i| i as f32).collect();
        scatter_block_to_plane(&mut plane, &block, 2, 1, RAW_STRATEGY_DCT, padded_w);
        // Row 0 of the block lands at (px=16, py=8).
        for col in 0..8 {
            assert_eq!(
                plane[8 * padded_w + 16 + col],
                col as f32,
                "row 0 col {col}"
            );
        }
        // Row 7 lands at (px=16, py=15).
        for col in 0..8 {
            assert_eq!(
                plane[15 * padded_w + 16 + col],
                (7 * 8 + col) as f32,
                "row 7 col {col}"
            );
        }
        // Adjacent untouched pixel at (15, 8) still 0.
        assert_eq!(plane[8 * padded_w + 15], 0.0);
        // Pixel just past the block (24, 8) still 0.
        assert_eq!(plane[8 * padded_w + 24], 0.0);
    }

    #[test]
    fn test_scatter_block_to_plane_dct16x16() {
        use crate::forks::transform::RAW_STRATEGY_DCT16X16;
        // 32×16 plane is too small; use 64×32 (8 wide × 4 tall blocks)
        let padded_w = 64_usize;
        let padded_h = 32_usize;
        let mut plane = vec![0.0_f32; padded_w * padded_h];
        let block = vec![0.5_f32; 256]; // 16×16 constant
        // Place a 16×16 block at (bx=2, by=1) → covers pixels
        // (16..32) × (8..24).
        scatter_block_to_plane(&mut plane, &block, 2, 1, RAW_STRATEGY_DCT16X16, padded_w);
        // Spot-check 4 corners + 1 center of the destination region.
        assert_eq!(plane[8 * padded_w + 16], 0.5); // top-left
        assert_eq!(plane[8 * padded_w + 31], 0.5); // top-right
        assert_eq!(plane[23 * padded_w + 16], 0.5); // bot-left
        assert_eq!(plane[23 * padded_w + 31], 0.5); // bot-right
        assert_eq!(plane[15 * padded_w + 23], 0.5); // mid
        // Just outside the block.
        assert_eq!(plane[8 * padded_w + 15], 0.0);
        assert_eq!(plane[24 * padded_w + 16], 0.0);
    }

    #[test]
    fn test_dispatch_restore_llf_dct8_writes_position_0() {
        use crate::forks::transform::RAW_STRATEGY_DCT;
        let mut coeffs = [0.7_f32; 64]; // seed AC positions
        let stride = dispatch_restore_llf(&mut coeffs, &[1.5_f32], RAW_STRATEGY_DCT);
        assert_eq!(stride, 8);
        assert_eq!(coeffs[0], 1.5);
        // Other positions stay seeded.
        for i in 1..64 {
            assert_eq!(coeffs[i], 0.7, "pos {i} unexpectedly modified");
        }
    }

    #[test]
    fn test_dispatch_restore_llf_dct16x16_writes_4_llf_positions() {
        use crate::forks::transform::RAW_STRATEGY_DCT16X16;
        let mut coeffs = [0.7_f32; 256];
        let stride =
            dispatch_restore_llf(&mut coeffs, &[0.5, 0.5, 0.5, 0.5], RAW_STRATEGY_DCT16X16);
        assert_eq!(stride, 16);
        // Constant DC c=0.5 → only LLF[0,0] = c (other 3 LLF positions ~0).
        assert!((coeffs[0] - 0.5).abs() < 1e-5);
        assert!(coeffs[1].abs() < 1e-5);
        assert!(coeffs[16].abs() < 1e-5);
        assert!(coeffs[17].abs() < 1e-5);
        // Other positions stay seeded.
        for &i in &[2_usize, 15, 18, 32, 64, 100, 255] {
            assert_eq!(coeffs[i], 0.7, "pos {i} unexpectedly modified");
        }
    }

    #[test]
    fn test_dispatch_restore_llf_dct32x32_writes_4x4_llf() {
        use crate::forks::transform::RAW_STRATEGY_DCT32X32;
        let mut coeffs = [0.0_f32; 1024];
        let stride = dispatch_restore_llf(&mut coeffs, &[0.5_f32; 16], RAW_STRATEGY_DCT32X32);
        assert_eq!(stride, 32);
        assert!((coeffs[0] - 0.5).abs() < 1e-5);
        // Verify no off-LLF position was touched (sample from far areas).
        for &i in &[5_usize, 32 * 4, 32 * 31 + 31] {
            assert_eq!(coeffs[i], 0.0);
        }
    }

    #[test]
    fn test_restore_llf_dct64x32_zero_in() {
        let r = restore_llf_dct64x32([0.0; 32]);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_restore_llf_dct64x32_constant_dc() {
        // Constant DC c → only out[0] should be ~c (others ~0 by symmetry).
        let c = 0.5_f32;
        let r = restore_llf_dct64x32([c; 32]);
        assert!((r[0] - c).abs() < 1e-4, "got {} expected {}", r[0], c);
        for i in 1..32 {
            assert!(r[i].abs() < 1e-4, "pos {i}: got {} expected 0", r[i]);
        }
    }

    #[test]
    fn test_restore_llf_dct32x64_zero_in() {
        let r = restore_llf_dct32x64([0.0; 32]);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_restore_llf_dct32x64_constant_dc() {
        let c = 0.5_f32;
        let r = restore_llf_dct32x64([c; 32]);
        assert!((r[0] - c).abs() < 1e-4, "got {} expected {}", r[0], c);
        for i in 1..32 {
            assert!(r[i].abs() < 1e-4, "pos {i}: got {} expected 0", r[i]);
        }
    }

    #[test]
    fn test_restore_llf_dct64x64_zero_in() {
        let r = restore_llf_dct64x64([0.0; 64]);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_restore_llf_dct64x64_constant_dc() {
        // Constant DC c: 8-pt DCT of [c]*8 = [8c, 0, ..., 0].
        // After 1/8: [c, 0, ...]. Transpose puts c column 0.
        // Second pass DCT on row 0 [c,0,...,0]: u-transform yields
        // a complex row. But for [c,0,0,0,0,0,0,0]: each dct1d_8
        // step splits into many components — let me just verify
        // shape/finiteness rather than predict the exact pattern.
        let c = 0.5_f32;
        let r = restore_llf_dct64x64([c; 64]);
        // After full 8x8 DCT of [c, c, ..., c] (=64 times constant c),
        // only the (0,0) frequency tap survives the symmetric input.
        // Our restoration: (8c)/8 per row × (8c)/8 per col = c at (0,0)
        // (since SCALE_64_TO_8[0] = 1.0).
        assert!(r[0].is_finite());
        assert!((r[0] - c).abs() < 1e-4, "got {} expected ~{}", r[0], c);
    }

    #[test]
    fn test_restore_llf_dct32x16_zero_in() {
        let r = restore_llf_dct32x16([0.0; 8]);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_restore_llf_dct32x16_constant_dc() {
        // Constant DC across 4×2: only LLF[0] should be non-zero.
        // 2-pt DCT [c,c]→[2c,0]; per-row 4 times → block = [[2c,0],[2c,0],...].
        // Transpose 4×2 → 2×4: t = [[2c,2c,2c,2c],[0,0,0,0]].
        // 4-pt DCT row 0: [2c,2c,2c,2c] → [8c, 0, 0, 0]; row 1 → [0,0,0,0].
        // Divide row 0 col 0: 8c / (1 * 1 * 8) = c. Other positions 0.
        let c = 0.5_f32;
        let r = restore_llf_dct32x16([c; 8]);
        assert!((r[0] - c).abs() < 1e-5);
        for i in 1..8 {
            assert!(r[i].abs() < 1e-5, "pos {i}: got {} expected 0", r[i]);
        }
    }

    #[test]
    fn test_restore_llf_dct16x32_zero_in() {
        let r = restore_llf_dct16x32([0.0; 8]);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_restore_llf_dct16x32_constant_dc() {
        let c = 0.5_f32;
        let r = restore_llf_dct16x32([c; 8]);
        assert!((r[0] - c).abs() < 1e-5);
        for i in 1..8 {
            assert!(r[i].abs() < 1e-5, "pos {i}: got {} expected 0", r[i]);
        }
    }

    #[test]
    fn test_restore_llf_dct32x32_zero_in_zero_out() {
        let r = restore_llf_dct32x32([0.0; 16]);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_restore_llf_dct32x32_constant_dc() {
        // dc_grid = constant c. Forward 4-pt DCT of [c,c,c,c] is
        // [4c, 0, 0, 0] (DC tap = sum, others zero by symmetry).
        // Per-row → block = [[4c,0,0,0],[4c,0,0,0],...]. Transpose →
        // [[4c,4c,4c,4c],[0,...],[0,...],[0,...]]. Forward 4-pt DCT
        // of row 0 [4c,4c,4c,4c] → [16c,0,0,0]; rows 1..3 stay zero.
        // After / (scale * 16): out[0] = 16c / (1*1*16) = c, others 0
        // (or scaled by 0).
        let c = 0.5_f32;
        let r = restore_llf_dct32x32([c; 16]);
        assert!((r[0] - c).abs() < 1e-5);
        for i in 1..16 {
            assert!(r[i].abs() < 1e-5, "pos {i}: got {} expected 0", r[i]);
        }
    }

    // ─── Upstream parity tests for dc_from_dct_* helpers ────────────
    //
    // Validates our hand-rolled forward DC-extraction helpers
    // against jxl-encoder's `pub` dc_from_dct_* functions. Together
    // with the roundtrip tests, this gives full G5.1-compliant
    // parity coverage on the LLF restoration path:
    //
    //   1. `dc_from_dct_X` (mine)  ≡  `jxl_encoder::vardct::dct::dc_from_dct_X` (upstream)  — these tests
    //   2. `restore_llf_dct_X(dc_from_dct_X(llf)) == llf` — roundtrips
    //
    // Together: both directions of the LLF restoration are verified
    // against upstream's exposed forward path, no longer just
    // against each other.

    /// Build a synthetic NxM coefficient block whose only non-zero
    /// values are at the LLF positions provided.
    fn synthesize_block_with_llf<const N: usize>(
        llf_grid: &[f32],
        llf_positions: &[(usize, usize)],
        rows: usize,
        cols: usize,
    ) -> [f32; N] {
        let _ = (rows, cols); // unused, kept for clarity
        debug_assert_eq!(llf_grid.len(), llf_positions.len());
        let mut block = [0.0_f32; N];
        for (i, &(r, c)) in llf_positions.iter().enumerate() {
            block[r * cols + c] = llf_grid[i];
        }
        block
    }

    #[test]
    fn test_dc_from_dct_32x32_matches_upstream() {
        // Synthetic LLF inputs at the 4×4 positions [iy*32+ix].
        // dc_from_dct_32x32 should match jxl_encoder upstream exactly.
        let trials: [[f32; 16]; 4] = [
            {
                let mut a = [0.0_f32; 16];
                a[0] = 1.0;
                a
            },
            {
                let mut a = [0.0_f32; 16];
                a[5] = 1.0;
                a
            },
            [
                3.14, -2.71, 1.41, 0.577, -1.0, 2.0, -3.0, 4.0, 0.1, -0.2, 0.3, -0.4, 5.5, -6.6,
                7.7, -8.8,
            ],
            core::array::from_fn(|i| (i as f32 * 0.13).sin() * 0.5),
        ];
        for (ti, trial) in trials.iter().enumerate() {
            // Build a 1024-coefficient block with LLF at positions
            // [iy*32+ix] for iy, ix in 0..4 (matching where
            // restore_llf_dct32x32's output goes).
            let mut llf_positions = alloc::vec::Vec::with_capacity(16);
            for iy in 0..4 {
                for ix in 0..4 {
                    llf_positions.push((iy, ix));
                }
            }
            let block: [f32; 1024] =
                synthesize_block_with_llf::<1024>(trial, &llf_positions, 32, 32);
            let mine = dc_from_dct_32x32(*trial);
            let theirs = jxl_encoder::vardct::dct::dc_from_dct_32x32(&block);
            for i in 0..16 {
                assert!(
                    (mine[i] - theirs[i]).abs() < 1e-4,
                    "trial {ti} pos {i}: mine={} theirs={}",
                    mine[i],
                    theirs[i]
                );
            }
        }
    }

    #[test]
    fn test_dc_from_dct_16x16_matches_upstream() {
        let trials: [[f32; 4]; 5] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [3.14, -2.71, 1.41, 0.577],
        ];
        for (ti, trial) in trials.iter().enumerate() {
            // Build a 256-coef block with LLF at [0, 1, 16, 17] (stride 16).
            let mut block = [0.0_f32; 256];
            block[0] = trial[0];
            block[1] = trial[1];
            block[16] = trial[2];
            block[17] = trial[3];
            let mine = dc_from_dct_16x16(*trial);
            let theirs = jxl_encoder::vardct::dct::dc_from_dct_16x16(&block);
            for i in 0..4 {
                assert!(
                    (mine[i] - theirs[i]).abs() < 1e-5,
                    "trial {ti} pos {i}: mine={} theirs={}",
                    mine[i],
                    theirs[i]
                );
            }
        }
    }

    #[test]
    fn test_dc_from_dct_16x8_matches_upstream() {
        // DCT16x8 LLF positions are [0] and [1] in the 8-stride layout.
        for &(l0, l1) in &[(1.0_f32, 0.0), (0.0, 1.0), (0.5, -0.3), (3.14, -2.71)] {
            let mut block = [0.0_f32; 128];
            block[0] = l0;
            block[1] = l1;
            let (mine0, mine1) = dc_from_dct_16x8_or_8x16(l0, l1);
            let theirs = jxl_encoder::vardct::dct::dc_from_dct_16x8(&block);
            assert!(
                (mine0 - theirs[0]).abs() < 1e-5,
                "(l0={l0}, l1={l1}) [0]: mine={mine0} theirs={}",
                theirs[0]
            );
            assert!(
                (mine1 - theirs[1]).abs() < 1e-5,
                "(l0={l0}, l1={l1}) [1]: mine={mine1} theirs={}",
                theirs[1]
            );
        }
    }

    #[test]
    fn test_dc_from_dct_16x32_matches_upstream() {
        let trials: [[f32; 8]; 4] = [
            {
                let mut a = [0.0_f32; 8];
                a[0] = 1.0;
                a
            },
            {
                let mut a = [0.0_f32; 8];
                a[3] = 1.0;
                a
            },
            [0.5, -0.3, 0.7, -0.2, 1.1, -1.7, 0.9, -2.3],
            [3.14, -2.71, 1.41, 0.577, -1.0, 2.0, -3.0, 4.0],
        ];
        for (ti, trial) in trials.iter().enumerate() {
            // 16×32 layout: stride 32, LLF at [iy*32+ix] for iy in 0..2, ix in 0..4.
            let mut block = [0.0_f32; 512];
            for iy in 0..2 {
                for ix in 0..4 {
                    block[iy * 32 + ix] = trial[iy * 4 + ix];
                }
            }
            let mine = dc_from_dct_16x32(*trial);
            let theirs = jxl_encoder::vardct::dct::dc_from_dct_16x32(&block);
            for i in 0..8 {
                assert!(
                    (mine[i] - theirs[i]).abs() < 1e-4,
                    "trial {ti} pos {i}: mine={} theirs={}",
                    mine[i],
                    theirs[i]
                );
            }
        }
    }

    #[test]
    fn test_dc_from_dct_64x32_matches_upstream() {
        let trials: [[f32; 32]; 3] = [
            {
                let mut a = [0.0_f32; 32];
                a[0] = 1.0;
                a
            },
            {
                let mut a = [0.0_f32; 32];
                a[15] = 1.0;
                a
            },
            core::array::from_fn(|i| (i as f32 * 0.21).cos() * 0.6),
        ];
        for (ti, trial) in trials.iter().enumerate() {
            // 64×32 layout: stride 64, LLF at [iy*64+ix] for iy in 0..4, ix in 0..8.
            let mut block = alloc::vec![0.0_f32; 2048];
            for iy in 0..4 {
                for ix in 0..8 {
                    block[iy * 64 + ix] = trial[iy * 8 + ix];
                }
            }
            let mine = dc_from_dct_64x32(*trial);
            let theirs = jxl_encoder::vardct::dct::dc_from_dct_64x32(&block);
            for i in 0..32 {
                assert!(
                    (mine[i] - theirs[i]).abs() < 5e-3,
                    "trial {ti} pos {i}: mine={} theirs={}",
                    mine[i],
                    theirs[i]
                );
            }
        }
    }

    #[test]
    fn test_dc_from_dct_32x64_matches_upstream() {
        let trials: [[f32; 32]; 3] = [
            {
                let mut a = [0.0_f32; 32];
                a[0] = 1.0;
                a
            },
            {
                let mut a = [0.0_f32; 32];
                a[15] = 1.0;
                a
            },
            core::array::from_fn(|i| (i as f32 * 0.17).sin() * 0.6),
        ];
        for (ti, trial) in trials.iter().enumerate() {
            // 32×64 layout: stride 64, LLF at [iy*64+ix] for iy in 0..4, ix in 0..8.
            let mut block = alloc::vec![0.0_f32; 2048];
            for iy in 0..4 {
                for ix in 0..8 {
                    block[iy * 64 + ix] = trial[iy * 8 + ix];
                }
            }
            let mine = dc_from_dct_32x64(*trial);
            let theirs = jxl_encoder::vardct::dct::dc_from_dct_32x64(&block);
            for i in 0..32 {
                assert!(
                    (mine[i] - theirs[i]).abs() < 5e-3,
                    "trial {ti} pos {i}: mine={} theirs={}",
                    mine[i],
                    theirs[i]
                );
            }
        }
    }

    #[test]
    fn test_dc_from_dct_64x64_matches_upstream() {
        let trials: [[f32; 64]; 3] = [
            {
                let mut a = [0.0_f32; 64];
                a[0] = 1.0;
                a
            },
            {
                let mut a = [0.0_f32; 64];
                a[27] = 1.0;
                a
            },
            core::array::from_fn(|i| (i as f32 * 0.13).sin() * 0.7),
        ];
        for (ti, trial) in trials.iter().enumerate() {
            let mut llf_positions = alloc::vec::Vec::with_capacity(64);
            for iy in 0..8 {
                for ix in 0..8 {
                    llf_positions.push((iy, ix));
                }
            }
            let block: alloc::vec::Vec<f32> = {
                let mut v = alloc::vec![0.0_f32; 4096];
                for (i, &(r, c)) in llf_positions.iter().enumerate() {
                    v[r * 64 + c] = trial[i];
                }
                v
            };
            let mine = dc_from_dct_64x64(*trial);
            let theirs = jxl_encoder::vardct::dct::dc_from_dct_64x64(&block);
            for i in 0..64 {
                assert!(
                    (mine[i] - theirs[i]).abs() < 5e-3,
                    "trial {ti} pos {i}: mine={} theirs={}",
                    mine[i],
                    theirs[i]
                );
            }
        }
    }

    #[test]
    fn test_dc_from_dct_32x16_matches_upstream() {
        let trials: [[f32; 8]; 3] = [
            {
                let mut a = [0.0_f32; 8];
                a[0] = 1.0;
                a
            },
            [0.5, -0.3, 0.7, -0.2, 1.1, -1.7, 0.9, -2.3],
            [3.14, -2.71, 1.41, 0.577, -1.0, 2.0, -3.0, 4.0],
        ];
        for (ti, trial) in trials.iter().enumerate() {
            // Position in 32-stride layout (= 16 cols + 16 zero).
            // LLF positions: [iy*32+ix] for iy in 0..2, ix in 0..4
            let mut block = [0.0_f32; 512];
            for iy in 0..2 {
                for ix in 0..4 {
                    block[iy * 32 + ix] = trial[iy * 4 + ix];
                }
            }
            let mine = dc_from_dct_32x16(*trial);
            let theirs = jxl_encoder::vardct::dct::dc_from_dct_32x16(&block);
            for i in 0..8 {
                assert!(
                    (mine[i] - theirs[i]).abs() < 1e-4,
                    "trial {ti} pos {i}: mine={} theirs={}",
                    mine[i],
                    theirs[i]
                );
            }
        }
    }

    #[test]
    fn test_dct64x64_forward_inverse_roundtrip() {
        // 5 trial inputs covering corner cases.
        let mut single0 = [0.0_f32; 64];
        single0[0] = 1.0;
        let mut single_mid = [0.0_f32; 64];
        single_mid[27] = 1.0;
        let mut single_last = [0.0_f32; 64];
        single_last[63] = 1.0;
        let mut sign_mixed = [0.0_f32; 64];
        for (i, v) in sign_mixed.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 0.5 } else { -0.5 };
        }
        let arbitrary: [f32; 64] = core::array::from_fn(|i| (i as f32 * 0.13).sin() * 0.7);
        for (ti, trial) in [single0, single_mid, single_last, sign_mixed, arbitrary]
            .iter()
            .enumerate()
        {
            let dc = dc_from_dct_64x64(*trial);
            let restored = restore_llf_dct64x64(dc);
            for i in 0..64 {
                assert!(
                    (restored[i] - trial[i]).abs() < 5e-3,
                    "trial {ti} pos {i}: got {} expected {}",
                    restored[i],
                    trial[i]
                );
            }
        }
    }

    #[test]
    fn test_dct64x32_forward_inverse_roundtrip() {
        let mut single0 = [0.0_f32; 32];
        single0[0] = 1.0;
        let mut single_mid = [0.0_f32; 32];
        single_mid[15] = 1.0;
        let mut single_last = [0.0_f32; 32];
        single_last[31] = 1.0;
        let arbitrary: [f32; 32] = core::array::from_fn(|i| (i as f32 * 0.21).cos() * 0.6);
        for (ti, trial) in [single0, single_mid, single_last, arbitrary]
            .iter()
            .enumerate()
        {
            let dc = dc_from_dct_64x32(*trial);
            let restored = restore_llf_dct64x32(dc);
            for i in 0..32 {
                assert!(
                    (restored[i] - trial[i]).abs() < 5e-3,
                    "trial {ti} pos {i}: got {} expected {}",
                    restored[i],
                    trial[i]
                );
            }
        }
    }

    #[test]
    fn test_dct32x64_forward_inverse_roundtrip() {
        let mut single0 = [0.0_f32; 32];
        single0[0] = 1.0;
        let mut single_mid = [0.0_f32; 32];
        single_mid[15] = 1.0;
        let mut single_last = [0.0_f32; 32];
        single_last[31] = 1.0;
        let arbitrary: [f32; 32] = core::array::from_fn(|i| (i as f32 * 0.17).sin() * 0.6);
        for (ti, trial) in [single0, single_mid, single_last, arbitrary]
            .iter()
            .enumerate()
        {
            let dc = dc_from_dct_32x64(*trial);
            let restored = restore_llf_dct32x64(dc);
            for i in 0..32 {
                assert!(
                    (restored[i] - trial[i]).abs() < 5e-3,
                    "trial {ti} pos {i}: got {} expected {}",
                    restored[i],
                    trial[i]
                );
            }
        }
    }

    #[test]
    fn test_dct32x32_forward_inverse_roundtrip() {
        // 6 trial inputs covering corner cases.
        let trials: [[f32; 16]; 6] = [
            // single non-zero LLF positions to catch index/sign errors
            {
                let mut a = [0.0_f32; 16];
                a[0] = 1.0;
                a
            },
            {
                let mut a = [0.0_f32; 16];
                a[5] = 1.0;
                a
            },
            {
                let mut a = [0.0_f32; 16];
                a[15] = 1.0;
                a
            },
            // sign-mixed
            {
                let mut a = [0.0_f32; 16];
                for (i, v) in a.iter_mut().enumerate() {
                    *v = if i % 2 == 0 { 0.5 } else { -0.5 };
                }
                a
            },
            // checkerboard
            {
                let mut a = [0.0_f32; 16];
                for (i, v) in a.iter_mut().enumerate() {
                    *v = if (i / 4 + i % 4) % 2 == 0 { 1.0 } else { -1.0 };
                }
                a
            },
            // arbitrary
            [
                3.14, -2.71, 1.41, 0.577, -1.0, 2.0, -3.0, 4.0, 0.1, -0.2, 0.3, -0.4, 5.5, -6.6,
                7.7, -8.8,
            ],
        ];
        for (ti, trial) in trials.iter().enumerate() {
            let dc = dc_from_dct_32x32(*trial);
            let restored = restore_llf_dct32x32(dc);
            for i in 0..16 {
                assert!(
                    (restored[i] - trial[i]).abs() < 1e-3,
                    "trial {ti} pos {i}: got {} expected {}",
                    restored[i],
                    trial[i]
                );
            }
        }
    }

    #[test]
    fn test_dct32x16_forward_inverse_roundtrip() {
        let trials: [[f32; 8]; 5] = [
            {
                let mut a = [0.0_f32; 8];
                a[0] = 1.0;
                a
            },
            {
                let mut a = [0.0_f32; 8];
                a[3] = 1.0;
                a
            },
            {
                let mut a = [0.0_f32; 8];
                a[7] = 1.0;
                a
            },
            [0.5, -0.3, 0.7, -0.2, 1.1, -1.7, 0.9, -2.3],
            [3.14, -2.71, 1.41, 0.577, -1.0, 2.0, -3.0, 4.0],
        ];
        for (ti, trial) in trials.iter().enumerate() {
            let dc = dc_from_dct_32x16(*trial);
            let restored = restore_llf_dct32x16(dc);
            for i in 0..8 {
                assert!(
                    (restored[i] - trial[i]).abs() < 1e-4,
                    "trial {ti} pos {i}: got {} expected {}",
                    restored[i],
                    trial[i]
                );
            }
        }
    }

    #[test]
    fn test_dct16x32_forward_inverse_roundtrip() {
        let trials: [[f32; 8]; 5] = [
            {
                let mut a = [0.0_f32; 8];
                a[0] = 1.0;
                a
            },
            {
                let mut a = [0.0_f32; 8];
                a[3] = 1.0;
                a
            },
            {
                let mut a = [0.0_f32; 8];
                a[7] = 1.0;
                a
            },
            [0.5, -0.3, 0.7, -0.2, 1.1, -1.7, 0.9, -2.3],
            [3.14, -2.71, 1.41, 0.577, -1.0, 2.0, -3.0, 4.0],
        ];
        for (ti, trial) in trials.iter().enumerate() {
            let dc = dc_from_dct_16x32(*trial);
            let restored = restore_llf_dct16x32(dc);
            for i in 0..8 {
                assert!(
                    (restored[i] - trial[i]).abs() < 1e-4,
                    "trial {ti} pos {i}: got {} expected {}",
                    restored[i],
                    trial[i]
                );
            }
        }
    }

    #[test]
    fn test_dct16x16_forward_inverse_roundtrip() {
        // Strong parity: forward then inverse should recover the LLF values.
        // Catches sign / scale / transpose bugs that the constant-DC test misses.
        for trial in &[
            [1.0_f32, 0.0, 0.0, 0.0],
            [0.0_f32, 1.0, 0.0, 0.0],
            [0.0_f32, 0.0, 1.0, 0.0],
            [0.0_f32, 0.0, 0.0, 1.0],
            [0.5_f32, -0.3, 0.7, -0.2],
            [3.14_f32, -2.71, 1.41, 0.577],
        ] {
            let dc = dc_from_dct_16x16(*trial);
            let restored = restore_llf_dct16x16(dc);
            for i in 0..4 {
                assert!(
                    (restored[i] - trial[i]).abs() < 1e-5,
                    "trial {trial:?} pos {i}: got {} expected {}",
                    restored[i],
                    trial[i]
                );
            }
        }
    }

    #[test]
    fn test_dct16x16_inverse_forward_roundtrip() {
        // Other direction: dc → llf → dc should recover the dc values.
        for trial in &[
            [1.0_f32, 0.0, 0.0, 0.0],
            [0.0_f32, 1.0, 0.0, 0.0],
            [0.5_f32, -0.3, 0.7, -0.2],
            [12.0_f32, -7.0, 4.5, -1.1],
        ] {
            let llf = restore_llf_dct16x16(*trial);
            let dc = dc_from_dct_16x16(llf);
            for i in 0..4 {
                assert!(
                    (dc[i] - trial[i]).abs() < 1e-5,
                    "trial {trial:?} pos {i}: got {} expected {}",
                    dc[i],
                    trial[i]
                );
            }
        }
    }

    #[test]
    fn test_dct16x8_or_8x16_forward_inverse_roundtrip() {
        // Both directions for the 2-point case.
        for &(l0, l1) in &[(1.0_f32, 0.0), (0.0, 1.0), (0.5, -0.3), (3.14, -2.71)] {
            let (dc0, dc1) = dc_from_dct_16x8_or_8x16(l0, l1);
            let restored = restore_llf_dct16x8_or_8x16(dc0, dc1);
            assert!((restored[0] - l0).abs() < 1e-5);
            assert!((restored[1] - l1).abs() < 1e-5);
        }
    }

    #[test]
    fn test_restore_llf_dct16x16_zero_dc_yields_zero_llf() {
        // All zeros in → all zeros out.
        let r = restore_llf_dct16x16([0.0; 4]);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_restore_llf_dct16x16_constant_dc_yields_dc_only() {
        // dc_grid = [c, c, c, c] → h00 = 4c, h01=h10=h11=0.
        // llf00 = 4c / (4 * s0^2) = c (since s0 = 1.0)
        let c = 0.5_f32;
        let r = restore_llf_dct16x16([c, c, c, c]);
        assert!((r[0] - c).abs() < 1e-6);
        assert!(r[1].abs() < 1e-6);
        assert!(r[2].abs() < 1e-6);
        assert!(r[3].abs() < 1e-6);
    }

    #[test]
    fn test_restore_dct8_dc_override_y() {
        // Y channel: dq_y[0] = quant_dc_y / inv_factor[1]
        // inv_factor[1] = 512 * scale_dc
        let mut dq_x = [0.0_f32; 64];
        let mut dq_y = [0.0_f32; 64];
        let mut dq_b = [0.0_f32; 64];
        let quant_dc_y = 100.0_f32;
        let scale_dc = 0.5_f32;
        restore_dct8_dc_override(
            &mut dq_x, &mut dq_y, &mut dq_b, 0.0, quant_dc_y, 0.0, scale_dc,
        );
        // dq_y[0] = 100 / (512 * 0.5) = 100 / 256 = 0.390625
        assert!((dq_y[0] - 0.390_625).abs() < 1e-6);
    }

    #[test]
    fn test_restore_dct8_dc_override_b_includes_y_cfl() {
        // B channel includes 0.5 * Y DC contribution.
        let mut dq_x = [0.0_f32; 64];
        let mut dq_y = [0.0_f32; 64];
        let mut dq_b = [0.0_f32; 64];
        // quant_dc_b=0, quant_dc_y=10, scale_dc=1.0
        // dq_b[0] = (0 + 10 * 0.5) / (256 * 1.0) = 5 / 256 = 0.01953125
        restore_dct8_dc_override(&mut dq_x, &mut dq_y, &mut dq_b, 0.0, 10.0, 0.0, 1.0);
        assert!((dq_b[0] - 0.019_531_25).abs() < 1e-6);
        // dq_x[0] = 0 / 4096 = 0
        assert_eq!(dq_x[0], 0.0);
        // dq_y[0] = 10 / 512 = 0.01953125
        assert!((dq_y[0] - 0.019_531_25).abs() < 1e-6);
    }

    #[test]
    fn test_restore_dct8_dc_override_batched_matches_per_block() {
        // Run both forms on the same inputs; outputs must agree exactly.
        const N: usize = 5;
        let mut dq_x_batch = vec![0.0_f32; N * 64];
        let mut dq_y_batch = vec![0.0_f32; N * 64];
        let mut dq_b_batch = vec![0.0_f32; N * 64];
        // Seed AC slots to ensure we don't touch them.
        for i in 0..N * 64 {
            if !i.is_multiple_of(64) {
                dq_x_batch[i] = (i as f32) * 0.001;
                dq_y_batch[i] = (i as f32) * 0.002;
                dq_b_batch[i] = (i as f32) * 0.003;
            }
        }
        let qx: Vec<f32> = (0..N).map(|b| 1.0 + b as f32 * 2.0).collect();
        let qy: Vec<f32> = (0..N).map(|b| 5.0 + b as f32 * 3.0).collect();
        let qb: Vec<f32> = (0..N).map(|b| -3.0 + b as f32).collect();
        let scale_dc = 0.7_f32;

        // Per-block reference.
        let mut dq_x_ref = dq_x_batch.clone();
        let mut dq_y_ref = dq_y_batch.clone();
        let mut dq_b_ref = dq_b_batch.clone();
        for b in 0..N {
            let block_x: &mut [f32; 64] = (&mut dq_x_ref[b * 64..b * 64 + 64]).try_into().unwrap();
            let block_y: &mut [f32; 64] = (&mut dq_y_ref[b * 64..b * 64 + 64]).try_into().unwrap();
            let block_b: &mut [f32; 64] = (&mut dq_b_ref[b * 64..b * 64 + 64]).try_into().unwrap();
            restore_dct8_dc_override(block_x, block_y, block_b, qx[b], qy[b], qb[b], scale_dc);
        }
        // Batched.
        restore_dct8_dc_override_batched(
            &mut dq_x_batch,
            &mut dq_y_batch,
            &mut dq_b_batch,
            &qx,
            &qy,
            &qb,
            scale_dc,
        );
        assert_eq!(dq_x_batch, dq_x_ref);
        assert_eq!(dq_y_batch, dq_y_ref);
        assert_eq!(dq_b_batch, dq_b_ref);
    }

    #[test]
    fn test_restore_dct8_dc_override_does_not_touch_ac() {
        // AC slots [1..64] must stay unchanged.
        let mut dq_x = [0.5_f32; 64];
        let mut dq_y = [0.7_f32; 64];
        let mut dq_b = [0.3_f32; 64];
        restore_dct8_dc_override(&mut dq_x, &mut dq_y, &mut dq_b, 1.0, 2.0, 3.0, 1.0);
        for i in 1..64 {
            assert_eq!(dq_x[i], 0.5);
            assert_eq!(dq_y[i], 0.7);
            assert_eq!(dq_b[i], 0.3);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_reconstruct_xyb_dct8_only_gpu_zero_input() {
        // All-zero quant + zero CfL → output is all zeros (DC=0, AC=0).
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let xb = 2_usize;
        let yb = 2_usize;
        let nb = xb * yb;
        let zeros_dc = vec![0.0_f32; nb];
        let zeros_ac_i = vec![0_i32; nb * 64];
        let weights_one = [1.0_f32; 64];
        let qac_qm = vec![1.0_f32; nb];
        let zero_factor = vec![0.0_f32; nb];

        let planes = reconstruct_xyb_dct8_only_gpu(
            &enc,
            &zeros_dc,
            &zeros_dc,
            &zeros_dc,
            &zeros_ac_i,
            &zeros_ac_i,
            &zeros_ac_i,
            &weights_one,
            &weights_one,
            &weights_one,
            &qac_qm,
            &qac_qm,
            &qac_qm,
            &zero_factor,
            &zero_factor,
            1.0,
            xb,
            yb,
        );
        for p in &planes {
            assert_eq!(p.len(), xb * 8 * yb * 8);
            for &v in p {
                assert!(v.abs() < 1e-6, "expected ~0, got {v}");
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_reconstruct_xyb_dct8_only_gpu_constant_dc() {
        // Constant DC across all blocks → constant output per channel.
        // Set quant_dc_y = 100, scale_dc = 1.0 → DC = 100/512 = 0.1953
        // After IDCT (which scales by 1/8 per dim → 1/8 from the DC tap),
        // each pixel = 0.1953 / 8 = 0.02441 (libjxl IDCT normalization).
        // We just check that the output is constant per plane and finite.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let xb = 1_usize;
        let yb = 1_usize;
        let nb = xb * yb;
        let dc_y = vec![100.0_f32; nb];
        let dc_zero = vec![0.0_f32; nb];
        let zeros_ac_i = vec![0_i32; nb * 64];
        let weights_one = [1.0_f32; 64];
        let qac_qm = vec![1.0_f32; nb];
        let zero_factor = vec![0.0_f32; nb];

        let planes = reconstruct_xyb_dct8_only_gpu(
            &enc,
            &dc_zero,
            &dc_y,
            &dc_zero,
            &zeros_ac_i,
            &zeros_ac_i,
            &zeros_ac_i,
            &weights_one,
            &weights_one,
            &weights_one,
            &qac_qm,
            &qac_qm,
            &qac_qm,
            &zero_factor,
            &zero_factor,
            1.0,
            xb,
            yb,
        );
        // Y plane should be constant non-zero; X plane zero; B plane non-zero
        // due to DC-CfL: dc_b = (0 + 100*0.5)/256 = 0.1953
        let v0_y = planes[1][0];
        for &v in &planes[1] {
            assert!((v - v0_y).abs() < 1e-5, "Y not constant: {v} vs {v0_y}");
        }
        assert!(v0_y.abs() > 0.0);
        for &v in &planes[0] {
            assert!(v.abs() < 1e-6, "X should be 0, got {v}");
        }
        // B is non-zero (Y CfL contribution).
        let v0_b = planes[2][0];
        assert!(v0_b.abs() > 0.0);
        for &v in &planes[2] {
            assert!((v - v0_b).abs() < 1e-5);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gab_smooth_uniform_gpu() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let w = 16;
        let h = 16;
        // Uniform image stays uniform under symmetric blur.
        let mut planes = [
            vec![0.5_f32; w * h],
            vec![0.3_f32; w * h],
            vec![0.7_f32; w * h],
        ];
        gab_smooth_gpu(&enc, &mut planes, w, h);
        for &v in &planes[0] {
            assert!((v - 0.5).abs() < 1e-4);
        }
        for &v in &planes[1] {
            assert!((v - 0.3).abs() < 1e-4);
        }
        for &v in &planes[2] {
            assert!((v - 0.7).abs() < 1e-4);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_xyb_to_linear_rgb_planar_gpu_finite() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n = 64;
        let xyb_x: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.001).collect();
        let xyb_y: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.005).collect();
        let xyb_b: Vec<f32> = (0..n).map(|i| 0.05 + (i as f32) * 0.003).collect();
        let mut r = vec![0.0_f32; n];
        let mut g = vec![0.0_f32; n];
        let mut b = vec![0.0_f32; n];
        xyb_to_linear_rgb_planar_gpu(&enc, &xyb_x, &xyb_y, &xyb_b, &mut r, &mut g, &mut b, n);
        for i in 0..n {
            assert!(r[i].is_finite(), "r[{i}] not finite");
            assert!(g[i].is_finite(), "g[{i}] not finite");
            assert!(b[i].is_finite(), "b[{i}] not finite");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_xyb_roundtrip_via_gpu() {
        // Forward XYB then inverse XYB on GPU should round-trip linear RGB.
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let n = 256;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n)
            .map(|i| 0.2 + 0.5 * ((i + 7) as f32 / n as f32))
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| 0.3 + 0.4 * ((i + 13) as f32 / n as f32))
            .collect();
        let (xx, xy, xb) = enc.xyb_from_linear_rgb(&r, &g, &b);
        let (r2, g2, b2) = enc.xyb_to_linear_rgb_planar(&xx, &xy, &xb);
        let mut max_err = 0.0_f32;
        for i in 0..n {
            max_err = max_err.max((r[i] - r2[i]).abs());
            max_err = max_err.max((g[i] - g2[i]).abs());
            max_err = max_err.max((b[i] - b2[i]).abs());
        }
        // XYB roundtrip is not bit-exact (cube-root → cube can drift) but
        // should be well below 1e-3 absolute on normal RGB inputs.
        assert!(
            max_err < 5e-4,
            "XYB roundtrip drift too large: {max_err:.3e}"
        );
    }
}
