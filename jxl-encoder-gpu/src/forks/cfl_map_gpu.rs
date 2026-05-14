// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! GPU compute_cfl_map: computes per-tile CfL multipliers (ytox, ytob)
//! directly from GPU-resident XYB planes, replacing the CPU
//! `compute_cfl_map` for the all-DCT8 fast path.
//!
//! Pipeline:
//!   1. gather_blocks_persistent on each XYB plane (8x8 blocks)
//!   2. dct_8x8_persistent on each
//!   3. cfl_collect: tile-major scatter + zero-DC + scale by inv_qm
//!      (one cube per block, 64 threads each)
//!   4. find_best_multipliers_newton_batch_gpu × 2 (X and B)
//!   5. download tiny ytox/ytob (a few KB)
//!
//! Key optimization: keeps everything GPU-resident through step 4,
//! eliminating the 60ms full-XYB DtoH download + 28ms CPU repack +
//! 9ms CPU compute_cfl_map for the fast path.
//!
//! Edge tiles: padded with 0.0 in the m/s buffers. Newton iterations
//! treat zeros as no-op (sum_aa, sum_ab unchanged), and the
//! regularization term `n_f * distance_mul * 0.5` with
//! `distance_mul = 1e-9` and max `n_f = 4096` is `~2e-6` — negligible
//! compared to typical `sum_aa` values, so the LS estimate matches
//! the CPU-correct per-tile-actual-n result.

use cubecl::prelude::*;

use crate::encoder::GpuEncoder;
use crate::persistent::GpuPlane;

/// libjxl `compute_cfl_map` constants.
const TILE_DIM_IN_BLOCKS: usize = 8;
const COEFFS_PER_BLOCK: usize = 64;
const VALUES_PER_TILE: usize = TILE_DIM_IN_BLOCKS * TILE_DIM_IN_BLOCKS * COEFFS_PER_BLOCK;
const K_DISTANCE_MULTIPLIER_AC: f32 = 1e-9;

/// libjxl quant weight tables for DCT8 channels 0 and 2.
fn inv_qm_table(channel: usize) -> [f32; 64] {
    let qw = jxl_encoder::__pre_quantized::quant_weights_dct8(channel);
    let mut inv = [0.0_f32; 64];
    for i in 0..64 {
        inv[i] = 1.0 / qw[i];
    }
    inv
}

/// CfL map output, mirrors `jxl_encoder::__pre_quantized::CflMap`
/// shape (i8 ytox/ytob arrays per tile).
pub struct CflMapResult {
    pub ytox: alloc::vec::Vec<i8>,
    pub ytob: alloc::vec::Vec<i8>,
    pub xsize_tiles: usize,
    pub ysize_tiles: usize,
}

/// Compute CfL map entirely on GPU. Inputs are GPU-resident XYB
/// planes (gpu-padded, gpu_pw × gpu_ph). Outputs are CPU-side
/// tiny i8 arrays.
///
/// `xsize_blocks`, `ysize_blocks`: CPU-aligned block grid (cpu_pw / 8,
/// cpu_ph / 8). The kernel filters out blocks past these bounds.
///
/// `use_newton`: true → call Newton-refined multiplier search; false →
/// LS-only (matching `compute_cfl_map`'s use_newton parameter).
pub fn compute_cfl_map_gpu_persistent<R: Runtime>(
    enc: &GpuEncoder<R>,
    xyb_x: &GpuPlane<R>,
    xyb_y: &GpuPlane<R>,
    xyb_b: &GpuPlane<R>,
    xsize_blocks: usize,
    ysize_blocks: usize,
    use_newton: bool,
    newton_eps: f32,
    newton_max_iters: usize,
) -> CflMapResult {
    let xsize_tiles = xsize_blocks.div_ceil(TILE_DIM_IN_BLOCKS);
    let ysize_tiles = ysize_blocks.div_ceil(TILE_DIM_IN_BLOCKS);
    let num_tiles = xsize_tiles * ysize_tiles;

    // Step 1: gather XYB into 8x8 blocks (GPU-padded layout).
    let g_x = enc.gather_blocks_persistent(xyb_x, 8, 8);
    let g_y = enc.gather_blocks_persistent(xyb_y, 8, 8);
    let g_b = enc.gather_blocks_persistent(xyb_b, 8, 8);

    // Step 2: DCT8 each.
    let dct_x = enc.dct_8x8_persistent(&g_x);
    let dct_y = enc.dct_8x8_persistent(&g_y);
    let dct_b = enc.dct_8x8_persistent(&g_b);

    // Step 3: tile-major scatter + zero-DC + scale.
    let inv_qm_x = inv_qm_table(0);
    let inv_qm_b = inv_qm_table(2);
    let h_iqx = enc
        .client_ref()
        .create_from_slice(f32::as_bytes(&inv_qm_x));
    let h_iqb = enc
        .client_ref()
        .create_from_slice(f32::as_bytes(&inv_qm_b));
    let out_floats = num_tiles * VALUES_PER_TILE;
    // No zero-init upload: the kernel writes EVERY slot (data for
    // active blocks, 0.0 for padding slots) so empty() suffices.
    // Avoids a 4 × 49 MB pageable HtoD upload at 12 MP.
    let h_m_yx = enc.client_ref().empty(out_floats * 4);
    let h_s_x = enc.client_ref().empty(out_floats * 4);
    let h_m_yb = enc.client_ref().empty(out_floats * 4);
    let h_s_b = enc.client_ref().empty(out_floats * 4);
    let in_floats = (dct_y.num_blocks() as usize) * COEFFS_PER_BLOCK;
    let gpu_xsize_blocks = (xyb_y.width() / 8) as u32;
    crate::launch::cfl_collect::cfl_collect::<R>(
        enc.client_ref(),
        dct_y.handle().clone(),
        dct_x.handle().clone(),
        dct_b.handle().clone(),
        h_iqx,
        h_iqb,
        h_m_yx.clone(),
        h_s_x.clone(),
        h_m_yb.clone(),
        h_s_b.clone(),
        num_tiles as u32,
        in_floats,
        out_floats,
        xsize_blocks as u32,
        ysize_blocks as u32,
        gpu_xsize_blocks,
        xsize_tiles as u32,
    );

    // Step 4: find_best_multiplier_newton (per-tile) for X and B.
    // bases: 0.0 for X, 1.0 for B (matching compute_cfl_map line 363, 373).
    let bases_x = alloc::vec![0.0f32; num_tiles];
    let bases_b = alloc::vec![1.0f32; num_tiles];
    let h_bx = enc
        .client_ref()
        .create_from_slice(f32::as_bytes(&bases_x));
    let h_bb = enc
        .client_ref()
        .create_from_slice(f32::as_bytes(&bases_b));
    let h_out_x = enc
        .client_ref()
        .create_from_slice(i32::as_bytes(&alloc::vec![0_i32; num_tiles]));
    let h_out_b = enc
        .client_ref()
        .create_from_slice(i32::as_bytes(&alloc::vec![0_i32; num_tiles]));

    if use_newton {
        crate::launch::cfl::find_best_multiplier_newton::<R>(
            enc.client_ref(),
            h_m_yx.clone(),
            h_s_x.clone(),
            h_bx,
            h_out_x.clone(),
            num_tiles as u32,
            VALUES_PER_TILE as u32,
            K_DISTANCE_MULTIPLIER_AC,
            newton_eps,
            newton_max_iters as u32,
        );
        crate::launch::cfl::find_best_multiplier_newton::<R>(
            enc.client_ref(),
            h_m_yb.clone(),
            h_s_b.clone(),
            h_bb,
            h_out_b.clone(),
            num_tiles as u32,
            VALUES_PER_TILE as u32,
            K_DISTANCE_MULTIPLIER_AC,
            newton_eps,
            newton_max_iters as u32,
        );
    } else {
        crate::launch::cfl::find_best_multiplier::<R>(
            enc.client_ref(),
            h_m_yx.clone(),
            h_s_x.clone(),
            h_bx,
            h_out_x.clone(),
            num_tiles as u32,
            VALUES_PER_TILE as u32,
            K_DISTANCE_MULTIPLIER_AC,
        );
        crate::launch::cfl::find_best_multiplier::<R>(
            enc.client_ref(),
            h_m_yb.clone(),
            h_s_b.clone(),
            h_bb,
            h_out_b.clone(),
            num_tiles as u32,
            VALUES_PER_TILE as u32,
            K_DISTANCE_MULTIPLIER_AC,
        );
    }

    // Step 5: ONE batched download of tiny i32 arrays (~12 KB total).
    let mut bytes = enc
        .client_ref()
        .read(alloc::vec![h_out_x, h_out_b]);
    let bb = bytes.pop().expect("read[1]");
    let xb = bytes.pop().expect("read[0]");
    let xs = i32::from_bytes(&xb);
    let bs = i32::from_bytes(&bb);
    let mut ytox = alloc::vec![0i8; num_tiles];
    let mut ytob = alloc::vec![0i8; num_tiles];
    for i in 0..num_tiles {
        ytox[i] = xs[i].clamp(-128, 127) as i8;
        ytob[i] = bs[i].clamp(-128, 127) as i8;
    }
    CflMapResult {
        ytox,
        ytob,
        xsize_tiles,
        ysize_tiles,
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use jxl_encoder::__pre_quantized::compute_cfl_map;

    type R = cubecl::cuda::CudaRuntime;

    /// Verify cfl_collect produces non-zero m/s buffers for non-zero
    /// input. Diagnostic for the parity test failure.
    #[test]
    fn cfl_collect_writes_data() {
        let enc: GpuEncoder<R> = GpuEncoder::new();
        let xsize_blocks = 8usize;
        let ysize_blocks = 8usize;
        let pw = 64;
        let ph = 64;
        let n = pw * ph;
        let mut x = alloc::vec![0.0f32; n];
        let mut y = alloc::vec![0.0f32; n];
        let mut b = alloc::vec![0.0f32; n];
        for i in 0..n {
            x[i] = (i as f32 * 0.1).sin();
            y[i] = 0.5 + (i as f32 * 0.13).cos();
            b[i] = 0.3 + (i as f32 * 0.17).sin();
        }
        let xx = enc.upload_plane(&x, pw as u32, ph as u32);
        let xy = enc.upload_plane(&y, pw as u32, ph as u32);
        let xb = enc.upload_plane(&b, pw as u32, ph as u32);
        let g_x = enc.gather_blocks_persistent(&xx, 8, 8);
        let g_y = enc.gather_blocks_persistent(&xy, 8, 8);
        let g_b = enc.gather_blocks_persistent(&xb, 8, 8);
        let dct_x = enc.dct_8x8_persistent(&g_x);
        let dct_y = enc.dct_8x8_persistent(&g_y);
        let dct_b = enc.dct_8x8_persistent(&g_b);
        let inv_qm_x = inv_qm_table(0);
        let inv_qm_b = inv_qm_table(2);
        let h_iqx = enc.client_ref().create_from_slice(f32::as_bytes(&inv_qm_x));
        let h_iqb = enc.client_ref().create_from_slice(f32::as_bytes(&inv_qm_b));
        let xsize_tiles = 1usize;
        let num_tiles = 1usize;
        let out_floats = num_tiles * VALUES_PER_TILE;
        let zeros: alloc::vec::Vec<f32> = alloc::vec![0.0; out_floats];
        let h_zero_bytes = f32::as_bytes(&zeros);
        let h_m_yx = enc.client_ref().create_from_slice(h_zero_bytes);
        let h_s_x = enc.client_ref().create_from_slice(h_zero_bytes);
        let h_m_yb = enc.client_ref().create_from_slice(h_zero_bytes);
        let h_s_b = enc.client_ref().create_from_slice(h_zero_bytes);
        let in_floats = (dct_y.num_blocks() as usize) * COEFFS_PER_BLOCK;
        let gpu_xsize_blocks = (xy.width() / 8) as u32;
        crate::launch::cfl_collect::cfl_collect::<R>(
            enc.client_ref(),
            dct_y.handle().clone(),
            dct_x.handle().clone(),
            dct_b.handle().clone(),
            h_iqx,
            h_iqb,
            h_m_yx.clone(),
            h_s_x.clone(),
            h_m_yb.clone(),
            h_s_b.clone(),
            num_tiles as u32,
            in_floats,
            out_floats,
            xsize_blocks as u32,
            ysize_blocks as u32,
            gpu_xsize_blocks,
            xsize_tiles as u32,
        );
        let mut bytes = enc.client_ref().read(alloc::vec![h_m_yx.clone(), h_s_x.clone()]);
        let sb = bytes.pop().expect("read[1]");
        let mb = bytes.pop().expect("read[0]");
        let m = f32::from_bytes(&mb);
        let s = f32::from_bytes(&sb);
        let m_max = m.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));
        let s_max = s.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));
        eprintln!("m_max={m_max:.4}, s_max={s_max:.4}");
        eprintln!("m[0..8]={:?}", &m[0..8]);
        eprintln!("s[0..8]={:?}", &s[0..8]);
        eprintln!("m[64..72]={:?}", &m[64..72]); // block 1 of tile 0
        assert!(m_max > 0.0, "m_yx should be non-zero");
        assert!(s_max > 0.0, "s_x should be non-zero");
    }

    /// Build a synthetic XYB image (small, deterministic) and verify
    /// the GPU CfL map matches the CPU reference within `±1` per tile
    /// (rounding noise on borderline values).
    #[test]
    fn cfl_map_gpu_matches_cpu_small() {
        let enc: GpuEncoder<R> = GpuEncoder::new();
        let xsize_blocks = 24usize; // 3 tiles
        let ysize_blocks = 16usize; // 2 tiles
        let pw = xsize_blocks * 8;
        let ph = ysize_blocks * 8;
        let n = pw * ph;

        let mut x = alloc::vec![0.0_f32; n];
        let mut y = alloc::vec![0.0_f32; n];
        let mut b = alloc::vec![0.0_f32; n];
        // High-frequency content so the AC coefficients have real
        // signal — Newton fitting on near-zero values is dominated by
        // FP noise.
        for py in 0..ph {
            for px in 0..pw {
                let i = py * pw + px;
                let fx = px as f32;
                let fy = py as f32;
                let v = (fx * 0.31 + fy * 0.17).sin()
                    + 0.5 * (fx * 0.79).cos()
                    + 0.3 * (fy * 0.41).sin();
                y[i] = 0.5 + 0.1 * v;
                x[i] = 0.02 * v + 0.005 * (fx * 0.61 + fy * 0.23).sin();
                b[i] = 0.6 + 0.08 * v + 0.015 * (fx * 0.43).cos();
            }
        }

        // Upload XYB to persistent GPU planes.
        let xx_g = enc.upload_plane(&x, pw as u32, ph as u32);
        let xy_g = enc.upload_plane(&y, pw as u32, ph as u32);
        let xb_g = enc.upload_plane(&b, pw as u32, ph as u32);

        let gpu = compute_cfl_map_gpu_persistent(
            &enc,
            &xx_g,
            &xy_g,
            &xb_g,
            xsize_blocks,
            ysize_blocks,
            true,
            1e-3,
            10,
        );

        let cpu = compute_cfl_map(
            &x,
            &y,
            &b,
            pw,
            ph,
            xsize_blocks,
            ysize_blocks,
            true,
            1e-3,
            10,
        );

        assert_eq!(gpu.xsize_tiles, cpu.xsize_tiles);
        assert_eq!(gpu.ysize_tiles, cpu.ysize_tiles);
        for t in 0..gpu.xsize_tiles * gpu.ysize_tiles {
            let gx = gpu.ytox[t] as i32;
            let cx = cpu.ytox[t] as i32;
            let gb = gpu.ytob[t] as i32;
            let cb = cpu.ytob[t] as i32;
            assert!(
                (gx - cx).abs() <= 1,
                "tile {t}: gpu ytox={gx}, cpu ytox={cx}"
            );
            assert!(
                (gb - cb).abs() <= 1,
                "tile {t}: gpu ytob={gb}, cpu ytob={cb}"
            );
        }
    }
}
