// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause) via jxl-encoder-simd.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! 8x8 forward and inverse DCT.
//!
//! Mirrors `jxl_encoder_simd::dct8::{dct_8x8_scalar, idct_8x8_scalar}`.
//!
//! Strategy: one thread per 8x8 block (cube_dim = 1, cube_count = num_blocks).
//! Per-block scratch lives in `SharedMemory<f32>::new(64)` (single-threaded
//! cube → effectively private after codegen). Higher-throughput cooperative
//! variant is a future optimization once parity is locked.

// Butterfly factorization is more readable as `x = x op y` than `x op= y`;
// keeping it parallel to the CPU `_scalar` reference for parity audits.
#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

const SQRT2: f32 = core::f32::consts::SQRT_2;
const ONE_OVER_SQRT2: f32 = 0.707_106_77;
const ONE_OVER_8: f32 = 0.125;

const WC_M4_0: f32 = 0.541_196_1;
const WC_M4_1: f32 = 1.306_563;
const WC_M8_0: f32 = 0.509_795_6;
const WC_M8_1: f32 = 0.601_344_9;
const WC_M8_2: f32 = 0.899_976_2;
const WC_M8_3: f32 = 2.562_915_5;
const INV_WC_M4_0: f32 = 1.0 / WC_M4_0;
const INV_WC_M4_1: f32 = 1.0 / WC_M4_1;
const INV_WC_M8_0: f32 = 1.0 / WC_M8_0;
const INV_WC_M8_1: f32 = 1.0 / WC_M8_1;
const INV_WC_M8_2: f32 = 1.0 / WC_M8_2;
const INV_WC_M8_3: f32 = 1.0 / WC_M8_3;

/// 1D 8-point DCT (libjxl factorization). Operates in-place on `mem`.
/// `base` is the offset of the row's first element in the shared array
/// (in usize units).
#[cube]
fn dct1d_8(mem: &mut SharedMemory<f32>, base: u32) {
    let b0 = base as usize;
    let m0 = mem[b0];
    let m1 = mem[b0 + 1usize];
    let m2 = mem[b0 + 2usize];
    let m3 = mem[b0 + 3usize];
    let m4 = mem[b0 + 4usize];
    let m5 = mem[b0 + 5usize];
    let m6 = mem[b0 + 6usize];
    let m7 = mem[b0 + 7usize];

    let mut t0 = m0 + m7;
    let mut t1 = m1 + m6;
    let mut t2 = m2 + m5;
    let mut t3 = m3 + m4;
    let mut t4 = m0 - m7;
    let mut t5 = m1 - m6;
    let mut t6 = m2 - m5;
    let mut t7 = m3 - m4;

    // dct1d_4 on [t0..t4]
    let a0 = t0 + t3;
    let a1 = t1 + t2;
    let a2 = t0 - t3;
    let a3 = t1 - t2;
    let b0v = a0 + a1;
    let b1v = a0 - a1;
    let a2s = a2 * WC_M4_0;
    let a3s = a3 * WC_M4_1;
    let c0 = a2s + a3s;
    let c1 = a2s - a3s;
    let c0_post = SQRT2 * c0 + c1;
    t0 = b0v;
    t2 = b1v;
    t1 = c0_post;
    t3 = c1;

    t4 = t4 * WC_M8_0;
    t5 = t5 * WC_M8_1;
    t6 = t6 * WC_M8_2;
    t7 = t7 * WC_M8_3;

    let a0 = t4 + t7;
    let a1 = t5 + t6;
    let a2 = t4 - t7;
    let a3 = t5 - t6;
    let b0v = a0 + a1;
    let b1v = a0 - a1;
    let a2s = a2 * WC_M4_0;
    let a3s = a3 * WC_M4_1;
    let c0 = a2s + a3s;
    let c1 = a2s - a3s;
    let c0_post = SQRT2 * c0 + c1;
    t4 = b0v;
    t6 = b1v;
    t5 = c0_post;
    t7 = c1;

    t4 = SQRT2 * t4 + t5;
    t5 = t5 + t6;
    t6 = t6 + t7;

    mem[b0] = t0;
    mem[b0 + 1usize] = t4;
    mem[b0 + 2usize] = t1;
    mem[b0 + 3usize] = t5;
    mem[b0 + 4usize] = t2;
    mem[b0 + 5usize] = t6;
    mem[b0 + 6usize] = t3;
    mem[b0 + 7usize] = t7;
}

/// 1D 8-point inverse DCT (libjxl factorization).
#[cube]
fn idct1d_8(mem: &mut SharedMemory<f32>, base: u32) {
    let b0 = base as usize;
    let f0 = mem[b0];
    let f1 = mem[b0 + 2usize];
    let f2 = mem[b0 + 4usize];
    let f3 = mem[b0 + 6usize];
    let mut s0 = mem[b0 + 1usize];
    let mut s1 = mem[b0 + 3usize];
    let mut s2 = mem[b0 + 5usize];
    let s3 = mem[b0 + 7usize];

    s2 = s2 - s3;
    s1 = s1 - s2;
    s0 = (s0 - s1) * ONE_OVER_SQRT2;

    // IDCT-4 on [s0, s1, s2, s3]
    let mut t0 = s0;
    let mut t1 = s2;
    let mut t2 = s1;
    let mut t3 = s3;
    t2 = (t2 - t3) * ONE_OVER_SQRT2;
    let a0 = t2 + t3;
    let a1 = t2 - t3;
    t2 = a0;
    t3 = a1;
    t2 = t2 * INV_WC_M4_0;
    t3 = t3 * INV_WC_M4_1;
    let a0 = t0 + t1;
    let a1 = t0 - t1;
    t0 = a0;
    t1 = a1;
    let so0 = t0 + t2;
    let so1 = t1 + t3;
    let so2 = t1 - t3;
    let so3 = t0 - t2;

    let sa0 = so0 * INV_WC_M8_0;
    let sa1 = so1 * INV_WC_M8_1;
    let sa2 = so2 * INV_WC_M8_2;
    let sa3 = so3 * INV_WC_M8_3;

    // IDCT-4 on [f0, f1, f2, f3]
    let mut g0 = f0;
    let mut g1 = f2;
    let mut g2 = f1;
    let mut g3 = f3;
    g2 = (g2 - g3) * ONE_OVER_SQRT2;
    let a0 = g2 + g3;
    let a1 = g2 - g3;
    g2 = a0;
    g3 = a1;
    g2 = g2 * INV_WC_M4_0;
    g3 = g3 * INV_WC_M4_1;
    let a0 = g0 + g1;
    let a1 = g0 - g1;
    g0 = a0;
    g1 = a1;
    let fo0 = g0 + g2;
    let fo1 = g1 + g3;
    let fo2 = g1 - g3;
    let fo3 = g0 - g2;

    mem[b0] = fo0 + sa0;
    mem[b0 + 1usize] = fo1 + sa1;
    mem[b0 + 2usize] = fo2 + sa2;
    mem[b0 + 3usize] = fo3 + sa3;
    mem[b0 + 4usize] = fo3 - sa3;
    mem[b0 + 5usize] = fo2 - sa2;
    mem[b0 + 6usize] = fo1 - sa1;
    mem[b0 + 7usize] = fo0 - sa0;
}

/// Forward 8x8 DCT for `num_blocks` contiguous blocks. One cube per block.
#[cube(launch_unchecked)]
pub fn dct_8x8_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;

    let mut scratch = SharedMemory::<f32>::new(64usize);
    let mut transposed = SharedMemory::<f32>::new(64usize);

    let mut r: u32 = 0u32;
    while r < 8u32 {
        let row_off = r * 8u32;
        let row_off_us = row_off as usize;
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = input[off + row_off_us + cu];
            c += 1u32;
        }
        dct1d_8(&mut scratch, row_off);
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let cu = c as usize;
            scratch[row_off_us + cu] = scratch[row_off_us + cu] * ONE_OVER_8;
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 8u32 {
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 8usize + ru] = scratch[ru * 8usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 8u32 {
        let row_off = r * 8u32;
        let row_off_us = row_off as usize;
        dct1d_8(&mut transposed, row_off);
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let cu = c as usize;
            transposed[row_off_us + cu] = transposed[row_off_us + cu] * ONE_OVER_8;
            c += 1u32;
        }
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        output[off + iu] = transposed[iu];
        i += 1u32;
    }
}

/// Wide-cube forward 8×8 DCT: each thread handles ONE complete block,
/// but cubes pack `WIDE_CUBE_DIM` threads to amortize launch overhead.
///
/// Per-thread scratch lives at `[UNIT_POS * 64 + ...]` of a
/// shared `WIDE_CUBE_DIM * 64` array. Each thread reads/writes its
/// own 64-float slice; no inter-thread communication.
///
/// Trades cube_dim=1's private-shared-memory simplicity for better
/// launch amortization and warp utilization.
///
/// Tuned via dct8_coop_bench sweep on RTX 5070 (sizes 256²-4096²,
/// 16/32/64 cube_dim variants):
///   16  →  4.25 ms at 1024² (limited by cube count)
///   32  →  3.76 ms at 1024²
///   64  →  3.67 ms at 1024² ← best at sweet-spot, also best at 4096²
/// 64 wins on aggregate but only by margin; 32 is also reasonable.
const WIDE_CUBE_DIM: u32 = 64;

#[cube(launch_unchecked)]
pub fn dct_8x8_wide_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;
    let unit = UNIT_POS;
    let private_base = unit * 64u32;
    let private_base_us = private_base as usize;

    let mut scratch = SharedMemory::<f32>::new((WIDE_CUBE_DIM * 64u32) as usize);
    let mut transposed = SharedMemory::<f32>::new((WIDE_CUBE_DIM * 64u32) as usize);

    let mut r: u32 = 0u32;
    while r < 8u32 {
        let row_off = r * 8u32;
        let row_off_us = row_off as usize;
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let cu = c as usize;
            scratch[private_base_us + row_off_us + cu] = input[off + row_off_us + cu];
            c += 1u32;
        }
        dct1d_8(&mut scratch, private_base + row_off);
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let cu = c as usize;
            scratch[private_base_us + row_off_us + cu] =
                scratch[private_base_us + row_off_us + cu] * ONE_OVER_8;
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 8u32 {
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[private_base_us + cu * 8usize + ru] =
                scratch[private_base_us + ru * 8usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 8u32 {
        let row_off = r * 8u32;
        let row_off_us = row_off as usize;
        dct1d_8(&mut transposed, private_base + row_off);
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let cu = c as usize;
            transposed[private_base_us + row_off_us + cu] =
                transposed[private_base_us + row_off_us + cu] * ONE_OVER_8;
            c += 1u32;
        }
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        output[off + iu] = transposed[private_base_us + iu];
        i += 1u32;
    }
}

/// Wide-cube inverse 8×8 DCT — symmetric to `dct_8x8_wide_kernel`.
/// Same per-thread private slice layout in shared memory.
#[cube(launch_unchecked)]
pub fn idct_8x8_wide_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;
    let unit = UNIT_POS;
    let private_base = unit * 64u32;
    let private_base_us = private_base as usize;

    let mut scratch = SharedMemory::<f32>::new((WIDE_CUBE_DIM * 64u32) as usize);
    let mut transposed = SharedMemory::<f32>::new((WIDE_CUBE_DIM * 64u32) as usize);

    // Load whole block into private scratch slice.
    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        scratch[private_base_us + iu] = input[off + iu];
        i += 1u32;
    }

    // Row pass.
    let mut r: u32 = 0u32;
    while r < 8u32 {
        idct1d_8(&mut scratch, private_base + r * 8u32);
        r += 1u32;
    }

    // Transpose.
    let mut r: u32 = 0u32;
    while r < 8u32 {
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[private_base_us + cu * 8usize + ru] =
                scratch[private_base_us + ru * 8usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    // Column pass (= row pass on transposed).
    let mut r: u32 = 0u32;
    while r < 8u32 {
        idct1d_8(&mut transposed, private_base + r * 8u32);
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        output[off + iu] = transposed[private_base_us + iu];
        i += 1u32;
    }
}

/// Cooperative forward 8×8 DCT: 8 threads per block, one per row.
///
/// Each cube processes one block. Within a cube, thread `r` (UNIT_POS)
/// owns row `r` — performs row load, dct1d_8 on its row, then column
/// pass on row `r` of the transposed buffer, then writes its row out.
///
/// Throughput target: 8× more block-parallelism than the cube_dim=1
/// kernel for the same num_blocks, so SMs stay busy at smaller image
/// sizes (where cube_dim=1 starves with too few cubes).
///
/// Launch: cube_dim = 8, cube_count = num_blocks.
#[cube(launch_unchecked)]
pub fn dct_8x8_coop_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = CUBE_POS;
    let row_idx = UNIT_POS;
    let n_blocks = input.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;
    let row_off = row_idx * 8u32;
    let row_off_us = row_off as usize;

    let mut scratch = SharedMemory::<f32>::new(64usize);
    let mut transposed = SharedMemory::<f32>::new(64usize);

    // ── Row pass ────────────────────────────────────────────────
    let mut c: u32 = 0u32;
    while c < 8u32 {
        let cu = c as usize;
        scratch[row_off_us + cu] = input[off + row_off_us + cu];
        c += 1u32;
    }
    sync_cube();
    dct1d_8(&mut scratch, row_off);
    let mut c: u32 = 0u32;
    while c < 8u32 {
        let cu = c as usize;
        scratch[row_off_us + cu] = scratch[row_off_us + cu] * ONE_OVER_8;
        c += 1u32;
    }
    sync_cube();

    // ── Transpose: thread r writes column r of `transposed`
    //    (= reads row r of `scratch` slot-by-slot into transposed[c, r]) ──
    let mut c: u32 = 0u32;
    while c < 8u32 {
        let cu = c as usize;
        let ru = row_idx as usize;
        transposed[cu * 8usize + ru] = scratch[ru * 8usize + cu];
        c += 1u32;
    }
    sync_cube();

    // ── Column pass (now row pass on transposed) ───────────────
    dct1d_8(&mut transposed, row_off);
    let mut c: u32 = 0u32;
    while c < 8u32 {
        let cu = c as usize;
        transposed[row_off_us + cu] = transposed[row_off_us + cu] * ONE_OVER_8;
        c += 1u32;
    }
    sync_cube();

    // ── Write out: thread r writes row r ───────────────────────
    let mut c: u32 = 0u32;
    while c < 8u32 {
        let cu = c as usize;
        output[off + row_off_us + cu] = transposed[row_off_us + cu];
        c += 1u32;
    }
}

/// Cooperative inverse 8×8 DCT — 8 threads per block, one per row.
/// Mirrors `dct_8x8_coop_kernel` for the inverse direction.
#[cube(launch_unchecked)]
pub fn idct_8x8_coop_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = CUBE_POS;
    let row_idx = UNIT_POS;
    let n_blocks = input.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;
    let row_off = row_idx * 8u32;
    let row_off_us = row_off as usize;

    let mut scratch = SharedMemory::<f32>::new(64usize);
    let mut transposed = SharedMemory::<f32>::new(64usize);

    // Load row into scratch.
    let mut c: u32 = 0u32;
    while c < 8u32 {
        let cu = c as usize;
        scratch[row_off_us + cu] = input[off + row_off_us + cu];
        c += 1u32;
    }
    sync_cube();
    idct1d_8(&mut scratch, row_off);
    sync_cube();

    // Transpose.
    let mut c: u32 = 0u32;
    while c < 8u32 {
        let cu = c as usize;
        let ru = row_idx as usize;
        transposed[cu * 8usize + ru] = scratch[ru * 8usize + cu];
        c += 1u32;
    }
    sync_cube();

    // Column pass (= row pass on transposed).
    idct1d_8(&mut transposed, row_off);
    sync_cube();

    let mut c: u32 = 0u32;
    while c < 8u32 {
        let cu = c as usize;
        output[off + row_off_us + cu] = transposed[row_off_us + cu];
        c += 1u32;
    }
}

/// Inverse 8x8 DCT for `num_blocks` contiguous blocks.
#[cube(launch_unchecked)]
pub fn idct_8x8_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = input.len() / 64usize;
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;

    let mut scratch = SharedMemory::<f32>::new(64usize);
    let mut transposed = SharedMemory::<f32>::new(64usize);

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        scratch[iu] = input[off + iu];
        i += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 8u32 {
        idct1d_8(&mut scratch, r * 8u32);
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 8u32 {
        let mut c: u32 = 0u32;
        while c < 8u32 {
            let ru = r as usize;
            let cu = c as usize;
            transposed[cu * 8usize + ru] = scratch[ru * 8usize + cu];
            c += 1u32;
        }
        r += 1u32;
    }

    let mut r: u32 = 0u32;
    while r < 8u32 {
        idct1d_8(&mut transposed, r * 8u32);
        r += 1u32;
    }

    let mut i: u32 = 0u32;
    while i < 64u32 {
        let iu = i as usize;
        output[off + iu] = transposed[iu];
        i += 1u32;
    }
}
