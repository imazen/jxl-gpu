// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! 8×8 mask-mean reduction for adaptive-quant field generation.
//!
//! One thread per output block. Reads up to 64 mask values from
//! the input plane and writes the mean to the output. Image-edge
//! clipping matches the CPU `prepare_strategy_search_plan_inner`
//! reduction loop:
//!
//! - blocks fully inside the image: average of all 64 pixels
//! - blocks partially outside (right / bottom edge): average of
//!   the in-image subset
//! - blocks fully outside (padded grid): emit `1.0` (matches the
//!   `if count > 0 ... else 1.0` fallback)
//!
//! Replaces a host pipeline that downloads the full mask plane
//! (48 MB at 12 MP) and runs a sequential CPU reduction. With
//! this kernel the mask stays on GPU and only the small
//! `xs8 × ys8` output (1.5 MB at 12 MP) crosses the wire.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn block_mask_mean_kernel(
    mask: &Array<f32>,
    out: &mut Array<f32>,
    width: u32,
    height: u32,
    padded_width: u32,
    blocks_per_row: u32,
    blocks_per_col: u32,
) {
    let idx = ABSOLUTE_POS;
    let bpr = blocks_per_row as usize;
    let bpc = blocks_per_col as usize;
    let n_blocks = bpr * bpc;
    if idx >= n_blocks {
        terminate!();
    }
    let by = idx / bpr;
    let bx = idx - by * bpr;

    let w = width as usize;
    let h = height as usize;
    let pw = padded_width as usize;
    let y0 = by * 8usize;
    let x0 = bx * 8usize;

    // Edge-clip extents. When the block sits entirely past the
    // image edge (rare padded-grid case) the inner loop iterates
    // zero times and `mean` keeps its initial 1.0 (matches the
    // CPU loop's `if count > 0 ... else 1.0` fallback).
    let y_end = usize::min(y0 + 8usize, h);
    let x_end = usize::min(x0 + 8usize, w);

    let mut sum = 0.0f32;
    let mut count = 0.0f32;
    for y in y0..y_end {
        let row_off = y * pw;
        for x in x0..x_end {
            sum += mask[row_off + x];
            count += 1.0f32;
        }
    }
    let mut mean = 1.0f32;
    if count > 0.0f32 {
        mean = sum / count;
    }
    out[idx] = mean;
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use crate::launch::aq_field::block_mask_mean;
    use cubecl::Runtime;
    use cubecl::cuda::CudaRuntime;
    extern crate alloc;

    fn cpu_reference(
        mask: &[f32],
        w: usize,
        h: usize,
        pw: usize,
        bpr: usize,
        bpc: usize,
    ) -> Vec<f32> {
        let mut out = vec![1.0f32; bpr * bpc];
        for by in 0..bpc {
            for bx in 0..bpr {
                let y0 = by * 8;
                let x0 = bx * 8;
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for y in y0..(y0 + 8).min(h) {
                    for x in x0..(x0 + 8).min(w) {
                        sum += mask[y * pw + x];
                        count += 1;
                    }
                }
                if count > 0 {
                    out[by * bpr + bx] = sum / count as f32;
                }
            }
        }
        out
    }

    #[test]
    fn block_mask_mean_matches_cpu() {
        let device = Default::default();
        let client = CudaRuntime::client(&device);

        let w = 137usize;
        let h = 91usize;
        let pw = w.div_ceil(8) * 8;
        let ph = h.div_ceil(8) * 8;
        let bpr = pw / 8;
        let bpc = ph / 8;

        let mut mask = vec![0.0f32; pw * ph];
        for y in 0..ph {
            for x in 0..pw {
                mask[y * pw + x] = ((y * 31 + x * 7) % 251) as f32 * 0.001;
            }
        }
        let expected = cpu_reference(&mask, w, h, pw, bpr, bpc);

        let mask_bytes_vec = f32_to_bytes(&mask);
        let mask_h = client.create_from_slice(&mask_bytes_vec);
        let out_h = client.empty(bpr * bpc * core::mem::size_of::<f32>());
        block_mask_mean::<CudaRuntime>(
            &client,
            mask_h,
            out_h.clone(),
            w as u32,
            h as u32,
            pw as u32,
            ph as u32,
        );
        let mut all = client.read(alloc::vec![out_h]);
        let bytes = all.remove(0);
        let got = bytes_to_f32(&bytes);

        for i in 0..expected.len() {
            assert!(
                (got[i] - expected[i]).abs() < 1e-5,
                "block {i}: gpu={} cpu={}",
                got[i],
                expected[i],
            );
        }
    }

    fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
