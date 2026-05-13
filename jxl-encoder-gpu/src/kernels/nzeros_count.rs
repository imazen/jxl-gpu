// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-block non-zero AC coefficient counter for DCT8 blocks.
//!
//! Mirrors `jxl_encoder::vardct::ac_group::num_nonzero_8x8_except_dc`:
//! one count per block (positions 1..64), max possible value is 63
//! (no clamping needed at u8 width).
//!
//! Output is u32 per block — caller converts to u8 (`nzeros`) and
//! u16 (`raw_nzeros`) on the host. Both upstream fields hold the
//! same value for DCT8 (covered_blocks == 1, no LLF shift).

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn nzeros_count_dct8_kernel(quant_ac: &Array<i32>, nzeros: &mut Array<u32>) {
    let block_idx = ABSOLUTE_POS;
    let n_blocks = nzeros.len();
    if block_idx >= n_blocks {
        terminate!();
    }
    let off = block_idx * 64usize;
    let mut count: u32 = 0u32;
    let mut idx: u32 = 1u32; // skip DC
    while idx < 64u32 {
        if quant_ac[off + idx as usize] != 0i32 {
            count += 1u32;
        }
        idx += 1u32;
    }
    nzeros[block_idx] = count;
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use crate::launch::nzeros_count::nzeros_count_dct8;
    use cubecl::Runtime;
    use cubecl::cuda::CudaRuntime;
    use cubecl::prelude::*;
    extern crate alloc;

    #[test]
    fn nzeros_count_dct8_matches_cpu_reference() {
        let device = Default::default();
        let client = CudaRuntime::client(&device);

        let n_blocks = 19usize;
        // Mix of dense and sparse blocks to exercise the count.
        let quant_ac: Vec<i32> = (0..n_blocks * 64)
            .map(|i| {
                let b = i / 64;
                let p = i % 64;
                if p == 0 {
                    9999 // DC always present, must be ignored
                } else if (b * 31 + p * 7) % 5 == 0 {
                    ((i as i32) % 7) - 3
                } else {
                    0
                }
            })
            .collect();

        let mut cpu_nzeros = vec![0u32; n_blocks];
        for b in 0..n_blocks {
            for p in 1..64 {
                if quant_ac[b * 64 + p] != 0 {
                    cpu_nzeros[b] += 1;
                }
            }
        }

        let h_q = client.create_from_slice(i32::as_bytes(&quant_ac));
        let h_n = client.empty(n_blocks * core::mem::size_of::<u32>());
        nzeros_count_dct8::<CudaRuntime>(&client, h_q, h_n.clone(), n_blocks as u32);
        let mut bytes = client.read(alloc::vec![h_n]);
        let nz_bytes = bytes.pop().unwrap();
        let gpu: Vec<u32> = u32::from_bytes(&nz_bytes).to_vec();

        for b in 0..n_blocks {
            assert_eq!(gpu[b], cpu_nzeros[b], "block {b}: gpu={} cpu={}", gpu[b], cpu_nzeros[b]);
        }
    }
}
