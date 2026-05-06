//! Phase 3 end-to-end integration demo.
//!
//! Runs cost grids for DCT8 + DCT16x16 + DCT32x32 on a synthetic image,
//! aggregates the per-8x8-sub-block costs to per-strategy-region costs,
//! and feeds the partition selector to produce final per-region strategy
//! decisions. This is the full Phase 3 Component 1 + Component 2 pipeline
//! working end-to-end.
//!
//! Component 3 (refactor `ac_strategy_search.rs` in `jxl-encoder`) remains
//! BLOCKED on cross-repo permission.

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(all(not(feature = "cuda"), not(feature = "wgpu"), feature = "cpu"))]
type Backend = cubecl::cpu::CpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "cpu")))]
fn main() {
    eprintln!("enable one of: --features cuda | wgpu | cpu");
    std::process::exit(2);
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]
fn main() {
    use cubecl::prelude::*;
    use jxl_encoder_gpu::pipeline::{
        Partition32x32, compute_cost_grid_dct16x16_single_channel,
        compute_cost_grid_dct32x32_single_channel,
        compute_cost_grid_dct8_single_channel, select_partitions_32x32,
    };

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // 8x8 blocks @ 8x8 = 64x64 pixel image = 4x4 16x16 regions = 2x2 32x32 regions
    const XB8: u32 = 8;
    const YB8: u32 = 8;
    const NB8: usize = (XB8 * YB8) as usize;
    const N_PIX: usize = NB8 * 64;

    // Half the image is smooth (low-freq), half is noisy (high-freq) to
    // exercise per-region strategy selection.
    let mut input_dct8 = vec![0.0f32; N_PIX];
    for by in 0..YB8 as usize {
        for bx in 0..XB8 as usize {
            let block_idx = by * (XB8 as usize) + bx;
            let smooth = bx < (XB8 as usize) / 2;
            for i in 0..64 {
                let r = (i / 8) as f32;
                let c = (i % 8) as f32;
                input_dct8[block_idx * 64 + i] = if smooth {
                    0.3 + 0.05 * (r + c)
                } else {
                    let v = ((block_idx * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
                    0.3 + 0.4 * v
                };
            }
        }
    }

    // The block_l2 launcher operates on 8x8 sub-blocks and treats input as
    // contiguous [num_blocks * 64] floats. For DCT8 we already have that
    // shape. For DCT16x16 we need [num_dct16_blocks * 256] (16x16 blocks
    // contiguous). For this demo, we approximate by repacking — a real
    // integration would extract from a strided plane.
    //
    // For DCT16x16 input: take 4 8x8 blocks per 16x16 region and pack as
    // 256 contiguous floats (row-major within 16x16).
    const XB16: u32 = XB8 / 2;
    const YB16: u32 = YB8 / 2;
    const NB16: usize = (XB16 * YB16) as usize;
    let mut input_dct16 = vec![0.0f32; NB16 * 256];
    for ry in 0..YB16 as usize {
        for rx in 0..XB16 as usize {
            let region_idx = ry * (XB16 as usize) + rx;
            for ly in 0..16 {
                for lx in 0..16 {
                    let by = ry * 2 + ly / 8;
                    let bx = rx * 2 + lx / 8;
                    let block_idx = by * (XB8 as usize) + bx;
                    let in_block_idx = (ly % 8) * 8 + (lx % 8);
                    input_dct16[region_idx * 256 + ly * 16 + lx] =
                        input_dct8[block_idx * 64 + in_block_idx];
                }
            }
        }
    }
    // Same pattern for DCT32x32 input.
    const XB32: u32 = XB8 / 4;
    const YB32: u32 = YB8 / 4;
    const NB32: usize = (XB32 * YB32) as usize;
    let mut input_dct32 = vec![0.0f32; NB32 * 1024];
    for ry in 0..YB32 as usize {
        for rx in 0..XB32 as usize {
            let region_idx = ry * (XB32 as usize) + rx;
            for ly in 0..32 {
                for lx in 0..32 {
                    let by = ry * 4 + ly / 8;
                    let bx = rx * 4 + lx / 8;
                    let block_idx = by * (XB8 as usize) + bx;
                    let in_block_idx = (ly % 8) * 8 + (lx % 8);
                    input_dct32[region_idx * 1024 + ly * 32 + lx] =
                        input_dct8[block_idx * 64 + in_block_idx];
                }
            }
        }
    }

    // Per-block weights replicated.
    fn weights_block(block_size: usize, side: usize) -> Vec<f32> {
        let mut w = vec![1.0f32; block_size];
        for i in 0..block_size {
            let r = (i / side) as f32;
            let c = (i % side) as f32;
            w[i] = 1.0 + 0.7 * (r + c);
        }
        w
    }
    let w_dct8 = {
        let per = weights_block(64, 8);
        let mut w = vec![0.0f32; NB8 * 64];
        for b in 0..NB8 {
            w[b * 64..b * 64 + 64].copy_from_slice(&per);
        }
        w
    };
    let w_dct16 = {
        let per = weights_block(256, 16);
        let mut w = vec![0.0f32; NB16 * 256];
        for b in 0..NB16 {
            w[b * 256..b * 256 + 256].copy_from_slice(&per);
        }
        w
    };
    let w_dct32 = {
        let per = weights_block(1024, 32);
        let mut w = vec![0.0f32; NB32 * 1024];
        for b in 0..NB32 {
            w[b * 1024..b * 1024 + 1024].copy_from_slice(&per);
        }
        w
    };

    let qac_dct8 = vec![1.7f32; NB8];
    let qac_dct16 = vec![1.7f32; NB16];
    let qac_dct32 = vec![1.7f32; NB32];
    let thresholds = [0.62f32; 4];

    // Run all three cost grids on the GPU.
    let h_in8 = client.create_from_slice(f32::as_bytes(&input_dct8));
    let h_w8 = client.create_from_slice(f32::as_bytes(&w_dct8));
    let h_qac8 = client.create_from_slice(f32::as_bytes(&qac_dct8));
    let h_thr8 = client.create_from_slice(f32::as_bytes(&thresholds[..]));
    let cg_dct8 = compute_cost_grid_dct8_single_channel::<Backend>(
        &client, h_in8, h_w8, h_qac8, h_thr8, XB8, YB8,
    );
    let cost_dct8: Vec<f32> = {
        let bytes = client.read_one(cg_dct8.costs).expect("cost_dct8");
        f32::from_bytes(&bytes).to_vec()
    };

    let h_in16 = client.create_from_slice(f32::as_bytes(&input_dct16));
    let h_w16 = client.create_from_slice(f32::as_bytes(&w_dct16));
    let h_qac16 = client.create_from_slice(f32::as_bytes(&qac_dct16));
    let h_thr16 = client.create_from_slice(f32::as_bytes(&thresholds[..]));
    let cg_dct16 = compute_cost_grid_dct16x16_single_channel::<Backend>(
        &client, h_in16, h_w16, h_qac16, h_thr16, XB16, YB16,
    );
    // The dct16 cost grid stores 4 sub-block (8x8) costs per 16x16 region.
    // Aggregate to per-region totals for the partition selector.
    let cost_dct16x16: Vec<f32> = {
        let bytes = client.read_one(cg_dct16.costs).expect("cost_dct16");
        let raw: &[f32] = f32::from_bytes(&bytes);
        let mut out = vec![0.0f32; NB16];
        // raw layout matches block_l2 output for 2x2 8x8 sub-blocks per 16x16:
        // (XB16*2) wide × (YB16*2) tall row-major. Aggregate 4 sub-blocks per region.
        let xb_sub = (XB16 * 2) as usize;
        for ry in 0..YB16 as usize {
            for rx in 0..XB16 as usize {
                let by = ry * 2;
                let bx = rx * 2;
                out[ry * (XB16 as usize) + rx] = raw[by * xb_sub + bx]
                    + raw[by * xb_sub + bx + 1]
                    + raw[(by + 1) * xb_sub + bx]
                    + raw[(by + 1) * xb_sub + bx + 1];
            }
        }
        out
    };

    let h_in32 = client.create_from_slice(f32::as_bytes(&input_dct32));
    let h_w32 = client.create_from_slice(f32::as_bytes(&w_dct32));
    let h_qac32 = client.create_from_slice(f32::as_bytes(&qac_dct32));
    let h_thr32 = client.create_from_slice(f32::as_bytes(&thresholds[..]));
    let cg_dct32 = compute_cost_grid_dct32x32_single_channel::<Backend>(
        &client, h_in32, h_w32, h_qac32, h_thr32, XB32, YB32,
    );
    // Same aggregation: 16 sub-blocks per 32x32 region.
    let cost_dct32x32: Vec<f32> = {
        let bytes = client.read_one(cg_dct32.costs).expect("cost_dct32");
        let raw: &[f32] = f32::from_bytes(&bytes);
        let mut out = vec![0.0f32; NB32];
        let xb_sub = (XB32 * 4) as usize;
        for ry in 0..YB32 as usize {
            for rx in 0..XB32 as usize {
                let by_base = ry * 4;
                let bx_base = rx * 4;
                let mut sum = 0.0f32;
                for dy in 0..4 {
                    for dx in 0..4 {
                        sum += raw[(by_base + dy) * xb_sub + bx_base + dx];
                    }
                }
                out[ry * (XB32 as usize) + rx] = sum;
            }
        }
        out
    };

    // Now feed the partition selector.
    let partitions = select_partitions_32x32(
        &cost_dct8,
        &cost_dct16x16,
        &cost_dct32x32,
        XB8 as usize,
        YB8 as usize,
    );

    println!("Phase 3 end-to-end demo:");
    println!("  Image: {}x{} pixels = {}x{} 32x32 regions", XB8 * 8, YB8 * 8, XB32, YB32);
    println!("  Cost grids: DCT8={} entries, DCT16x16={}, DCT32x32={}",
        cost_dct8.len(), cost_dct16x16.len(), cost_dct32x32.len());
    println!("  Partition decisions per 32x32 region:");
    for (i, p) in partitions.iter().enumerate() {
        let rx = i % (XB32 as usize);
        let ry = i / (XB32 as usize);
        let label = match p {
            Partition32x32::Dct32x32 => "DCT32x32".to_string(),
            Partition32x32::TwoDct32x16Horizontal => "2×DCT32x16".to_string(),
            Partition32x32::TwoDct16x32Vertical => "2×DCT16x32".to_string(),
            Partition32x32::Sub16x16(subs) => format!("Sub16x16{:?}", subs),
        };
        println!("    region ({}, {}): {}", rx, ry, label);
    }

    println!("\n✓ Phase 3 Components 1+2 chain works end-to-end.");
    println!("  (Component 3 — jxl-encoder integration — remains BLOCKED.)");
}
