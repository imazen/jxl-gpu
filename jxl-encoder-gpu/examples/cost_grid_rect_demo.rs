//! Phase 3 prototype: cost grids for the rectangular DCT16x8 + DCT8x16
//! strategies on a synthetic image.
//!
//! Mirrors `cost_grid_demo` but exercises the rectangular variants.
//! Same single-channel proxy cost as the DCT8 / DCT16x16 / DCT32x32 /
//! DCT64x64 grids.

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
        compute_cost_grid_dct8x16_single_channel, compute_cost_grid_dct16x8_single_channel,
    };

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // 8x6 rect blocks = 128x48 (16x8) and 64x96 (8x16) image equivalents
    // (cost grids are reported per-rect-block, not per-pixel; we just
    // pick a multiple of the rect to keep the math clean).
    const XB: usize = 8;
    const YB: usize = 6;
    const NB: usize = XB * YB;
    const N_COEF: usize = NB * 128;

    // Synthetic block-major Y channel.
    let mut input = vec![0.0f32; N_COEF];
    for b in 0..NB {
        for i in 0..128 {
            let v = ((b * 11 + i * 17).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
            input[b * 128 + i] = 0.3 + 0.04 * v;
        }
    }

    // Standard rect quant matrix (replicated per block; 128 weights each).
    let mut weights_per = vec![1.0f32; 128];
    for i in 0..128 {
        weights_per[i] = 1.0 + 0.5 * (i as f32 / 128.0);
    }
    let mut weights = vec![0.0f32; N_COEF];
    for b in 0..NB {
        weights[b * 128..b * 128 + 128].copy_from_slice(&weights_per);
    }

    let qac_qm = vec![1.7f32; NB];
    let thresholds = [0.62f32, 0.62, 0.62, 0.62];

    let h_in = client.create_from_slice(f32::as_bytes(&input));
    let h_w = client.create_from_slice(f32::as_bytes(&weights));
    let h_qac = client.create_from_slice(f32::as_bytes(&qac_qm));
    let h_thr = client.create_from_slice(f32::as_bytes(&thresholds[..]));

    // ---- DCT16×8 ----
    let cg16x8 = compute_cost_grid_dct16x8_single_channel::<Backend>(
        &client,
        h_in.clone(),
        h_w.clone(),
        h_qac.clone(),
        h_thr.clone(),
        XB as u32,
        YB as u32,
    );
    let bytes = client.read_one(cg16x8.costs).expect("read 16x8");
    let costs_16x8: &[f32] = f32::from_bytes(&bytes);
    let mean_16x8 = costs_16x8.iter().sum::<f32>() / (costs_16x8.len() as f32);
    let max_16x8 = costs_16x8.iter().fold(0.0f32, |a, &b| a.max(b));
    println!(
        "DCT16×8 cost grid: {} per-8×8 costs (= {} 16×8 rect blocks × 2 sub-cells), mean={:.4} max={:.4}",
        costs_16x8.len(),
        cg16x8.xsize_blocks * cg16x8.ysize_blocks,
        mean_16x8,
        max_16x8,
    );
    let ok_16x8 = costs_16x8.iter().all(|&c| c >= 0.0 && c.is_finite());
    if !ok_16x8 {
        eprintln!("✗ DCT16×8 produced negative or non-finite costs");
        std::process::exit(1);
    }

    // ---- DCT8×16 ----
    let cg8x16 = compute_cost_grid_dct8x16_single_channel::<Backend>(
        &client, h_in, h_w, h_qac, h_thr, XB as u32, YB as u32,
    );
    let bytes = client.read_one(cg8x16.costs).expect("read 8x16");
    let costs_8x16: &[f32] = f32::from_bytes(&bytes);
    let mean_8x16 = costs_8x16.iter().sum::<f32>() / (costs_8x16.len() as f32);
    let max_8x16 = costs_8x16.iter().fold(0.0f32, |a, &b| a.max(b));
    println!(
        "DCT8×16  cost grid: {} per-8×8 costs (= {} 8×16 rect blocks × 2 sub-cells), mean={:.4} max={:.4}",
        costs_8x16.len(),
        cg8x16.xsize_blocks * cg8x16.ysize_blocks,
        mean_8x16,
        max_8x16,
    );
    let ok_8x16 = costs_8x16.iter().all(|&c| c >= 0.0 && c.is_finite());
    if !ok_8x16 {
        eprintln!("✗ DCT8×16 produced negative or non-finite costs");
        std::process::exit(1);
    }

    println!("\n✓ Both rectangular DCT16 cost grids produced finite non-negative costs.");
}
