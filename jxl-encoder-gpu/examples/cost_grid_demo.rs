//! Phase 3 prototype: cost grid for DCT8 strategy on a synthetic image.
//!
//! Demonstrates the whole-image-per-strategy pipeline composition:
//! DCT → quantize → dequant → IDCT → block_l2.
//!
//! Output is a per-block cost (proxy for entropy + pixel_loss combined).

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
    use jxl_encoder_gpu::pipeline::compute_cost_grid_dct8_single_channel;

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // 16x12 blocks = 128x96 input
    const XB: usize = 16;
    const YB: usize = 12;
    const NB: usize = XB * YB;
    const N_COEF: usize = NB * 64;

    // Synthetic block-major Y channel (mix of low and high freq content)
    let mut input = vec![0.0f32; N_COEF];
    for b in 0..NB {
        for i in 0..64 {
            let r = (i / 8) as f32;
            let c = (i % 8) as f32;
            let v = ((b * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
            // DC + smooth gradient + high-freq noise mix
            input[b * 64 + i] = 0.3 + 0.05 * (r + c) + 0.02 * v;
        }
    }

    // Standard DCT8 quant matrix (replicated per block)
    let mut weights_per = vec![1.0f32; 64];
    for i in 0..64 {
        let r = (i / 8) as f32;
        let c = (i % 8) as f32;
        weights_per[i] = 1.0 + 0.7 * (r + c);
    }
    let mut weights = vec![0.0f32; N_COEF];
    for b in 0..NB {
        weights[b * 64..b * 64 + 64].copy_from_slice(&weights_per);
    }

    let mut qac_qm = vec![0.0f32; NB];
    for b in 0..NB {
        qac_qm[b] = 1.7;
    }
    let thresholds = [0.62f32, 0.62, 0.62, 0.62];

    let h_in = client.create_from_slice(f32::as_bytes(&input));
    let h_w = client.create_from_slice(f32::as_bytes(&weights));
    let h_qac = client.create_from_slice(f32::as_bytes(&qac_qm));
    let h_thr = client.create_from_slice(f32::as_bytes(&thresholds[..]));

    let result = compute_cost_grid_dct8_single_channel::<Backend>(
        &client, h_in, h_w, h_qac, h_thr, XB as u32, YB as u32,
    );

    let bytes = client.read_one(result.costs).expect("read costs");
    let costs: &[f32] = f32::from_bytes(&bytes);

    println!(
        "Cost grid: {}x{} blocks, {} costs total",
        result.xsize_blocks,
        result.ysize_blocks,
        costs.len()
    );
    println!("Sample costs (first 8 blocks):");
    for (i, c) in costs.iter().take(8).enumerate() {
        println!("  block {}: cost = {:.6}", i, c);
    }
    let max = costs.iter().fold(0.0f32, |a, &b| a.max(b));
    let min = costs.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let mean = costs.iter().sum::<f32>() / (costs.len() as f32);
    println!("Cost range: min = {min:.6}, max = {max:.6}, mean = {mean:.6}");

    // Sanity: all costs should be non-negative (block_l2 is sum of squares).
    let all_nonneg = costs.iter().all(|&c| c >= 0.0 && c.is_finite());
    if !all_nonneg {
        eprintln!("✗ FAILED: costs contain negative or non-finite values");
        std::process::exit(1);
    }

    println!(
        "\n✓ Phase 3 prototype: cost grid for DCT8 single-channel composition runs end-to-end."
    );
    println!("  (Full Phase 3: 3-channel + CfL + multiple strategies + partition selector.)");
}
