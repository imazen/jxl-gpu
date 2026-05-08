//! Phase 3 prototype: cost grids for the DCT4 sub-block family
//! (DCT4×4, DCT4×8, DCT8×4) on a synthetic image.
//!
//! All three operate on 64-coeff blocks (8×8 layout with sub-block
//! structure), so they reuse the dct8 quantize + dequant kernels.

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
        compute_cost_grid_dct4x4_single_channel, compute_cost_grid_dct4x8_single_channel,
        compute_cost_grid_dct8x4_single_channel,
    };

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    const XB: usize = 16;
    const YB: usize = 12;
    const NB: usize = XB * YB;
    const N_COEF: usize = NB * 64;

    let mut input = vec![0.0f32; N_COEF];
    for b in 0..NB {
        for i in 0..64 {
            let r = (i / 8) as f32;
            let c = (i % 8) as f32;
            let v = ((b * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
            input[b * 64 + i] = 0.3 + 0.05 * (r + c) + 0.02 * v;
        }
    }

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

    let qac_qm = vec![1.7f32; NB];
    let thresholds = [0.62f32, 0.62, 0.62, 0.62];

    let h_in = client.create_from_slice(f32::as_bytes(&input));
    let h_w = client.create_from_slice(f32::as_bytes(&weights));
    let h_qac = client.create_from_slice(f32::as_bytes(&qac_qm));
    let h_thr = client.create_from_slice(f32::as_bytes(&thresholds[..]));

    let report = |name: &str, costs: &[f32]| {
        let mean = costs.iter().sum::<f32>() / (costs.len() as f32);
        let max = costs.iter().fold(0.0f32, |a, &b| a.max(b));
        let min = costs.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        println!(
            "{name:>8}: {} costs, min={min:.4} mean={mean:.4} max={max:.4}",
            costs.len()
        );
        let ok = costs.iter().all(|&c| c >= 0.0 && c.is_finite());
        if !ok {
            eprintln!("  ✗ {name} produced negative or non-finite costs");
            std::process::exit(1);
        }
    };

    let cg4x4 = compute_cost_grid_dct4x4_single_channel::<Backend>(
        &client,
        h_in.clone(),
        h_w.clone(),
        h_qac.clone(),
        h_thr.clone(),
        XB as u32,
        YB as u32,
    );
    let bytes = client.read_one(cg4x4.costs).expect("read 4x4");
    report("DCT4×4", f32::from_bytes(&bytes));

    let cg4x8 = compute_cost_grid_dct4x8_single_channel::<Backend>(
        &client,
        h_in.clone(),
        h_w.clone(),
        h_qac.clone(),
        h_thr.clone(),
        XB as u32,
        YB as u32,
    );
    let bytes = client.read_one(cg4x8.costs).expect("read 4x8");
    report("DCT4×8", f32::from_bytes(&bytes));

    let cg8x4 = compute_cost_grid_dct8x4_single_channel::<Backend>(
        &client, h_in, h_w, h_qac, h_thr, XB as u32, YB as u32,
    );
    let bytes = client.read_one(cg8x4.costs).expect("read 8x4");
    report("DCT8×4", f32::from_bytes(&bytes));

    println!("\n✓ All three DCT4-family cost grids produced finite non-negative costs.");
}
