//! Parity test: GPU pixel_loss vs `jxl_encoder_simd::pixel_domain_loss_scalar`.
//!
//! ```sh
//! cargo run --release --example pixel_loss_parity --features cuda
//! ```

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
    use jxl_encoder_gpu::launch::pixel_loss::pixel_loss;
    use jxl_encoder_simd::pixel_domain_loss_scalar;

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // 16x16 blocks of 8x8 pixels each. Mask is the full image.
    const BW: usize = 8;
    const BH: usize = 8;
    const NX: usize = 16;
    const NY: usize = 16;
    const NB: usize = NX * NY;
    const MASK_STRIDE: usize = NX * BW;
    const MASK_LEN: usize = MASK_STRIDE * (NY * BH);
    const ERR_LEN: usize = NB * BW * BH;
    const MASK_OFFSET: f32 = 0.05;

    // Synthesize per-block error rows + mask plane.
    let mut pixel_error = vec![0.0f32; ERR_LEN];
    for b in 0..NB {
        for i in 0..(BW * BH) {
            let s = ((b + i).wrapping_mul(7) % 251) as f32 / 251.0 - 0.5;
            // Scale to small but non-trivial errors so m8 doesn't underflow f32.
            pixel_error[b * BW * BH + i] = 0.05 * s + 0.001 * (b as f32 / NB as f32);
        }
    }

    let mut mask = vec![0.0f32; MASK_LEN];
    for i in 0..MASK_LEN {
        let u = (i as f32) / (MASK_LEN as f32);
        mask[i] = 0.3 + 0.7 * u; // mask in [0.3, 1.0]
    }

    // Per-block mask_row_base = (block_y * BH) * MASK_STRIDE + block_x * BW
    let mut mask_row_base = vec![0u32; NB];
    for by in 0..NY {
        for bx in 0..NX {
            mask_row_base[by * NX + bx] = (by * BH * MASK_STRIDE + bx * BW) as u32;
        }
    }

    // CPU reference per block.
    let mut cpu_out = vec![0.0f64; NB];
    for b in 0..NB {
        let err_slice = &pixel_error[b * BW * BH..b * BW * BH + BW * BH];
        cpu_out[b] = pixel_domain_loss_scalar(
            err_slice,
            &mask,
            mask_row_base[b] as usize,
            MASK_STRIDE,
            MASK_OFFSET,
            BW,
            BH,
        );
    }

    // GPU.
    let h_err = client.create_from_slice(f32::as_bytes(&pixel_error));
    let h_mask = client.create_from_slice(f32::as_bytes(&mask));
    let h_mrb = client.create_from_slice(u32::as_bytes(&mask_row_base));
    let h_out = client.create_from_slice(f64::as_bytes(&vec![0.0f64; NB]));
    pixel_loss::<Backend>(
        &client,
        h_err,
        h_mask,
        h_mrb,
        h_out.clone(),
        NB as u32,
        MASK_LEN,
        MASK_STRIDE as u32,
        MASK_OFFSET,
        BW as u32,
        BH as u32,
    );
    let bytes = client.read_one(h_out).expect("read pixel_loss");
    let gpu: &[f64] = f64::from_bytes(&bytes);

    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut max_idx = 0usize;
    for (i, (&g, &c)) in gpu.iter().zip(cpu_out.iter()).enumerate() {
        let d = (g - c).abs();
        let r = if c.abs() > 1e-30 { d / c.abs() } else { 0.0 };
        if d > max_abs {
            max_abs = d;
            max_idx = i;
        }
        if r > max_rel {
            max_rel = r;
        }
    }
    println!(
        "pixel_loss parity ({NB} blocks, 8x8): max|Δ| = {max_abs:.3e}, max|rel| = {max_rel:.3e} at block {max_idx}"
    );
    // 8th power chain — sub-ulp f64 ops compound; relative tol 1e-12 is generous.
    assert!(
        max_rel < 1e-10,
        "pixel_loss relative diverges: {max_rel:.3e}"
    );
    println!("\n✓ pixel_loss parity OK (relative tolerance 1e-10).");
}
