//! Parity test: GPU DCT8 / IDCT8 vs `jxl_encoder_simd::dct8::*_scalar`.
//!
//! ```sh
//! cargo run --release --example dct8_parity --features cuda
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
    use jxl_encoder_gpu::launch::dct8::{dct_8x8, idct_8x8};
    use jxl_encoder_simd::{dct_8x8_scalar, idct_8x8_scalar};

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // Cover several blocks; deterministic distinct content per block.
    const NB: usize = 256;
    const N: usize = NB * 64;

    let mut input = vec![0.0f32; N];
    for b in 0..NB {
        for i in 0..64 {
            // Mix of low-frequency, high-frequency, and DC content.
            let v = ((b * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
            input[b * 64 + i] = v + 0.1 * (b as f32 / NB as f32);
        }
    }

    // CPU reference forward.
    let mut cpu_dct = vec![0.0f32; N];
    for b in 0..NB {
        let in_blk: &[f32; 64] = (&input[b * 64..b * 64 + 64]).try_into().unwrap();
        let out_blk: &mut [f32; 64] = (&mut cpu_dct[b * 64..b * 64 + 64]).try_into().unwrap();
        dct_8x8_scalar(in_blk, out_blk);
    }

    // GPU forward.
    let h_in = client.create_from_slice(f32::as_bytes(&input));
    let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    dct_8x8::<Backend>(&client, h_in, h_out.clone(), NB as u32);
    let bytes = client.read_one(h_out).expect("read dct");
    let gpu_dct: &[f32] = f32::from_bytes(&bytes);

    let (mf, mfp) = max_abs_diff(gpu_dct, &cpu_dct);
    println!("DCT8 forward parity (256 blocks):  max|Δ| = {mf:.3e} at {mfp}");

    // CPU reference inverse on the (CPU) DCT output.
    let mut cpu_idct = vec![0.0f32; N];
    for b in 0..NB {
        let in_blk: &[f32; 64] = (&cpu_dct[b * 64..b * 64 + 64]).try_into().unwrap();
        let out_blk: &mut [f32; 64] = (&mut cpu_idct[b * 64..b * 64 + 64]).try_into().unwrap();
        idct_8x8_scalar(in_blk, out_blk);
    }

    // GPU IDCT on CPU's DCT output (so input matches CPU IDCT).
    let h_idct_in = client.create_from_slice(f32::as_bytes(&cpu_dct));
    let h_idct_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    idct_8x8::<Backend>(&client, h_idct_in, h_idct_out.clone(), NB as u32);
    let bytes = client.read_one(h_idct_out).expect("read idct");
    let gpu_idct: &[f32] = f32::from_bytes(&bytes);

    let (mi, mip) = max_abs_diff(gpu_idct, &cpu_idct);
    println!("IDCT8 parity (CPU DCT → GPU IDCT): max|Δ| = {mi:.3e} at {mip}");

    // Roundtrip sanity: GPU DCT then GPU IDCT should ~match input.
    let h_rt_dct_in = client.create_from_slice(f32::as_bytes(&input));
    let h_rt_dct_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    dct_8x8::<Backend>(&client, h_rt_dct_in, h_rt_dct_out.clone(), NB as u32);
    let h_rt_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    idct_8x8::<Backend>(&client, h_rt_dct_out, h_rt_out.clone(), NB as u32);
    let bytes = client.read_one(h_rt_out).expect("read rt");
    let gpu_rt: &[f32] = f32::from_bytes(&bytes);
    let (mrt, mrtp) = max_abs_diff(gpu_rt, &input);
    println!("GPU DCT→IDCT roundtrip:            max|Δ| = {mrt:.3e} at {mrtp}");

    // DCT/IDCT chain has ~6 multiplies + 4 adds in 1D × 2 passes = ~20 ops
    // per coefficient, so ulp-level tolerance is reasonable. The CPU scalar
    // path uses `mul_add` for the SQRT2 step, so FMA-contraction parity is
    // close to bit-exact; rest is plain ops with the same FMA exposure.
    let tol = 5e-6_f32;
    assert!(mf < tol, "DCT8 forward: {mf:.3e} >= {tol:.0e}");
    assert!(mi < tol, "IDCT8: {mi:.3e} >= {tol:.0e}");
    // Roundtrip tolerance is looser because errors compound through both
    // transforms and the input has unit-magnitude content.
    assert!(mrt < 1e-5, "Roundtrip: {mrt:.3e} >= 1e-5");

    println!("\n✓ DCT8 + IDCT8 parity OK.");
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]
fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut m = 0.0f32;
    let mut p = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > m {
            m = d;
            p = i;
        }
    }
    (m, p)
}
