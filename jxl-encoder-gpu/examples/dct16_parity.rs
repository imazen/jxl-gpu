//! Parity test: GPU DCT16x16 / IDCT16x16 vs `jxl_encoder_simd` scalar.
//!
//! ```sh
//! cargo run --release --example dct16_parity --features cuda
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
    use jxl_encoder_gpu::launch::dct16::{dct_16x16, idct_16x16};
    use jxl_encoder_simd::{dct_16x16_scalar, idct_16x16_scalar};

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    const NB: usize = 64;
    const N: usize = NB * 256;

    let mut input = vec![0.0f32; N];
    for b in 0..NB {
        for i in 0..256 {
            let v = ((b * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
            input[b * 256 + i] = v + 0.05 * (b as f32 / NB as f32);
        }
    }

    let mut cpu_dct = vec![0.0f32; N];
    for b in 0..NB {
        let inb: &[f32; 256] = (&input[b * 256..b * 256 + 256]).try_into().unwrap();
        let outb: &mut [f32; 256] = (&mut cpu_dct[b * 256..b * 256 + 256]).try_into().unwrap();
        dct_16x16_scalar(inb, outb);
    }

    let h_in = client.create_from_slice(f32::as_bytes(&input));
    let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    dct_16x16::<Backend>(&client, h_in, h_out.clone(), NB as u32);
    let bytes = client.read_one(h_out).expect("read dct");
    let gpu_dct: &[f32] = f32::from_bytes(&bytes);
    let (mf, mfp) = max_abs_diff(gpu_dct, &cpu_dct);
    println!("DCT16x16 forward parity ({NB} blocks):  max|Δ| = {mf:.3e} at {mfp}");

    let mut cpu_idct = vec![0.0f32; N];
    for b in 0..NB {
        let inb: &[f32; 256] = (&cpu_dct[b * 256..b * 256 + 256]).try_into().unwrap();
        let outb: &mut [f32; 256] = (&mut cpu_idct[b * 256..b * 256 + 256]).try_into().unwrap();
        idct_16x16_scalar(inb, outb);
    }

    let h_idct_in = client.create_from_slice(f32::as_bytes(&cpu_dct));
    let h_idct_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    idct_16x16::<Backend>(&client, h_idct_in, h_idct_out.clone(), NB as u32);
    let bytes = client.read_one(h_idct_out).expect("read idct");
    let gpu_idct: &[f32] = f32::from_bytes(&bytes);
    let (mi, mip) = max_abs_diff(gpu_idct, &cpu_idct);
    println!("IDCT16x16 parity (CPU DCT → GPU IDCT): max|Δ| = {mi:.3e} at {mip}");

    // Roundtrip
    let h_rt_in = client.create_from_slice(f32::as_bytes(&input));
    let h_rt_dct = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    dct_16x16::<Backend>(&client, h_rt_in, h_rt_dct.clone(), NB as u32);
    let h_rt_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    idct_16x16::<Backend>(&client, h_rt_dct, h_rt_out.clone(), NB as u32);
    let bytes = client.read_one(h_rt_out).expect("read rt");
    let gpu_rt: &[f32] = f32::from_bytes(&bytes);
    let (mrt, mrtp) = max_abs_diff(gpu_rt, &input);
    println!("GPU DCT→IDCT roundtrip:                max|Δ| = {mrt:.3e} at {mrtp}");

    // Larger butterflies → more ops → looser tolerance.
    let tol_fwd = 5e-6_f32;
    let tol_inv = 1e-5_f32;
    let tol_rt = 5e-5_f32;
    assert!(mf < tol_fwd, "DCT16x16 forward: {mf:.3e} >= {tol_fwd:.0e}");
    assert!(mi < tol_inv, "IDCT16x16: {mi:.3e} >= {tol_inv:.0e}");
    assert!(mrt < tol_rt, "Roundtrip: {mrt:.3e} >= {tol_rt:.0e}");
    println!("\n✓ DCT16x16 + IDCT16x16 parity OK.");
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
