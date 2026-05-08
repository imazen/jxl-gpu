//! Parity test: GPU DCT2X2 transform vs
//! `jxl_encoder::vardct::dct::{dct2x2_transform, inverse_dct2x2_transform}`.
//!
//! ```sh
//! cargo run --release --example dct2x2_parity --features cuda
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
    use jxl_encoder::vardct::dct::{dct2x2_transform, inverse_dct2x2_transform};
    use jxl_encoder_gpu::launch::dct2x2::{dct2x2_forward, dct2x2_inverse};

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    const N_BLOCKS: usize = 32;
    const N: usize = N_BLOCKS * 64;

    let mut input = vec![0.0f32; N];
    for b in 0..N_BLOCKS {
        for i in 0..64 {
            let v = ((b * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
            input[b * 64 + i] = 0.3 + 0.4 * v;
        }
    }

    // CPU forward.
    let mut cpu_fwd = vec![0.0f32; N];
    for b in 0..N_BLOCKS {
        let pixels: &[f32; 64] = (&input[b * 64..b * 64 + 64]).try_into().unwrap();
        let coeffs: &mut [f32; 64] = (&mut cpu_fwd[b * 64..b * 64 + 64]).try_into().unwrap();
        dct2x2_transform(pixels, coeffs);
    }

    // GPU forward.
    let h_in = client.create_from_slice(f32::as_bytes(&input));
    let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    dct2x2_forward::<Backend>(&client, h_in, h_out.clone(), N_BLOCKS as u32);
    let bytes = client.read_one(h_out).expect("read fwd");
    let gpu_fwd: &[f32] = f32::from_bytes(&bytes);

    let (m, p) = max_abs_diff(gpu_fwd, &cpu_fwd);
    let ok_fwd = m < 1e-6;
    println!(
        "dct2x2_forward (n={N_BLOCKS}):  max|Δ| = {m:.3e} at idx {p}  {}",
        if ok_fwd { "✓" } else { "✗" }
    );

    // CPU inverse on the CPU forward result.
    let mut cpu_round = vec![0.0f32; N];
    for b in 0..N_BLOCKS {
        let coeffs: &[f32; 64] = (&cpu_fwd[b * 64..b * 64 + 64]).try_into().unwrap();
        let pixels: &mut [f32; 64] = (&mut cpu_round[b * 64..b * 64 + 64]).try_into().unwrap();
        inverse_dct2x2_transform(coeffs, pixels);
    }

    // GPU inverse on the same CPU forward output.
    let h_in2 = client.create_from_slice(f32::as_bytes(&cpu_fwd));
    let h_out2 = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    dct2x2_inverse::<Backend>(&client, h_in2, h_out2.clone(), N_BLOCKS as u32);
    let bytes = client.read_one(h_out2).expect("read inv");
    let gpu_inv: &[f32] = f32::from_bytes(&bytes);

    let (m2, p2) = max_abs_diff(gpu_inv, &cpu_round);
    let ok_inv = m2 < 1e-6;
    println!(
        "dct2x2_inverse (n={N_BLOCKS}):  max|Δ| = {m2:.3e} at idx {p2}  {}",
        if ok_inv { "✓" } else { "✗" }
    );

    // Roundtrip GPU forward → GPU inverse vs original input.
    let h_round_in = client.create_from_slice(f32::as_bytes(gpu_fwd));
    let h_round_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    dct2x2_inverse::<Backend>(&client, h_round_in, h_round_out.clone(), N_BLOCKS as u32);
    let bytes = client.read_one(h_round_out).expect("read round");
    let gpu_round: &[f32] = f32::from_bytes(&bytes);
    let (m3, p3) = max_abs_diff(gpu_round, &input);
    let ok_round = m3 < 1e-5;
    println!(
        "dct2x2 roundtrip (n={N_BLOCKS}):  max|Δ| vs original = {m3:.3e} at idx {p3}  {}",
        if ok_round { "✓" } else { "✗" }
    );

    if !ok_fwd || !ok_inv || !ok_round {
        std::process::exit(1);
    }
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
