//! Parity test: GPU AFV 4x4 DCT vs CPU naive matmul using the same
//! basis matrix. Validates that the kernel computes the spec'd
//! `coeffs[j] = sum_i (basis_t[i][j] * pixels[i])` correctly, not
//! that our basis matrix matches upstream's bit-for-bit (the matrix
//! is private upstream; cross-validation against
//! `afv_transform_from_pixels` would require modeling the full DC
//! packing and is left to a separate end-to-end test).
//!
//! ```sh
//! cargo run --release --example afv_dct_4x4_parity --features cuda
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
fn cpu_afv_dct_4x4(pixels: &[f32; 16], basis_t: &[f32; 256], coeffs: &mut [f32; 16]) {
    *coeffs = [0.0; 16];
    for i in 0..16 {
        let p = pixels[i];
        for j in 0..16 {
            coeffs[j] += basis_t[i * 16 + j] * p;
        }
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]
fn cpu_afv_idct_4x4(coeffs: &[f32; 16], basis_t: &[f32; 256], pixels: &mut [f32; 16]) {
    for i in 0..16 {
        let mut sum = 0.0_f32;
        for j in 0..16 {
            sum += basis_t[i * 16 + j] * coeffs[j];
        }
        pixels[i] = sum;
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]
fn main() {
    use cubecl::prelude::*;
    use jxl_encoder_gpu::kernels::afv::AFV4X4_BASIS_TRANSPOSE;
    use jxl_encoder_gpu::launch::afv::{afv_dct_4x4, afv_idct_4x4};

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    const N_BLOCKS: usize = 32;
    const N: usize = N_BLOCKS * 16;

    // Synthetic 16-pixel sub-blocks.
    let mut input = vec![0.0_f32; N];
    for b in 0..N_BLOCKS {
        for i in 0..16 {
            let v = ((b * 7 + i * 11).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
            input[b * 16 + i] = 0.3 + 0.4 * v;
        }
    }

    // CPU forward.
    let mut cpu_fwd = vec![0.0_f32; N];
    for b in 0..N_BLOCKS {
        let pixels: &[f32; 16] = (&input[b * 16..b * 16 + 16]).try_into().unwrap();
        let coeffs: &mut [f32; 16] =
            (&mut cpu_fwd[b * 16..b * 16 + 16]).try_into().unwrap();
        cpu_afv_dct_4x4(pixels, &AFV4X4_BASIS_TRANSPOSE, coeffs);
    }

    // GPU forward.
    let h_in = client.create_from_slice(f32::as_bytes(&input));
    let h_basis = client.create_from_slice(f32::as_bytes(&AFV4X4_BASIS_TRANSPOSE));
    let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N]));
    afv_dct_4x4::<Backend>(
        &client,
        h_in.clone(),
        h_basis.clone(),
        h_out.clone(),
        N_BLOCKS as u32,
    );
    let bytes = client.read_one(h_out).expect("read fwd");
    let gpu_fwd: &[f32] = f32::from_bytes(&bytes);
    let (m_fwd, p_fwd) = max_abs_diff(gpu_fwd, &cpu_fwd);
    let ok_fwd = m_fwd < 1e-5;
    println!(
        "afv_dct_4x4 (n={N_BLOCKS}):  max|Δ| = {m_fwd:.3e} at idx {p_fwd}  {}",
        if ok_fwd { "✓" } else { "✗" }
    );

    // CPU inverse on CPU forward (roundtrip check).
    let mut cpu_round = vec![0.0_f32; N];
    for b in 0..N_BLOCKS {
        let coeffs: &[f32; 16] = (&cpu_fwd[b * 16..b * 16 + 16]).try_into().unwrap();
        let pixels: &mut [f32; 16] =
            (&mut cpu_round[b * 16..b * 16 + 16]).try_into().unwrap();
        cpu_afv_idct_4x4(coeffs, &AFV4X4_BASIS_TRANSPOSE, pixels);
    }

    // GPU inverse on CPU forward.
    let h_in2 = client.create_from_slice(f32::as_bytes(&cpu_fwd));
    let h_out2 = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N]));
    afv_idct_4x4::<Backend>(
        &client,
        h_in2,
        h_basis.clone(),
        h_out2.clone(),
        N_BLOCKS as u32,
    );
    let bytes = client.read_one(h_out2).expect("read inv");
    let gpu_inv: &[f32] = f32::from_bytes(&bytes);
    let (m_inv, p_inv) = max_abs_diff(gpu_inv, &cpu_round);
    let ok_inv = m_inv < 1e-5;
    println!(
        "afv_idct_4x4 (n={N_BLOCKS}): max|Δ| = {m_inv:.3e} at idx {p_inv}  {}",
        if ok_inv { "✓" } else { "✗" }
    );

    // GPU forward → GPU inverse roundtrip.
    let h_round_in = client.create_from_slice(f32::as_bytes(gpu_fwd));
    let h_round_out = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N]));
    afv_idct_4x4::<Backend>(
        &client,
        h_round_in,
        h_basis,
        h_round_out.clone(),
        N_BLOCKS as u32,
    );
    let bytes = client.read_one(h_round_out).expect("read round");
    let gpu_round: &[f32] = f32::from_bytes(&bytes);
    let (m_rt, p_rt) = max_abs_diff(gpu_round, &input);
    let ok_rt = m_rt < 1e-4;
    println!(
        "afv roundtrip (n={N_BLOCKS}):  max|Δ| vs input = {m_rt:.3e} at idx {p_rt}  {}",
        if ok_rt { "✓" } else { "✗" }
    );

    if !ok_fwd || !ok_inv || !ok_rt {
        std::process::exit(1);
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]
fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut m = 0.0_f32;
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
