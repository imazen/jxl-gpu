//! Parity test: GPU raw inverse 4×4 + 4×8 DCT vs
//! `jxl_encoder::vardct::dct::{idct_4x4, idct_4x8}`.
//!
//! ```sh
//! cargo run --release --example idct4_raw_parity --features cuda
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
    use jxl_encoder::vardct::dct::{idct_4x4, idct_4x8, dct_4x4, dct_4x8};
    use jxl_encoder_gpu::launch::idct4_raw::{idct_4x4_raw, idct_4x8_raw};

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // Generate inputs by running upstream forward DCT on synthetic pixels;
    // this gives us realistic coefficient distributions for the inverse test.
    {
        const N_BLOCKS: usize = 16;
        const N: usize = N_BLOCKS * 16;
        let mut pixels = vec![0.0_f32; N];
        for b in 0..N_BLOCKS {
            for i in 0..16 {
                let v = ((b * 7 + i * 11).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
                pixels[b * 16 + i] = 0.3 + 0.4 * v;
            }
        }
        let mut coeffs = vec![0.0_f32; N];
        for b in 0..N_BLOCKS {
            let p: &[f32; 16] = (&pixels[b * 16..b * 16 + 16]).try_into().unwrap();
            let c: &mut [f32; 16] = (&mut coeffs[b * 16..b * 16 + 16]).try_into().unwrap();
            dct_4x4(p, c);
        }
        let mut cpu_out = vec![0.0_f32; N];
        for b in 0..N_BLOCKS {
            let c: &[f32; 16] = (&coeffs[b * 16..b * 16 + 16]).try_into().unwrap();
            let p: &mut [f32; 16] = (&mut cpu_out[b * 16..b * 16 + 16]).try_into().unwrap();
            idct_4x4(c, p);
        }
        let h_in = client.create_from_slice(f32::as_bytes(&coeffs));
        let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N]));
        idct_4x4_raw::<Backend>(&client, h_in, h_out.clone(), N_BLOCKS as u32);
        let bytes = client.read_one(h_out).expect("read 4x4");
        let gpu: &[f32] = f32::from_bytes(&bytes);
        let (m, p) = max_abs_diff(gpu, &cpu_out);
        let ok = m < 1e-5;
        println!(
            "idct_4x4_raw (n={N_BLOCKS}): max|Δ| = {m:.3e} at idx {p}  {}",
            if ok { "✓" } else { "✗" }
        );
        if !ok {
            eprintln!("  cpu[{p}]={:.6}  gpu[{p}]={:.6}", cpu_out[p], gpu[p]);
            std::process::exit(1);
        }
    }

    // 4×8.
    {
        const N_BLOCKS: usize = 16;
        const N: usize = N_BLOCKS * 32;
        let mut pixels = vec![0.0_f32; N];
        for b in 0..N_BLOCKS {
            for i in 0..32 {
                let v = ((b * 13 + i * 17).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
                pixels[b * 32 + i] = 0.3 + 0.4 * v;
            }
        }
        let mut coeffs = vec![0.0_f32; N];
        for b in 0..N_BLOCKS {
            let p: &[f32; 32] = (&pixels[b * 32..b * 32 + 32]).try_into().unwrap();
            let c: &mut [f32; 32] = (&mut coeffs[b * 32..b * 32 + 32]).try_into().unwrap();
            dct_4x8(p, c);
        }
        let mut cpu_out = vec![0.0_f32; N];
        for b in 0..N_BLOCKS {
            let c: &[f32; 32] = (&coeffs[b * 32..b * 32 + 32]).try_into().unwrap();
            let p: &mut [f32; 32] = (&mut cpu_out[b * 32..b * 32 + 32]).try_into().unwrap();
            idct_4x8(c, p);
        }
        let h_in = client.create_from_slice(f32::as_bytes(&coeffs));
        let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; N]));
        idct_4x8_raw::<Backend>(&client, h_in, h_out.clone(), N_BLOCKS as u32);
        let bytes = client.read_one(h_out).expect("read 4x8");
        let gpu: &[f32] = f32::from_bytes(&bytes);
        let (m, p) = max_abs_diff(gpu, &cpu_out);
        let ok = m < 1e-5;
        println!(
            "idct_4x8_raw (n={N_BLOCKS}): max|Δ| = {m:.3e} at idx {p}  {}",
            if ok { "✓" } else { "✗" }
        );
        if !ok {
            eprintln!("  cpu[{p}]={:.6}  gpu[{p}]={:.6}", cpu_out[p], gpu[p]);
            std::process::exit(1);
        }
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
