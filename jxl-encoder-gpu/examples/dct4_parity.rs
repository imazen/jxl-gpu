//! Parity test: GPU DCT4 family (4x4, 4x8, 8x4 + matching IDCTs) vs
//! `jxl_encoder_simd::dct4::*_full_scalar`.
//!
//! ```sh
//! cargo run --release --example dct4_parity --features cuda
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
    use jxl_encoder_gpu::launch::dct4::{
        dct_4x4_full, dct_4x8_full, dct_8x4_full, idct_4x4_full, idct_4x8_full, idct_8x4_full,
    };
    use jxl_encoder_simd::{
        dct_4x4_full_scalar, dct_4x8_full_scalar, dct_8x4_full_scalar, idct_4x4_full_scalar,
        idct_4x8_full_scalar, idct_8x4_full_scalar,
    };

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    const NB: usize = 64;
    const N: usize = NB * 64;

    let mut input = vec![0.0f32; N];
    for b in 0..NB {
        for i in 0..64 {
            let v = ((b * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
            input[b * 64 + i] = v + 0.05 * (b as f32 / NB as f32);
        }
    }

    let mut all_ok = true;

    macro_rules! check_pair {
        ($name:literal, $rt:expr, $tol_fwd:expr, $tol_inv:expr,
         $gpu_fwd:path, $gpu_inv:path, $cpu_fwd:path, $cpu_inv:path) => {{
            // CPU forward
            let mut cpu_dct = vec![0.0f32; N];
            for b in 0..NB {
                let inb: &[f32; 64] = (&input[b * 64..b * 64 + 64]).try_into().unwrap();
                let outb: &mut [f32; 64] =
                    (&mut cpu_dct[b * 64..b * 64 + 64]).try_into().unwrap();
                $cpu_fwd(inb, outb);
            }

            // GPU forward
            let h_in = client.create_from_slice(f32::as_bytes(&input));
            let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
            $gpu_fwd(&client, h_in, h_out.clone(), NB as u32);
            let bytes = client.read_one(h_out).expect("read fwd");
            let gpu_dct: &[f32] = f32::from_bytes(&bytes);
            let (mf, mfp) = max_abs_diff(gpu_dct, &cpu_dct);
            let ok_f = mf < $tol_fwd;
            println!(
                concat!("DCT", $name, " forward parity ({} blocks): max|Δ| = {:.3e} at {}  {}"),
                NB, mf, mfp, if ok_f { "✓" } else { "✗" }
            );
            all_ok &= ok_f;

            // CPU inverse on CPU forward output
            let mut cpu_idct = vec![0.0f32; N];
            for b in 0..NB {
                let inb: &[f32; 64] = (&cpu_dct[b * 64..b * 64 + 64]).try_into().unwrap();
                let outb: &mut [f32; 64] =
                    (&mut cpu_idct[b * 64..b * 64 + 64]).try_into().unwrap();
                $cpu_inv(inb, outb);
            }

            // GPU inverse on CPU forward output
            let h_idct_in = client.create_from_slice(f32::as_bytes(&cpu_dct));
            let h_idct_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
            $gpu_inv(&client, h_idct_in, h_idct_out.clone(), NB as u32);
            let bytes = client.read_one(h_idct_out).expect("read inv");
            let gpu_idct: &[f32] = f32::from_bytes(&bytes);
            let (mi, mip) = max_abs_diff(gpu_idct, &cpu_idct);
            let ok_i = mi < $tol_inv;
            println!(
                concat!("IDCT", $name, " parity (CPU DCT → GPU IDCT):  max|Δ| = {:.3e} at {}  {}"),
                mi, mip, if ok_i { "✓" } else { "✗" }
            );
            all_ok &= ok_i;

            if $rt {
                let h_rt_in = client.create_from_slice(f32::as_bytes(&input));
                let h_rt_dct = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
                $gpu_fwd(&client, h_rt_in, h_rt_dct.clone(), NB as u32);
                let h_rt_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
                $gpu_inv(&client, h_rt_dct, h_rt_out.clone(), NB as u32);
                let bytes = client.read_one(h_rt_out).expect("read rt");
                let gpu_rt: &[f32] = f32::from_bytes(&bytes);
                let (mrt, mrtp) = max_abs_diff(gpu_rt, &input);
                let ok_rt = mrt < 5e-5;
                println!(
                    concat!("GPU DCT", $name, "→IDCT roundtrip:           max|Δ| = {:.3e} at {}  {}"),
                    mrt, mrtp, if ok_rt { "✓" } else { "✗" }
                );
                all_ok &= ok_rt;
            }
        }};
    }

    check_pair!("4x4_full", true, 1e-6, 5e-6, dct_4x4_full::<Backend>, idct_4x4_full::<Backend>, dct_4x4_full_scalar, idct_4x4_full_scalar);
    check_pair!("4x8_full", true, 1e-6, 5e-6, dct_4x8_full::<Backend>, idct_4x8_full::<Backend>, dct_4x8_full_scalar, idct_4x8_full_scalar);
    check_pair!("8x4_full", true, 1e-6, 5e-6, dct_8x4_full::<Backend>, idct_8x4_full::<Backend>, dct_8x4_full_scalar, idct_8x4_full_scalar);

    if all_ok {
        println!("\n✓ DCT4 family (6 kernels) parity OK.");
    } else {
        eprintln!("\n✗ DCT4 family parity FAILED.");
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
