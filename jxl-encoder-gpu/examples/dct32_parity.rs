//! Parity test: GPU DCT32 family (32x32, 32x16, 16x32 + IDCTs) vs
//! `jxl_encoder_simd::dct32::*_scalar` and `idct32::*_scalar`.
//!
//! ```sh
//! cargo run --release --example dct32_parity --features cuda
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
    use jxl_encoder_gpu::launch::dct32::{
        dct_16x32, dct_32x16, dct_32x32, idct_16x32, idct_32x16, idct_32x32,
    };
    use jxl_encoder_simd::{
        dct_16x32_scalar, dct_32x16_scalar, dct_32x32_scalar, idct_16x32_scalar,
        idct_32x16_scalar, idct_32x32_scalar,
    };

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    let mut all_ok = true;

    // -----------------------------------------------------------------------
    // 32x32
    // -----------------------------------------------------------------------
    {
        const NB: usize = 16;
        const SZ: usize = 1024;
        const N: usize = NB * SZ;
        let mut input = vec![0.0f32; N];
        for b in 0..NB {
            for i in 0..SZ {
                let v = ((b * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
                input[b * SZ + i] = v + 0.05 * (b as f32 / NB as f32);
            }
        }

        let mut cpu_dct = vec![0.0f32; N];
        for b in 0..NB {
            let inb: &[f32; SZ] = (&input[b * SZ..b * SZ + SZ]).try_into().unwrap();
            let outb: &mut [f32; SZ] =
                (&mut cpu_dct[b * SZ..b * SZ + SZ]).try_into().unwrap();
            dct_32x32_scalar(inb, outb);
        }
        let h_in = client.create_from_slice(f32::as_bytes(&input));
        let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        dct_32x32::<Backend>(&client, h_in, h_out.clone(), NB as u32);
        let bytes = client.read_one(h_out).expect("read");
        let gpu: &[f32] = f32::from_bytes(&bytes);
        let (mf, mfp) = max_abs_diff(gpu, &cpu_dct);
        let ok_f = mf < 1e-5;
        println!(
            "DCT32x32 forward parity ({NB} blocks): max|Δ| = {mf:.3e} at {mfp}  {}",
            if ok_f { "✓" } else { "✗" }
        );
        all_ok &= ok_f;

        let mut cpu_idct = vec![0.0f32; N];
        for b in 0..NB {
            let inb: &[f32; SZ] = (&cpu_dct[b * SZ..b * SZ + SZ]).try_into().unwrap();
            let outb: &mut [f32; SZ] =
                (&mut cpu_idct[b * SZ..b * SZ + SZ]).try_into().unwrap();
            idct_32x32_scalar(inb, outb);
        }
        let h_idct_in = client.create_from_slice(f32::as_bytes(&cpu_dct));
        let h_idct_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        idct_32x32::<Backend>(&client, h_idct_in, h_idct_out.clone(), NB as u32);
        let bytes = client.read_one(h_idct_out).expect("read");
        let gpu: &[f32] = f32::from_bytes(&bytes);
        let (mi, mip) = max_abs_diff(gpu, &cpu_idct);
        let ok_i = mi < 1e-4;
        println!(
            "IDCT32x32 parity:                       max|Δ| = {mi:.3e} at {mip}  {}",
            if ok_i { "✓" } else { "✗" }
        );
        all_ok &= ok_i;

        // Roundtrip
        let h_rt_in = client.create_from_slice(f32::as_bytes(&input));
        let h_rt_dct = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        dct_32x32::<Backend>(&client, h_rt_in, h_rt_dct.clone(), NB as u32);
        let h_rt_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        idct_32x32::<Backend>(&client, h_rt_dct, h_rt_out.clone(), NB as u32);
        let bytes = client.read_one(h_rt_out).expect("read");
        let gpu_rt: &[f32] = f32::from_bytes(&bytes);
        let (mrt, mrtp) = max_abs_diff(gpu_rt, &input);
        let ok_rt = mrt < 1e-4;
        println!(
            "GPU DCT32x32→IDCT roundtrip:           max|Δ| = {mrt:.3e} at {mrtp}  {}",
            if ok_rt { "✓" } else { "✗" }
        );
        all_ok &= ok_rt;
    }

    // -----------------------------------------------------------------------
    // 32x16 and 16x32
    // -----------------------------------------------------------------------
    macro_rules! check_512 {
        ($name:literal, $rt:expr, $gpu_fwd:path, $gpu_inv:path, $cpu_fwd:path, $cpu_inv:path) => {{
            const NB: usize = 16;
            const SZ: usize = 512;
            const N: usize = NB * SZ;
            let mut input = vec![0.0f32; N];
            for b in 0..NB {
                for i in 0..SZ {
                    let v = ((b * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
                    input[b * SZ + i] = v + 0.05 * (b as f32 / NB as f32);
                }
            }

            let mut cpu_dct = vec![0.0f32; N];
            for b in 0..NB {
                let inb: &[f32; SZ] = (&input[b * SZ..b * SZ + SZ]).try_into().unwrap();
                let outb: &mut [f32; SZ] =
                    (&mut cpu_dct[b * SZ..b * SZ + SZ]).try_into().unwrap();
                $cpu_fwd(inb, outb);
            }
            let h_in = client.create_from_slice(f32::as_bytes(&input));
            let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
            $gpu_fwd(&client, h_in, h_out.clone(), NB as u32);
            let bytes = client.read_one(h_out).expect("read");
            let gpu: &[f32] = f32::from_bytes(&bytes);
            let (mf, mfp) = max_abs_diff(gpu, &cpu_dct);
            let ok_f = mf < 1e-5;
            println!(
                concat!("DCT", $name, " forward parity ({} blocks): max|Δ| = {:.3e} at {}  {}"),
                NB, mf, mfp, if ok_f { "✓" } else { "✗" }
            );
            all_ok &= ok_f;

            let mut cpu_idct = vec![0.0f32; N];
            for b in 0..NB {
                let inb: &[f32; SZ] = (&cpu_dct[b * SZ..b * SZ + SZ]).try_into().unwrap();
                let outb: &mut [f32; SZ] =
                    (&mut cpu_idct[b * SZ..b * SZ + SZ]).try_into().unwrap();
                $cpu_inv(inb, outb);
            }
            let h_idct_in = client.create_from_slice(f32::as_bytes(&cpu_dct));
            let h_idct_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
            $gpu_inv(&client, h_idct_in, h_idct_out.clone(), NB as u32);
            let bytes = client.read_one(h_idct_out).expect("read");
            let gpu: &[f32] = f32::from_bytes(&bytes);
            let (mi, mip) = max_abs_diff(gpu, &cpu_idct);
            let ok_i = mi < 1e-4;
            println!(
                concat!("IDCT", $name, " parity:                       max|Δ| = {:.3e} at {}  {}"),
                mi, mip, if ok_i { "✓" } else { "✗" }
            );
            all_ok &= ok_i;

            if $rt {
                let h_rt_in = client.create_from_slice(f32::as_bytes(&input));
                let h_rt_dct = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
                $gpu_fwd(&client, h_rt_in, h_rt_dct.clone(), NB as u32);
                let h_rt_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
                $gpu_inv(&client, h_rt_dct, h_rt_out.clone(), NB as u32);
                let bytes = client.read_one(h_rt_out).expect("read");
                let gpu_rt: &[f32] = f32::from_bytes(&bytes);
                let (mrt, mrtp) = max_abs_diff(gpu_rt, &input);
                let ok_rt = mrt < 1e-4;
                println!(
                    concat!("GPU DCT", $name, "→IDCT roundtrip:           max|Δ| = {:.3e} at {}  {}"),
                    mrt, mrtp, if ok_rt { "✓" } else { "✗" }
                );
                all_ok &= ok_rt;
            } else {
                println!(
                    concat!("GPU DCT", $name, "→IDCT roundtrip:           skipped (asymmetric layout)")
                );
            }
        }};
    }

    check_512!("32x16", true, dct_32x16::<Backend>, idct_32x16::<Backend>, dct_32x16_scalar, idct_32x16_scalar);
    check_512!("16x32", true, dct_16x32::<Backend>, idct_16x32::<Backend>, dct_16x32_scalar, idct_16x32_scalar);

    if all_ok {
        println!("\n✓ DCT32 family (6 kernels) parity OK.");
    } else {
        eprintln!("\n✗ DCT32 family parity FAILED.");
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
