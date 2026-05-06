//! Parity test: GPU rectangular DCT16 (16x8, 8x16) + matching IDCT vs
//! `jxl_encoder_simd` scalar references.
//!
//! ```sh
//! cargo run --release --example dct16_rect_parity --features cuda
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
    use jxl_encoder_gpu::launch::dct16::{dct_16x8, dct_8x16, idct_16x8, idct_8x16};
    use jxl_encoder_simd::{
        dct_16x8_scalar, dct_8x16_scalar, idct_16x8_scalar, idct_8x16_scalar,
    };

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    const NB: usize = 64;
    const N: usize = NB * 128;

    let mut input = vec![0.0f32; N];
    for b in 0..NB {
        for i in 0..128 {
            let v = ((b * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
            input[b * 128 + i] = v + 0.05 * (b as f32 / NB as f32);
        }
    }

    let mut all_ok = true;

    // Note on roundtrip: dct_16x8 outputs in 8x16 layout (transposed),
    // while idct_16x8 expects 16x8 layout input. They are NOT direct
    // inverses without an intermediate transpose (verified: CPU
    // dct_16x8_scalar → idct_16x8_scalar also produces ~1.0 abs error).
    // The encoder pipeline transposes between them. dct_8x16 +
    // idct_8x16 use the same 8x16 layout so they DO roundtrip.
    // Per-direction parity is the load-bearing invariant; each kernel
    // matches its CPU counterpart bit-near-exactly.
    macro_rules! check {
        ($name:literal, $rt:expr, $gpu_fwd:path, $gpu_inv:path, $cpu_fwd:path, $cpu_inv:path) => {{
            let mut cpu_dct = vec![0.0f32; N];
            for b in 0..NB {
                let inb: &[f32; 128] = (&input[b * 128..b * 128 + 128]).try_into().unwrap();
                let outb: &mut [f32; 128] =
                    (&mut cpu_dct[b * 128..b * 128 + 128]).try_into().unwrap();
                $cpu_fwd(inb, outb);
            }

            let h_in = client.create_from_slice(f32::as_bytes(&input));
            let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
            $gpu_fwd(&client, h_in, h_out.clone(), NB as u32);
            let bytes = client.read_one(h_out).expect("read fwd");
            let gpu_dct: &[f32] = f32::from_bytes(&bytes);
            let (mf, mfp) = max_abs_diff(gpu_dct, &cpu_dct);
            let ok_f = mf < 1e-5;
            println!(
                concat!("DCT", $name, " forward parity ({} blocks): max|Δ| = {:.3e} at {}  {}"),
                NB, mf, mfp, if ok_f { "✓" } else { "✗" }
            );
            all_ok &= ok_f;

            let mut cpu_idct = vec![0.0f32; N];
            for b in 0..NB {
                let inb: &[f32; 128] = (&cpu_dct[b * 128..b * 128 + 128]).try_into().unwrap();
                let outb: &mut [f32; 128] =
                    (&mut cpu_idct[b * 128..b * 128 + 128]).try_into().unwrap();
                $cpu_inv(inb, outb);
            }

            let h_idct_in = client.create_from_slice(f32::as_bytes(&cpu_dct));
            let h_idct_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
            $gpu_inv(&client, h_idct_in, h_idct_out.clone(), NB as u32);
            let bytes = client.read_one(h_idct_out).expect("read inv");
            let gpu_idct: &[f32] = f32::from_bytes(&bytes);
            let (mi, mip) = max_abs_diff(gpu_idct, &cpu_idct);
            let ok_i = mi < 1e-5;
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
            } else {
                println!(
                    concat!("GPU DCT", $name, "→IDCT roundtrip:           skipped (asymmetric layout — see comment)")
                );
            }
        }};
    }

    check!("16x8", false, dct_16x8::<Backend>, idct_16x8::<Backend>, dct_16x8_scalar, idct_16x8_scalar);
    check!("8x16", true,  dct_8x16::<Backend>, idct_8x16::<Backend>, dct_8x16_scalar, idct_8x16_scalar);

    if all_ok {
        println!("\n✓ DCT16 rectangular family parity OK.");
    } else {
        eprintln!("\n✗ DCT16 rectangular parity FAILED.");
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
