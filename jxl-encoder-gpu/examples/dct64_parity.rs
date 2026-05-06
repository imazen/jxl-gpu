//! Parity test: GPU DCT64 family vs `jxl_encoder_simd::{dct64,idct64}::*_scalar`.

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
    use jxl_encoder_gpu::launch::dct64::{
        dct_32x64, dct_64x32, dct_64x64, idct_32x64, idct_64x32, idct_64x64,
    };
    use jxl_encoder_simd::{
        dct_32x64_scalar, dct_64x32_scalar, dct_64x64_scalar, idct_32x64_scalar, idct_64x32_scalar,
        idct_64x64_scalar,
    };

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    let mut all_ok = true;

    macro_rules! check {
        ($name:literal, $sz:expr, $rt:expr,
         $gpu_fwd:path, $gpu_inv:path, $cpu_fwd:path, $cpu_inv:path) => {{
            const NB: usize = 4;
            let sz: usize = $sz;
            let n: usize = NB * sz;
            let mut input = vec![0.0f32; n];
            for b in 0..NB {
                for i in 0..sz {
                    let v = ((b * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
                    input[b * sz + i] = v + 0.05 * (b as f32 / NB as f32);
                }
            }

            // CPU forward
            let mut cpu_dct = vec![0.0f32; n];
            for b in 0..NB {
                let inb_slice = &input[b * sz..b * sz + sz];
                let outb_slice = &mut cpu_dct[b * sz..b * sz + sz];
                let inb = unsafe { &*(inb_slice.as_ptr() as *const [f32; $sz]) };
                let outb = unsafe { &mut *(outb_slice.as_mut_ptr() as *mut [f32; $sz]) };
                $cpu_fwd(inb, outb);
            }
            let h_in = client.create_from_slice(f32::as_bytes(&input));
            let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
            $gpu_fwd(&client, h_in, h_out.clone(), NB as u32);
            let bytes = client.read_one(h_out).expect("read");
            let gpu: &[f32] = f32::from_bytes(&bytes);
            let (mf, mfp) = max_abs_diff(gpu, &cpu_dct);
            let ok_f = mf < 1e-4;
            println!(
                concat!("DCT", $name, " forward parity: max|Δ| = {:.3e} at {}  {}"),
                mf,
                mfp,
                if ok_f { "✓" } else { "✗" }
            );
            all_ok &= ok_f;

            // CPU inverse on CPU forward
            let mut cpu_idct = vec![0.0f32; n];
            for b in 0..NB {
                let inb_slice = &cpu_dct[b * sz..b * sz + sz];
                let outb_slice = &mut cpu_idct[b * sz..b * sz + sz];
                let inb = unsafe { &*(inb_slice.as_ptr() as *const [f32; $sz]) };
                let outb = unsafe { &mut *(outb_slice.as_mut_ptr() as *mut [f32; $sz]) };
                $cpu_inv(inb, outb);
            }
            let h_idct_in = client.create_from_slice(f32::as_bytes(&cpu_dct));
            let h_idct_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
            $gpu_inv(&client, h_idct_in, h_idct_out.clone(), NB as u32);
            let bytes = client.read_one(h_idct_out).expect("read");
            let gpu: &[f32] = f32::from_bytes(&bytes);
            let (mi, mip) = max_abs_diff(gpu, &cpu_idct);
            let ok_i = mi < 1e-3;
            println!(
                concat!(
                    "IDCT",
                    $name,
                    " parity:           max|Δ| = {:.3e} at {}  {}"
                ),
                mi,
                mip,
                if ok_i { "✓" } else { "✗" }
            );
            all_ok &= ok_i;

            if $rt {
                let h_rt_in = client.create_from_slice(f32::as_bytes(&input));
                let h_rt_dct = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
                $gpu_fwd(&client, h_rt_in, h_rt_dct.clone(), NB as u32);
                let h_rt_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
                $gpu_inv(&client, h_rt_dct, h_rt_out.clone(), NB as u32);
                let bytes = client.read_one(h_rt_out).expect("read");
                let gpu_rt: &[f32] = f32::from_bytes(&bytes);
                let (mrt, mrtp) = max_abs_diff(gpu_rt, &input);
                let ok_rt = mrt < 1e-3;
                println!(
                    concat!(
                        "GPU DCT",
                        $name,
                        "→IDCT roundtrip: max|Δ| = {:.3e} at {}  {}"
                    ),
                    mrt,
                    mrtp,
                    if ok_rt { "✓" } else { "✗" }
                );
                all_ok &= ok_rt;
            }
        }};
    }

    check!(
        "64x64",
        4096,
        true,
        dct_64x64::<Backend>,
        idct_64x64::<Backend>,
        dct_64x64_scalar,
        idct_64x64_scalar
    );
    check!(
        "64x32",
        2048,
        true,
        dct_64x32::<Backend>,
        idct_64x32::<Backend>,
        dct_64x32_scalar,
        idct_64x32_scalar
    );
    check!(
        "32x64",
        2048,
        true,
        dct_32x64::<Backend>,
        idct_32x64::<Backend>,
        dct_32x64_scalar,
        idct_32x64_scalar
    );

    if all_ok {
        println!("\n✓ DCT64 family (6 kernels) parity OK.");
    } else {
        eprintln!("\n✗ DCT64 family parity FAILED.");
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
