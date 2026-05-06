//! Parity test: entropy_coeffs (pixel + coeff modes) vs scalar.

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
    use jxl_encoder_gpu::launch::entropy::{
        entropy_coeffs_coeff as gpu_coeff, entropy_coeffs_pixel as gpu_pixel,
    };
    use jxl_encoder_simd::entropy_coeffs_scalar;

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    const NB: usize = 64;
    const N: usize = 64;
    const TOTAL: usize = NB * N;

    let mut block_c = vec![0.0f32; TOTAL];
    let mut block_y = vec![0.0f32; TOTAL];
    let mut weights = vec![1.0f32; TOTAL];
    let mut inv_weights = vec![1.0f32; TOTAL];
    for b in 0..NB {
        for i in 0..N {
            let r = (i / 8) as f32;
            let c = (i % 8) as f32;
            let s = if (b + i).is_multiple_of(2) { 1.0 } else { -1.0 };
            block_c[b * N + i] =
                s * 12.7 / (1.0 + 0.4 * (r * r + c * c).sqrt()) + 0.13 * (b as f32);
            block_y[b * N + i] = 0.5 * block_c[b * N + i] + 0.05 * (b as f32);
            weights[b * N + i] = 1.0 + 0.7 * (r + c);
            inv_weights[b * N + i] = 1.0 / weights[b * N + i];
        }
    }

    let cmap_factor = -0.05f32;
    let quant = 1.7f32;
    let k_cost_delta = 10.833f32;
    let k_cost2 = 0.45f32;

    let mut all_ok = true;

    // -- pixel mode --
    {
        let mut cpu_err = vec![0.0f32; TOTAL];
        let mut cpu_out = vec![0.0f32; NB * 4];
        for b in 0..NB {
            let r = jxl_encoder_simd::entropy_coeffs_scalar(
                &block_c[b * N..b * N + N],
                &block_y[b * N..b * N + N],
                &weights[b * N..b * N + N],
                &inv_weights[b * N..b * N + N],
                N,
                cmap_factor,
                quant,
                k_cost_delta,
                0.0,
                true,
                &mut cpu_err[b * N..b * N + N],
            );
            cpu_out[b * 4] = r.entropy_sum;
            cpu_out[b * 4 + 1] = r.nzeros_sum;
            cpu_out[b * 4 + 2] = r.info_loss_sum;
            cpu_out[b * 4 + 3] = r.info_loss2_sum;
        }
        let h_c = client.create_from_slice(f32::as_bytes(&block_c));
        let h_y = client.create_from_slice(f32::as_bytes(&block_y));
        let h_w = client.create_from_slice(f32::as_bytes(&weights));
        let h_iw = client.create_from_slice(f32::as_bytes(&inv_weights));
        let h_err = client.create_from_slice(f32::as_bytes(&vec![0.0f32; TOTAL]));
        let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; NB * 4]));
        gpu_pixel::<Backend>(
            &client,
            h_c,
            h_y,
            h_w,
            h_iw,
            h_err.clone(),
            h_out.clone(),
            NB as u32,
            N as u32,
            cmap_factor,
            quant,
            k_cost_delta,
        );
        let bytes = client.read_one(h_out).expect("read out");
        let gpu_out: &[f32] = f32::from_bytes(&bytes);
        let bytes_e = client.read_one(h_err).expect("read err");
        let gpu_err: &[f32] = f32::from_bytes(&bytes_e);

        let m_out = gpu_out
            .iter()
            .zip(cpu_out.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        let m_err = gpu_err
            .iter()
            .zip(cpu_err.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        let ok = m_out < 1e-3 && m_err < 1e-5;
        println!(
            "entropy_coeffs_pixel ({NB} blocks of {N}): out_max|Δ|={m_out:.3e}, err_max|Δ|={m_err:.3e}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    // -- coeff mode --
    {
        let mut cpu_err = vec![0.0f32; TOTAL];
        let mut cpu_out = vec![0.0f32; NB * 4];
        for b in 0..NB {
            let r = jxl_encoder_simd::entropy_coeffs_scalar(
                &block_c[b * N..b * N + N],
                &block_y[b * N..b * N + N],
                &weights[b * N..b * N + N],
                &inv_weights[b * N..b * N + N],
                N,
                cmap_factor,
                quant,
                k_cost_delta,
                k_cost2,
                false,
                &mut cpu_err[b * N..b * N + N],
            );
            cpu_out[b * 4] = r.entropy_sum;
            cpu_out[b * 4 + 1] = r.nzeros_sum;
            cpu_out[b * 4 + 2] = r.info_loss_sum;
            cpu_out[b * 4 + 3] = r.info_loss2_sum;
        }
        let h_c = client.create_from_slice(f32::as_bytes(&block_c));
        let h_y = client.create_from_slice(f32::as_bytes(&block_y));
        let h_iw = client.create_from_slice(f32::as_bytes(&inv_weights));
        let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; NB * 4]));
        gpu_coeff::<Backend>(
            &client,
            h_c,
            h_y,
            h_iw,
            h_out.clone(),
            NB as u32,
            N as u32,
            cmap_factor,
            quant,
            k_cost_delta,
            k_cost2,
        );
        let bytes = client.read_one(h_out).expect("read out");
        let gpu_out: &[f32] = f32::from_bytes(&bytes);
        let m_out = gpu_out
            .iter()
            .zip(cpu_out.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        let ok = m_out < 1e-3;
        println!(
            "entropy_coeffs_coeff ({NB} blocks of {N}): out_max|Δ|={m_out:.3e}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    if all_ok {
        println!("\n✓ entropy_coeffs (both modes) parity OK.");
    } else {
        eprintln!("\n✗ entropy_coeffs parity FAILED.");
        std::process::exit(1);
    }
    let _ = entropy_coeffs_scalar;
}
