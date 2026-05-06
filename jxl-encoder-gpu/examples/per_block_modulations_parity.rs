//! Parity test: per_block_modulations vs scalar.

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
    use jxl_encoder_gpu::launch::adaptive_quant::per_block_modulations as gpu_pbm;
    use jxl_encoder_simd::per_block_modulations_scalar;

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // 16x12 blocks = 128x96 image
    const XB: usize = 16;
    const YB: usize = 12;
    const W: usize = XB * 8;
    const H: usize = YB * 8;
    const N: usize = W * H;
    const STRIDE: usize = W;

    let mut xyb_x = vec![0.0f32; N];
    let mut xyb_y = vec![0.0f32; N];
    let mut xyb_b = vec![0.0f32; N];
    for i in 0..N {
        let u = (i as f32) / (N as f32);
        let v = ((i * 17) % 251) as f32 / 251.0 - 0.5;
        xyb_x[i] = 0.05 * v - 0.01 * u;
        xyb_y[i] = 0.4 * (u - 0.5) + 0.005 * v;
        xyb_b[i] = -0.2 * u + 0.003 * v;
    }

    let n_blocks = XB * YB;
    let aq_map_stride = XB;
    let mut aq_map = vec![0.0f32; n_blocks];
    for b in 0..n_blocks {
        // pre-erosion-like values: small positive
        aq_map[b] = 0.5 + 0.3 * ((b as f32 / n_blocks as f32) - 0.5);
    }

    let butteraugli_target = 1.0f32;
    let scale = 0.39f32;

    // CPU
    let mut cpu_aq = aq_map.clone();
    per_block_modulations_scalar(
        &xyb_x,
        &xyb_y,
        &xyb_b,
        STRIDE,
        butteraugli_target,
        scale,
        0,
        0,
        XB,
        YB,
        &mut cpu_aq,
        aq_map_stride,
    );

    // GPU
    let h_x = client.create_from_slice(f32::as_bytes(&xyb_x));
    let h_y = client.create_from_slice(f32::as_bytes(&xyb_y));
    let h_b = client.create_from_slice(f32::as_bytes(&xyb_b));
    let h_aq = client.create_from_slice(f32::as_bytes(&aq_map));
    gpu_pbm::<Backend>(
        &client,
        h_x,
        h_y,
        h_b,
        h_aq.clone(),
        N,
        n_blocks,
        STRIDE as u32,
        aq_map_stride as u32,
        0,
        0,
        XB as u32,
        YB as u32,
        butteraugli_target,
        scale,
    );
    let bytes = client.read_one(h_aq).expect("read");
    let gpu_aq: &[f32] = f32::from_bytes(&bytes);

    let m = gpu_aq
        .iter()
        .zip(cpu_aq.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    let m_rel = gpu_aq
        .iter()
        .zip(cpu_aq.iter())
        .map(|(g, c)| {
            if c.abs() > 1e-30 {
                (g - c).abs() / c.abs()
            } else {
                0.0
            }
        })
        .fold(0.0f32, f32::max);
    println!("per_block_modulations ({XB}x{YB} blocks): max|Δ|={m:.3e}, max|rel|={m_rel:.3e}");
    // Output passes through fast_pow2f which spans many orders of magnitude
    // (~0.3 to ~6500 in this test). Use relative tolerance only — absolute
    // would need to track output range. 5e-4 relative is well within FMA
    // noise budget for the polynomial chain (fast_log2f + fast_pow2f +
    // ratio_of_deriv_inverted ~5 multiplies + 64-pixel reduction).
    let ok = m_rel < 5e-4;
    if ok {
        println!("\n✓ per_block_modulations parity OK.");
    } else {
        eprintln!("\n✗ per_block_modulations parity FAILED.");
        std::process::exit(1);
    }
}
