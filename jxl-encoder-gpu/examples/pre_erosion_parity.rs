//! Parity test: compute_pre_erosion vs scalar.

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
    use jxl_encoder_gpu::launch::adaptive_quant::compute_pre_erosion as gpu_pe;
    use jxl_encoder_simd::compute_pre_erosion_scalar;

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    const W: usize = 256;
    const H: usize = 192;
    const N: usize = W * H;

    // Synthetic Y plane (small XYB-y values).
    let mut xyb_y = vec![0.0f32; N];
    for i in 0..N {
        let u = (i as f32) / (N as f32);
        let v = ((i * 17) % 251) as f32 / 251.0 - 0.5;
        xyb_y[i] = 0.4 * (u - 0.5) + 0.005 * v;
    }

    // Tile bounds (typical encoder tile).
    let tile_x0: usize = 32;
    let tile_y0: usize = 24;
    let tile_x1: usize = 224;
    let tile_y1: usize = 168;

    // CPU
    let (cpu_pe, cpu_w, cpu_h) =
        compute_pre_erosion_scalar(&xyb_y, W, H, tile_x0, tile_y0, tile_x1, tile_y1);

    // Recompute matching x0, y_start to pass to GPU
    let x0 = tile_x0.saturating_sub(4);
    let y_start = tile_y0.saturating_sub(4);

    let n_out = cpu_w * cpu_h;
    let h_y = client.create_from_slice(f32::as_bytes(&xyb_y));
    let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n_out]));
    gpu_pe::<Backend>(
        &client,
        h_y,
        h_out.clone(),
        N,
        W as u32,
        H as u32,
        x0 as u32,
        y_start as u32,
        cpu_w as u32,
        cpu_h as u32,
    );
    let bytes = client.read_one(h_out).expect("read");
    let gpu_pe_out: &[f32] = f32::from_bytes(&bytes);

    let m = gpu_pe_out
        .iter()
        .zip(cpu_pe.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    let m_rel = gpu_pe_out
        .iter()
        .zip(cpu_pe.iter())
        .map(|(g, c)| {
            if c.abs() > 1e-30 {
                (g - c).abs() / c.abs()
            } else {
                0.0
            }
        })
        .fold(0.0f32, f32::max);
    println!(
        "compute_pre_erosion ({W}x{H} → {cpu_w}x{cpu_h}): max|Δ|={m:.3e}, max|rel|={m_rel:.3e}"
    );
    let ok = m_rel < 5e-4;
    if ok {
        println!("\n✓ compute_pre_erosion parity OK.");
    } else {
        eprintln!("\n✗ compute_pre_erosion parity FAILED.");
        std::process::exit(1);
    }
}
