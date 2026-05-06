//! Parity test: pad_plane + epf_step2 vs jxl_encoder_simd scalars.

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
    use jxl_encoder_gpu::launch::epf::{
        epf_step1 as gpu_epf1, epf_step2 as gpu_epf2, pad_plane as gpu_pad,
    };
    use jxl_encoder_simd::{epf_step1_scalar, epf_step2_scalar, pad_plane};

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    let mut all_ok = true;

    // ---- pad_plane ----
    {
        const W: usize = 128;
        const H: usize = 96;
        const PAD: usize = 4;
        let mut src = vec![0.0f32; W * H];
        for i in 0..(W * H) {
            src[i] = 0.4 * ((i * 13) as f32 / (W * H) as f32) - 0.1;
        }
        let cpu = pad_plane(&src, W, H, PAD);

        let dst_w = W + 2 * PAD;
        let dst_h = H + 2 * PAD;
        let dst_n = dst_w * dst_h;
        let h_src = client.create_from_slice(f32::as_bytes(&src));
        let h_dst = client.create_from_slice(f32::as_bytes(&vec![0.0f32; dst_n]));
        gpu_pad::<Backend>(
            &client,
            h_src,
            h_dst.clone(),
            W as u32,
            H as u32,
            PAD as u32,
        );
        let bytes = client.read_one(h_dst).expect("read");
        let gpu: &[f32] = f32::from_bytes(&bytes);

        let max_diff = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        let ok = max_diff == 0.0; // pure copy, must be bit-exact
        println!(
            "pad_plane ({W}x{H}, pad={PAD}): max|Δ| = {max_diff:.3e}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    // ---- epf_step2 ----
    {
        const PAD: usize = 3; // EPF needs >= 1 for cross kernel; use 3 for both steps
        const XB: usize = 8;
        const YB: usize = 6;
        const W: usize = XB * 8;
        const H: usize = YB * 8;
        const N: usize = W * H;
        const STRIDE: usize = W + 2 * PAD;
        const PADDED_N: usize = STRIDE * (H + 2 * PAD);

        // Synthetic XYB content (small magnitudes typical of opsin space).
        let mut raw_x = vec![0.0f32; N];
        let mut raw_y = vec![0.0f32; N];
        let mut raw_b = vec![0.0f32; N];
        for i in 0..N {
            let u = (i as f32) / (N as f32);
            let v = ((i * 17) % 251) as f32 / 251.0 - 0.5;
            raw_x[i] = 0.05 * v - 0.01 * u;
            raw_y[i] = 0.4 * (u - 0.5) + 0.005 * v;
            raw_b[i] = -0.2 * u + 0.003 * v;
        }
        let in_x = pad_plane(&raw_x, W, H, PAD);
        let in_y = pad_plane(&raw_y, W, H, PAD);
        let in_b = pad_plane(&raw_b, W, H, PAD);

        // inv_sigma: typical values ~0.1-0.5 per block, occasional zeros
        let n_blocks = XB * YB;
        let mut inv_sigma = vec![0.0f32; n_blocks];
        for b in 0..n_blocks {
            inv_sigma[b] = if b.is_multiple_of(7) {
                0.0
            } else {
                0.1 + 0.4 * (b as f32 / n_blocks as f32)
            };
        }

        let sigma_scale = 0.6f32;
        let border_sigma_mul = 0.667f32;

        // CPU
        let mut cpu_x = vec![0.0f32; N];
        let mut cpu_y = vec![0.0f32; N];
        let mut cpu_b = vec![0.0f32; N];
        epf_step2_scalar(
            &in_x,
            &in_y,
            &in_b,
            &mut cpu_x,
            &mut cpu_y,
            &mut cpu_b,
            &inv_sigma,
            XB,
            W,
            H,
            STRIDE,
            PAD,
            sigma_scale,
            border_sigma_mul,
        );

        // GPU
        let h_ix = client.create_from_slice(f32::as_bytes(&in_x));
        let h_iy = client.create_from_slice(f32::as_bytes(&in_y));
        let h_ib = client.create_from_slice(f32::as_bytes(&in_b));
        let h_ox = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        let h_oy = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        let h_ob = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        let h_is = client.create_from_slice(f32::as_bytes(&inv_sigma));
        gpu_epf2::<Backend>(
            &client,
            h_ix,
            h_iy,
            h_ib,
            h_ox.clone(),
            h_oy.clone(),
            h_ob.clone(),
            h_is,
            W as u32,
            H as u32,
            XB as u32,
            YB as u32,
            PAD as u32,
            sigma_scale,
            border_sigma_mul,
        );
        let xb = client.read_one(h_ox).expect("x");
        let yb = client.read_one(h_oy).expect("y");
        let bb = client.read_one(h_ob).expect("b");
        let gx: &[f32] = f32::from_bytes(&xb);
        let gy: &[f32] = f32::from_bytes(&yb);
        let gbo: &[f32] = f32::from_bytes(&bb);

        let mx = gx
            .iter()
            .zip(cpu_x.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        let my = gy
            .iter()
            .zip(cpu_y.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        let mbo = gbo
            .iter()
            .zip(cpu_b.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        // EPF chains absolute-value SADs across 3 channels through a divide.
        // FMA contraction can drift by a few ulps. 1e-4 absolute is generous
        // for opsin-magnitude (~0.1 typical) — ~1e-3 relative.
        let tol = 1e-4f32;
        let _ = PADDED_N;
        let ok = mx < tol && my < tol && mbo < tol;
        println!(
            "epf_step2 ({W}x{H}, {XB}x{YB} blocks): X {mx:.3e}, Y {my:.3e}, B {mbo:.3e}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    // ---- epf_step1 ----
    {
        const PAD: usize = 3;
        const XB: usize = 8;
        const YB: usize = 6;
        const W: usize = XB * 8;
        const H: usize = YB * 8;
        const N: usize = W * H;
        const STRIDE: usize = W + 2 * PAD;

        let mut raw_x = vec![0.0f32; N];
        let mut raw_y = vec![0.0f32; N];
        let mut raw_b = vec![0.0f32; N];
        for i in 0..N {
            let u = (i as f32) / (N as f32);
            let v = ((i * 17) % 251) as f32 / 251.0 - 0.5;
            raw_x[i] = 0.05 * v - 0.01 * u;
            raw_y[i] = 0.4 * (u - 0.5) + 0.005 * v;
            raw_b[i] = -0.2 * u + 0.003 * v;
        }
        let in_x = pad_plane(&raw_x, W, H, PAD);
        let in_y = pad_plane(&raw_y, W, H, PAD);
        let in_b = pad_plane(&raw_b, W, H, PAD);

        let n_blocks = XB * YB;
        let mut inv_sigma = vec![0.0f32; n_blocks];
        for b in 0..n_blocks {
            inv_sigma[b] = if b.is_multiple_of(7) {
                0.0
            } else {
                0.1 + 0.4 * (b as f32 / n_blocks as f32)
            };
        }

        let sigma_scale = 0.65f32;
        let border_sigma_mul = 0.667f32;

        let mut cpu_x = vec![0.0f32; N];
        let mut cpu_y = vec![0.0f32; N];
        let mut cpu_b = vec![0.0f32; N];
        epf_step1_scalar(
            &in_x,
            &in_y,
            &in_b,
            &mut cpu_x,
            &mut cpu_y,
            &mut cpu_b,
            &inv_sigma,
            XB,
            W,
            H,
            STRIDE,
            PAD,
            sigma_scale,
            border_sigma_mul,
        );

        let h_ix = client.create_from_slice(f32::as_bytes(&in_x));
        let h_iy = client.create_from_slice(f32::as_bytes(&in_y));
        let h_ib = client.create_from_slice(f32::as_bytes(&in_b));
        let h_ox = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        let h_oy = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        let h_ob = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        let h_is = client.create_from_slice(f32::as_bytes(&inv_sigma));
        gpu_epf1::<Backend>(
            &client,
            h_ix,
            h_iy,
            h_ib,
            h_ox.clone(),
            h_oy.clone(),
            h_ob.clone(),
            h_is,
            W as u32,
            H as u32,
            XB as u32,
            YB as u32,
            PAD as u32,
            sigma_scale,
            border_sigma_mul,
        );
        let xb = client.read_one(h_ox).expect("x");
        let yb = client.read_one(h_oy).expect("y");
        let bb = client.read_one(h_ob).expect("b");
        let gx: &[f32] = f32::from_bytes(&xb);
        let gy: &[f32] = f32::from_bytes(&yb);
        let gbo: &[f32] = f32::from_bytes(&bb);

        let mx = gx
            .iter()
            .zip(cpu_x.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        let my = gy
            .iter()
            .zip(cpu_y.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        let mbo = gbo
            .iter()
            .zip(cpu_b.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        // EPF step1 has 5x more SAD ops + 4x weighted sums → more FMA noise
        // than step2. Tolerance scaled accordingly.
        let tol = 5e-4f32;
        let ok = mx < tol && my < tol && mbo < tol;
        println!(
            "epf_step1 ({W}x{H}, {XB}x{YB} blocks): X {mx:.3e}, Y {my:.3e}, B {mbo:.3e}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    if all_ok {
        println!("\n✓ pad_plane + epf_step1 + epf_step2 parity OK.");
    } else {
        eprintln!("\n✗ EPF parity FAILED.");
        std::process::exit(1);
    }
}
