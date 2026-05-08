//! Parity test: GPU `epf_step0_kernel` vs an inline CPU reference
//! that mirrors `jxl_encoder::vardct::epf::epf_step0_strip` exactly.
//!
//! We can't reuse a `jxl_encoder_simd::epf_step0_scalar` because step 0
//! is currently only implemented inside the upstream `jxl-encoder` crate
//! (as the private `fn epf_step0_strip`). The inline reference is a
//! line-by-line port of the upstream serial logic.

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
    use jxl_encoder_gpu::launch::epf::{epf_step0 as gpu_epf0, pad_plane as gpu_pad};

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    const W: usize = 64;
    const H: usize = 48;
    const PAD: usize = 3;
    const XB: usize = W / 8;
    const YB: usize = H / 8;
    const N: usize = W * H;
    const STRIDE: usize = W + 2 * PAD;
    const N_PADDED: usize = STRIDE * (H + 2 * PAD);

    // Synthetic XYB-like planes.
    let mut raw_x = vec![0.0f32; N];
    let mut raw_y = vec![0.0f32; N];
    let mut raw_b = vec![0.0f32; N];
    for i in 0..N {
        let u = (i as f32) / (N as f32);
        let v = (((i * 17) % N) as f32) / (N as f32);
        raw_x[i] = 0.005 * (u - 0.5);
        raw_y[i] = 0.4 * (u - 0.5) + 0.005 * v;
        raw_b[i] = -0.2 * u + 0.003 * v;
    }

    // Pad on GPU and read back so CPU + GPU consume identical padded inputs.
    let pad_one = |raw: &[f32]| -> Vec<f32> {
        let h_src = client.create_from_slice(f32::as_bytes(raw));
        let h_dst = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N_PADDED]));
        gpu_pad::<Backend>(
            &client,
            h_src,
            h_dst.clone(),
            W as u32,
            H as u32,
            PAD as u32,
        );
        let bytes = client.read_one(h_dst).expect("pad");
        f32::from_bytes(&bytes).to_vec()
    };
    let in_x = pad_one(&raw_x);
    let in_y = pad_one(&raw_y);
    let in_b = pad_one(&raw_b);

    // inv_sigma map: typical values 0.1-0.5 with occasional zeros (no-filter).
    let n_blocks = XB * YB;
    let mut inv_sigma = vec![0.0f32; n_blocks];
    for b in 0..n_blocks {
        inv_sigma[b] = if b.is_multiple_of(7) {
            0.0
        } else {
            0.1 + 0.4 * (b as f32 / n_blocks as f32)
        };
    }

    // CPU reference: call upstream's epf_step0_strip directly via the
    // __internals cargo feature. G5.1-compliant — no hand-rolled
    // re-derivation. Constants (EPF_PASS0_SIGMA_SCALE,
    // EPF_BORDER_SAD_MUL, EPF_CHANNEL_SCALE, EPF0_NEIGHBORS) live in
    // the upstream module and don't need re-declaration here.
    const EPF_PASS0_SIGMA_SCALE: f32 = 0.9;
    const EPF_BORDER_SAD_MUL: f32 = 2.0 / 3.0;
    let sigma_scale = EPF_PASS0_SIGMA_SCALE * 1.65;
    let border_sigma_mul = EPF_BORDER_SAD_MUL;

    let mut cpu_x = vec![0.0f32; N];
    let mut cpu_y = vec![0.0f32; N];
    let mut cpu_b = vec![0.0f32; N];
    jxl_encoder::__internals::epf_step0_strip_free(
        [&in_x, &in_y, &in_b],
        &inv_sigma,
        XB,
        W,
        0, // py_start
        H, // rows = full image height
        STRIDE,
        PAD,
        &mut cpu_x,
        &mut cpu_y,
        &mut cpu_b,
    );

    // GPU
    let h_ix = client.create_from_slice(f32::as_bytes(&in_x));
    let h_iy = client.create_from_slice(f32::as_bytes(&in_y));
    let h_ib = client.create_from_slice(f32::as_bytes(&in_b));
    let h_ox = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    let h_oy = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    let h_ob = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    let h_is = client.create_from_slice(f32::as_bytes(&inv_sigma));
    gpu_epf0::<Backend>(
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

    let max_d = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0_f32, f32::max)
    };
    let mx = max_d(gx, &cpu_x);
    let my = max_d(gy, &cpu_y);
    let mbo = max_d(gbo, &cpu_b);

    // EPF chains 12 absolute-value SADs per pixel through a divide. Allow
    // a small FP epsilon. Step 1's parity example uses the same gate.
    const TOL: f32 = 5e-5;
    let ok = mx < TOL && my < TOL && mbo < TOL;
    println!(
        "epf_step0 max|Δ| X={mx:.3e} Y={my:.3e} B={mbo:.3e} (tol={TOL:.0e}) {}",
        if ok { "OK" } else { "FAIL" }
    );
    if !ok {
        std::process::exit(1);
    }
}
