//! Parity test: GPU `xyb_forward` / `xyb_inverse` vs
//! `jxl_encoder_simd::xyb::*_scalar`.
//!
//! Run with the matching backend feature enabled:
//!
//! ```sh
//! cargo run --release --example xyb_parity --features cuda
//! ```
//!
//! Reports `max_abs_diff` per channel and asserts `< 1e-5` absolute
//! (opsin-scale tolerance per CLAUDE.md).

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
    use jxl_encoder_gpu::launch::xyb::{xyb_forward, xyb_inverse};
    use jxl_encoder_simd::{forward_xyb_scalar, inverse_xyb_planar_scalar};

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // Deterministic test pattern: 256x256 mix of low/mid/high luminance
    // and saturated colors.
    const W: usize = 256;
    const H: usize = 256;
    const N: usize = W * H;
    let mut r = vec![0.0f32; N];
    let mut g = vec![0.0f32; N];
    let mut b = vec![0.0f32; N];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let u = (x as f32) / (W as f32 - 1.0);
            let v = (y as f32) / (H as f32 - 1.0);
            r[i] = u * (0.7 + 0.3 * v);
            g[i] = (1.0 - u) * (0.7 + 0.3 * v);
            b[i] = 0.5 * (u + v);
        }
    }

    // CPU reference forward.
    let mut x_cpu = vec![0.0f32; N];
    let mut y_cpu = vec![0.0f32; N];
    let mut b_cpu = vec![0.0f32; N];
    forward_xyb_scalar(&r, &g, &b, &mut x_cpu, &mut y_cpu, &mut b_cpu, N);

    // GPU forward.
    let h_r = client.create_from_slice(f32::as_bytes(&r));
    let h_g = client.create_from_slice(f32::as_bytes(&g));
    let h_b = client.create_from_slice(f32::as_bytes(&b));
    let h_x_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    let h_y_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    let h_b_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));

    xyb_forward::<Backend>(
        &client,
        h_r.clone(),
        h_g.clone(),
        h_b.clone(),
        h_x_out.clone(),
        h_y_out.clone(),
        h_b_out.clone(),
        N as u32,
    );

    let x_bytes = client.read_one(h_x_out).expect("read x");
    let y_bytes = client.read_one(h_y_out).expect("read y");
    let b_bytes = client.read_one(h_b_out).expect("read b");
    let x_gpu: &[f32] = f32::from_bytes(&x_bytes);
    let y_gpu: &[f32] = f32::from_bytes(&y_bytes);
    let b_gpu: &[f32] = f32::from_bytes(&b_bytes);

    let (mx, mxp) = max_abs_diff(x_gpu, &x_cpu);
    let (my, myp) = max_abs_diff(y_gpu, &y_cpu);
    let (mb, mbp) = max_abs_diff(b_gpu, &b_cpu);

    println!("FORWARD parity (256x256 ramp):");
    println!("  X: max|Δ| = {mx:.3e} at idx {mxp}");
    println!("  Y: max|Δ| = {my:.3e} at idx {myp}");
    println!("  B: max|Δ| = {mb:.3e} at idx {mbp}");

    let tol_forward = 1e-5_f32;
    assert!(
        mx < tol_forward,
        "X channel diverges: {mx:.3e} >= {tol_forward:.0e}"
    );
    assert!(
        my < tol_forward,
        "Y channel diverges: {my:.3e} >= {tol_forward:.0e}"
    );
    assert!(
        mb < tol_forward,
        "B channel diverges: {mb:.3e} >= {tol_forward:.0e}"
    );

    // Inverse: feed CPU XYB through GPU inverse, compare to CPU inverse.
    let mut r_cpu_back = vec![0.0f32; N];
    let mut g_cpu_back = vec![0.0f32; N];
    let mut b_cpu_back = vec![0.0f32; N];
    inverse_xyb_planar_scalar(
        &x_cpu,
        &y_cpu,
        &b_cpu,
        &mut r_cpu_back,
        &mut g_cpu_back,
        &mut b_cpu_back,
        N,
    );

    let h_xc = client.create_from_slice(f32::as_bytes(&x_cpu));
    let h_yc = client.create_from_slice(f32::as_bytes(&y_cpu));
    let h_bc = client.create_from_slice(f32::as_bytes(&b_cpu));
    let h_rg = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    let h_gg = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    let h_bg = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));

    xyb_inverse::<Backend>(
        &client,
        h_xc,
        h_yc,
        h_bc,
        h_rg.clone(),
        h_gg.clone(),
        h_bg.clone(),
        N as u32,
    );

    let r_bytes = client.read_one(h_rg).expect("read r");
    let g_bytes = client.read_one(h_gg).expect("read g");
    let b_bytes = client.read_one(h_bg).expect("read b");
    let r_gpu: &[f32] = f32::from_bytes(&r_bytes);
    let g_gpu: &[f32] = f32::from_bytes(&g_bytes);
    let b_gpu: &[f32] = f32::from_bytes(&b_bytes);

    let (mr, mrp) = max_abs_diff(r_gpu, &r_cpu_back);
    let (mg, mgp) = max_abs_diff(g_gpu, &g_cpu_back);
    let (mbi, mbip) = max_abs_diff(b_gpu, &b_cpu_back);

    println!("\nINVERSE parity (CPU XYB → GPU vs CPU planar RGB):");
    println!("  R: max|Δ| = {mr:.3e} at idx {mrp}");
    println!("  G: max|Δ| = {mg:.3e} at idx {mgp}");
    println!("  B: max|Δ| = {mbi:.3e} at idx {mbip}");

    let tol_inv = 1e-5_f32;
    assert!(mr < tol_inv, "R inverse diverges: {mr:.3e}");
    assert!(mg < tol_inv, "G inverse diverges: {mg:.3e}");
    assert!(mbi < tol_inv, "B inverse diverges: {mbi:.3e}");

    println!("\n✓ XYB forward + inverse parity OK (tolerance < 1e-5).");
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
