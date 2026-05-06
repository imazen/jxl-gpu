//! Parity test: GPU dequant_dct8 vs `jxl_encoder_simd::dequant_dct8_scalar`.
//!
//! ```sh
//! cargo run --release --example dequant_parity --features cuda
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
    use jxl_encoder_gpu::launch::dequant::dequant_dct8;
    use jxl_encoder_simd::dequant_dct8_scalar;

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    const NB: usize = 64;
    const N: usize = NB * 64;

    // Synthesize quant_ac (typical post-quantization integer values).
    let mut quant = [vec![0i32; N], vec![0i32; N], vec![0i32; N]];
    for c in 0..3 {
        for b in 0..NB {
            for i in 0..64 {
                let r = (i / 8) as i32;
                let col = (i % 8) as i32;
                let v = (((b as i32 + c as i32) * 7 + i as i32 * 13) % 23) - 11;
                quant[c][b * 64 + i] = if i == 0 { 0 } else { v - r * col / 4 };
            }
        }
    }

    // Per-block weights (replicated, one weight table per block).
    let mut weights_per = vec![0.0f32; 64];
    for i in 0..64 {
        let r = (i / 8) as f32;
        let col = (i % 8) as f32;
        weights_per[i] = 1.0 + 0.5 * (r + col);
    }
    let mut weights = [vec![0.0f32; N], vec![0.0f32; N], vec![0.0f32; N]];
    for c in 0..3 {
        for b in 0..NB {
            weights[c][b * 64..b * 64 + 64].copy_from_slice(&weights_per);
        }
    }

    let mut qac_qm = [vec![0.0f32; NB], vec![0.0f32; NB], vec![0.0f32; NB]];
    for b in 0..NB {
        qac_qm[0][b] = 0.6 + 0.05 * (b as f32);
        qac_qm[1][b] = 0.7 + 0.05 * (b as f32);
        qac_qm[2][b] = 0.5 + 0.05 * (b as f32);
    }

    let mut x_factor = vec![0.0f32; NB];
    let mut b_factor = vec![0.0f32; NB];
    for b in 0..NB {
        x_factor[b] = -0.05 + 0.001 * (b as f32);
        b_factor[b] = 0.4 + 0.002 * (b as f32);
    }

    // CPU reference.
    let mut cpu_out = [vec![0.0f32; N], vec![0.0f32; N], vec![0.0f32; N]];
    for b in 0..NB {
        let qx: &[i32; 64] = (&quant[0][b * 64..b * 64 + 64]).try_into().unwrap();
        let qy: &[i32; 64] = (&quant[1][b * 64..b * 64 + 64]).try_into().unwrap();
        let qb: &[i32; 64] = (&quant[2][b * 64..b * 64 + 64]).try_into().unwrap();
        let wx: &[f32; 64] = (&weights_per[..]).try_into().unwrap();
        let wy = wx;
        let wb = wx;
        let (a, rest) = cpu_out.split_at_mut(1);
        let (b_arr, c_arr) = rest.split_at_mut(1);
        let ox: &mut [f32; 64] = (&mut a[0][b * 64..b * 64 + 64]).try_into().unwrap();
        let oy: &mut [f32; 64] = (&mut b_arr[0][b * 64..b * 64 + 64]).try_into().unwrap();
        let ob: &mut [f32; 64] = (&mut c_arr[0][b * 64..b * 64 + 64]).try_into().unwrap();
        dequant_dct8_scalar(
            qx,
            qy,
            qb,
            wx,
            wy,
            wb,
            [qac_qm[0][b], qac_qm[1][b], qac_qm[2][b]],
            x_factor[b],
            b_factor[b],
            ox,
            oy,
            ob,
        );
    }

    // GPU.
    let h_qx = client.create_from_slice(i32::as_bytes(&quant[0]));
    let h_qy = client.create_from_slice(i32::as_bytes(&quant[1]));
    let h_qb = client.create_from_slice(i32::as_bytes(&quant[2]));
    let h_wx = client.create_from_slice(f32::as_bytes(&weights[0]));
    let h_wy = client.create_from_slice(f32::as_bytes(&weights[1]));
    let h_wb = client.create_from_slice(f32::as_bytes(&weights[2]));
    let h_qmx = client.create_from_slice(f32::as_bytes(&qac_qm[0]));
    let h_qmy = client.create_from_slice(f32::as_bytes(&qac_qm[1]));
    let h_qmb = client.create_from_slice(f32::as_bytes(&qac_qm[2]));
    let h_xf = client.create_from_slice(f32::as_bytes(&x_factor));
    let h_bf = client.create_from_slice(f32::as_bytes(&b_factor));
    let h_ox = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    let h_oy = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    let h_ob = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));

    dequant_dct8::<Backend>(
        &client,
        h_qx,
        h_qy,
        h_qb,
        h_wx,
        h_wy,
        h_wb,
        h_qmx,
        h_qmy,
        h_qmb,
        h_xf,
        h_bf,
        h_ox.clone(),
        h_oy.clone(),
        h_ob.clone(),
        NB as u32,
    );
    let xb = client.read_one(h_ox).expect("x");
    let yb = client.read_one(h_oy).expect("y");
    let bb = client.read_one(h_ob).expect("b");
    let gx: &[f32] = f32::from_bytes(&xb);
    let gy: &[f32] = f32::from_bytes(&yb);
    let gbo: &[f32] = f32::from_bytes(&bb);

    let (mx, mxp) = max_abs_diff(gx, &cpu_out[0]);
    let (my, myp) = max_abs_diff(gy, &cpu_out[1]);
    let (mb, mbp) = max_abs_diff(gbo, &cpu_out[2]);
    println!("dequant X: max|Δ| = {mx:.3e} at {mxp}");
    println!("dequant Y: max|Δ| = {my:.3e} at {myp}");
    println!("dequant B: max|Δ| = {mb:.3e} at {mbp}");

    // Y channel is a single multiply chain (no FMA opportunity) → bit-exact.
    // X/B channels include a CfL `+ x_factor * dq_y` term that the GPU
    // contracts to FMA but the CPU `_scalar` reference does not. Same
    // FMA-noise pattern as mask1x1 (gotcha G6.4); 5e-5 absolute on
    // dequant magnitudes ~1-10 is ~5e-6 relative — well below the
    // sensitivity of downstream cost-model AC-strategy decisions.
    let tol_y = 1e-7_f32;
    let tol_xb = 5e-5_f32;
    assert!(my < tol_y, "Y diverges (no FMA expected): {my:.3e}");
    assert!(mx < tol_xb, "X diverges: {mx:.3e}");
    assert!(mb < tol_xb, "B diverges: {mb:.3e}");
    println!("\n✓ dequant_dct8 parity OK (Y bit-exact; X/B {tol_xb:.0e} FMA noise).");
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
