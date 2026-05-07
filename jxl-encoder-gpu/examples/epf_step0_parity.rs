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
        gpu_pad::<Backend>(&client, h_src, h_dst.clone(), W as u32, H as u32, PAD as u32);
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

    // Constants copied from upstream `jxl_encoder::vardct::epf`.
    const EPF_PASS0_SIGMA_SCALE: f32 = 0.9;
    const EPF_BORDER_SAD_MUL: f32 = 2.0 / 3.0;
    const EPF_CHANNEL_SCALE: [f32; 3] = [40.0, 5.0, 3.5];
    const NEIGHBORS: [(isize, isize); 12] = [
        (-2, 0),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -2),
        (0, -1),
        (0, 1),
        (0, 2),
        (1, -1),
        (1, 0),
        (1, 1),
        (2, 0),
    ];
    let sigma_scale = EPF_PASS0_SIGMA_SCALE * 1.65;
    let border_sigma_mul = EPF_BORDER_SAD_MUL;

    // CPU reference: line-by-line port of upstream epf_step0_strip.
    let pad = PAD;
    let mut cpu_x = vec![0.0f32; N];
    let mut cpu_y = vec![0.0f32; N];
    let mut cpu_b = vec![0.0f32; N];
    let planes = [&in_x[..], &in_y[..], &in_b[..]];
    for py in 0..H {
        for px in 0..W {
            let by = py / 8;
            let bx = px / 8;
            let is = inv_sigma[by * XB + bx];
            let oidx = py * W + px;
            if is == 0.0 {
                let pidx = (py + pad) * STRIDE + (px + pad);
                cpu_x[oidx] = planes[0][pidx];
                cpu_y[oidx] = planes[1][pidx];
                cpu_b[oidx] = planes[2][pidx];
                continue;
            }
            let mod_x = px % 8;
            let mod_y = py % 8;
            let at_border = mod_x == 0 || mod_x == 7 || mod_y == 0 || mod_y == 7;
            let bm = if at_border { border_sigma_mul } else { 1.0 };
            let eff_is = is * sigma_scale * bm;

            let cx = px + pad;
            let cy = py + pad;
            let center_idx = cy * STRIDE + cx;
            let mut total_w = 1.0_f32;
            let mut sum_x = planes[0][center_idx];
            let mut sum_y = planes[1][center_idx];
            let mut sum_b = planes[2][center_idx];

            for &(dy, dx) in &NEIGHBORS {
                let nx = (cx as isize + dx) as usize;
                let ny = (cy as isize + dy) as usize;
                // sad_3x3_plus
                let c_off = [
                    cy * STRIDE + cx,
                    (cy - 1) * STRIDE + cx,
                    cy * STRIDE + (cx - 1),
                    cy * STRIDE + (cx + 1),
                    (cy + 1) * STRIDE + cx,
                ];
                let n_off = [
                    ny * STRIDE + nx,
                    (ny - 1) * STRIDE + nx,
                    ny * STRIDE + (nx - 1),
                    ny * STRIDE + (nx + 1),
                    (ny + 1) * STRIDE + nx,
                ];
                let mut sad = 0.0_f32;
                for i in 0..5 {
                    for c in 0..3 {
                        sad += (planes[c][c_off[i]] - planes[c][n_off[i]]).abs()
                            * EPF_CHANNEL_SCALE[c];
                    }
                }
                let weight = (sad * eff_is + 1.0).max(0.0);
                let n_idx = ny * STRIDE + nx;
                total_w += weight;
                sum_x += weight * planes[0][n_idx];
                sum_y += weight * planes[1][n_idx];
                sum_b += weight * planes[2][n_idx];
            }
            let inv_tw = 1.0 / total_w;
            cpu_x[oidx] = sum_x * inv_tw;
            cpu_y[oidx] = sum_y * inv_tw;
            cpu_b[oidx] = sum_b * inv_tw;
        }
    }

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
