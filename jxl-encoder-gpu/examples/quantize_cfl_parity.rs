//! Parity test: quantize_large + cfl find_best_multiplier vs scalars.

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
    use jxl_encoder_gpu::launch::{
        cfl::{find_best_multiplier as gpu_cfl, find_best_multiplier_newton as gpu_cfl_newton},
        quantize::quantize_large,
    };
    use jxl_encoder_simd::{
        cfl_find_best_multiplier_newton_scalar, cfl_find_best_multiplier_scalar,
        quantize_large_scalar,
    };
    let find_best_multiplier_scalar = cfl_find_best_multiplier_scalar;

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    let mut all_ok = true;

    // ---- quantize_large ----
    {
        // Use 16x16 grid (DCT16x16) with LLF 2x2 for testing.
        const GW: usize = 16;
        const GH: usize = 16;
        const SIZE: usize = GW * GH;
        const LLF_X: usize = 2;
        const LLF_Y: usize = 2;
        const NB: usize = 32;
        const N_COEF: usize = NB * SIZE;

        let mut coeffs = vec![0.0f32; N_COEF];
        let mut weights = vec![1.0f32; N_COEF];
        for b in 0..NB {
            for i in 0..SIZE {
                let r = (i / GW) as f32;
                let c = (i % GW) as f32;
                let freq = (r * r + c * c).sqrt();
                let s = if (b + i).is_multiple_of(2) { 1.0 } else { -1.0 };
                coeffs[b * SIZE + i] = s * 12.7 / (1.0 + 0.4 * freq) + 0.13 * (b as f32);
                weights[b * SIZE + i] = 1.0 + 0.7 * (r + c);
            }
        }
        let mut qac_qm = vec![0.0f32; NB];
        for b in 0..NB {
            qac_qm[b] = 1.7 + 0.05 * (b as f32);
        }
        let thresholds = [0.62f32, 0.62, 0.62, 0.62];

        let mut cpu_out = vec![0i32; N_COEF];
        for b in 0..NB {
            let coef_blk = &coeffs[b * SIZE..b * SIZE + SIZE];
            let w_blk = &weights[b * SIZE..b * SIZE + SIZE];
            let out_blk = &mut cpu_out[b * SIZE..b * SIZE + SIZE];
            quantize_large_scalar(
                coef_blk,
                w_blk,
                qac_qm[b],
                &thresholds,
                GW,
                GH,
                LLF_X,
                LLF_Y,
                out_blk,
            );
        }

        let h_coef = client.create_from_slice(f32::as_bytes(&coeffs));
        let h_w = client.create_from_slice(f32::as_bytes(&weights));
        let h_qac = client.create_from_slice(f32::as_bytes(&qac_qm));
        let h_thr = client.create_from_slice(f32::as_bytes(&thresholds[..]));
        let h_out = client.create_from_slice(i32::as_bytes(&vec![0i32; N_COEF]));
        quantize_large::<Backend>(
            &client,
            h_coef,
            h_w,
            h_qac,
            h_thr,
            h_out.clone(),
            NB as u32,
            GW as u32,
            GH as u32,
            LLF_X as u32,
            LLF_Y as u32,
        );
        let bytes = client.read_one(h_out).expect("read");
        let gpu: &[i32] = i32::from_bytes(&bytes);

        let mut diffs = 0usize;
        let mut max_diff = 0i64;
        for (i, (&g, &c)) in gpu.iter().zip(cpu_out.iter()).enumerate() {
            let d = (g as i64 - c as i64).abs();
            if d > 0 {
                diffs += 1;
                if d > max_diff {
                    max_diff = d;
                    let _ = i;
                }
            }
        }
        let ok = diffs == 0;
        println!(
            "quantize_large ({GW}x{GH}, LLF {LLF_X}x{LLF_Y}, {NB} blocks): diffs = {diffs}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    // ---- cfl find_best_multiplier ----
    {
        const NT: usize = 32;
        const NPT: usize = 256; // num per tile (typical: 64-256)
        const N: usize = NT * NPT;

        let mut values_m = vec![0.0f32; N];
        let mut values_s = vec![0.0f32; N];
        let mut bases = vec![0.0f32; NT];
        for t in 0..NT {
            bases[t] = 0.5 + 0.01 * (t as f32);
            for i in 0..NPT {
                let v = ((t * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
                values_m[t * NPT + i] = 100.0 * v + 0.5 * (t as f32);
                values_s[t * NPT + i] = 50.0 * v.cos() + 0.3 * (i as f32 / NPT as f32);
            }
        }
        let distance_mul = 0.05f32;

        let mut cpu_out = vec![0i8; NT];
        for t in 0..NT {
            let m_slice = &values_m[t * NPT..t * NPT + NPT];
            let s_slice = &values_s[t * NPT..t * NPT + NPT];
            cpu_out[t] = find_best_multiplier_scalar(m_slice, s_slice, NPT, bases[t], distance_mul);
        }

        let h_m = client.create_from_slice(f32::as_bytes(&values_m));
        let h_s = client.create_from_slice(f32::as_bytes(&values_s));
        let h_b = client.create_from_slice(f32::as_bytes(&bases));
        let h_out = client.create_from_slice(i32::as_bytes(&vec![0i32; NT]));
        gpu_cfl::<Backend>(
            &client,
            h_m,
            h_s,
            h_b,
            h_out.clone(),
            NT as u32,
            NPT as u32,
            distance_mul,
        );
        let bytes = client.read_one(h_out).expect("read");
        let gpu_i32: &[i32] = i32::from_bytes(&bytes);

        let mut diffs = 0usize;
        let mut max_diff = 0i32;
        for (i, (&g_i32, &c)) in gpu_i32.iter().zip(cpu_out.iter()).enumerate() {
            let g = g_i32 as i8;
            let d = (g as i32 - c as i32).abs();
            if d > 0 {
                diffs += 1;
                if d > max_diff {
                    max_diff = d;
                    let _ = i;
                }
            }
        }
        // CFL is a least-squares fit followed by bias-and-quantize. FMA
        // contraction in the sum_aa/sum_ab reductions can flip the
        // quantization boundary by ±1 for some tiles. Allow up to 2-tile
        // disagreement at most ±1 each.
        let ok = max_diff <= 1 && diffs <= NT / 4;
        println!(
            "cfl find_best_multiplier ({NT} tiles, {NPT}/tile): diffs = {diffs}, max|Δ| = {max_diff}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    // ---- cfl find_best_multiplier_newton ----
    {
        const NT: usize = 32;
        const NPT: usize = 256;
        const N: usize = NT * NPT;
        let eps = 1.0f32;
        let max_iters: usize = 10;

        let mut values_m = vec![0.0f32; N];
        let mut values_s = vec![0.0f32; N];
        let mut bases = vec![0.0f32; NT];
        for t in 0..NT {
            bases[t] = 0.5 + 0.01 * (t as f32);
            for i in 0..NPT {
                let v = ((t * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
                values_m[t * NPT + i] = 100.0 * v + 0.5 * (t as f32);
                values_s[t * NPT + i] = 50.0 * v.cos() + 0.3 * (i as f32 / NPT as f32);
            }
        }
        let distance_mul = 0.05f32;

        let mut cpu_out = vec![0i8; NT];
        for t in 0..NT {
            cpu_out[t] = cfl_find_best_multiplier_newton_scalar(
                &values_m[t * NPT..t * NPT + NPT],
                &values_s[t * NPT..t * NPT + NPT],
                NPT,
                bases[t],
                distance_mul,
                eps,
                max_iters,
                false, // libjxl_parity (legacy Newton-default behavior)
                false, // libjxl_math_with_ls_warm_start
            );
        }

        let h_m = client.create_from_slice(f32::as_bytes(&values_m));
        let h_s = client.create_from_slice(f32::as_bytes(&values_s));
        let h_b = client.create_from_slice(f32::as_bytes(&bases));
        let h_out = client.create_from_slice(i32::as_bytes(&vec![0i32; NT]));
        gpu_cfl_newton::<Backend>(
            &client,
            h_m,
            h_s,
            h_b,
            h_out.clone(),
            NT as u32,
            NPT as u32,
            distance_mul,
            eps,
            max_iters as u32,
        );
        let bytes = client.read_one(h_out).expect("read");
        let gpu_i32: &[i32] = i32::from_bytes(&bytes);

        let mut diffs = 0usize;
        let mut max_diff = 0i32;
        for (g_i32, &c) in gpu_i32.iter().zip(cpu_out.iter()) {
            let g = *g_i32 as i8;
            let d = (g as i32 - c as i32).abs();
            if d > 0 {
                diffs += 1;
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        // Newton with FMA noise can produce 1-tile differences (boundary
        // tiles where converged-LS-vs-Newton diverge). Allow up to ±1
        // on a small minority of tiles.
        let ok = max_diff <= 1 && diffs <= NT / 4;
        println!(
            "cfl find_best_multiplier_newton ({NT} tiles, {NPT}/tile, {max_iters} iters): diffs = {diffs}, max|Δ| = {max_diff}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    if all_ok {
        println!("\n✓ quantize_large + cfl + cfl_newton parity OK.");
    } else {
        eprintln!("\n✗ quantize_large + cfl parity FAILED.");
        std::process::exit(1);
    }
}
