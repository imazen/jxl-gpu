//! Combined parity test: `block_l2` + `quantize_dct8` vs the scalar
//! references in `jxl-encoder-simd`.
//!
//! ```sh
//! cargo run --release --example phase2_small_parity --features cuda
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
    use jxl_encoder_gpu::launch::{block_l2::block_l2, quantize::quantize_dct8};
    use jxl_encoder_simd::{compute_block_l2_errors_scalar, quantize_dct8_scalar};

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    let mut all_ok = true;

    // ---- block_l2 ----
    {
        // 16 x 12 blocks = 128 x 96 padded image.
        const XB: usize = 16;
        const YB: usize = 12;
        const PADDED_W: usize = XB * 8;
        const H: usize = YB * 8;
        const N: usize = PADDED_W * H;

        let mut orig = [
            vec![0.0f32; N],
            vec![0.0f32; N],
            vec![0.0f32; N],
        ];
        let mut recon = [
            vec![0.0f32; N],
            vec![0.0f32; N],
            vec![0.0f32; N],
        ];
        let mut mask = vec![0.0f32; N];
        for i in 0..N {
            let u = (i as f32) / (N as f32);
            // Mix of low-mag and higher-mag values for the 8x8 reduction.
            orig[0][i] = 0.4 * u - 0.2;
            orig[1][i] = 0.3 * (1.0 - u) + 0.05;
            orig[2][i] = 0.1 * u + 0.02;
            recon[0][i] = orig[0][i] + 0.005 * ((i as f32 * 0.31).sin());
            recon[1][i] = orig[1][i] + 0.003 * ((i as f32 * 0.27).cos());
            recon[2][i] = orig[2][i] + 0.001 * ((i as f32 * 0.19).sin());
            mask[i] = 0.4 + 0.6 * u;
        }

        let cpu_out = compute_block_l2_errors_scalar(
            [&orig[0], &orig[1], &orig[2]],
            [&recon[0], &recon[1], &recon[2]],
            &mask,
            XB,
            YB,
            PADDED_W,
            XB * YB,
        );

        let h_ox = client.create_from_slice(f32::as_bytes(&orig[0]));
        let h_oy = client.create_from_slice(f32::as_bytes(&orig[1]));
        let h_ob = client.create_from_slice(f32::as_bytes(&orig[2]));
        let h_rx = client.create_from_slice(f32::as_bytes(&recon[0]));
        let h_ry = client.create_from_slice(f32::as_bytes(&recon[1]));
        let h_rb = client.create_from_slice(f32::as_bytes(&recon[2]));
        let h_m = client.create_from_slice(f32::as_bytes(&mask));
        let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; XB * YB]));

        block_l2::<Backend>(
            &client,
            h_ox,
            h_oy,
            h_ob,
            h_rx,
            h_ry,
            h_rb,
            h_m,
            h_out.clone(),
            XB as u32,
            YB as u32,
            PADDED_W as u32,
        );
        let bytes = client.read_one(h_out).expect("read block_l2");
        let gpu: &[f32] = f32::from_bytes(&bytes);
        let (m, p) = max_abs_diff(gpu, &cpu_out);
        // 64-pixel × 3-channel reduction; ulp-level is fine.
        let ok = m < 5e-5;
        println!(
            "block_l2 ({XB}x{YB} blocks):       max|Δ| = {m:.3e} at block {p}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    // ---- quantize_dct8 ----
    {
        const NB: usize = 64;
        const N_COEF: usize = NB * 64;

        // Synthesize plausible DCT coefficients: DC large, AC decreasing
        // with frequency. Avoid hitting exact half-integer values after
        // quantization to keep round-ties-even agreement clean even
        // before the manual ties-to-even kernel implementation.
        let mut coeffs = vec![0.0f32; N_COEF];
        for b in 0..NB {
            for i in 0..64 {
                let r = (i / 8) as f32;
                let c = (i % 8) as f32;
                let freq = (r * r + c * c).sqrt();
                // deterministic pseudo-random sign
                let s = if (b + i).is_multiple_of(2) { 1.0 } else { -1.0 };
                coeffs[b * 64 + i] = s * 12.7 / (1.0 + 0.4 * freq) + 0.13 * (b as f32);
            }
        }

        // Use a single weights table (DCT8 standard quant matrix-ish);
        // replicate per block so the layout matches the GPU launcher.
        let mut weights_per_block = vec![1.0f32; 64];
        for i in 0..64 {
            let r = (i / 8) as f32;
            let c = (i % 8) as f32;
            weights_per_block[i] = 1.0 + 0.7 * (r + c);
        }
        let mut weights = vec![0.0f32; N_COEF];
        for b in 0..NB {
            weights[b * 64..b * 64 + 64].copy_from_slice(&weights_per_block);
        }

        let mut qac_qm = vec![0.0f32; NB];
        for b in 0..NB {
            qac_qm[b] = 1.7 + 0.05 * (b as f32);
        }

        let thresholds = [0.62f32, 0.62, 0.62, 0.62];

        // CPU reference per block.
        let mut cpu_out = vec![0i32; N_COEF];
        for b in 0..NB {
            let coef_blk: &[f32; 64] = (&coeffs[b * 64..b * 64 + 64]).try_into().unwrap();
            let w_blk: &[f32; 64] = (&weights_per_block[..]).try_into().unwrap();
            let out_blk: &mut [i32; 64] = (&mut cpu_out[b * 64..b * 64 + 64]).try_into().unwrap();
            quantize_dct8_scalar(coef_blk, w_blk, qac_qm[b], &thresholds, out_blk);
        }

        let h_coef = client.create_from_slice(f32::as_bytes(&coeffs));
        let h_w = client.create_from_slice(f32::as_bytes(&weights));
        let h_qac = client.create_from_slice(f32::as_bytes(&qac_qm));
        let h_thr = client.create_from_slice(f32::as_bytes(&thresholds[..]));
        let h_out = client.create_from_slice(i32::as_bytes(&vec![0i32; N_COEF]));
        quantize_dct8::<Backend>(
            &client,
            h_coef,
            h_w,
            h_qac,
            h_thr,
            h_out.clone(),
            NB as u32,
        );
        let bytes = client.read_one(h_out).expect("read quantize");
        let gpu: &[i32] = i32::from_bytes(&bytes);

        let mut diffs = 0usize;
        let mut max_diff = 0i64;
        let mut max_diff_idx = 0usize;
        for (i, (&g, &c)) in gpu.iter().zip(cpu_out.iter()).enumerate() {
            let d = (g as i64 - c as i64).abs();
            if d > 0 {
                diffs += 1;
            }
            if d > max_diff {
                max_diff = d;
                max_diff_idx = i;
            }
        }
        // We expect zero divergence — round-ties-even is implemented
        // bit-exact in the kernel.
        let ok = diffs == 0;
        println!(
            "quantize_dct8 ({NB} blocks):    diffs = {diffs}, max|Δ| = {max_diff} at coef {max_diff_idx}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    if all_ok {
        println!("\n✓ Phase-2 small kernels parity OK.");
    } else {
        eprintln!("\n✗ Phase-2 small kernels parity FAILED.");
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
