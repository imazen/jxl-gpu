//! Combined Phase-1 parity test: gab, gaborish_5x5, mask1x1 each vs the
//! `_scalar` reference in `jxl-encoder-simd`.
//!
//! ```sh
//! cargo run --release --example phase1_parity --features cuda
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
    use jxl_encoder_gpu::launch::{gab::gab_smooth, gaborish::gaborish_5x5, mask1x1::mask1x1};
    use jxl_encoder_simd::{compute_mask1x1_scalar, gab_smooth_scalar, gaborish_5x5_scalar};

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // Use a non-square non-power-of-2 size to catch boundary bugs.
    const W: usize = 257;
    const H: usize = 191;
    const N: usize = W * H;

    // Deterministic test pattern with a few sharp edges.
    let mut input = vec![0.0f32; N];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let u = (x as f32) / (W as f32 - 1.0);
            let v = (y as f32) / (H as f32 - 1.0);
            let edge = if (x + y).is_multiple_of(53) { 1.0 } else { 0.0 };
            input[i] = 0.4 * u + 0.3 * v + 0.2 * edge;
        }
    }

    let mut all_ok = true;

    // ---- gab_smooth (3x3) ----
    {
        let wc = 0.5_f32;
        let w1 = 0.1_f32;
        let w2 = 0.025_f32;

        let mut cpu_out = vec![0.0f32; N];
        gab_smooth_scalar(&mut cpu_out, &input, W, H, wc, w1, w2);

        let h_in = client.create_from_slice(f32::as_bytes(&input));
        let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        gab_smooth::<Backend>(
            &client,
            h_in,
            h_out.clone(),
            W as u32,
            H as u32,
            wc,
            w1,
            w2,
        );
        let bytes = client.read_one(h_out).expect("read gab");
        let gpu: &[f32] = f32::from_bytes(&bytes);
        let (m, p) = max_abs_diff(gpu, &cpu_out);
        let ok = m < 1e-5;
        println!(
            "gab_smooth (3x3, {W}x{H}):       max|Δ| = {m:.3e} at {p}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    // ---- gaborish_5x5 ----
    {
        // Realistic-ish weights normalized to ~1.0
        let wc = 1.04_f32;
        let wr = -0.005_f32;
        let wd = -0.0025_f32;
        let w_big_r = -0.0008_f32;
        let wl = -0.0005_f32;
        let w_big_d = -0.0001_f32;

        let mut cpu_out = vec![0.0f32; N];
        gaborish_5x5_scalar(
            &mut cpu_out,
            &input,
            W,
            H,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );

        let h_in = client.create_from_slice(f32::as_bytes(&input));
        let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        gaborish_5x5::<Backend>(
            &client,
            h_in,
            h_out.clone(),
            W as u32,
            H as u32,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );
        let bytes = client.read_one(h_out).expect("read gaborish");
        let gpu: &[f32] = f32::from_bytes(&bytes);
        let (m, p) = max_abs_diff(gpu, &cpu_out);
        let ok = m < 1e-5;
        println!(
            "gaborish_5x5 ({W}x{H}):           max|Δ| = {m:.3e} at {p}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    // ---- mask1x1 ----
    {
        // Use values in opsin (post-cbrt) range — small magnitudes around 0.
        let mut xyb_y = vec![0.0f32; N];
        for i in 0..N {
            xyb_y[i] = (input[i] - 0.5) * 0.4;
        }

        let mut cpu_out = vec![0.0f32; N];
        compute_mask1x1_scalar(&xyb_y, W, H, &mut cpu_out);

        let h_in = client.create_from_slice(f32::as_bytes(&xyb_y));
        let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
        mask1x1::<Backend>(&client, h_in, h_out.clone(), W as u32, H as u32);
        let bytes = client.read_one(h_out).expect("read mask1x1");
        let gpu: &[f32] = f32::from_bytes(&bytes);
        let (m, p) = max_abs_diff(gpu, &cpu_out);
        // mask1x1 chains a fast_log2f polynomial through a reciprocal.
        // CPU scalar `_scalar` path uses plain `+`/`*` (no `mul_add`), so it
        // may or may not FMA-contract; the GPU (CUDA default) DOES contract.
        // The resulting ulp drift accumulates through 5 multiplies + a
        // divide. At output values ~100 (small `diff`), absolute diff
        // ~7e-4 = ~7e-6 relative — well below the perceptual-masking
        // sensitivity threshold (mask1x1 drives quant-step decisions where
        // ~5% changes flip block strategy). Tolerance documented here:
        let ok = m < 1.5e-3;
        println!(
            "mask1x1 ({W}x{H}):                max|Δ| = {m:.3e} at {p}  {}",
            if ok { "✓" } else { "✗" }
        );
        all_ok &= ok;
    }

    if all_ok {
        println!("\n✓ Phase-1 parity OK (3 of 4 kernels — XYB has its own example).");
    } else {
        eprintln!("\n✗ Phase-1 parity FAILED — see lines above.");
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
