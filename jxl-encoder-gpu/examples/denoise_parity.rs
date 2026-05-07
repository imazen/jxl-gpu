//! Parity test: GPU `denoise` vs `jxl_encoder_simd::denoise_channel_scalar`.
//!
//! ```sh
//! cargo run --release --example denoise_parity --features cuda
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
    use jxl_encoder_gpu::launch::denoise::denoise;
    use jxl_encoder_simd::denoise_channel_scalar;

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // Non-square non-power-of-2 to catch boundary bugs.
    const W: usize = 257;
    const H: usize = 191;
    const N: usize = W * H;

    // Deterministic XYB-like input — small values, mix of smooth + edges.
    let mut orig = vec![0.0f32; N];
    let mut y_chan = vec![0.0f32; N];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let u = (x as f32) / (W as f32 - 1.0);
            let v = (y as f32) / (H as f32 - 1.0);
            let edge = if (x + y).is_multiple_of(43) { 0.4 } else { 0.0 };
            // X channel: signed
            orig[i] = (u - 0.5) * 0.6 + edge - 0.2 * v;
            // Y intensity: positive, in [0, 1] range
            y_chan[i] = 0.05 + 0.7 * u + 0.2 * v;
        }
    }

    // Realistic noise LUT (8 points), chosen so most pixels exceed EPS.
    let noise_lut: [f32; 8] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.20, 0.10];
    let denoise_scale = 0.25_f32 / (1.0_f32 * 1.4_f32); // canonical d=1.0 quality_coef

    let mut cpu_out = orig.clone();
    denoise_channel_scalar(
        &mut cpu_out,
        &orig,
        &y_chan,
        W,
        H,
        &noise_lut,
        denoise_scale,
    );

    let h_orig = client.create_from_slice(f32::as_bytes(&orig));
    let h_y = client.create_from_slice(f32::as_bytes(&y_chan));
    let h_lut = client.create_from_slice(f32::as_bytes(&noise_lut));
    let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    denoise::<Backend>(
        &client,
        h_orig,
        h_y,
        h_lut,
        h_out.clone(),
        W as u32,
        H as u32,
        denoise_scale,
    );
    let bytes = client.read_one(h_out).expect("read denoise");
    let gpu: &[f32] = f32::from_bytes(&bytes);

    let (m, p) = max_abs_diff(gpu, &cpu_out);
    // The Wiener weight is a ratio of variances, so values can range ~[0, 1]
    // with FMA-contraction differences accumulating through the 5x5 sum.
    // Tolerance: 1e-5 covers the tightest reasonable contraction noise.
    let ok = m < 1e-5;
    println!(
        "denoise ({W}x{H}):                max|Δ| = {m:.3e} at idx {p}  {}",
        if ok { "✓" } else { "✗" }
    );

    if !ok {
        eprintln!(
            "\n  cpu[{p}] = {:.6}\n  gpu[{p}] = {:.6}\n  orig[{p}] = {:.6}\n  y[{p}] = {:.6}",
            cpu_out[p], gpu[p], orig[p], y_chan[p],
        );
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
