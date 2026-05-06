//! Phase 4 proof-of-life: GpuEncoder dependency works.
//!
//! Encodes a tiny synthetic 32x32 image via the (currently CPU-delegated)
//! `GpuEncoder::encode_lossy_via_cpu` and verifies non-empty output bytes.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder::api::{LossyConfig, PixelLayout};
    use jxl_encoder_gpu::encoder::GpuEncoder;

    type Backend = cubecl::cuda::CudaRuntime;

    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    // 32x32 RGB8 input
    const W: u32 = 32;
    const H: u32 = 32;
    let mut pixels = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            pixels.push(((x * 8) & 0xFF) as u8);
            pixels.push(((y * 8) & 0xFF) as u8);
            pixels.push((((x + y) * 4) & 0xFF) as u8);
        }
    }

    let config = LossyConfig::new(1.0).with_effort(7);

    let bytes = enc
        .encode_lossy_via_cpu(&config, &pixels, W, H, PixelLayout::Rgb8)
        .expect("encode failed");

    println!(
        "Phase 4 smoke test: encoded {}x{} RGB8 → {} JXL bytes",
        W,
        H,
        bytes.len()
    );

    // Sanity: JXL files start with the bare codestream marker (0xFF 0x0A)
    // OR a JXL container box (sig 0x0000000C 'JXL ').
    let starts_ok = bytes.starts_with(&[0xFF, 0x0A])
        || bytes.starts_with(&[0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ']);
    assert!(starts_ok, "output doesn't start with JXL signature");
    assert!(bytes.len() > 16, "output suspiciously small");

    println!("✓ Phase 4 dependency on jxl-encoder works end-to-end.");
    println!("  (GPU kernels not yet wired into the bitstream — but next test runs GPU XYB:)\n");

    // Standalone GPU XYB demo on linear-RGB input.
    let n = (W * H) as usize;
    let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
    let g: Vec<f32> = (0..n).map(|i| 0.5 - 0.4 * (i as f32 / n as f32)).collect();
    let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * ((i % 17) as f32 / 17.0)).collect();
    let (x, y, b_out) = enc.xyb_from_linear_rgb(&r, &g, &b);
    println!(
        "GPU XYB on {} pixels: X[0]={:.4}, Y[0]={:.4}, B[0]={:.4}",
        n, x[0], y[0], b_out[0]
    );
    assert_eq!(x.len(), n);
    assert_eq!(y.len(), n);
    assert_eq!(b_out.len(), n);
    assert!(x.iter().all(|v| v.is_finite()));
    assert!(y.iter().all(|v| v.is_finite()));
    assert!(b_out.iter().all(|v| v.is_finite()));
    println!("✓ GpuEncoder::xyb_from_linear_rgb produces finite XYB.");

    // mask1x1 on the Y channel (same shape as input).
    let mask = enc.mask1x1_field(&y, W, H);
    assert_eq!(mask.len(), n);
    assert!(mask.iter().all(|v| v.is_finite() && *v > 0.0));
    println!("✓ GpuEncoder::mask1x1_field produced {} positive finite values.", mask.len());

    // DCT8 on a 4-block batch of synthetic 8x8 inputs.
    let blocks: Vec<f32> = (0..(4 * 64))
        .map(|i| 0.5 + 0.1 * ((i as f32 * 0.31).sin()))
        .collect();
    let dct = enc.dct_8x8_blocks(&blocks);
    assert_eq!(dct.len(), 4 * 64);
    assert!(dct.iter().all(|v| v.is_finite()));
    println!("✓ GpuEncoder::dct_8x8_blocks produced {} finite coefficients.", dct.len());

    // Equivalence test: GPU XYB vs jxl-encoder-simd CPU XYB on 256x256 random.
    {
        const N2: usize = 256 * 256;
        // Deterministic pseudo-random linear-RGB
        let mut r2 = vec![0.0f32; N2];
        let mut g2 = vec![0.0f32; N2];
        let mut b2 = vec![0.0f32; N2];
        for i in 0..N2 {
            let u = ((i.wrapping_mul(2654435761)) & 0xFFFF) as f32 / 65535.0;
            let v = ((i.wrapping_mul(0x9E3779B9)) & 0xFFFF) as f32 / 65535.0;
            let w = ((i.wrapping_mul(0xBF58476D)) & 0xFFFF) as f32 / 65535.0;
            r2[i] = u;
            g2[i] = v;
            b2[i] = w;
        }
        let (gpu_x, gpu_y, gpu_b) = enc.xyb_from_linear_rgb(&r2, &g2, &b2);
        let mut cpu_x = vec![0.0f32; N2];
        let mut cpu_y = vec![0.0f32; N2];
        let mut cpu_b = vec![0.0f32; N2];
        jxl_encoder_simd::forward_xyb_scalar(&r2, &g2, &b2, &mut cpu_x, &mut cpu_y, &mut cpu_b, N2);

        let max_x = gpu_x.iter().zip(&cpu_x).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_y = gpu_y.iter().zip(&cpu_y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_b = gpu_b.iter().zip(&cpu_b).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!(
            "GPU vs CPU XYB on 256x256 random: max|Δ| X={max_x:.3e}, Y={max_y:.3e}, B={max_b:.3e}"
        );
        assert!(max_x < 5e-6, "X channel diverges: {max_x:.3e}");
        assert!(max_y < 5e-6, "Y channel diverges: {max_y:.3e}");
        assert!(max_b < 5e-6, "B channel diverges: {max_b:.3e}");
        println!("✓ GpuEncoder::xyb_from_linear_rgb matches jxl-encoder-simd CPU within 5e-6.");
    }
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
