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
    println!("  (GPU kernels not yet wired in — that's the next series of replacements.)");
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
