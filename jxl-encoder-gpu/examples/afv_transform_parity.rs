//! End-to-end parity test for the full AFV0-3 transform composition.
//!
//! Compares `forks::afv::afv_transform_gpu` (host composition of
//! AFV-DCT-4×4 + raw DCT-4×4 + raw DCT-4×8 + DC packing) against an
//! inline CPU reference using the same primitives — upstream's
//! `jxl_encoder::vardct::afv::afv_transform_from_pixels` is private.
//!
//! ```sh
//! cargo run --release --example afv_transform_parity --features cuda
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
fn cpu_afv_transform(
    basis_t: &[f32; 256],
    pixels: &[f32; 64],
    afv_kind: usize,
) -> [f32; 64] {
    use jxl_encoder::vardct::dct::{dct_4x4, dct_4x8};
    use jxl_encoder_gpu::forks::afv::{
        extract_afv_corner, extract_dct4_corner, extract_dct4x8_half, pack_afv_dcs,
    };

    let mut coeffs = [0.0_f32; 64];

    // Step 1: AFV 4×4 DCT (matmul against basis_t).
    let afv_corner = extract_afv_corner(pixels, afv_kind);
    let mut afv_coeffs = [0.0_f32; 16];
    for i in 0..16 {
        let p = afv_corner[i];
        for j in 0..16 {
            afv_coeffs[j] += basis_t[i * 16 + j] * p;
        }
    }
    for iy in 0..4 {
        for ix in 0..4 {
            coeffs[iy * 2 * 8 + ix * 2] = afv_coeffs[iy * 4 + ix];
        }
    }

    // Step 2: Raw 4×4 DCT.
    let dct4_corner = extract_dct4_corner(pixels, afv_kind);
    let mut dct4_coeffs = [0.0_f32; 16];
    dct_4x4(&dct4_corner, &mut dct4_coeffs);
    for iy in 0..4 {
        for ix in 0..4 {
            coeffs[iy * 2 * 8 + ix * 2 + 1] = dct4_coeffs[iy * 4 + ix];
        }
    }

    // Step 3: Raw 4×8 DCT.
    let dct4x8_half = extract_dct4x8_half(pixels, afv_kind);
    let mut dct4x8_coeffs = [0.0_f32; 32];
    dct_4x8(&dct4x8_half, &mut dct4x8_coeffs);
    for iy in 0..4 {
        for ix in 0..8 {
            coeffs[(1 + iy * 2) * 8 + ix] = dct4x8_coeffs[iy * 8 + ix];
        }
    }

    // Step 4: DC pack.
    pack_afv_dcs(&mut coeffs);
    coeffs
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::afv::afv_transform_gpu;
    use jxl_encoder_gpu::kernels::afv::AFV4X4_BASIS_TRANSPOSE;

    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    // Synthetic 8×8 test block.
    let pixels: [f32; 64] =
        core::array::from_fn(|i| 0.3 + 0.4 * ((i as f32) * 0.011).sin());

    let mut max_diff_overall = 0.0_f32;
    let mut all_ok = true;
    for afv_kind in 0..4 {
        let cpu_out = cpu_afv_transform(&AFV4X4_BASIS_TRANSPOSE, &pixels, afv_kind);
        let gpu_out = afv_transform_gpu(&enc, &AFV4X4_BASIS_TRANSPOSE, &pixels, afv_kind);

        let mut max_diff = 0.0_f32;
        let mut max_pos = 0usize;
        for i in 0..64 {
            let d = (cpu_out[i] - gpu_out[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_pos = i;
            }
        }
        let ok = max_diff < 1e-4; // basis matrix is hand-truncated; allow some drift
        let mark = if ok { "✓" } else { "✗" };
        println!(
            "afv_kind={afv_kind}: max|Δ| = {max_diff:.3e} at idx {max_pos} {mark}"
        );
        if !ok {
            all_ok = false;
            eprintln!(
                "  cpu[{max_pos}] = {:.6}, gpu[{max_pos}] = {:.6}",
                cpu_out[max_pos], gpu_out[max_pos]
            );
        }
        if max_diff > max_diff_overall {
            max_diff_overall = max_diff;
        }
    }

    println!("\noverall max|Δ| = {max_diff_overall:.3e}");
    if !all_ok {
        std::process::exit(1);
    }
}
