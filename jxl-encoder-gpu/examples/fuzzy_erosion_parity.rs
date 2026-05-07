//! Parity test: GPU fuzzy_erosion vs an inline CPU reference matching
//! `jxl_encoder::vardct::adaptive_quant::fuzzy_erosion` (which is
//! crate-private upstream, so we re-implement it here).
//!
//! ```sh
//! cargo run --release --example fuzzy_erosion_parity --features cuda
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
#[allow(clippy::too_many_arguments)]
fn cpu_fuzzy_erosion(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    from_x0: usize,
    from_y0: usize,
    region_w: usize,
    region_h: usize,
    k_mul: [f32; 4],
) -> Vec<f32> {
    let out_w = region_w / 2;
    let out_h = region_h / 2;
    let mut out = vec![0.0_f32; out_w * out_h];

    let store_min4 = |v: f32, m: &mut [f32; 4]| {
        if v < m[3] {
            if v < m[0] {
                m[3] = m[2];
                m[2] = m[1];
                m[1] = m[0];
                m[0] = v;
            } else if v < m[1] {
                m[3] = m[2];
                m[2] = m[1];
                m[1] = v;
            } else if v < m[2] {
                m[3] = m[2];
                m[2] = v;
            } else {
                m[3] = v;
            }
        }
    };

    for fy in 0..region_h {
        let y = fy + from_y0;
        let ym1 = if y >= 1 { y - 1 } else { y };
        let yp1 = if y + 1 < src_h { y + 1 } else { y };
        for fx in 0..region_w {
            let x = fx + from_x0;
            let xm1 = if x >= 1 { x - 1 } else { x };
            let xp1 = if x + 1 < src_w { x + 1 } else { x };
            let center = src[y * src_w + x];
            let left = src[y * src_w + xm1];
            let right = src[y * src_w + xp1];
            let tl = src[ym1 * src_w + xm1];
            let top = src[ym1 * src_w + x];
            let tr = src[ym1 * src_w + xp1];
            let bl = src[yp1 * src_w + xm1];
            let bot = src[yp1 * src_w + x];
            let br = src[yp1 * src_w + xp1];

            let mut m = [center, left, right, tl];
            // Sort first 4
            for _ in 0..4 {
                for j in 0..3 {
                    if m[j] > m[j + 1] {
                        m.swap(j, j + 1);
                    }
                }
            }
            store_min4(top, &mut m);
            store_min4(tr, &mut m);
            store_min4(bl, &mut m);
            store_min4(bot, &mut m);
            store_min4(br, &mut m);

            let v = k_mul[0] * m[0] + k_mul[1] * m[1] + k_mul[2] * m[2] + k_mul[3] * m[3];
            let ox = fx / 2;
            let oy = fy / 2;
            if fx % 2 == 0 && fy % 2 == 0 {
                out[oy * out_w + ox] = v;
            } else {
                out[oy * out_w + ox] += v;
            }
        }
    }
    out
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]
fn main() {
    use cubecl::prelude::*;
    use jxl_encoder_gpu::launch::fuzzy_erosion::{fuzzy_erosion, fuzzy_erosion_kmul};

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // Source plane: non-square, non-power-of-2 to catch boundary bugs.
    const SRC_W: usize = 73;
    const SRC_H: usize = 51;
    const FROM_X0: usize = 4;
    const FROM_Y0: usize = 6;
    const REGION_W: usize = 64;
    const REGION_H: usize = 40;
    const OUT_W: usize = REGION_W / 2;
    const OUT_H: usize = REGION_H / 2;

    let src: Vec<f32> = (0..SRC_W * SRC_H)
        .map(|i| 0.3 + 0.4 * (i as f32 * 0.0173).sin())
        .collect();

    let k_mul = fuzzy_erosion_kmul(1.0);
    println!(
        "k_mul (butteraugli=1.0) = [{:.6}, {:.6}, {:.6}, {:.6}]",
        k_mul[0], k_mul[1], k_mul[2], k_mul[3]
    );

    let cpu_out = cpu_fuzzy_erosion(&src, SRC_W, SRC_H, FROM_X0, FROM_Y0, REGION_W, REGION_H, k_mul);
    assert_eq!(cpu_out.len(), OUT_W * OUT_H);

    let h_src = client.create_from_slice(f32::as_bytes(&src));
    let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; OUT_W * OUT_H]));
    fuzzy_erosion::<Backend>(
        &client,
        h_src,
        h_out.clone(),
        SRC_W as u32,
        SRC_H as u32,
        FROM_X0 as u32,
        FROM_Y0 as u32,
        OUT_W as u32,
        OUT_H as u32,
        k_mul,
    );
    let bytes = client.read_one(h_out).expect("read fuzzy_erosion");
    let gpu_out: &[f32] = f32::from_bytes(&bytes);

    let mut max_diff = 0.0_f32;
    let mut max_pos = 0usize;
    for (i, (&c, &g)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
        let d = (c - g).abs();
        if d > max_diff {
            max_diff = d;
            max_pos = i;
        }
    }
    let ok = max_diff < 1e-6;
    println!(
        "fuzzy_erosion ({}×{} src → {}×{} out): max|Δ| = {:.3e} at idx {}  {}",
        SRC_W,
        SRC_H,
        OUT_W,
        OUT_H,
        max_diff,
        max_pos,
        if ok { "✓" } else { "✗" }
    );
    if !ok {
        eprintln!("  cpu[{max_pos}]={:.6}  gpu[{max_pos}]={:.6}", cpu_out[max_pos], gpu_out[max_pos]);
        std::process::exit(1);
    }
}
