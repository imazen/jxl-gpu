//! Per-strategy cost comparison on a real image.
//!
//! Runs all six 8×8-tier cost grids (DCT8, DCT4×4, DCT4×8, DCT8×4,
//! IDENTITY, DCT2X2) on the same XYB Y-plane and reports cost
//! statistics + per-block "winner" counts. Tells you which strategies
//! would actually get picked if a finer selector existed.
//!
//! All six cost grids share the 64-coeff layout, so they consume the
//! same per-block input + weights + qac + thresholds. The Y plane is
//! repacked to per-8×8-block layout once, then fed to each grid.

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
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::pipeline::{
        compute_cost_grid_dct2x2_single_channel, compute_cost_grid_dct4x4_single_channel,
        compute_cost_grid_dct4x8_single_channel, compute_cost_grid_dct8_single_channel,
        compute_cost_grid_dct8x4_single_channel, compute_cost_grid_identity_single_channel,
    };

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let image_path = std::env::var("IMAGE_PATH").unwrap_or_else(|_| {
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
            .to_string()
    });
    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let w = (w / 8) * 8;
    let h = (h / 8) * 8;
    let pixels: Vec<u8> = img.into_raw();
    let n = (w as usize) * (h as usize);

    let to_linear = |c: u8| (c as f32 / 255.0).powf(2.4);
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = (y * (w as usize) + x) * 3;
            r.push(to_linear(pixels[i]));
            g.push(to_linear(pixels[i + 1]));
            b.push(to_linear(pixels[i + 2]));
        }
    }
    let (_xx, xy, _xb) = enc.xyb_from_linear_rgb(&r, &g, &b);

    let xb8 = (w as usize) / 8;
    let yb8 = (h as usize) / 8;
    let nb8 = xb8 * yb8;
    let mut y_blocks = vec![0.0f32; nb8 * 64];
    for by in 0..yb8 {
        for bx in 0..xb8 {
            for ly in 0..8 {
                for lx in 0..8 {
                    y_blocks[(by * xb8 + bx) * 64 + ly * 8 + lx] =
                        xy[(by * 8 + ly) * (w as usize) + (bx * 8 + lx)];
                }
            }
        }
    }
    let mut weights_per = vec![1.0f32; 64];
    for i in 0..64 {
        let r = (i / 8) as f32;
        let c = (i % 8) as f32;
        weights_per[i] = 1.0 + 0.7 * (r + c);
    }
    let mut weights = vec![0.0f32; nb8 * 64];
    for k in 0..nb8 {
        weights[k * 64..k * 64 + 64].copy_from_slice(&weights_per);
    }
    let qac_qm = vec![1.7f32; nb8];
    let thresholds = [0.62f32; 4];

    let upload_f = |v: &[f32]| client.create_from_slice(f32::as_bytes(v));

    let run = |name: &str, costs: Vec<f32>| -> (String, Vec<f32>) {
        (name.to_string(), costs)
    };

    let read = |handle, label: &str| -> Vec<f32> {
        let bytes = client.read_one(handle).expect(label);
        f32::from_bytes(&bytes).to_vec()
    };

    // Compute all six grids.
    let strategies: Vec<(String, Vec<f32>)> = vec![
        run("DCT8", read(
            compute_cost_grid_dct8_single_channel::<Backend>(
                &client,
                upload_f(&y_blocks),
                upload_f(&weights),
                upload_f(&qac_qm),
                upload_f(&thresholds[..]),
                xb8 as u32,
                yb8 as u32,
            )
            .costs,
            "dct8",
        )),
        run("DCT4×4", read(
            compute_cost_grid_dct4x4_single_channel::<Backend>(
                &client,
                upload_f(&y_blocks),
                upload_f(&weights),
                upload_f(&qac_qm),
                upload_f(&thresholds[..]),
                xb8 as u32,
                yb8 as u32,
            )
            .costs,
            "dct4x4",
        )),
        run("DCT4×8", read(
            compute_cost_grid_dct4x8_single_channel::<Backend>(
                &client,
                upload_f(&y_blocks),
                upload_f(&weights),
                upload_f(&qac_qm),
                upload_f(&thresholds[..]),
                xb8 as u32,
                yb8 as u32,
            )
            .costs,
            "dct4x8",
        )),
        run("DCT8×4", read(
            compute_cost_grid_dct8x4_single_channel::<Backend>(
                &client,
                upload_f(&y_blocks),
                upload_f(&weights),
                upload_f(&qac_qm),
                upload_f(&thresholds[..]),
                xb8 as u32,
                yb8 as u32,
            )
            .costs,
            "dct8x4",
        )),
        run("IDENTITY", read(
            compute_cost_grid_identity_single_channel::<Backend>(
                &client,
                upload_f(&y_blocks),
                upload_f(&weights),
                upload_f(&qac_qm),
                upload_f(&thresholds[..]),
                xb8 as u32,
                yb8 as u32,
            )
            .costs,
            "identity",
        )),
        run("DCT2X2", read(
            compute_cost_grid_dct2x2_single_channel::<Backend>(
                &client,
                upload_f(&y_blocks),
                upload_f(&weights),
                upload_f(&qac_qm),
                upload_f(&thresholds[..]),
                xb8 as u32,
                yb8 as u32,
            )
            .costs,
            "dct2x2",
        )),
    ];

    println!("=== strategy_cost_comparison_demo ===");
    println!("Image: {image_path}");
    println!("Crop: {w}×{h} ({nb8} 8×8 blocks)\n");

    let stats = |c: &[f32]| -> (f32, f32, f32) {
        let n = c.len() as f32;
        let mean = c.iter().sum::<f32>() / n;
        let min = c.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = c.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        (min, mean, max)
    };
    println!(
        "{:<10}  {:>12}  {:>12}  {:>12}",
        "strategy", "min", "mean", "max"
    );
    for (name, costs) in &strategies {
        let (mn, me, mx) = stats(costs);
        println!("{name:<10}  {mn:>12.4}  {me:>12.4}  {mx:>12.4}");
    }

    // Per-block "winner" tally.
    let mut wins = vec![0_usize; strategies.len()];
    for i in 0..nb8 {
        let mut best = 0;
        let mut best_cost = strategies[0].1[i];
        for (j, (_, costs)) in strategies.iter().enumerate().skip(1) {
            if costs[i] < best_cost {
                best_cost = costs[i];
                best = j;
            }
        }
        wins[best] += 1;
    }
    println!("\nPer-block winner (lowest cost) over {nb8} 8×8 blocks:");
    for (i, (name, _)) in strategies.iter().enumerate() {
        let pct = 100.0 * wins[i] as f32 / nb8 as f32;
        println!("  {name:<10}: {:>6}  ({:>5.1}%)", wins[i], pct);
    }
    println!(
        "\nNotes:\n- Per-strategy lowest-cost on EACH 8×8 cell — assumes the selector\n  can freely choose any 8×8-tier strategy per cell, which the current\n  Partition16x16::FourDct8x8 doesn't yet support (it forces all 4\n  sub-cells to DCT8). Real-world value depends on extending the\n  selector to per-cell strategy choice.\n- Identical costs across all six strategies on this run indicate the\n  quant params (synthetic weights `1 + 0.7*(r+c)` + qac=1.7) zeroed\n  all but DC coefficients, so every strategy reconstructs to the\n  same constant. Real DCT8 quant matrices (much heavier on high\n  freqs, lighter on low freqs) preserve more coefficients and\n  differentiate the strategies. Plug those in via `compute_cost_grid_*`\n  to see meaningful per-strategy variation."
    );
}
