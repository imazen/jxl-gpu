//! 3-channel DCT8 cost grid on a real CLIC2025 image.
//!
//! Mirrors `phase3_real_image_demo` but uses
//! `compute_cost_grid_dct8_xyb` — pays the full XYB-weighted +
//! mask1x1-modulated reconstruction cost, instead of the
//! single-channel proxy.
//!
//! Reports cost statistics (min/mean/max/std) for the proper 3-channel
//! grid so they can be compared against the single-channel proxy.

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
    use jxl_encoder_gpu::forks::adaptive_quant::compute_mask1x1_gpu;
    use jxl_encoder_gpu::pipeline::{
        compute_cost_grid_dct8_single_channel, compute_cost_grid_dct8_xyb,
        compute_cost_grid_dct32x32_xyb,
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

    let to_linear = |c: u8| (c as f32 / 255.0).powf(2.4);
    let n = (w as usize) * (h as usize);
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
    let (xx, xy, xb) = enc.xyb_from_linear_rgb(&r, &g, &b);

    // Per-pixel mask1x1 (length = w*h).
    let mask1x1 = compute_mask1x1_gpu(&enc, &xy, w as usize, h as usize);

    let xb_blocks = (w as usize) / 8;
    let yb_blocks = (h as usize) / 8;
    let nb = xb_blocks * yb_blocks;

    // Repack each XYB plane into per-block layout.
    let repack = |plane: &[f32]| {
        let mut out = vec![0.0f32; nb * 64];
        for by in 0..yb_blocks {
            for bx in 0..xb_blocks {
                for ly in 0..8 {
                    for lx in 0..8 {
                        out[(by * xb_blocks + bx) * 64 + ly * 8 + lx] =
                            plane[(by * 8 + ly) * (w as usize) + (bx * 8 + lx)];
                    }
                }
            }
        }
        out
    };
    let bx = repack(&xx);
    let by = repack(&xy);
    let bb = repack(&xb);

    // Per-channel weights (shaped like a DCT8 quant matrix; fake values).
    let mut wts = vec![1.0f32; 64];
    for i in 0..64 {
        let r = (i / 8) as f32;
        let c = (i % 8) as f32;
        wts[i] = 1.0 + 0.7 * (r + c);
    }
    // Three different magnitude scales so each channel has slightly
    // different quant; mimics actual XYB matrix differences.
    let wx_per: Vec<f32> = wts.iter().map(|w| w * 0.6).collect();
    let wy_per = wts.clone();
    let wb_per: Vec<f32> = wts.iter().map(|w| w * 1.3).collect();

    let replicate = |per: &[f32]| {
        let mut out = vec![0.0f32; nb * per.len()];
        for k in 0..nb {
            out[k * per.len()..(k + 1) * per.len()].copy_from_slice(per);
        }
        out
    };
    let wx = replicate(&wx_per);
    let wy = replicate(&wy_per);
    let wb = replicate(&wb_per);
    let qac = vec![1.7f32; nb];
    let thr_y = [0.56_f32, 0.62, 0.62, 0.62];
    let thr_xb = [0.58_f32, 0.62, 0.62, 0.62];

    // Upload once, share across both grid runs.
    let h_bx = client.create_from_slice(f32::as_bytes(&bx));
    let h_by = client.create_from_slice(f32::as_bytes(&by));
    let h_bb = client.create_from_slice(f32::as_bytes(&bb));
    let h_wx = client.create_from_slice(f32::as_bytes(&wx));
    let h_wy = client.create_from_slice(f32::as_bytes(&wy));
    let h_wb = client.create_from_slice(f32::as_bytes(&wb));
    let h_qac_x = client.create_from_slice(f32::as_bytes(&qac));
    let h_qac_y = client.create_from_slice(f32::as_bytes(&qac));
    let h_qac_b = client.create_from_slice(f32::as_bytes(&qac));
    let h_thr_y = client.create_from_slice(f32::as_bytes(&thr_y[..]));
    let h_thr_xb_a = client.create_from_slice(f32::as_bytes(&thr_xb[..]));
    let h_thr_xb_b = client.create_from_slice(f32::as_bytes(&thr_xb[..]));
    let h_mask = client.create_from_slice(f32::as_bytes(&mask1x1));

    // Single-channel proxy on Y for comparison.
    let cg_proxy = compute_cost_grid_dct8_single_channel::<Backend>(
        &client,
        h_by.clone(),
        h_wy.clone(),
        client.create_from_slice(f32::as_bytes(&qac)),
        client.create_from_slice(f32::as_bytes(&thr_y[..])),
        xb_blocks as u32,
        yb_blocks as u32,
    );
    let costs_proxy: Vec<f32> = {
        let bytes = client.read_one(cg_proxy.costs).expect("proxy");
        f32::from_bytes(&bytes).to_vec()
    };

    // Proper 3-channel cost.
    let cg = compute_cost_grid_dct8_xyb::<Backend>(
        &client, h_bx, h_by, h_bb, h_wx, h_wy, h_wb, h_qac_x, h_qac_y, h_qac_b, h_thr_xb_a,
        h_thr_y, h_thr_xb_b, h_mask, xb_blocks as u32, yb_blocks as u32,
    );
    let costs_xyb: Vec<f32> = {
        let bytes = client.read_one(cg.costs).expect("xyb");
        f32::from_bytes(&bytes).to_vec()
    };

    let stats = |c: &[f32]| -> (f32, f32, f32, f32) {
        let n = c.len() as f32;
        let mean = c.iter().sum::<f32>() / n;
        let min = c.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = c.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let var = c.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        (min, mean, max, var.sqrt())
    };

    println!("=== cost_grid_3channel_demo ===");
    println!("Image: {image_path}");
    println!("Crop: {w}×{h} ({nb} 8×8 blocks)\n");
    let (mn1, me1, mx1, sd1) = stats(&costs_proxy);
    let (mn3, me3, mx3, sd3) = stats(&costs_xyb);
    println!(
        "Single-channel (Y proxy):  min={mn1:.4}  mean={me1:.4}  max={mx1:.4}  std={sd1:.4}  (std/mean={:.3})",
        sd1 / me1.max(1e-9)
    );
    println!(
        "3-channel (XYB + mask):    min={mn3:.4}  mean={me3:.4}  max={mx3:.4}  std={sd3:.4}  (std/mean={:.3})",
        sd3 / me3.max(1e-9)
    );

    // Sanity: both should produce non-degenerate distributions on real content.
    if sd1 / me1.max(1e-9) < 0.05 || sd3 / me3.max(1e-9) < 0.05 {
        eprintln!("\n✗ Cost grid distribution unexpectedly flat — check input?");
        std::process::exit(1);
    }
    println!("\n✓ DCT8 cost grids (proxy + 3-channel) produced varying costs on real content.");

    // ---- DCT32x32 3-channel (per-region) ----
    // Repack each XYB plane into 32x32 region layout (1024 floats each).
    let xb32 = xb_blocks / 4;
    let yb32 = yb_blocks / 4;
    let nb32 = xb32 * yb32;
    let repack32 = |plane: &[f32]| {
        let mut out = vec![0.0f32; nb32 * 1024];
        for ry in 0..yb32 {
            for rx in 0..xb32 {
                for ly in 0..32 {
                    for lx in 0..32 {
                        out[(ry * xb32 + rx) * 1024 + ly * 32 + lx] =
                            plane[(ry * 32 + ly) * (w as usize) + (rx * 32 + lx)];
                    }
                }
            }
        }
        out
    };
    let bx32 = repack32(&xx);
    let by32 = repack32(&xy);
    let bb32 = repack32(&xb);
    let mut wts32 = vec![1.0f32; 1024];
    for i in 0..1024 {
        wts32[i] = 1.0 + 0.5 * (i as f32 / 1024.0);
    }
    let wx32: Vec<f32> = wts32.iter().map(|w| w * 0.6).collect();
    let wy32 = wts32.clone();
    let wb32: Vec<f32> = wts32.iter().map(|w| w * 1.3).collect();
    let replicate32 = |per: &[f32]| {
        let mut out = vec![0.0f32; nb32 * per.len()];
        for k in 0..nb32 {
            out[k * per.len()..(k + 1) * per.len()].copy_from_slice(per);
        }
        out
    };
    let qac32 = vec![1.7f32; nb32];
    let cg32 = compute_cost_grid_dct32x32_xyb::<Backend>(
        &client,
        client.create_from_slice(f32::as_bytes(&bx32)),
        client.create_from_slice(f32::as_bytes(&by32)),
        client.create_from_slice(f32::as_bytes(&bb32)),
        client.create_from_slice(f32::as_bytes(&replicate32(&wx32))),
        client.create_from_slice(f32::as_bytes(&replicate32(&wy32))),
        client.create_from_slice(f32::as_bytes(&replicate32(&wb32))),
        client.create_from_slice(f32::as_bytes(&qac32)),
        client.create_from_slice(f32::as_bytes(&qac32)),
        client.create_from_slice(f32::as_bytes(&qac32)),
        client.create_from_slice(f32::as_bytes(&thr_xb[..])),
        client.create_from_slice(f32::as_bytes(&thr_y[..])),
        client.create_from_slice(f32::as_bytes(&thr_xb[..])),
        client.create_from_slice(f32::as_bytes(&mask1x1)),
        xb32 as u32,
        yb32 as u32,
    );
    let costs32_raw: Vec<f32> = {
        let bytes = client.read_one(cg32.costs).expect("dct32 xyb");
        f32::from_bytes(&bytes).to_vec()
    };
    // Aggregate 16 sub-cells per 32x32 region.
    let xb_sub = xb32 * 4;
    let mut costs32 = vec![0.0f32; nb32];
    for ry in 0..yb32 {
        for rx in 0..xb32 {
            let by = ry * 4;
            let bx = rx * 4;
            let mut s = 0.0_f32;
            for dy in 0..4 {
                for dx in 0..4 {
                    s += costs32_raw[(by + dy) * xb_sub + bx + dx];
                }
            }
            costs32[ry * xb32 + rx] = s;
        }
    }
    let (mn, me, mx, sd) = stats(&costs32);
    println!(
        "\nDCT32x32 (3-ch XYB+mask):  min={mn:.4}  mean={me:.4}  max={mx:.4}  std={sd:.4}  (std/mean={:.3})",
        sd / me.max(1e-9)
    );
    if sd / me.max(1e-9) < 0.05 {
        eprintln!("✗ DCT32x32 cost grid distribution unexpectedly flat");
        std::process::exit(1);
    }
    println!("✓ DCT32x32 3-channel cost grid produced varying costs on real content.");
}
