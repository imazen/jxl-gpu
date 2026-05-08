//! Phase 3 end-to-end on a real image with PROPER 3-channel cost grids.
//!
//! Like `phase3_real_image_demo` but uses
//! `compute_cost_grid_dct8_xyb` + `compute_cost_grid_dct16x16_xyb`
//! instead of the single-channel proxies. The cost grids see the
//! full XYB-weighted reconstruction error masked by per-pixel
//! perceptual sensitivity (mask1x1) — the same cost the VarDCT
//! encoder pays for in production.
//!
//! Compares the partition pick distribution between the proxy-based
//! and proper-cost runs to see how much the choice changes when the
//! cost model has the full discriminative power.

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
        CostGrids16x16, Partition16x16, compute_cost_grid_dct8_xyb, compute_cost_grid_dct8x16_xyb,
        compute_cost_grid_dct16x8_xyb, compute_cost_grid_dct16x16_xyb, select_partitions_16x16,
        select_partitions_16x16_full,
    };
    use jxl_encoder_gpu::quant_weights::{
        dct8_weights, dct16x8_weights, dct16x16_weights, replicate_weights,
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
    let w = (w / 16) * 16;
    let h = (h / 16) * 16;
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
    let (xx, xy, xb) = enc.xyb_from_linear_rgb(&r, &g, &b);
    let mask1x1 = compute_mask1x1_gpu(&enc, &xy, w as usize, h as usize);

    // Per-block-layout repackers.
    let xb8 = (w as usize) / 8;
    let yb8 = (h as usize) / 8;
    let nb8 = xb8 * yb8;
    let repack8 = |plane: &[f32]| {
        let mut out = vec![0.0f32; nb8 * 64];
        for by in 0..yb8 {
            for bx in 0..xb8 {
                for ly in 0..8 {
                    for lx in 0..8 {
                        out[(by * xb8 + bx) * 64 + ly * 8 + lx] =
                            plane[(by * 8 + ly) * (w as usize) + (bx * 8 + lx)];
                    }
                }
            }
        }
        out
    };
    let xb16 = xb8 / 2;
    let yb16 = yb8 / 2;
    let nb16 = xb16 * yb16;
    let repack16 = |plane: &[f32]| {
        let mut out = vec![0.0f32; nb16 * 256];
        for ry in 0..yb16 {
            for rx in 0..xb16 {
                for ly in 0..16 {
                    for lx in 0..16 {
                        out[(ry * xb16 + rx) * 256 + ly * 16 + lx] =
                            plane[(ry * 16 + ly) * (w as usize) + (rx * 16 + lx)];
                    }
                }
            }
        }
        out
    };
    let bx8 = repack8(&xx);
    let by8 = repack8(&xy);
    let bb8 = repack8(&xb);
    let bx16 = repack16(&xx);
    let by16 = repack16(&xy);
    let bb16 = repack16(&xb);

    // Real libjxl per-channel quant weights via quant_weights module.
    // (Cost grids are now backed by the actual upstream tables; mock
    // weights would zero all but DC under typical qac and produce
    // degenerate identical costs across strategies.)
    let dct8_all = dct8_weights();
    let wx8 = replicate_weights(&dct8_all[..64], nb8);
    let wy8 = replicate_weights(&dct8_all[64..128], nb8);
    let wb8 = replicate_weights(&dct8_all[128..192], nb8);
    let dct16_all = dct16x16_weights();
    let wx16 = replicate_weights(&dct16_all[..256], nb16);
    let wy16 = replicate_weights(&dct16_all[256..512], nb16);
    let wb16 = replicate_weights(&dct16_all[512..768], nb16);
    let qac8 = vec![1.7f32; nb8];
    let qac16 = vec![1.7f32; nb16];
    let thr_y = [0.56f32, 0.62, 0.62, 0.62];
    let thr_xb = [0.58f32, 0.62, 0.62, 0.62];

    let upload = |v: &[f32]| client.create_from_slice(f32::as_bytes(v));
    let h_thr_y_a = upload(&thr_y[..]);
    let h_thr_y_b = upload(&thr_y[..]);
    let h_thr_xb_a = upload(&thr_xb[..]);
    let h_thr_xb_b = upload(&thr_xb[..]);
    let h_thr_xb_c = upload(&thr_xb[..]);
    let h_thr_xb_d = upload(&thr_xb[..]);
    let h_mask = upload(&mask1x1);

    // ---- DCT8 3-channel cost grid ----
    let cg_dct8 = compute_cost_grid_dct8_xyb::<Backend>(
        &client,
        upload(&bx8),
        upload(&by8),
        upload(&bb8),
        upload(&wx8),
        upload(&wy8),
        upload(&wb8),
        upload(&qac8),
        upload(&qac8),
        upload(&qac8),
        h_thr_xb_a,
        h_thr_y_a,
        h_thr_xb_b,
        h_mask.clone(),
        xb8 as u32,
        yb8 as u32,
    );
    let cost_dct8: Vec<f32> = {
        let bytes = client.read_one(cg_dct8.costs).expect("dct8 xyb");
        f32::from_bytes(&bytes).to_vec()
    };

    // ---- DCT16x16 3-channel cost grid (aggregated to per-region) ----
    let cg_dct16 = compute_cost_grid_dct16x16_xyb::<Backend>(
        &client,
        upload(&bx16),
        upload(&by16),
        upload(&bb16),
        upload(&wx16),
        upload(&wy16),
        upload(&wb16),
        upload(&qac16),
        upload(&qac16),
        upload(&qac16),
        h_thr_xb_c,
        h_thr_y_b,
        h_thr_xb_d,
        h_mask,
        xb16 as u32,
        yb16 as u32,
    );
    let cost_dct16x16: Vec<f32> = {
        let bytes = client.read_one(cg_dct16.costs).expect("dct16 xyb");
        let raw: &[f32] = f32::from_bytes(&bytes);
        let xb_sub = xb16 * 2;
        let mut out = vec![0.0f32; nb16];
        for ry in 0..yb16 {
            for rx in 0..xb16 {
                let by = ry * 2;
                let bx = rx * 2;
                out[ry * xb16 + rx] = raw[by * xb_sub + bx]
                    + raw[by * xb_sub + bx + 1]
                    + raw[(by + 1) * xb_sub + bx]
                    + raw[(by + 1) * xb_sub + bx + 1];
            }
        }
        out
    };

    let partitions = select_partitions_16x16(&cost_dct8, &cost_dct16x16, xb8, yb8);
    let mut counts = [0_usize; 4];
    for &p in &partitions {
        counts[match p {
            Partition16x16::Dct16x16 => 0,
            Partition16x16::TwoDct16x8Horizontal => 1,
            Partition16x16::TwoDct8x16Vertical => 2,
            Partition16x16::FourDct8x8 => 3,
            Partition16x16::FourSubBlocks(_) => continue, // demo doesn't pass sub-block grids
        }] += 1;
    }

    let stats = |c: &[f32]| -> (f32, f32, f32, f32) {
        let n = c.len() as f32;
        let mean = c.iter().sum::<f32>() / n;
        let min = c.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = c.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let var = c.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        (min, mean, max, var.sqrt())
    };
    let (mn8, me8, mx8, sd8) = stats(&cost_dct8);
    let (mn16, me16, mx16, sd16) = stats(&cost_dct16x16);

    println!("=== phase3_xyb_real_image_demo ===");
    println!("Image: {image_path}");
    println!("Crop: {w}×{h} ({nb8} 8×8 blocks, {nb16} 16×16 regions)\n");
    println!(
        "DCT8 (3-ch XYB+mask):     min={mn8:.4}  mean={me8:.4}  max={mx8:.4}  std={sd8:.4}  (std/mean={:.3})",
        sd8 / me8.max(1e-9)
    );
    println!(
        "DCT16x16 (3-ch XYB+mask): min={mn16:.4}  mean={me16:.4}  max={mx16:.4}  std={sd16:.4}  (std/mean={:.3})",
        sd16 / me16.max(1e-9)
    );
    println!("\nPartition decisions over {} regions:", partitions.len());
    println!(
        "  DCT16×16          : {:>5}  ({:>5.1}%)",
        counts[0],
        100.0 * counts[0] as f32 / partitions.len() as f32
    );
    println!(
        "  Two DCT16×8 horiz : {:>5}  ({:>5.1}%)",
        counts[1],
        100.0 * counts[1] as f32 / partitions.len() as f32
    );
    println!(
        "  Two DCT8×16 vert  : {:>5}  ({:>5.1}%)",
        counts[2],
        100.0 * counts[2] as f32 / partitions.len() as f32
    );
    println!(
        "  Four DCT8×8       : {:>5}  ({:>5.1}%)",
        counts[3],
        100.0 * counts[3] as f32 / partitions.len() as f32
    );
    println!("\n(Compare with phase3_real_image_demo — single-channel proxy on the same image.)");

    // ---- Now add the rect 3-channel cost grids and re-pick. ----
    // DCT16×8 (16-tall × 8-wide rect): repack each XYB plane into
    // [rect_y * xb_rect + rect_x][16 rows × 8 cols].
    let xb_16x8 = (w as usize) / 8;
    let yb_16x8 = (h as usize) / 16;
    let nb_16x8 = xb_16x8 * yb_16x8;
    let repack_16x8 = |plane: &[f32]| {
        let mut out = vec![0.0f32; nb_16x8 * 128];
        for ry in 0..yb_16x8 {
            for rx in 0..xb_16x8 {
                for ly in 0..16 {
                    for lx in 0..8 {
                        out[(ry * xb_16x8 + rx) * 128 + ly * 8 + lx] =
                            plane[(ry * 16 + ly) * (w as usize) + (rx * 8 + lx)];
                    }
                }
            }
        }
        out
    };
    // DCT8×16 (8-tall × 16-wide rect): [rect_y * xb_rect + rect_x][8 rows × 16 cols].
    let xb_8x16 = (w as usize) / 16;
    let yb_8x16 = (h as usize) / 8;
    let nb_8x16 = xb_8x16 * yb_8x16;
    let repack_8x16 = |plane: &[f32]| {
        let mut out = vec![0.0f32; nb_8x16 * 128];
        for ry in 0..yb_8x16 {
            for rx in 0..xb_8x16 {
                for ly in 0..8 {
                    for lx in 0..16 {
                        out[(ry * xb_8x16 + rx) * 128 + ly * 16 + lx] =
                            plane[(ry * 8 + ly) * (w as usize) + (rx * 16 + lx)];
                    }
                }
            }
        }
        out
    };
    let bx_16x8 = repack_16x8(&xx);
    let by_16x8 = repack_16x8(&xy);
    let bb_16x8 = repack_16x8(&xb);
    let bx_8x16 = repack_8x16(&xx);
    let by_8x16 = repack_8x16(&xy);
    let bb_8x16 = repack_8x16(&xb);

    // Real DCT16x8 weights (8 rows × 16 cols = 128 floats per channel).
    let dct16x8_all = dct16x8_weights();
    let wx_rect = replicate_weights(&dct16x8_all[..128], nb_16x8);
    let wy_rect = replicate_weights(&dct16x8_all[128..256], nb_16x8);
    let wb_rect = replicate_weights(&dct16x8_all[256..384], nb_16x8);
    let qac_16x8 = vec![1.7f32; nb_16x8];
    let cg_16x8 = compute_cost_grid_dct16x8_xyb::<Backend>(
        &client,
        upload(&bx_16x8),
        upload(&by_16x8),
        upload(&bb_16x8),
        upload(&wx_rect),
        upload(&wy_rect),
        upload(&wb_rect),
        upload(&qac_16x8),
        upload(&qac_16x8),
        upload(&qac_16x8),
        upload(&thr_xb[..]),
        upload(&thr_y[..]),
        upload(&thr_xb[..]),
        upload(&mask1x1),
        xb_16x8 as u32,
        yb_16x8 as u32,
    );
    // block_l2 returns per-8×8 sub-cell costs; aggregate 2 per rect.
    let cost_dct16x8: Vec<f32> = {
        let bytes = client.read_one(cg_16x8.costs).expect("dct16x8");
        let raw: &[f32] = f32::from_bytes(&bytes);
        let mut out = vec![0.0f32; nb_16x8];
        for ry in 0..yb_16x8 {
            for rx in 0..xb_16x8 {
                out[ry * xb_16x8 + rx] =
                    raw[ry * (xb_16x8 * 2) + 2 * rx] + raw[ry * (xb_16x8 * 2) + 2 * rx + 1];
            }
        }
        out
    };

    // DCT8x16 shares the DCT16x8 weight table per upstream.
    let wx_rect2 = replicate_weights(&dct16x8_all[..128], nb_8x16);
    let wy_rect2 = replicate_weights(&dct16x8_all[128..256], nb_8x16);
    let wb_rect2 = replicate_weights(&dct16x8_all[256..384], nb_8x16);
    let qac_8x16 = vec![1.7f32; nb_8x16];
    let cg_8x16 = compute_cost_grid_dct8x16_xyb::<Backend>(
        &client,
        upload(&bx_8x16),
        upload(&by_8x16),
        upload(&bb_8x16),
        upload(&wx_rect2),
        upload(&wy_rect2),
        upload(&wb_rect2),
        upload(&qac_8x16),
        upload(&qac_8x16),
        upload(&qac_8x16),
        upload(&thr_xb[..]),
        upload(&thr_y[..]),
        upload(&thr_xb[..]),
        upload(&mask1x1),
        xb_8x16 as u32,
        yb_8x16 as u32,
    );
    let cost_dct8x16: Vec<f32> = {
        let bytes = client.read_one(cg_8x16.costs).expect("dct8x16");
        let raw: &[f32] = f32::from_bytes(&bytes);
        let mut out = vec![0.0f32; nb_8x16];
        for ry in 0..yb_8x16 {
            for rx in 0..xb_8x16 {
                out[ry * xb_8x16 + rx] =
                    raw[2 * ry * xb_8x16 + rx] + raw[(2 * ry + 1) * xb_8x16 + rx];
            }
        }
        out
    };

    let extra = CostGrids16x16 {
        dct_16x8: Some(&cost_dct16x8),
        dct_8x16: Some(&cost_dct8x16),
        ..Default::default()
    };
    let partitions_full = select_partitions_16x16_full(&cost_dct8, &cost_dct16x16, extra, xb8, yb8);
    let mut counts_full = [0_usize; 4];
    for &p in &partitions_full {
        counts_full[match p {
            Partition16x16::Dct16x16 => 0,
            Partition16x16::TwoDct16x8Horizontal => 1,
            Partition16x16::TwoDct8x16Vertical => 2,
            Partition16x16::FourDct8x8 => 3,
            Partition16x16::FourSubBlocks(_) => continue, // demo doesn't pass sub-block grids
        }] += 1;
    }

    let (mn1, me1, mx1, sd1) = stats(&cost_dct16x8);
    let (mn2, me2, mx2, sd2) = stats(&cost_dct8x16);
    println!("\nRect 3-channel cost grids:");
    println!(
        "  DCT16×8:  min={mn1:.4}  mean={me1:.4}  max={mx1:.4}  std={sd1:.4}  (std/mean={:.3})",
        sd1 / me1.max(1e-9)
    );
    println!(
        "  DCT8×16:  min={mn2:.4}  mean={me2:.4}  max={mx2:.4}  std={sd2:.4}  (std/mean={:.3})",
        sd2 / me2.max(1e-9)
    );
    println!(
        "\nFull 4-strategy 16×16 partition decisions over {} regions:",
        partitions_full.len()
    );
    let pct = |c: usize| 100.0 * c as f32 / partitions_full.len() as f32;
    println!(
        "  DCT16×16          : {:>5}  ({:>5.1}%)",
        counts_full[0],
        pct(counts_full[0])
    );
    println!(
        "  Two DCT16×8 horiz : {:>5}  ({:>5.1}%)",
        counts_full[1],
        pct(counts_full[1])
    );
    println!(
        "  Two DCT8×16 vert  : {:>5}  ({:>5.1}%)",
        counts_full[2],
        pct(counts_full[2])
    );
    println!(
        "  Four DCT8×8       : {:>5}  ({:>5.1}%)",
        counts_full[3],
        pct(counts_full[3])
    );
    println!(
        "\n2-strategy → 4-strategy delta:\n  DCT16×16: {} → {} ({:+})\n  4-DCT8×8: {} → {} ({:+})\n  +rect picks: {}",
        counts[0],
        counts_full[0],
        counts_full[0] as i64 - counts[0] as i64,
        counts[3],
        counts_full[3],
        counts_full[3] as i64 - counts[3] as i64,
        counts_full[1] + counts_full[2],
    );
}
