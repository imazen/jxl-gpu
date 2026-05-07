//! Corpus-level validation of rect strategy picks across CLIC2025.
//!
//! Walks a directory of PNGs and runs the full 4-strategy 16×16
//! partition selector (DCT16×16 / 2×DCT16×8 / 2×DCT8×16 / 4×DCT8×8)
//! on each, with proper 3-channel XYB+mask cost grids. Reports per-
//! image pick distribution + corpus-aggregate percentages.
//!
//! Confirms whether the "rect strategies win ~60% of picks" finding
//! from `phase3_xyb_real_image_demo` generalizes across content.
//!
//! Usage:
//!   cargo run --release --features cuda --example corpus_rect_picks_demo
//!
//! Optional env vars:
//!   CORPUS_DIR   Directory of PNGs (default: codec-corpus/clic2025-1024)
//!   MAX_IMAGES   Cap on images to process (default: 8)

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
        CostGrids16x16, Partition16x16, compute_cost_grid_dct8_xyb,
        compute_cost_grid_dct8x16_xyb, compute_cost_grid_dct16x8_xyb,
        compute_cost_grid_dct16x16_xyb, select_partitions_16x16_full,
    };

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let corpus_dir = std::env::var("CORPUS_DIR")
        .unwrap_or_else(|_| "/home/lilith/work/codec-corpus/clic2025-1024".to_string());
    let max_images: usize = std::env::var("MAX_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&corpus_dir)
        .unwrap_or_else(|e| panic!("read_dir {corpus_dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
        .collect();
    paths.sort();
    paths.truncate(max_images);

    println!("=== corpus_rect_picks_demo ===");
    println!("Corpus: {corpus_dir}");
    println!("Images: {}\n", paths.len());

    println!(
        "{:<20}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}",
        "image", "16x16%", "16x8%", "8x16%", "8x8%", "rect%"
    );

    let mut total = [0_usize; 4];
    let mut total_regions = 0_usize;

    let upload = |c: &ComputeClient<Backend>, v: &[f32]| {
        c.create_from_slice(f32::as_bytes(v))
    };

    for path in &paths {
        let img = match image::open(path) {
            Ok(i) => i.to_rgb8(),
            Err(_) => continue,
        };
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

        let xb8 = (w as usize) / 8;
        let yb8 = (h as usize) / 8;
        let nb8 = xb8 * yb8;
        let xb16 = xb8 / 2;
        let yb16 = yb8 / 2;
        let nb16 = xb16 * yb16;
        let xb_16x8 = xb8;
        let yb_16x8 = yb8 / 2;
        let nb_16x8 = xb_16x8 * yb_16x8;
        let xb_8x16 = xb8 / 2;
        let yb_8x16 = yb8;
        let nb_8x16 = xb_8x16 * yb_8x16;

        let repack = |plane: &[f32], blk_w: usize, blk_h: usize, xb: usize, yb: usize| {
            let n_per_block = blk_w * blk_h;
            let mut out = vec![0.0f32; xb * yb * n_per_block];
            for ry in 0..yb {
                for rx in 0..xb {
                    for ly in 0..blk_h {
                        for lx in 0..blk_w {
                            out[(ry * xb + rx) * n_per_block + ly * blk_w + lx] =
                                plane[(ry * blk_h + ly) * (w as usize) + (rx * blk_w + lx)];
                        }
                    }
                }
            }
            out
        };

        // Real libjxl quant weights for all four strategies.
        let dct8_all = jxl_encoder_gpu::quant_weights::dct8_weights();
        let dct16_all = jxl_encoder_gpu::quant_weights::dct16x16_weights();
        let dct16x8_all = jxl_encoder_gpu::quant_weights::dct16x8_weights();
        let replicate = |per: &[f32], k: usize| {
            let mut out = vec![0.0f32; k * per.len()];
            for j in 0..k {
                out[j * per.len()..(j + 1) * per.len()].copy_from_slice(per);
            }
            out
        };
        let qac8 = vec![1.7f32; nb8];
        let qac16 = vec![1.7f32; nb16];
        let qac_16x8 = vec![1.7f32; nb_16x8];
        let qac_8x16 = vec![1.7f32; nb_8x16];
        let thr_y = [0.56_f32, 0.62, 0.62, 0.62];
        let thr_xb = [0.58_f32, 0.62, 0.62, 0.62];

        let h_mask = upload(&client, &mask1x1);

        // DCT8 (3-ch)
        let cg8 = compute_cost_grid_dct8_xyb::<Backend>(
            &client,
            upload(&client, &repack(&xx, 8, 8, xb8, yb8)),
            upload(&client, &repack(&xy, 8, 8, xb8, yb8)),
            upload(&client, &repack(&xb, 8, 8, xb8, yb8)),
            upload(&client, &replicate(&dct8_all[..64], nb8)),
            upload(&client, &replicate(&dct8_all[64..128], nb8)),
            upload(&client, &replicate(&dct8_all[128..192], nb8)),
            upload(&client, &qac8),
            upload(&client, &qac8),
            upload(&client, &qac8),
            upload(&client, &thr_xb[..]),
            upload(&client, &thr_y[..]),
            upload(&client, &thr_xb[..]),
            h_mask.clone(),
            xb8 as u32,
            yb8 as u32,
        );
        let cost_dct8: Vec<f32> = {
            let bytes = client.read_one(cg8.costs).expect("dct8");
            f32::from_bytes(&bytes).to_vec()
        };

        // DCT16x16 (3-ch, aggregated to per-region)
        let cg16 = compute_cost_grid_dct16x16_xyb::<Backend>(
            &client,
            upload(&client, &repack(&xx, 16, 16, xb16, yb16)),
            upload(&client, &repack(&xy, 16, 16, xb16, yb16)),
            upload(&client, &repack(&xb, 16, 16, xb16, yb16)),
            upload(&client, &replicate(&dct16_all[..256], nb16)),
            upload(&client, &replicate(&dct16_all[256..512], nb16)),
            upload(&client, &replicate(&dct16_all[512..768], nb16)),
            upload(&client, &qac16),
            upload(&client, &qac16),
            upload(&client, &qac16),
            upload(&client, &thr_xb[..]),
            upload(&client, &thr_y[..]),
            upload(&client, &thr_xb[..]),
            h_mask.clone(),
            xb16 as u32,
            yb16 as u32,
        );
        let cost_dct16x16: Vec<f32> = {
            let bytes = client.read_one(cg16.costs).expect("dct16");
            let raw: &[f32] = f32::from_bytes(&bytes);
            let xs = xb16 * 2;
            let mut out = vec![0.0f32; nb16];
            for ry in 0..yb16 {
                for rx in 0..xb16 {
                    let by = ry * 2;
                    let bx = rx * 2;
                    out[ry * xb16 + rx] = raw[by * xs + bx]
                        + raw[by * xs + bx + 1]
                        + raw[(by + 1) * xs + bx]
                        + raw[(by + 1) * xs + bx + 1];
                }
            }
            out
        };

        // DCT16×8 rect (3-ch, aggregated to per-rect)
        let cg_16x8 = compute_cost_grid_dct16x8_xyb::<Backend>(
            &client,
            upload(&client, &repack(&xx, 8, 16, xb_16x8, yb_16x8)),
            upload(&client, &repack(&xy, 8, 16, xb_16x8, yb_16x8)),
            upload(&client, &repack(&xb, 8, 16, xb_16x8, yb_16x8)),
            upload(&client, &replicate(&dct16x8_all[..128], nb_16x8)),
            upload(&client, &replicate(&dct16x8_all[128..256], nb_16x8)),
            upload(&client, &replicate(&dct16x8_all[256..384], nb_16x8)),
            upload(&client, &qac_16x8),
            upload(&client, &qac_16x8),
            upload(&client, &qac_16x8),
            upload(&client, &thr_xb[..]),
            upload(&client, &thr_y[..]),
            upload(&client, &thr_xb[..]),
            h_mask.clone(),
            xb_16x8 as u32,
            yb_16x8 as u32,
        );
        let cost_dct16x8: Vec<f32> = {
            let bytes = client.read_one(cg_16x8.costs).expect("dct16x8");
            let raw: &[f32] = f32::from_bytes(&bytes);
            let mut out = vec![0.0f32; nb_16x8];
            for ry in 0..yb_16x8 {
                for rx in 0..xb_16x8 {
                    out[ry * xb_16x8 + rx] = raw[ry * (xb_16x8 * 2) + 2 * rx]
                        + raw[ry * (xb_16x8 * 2) + 2 * rx + 1];
                }
            }
            out
        };

        // DCT8×16 rect (3-ch, aggregated to per-rect)
        let cg_8x16 = compute_cost_grid_dct8x16_xyb::<Backend>(
            &client,
            upload(&client, &repack(&xx, 16, 8, xb_8x16, yb_8x16)),
            upload(&client, &repack(&xy, 16, 8, xb_8x16, yb_8x16)),
            upload(&client, &repack(&xb, 16, 8, xb_8x16, yb_8x16)),
            upload(&client, &replicate(&dct16x8_all[..128], nb_8x16)),
            upload(&client, &replicate(&dct16x8_all[128..256], nb_8x16)),
            upload(&client, &replicate(&dct16x8_all[256..384], nb_8x16)),
            upload(&client, &qac_8x16),
            upload(&client, &qac_8x16),
            upload(&client, &qac_8x16),
            upload(&client, &thr_xb[..]),
            upload(&client, &thr_y[..]),
            upload(&client, &thr_xb[..]),
            h_mask,
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
        let parts = select_partitions_16x16_full(&cost_dct8, &cost_dct16x16, extra, xb8, yb8);

        let mut counts = [0_usize; 4];
        for &p in &parts {
            counts[match p {
                Partition16x16::Dct16x16 => 0,
                Partition16x16::TwoDct16x8Horizontal => 1,
                Partition16x16::TwoDct8x16Vertical => 2,
                Partition16x16::FourDct8x8 => 3,
                Partition16x16::FourSubBlocks(_) => continue, // demo doesn't pass sub-block grids
            }] += 1;
        }

        let np = parts.len();
        for i in 0..4 {
            total[i] += counts[i];
        }
        total_regions += np;

        let pct = |c: usize| 100.0 * c as f32 / np as f32;
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        println!(
            "{:<20}  {:>5.1}%  {:>5.1}%  {:>5.1}%  {:>5.1}%  {:>5.1}%",
            name.chars().take(20).collect::<String>(),
            pct(counts[0]),
            pct(counts[1]),
            pct(counts[2]),
            pct(counts[3]),
            pct(counts[1] + counts[2]),
        );
    }

    let pct = |c: usize| 100.0 * c as f32 / total_regions as f32;
    println!("\n=== Aggregate ({} regions across {} images) ===", total_regions, paths.len());
    println!("  DCT16×16          : {:>6}  ({:>5.1}%)", total[0], pct(total[0]));
    println!("  Two DCT16×8 horiz : {:>6}  ({:>5.1}%)", total[1], pct(total[1]));
    println!("  Two DCT8×16 vert  : {:>6}  ({:>5.1}%)", total[2], pct(total[2]));
    println!("  Four DCT8×8       : {:>6}  ({:>5.1}%)", total[3], pct(total[3]));
    println!("  rect (16×8 + 8×16): {:>6}  ({:>5.1}%)", total[1] + total[2], pct(total[1] + total[2]));
}
