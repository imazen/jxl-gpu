//! Corpus-level validation of the 7-strategy 16×16 partition selector
//! with per-cell sub-block choice across CLIC2025.
//!
//! Walks a directory of PNGs and runs the full selector
//! (DCT16×16 + 2×DCT16×8 + 2×DCT8×16 + Four{DCT8|SubBlocks}) on each,
//! then aggregates partition + sub-strategy pick distributions across
//! the corpus. Confirms whether the single-image finding from
//! `phase3_subblock_real_image_demo` (pure 4-DCT8 nearly vanishes,
//! ~33% FourSubBlocks) generalizes.
//!
//! Usage:
//!   cargo run --release --features cuda --example corpus_subblock_picks_demo
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
    use jxl_encoder_gpu::pipeline::{
        CostGrids16x16, Partition16x16, SubBlockCostGrids, SubStrategy,
        compute_cost_grid_dct2x2_single_channel, compute_cost_grid_dct4x4_single_channel,
        compute_cost_grid_dct4x8_single_channel, compute_cost_grid_dct8_single_channel,
        compute_cost_grid_dct8x4_single_channel, compute_cost_grid_dct8x16_single_channel,
        compute_cost_grid_dct16x8_single_channel, compute_cost_grid_dct16x16_single_channel,
        compute_cost_grid_identity_single_channel, select_partitions_16x16_full,
    };
    use jxl_encoder_gpu::quant_weights::{
        dct8_weights_per_channel, dct16x8_weights, dct16x16_weights, replicate_weights,
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

    println!("=== corpus_subblock_picks_demo ===");
    println!("Corpus: {corpus_dir}");
    println!("Images: {}\n", paths.len());

    let mut total_partitions = [0_usize; 5];
    let mut total_subs = [0_usize; 6];
    let mut total_regions = 0_usize;

    let upload_f = |c: &ComputeClient<Backend>, v: &[f32]| c.create_from_slice(f32::as_bytes(v));

    let (_wx, wy_per, _wb) = dct8_weights_per_channel();

    println!(
        "{:<22}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}",
        "image", "16×16", "16×8", "8×16", "DCT8", "Sub"
    );

    for path in &paths {
        let img = match image::open(path) {
            Ok(i) => i.to_rgb8(),
            Err(_) => continue,
        };
        let (w, h) = img.dimensions();
        let w = (w / 16) * 16;
        let h = (h / 16) * 16;
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
        let (_xx, xy, _xb) = enc.xyb_from_linear_rgb(&r, &g, &b);

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

        // Repackers per layout.
        let repack_8 = |plane: &[f32]| {
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
        let repack_16 = |plane: &[f32]| {
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

        let y8 = repack_8(&xy);
        let y16 = repack_16(&xy);
        let y_16x8 = repack_16x8(&xy);
        let y_8x16 = repack_8x16(&xy);

        let weights8 = replicate_weights(&wy_per, nb8);
        // Real DCT16x16 / DCT16x8 / DCT8x16 weights via quant_weights —
        // single-channel cost grids use the Y-channel slices.
        let dct16_all = dct16x16_weights();
        let weights16 = replicate_weights(&dct16_all[256..512], nb16);
        let dct16x8_all = dct16x8_weights();
        let weights16x8 = replicate_weights(&dct16x8_all[128..256], nb_16x8);
        let weights8x16 = replicate_weights(&dct16x8_all[128..256], nb_8x16);
        let qac8 = vec![1.7f32; nb8];
        let qac16 = vec![1.7f32; nb16];
        let qac_16x8 = vec![1.7f32; nb_16x8];
        let qac_8x16 = vec![1.7f32; nb_8x16];
        let thresholds = [0.62f32; 4];
        let h_thr = upload_f(&client, &thresholds[..]);

        let read = |handle, label: &str| -> Vec<f32> {
            let bytes = client.read_one(handle).expect(label);
            f32::from_bytes(&bytes).to_vec()
        };

        macro_rules! grid64 {
            ($fname:ident) => {
                read(
                    $fname::<Backend>(
                        &client,
                        upload_f(&client, &y8),
                        upload_f(&client, &weights8),
                        upload_f(&client, &qac8),
                        h_thr.clone(),
                        xb8 as u32,
                        yb8 as u32,
                    )
                    .costs,
                    stringify!($fname),
                )
            };
        }
        let cost_dct8 = grid64!(compute_cost_grid_dct8_single_channel);
        let cost_dct4x4 = grid64!(compute_cost_grid_dct4x4_single_channel);
        let cost_dct4x8 = grid64!(compute_cost_grid_dct4x8_single_channel);
        let cost_dct8x4 = grid64!(compute_cost_grid_dct8x4_single_channel);
        let cost_identity = grid64!(compute_cost_grid_identity_single_channel);
        let cost_dct2x2 = grid64!(compute_cost_grid_dct2x2_single_channel);

        let cg_dct16 = compute_cost_grid_dct16x16_single_channel::<Backend>(
            &client,
            upload_f(&client, &y16),
            upload_f(&client, &weights16),
            upload_f(&client, &qac16),
            h_thr.clone(),
            xb16 as u32,
            yb16 as u32,
        );
        let cost_dct16x16: Vec<f32> = {
            let raw = read(cg_dct16.costs, "dct16");
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
        let cg_16x8 = compute_cost_grid_dct16x8_single_channel::<Backend>(
            &client,
            upload_f(&client, &y_16x8),
            upload_f(&client, &weights16x8),
            upload_f(&client, &qac_16x8),
            h_thr.clone(),
            xb_16x8 as u32,
            yb_16x8 as u32,
        );
        let cost_dct16x8: Vec<f32> = {
            let raw = read(cg_16x8.costs, "16x8");
            let mut out = vec![0.0f32; nb_16x8];
            for ry in 0..yb_16x8 {
                for rx in 0..xb_16x8 {
                    out[ry * xb_16x8 + rx] =
                        raw[ry * (xb_16x8 * 2) + 2 * rx] + raw[ry * (xb_16x8 * 2) + 2 * rx + 1];
                }
            }
            out
        };
        let cg_8x16 = compute_cost_grid_dct8x16_single_channel::<Backend>(
            &client,
            upload_f(&client, &y_8x16),
            upload_f(&client, &weights8x16),
            upload_f(&client, &qac_8x16),
            h_thr,
            xb_8x16 as u32,
            yb_8x16 as u32,
        );
        let cost_dct8x16: Vec<f32> = {
            let raw = read(cg_8x16.costs, "8x16");
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
            sub_blocks: SubBlockCostGrids {
                dct4x4: Some(&cost_dct4x4),
                dct4x8: Some(&cost_dct4x8),
                dct8x4: Some(&cost_dct8x4),
                identity: Some(&cost_identity),
                dct2x2: Some(&cost_dct2x2),
            },
        };
        let parts = select_partitions_16x16_full(&cost_dct8, &cost_dct16x16, extra, xb8, yb8);

        let mut counts = [0_usize; 5];
        let mut subs = [0_usize; 6];
        for &p in &parts {
            match p {
                Partition16x16::Dct16x16 => counts[0] += 1,
                Partition16x16::TwoDct16x8Horizontal => counts[1] += 1,
                Partition16x16::TwoDct8x16Vertical => counts[2] += 1,
                Partition16x16::FourDct8x8 => counts[3] += 1,
                Partition16x16::FourSubBlocks(s) => {
                    counts[4] += 1;
                    for &c in &s {
                        subs[match c {
                            SubStrategy::Dct8 => 0,
                            SubStrategy::Dct4x4 => 1,
                            SubStrategy::Dct4x8 => 2,
                            SubStrategy::Dct8x4 => 3,
                            SubStrategy::Identity => 4,
                            SubStrategy::Dct2x2 => 5,
                        }] += 1;
                    }
                }
            }
        }

        let np = parts.len();
        let pct = |c: usize| 100.0 * c as f32 / np as f32;
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        println!(
            "{:<22}  {:>4.1}%  {:>4.1}%  {:>4.1}%  {:>4.1}%  {:>4.1}%",
            name.chars().take(22).collect::<String>(),
            pct(counts[0]),
            pct(counts[1]),
            pct(counts[2]),
            pct(counts[3]),
            pct(counts[4]),
        );
        for i in 0..5 {
            total_partitions[i] += counts[i];
        }
        for i in 0..6 {
            total_subs[i] += subs[i];
        }
        total_regions += np;
    }

    let pct = |c: usize, t: usize| 100.0 * c as f32 / t as f32;
    println!(
        "\n=== Aggregate ({} regions across {} images) ===",
        total_regions,
        paths.len()
    );
    println!(
        "  DCT16×16          : {:>6}  ({:>5.1}%)",
        total_partitions[0],
        pct(total_partitions[0], total_regions)
    );
    println!(
        "  Two DCT16×8 horiz : {:>6}  ({:>5.1}%)",
        total_partitions[1],
        pct(total_partitions[1], total_regions)
    );
    println!(
        "  Two DCT8×16 vert  : {:>6}  ({:>5.1}%)",
        total_partitions[2],
        pct(total_partitions[2], total_regions)
    );
    println!(
        "  Four DCT8×8       : {:>6}  ({:>5.1}%)",
        total_partitions[3],
        pct(total_partitions[3], total_regions)
    );
    println!(
        "  Four SubBlocks    : {:>6}  ({:>5.1}%)",
        total_partitions[4],
        pct(total_partitions[4], total_regions)
    );

    let sub_total: usize = total_subs.iter().sum();
    if sub_total > 0 {
        println!("\nWithin {} sub-block cells:", sub_total);
        let names = ["DCT8", "DCT4×4", "DCT4×8", "DCT8×4", "IDENTITY", "DCT2X2"];
        for (i, name) in names.iter().enumerate() {
            println!(
                "  {name:<10}: {:>6}  ({:>5.1}%)",
                total_subs[i],
                pct(total_subs[i], sub_total)
            );
        }
    }
}
