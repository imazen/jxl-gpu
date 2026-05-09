//! Phase 3 end-to-end on a real image with PER-CELL sub-block strategy
//! choice — runs the full 7-strategy 16×16 partition selector
//! (DCT16×16 + 2× DCT16×8 + 2× DCT8×16 + 4-cell sub-block where each
//! cell independently picks DCT8 / DCT4×4 / DCT4×8 / DCT8×4 / IDENTITY
//! / DCT2X2) and reports partition + sub-strategy pick distributions.
//!
//! This is the most expressive selector configuration available — the
//! union of all 16x16-tier alternatives plus the per-cell 8x8-tier
//! choice that `Partition16x16::FourSubBlocks` enables.
//!
//! Uses real libjxl DCT8 quant weights via [`quant_weights::dct8_weights`].

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
    let (_xx, xy, _xb) = enc.xyb_from_linear_rgb(&r, &g, &b);

    // 8×8 block-major Y plane.
    let xb8 = (w as usize) / 8;
    let yb8 = (h as usize) / 8;
    let nb8 = xb8 * yb8;
    let mut y8 = vec![0.0f32; nb8 * 64];
    for by in 0..yb8 {
        for bx in 0..xb8 {
            for ly in 0..8 {
                for lx in 0..8 {
                    y8[(by * xb8 + bx) * 64 + ly * 8 + lx] =
                        xy[(by * 8 + ly) * (w as usize) + (bx * 8 + lx)];
                }
            }
        }
    }
    // 16×16 region-major Y plane.
    let xb16 = xb8 / 2;
    let yb16 = yb8 / 2;
    let nb16 = xb16 * yb16;
    let mut y16 = vec![0.0f32; nb16 * 256];
    for ry in 0..yb16 {
        for rx in 0..xb16 {
            for ly in 0..16 {
                for lx in 0..16 {
                    y16[(ry * xb16 + rx) * 256 + ly * 16 + lx] =
                        xy[(ry * 16 + ly) * (w as usize) + (rx * 16 + lx)];
                }
            }
        }
    }
    // 16×8 (8 wide × 16 tall pixels per rect).
    let xb_16x8 = xb8;
    let yb_16x8 = yb8 / 2;
    let nb_16x8 = xb_16x8 * yb_16x8;
    let mut y_16x8 = vec![0.0f32; nb_16x8 * 128];
    for ry in 0..yb_16x8 {
        for rx in 0..xb_16x8 {
            for ly in 0..16 {
                for lx in 0..8 {
                    y_16x8[(ry * xb_16x8 + rx) * 128 + ly * 8 + lx] =
                        xy[(ry * 16 + ly) * (w as usize) + (rx * 8 + lx)];
                }
            }
        }
    }
    // 8×16 (16 wide × 8 tall pixels per rect).
    let xb_8x16 = xb8 / 2;
    let yb_8x16 = yb8;
    let nb_8x16 = xb_8x16 * yb_8x16;
    let mut y_8x16 = vec![0.0f32; nb_8x16 * 128];
    for ry in 0..yb_8x16 {
        for rx in 0..xb_8x16 {
            for ly in 0..8 {
                for lx in 0..16 {
                    y_8x16[(ry * xb_8x16 + rx) * 128 + ly * 16 + lx] =
                        xy[(ry * 8 + ly) * (w as usize) + (rx * 16 + lx)];
                }
            }
        }
    }

    // Real DCT8 Y-channel weights, replicated per-block.
    let (_wx, wy_per, _wb) = dct8_weights_per_channel();
    let weights8 = replicate_weights(&wy_per, nb8);
    // Real DCT16x16 / DCT16x8 / DCT8x16 weights via quant_weights —
    // the demo uses the Y-channel slice for single-channel cost grids.
    let dct16_all = dct16x16_weights();
    let weights16 = replicate_weights(&dct16_all[256..512], nb16);
    let dct16x8_all = dct16x8_weights();
    let weights16x8 = replicate_weights(&dct16x8_all[128..256], nb_16x8);
    // DCT8x16 shares the DCT16x8 weight table per upstream.
    let weights8x16 = replicate_weights(&dct16x8_all[128..256], nb_8x16);
    let qac8 = vec![1.7f32; nb8];
    let qac16 = vec![1.7f32; nb16];
    let qac_16x8 = vec![1.7f32; nb_16x8];
    let qac_8x16 = vec![1.7f32; nb_8x16];
    let thresholds = [0.62f32; 4];

    let upload = |v: &[f32]| client.create_from_slice(f32::as_bytes(v));
    let h_thr = upload(&thresholds[..]);

    // --- Compute all 9 cost grids (5 per-8×8-cell sub-strategies + DCT8 +
    //     DCT16x16 + 2 rect) ---
    let read = |handle, label: &str| -> Vec<f32> {
        let bytes = client.read_one(handle).expect(label);
        f32::from_bytes(&bytes).to_vec()
    };

    macro_rules! grid64 {
        ($fname:ident) => {
            read(
                $fname::<Backend>(
                    &client,
                    upload(&y8),
                    upload(&weights8),
                    upload(&qac8),
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

    // DCT16x16 — different shape (256-coef).
    let cg_dct16 = compute_cost_grid_dct16x16_single_channel::<Backend>(
        &client,
        upload(&y16),
        upload(&weights16),
        upload(&qac16),
        h_thr.clone(),
        xb16 as u32,
        yb16 as u32,
    );
    // Aggregate 4 sub-cells per 16×16 region.
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
    // DCT16x8 — 128-coef rect.
    let cg_16x8 = compute_cost_grid_dct16x8_single_channel::<Backend>(
        &client,
        upload(&y_16x8),
        upload(&weights16x8),
        upload(&qac_16x8),
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
        upload(&y_8x16),
        upload(&weights8x16),
        upload(&qac_8x16),
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

    // --- 7-strategy partition selection ---
    let extra = CostGrids16x16 {
        dct_16x8: Some(&cost_dct16x8),
        dct_8x16: Some(&cost_dct8x16),
        sub_blocks: SubBlockCostGrids {
            dct4x4: Some(&cost_dct4x4),
            dct4x8: Some(&cost_dct4x8),
            dct8x4: Some(&cost_dct8x4),
            identity: Some(&cost_identity),
            dct2x2: Some(&cost_dct2x2),
            afv0: None,
            afv1: None,
            afv2: None,
            afv3: None,
        },
    };
    let parts = select_partitions_16x16_full(&cost_dct8, &cost_dct16x16, extra, xb8, yb8);

    let mut counts = [0_usize; 5]; // 0=16x16, 1=16x8, 2=8x16, 3=4-DCT8, 4=4-SubBlocks
    let mut sub_counts = [0_usize; 6]; // SubStrategy variants
    for &p in &parts {
        match p {
            Partition16x16::Dct16x16 => counts[0] += 1,
            Partition16x16::TwoDct16x8Horizontal => counts[1] += 1,
            Partition16x16::TwoDct8x16Vertical => counts[2] += 1,
            Partition16x16::FourDct8x8 => counts[3] += 1,
            Partition16x16::FourSubBlocks(s) => {
                counts[4] += 1;
                for &c in &s {
                    sub_counts[match c {
                        SubStrategy::Dct8 => 0,
                        SubStrategy::Dct4x4 => 1,
                        SubStrategy::Dct4x8 => 2,
                        SubStrategy::Dct8x4 => 3,
                        SubStrategy::Identity => 4,
                        SubStrategy::Dct2x2 => 5,
                        // AFV variants share the SubBlocks bucket with no
                        // separate breakdown in this demo; treat as DCT8
                        // for histogram purposes (rare in practice).
                        SubStrategy::Afv0
                        | SubStrategy::Afv1
                        | SubStrategy::Afv2
                        | SubStrategy::Afv3 => 0,
                    }] += 1;
                }
            }
        }
    }

    let total = parts.len();
    let pct = |c: usize, t: usize| 100.0 * c as f32 / t as f32;

    println!("=== phase3_subblock_real_image_demo ===");
    println!("Image: {image_path}");
    println!(
        "Crop:  {w}×{h} ({nb8} 8×8 blocks, {} 16×16 regions)\n",
        total
    );
    println!("16×16-tier partition picks:");
    println!(
        "  DCT16×16          : {:>6}  ({:>5.1}%)",
        counts[0],
        pct(counts[0], total)
    );
    println!(
        "  Two DCT16×8 horiz : {:>6}  ({:>5.1}%)",
        counts[1],
        pct(counts[1], total)
    );
    println!(
        "  Two DCT8×16 vert  : {:>6}  ({:>5.1}%)",
        counts[2],
        pct(counts[2], total)
    );
    println!(
        "  Four DCT8×8       : {:>6}  ({:>5.1}%)",
        counts[3],
        pct(counts[3], total)
    );
    println!(
        "  Four SubBlocks    : {:>6}  ({:>5.1}%)",
        counts[4],
        pct(counts[4], total)
    );

    if counts[4] > 0 {
        let sub_total: usize = sub_counts.iter().sum();
        println!(
            "\nWithin {} FourSubBlocks regions ({} 8×8 cells), per-cell strategy:",
            counts[4], sub_total
        );
        let names = ["DCT8", "DCT4×4", "DCT4×8", "DCT8×4", "IDENTITY", "DCT2X2"];
        for (i, name) in names.iter().enumerate() {
            println!(
                "  {name:<10}: {:>6}  ({:>5.1}%)",
                sub_counts[i],
                pct(sub_counts[i], sub_total)
            );
        }
    }
}
