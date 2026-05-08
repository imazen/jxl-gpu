//! Phase 3 prototype: rect-aware 16×16 partition selection on a
//! synthetic mixed-content image.
//!
//! Composes the four 16×16-region strategies (Dct16x16, two Dct16x8
//! horizontal, two Dct8x16 vertical, four Dct8x8) by feeding the
//! corresponding cost grids into `select_partitions_16x16_full`.
//!
//! Validates that the new rectangular cost grids
//! (`compute_cost_grid_dct16x8_single_channel`,
//! `compute_cost_grid_dct8x16_single_channel`) interoperate with the
//! existing host-side partition selector and produce a valid
//! per-region pick distribution.
//!
//! Note: this demo's per-rect content packing uses the natural row-
//! major layout, which doesn't perfectly exploit the kernel's
//! "16×8 = 16-tall × 8-wide" libjxl convention. Content-driven
//! preference between rect orientations on this synthetic input is
//! therefore approximate; the demo's value is in API composition, not
//! content-driven selection accuracy.

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
    use jxl_encoder_gpu::pipeline::{
        CostGrids16x16, Partition16x16, compute_cost_grid_dct8_single_channel,
        compute_cost_grid_dct8x16_single_channel, compute_cost_grid_dct16x8_single_channel,
        compute_cost_grid_dct16x16_single_channel, select_partitions_16x16_full,
    };

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // 8×8 = 64 8×8 blocks = 64×64 pixel image = 16 16×16 regions (4×4).
    const XB8: u32 = 8;
    const YB8: u32 = 8;
    const NB8: usize = (XB8 * YB8) as usize;

    // Mixed content: 4×4 pattern of 16×16 region types.
    //   region (rx, ry) ∈ {smooth, h-edge, v-edge, noise} cycling
    //   so the picker has each kind to choose from.
    let region_kind = |rx: usize, ry: usize| -> u8 { ((rx + 2 * ry) % 4) as u8 };

    let mut input8 = vec![0.0f32; NB8 * 64];
    for by in 0..YB8 as usize {
        for bx in 0..XB8 as usize {
            let block_idx = by * XB8 as usize + bx;
            let ry = by / 2;
            let rx = bx / 2;
            let kind = region_kind(rx, ry);
            for i in 0..64 {
                let r = (i / 8) as f32;
                let c = (i % 8) as f32;
                let v = ((block_idx * 7 + i * 13).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
                input8[block_idx * 64 + i] = match kind {
                    0 => 0.3 + 0.05 * (r + c),          // smooth diag
                    1 => 0.3 + 0.4 * ((c / 7.0) - 0.5), // strong horizontal gradient (favors DCT8×16)
                    2 => 0.3 + 0.4 * ((r / 7.0) - 0.5), // strong vertical gradient (favors DCT16×8)
                    _ => 0.3 + 0.4 * v,                 // noise (favors DCT8×8)
                };
            }
        }
    }

    // Repack into the layouts each cost-grid kernel needs.
    // DCT8 layout: per-8×8 contiguous (already correct).
    // DCT16×16 layout: per-16×16 contiguous (256 floats / region, row-major within region).
    // DCT16×8 layout: per-16×8 contiguous (128 floats / rect block, row-major).
    // DCT8×16 layout: per-8×16 contiguous (128 floats / rect block, row-major).

    const NB16: usize = NB8 / 4; // 4×4 = 16 regions
    let mut input16 = vec![0.0f32; NB16 * 256];
    for ry in 0..YB8 as usize / 2 {
        for rx in 0..XB8 as usize / 2 {
            let region = ry * (XB8 as usize / 2) + rx;
            for ly in 0..16 {
                for lx in 0..16 {
                    let by = ry * 2 + ly / 8;
                    let bx = rx * 2 + lx / 8;
                    let block = by * XB8 as usize + bx;
                    let inb = (ly % 8) * 8 + (lx % 8);
                    input16[region * 256 + ly * 16 + lx] = input8[block * 64 + inb];
                }
            }
        }
    }

    // DCT16×8 grid: 16×8 rect blocks. Image is 64×64 = 4 wide × 8 tall in
    // 16×8 units. Each rect block = 128 floats, row-major within rect.
    const X_16X8: usize = 4;
    const Y_16X8: usize = 8;
    const N_16X8: usize = X_16X8 * Y_16X8;
    let mut input16x8 = vec![0.0f32; N_16X8 * 128];
    for ry in 0..Y_16X8 {
        for rx in 0..X_16X8 {
            let rect = ry * X_16X8 + rx;
            for ly in 0..8 {
                for lx in 0..16 {
                    let by = ry;
                    let bx = rx * 2 + lx / 8;
                    let block = by * XB8 as usize + bx;
                    let inb = ly * 8 + (lx % 8);
                    input16x8[rect * 128 + ly * 16 + lx] = input8[block * 64 + inb];
                }
            }
        }
    }

    // DCT8×16 grid: 8×16 rect blocks. Image is 64×64 = 8 wide × 4 tall in
    // 8×16 units.
    const X_8X16: usize = 8;
    const Y_8X16: usize = 4;
    const N_8X16: usize = X_8X16 * Y_8X16;
    let mut input8x16 = vec![0.0f32; N_8X16 * 128];
    for ry in 0..Y_8X16 {
        for rx in 0..X_8X16 {
            let rect = ry * X_8X16 + rx;
            for ly in 0..16 {
                for lx in 0..8 {
                    let by = ry * 2 + ly / 8;
                    let bx = rx;
                    let block = by * XB8 as usize + bx;
                    let inb = (ly % 8) * 8 + lx;
                    input8x16[rect * 128 + ly * 8 + lx] = input8[block * 64 + inb];
                }
            }
        }
    }

    let weights_block = |size: usize, side: usize| {
        let mut w = vec![1.0f32; size];
        for i in 0..size {
            let r = (i / side) as f32;
            let c = (i % side) as f32;
            w[i] = 1.0 + 0.7 * (r + c);
        }
        w
    };
    let replicate = |per: &[f32], n: usize| {
        let mut out = vec![0.0f32; n * per.len()];
        for b in 0..n {
            out[b * per.len()..(b + 1) * per.len()].copy_from_slice(per);
        }
        out
    };

    let w8 = replicate(&weights_block(64, 8), NB8);
    let w16 = replicate(&weights_block(256, 16), NB16);
    let w16x8 = replicate(&weights_block(128, 16), N_16X8);
    let w8x16 = replicate(&weights_block(128, 8), N_8X16);

    let qac8 = vec![1.7f32; NB8];
    let qac16 = vec![1.7f32; NB16];
    let qac16x8 = vec![1.7f32; N_16X8];
    let qac8x16 = vec![1.7f32; N_8X16];
    let thresholds = [0.62f32; 4];
    let h_thr = client.create_from_slice(f32::as_bytes(&thresholds[..]));

    // ---- DCT8 ----
    let cg_dct8 = compute_cost_grid_dct8_single_channel::<Backend>(
        &client,
        client.create_from_slice(f32::as_bytes(&input8)),
        client.create_from_slice(f32::as_bytes(&w8)),
        client.create_from_slice(f32::as_bytes(&qac8)),
        h_thr.clone(),
        XB8,
        YB8,
    );
    let cost_dct8: Vec<f32> = {
        let b = client.read_one(cg_dct8.costs).expect("dct8");
        f32::from_bytes(&b).to_vec()
    };

    // ---- DCT16×16 (per-region aggregate from 4 sub-cell costs) ----
    let cg_dct16 = compute_cost_grid_dct16x16_single_channel::<Backend>(
        &client,
        client.create_from_slice(f32::as_bytes(&input16)),
        client.create_from_slice(f32::as_bytes(&w16)),
        client.create_from_slice(f32::as_bytes(&qac16)),
        h_thr.clone(),
        XB8 / 2,
        YB8 / 2,
    );
    let cost_dct16x16: Vec<f32> = {
        let bytes = client.read_one(cg_dct16.costs).expect("dct16");
        let raw: &[f32] = f32::from_bytes(&bytes);
        let xb16 = (XB8 / 2) as usize;
        let yb16 = (YB8 / 2) as usize;
        let xb_sub = xb16 * 2;
        let mut out = vec![0.0f32; xb16 * yb16];
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

    // ---- DCT16×8 (rect, 2 sub-cells per rect) ----
    let cg_16x8 = compute_cost_grid_dct16x8_single_channel::<Backend>(
        &client,
        client.create_from_slice(f32::as_bytes(&input16x8)),
        client.create_from_slice(f32::as_bytes(&w16x8)),
        client.create_from_slice(f32::as_bytes(&qac16x8)),
        h_thr.clone(),
        X_16X8 as u32,
        Y_16X8 as u32,
    );
    let cost_dct16x8: Vec<f32> = {
        let bytes = client.read_one(cg_16x8.costs).expect("16x8");
        let raw: &[f32] = f32::from_bytes(&bytes);
        let mut out = vec![0.0f32; X_16X8 * Y_16X8];
        for ry in 0..Y_16X8 {
            for rx in 0..X_16X8 {
                out[ry * X_16X8 + rx] =
                    raw[ry * (X_16X8 * 2) + 2 * rx] + raw[ry * (X_16X8 * 2) + 2 * rx + 1];
            }
        }
        out
    };

    // ---- DCT8×16 (rect, 2 sub-cells per rect) ----
    let cg_8x16 = compute_cost_grid_dct8x16_single_channel::<Backend>(
        &client,
        client.create_from_slice(f32::as_bytes(&input8x16)),
        client.create_from_slice(f32::as_bytes(&w8x16)),
        client.create_from_slice(f32::as_bytes(&qac8x16)),
        h_thr,
        X_8X16 as u32,
        Y_8X16 as u32,
    );
    let cost_dct8x16: Vec<f32> = {
        let bytes = client.read_one(cg_8x16.costs).expect("8x16");
        let raw: &[f32] = f32::from_bytes(&bytes);
        let mut out = vec![0.0f32; X_8X16 * Y_8X16];
        for ry in 0..Y_8X16 {
            for rx in 0..X_8X16 {
                out[ry * X_8X16 + rx] = raw[2 * ry * X_8X16 + rx] + raw[(2 * ry + 1) * X_8X16 + rx];
            }
        }
        out
    };

    // ---- Selector (with rect cost grids) ----
    let extra = CostGrids16x16 {
        dct_16x8: Some(&cost_dct16x8),
        dct_8x16: Some(&cost_dct8x16),
        ..Default::default()
    };
    let partitions = select_partitions_16x16_full(
        &cost_dct8,
        &cost_dct16x16,
        extra,
        XB8 as usize,
        YB8 as usize,
    );

    // Count partition kinds.
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

    println!("=== phase3_rect_selector_demo ===");
    println!("Image: 64×64, 16 regions (4×4), 4-kind cycling content");
    println!(
        "\nPer-strategy partition counts (out of {}):",
        partitions.len()
    );
    println!("  DCT16×16          : {}", counts[0]);
    println!("  Two DCT16×8 horiz : {}", counts[1]);
    println!("  Two DCT8×16 vert  : {}", counts[2]);
    println!("  Four DCT8×8       : {}", counts[3]);
    println!("\nPartition map (region grid, top-left → bottom-right):");
    let xb16 = (XB8 / 2) as usize;
    let yb16 = (YB8 / 2) as usize;
    for ry in 0..yb16 {
        let row: Vec<&str> = (0..xb16)
            .map(|rx| match partitions[ry * xb16 + rx] {
                Partition16x16::Dct16x16 => "16×16",
                Partition16x16::TwoDct16x8Horizontal => "16×8 ",
                Partition16x16::TwoDct8x16Vertical => "8×16 ",
                Partition16x16::FourDct8x8 => "8×8  ",
                Partition16x16::FourSubBlocks(_) => "sub  ",
            })
            .collect();
        println!("  {}", row.join("  "));
    }
}
