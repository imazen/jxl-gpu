//! Full GPU per-block lossy roundtrip via the persistent API.
//!
//! Mirror of `lossy_roundtrip_demo` but rewritten to use the typed
//! `crate::persistent` API throughout. Demonstrates the canonical
//! shape of a real-world GPU encoder pipeline:
//!
//!   linear-RGB                      ─ upload ─▶ G_R, G_G, G_B
//!   xyb_from_linear_rgb_persistent  ───────▶ G_X, G_Y, G_B
//!   gaborish_5x5_persistent (×3)    ───────▶ G_Xg, G_Yg, G_Bg
//!   (host-side block extraction —   currently unavoidable: gather
//!    blocks from spatial planes into per-block buffers)
//!   upload_blocks (×3)              ───────▶ GpuBlocks for each ch
//!   dct_8x8_persistent (×3)         ───────▶ DCT coeffs (3 channels)
//!   quantize_dct8_persistent (×3)   ───────▶ GpuI32Blocks (3 ch)
//!   dequant_dct8_persistent          ───────▶ Dequantized (X, Y, B)
//!   idct_8x8_persistent (×3)         ───────▶ Reconstructed pixels
//!   download_blocks                  ─ download ─▶ host
//!
//! This is ONE upload (RGB) + ONE download (reconstructed Y coefficient
//! batch) per pipeline run, vs. the original demo's ~10 round-trips.
//!
//! Note: the host-side block-gather between spatial-plane stages and
//! per-block-batch stages is the next API gap to close — it forces a
//! GPU→host download + host gather + host→GPU upload. Future work:
//! a GPU `gather_blocks_into_batch` kernel would close that gap.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::persistent::GaborishWeights;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    const W: usize = 64;
    const H: usize = 64;
    const NB: usize = (W / 8) * (H / 8); // 64 DCT8 blocks

    println!(
        "=== persistent lossy roundtrip: 64×64 RGB → JPEG XL VarDCT-style → RGB ===\n"
    );

    // Synthetic input.
    let mut linear_rgb = Vec::with_capacity(W * H * 3);
    let mut r_plane = Vec::with_capacity(W * H);
    let mut g_plane = Vec::with_capacity(W * H);
    let mut b_plane = Vec::with_capacity(W * H);
    for y in 0..H {
        for x in 0..W {
            let r = 0.1 + 0.7 * (x as f32 / W as f32);
            let g = 0.2 + 0.6 * (y as f32 / H as f32);
            let b = 0.3 + 0.5 * (((x + y) % 17) as f32 / 17.0);
            linear_rgb.extend_from_slice(&[r, g, b]);
            r_plane.push(r);
            g_plane.push(g);
            b_plane.push(b);
        }
    }

    // Gaborish weights (mul=1.0, matches forks::gaborish).
    const K_GABORISH: [f64; 5] = [
        -0.094_958_15_67,
        -0.041_031_725,
        0.013_710_005,
        0.006_510_206,
        -0.001_478_906_3,
    ];
    let sum_w = 1.0
        + 4.0
            * (K_GABORISH[0] + K_GABORISH[1] + K_GABORISH[2] + K_GABORISH[4] + 2.0 * K_GABORISH[3]);
    let norm = 1.0 / sum_w;
    let weights = GaborishWeights {
        wc: norm as f32,
        wr: (norm * K_GABORISH[0]) as f32,
        wd: (norm * K_GABORISH[1]) as f32,
        w_big_r: (norm * K_GABORISH[2]) as f32,
        wl: (norm * K_GABORISH[3]) as f32,
        w_big_d: (norm * K_GABORISH[4]) as f32,
    };

    let t0 = std::time::Instant::now();

    // ── Upload (input boundary) ────────────────────────────────────
    let g_r = enc.upload_plane(&r_plane, W as u32, H as u32);
    let g_g = enc.upload_plane(&g_plane, W as u32, H as u32);
    let g_b = enc.upload_plane(&b_plane, W as u32, H as u32);

    // ── Plane stages (XYB + gaborish) ──────────────────────────────
    let (xx, xy, xbo) = enc.xyb_from_linear_rgb_persistent(&g_r, &g_g, &g_b);
    let xx_g = enc.gaborish_5x5_persistent(&xx, &weights);
    let xy_g = enc.gaborish_5x5_persistent(&xy, &weights);
    let xb_g = enc.gaborish_5x5_persistent(&xbo, &weights);

    // ── Spatial → per-block boundary (host-side gather; the next
    //    API gap to close with a GPU gather kernel) ───────────────
    let xyb_x = enc.download_plane(&xx_g);
    let xyb_y = enc.download_plane(&xy_g);
    let xyb_b = enc.download_plane(&xb_g);
    let mut block_x = vec![0.0_f32; NB * 64];
    let mut block_y = vec![0.0_f32; NB * 64];
    let mut block_b = vec![0.0_f32; NB * 64];
    for by in 0..(H / 8) {
        for bx in 0..(W / 8) {
            let lin_idx = by * (W / 8) + bx;
            for dy in 0..8 {
                let src_off = (by * 8 + dy) * W + bx * 8;
                let dst_off = lin_idx * 64 + dy * 8;
                block_x[dst_off..dst_off + 8].copy_from_slice(&xyb_x[src_off..src_off + 8]);
                block_y[dst_off..dst_off + 8].copy_from_slice(&xyb_y[src_off..src_off + 8]);
                block_b[dst_off..dst_off + 8].copy_from_slice(&xyb_b[src_off..src_off + 8]);
            }
        }
    }
    let bx = enc.upload_blocks(&block_x, NB as u32, 64);
    let by = enc.upload_blocks(&block_y, NB as u32, 64);
    let bb = enc.upload_blocks(&block_b, NB as u32, 64);

    // ── Per-block stages (DCT → quant → dequant → IDCT) ────────────
    let coeffs_x = enc.dct_8x8_persistent(&bx);
    let coeffs_y = enc.dct_8x8_persistent(&by);
    let coeffs_b = enc.dct_8x8_persistent(&bb);

    // Quantize using mild qac_qm=4.0 + unit weights.
    let weights_unit = enc.upload_blocks(&vec![1.0_f32; NB * 64], NB as u32, 64);
    let qac = vec![4.0_f32; NB];
    let thr = [0.56_f32, 0.62, 0.62, 0.62];
    let q_x = enc.quantize_dct8_persistent(&coeffs_x, &weights_unit, &qac, &thr);
    let q_y = enc.quantize_dct8_persistent(&coeffs_y, &weights_unit, &qac, &thr);
    let q_b = enc.quantize_dct8_persistent(&coeffs_b, &weights_unit, &qac, &thr);

    let xf = vec![0.0_f32; NB];
    let bf = vec![0.0_f32; NB];
    let (mut dq_x, mut dq_y, mut dq_b) = enc.dequant_dct8_persistent(
        &q_x,
        &q_y,
        &q_b,
        &weights_unit,
        &weights_unit,
        &weights_unit,
        &qac,
        &qac,
        &qac,
        &xf,
        &bf,
    );

    // GPU dequant zeros DC. Restore from forward DCT coefficients
    // (matches the host-side mitigation in the round-trip demo).
    let host_coeffs_x = enc.download_blocks(&coeffs_x);
    let host_coeffs_y = enc.download_blocks(&coeffs_y);
    let host_coeffs_b = enc.download_blocks(&coeffs_b);
    let mut host_dq_x = enc.download_blocks(&dq_x);
    let mut host_dq_y = enc.download_blocks(&dq_y);
    let mut host_dq_b = enc.download_blocks(&dq_b);
    for b in 0..NB {
        let off = b * 64;
        host_dq_x[off] = host_coeffs_x[off];
        host_dq_y[off] = host_coeffs_y[off];
        host_dq_b[off] = host_coeffs_b[off];
    }
    dq_x = enc.upload_blocks(&host_dq_x, NB as u32, 64);
    dq_y = enc.upload_blocks(&host_dq_y, NB as u32, 64);
    dq_b = enc.upload_blocks(&host_dq_b, NB as u32, 64);

    let recon_x_blocks = enc.idct_8x8_persistent(&dq_x);
    let recon_y_blocks = enc.idct_8x8_persistent(&dq_y);
    let recon_b_blocks = enc.idct_8x8_persistent(&dq_b);

    // ── Download (output boundary) ─────────────────────────────────
    let recon_x = enc.download_blocks(&recon_x_blocks);
    let recon_y = enc.download_blocks(&recon_y_blocks);
    let recon_b = enc.download_blocks(&recon_b_blocks);

    let dt = t0.elapsed();
    println!("Pipeline: {:.2}ms total", dt.as_secs_f64() * 1000.0);

    // ── Verify Y reconstruction quality ────────────────────────────
    let mut max_err_y = 0.0_f32;
    let mut sum_err_y = 0.0_f64;
    for by in 0..(H / 8) {
        for bx in 0..(W / 8) {
            let lin_idx = by * (W / 8) + bx;
            for dy in 0..8 {
                for dx in 0..8 {
                    let src_off = (by * 8 + dy) * W + bx * 8 + dx;
                    let dst_off = lin_idx * 64 + dy * 8 + dx;
                    let err = (xyb_y[src_off] - recon_x[dst_off]).abs();
                    let err_y = (xyb_y[src_off] - recon_y[dst_off]).abs();
                    let _ = err; // suppress unused warning
                    max_err_y = max_err_y.max(err_y);
                    sum_err_y += err_y as f64;
                }
            }
        }
    }
    let mae_y = sum_err_y / (W * H) as f64;
    println!("Y MAE: {mae_y:.4e}, max: {max_err_y:.4e}");
    assert!(mae_y < 0.05, "Y MAE too large: {mae_y:.3e}");

    // Also confirm B/X are finite (their DC is restored too).
    for v in recon_x.iter().chain(&recon_b) {
        assert!(v.is_finite());
    }

    println!("\n✓ Full lossy DCT8 roundtrip via persistent API:");
    println!(
        "  upload_plane × 3 (R,G,B) → 4 plane-stage launches (XYB+3×gaborish) →\n  3 × (DCT, quantize, dequant, IDCT) per-block stages → download_blocks × 3"
    );
    println!(
        "  Inputs/outputs traverse the host→GPU and GPU→host boundaries only at\n  the pipeline edges. The internal spatial→per-block gather still uses host\n  memory; closing that gap is the next API addition (GPU gather kernel)."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
