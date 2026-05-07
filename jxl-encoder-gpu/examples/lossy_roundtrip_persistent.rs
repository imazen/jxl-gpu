//! Full GPU per-block lossy roundtrip via the persistent API.
//!
//! End-to-end on-GPU pipeline mirroring a real VarDCT lossy path:
//!
//!   linear-RGB                          ─ upload ─▶
//!   xyb_from_linear_rgb_persistent      ─────────▶
//!   gaborish_5x5_persistent (×3)        ─────────▶
//!   gather_blocks_persistent (×3)       ─────────▶ (GPU spatial→blocks)
//!   dct_8x8_persistent (×3)             ─────────▶
//!   quantize_dct8_persistent (×3)       ─────────▶ (i32 quantized)
//!   dequant_dct8_persistent              ─────────▶ (3-channel batched)
//!   idct_8x8_persistent (×3)             ─────────▶
//!   scatter_blocks_persistent (×3)      ─────────▶ (GPU blocks→spatial)
//!                                       ─ download ─▶ host
//!
//! With the gather/scatter GPU kernels (added 2026-05-06), the
//! pipeline now traverses the host↔GPU boundary only at the input
//! upload and final download. Internal stages stay GPU-resident.
//!
//! DC handling: the GPU quantize_dct8 kernel always zeros DC (in the
//! real encoder, DC has its own quant + entropy coding via dc_coding).
//! We bridge by calling restore_dc_persistent — a small GPU kernel
//! that copies the DC slot from the forward DCT output into the
//! dequantized buffers, leaving AC untouched. No host hop required.
//! In a real encoder, dc_coding handles DC properly (separate quant
//! + entropy); this restore is just a roundtrip-demo bridge.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::persistent::GaborishWeights;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    const W: usize = 64;
    const H: usize = 64;
    const NB: usize = (W / 8) * (H / 8); // 64 DCT8 blocks

    println!("=== full-GPU lossy roundtrip (gather/scatter on-GPU): 64×64 ===\n");

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

    // Gaborish weights (mul=1.0).
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

    // ── Plane stages: XYB + gaborish ───────────────────────────────
    let (xx, xy, xbo) = enc.xyb_from_linear_rgb_persistent(&g_r, &g_g, &g_b);
    let xx_g = enc.gaborish_5x5_persistent(&xx, &weights);
    let xy_g = enc.gaborish_5x5_persistent(&xy, &weights);
    let xb_g = enc.gaborish_5x5_persistent(&xbo, &weights);

    // ── Spatial → per-block via GPU gather (NO host transfer) ──────
    let bx_g = enc.gather_blocks_persistent(&xx_g, 8, 8);
    let by_g = enc.gather_blocks_persistent(&xy_g, 8, 8);
    let bb_g = enc.gather_blocks_persistent(&xb_g, 8, 8);

    // ── Per-block stages: DCT → quant → dequant → IDCT ─────────────
    let coeffs_x = enc.dct_8x8_persistent(&bx_g);
    let coeffs_y = enc.dct_8x8_persistent(&by_g);
    let coeffs_b = enc.dct_8x8_persistent(&bb_g);

    let weights_unit = enc.upload_blocks(&vec![1.0_f32; NB * 64], NB as u32, 64);
    let qac = vec![4.0_f32; NB];
    let thr = [0.56_f32, 0.62, 0.62, 0.62];
    let q_x = enc.quantize_dct8_persistent(&coeffs_x, &weights_unit, &qac, &thr);
    let q_y = enc.quantize_dct8_persistent(&coeffs_y, &weights_unit, &qac, &thr);
    let q_b = enc.quantize_dct8_persistent(&coeffs_b, &weights_unit, &qac, &thr);

    let xf = vec![0.0_f32; NB];
    let bf = vec![0.0_f32; NB];
    let (dq_x, dq_y, dq_b) = enc.dequant_dct8_persistent(
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

    // DC restoration on-GPU: copy DC slot from forward DCT outputs
    // into the dequantized buffers (no host transfer).
    enc.restore_dc_persistent(&coeffs_x, &dq_x);
    enc.restore_dc_persistent(&coeffs_y, &dq_y);
    enc.restore_dc_persistent(&coeffs_b, &dq_b);

    let recon_x_blocks = enc.idct_8x8_persistent(&dq_x);
    let recon_y_blocks = enc.idct_8x8_persistent(&dq_y);
    let recon_b_blocks = enc.idct_8x8_persistent(&dq_b);

    // ── Per-block → spatial via GPU scatter ────────────────────────
    let recon_x_plane = enc.scatter_blocks_persistent(&recon_x_blocks, W as u32, H as u32, 8, 8);
    let recon_y_plane = enc.scatter_blocks_persistent(&recon_y_blocks, W as u32, H as u32, 8, 8);
    let recon_b_plane = enc.scatter_blocks_persistent(&recon_b_blocks, W as u32, H as u32, 8, 8);

    // ── Inverse XYB on GPU → linear RGB ────────────────────────────
    let (rgb_r, rgb_g, rgb_b) =
        enc.xyb_to_linear_rgb_planar_persistent(&recon_x_plane, &recon_y_plane, &recon_b_plane);

    // ── Download (output boundary) ─────────────────────────────────
    let r_out = enc.download_plane(&rgb_r);
    let g_out = enc.download_plane(&rgb_g);
    let b_out = enc.download_plane(&rgb_b);
    let dt = t0.elapsed();
    println!("Pipeline: {:.2}ms total", dt.as_secs_f64() * 1000.0);

    // ── Verify final RGB reconstruction ────────────────────────────
    let mut sum_r = 0.0_f64;
    let mut sum_g = 0.0_f64;
    let mut sum_b_acc = 0.0_f64;
    let mut max_r = 0.0_f32;
    let mut max_g = 0.0_f32;
    let mut max_b = 0.0_f32;
    for i in 0..(W * H) {
        let dr = (r_plane[i] - r_out[i]).abs();
        let dg = (g_plane[i] - g_out[i]).abs();
        let db = (b_plane[i] - b_out[i]).abs();
        sum_r += dr as f64;
        sum_g += dg as f64;
        sum_b_acc += db as f64;
        max_r = max_r.max(dr);
        max_g = max_g.max(dg);
        max_b = max_b.max(db);
    }
    let n = (W * H) as f64;
    println!(
        "RGB MAE: R={:.4e} G={:.4e} B={:.4e}",
        sum_r / n,
        sum_g / n,
        sum_b_acc / n
    );
    println!("RGB max: R={:.4e} G={:.4e} B={:.4e}", max_r, max_g, max_b);
    for v in r_out.iter().chain(&g_out).chain(&b_out) {
        assert!(v.is_finite(), "non-finite pixel: {v}");
    }
    assert!(sum_r / n < 0.05, "R MAE: {:.3e}", sum_r / n);
    assert!(sum_g / n < 0.05, "G MAE: {:.3e}", sum_g / n);
    assert!(sum_b_acc / n < 0.15, "B MAE: {:.3e}", sum_b_acc / n);

    println!("\n✓ Full GPU lossy DCT8 roundtrip via persistent API:");
    println!(
        "  upload_plane × 3 → XYB → gaborish × 3 → gather × 3 →\n  DCT × 3 → quantize × 3 → dequant → restore_dc × 3 (GPU) →\n  IDCT × 3 → scatter × 3 → XYB inverse → download_plane × 3"
    );
    println!(
        "  ALL stages run on-GPU end-to-end. Only host-↔-GPU transfers\n  are 3 input uploads + 3 output downloads at the pipeline edges.\n  Mid-pipeline data never leaves the GPU."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
