// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! High-level GPU lossy encoder facade.
//!
//! Wraps the persistent API into a single-call `encode_one` (one-shot
//! image lossy roundtrip on GPU) and `encode_many` (batch — same image,
//! multiple quality settings, with one-time input upload).
//!
//! ## What it does today
//!
//! Runs the canonical lossy DCT8 pipeline end-to-end on GPU:
//!
//!   linear-RGB → XYB → gaborish ×3 → gather ×3 →
//!   DCT8 wide ×3 → quantize ×3 → dequant → DC-restore → IDCT8 ×3 →
//!   scatter ×3 → XYB inverse → linear-RGB
//!
//! This is the same pipeline lossy_pipeline_throughput benchmarks
//! against CPU — currently 1.05-3.95× faster than CPU AVX2 at sizes
//! 256² → 2048² (RTX 5070 + Ryzen 9 7950X).
//!
//! ## What it does NOT do
//!
//! - Does not produce JXL bitstream bytes (no entropy coding /
//!   container yet — this is a roundtrip path for measuring quality
//!   + algorithmic correctness, not a complete encoder).
//! - Does not implement DC quant + entropy coding (DC restore is a
//!   passthrough; real encoders use jxl-encoder's dc_coding).
//! - DCT8 only (one strategy). The 13 strategies are available
//!   individually via [`crate::persistent`] but a strategy-search +
//!   per-block dispatch isn't yet wrapped here.
//!
//! For full bitstream encoding today, use
//! [`crate::encoder::GpuEncoder::encode_lossy_via_cpu`] which delegates
//! to jxl-encoder.

use alloc::vec;
use alloc::vec::Vec;

use cubecl::Runtime;

use crate::encoder::GpuEncoder;
use crate::persistent::{GaborishWeights, GpuBlocks, GpuPlane};

/// libjxl gaborish K_GABORISH constants for mul=1.0. Matches
/// `forks::gaborish::compute_weights(1.0)` bit-for-bit.
const K_GABORISH: [f64; 5] = [
    -0.094_958_15_67,
    -0.041_031_725,
    0.013_710_005,
    0.006_510_206,
    -0.001_478_906_3,
];

fn default_gaborish_weights() -> GaborishWeights {
    let sum_w = 1.0
        + 4.0
            * (K_GABORISH[0] + K_GABORISH[1] + K_GABORISH[2] + K_GABORISH[4] + 2.0 * K_GABORISH[3]);
    let norm = 1.0 / sum_w;
    GaborishWeights {
        wc: norm as f32,
        wr: (norm * K_GABORISH[0]) as f32,
        wd: (norm * K_GABORISH[1]) as f32,
        w_big_r: (norm * K_GABORISH[2]) as f32,
        wl: (norm * K_GABORISH[3]) as f32,
        w_big_d: (norm * K_GABORISH[4]) as f32,
    }
}

/// High-level lossy DCT8 encoder for a fixed image size. Pre-allocates
/// the static inputs (gaborish weights, dead-zone thresholds, unit
/// quant matrix) once at construction so they don't re-upload per call.
///
/// Construct one per (width, height) to amortize the static-input
/// upload across many encodes. Both width and height must be multiples
/// of 8 (DCT8 block size).
pub struct LossyEncoder<R: Runtime> {
    width: u32,
    height: u32,
    num_blocks: u32,
    weights_g: GpuBlocks<R>,
    weights: GaborishWeights,
    thresholds: [f32; 4],
}

impl<R: Runtime> LossyEncoder<R> {
    /// Construct a lossy encoder for a given image size.
    /// Width and height must be multiples of 8.
    pub fn new(enc: &GpuEncoder<R>, width: u32, height: u32) -> Self {
        assert!(width % 8 == 0 && height % 8 == 0, "width/height must be multiples of 8");
        let num_blocks = (width / 8) * (height / 8);
        // Unit weights — one float per coefficient. Real encoder would
        // upload per-strategy quant matrix; for the demo path unit
        // weights × per-block scale suffices.
        let weights_g = enc.upload_blocks(
            &vec![1.0_f32; (num_blocks as usize) * 64],
            num_blocks,
            64,
        );
        Self {
            width,
            height,
            num_blocks,
            weights_g,
            weights: default_gaborish_weights(),
            thresholds: [0.56, 0.62, 0.62, 0.62],
        }
    }

    /// Run the full lossy pipeline on RGB input. Returns reconstructed
    /// RGB (linear, planar). `qac_qm` is the per-block quantize scale
    /// (broadcast to all blocks); larger = more aggressive quant.
    ///
    /// Single-shot: input uploaded, pipeline run, output downloaded.
    /// For batch workloads, use [`Self::encode_many`].
    pub fn encode_one(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        qac_qm: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let g_r = enc.upload_plane(r, self.width, self.height);
        let g_g = enc.upload_plane(g, self.width, self.height);
        let g_b = enc.upload_plane(b, self.width, self.height);
        self.run_pipeline(enc, &g_r, &g_g, &g_b, qac_qm)
    }

    /// Run the full pipeline at multiple `qac_qm` settings on the same
    /// input. Input uploaded ONCE; all subsequent iterations re-use the
    /// uploaded handles.
    ///
    /// Per `lossy_pipeline_repeated_input`, this is ~5× faster than
    /// calling [`Self::encode_one`] in a loop at 2048² × 5 settings.
    pub fn encode_many(
        &self,
        enc: &GpuEncoder<R>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        qac_settings: &[f32],
    ) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let g_r = enc.upload_plane(r, self.width, self.height);
        let g_g = enc.upload_plane(g, self.width, self.height);
        let g_b = enc.upload_plane(b, self.width, self.height);
        qac_settings
            .iter()
            .map(|&qac_qm| self.run_pipeline(enc, &g_r, &g_g, &g_b, qac_qm))
            .collect()
    }

    /// Internal pipeline body. Operates on pre-uploaded GPU planes;
    /// downloads the reconstructed RGB at the end.
    fn run_pipeline(
        &self,
        enc: &GpuEncoder<R>,
        g_r: &GpuPlane<R>,
        g_g: &GpuPlane<R>,
        g_b: &GpuPlane<R>,
        qac_qm: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let qac_vec = vec![qac_qm; self.num_blocks as usize];
        let xf = vec![0.0_f32; self.num_blocks as usize];
        let bf = vec![0.0_f32; self.num_blocks as usize];

        let (xx, xy, xb) = enc.xyb_from_linear_rgb_persistent(g_r, g_g, g_b);
        let xx_g = enc.gaborish_5x5_persistent(&xx, &self.weights);
        let xy_g = enc.gaborish_5x5_persistent(&xy, &self.weights);
        let xb_g = enc.gaborish_5x5_persistent(&xb, &self.weights);
        let bx_g = enc.gather_blocks_persistent(&xx_g, 8, 8);
        let by_g = enc.gather_blocks_persistent(&xy_g, 8, 8);
        let bb_g = enc.gather_blocks_persistent(&xb_g, 8, 8);
        let coeffs_x = enc.dct_8x8_wide_persistent(&bx_g);
        let coeffs_y = enc.dct_8x8_wide_persistent(&by_g);
        let coeffs_b = enc.dct_8x8_wide_persistent(&bb_g);
        let q_x =
            enc.quantize_dct8_persistent(&coeffs_x, &self.weights_g, &qac_vec, &self.thresholds);
        let q_y =
            enc.quantize_dct8_persistent(&coeffs_y, &self.weights_g, &qac_vec, &self.thresholds);
        let q_b =
            enc.quantize_dct8_persistent(&coeffs_b, &self.weights_g, &qac_vec, &self.thresholds);
        let (dq_x, dq_y, dq_b) = enc.dequant_dct8_persistent(
            &q_x,
            &q_y,
            &q_b,
            &self.weights_g,
            &self.weights_g,
            &self.weights_g,
            &qac_vec,
            &qac_vec,
            &qac_vec,
            &xf,
            &bf,
        );
        enc.restore_dc_persistent(&coeffs_x, &dq_x);
        enc.restore_dc_persistent(&coeffs_y, &dq_y);
        enc.restore_dc_persistent(&coeffs_b, &dq_b);
        let recon_x_b = enc.idct_8x8_wide_persistent(&dq_x);
        let recon_y_b = enc.idct_8x8_wide_persistent(&dq_y);
        let recon_b_b = enc.idct_8x8_wide_persistent(&dq_b);
        let recon_x_p = enc.scatter_blocks_persistent(&recon_x_b, self.width, self.height, 8, 8);
        let recon_y_p = enc.scatter_blocks_persistent(&recon_y_b, self.width, self.height, 8, 8);
        let recon_b_p = enc.scatter_blocks_persistent(&recon_b_b, self.width, self.height, 8, 8);
        let (rgb_r, rgb_g, rgb_b) =
            enc.xyb_to_linear_rgb_planar_persistent(&recon_x_p, &recon_y_p, &recon_b_p);
        (
            enc.download_plane(&rgb_r),
            enc.download_plane(&rgb_g),
            enc.download_plane(&rgb_b),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_one_shot() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy = LossyEncoder::new(&enc, 64, 64);
        let n = 64 * 64;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        let (rr, gg, bb) = lossy.encode_one(&enc, &r, &g, &b, 4.0);
        assert_eq!(rr.len(), n);
        assert_eq!(gg.len(), n);
        assert_eq!(bb.len(), n);
        for v in rr.iter().chain(&gg).chain(&bb) {
            assert!(v.is_finite());
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lossy_encoder_many_settings() {
        type B = cubecl::cuda::CudaRuntime;
        let enc: GpuEncoder<B> = GpuEncoder::new();
        let lossy = LossyEncoder::new(&enc, 32, 32);
        let n = 32 * 32;
        let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.6 * (i as f32 / n as f32)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.5 * (i as f32 / n as f32)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.3 + 0.4 * (i as f32 / n as f32)).collect();
        let qacs = [1.0_f32, 2.0, 4.0, 8.0];
        let outputs = lossy.encode_many(&enc, &r, &g, &b, &qacs);
        assert_eq!(outputs.len(), 4);
        for (i, (rr, _, _)) in outputs.iter().enumerate() {
            assert_eq!(rr.len(), n, "output {i} wrong length");
        }
    }
}
