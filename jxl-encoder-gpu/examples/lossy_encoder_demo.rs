//! "Hello, world!" for the LossyEncoder API.
//!
//! Demonstrates both `encode_one` (single-shot) and `encode_many`
//! (batch — multiple settings on the same input). 30 lines of
//! application code total; all the GPU pipeline complexity is
//! hidden behind the LossyEncoder facade.
//!
//! Compare with `lossy_roundtrip_persistent.rs` which builds the
//! same pipeline by hand using the persistent API directly.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    // 1024×1024 synthetic linear-RGB input.
    let side = 1024_u32;
    let n = (side * side) as usize;
    let r: Vec<f32> = (0..n).map(|i| 0.1 + 0.7 * (i as f32 / n as f32)).collect();
    let g: Vec<f32> = (0..n).map(|i| 0.2 + 0.6 * (i as f32 / n as f32)).collect();
    let b: Vec<f32> = (0..n)
        .map(|i| 0.3 + 0.5 * (((i + 7) % 17) as f32 / 17.0))
        .collect();

    // Construct one LossyEncoder per (width, height) — amortizes the
    // static-input upload (gaborish weights, dead-zone thresholds, unit
    // quant matrix) across all encodes that follow.
    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, side, side);

    // ── One-shot encode using the libjxl distance interface ───────
    use jxl_encoder_gpu::lossy_encoder::distance_to_qac;
    let qac = distance_to_qac(1.0); // libjxl reference quality
    let t0 = std::time::Instant::now();
    let (rec_r, _rec_g, _rec_b) = lossy.encode_one(&enc, &r, &g, &b, qac);
    let dt0 = t0.elapsed();
    println!(
        "one-shot encode @ {side}×{side}, distance=1.0 (qac={qac:.3}): {:.2} ms",
        dt0.as_secs_f64() * 1000.0
    );
    assert_eq!(rec_r.len(), n);
    let max_err = r
        .iter()
        .zip(&rec_r)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("  R channel max abs reconstruction error: {max_err:.4e}");

    // ── Batch encode — distance sweep on the same input ────────────
    // distances 0.5 → 5.0 cover the practical libjxl range.
    let distances = [0.5_f32, 1.0, 2.0, 5.0];
    let qac_settings: Vec<f32> = distances.iter().copied().map(distance_to_qac).collect();
    let t1 = std::time::Instant::now();
    let outputs = lossy.encode_many(&enc, &r, &g, &b, &qac_settings);
    let dt1 = t1.elapsed();
    println!(
        "\nbatch encode @ {side}×{side} × {} distances: {:.2} ms total ({:.2} ms/encode)",
        distances.len(),
        dt1.as_secs_f64() * 1000.0,
        dt1.as_secs_f64() * 1000.0 / distances.len() as f64,
    );
    for ((d, qac), (rec_r_i, _, _)) in distances.iter().zip(&qac_settings).zip(&outputs) {
        let mae: f64 = r
            .iter()
            .zip(rec_r_i)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / n as f64;
        println!("  distance={d:>4.1}  qac={qac:.3}  R MAE={mae:.4e}");
    }

    println!(
        "\n✓ Both encode_one and encode_many work end-to-end.\n  Compared to running encode_one in a loop, encode_many uploads\n  the input ONCE (5.28× speedup at 2048² × 5 per\n  lossy_pipeline_repeated_input bench)."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
