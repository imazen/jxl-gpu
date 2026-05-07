//! Corpus-level AQ benefit measurement: runs uniform vs content-
//! driven AQ over many images at multiple distances and reports
//! aggregate SSIMULACRA2 statistics.
//!
//! For each image in the corpus directory, encodes at the full
//! distance grid two ways (uniform / AQ-centered) and accumulates
//! per-distance ssim2 deltas. Reports mean Δssim2, win count, and
//! worst-case regression per distance.
//!
//! This is the validation harness for the AQ tradeoff: the
//! single-image quality_sweep_with_aq_demo shows what happens on one
//! photo; this shows whether the win holds across content or is
//! image-specific.
//!
//! Usage:
//!   cargo run --release --features cuda --example corpus_aq_sweep_demo
//!
//! Optional env vars:
//!   CORPUS_DIR   Directory of PNGs (default: codec-corpus/clic2025-1024)
//!   MAX_IMAGES   Cap on images to process (default: 8)

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::{LossyEncoder, distance_to_qac};

    type Backend = cubecl::cuda::CudaRuntime;
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

    let distances = [0.5_f32, 1.0, 2.0, 4.0, 8.0];
    let qacs_uniform: Vec<f32> = distances.iter().copied().map(distance_to_qac).collect();

    println!("=== corpus_aq_sweep_demo ===");
    println!(
        "Corpus: {corpus_dir}\nImages: {}\nDistances: {:?}\n",
        paths.len(),
        distances
    );

    // Per-distance accumulators across the corpus.
    let nd = distances.len();
    let mut sum_dssim2 = vec![0.0_f64; nd];
    let mut sum_un = vec![0.0_f64; nd];
    let mut sum_aq = vec![0.0_f64; nd];
    let mut wins = vec![0_usize; nd];
    let mut losses = vec![0_usize; nd];
    let mut worst_loss: Vec<(f64, String)> = vec![(0.0, String::new()); nd];

    for (img_idx, path) in paths.iter().enumerate() {
        let img = match image::open(path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                eprintln!("[skip] {path:?}: {e}");
                continue;
            }
        };
        let (w, h) = img.dimensions();
        let rgb_in: Vec<u8> = img.into_raw();
        let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);

        let out_un = lossy.encode_many_srgb_u8(&enc, &rgb_in, &qacs_uniform);
        let out_aq = lossy.encode_many_with_aq_srgb_u8(&enc, &rgb_in, &distances);

        let to_rgb3 = |buf: &[u8]| -> Vec<[u8; 3]> {
            buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
        };
        let src = to_rgb3(&rgb_in);
        let src_img = imgref::ImgVec::new(src, w as usize, h as usize);

        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        print!("[{:>2}/{}] {:>15.15}  ", img_idx + 1, paths.len(), name);

        for (di, _) in distances.iter().enumerate() {
            let dst_un = imgref::ImgVec::new(to_rgb3(&out_un[di]), w as usize, h as usize);
            let dst_aq = imgref::ImgVec::new(to_rgb3(&out_aq[di]), w as usize, h as usize);
            let s_un = fast_ssim2::compute_ssimulacra2(src_img.as_ref(), dst_un.as_ref())
                .expect("ssim2 un") as f64;
            let s_aq = fast_ssim2::compute_ssimulacra2(src_img.as_ref(), dst_aq.as_ref())
                .expect("ssim2 aq") as f64;
            let delta = s_aq - s_un;
            sum_dssim2[di] += delta;
            sum_un[di] += s_un;
            sum_aq[di] += s_aq;
            if delta > 0.05 {
                wins[di] += 1;
            } else if delta < -0.05 {
                losses[di] += 1;
                if delta < worst_loss[di].0 {
                    worst_loss[di] = (delta, name.to_string());
                }
            }
            print!(" {:>+5.2}", delta);
        }
        println!();
    }

    let nf = paths.len() as f64;
    println!("\n=== Aggregate (n={}) ===", paths.len());
    println!(
        "  {:>5}  {:>10}  {:>10}  {:>10}  {:>5}  {:>5}  {:>22}",
        "dist", "uniform µ", "AQ µ", "Δssim2 µ", "wins", "loss", "worst-loss image (Δ)"
    );
    for (di, d) in distances.iter().enumerate() {
        let (worst_d, worst_name) = &worst_loss[di];
        let worst_str = if losses[di] > 0 {
            format!(
                "{:>15.15} ({:+.2})",
                worst_name.chars().take(15).collect::<String>(),
                worst_d
            )
        } else {
            String::from("(none)")
        };
        println!(
            "  {:>5.2}  {:>10.2}  {:>10.2}  {:>+10.2}  {:>5}  {:>5}  {:>22}",
            d,
            sum_un[di] / nf,
            sum_aq[di] / nf,
            sum_dssim2[di] / nf,
            wins[di],
            losses[di],
            worst_str,
        );
    }
    println!(
        "\nwin = Δssim2 > +0.05, loss = Δssim2 < -0.05, tie otherwise.\nAQ should win on most images at d>=2.0; near-tie at d=0.5 is\nexpected (less heavy-quant headroom to redistribute)."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
