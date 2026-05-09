//! Inspect mask1x1 statistics on diverse content to evaluate whether
//! mean/variance can serve as a discriminator for "screenshot-like
//! content where strat-search regresses" vs "photo content where
//! strat-search is at parity".
//!
//! Usage: `IMAGE_PATH=foo.png cargo run --release --features 'cuda
//! encoder' --example mask1x1_content_stats`

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let image_path = std::env::var("IMAGE_PATH").unwrap_or_else(|_| {
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
            .to_string()
    });

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("failed to open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let pixels: Vec<u8> = img.into_raw();
    let n = (w * h) as usize;

    let to_linear = |c: u8| -> f32 {
        let f = c as f32 / 255.0;
        if f <= 0.04045 {
            f / 12.92
        } else {
            ((f + 0.055) / 1.055).powf(2.4)
        }
    };
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for chunk in pixels.chunks_exact(3) {
        r.push(to_linear(chunk[0]));
        g.push(to_linear(chunk[1]));
        b.push(to_linear(chunk[2]));
    }

    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);
    let block_means = lossy.compute_block_mask_means(&enc, &r, &g, &b);

    let nb = block_means.len();
    let mean = block_means.iter().sum::<f32>() / nb as f32;
    let mut sorted = block_means.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[nb / 2];
    let p10 = sorted[nb / 10];
    let p90 = sorted[(nb * 9) / 10];
    let min = sorted[0];
    let max = sorted[nb - 1];
    let var: f32 = block_means.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / nb as f32;
    let stddev = var.sqrt();
    let cv = stddev / mean;

    let basename = std::path::Path::new(&image_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    println!(
        "{basename:<40} {w}x{h}  nb={nb}  mean={mean:.4}  median={median:.4}  p10={p10:.4}  p90={p90:.4}  min={min:.4}  max={max:.4}  stddev={stddev:.4}  cv={cv:.3}"
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
