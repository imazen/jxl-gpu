// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Dump GPU XYB output ranges for comparison with the CPU encoder's
//! `EncoderPrecomputed::compute(...)` output. Used to debug the
//! bitstream-decode failure in
//! `encode_lossy_to_bitstream_via_precomputed`.

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
}

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;
    type B = cubecl::cuda::CudaRuntime;

    let width = 64u32;
    let height = 64u32;
    let n = (width as usize) * (height as usize);
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for y in 0..height {
        for x in 0..width {
            r.push((x as f32 / width as f32) * 0.5);
            g.push((y as f32 / height as f32) * 0.5);
            b.push(0.25);
        }
    }
    let enc: GpuEncoder<B> = GpuEncoder::new();
    let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, width, height);
    let plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, 1.0);
    let xyb_x = enc.download_plane(&plan.xyb_x_gpu);
    let xyb_y = enc.download_plane(&plan.xyb_y_gpu);
    let xyb_b = enc.download_plane(&plan.xyb_b_gpu);
    println!("GPU xyb_x[0..6]: {:?}", &xyb_x[..6]);
    println!("GPU xyb_y[0..6]: {:?}", &xyb_y[..6]);
    println!("GPU xyb_b[0..6]: {:?}", &xyb_b[..6]);
    let mn_x = xyb_x.iter().copied().fold(f32::INFINITY, f32::min);
    let mx_x = xyb_x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mn_y = xyb_y.iter().copied().fold(f32::INFINITY, f32::min);
    let mx_y = xyb_y.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mn_b = xyb_b.iter().copied().fold(f32::INFINITY, f32::min);
    let mx_b = xyb_b.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!("GPU xyb_x range: [{}, {}]", mn_x, mx_x);
    println!("GPU xyb_y range: [{}, {}]", mn_y, mx_y);
    println!("GPU xyb_b range: [{}, {}]", mn_b, mx_b);
    let (pw, ph) = lossy.padded_dimensions();
    println!("GPU padded {}x{}", pw, ph);
}
