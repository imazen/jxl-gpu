// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Diagnostic for the e8/e9 butteraugli loop: per-iter measured
//! distance distribution, score, and tile_dist histogram.

#[cfg(not(feature = "butteraugli-loop"))]
fn main() {
    eprintln!("buttloop_trace requires --features butteraugli-loop");
}

#[cfg(feature = "butteraugli-loop")]
fn main() {
    use cubecl::cuda::CudaRuntime as Backend;
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::forks::butteraugli_loop::{
        ButteraugliLoopGpu, refine_aq_field_gpu_with_strategy_search_persistent,
    };
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

    let mut args = std::env::args().skip(1);
    let mut image_path: Option<String> = None;
    let mut distance: f32 = 1.0;
    let mut iters: usize = 4;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--image" => image_path = args.next(),
            "--distance" => distance = args.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(4),
            _ => {}
        }
    }
    let image_path = image_path.expect("--image PATH required");
    let img = image::open(&image_path).expect("decode").to_rgb8();
    let (w, h) = img.dimensions();
    let pixels_u8: Vec<u8> = img.into_raw();
    let n = (w as usize) * (h as usize);
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
    for chunk in pixels_u8.chunks_exact(3) {
        r.push(to_linear(chunk[0]));
        g.push(to_linear(chunk[1]));
        b.push(to_linear(chunk[2]));
    }

    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);
    let mut bg: ButteraugliLoopGpu<Backend> = ButteraugliLoopGpu::new_multires(&enc, w, h);
    let _ = bg.set_reference(&pixels_u8);

    let plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
    let initial_aq = plan.quant_field_float.clone();

    println!(
        "buttloop_trace: {}x{} ({:.2} MP), distance={}, iters={}",
        w,
        h,
        (w as f32 * h as f32) / 1e6,
        distance,
        iters
    );

    let mut iter_idx = 0;
    let trace = |t: jxl_encoder_gpu::forks::butteraugli_loop::RefineIterTrace| {
        let n = t.tile_dist.len() as f32;
        let mut hist = [0u32; 5];
        let mut sum = 0.0f32;
        let mut mx = 0.0f32;
        let mut mn = f32::INFINITY;
        for &d in &t.tile_dist {
            let r = d / distance;
            sum += r;
            if r > mx {
                mx = r;
            }
            if r < mn {
                mn = r;
            }
            let bin = if r < 0.5 {
                0
            } else if r < 0.9 {
                1
            } else if r <= 1.0 {
                2
            } else if r <= 1.5 {
                3
            } else {
                4
            };
            hist[bin] += 1;
        }
        let avg = sum / n;
        println!(
            "iter {} score={:.3} pnorm3={:.3} | tile_dist/target: avg={:.3} min={:.3} max={:.3} | hist [<0.5:{} <0.9:{} <=1:{} <=1.5:{} >1.5:{}]",
            t.iter, t.score, t.pnorm_3, avg, mn, mx, hist[0], hist[1], hist[2], hist[3], hist[4],
        );
        let _ = iter_idx;
    };

    let _refined = refine_aq_field_gpu_with_strategy_search_persistent(
        &enc,
        &lossy,
        &mut bg,
        &r,
        &g,
        &b,
        &pixels_u8,
        &initial_aq,
        distance,
        iters,
        trace,
    )
    .expect("refine");
}
