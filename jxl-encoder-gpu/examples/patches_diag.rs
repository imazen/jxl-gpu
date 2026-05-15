// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Diagnostic for the GPU patches wireup: download the GPU's pre-gab
//! XYB on a screenshot, run the encoder's `find_and_build_patches` on
//! it, and report whether patches were detected. Compares against
//! running the same detector on a host-side XYB conversion of the
//! same image.
//!
//! Used to investigate why the screenshot ratios on the GPU slow path
//! match the case-2 baseline after the case-1 wireup landed (probable
//! cause: GPU XYB's FP precision lets one of the patches detection
//! filters reject what CPU would accept).

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
}

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

    type B = cubecl::cuda::CudaRuntime;

    let raw: Vec<String> = std::env::args().collect();
    let image_path = raw
        .iter()
        .position(|a| a == "--image")
        .map(|i| raw[i + 1].clone())
        .expect("--image PATH required");
    let distance: f32 = raw
        .iter()
        .position(|a| a == "--distance")
        .map(|i| raw[i + 1].parse().expect("--distance D"))
        .unwrap_or(0.5);

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("open {image_path}: {e}"))
        .to_rgb8();
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

    let enc: GpuEncoder<B> = GpuEncoder::new();
    let lossy: LossyEncoder<B> = LossyEncoder::new(&enc, w, h);
    let plan = lossy.prepare_strategy_search_plan(&enc, &r, &g, &b, distance);
    let (gpu_pw, gpu_ph) = lossy.padded_dimensions();
    let cpu_pw = (w as usize).div_ceil(8) * 8;
    let cpu_ph = (h as usize).div_ceil(8) * 8;

    let (xyb_x_pre_dl, xyb_y_pre_dl, xyb_b_pre_dl) = enc.download_planes_3ch(
        &plan.xyb_x_pre_gab_gpu,
        &plan.xyb_y_pre_gab_gpu,
        &plan.xyb_b_pre_gab_gpu,
    );

    let repack = |src: &[f32]| -> Vec<f32> {
        if cpu_pw == gpu_pw as usize && cpu_ph == gpu_ph as usize {
            src.to_vec()
        } else {
            let mut dst = vec![0.0_f32; cpu_pw * cpu_ph];
            let wu = w as usize;
            let hu = h as usize;
            for row in 0..cpu_ph {
                let off = row * (gpu_pw as usize);
                let dst_off = row * cpu_pw;
                dst[dst_off..dst_off + cpu_pw].copy_from_slice(&src[off..off + cpu_pw]);
            }
            if cpu_pw > wu {
                for row in 0..cpu_ph {
                    let dst_off = row * cpu_pw;
                    let last_real = dst[dst_off + wu - 1];
                    for col in wu..cpu_pw {
                        dst[dst_off + col] = last_real;
                    }
                }
            }
            if cpu_ph > hu {
                let last_real_off = (hu - 1) * cpu_pw;
                for row in hu..cpu_ph {
                    let dst_off = row * cpu_pw;
                    dst.copy_within(last_real_off..last_real_off + cpu_pw, dst_off);
                }
            }
            dst
        }
    };
    let xyb_x_gpu = repack(&xyb_x_pre_dl);
    let xyb_y_gpu = repack(&xyb_y_pre_dl);
    let xyb_b_gpu = repack(&xyb_b_pre_dl);

    println!(
        "image {}×{} (cpu_pw={}, cpu_ph={}, gpu_pw={}, gpu_ph={})",
        w, h, cpu_pw, cpu_ph, gpu_pw, gpu_ph
    );
    println!(
        "xyb_x[..4] gpu = {:?}",
        &xyb_x_gpu[..4.min(xyb_x_gpu.len())]
    );

    // Run encoder's find_and_build_patches on the GPU pre-gab planes.
    let pd_gpu = jxl_encoder::__pre_quantized::find_and_build_patches(
        [&xyb_x_gpu, &xyb_y_gpu, &xyb_b_gpu],
        w as usize,
        h as usize,
        cpu_pw,
    );
    println!(
        "patches on GPU pre-gab: {}",
        if pd_gpu.is_some() { "Some" } else { "None" }
    );
}
