//! Persistent-buffer GPU pipeline benchmark.
//!
//! Demonstrates the win pattern that the round-trip API can't show:
//! upload input ONCE, run a multi-stage pipeline on GPU, download
//! ONCE at the end. The intermediate stages keep their buffers
//! resident on GPU — no PCIe roundtrip per stage.
//!
//! Pipeline (mirrors real encoder front-end):
//!   linear-RGB → XYB (3 channels) → gaborish_5x5 (3 channels) →
//!   mask1x1 (Y only)
//!
//! Compares:
//! - **GPU persistent**: 1 upload (3×n f32) + 5 launches + 1 download
//!   (1×n f32 mask). All intermediates stay on-GPU.
//! - **GPU round-trip** (via GpuEncoder facade): every stage uploads
//!   AND downloads — the worst case.
//! - **CPU**: dispatched-SIMD via jxl_encoder_simd.
//!
//! Expected: persistent GPU should be much closer to (or beat) CPU
//! at moderate sizes; round-trip GPU stays slow.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use cubecl::Runtime;
    use cubecl::prelude::*;
    use jxl_encoder_gpu::encoder::GpuEncoder;
    use jxl_encoder_gpu::launch::gaborish::gaborish_5x5;
    use jxl_encoder_gpu::launch::mask1x1::mask1x1;
    use jxl_encoder_gpu::launch::xyb::xyb_forward;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let device = <Backend as Runtime>::Device::default();
    let client = <Backend as Runtime>::client(&device);

    let sizes: Vec<usize> = std::env::var("SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![256, 512, 1024, 2048, 4096]);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    // Gaborish weights for mul=1.0 (matches forks::gaborish).
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
    let wc = norm as f32;
    let wr = (norm * K_GABORISH[0]) as f32;
    let wd = (norm * K_GABORISH[1]) as f32;
    let w_big_r = (norm * K_GABORISH[2]) as f32;
    let wl = (norm * K_GABORISH[3]) as f32;
    let w_big_d = (norm * K_GABORISH[4]) as f32;

    println!("=== persistent-buffer pipeline: XYB + gaborish + mask1x1 ===");
    println!("Iters per size: {iters} (1 warmup + {} sampled)\n", iters - 1);
    println!(
        "{:>6}  {:>9}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
        "side", "MP", "CPU ms", "GPU rt ms", "GPU pers ms", "vs CPU", "vs rt"
    );
    println!("{}", "─".repeat(70));

    for &side in &sizes {
        let n = side * side;
        let mp = n as f64 / 1e6;

        // Synthetic deterministic input.
        let r: Vec<f32> = (0..n)
            .map(|i| 0.1 + 0.6 * ((i.wrapping_mul(2654435761) & 0xFFFF) as f32 / 65535.0))
            .collect();
        let g: Vec<f32> = (0..n)
            .map(|i| 0.2 + 0.5 * ((i.wrapping_mul(0x9E3779B9) & 0xFFFF) as f32 / 65535.0))
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| 0.3 + 0.4 * ((i.wrapping_mul(0xBF58476D) & 0xFFFF) as f32 / 65535.0))
            .collect();

        // ── CPU baseline ───────────────────────────────────────────
        let mut cx = vec![0.0_f32; n];
        let mut cy = vec![0.0_f32; n];
        let mut cb = vec![0.0_f32; n];
        let mut sx = vec![0.0_f32; n];
        let mut sy = vec![0.0_f32; n];
        let mut sb = vec![0.0_f32; n];
        let mut mask = vec![0.0_f32; n];
        let mut cpu_times = Vec::with_capacity(iters);
        for i in 0..iters {
            let t = std::time::Instant::now();
            jxl_encoder_simd::linear_rgb_to_xyb_batch(&r, &g, &b, &mut cx, &mut cy, &mut cb);
            jxl_encoder_simd::gaborish_5x5_channel(
                &mut cx, &mut sx, side, side, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            jxl_encoder_simd::gaborish_5x5_channel(
                &mut cy, &mut sy, side, side, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            jxl_encoder_simd::gaborish_5x5_channel(
                &mut cb, &mut sb, side, side, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            jxl_encoder_simd::compute_mask1x1(&cy, side, side, &mut mask);
            let dt = t.elapsed();
            if i > 0 {
                cpu_times.push(dt);
            }
        }
        cpu_times.sort();
        let cpu_med = cpu_times[cpu_times.len() / 2];

        // ── GPU round-trip (via GpuEncoder facade) ─────────────────
        let mut rt_times = Vec::with_capacity(iters);
        for i in 0..iters {
            let t = std::time::Instant::now();
            let (xyb_x, mut xyb_y, xyb_b) = enc.xyb_from_linear_rgb(&r, &g, &b);
            let _gx = enc.gaborish_5x5_channel(
                &xyb_x, side as u32, side as u32, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            let gy = enc.gaborish_5x5_channel(
                &xyb_y, side as u32, side as u32, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            let _gb = enc.gaborish_5x5_channel(
                &xyb_b, side as u32, side as u32, wc, wr, wd, w_big_r, wl, w_big_d,
            );
            let _ = enc.mask1x1_field(&gy, side as u32, side as u32);
            // Use xyb_y to suppress unused-mut warning under iteration
            xyb_y[0] = xyb_y[0];
            let dt = t.elapsed();
            if i > 0 {
                rt_times.push(dt);
            }
        }
        rt_times.sort();
        let rt_med = rt_times[rt_times.len() / 2];

        // ── GPU persistent (handles never leave GPU until final read) ──
        let mut pers_times = Vec::with_capacity(iters);
        for i in 0..iters {
            let t = std::time::Instant::now();
            // Upload R, G, B once
            let h_r = client.create_from_slice(f32::as_bytes(&r));
            let h_g = client.create_from_slice(f32::as_bytes(&g));
            let h_b = client.create_from_slice(f32::as_bytes(&b));
            // Allocate output handles
            let h_x = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            let h_y = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            let h_bo = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            let h_gx = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            let h_gy = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            let h_gb = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            let h_mask = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

            // Stage 1: XYB
            xyb_forward::<Backend>(
                &client,
                h_r,
                h_g,
                h_b,
                h_x.clone(),
                h_y.clone(),
                h_bo.clone(),
                n as u32,
            );
            // Stage 2: gaborish on each channel
            gaborish_5x5::<Backend>(
                &client,
                h_x.clone(),
                h_gx,
                side as u32,
                side as u32,
                wc,
                wr,
                wd,
                w_big_r,
                wl,
                w_big_d,
            );
            gaborish_5x5::<Backend>(
                &client,
                h_y.clone(),
                h_gy.clone(),
                side as u32,
                side as u32,
                wc,
                wr,
                wd,
                w_big_r,
                wl,
                w_big_d,
            );
            gaborish_5x5::<Backend>(
                &client,
                h_bo.clone(),
                h_gb,
                side as u32,
                side as u32,
                wc,
                wr,
                wd,
                w_big_r,
                wl,
                w_big_d,
            );
            // Stage 3: mask1x1 on the (gaborished) Y channel
            mask1x1::<Backend>(
                &client,
                h_gy,
                h_mask.clone(),
                side as u32,
                side as u32,
            );
            // Final download (just the mask)
            let _ = client.read_one(h_mask).expect("read mask");
            let dt = t.elapsed();
            if i > 0 {
                pers_times.push(dt);
            }
        }
        pers_times.sort();
        let pers_med = pers_times[pers_times.len() / 2];

        let cpu_ms = cpu_med.as_secs_f64() * 1000.0;
        let rt_ms = rt_med.as_secs_f64() * 1000.0;
        let pers_ms = pers_med.as_secs_f64() * 1000.0;
        let vs_cpu = cpu_med.as_secs_f64() / pers_med.as_secs_f64();
        let vs_rt = rt_med.as_secs_f64() / pers_med.as_secs_f64();
        println!(
            "{:>6}  {:>9.3}  {:>10.3}  {:>10.3}  {:>10.3}  {:>6.2}×  {:>6.2}×",
            side, mp, cpu_ms, rt_ms, pers_ms, vs_cpu, vs_rt
        );
    }
    println!(
        "\n  vs CPU: persistent-GPU vs CPU AVX2 (>1.0 = GPU wins)\n  vs rt:  persistent-GPU vs round-trip-GPU API (always >1)"
    );
    println!(
        "\nThis is the pattern future GpuEncoder work needs to expose: persistent\n  Handle<R>-typed buffers + chained launches + single download at pipeline end."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
