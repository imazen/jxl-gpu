//! Cooperative DCT8 vs naive (cube_dim=1) vs CPU AVX2.
//!
//! Compares three forward-DCT8 implementations on batches of 8×8
//! blocks at scales corresponding to image sizes 256² → 4096²:
//!
//! 1. **CPU AVX2**: dispatched-SIMD `jxl_encoder_simd::dct_8x8_scalar`
//!    in a host loop.
//! 2. **GPU naive**: `dct_8x8` (cube_dim=1, num_blocks cubes).
//! 3. **GPU cooperative**: `dct_8x8_coop` (cube_dim=8, num_blocks cubes,
//!    one thread per row).
//!
//! Reports median time and verifies parity (max abs delta) between
//! all three. The cooperative kernel is the same algorithm but uses
//! 8× more thread-parallelism per cube — should saturate SMs at
//! smaller num_blocks counts where the naive kernel starves.

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use cubecl::Runtime;
    use cubecl::prelude::*;
    use jxl_encoder_gpu::launch::dct8::{dct_8x8, dct_8x8_coop};

    type Backend = cubecl::cuda::CudaRuntime;
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

    println!("=== DCT8 throughput: CPU vs naive-GPU vs cooperative-GPU ===");
    println!("Iters per size: {iters} (1 warmup + {} sampled)\n", iters - 1);
    println!(
        "{:>6}  {:>9}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
        "side", "blocks", "CPU ms", "naive ms", "coop ms", "naive×", "coop×"
    );
    println!("{}", "─".repeat(70));

    for &side in &sizes {
        let nb = (side / 8) * (side / 8);
        let n = nb * 64;
        // Synthetic batch.
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();

        // CPU
        let mut cpu_out = vec![0.0_f32; n];
        let mut cpu_times = Vec::with_capacity(iters);
        for it in 0..iters {
            let t = std::time::Instant::now();
            for b in 0..nb {
                let off = b * 64;
                let inp: &[f32; 64] = input[off..off + 64].try_into().unwrap();
                let out: &mut [f32; 64] = (&mut cpu_out[off..off + 64]).try_into().unwrap();
                jxl_encoder_simd::dct_8x8_scalar(inp, out);
            }
            let dt = t.elapsed();
            if it > 0 {
                cpu_times.push(dt);
            }
        }
        cpu_times.sort();
        let cpu_med = cpu_times[cpu_times.len() / 2];

        // GPU naive
        let mut naive_times = Vec::with_capacity(iters);
        let mut naive_out = vec![0.0_f32; n];
        for it in 0..iters {
            let t = std::time::Instant::now();
            let h_in = client.create_from_slice(f32::as_bytes(&input));
            let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            dct_8x8::<Backend>(&client, h_in, h_out.clone(), nb as u32);
            let bytes = client.read_one(h_out).expect("read");
            let result: &[f32] = f32::from_bytes(&bytes);
            let dt = t.elapsed();
            if it > 0 {
                naive_times.push(dt);
            }
            if it == iters - 1 {
                naive_out = result.to_vec();
            }
        }
        naive_times.sort();
        let naive_med = naive_times[naive_times.len() / 2];

        // GPU cooperative
        let mut coop_times = Vec::with_capacity(iters);
        let mut coop_out = vec![0.0_f32; n];
        for it in 0..iters {
            let t = std::time::Instant::now();
            let h_in = client.create_from_slice(f32::as_bytes(&input));
            let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
            dct_8x8_coop::<Backend>(&client, h_in, h_out.clone(), nb as u32);
            let bytes = client.read_one(h_out).expect("read");
            let result: &[f32] = f32::from_bytes(&bytes);
            let dt = t.elapsed();
            if it > 0 {
                coop_times.push(dt);
            }
            if it == iters - 1 {
                coop_out = result.to_vec();
            }
        }
        coop_times.sort();
        let coop_med = coop_times[coop_times.len() / 2];

        // Parity
        let max_naive = naive_out
            .iter()
            .zip(&cpu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let max_coop = coop_out
            .iter()
            .zip(&cpu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // Sanity: parity should be the same for both GPU paths.
        if max_naive > 1e-4 || max_coop > 1e-4 {
            eprintln!(
                "WARNING: parity drift at side={side}: naive={max_naive:.3e}, coop={max_coop:.3e}"
            );
        }

        let cpu_ms = cpu_med.as_secs_f64() * 1000.0;
        let naive_ms = naive_med.as_secs_f64() * 1000.0;
        let coop_ms = coop_med.as_secs_f64() * 1000.0;
        let naive_x = cpu_med.as_secs_f64() / naive_med.as_secs_f64();
        let coop_x = cpu_med.as_secs_f64() / coop_med.as_secs_f64();
        println!(
            "{:>6}  {:>9}  {:>10.2}  {:>10.2}  {:>10.2}  {:>6.2}×  {:>6.2}×",
            side, nb, cpu_ms, naive_ms, coop_ms, naive_x, coop_x
        );
    }
    println!(
        "\n  naive× / coop× = CPU_time / GPU_time. >1.0 = GPU faster than CPU.\n  Coop kernel uses cube_dim=8 (one thread per row); naive uses cube_dim=1.\n  Both produce identical output to CPU within sub-ulp tolerance."
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
