//! Real-image roundtrip: PNG → encode (CPU via GpuEncoder dep) →
//! decode via jxl-rs → compare.
//!
//! Validates that the JXL bytes our encoder produces are decodable by
//! jxl-rs (a fully-Rust independent implementation), not just by the
//! libjxl reference. This is the "pure Rust" half of the roundtrip
//! verification matrix; `real_image_encode` validates with djxl.
//!
//! Intentionally simple: decodes the first frame to a planar RGB f32
//! buffer (jxl-rs's natural pixel format), scans for NaN, prints
//! basic stats. Doesn't compare against the original PNG pixel-by-
//! pixel — the encoder is lossy at d=1 so a perceptual metric would
//! be needed for that, and the goal here is "decoder didn't reject
//! our bytes."

#[cfg(all(feature = "cuda", feature = "encoder"))]
fn main() {
    use jxl::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};
    use jxl_encoder::api::{LossyConfig, PixelLayout};
    use jxl_encoder_gpu::encoder::GpuEncoder;

    type Backend = cubecl::cuda::CudaRuntime;
    let enc: GpuEncoder<Backend> = GpuEncoder::new();

    let image_path = std::env::var("IMAGE_PATH").unwrap_or_else(|_| {
        "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
            .to_string()
    });
    let distance: f32 = std::env::var("DISTANCE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.0);
    let effort: u8 = std::env::var("EFFORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    println!("=== jxl-rs roundtrip test ===");
    println!("Image:    {image_path}");
    println!("Distance: {distance}");
    println!("Effort:   {effort}\n");

    let img = image::open(&image_path)
        .unwrap_or_else(|e| panic!("failed to open {image_path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let pixels: Vec<u8> = img.into_raw();
    println!("Loaded {w}×{h} RGB8");

    // ── Encode ─────────────────────────────────────────────────────
    let config = LossyConfig::new(distance).with_effort(effort);
    let t0 = std::time::Instant::now();
    let bytes = enc
        .encode_lossy_via_cpu(&config, &pixels, w, h, PixelLayout::Rgb8)
        .expect("encode failed");
    let encode_dt = t0.elapsed();
    let bpp = (bytes.len() * 8) as f64 / (w as f64 * h as f64);
    println!(
        "Encoded:  {} bytes ({:.3} bpp) in {:.2}s",
        bytes.len(),
        bpp,
        encode_dt.as_secs_f64()
    );

    // ── Decode via jxl-rs ──────────────────────────────────────────
    let t1 = std::time::Instant::now();
    let mut input: &[u8] = &bytes;
    let options = JxlDecoderOptions::default();
    let initialized: JxlDecoder<states::Initialized> = JxlDecoder::new(options);

    // Step 1: process until we have image info
    let mut decoder_with_image_info = match initialized.process(&mut input).expect("decode init") {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => {
            panic!("decoder reported NeedsMoreInput on initial process — full bytes provided");
        }
    };

    let basic_info = decoder_with_image_info.basic_info().clone();
    println!(
        "jxl-rs:   parsed header — {}×{}, bit_depth={:?}",
        basic_info.size.0,
        basic_info.size.1,
        basic_info.bit_depth.bits_per_sample()
    );
    assert_eq!(basic_info.size, (w as usize, h as usize));

    // Step 2: request F32 RGB output (matches our encode input format,
    // modulo the sRGB transfer function which jxl-rs applies).
    let default_format = decoder_with_image_info.current_pixel_format().clone();
    let requested_format = JxlPixelFormat {
        color_type: default_format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: default_format
            .extra_channel_format
            .iter()
            .map(|_| Some(JxlDataFormat::f32()))
            .collect(),
    };
    decoder_with_image_info.set_pixel_format(requested_format);

    let pixel_format = decoder_with_image_info.current_pixel_format().clone();
    let num_channels = pixel_format.color_type.samples_per_pixel();
    println!("Format:   {num_channels}-channel f32");
    let buffer_w = basic_info.size.0;
    let buffer_h = basic_info.size.1;

    // Allocate the output buffer (interleaved channels, single planar f32 image).
    let mut color_buf = Image::<f32>::new_with_value(
        (buffer_w * num_channels, buffer_h),
        f32::NAN,
    )
    .expect("alloc color buf");
    let extra_buf_count = pixel_format
        .extra_channel_format
        .iter()
        .filter(|x| x.is_some())
        .count();
    let mut extra_bufs: Vec<Image<f32>> = (0..extra_buf_count)
        .map(|_| {
            Image::<f32>::new_with_value((buffer_w, buffer_h), f32::NAN).expect("alloc extra")
        })
        .collect();

    // Build the API output buffers.
    let mut all_imgs: Vec<&mut Image<f32>> =
        std::iter::once(&mut color_buf).chain(extra_bufs.iter_mut()).collect();
    let mut api_buffers: Vec<JxlOutputBuffer<'_>> = all_imgs
        .iter_mut()
        .map(|b| {
            let size = b.size();
            JxlOutputBuffer::from_image_rect_mut(
                b.get_rect_mut(Rect { origin: (0, 0), size }).into_raw(),
            )
        })
        .collect();

    // Step 3: process WithImageInfo → WithFrameInfo (advances frame parsing).
    let decoder_with_frame_info = match decoder_with_image_info
        .process(&mut input)
        .expect("decode frame info")
    {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("NeedsMoreInput at frame info"),
    };

    // Step 4: process WithFrameInfo → WithImageInfo (writes pixel data).
    let _decoder_back_to_image = match decoder_with_frame_info
        .process(&mut input, &mut api_buffers)
        .expect("decode pixels")
    {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("NeedsMoreInput at frame body"),
    };
    // Drop api_buffers to release the borrow on color_buf so we can read it.
    drop(api_buffers);
    let decode_dt = t1.elapsed();
    println!("Decoded:  in {:.2}s", decode_dt.as_secs_f64());

    // ── Verify ─────────────────────────────────────────────────────
    let (xs, ys) = color_buf.size();
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut nan_count = 0usize;
    for y in 0..ys {
        for &v in color_buf.row(y).iter() {
            if v.is_nan() {
                nan_count += 1;
            } else {
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
        }
    }
    println!(
        "Pixels:   {} samples ({}×{}), range [{min_v:.4}, {max_v:.4}], NaN count={nan_count}",
        xs * ys,
        xs,
        ys
    );
    assert_eq!(nan_count, 0, "decoder left NaN pixels in the output");
    // Note: with JxlDataFormat::f32() jxl-rs decodes to LINEAR f32 by default
    // (not sRGB), so values can slightly exceed [0, 1] due to gamut/lossy
    // recovery. We just check that values are bounded and finite.
    assert!(
        min_v >= -0.5 && max_v <= 1.5,
        "decoded f32 range wildly out of bounds: [{min_v}, {max_v}]"
    );

    println!(
        "\n✓ jxl-rs decoded our encoder's output cleanly. {w}×{h} → {} bytes → decoded in\n  {:.2}s with no NaN, finite output in linear-RGB f32.",
        bytes.len(),
        decode_dt.as_secs_f64()
    );
}

#[cfg(not(all(feature = "cuda", feature = "encoder")))]
fn main() {
    eprintln!("requires --features 'cuda encoder'");
    std::process::exit(2);
}
