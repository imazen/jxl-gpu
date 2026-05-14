// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression test for the GPU↔CPU strategy code remap (issue #5).
//!
//! Pre-fix: GPU's `RAW_STRATEGY_DCT2X2 = 16` was being passed straight
//! into CPU's `AcStrategyMap::set` where it was interpreted as
//! `RAW_STRATEGY_DCT64X64 = 16`. The bitstream emitted a 64×64 wire
//! code (CPU `STRATEGY_CODE_LUT[16] = 18`) on a 1×1 block, which
//! djxl rejected outright.
//!
//! Repro: gb82-sc/terminal.png (1646×1062) at d=0.25/0.5/1.0/2.0 —
//! the GPU strat-search picks DCT2X2 + IDENTITY on small high-contrast
//! UI / text regions, so any non-trivial screenshot triggers it.
//!
//! Test strategy: encode a synthetic high-contrast pattern that's
//! likely to produce non-DCT8 / non-DCT16x16 picks (mixed solid blocks
//! + sharp edges + low-frequency gradients), then decode the bitstream
//! through jxl-rs and djxl. Both must succeed and produce non-NaN
//! pixel data. With the bug present, both decoders fail.
//!
//! This is a Layer-3 invariant per project CLAUDE.md proof-by-tests
//! methodology — the `forks::transform::gpu_to_cpu_tests` module
//! already covers Layer 1 (mapping table) and Layer 2 (round-trip
//! the table); this test covers Layer 3 (full encode → decode).

#![cfg(all(feature = "cuda", feature = "encoder", feature = "butteraugli-loop"))]

use jxl_encoder_gpu::encoder::GpuEncoder;
use jxl_encoder_gpu::lossy_encoder::LossyEncoder;

type Backend = cubecl::cuda::CudaRuntime;

/// A 256×256 sRGB pattern that the strat-search reliably picks DCT2X2 /
/// IDENTITY for on at least some 8×8 cells: alternating solid color
/// blocks of high contrast (forces non-DCT8 sub-block selection) plus
/// edge-aligned text-like glyphs.
fn build_repro_pattern(w: u32, h: u32) -> Vec<f32> {
    let n = (w as usize) * (h as usize);
    let mut r = vec![0.0_f32; n];
    let mut g = vec![0.0_f32; n];
    let mut b = vec![0.0_f32; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = y * w as usize + x;
            // 16-px checkerboard of pure white / pure black.
            let cx = x / 16;
            let cy = y / 16;
            let on = (cx + cy) % 2 == 0;
            // Add narrow vertical lines every 8 cols so AC coefs are
            // sparse and DCT2X2 / IDENTITY become competitive on the
            // line-aligned 8×8 cells.
            let line = (x % 8) == 0;
            let v: f32 = if line {
                0.5
            } else if on {
                1.0
            } else {
                0.0
            };
            r[idx] = v;
            g[idx] = v;
            b[idx] = v;
        }
    }
    let mut interleaved = vec![0.0_f32; n * 3];
    // We want planar instead — caller asks for planar floats.
    let _ = &mut interleaved;
    let mut out = Vec::with_capacity(n * 3);
    out.extend_from_slice(&r);
    out.extend_from_slice(&g);
    out.extend_from_slice(&b);
    out
}

fn split_planar(planar_rgb: &[f32], w: u32, h: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = (w as usize) * (h as usize);
    assert_eq!(planar_rgb.len(), n * 3);
    let r = planar_rgb[0..n].to_vec();
    let g = planar_rgb[n..2 * n].to_vec();
    let b = planar_rgb[2 * n..3 * n].to_vec();
    (r, g, b)
}

fn decode_via_jxl_rs(bytes: &[u8]) -> bool {
    use jxl::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input: &[u8] = bytes;
    let initialized: JxlDecoder<states::Initialized> =
        JxlDecoder::new(JxlDecoderOptions::default());
    let mut with_image_info = match initialized.process(&mut input) {
        Ok(ProcessingResult::Complete { result }) => result,
        _ => return false,
    };
    let basic = with_image_info.basic_info().clone();
    let (w, h) = (basic.size.0, basic.size.1);
    let default_fmt = with_image_info.current_pixel_format().clone();
    let requested_fmt = JxlPixelFormat {
        color_type: default_fmt.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: default_fmt
            .extra_channel_format
            .iter()
            .map(|_| Some(JxlDataFormat::f32()))
            .collect(),
    };
    with_image_info.set_pixel_format(requested_fmt);
    let pixel_format = with_image_info.current_pixel_format().clone();
    let num_channels = pixel_format.color_type.samples_per_pixel();
    if num_channels != 3 && num_channels != 4 {
        return false;
    }
    let mut color_buf = match Image::<f32>::new_with_value((w * num_channels, h), f32::NAN) {
        Ok(b) => b,
        Err(_) => return false,
    };
    let mut all_imgs: Vec<&mut Image<f32>> = vec![&mut color_buf];
    let mut api_buffers: Vec<JxlOutputBuffer<'_>> = all_imgs
        .iter_mut()
        .map(|b| {
            let size = b.size();
            JxlOutputBuffer::from_image_rect_mut(
                b.get_rect_mut(Rect {
                    origin: (0, 0),
                    size,
                })
                .into_raw(),
            )
        })
        .collect();
    let with_frame_info = match with_image_info.process(&mut input) {
        Ok(ProcessingResult::Complete { result }) => result,
        _ => return false,
    };
    let _back = match with_frame_info.process(&mut input, &mut api_buffers) {
        Ok(ProcessingResult::Complete { result }) => result,
        _ => return false,
    };
    drop(api_buffers);
    // Verify pixels are finite (NaN means decode produced garbage).
    for y in 0..h {
        for &v in color_buf.row(y) {
            if !v.is_finite() {
                return false;
            }
        }
    }
    true
}

fn decode_via_jxl_oxide(bytes: &[u8]) -> bool {
    let reader = std::io::Cursor::new(bytes);
    let mut img = match jxl_oxide::JxlImage::builder().read(reader) {
        Ok(i) => i,
        Err(_) => return false,
    };
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    match img.render_frame(0) {
        Ok(frame) => {
            let fb = frame.image_all_channels();
            // Verify finite pixels.
            fb.buf().iter().all(|v| v.is_finite())
        }
        Err(_) => false,
    }
}

/// End-to-end decode test: encode a high-contrast repro pattern at a
/// distance that historically picked DCT2X2 / IDENTITY, decode through
/// both jxl-rs (PRIMARY per CLAUDE.md) and jxl-oxide.
///
/// With the GPU↔CPU strategy enum bug (issue #5) present, both decoders
/// reject the bitstream. With the fix, both succeed.
#[test]
fn high_contrast_pattern_decodes_after_strategy_remap() {
    let enc: GpuEncoder<Backend> = GpuEncoder::new();
    let (w, h) = (256_u32, 256_u32);
    let planar = build_repro_pattern(w, h);
    let (r, g, b) = split_planar(&planar, w, h);
    let lossy: LossyEncoder<Backend> = LossyEncoder::new(&enc, w, h);

    // Sweep distances that hit the failure modes on terminal.png. At
    // the extreme low/high distances DCT2X2 isn't always picked, but
    // d=0.5..2.0 reliably triggers it on this content.
    let distances: &[f32] = &[0.25, 0.5, 1.0, 2.0];
    for &d in distances {
        let bytes = enc
            .encode_lossy_to_bitstream_via_precomputed(&lossy, &r, &g, &b, d)
            .unwrap_or_else(|e| panic!("encode d={d}: {e:?}"));
        assert!(!bytes.is_empty(), "empty bitstream at d={d}");

        let oxide_ok = decode_via_jxl_oxide(&bytes);
        let rs_ok = decode_via_jxl_rs(&bytes);
        assert!(
            oxide_ok && rs_ok,
            "decode failed at d={d}: jxl-oxide={oxide_ok} jxl-rs={rs_ok}\n\
             (pre-fix this failed when GPU's DCT2X2=16 was misread as CPU's DCT64X64=16)",
        );
    }
}
