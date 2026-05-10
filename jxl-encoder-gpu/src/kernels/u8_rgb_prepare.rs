// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Fused u8 sRGB RGB → planar f32 linear with edge-replication pad.
//!
//! Replaces the host pipeline `u8 → host sRGB→linear → 3× planar f32
//! → host pad_to_alignment → upload_planes_3ch` with a single GPU
//! kernel that takes interleaved u8 RGB and produces 3 padded
//! linear-f32 planes.
//!
//! Wins (estimated for 16 MP / 4909×3260):
//!   - upload bandwidth: 4× (48 MB raw u8 vs 192 MB converted f32)
//!   - skips host pad_to_alignment (~96 ms host memcpy)
//!   - skips host sRGB→linear loop (~400 ms host CPU at 16 MP for
//!     48M powf calls — though that's currently before our timed
//!     section, it's still real wall-clock for callers)
//!
//! Math matches the corpus regression test's host `to_linear`
//! (full sRGB EOTF, not pure gamma 2.2):
//!   ```text
//!   f = c / 255.0
//!   if f <= 0.04045 → linear = f / 12.92
//!   else            → linear = ((f + 0.055) / 1.055)^2.4
//!   ```

use cubecl::prelude::*;

/// One thread per output pixel. Reads interleaved RGB at the
/// clamped-to-edge source location, converts to linear f32, writes
/// into 3 separate planar output buffers.
///
/// Layout:
/// - `src`: `src_width * src_height * 3` u8 (interleaved RGB, row-major)
/// - `out_r/g/b`: `padded_width * padded_height` f32 (planar)
/// - For output (ox, oy):
///     src_x = min(ox, src_width - 1)
///     src_y = min(oy, src_height - 1)
///     src_off = (src_y * src_width + src_x) * 3
///     load src[src_off..src_off+3]
///     out_r/g/b[oy * padded_width + ox] = sRGB→linear(src bytes)
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
pub fn u8_rgb_to_linear_planar_padded_kernel(
    src: &Array<u8>,
    out_r: &mut Array<f32>,
    out_g: &mut Array<f32>,
    out_b: &mut Array<f32>,
    src_width: u32,
    src_height: u32,
    padded_width: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= out_r.len() {
        terminate!();
    }
    let pw = padded_width as usize;
    let oy = idx / pw;
    let ox = idx - oy * pw;

    // Clamp-to-edge for padded pixels outside the source rectangle.
    let sw = src_width as usize;
    let sh = src_height as usize;
    let sx = if ox < sw { ox } else { sw - 1usize };
    let sy = if oy < sh { oy } else { sh - 1usize };
    let src_off = (sy * sw + sx) * 3usize;

    let r_byte = src[src_off];
    let g_byte = src[src_off + 1usize];
    let b_byte = src[src_off + 2usize];

    // sRGB EOTF inline (cubecl 0.10 rejects typed f32 let-bindings).
    let r_norm = (r_byte as f32) * (1.0f32 / 255.0f32);
    let g_norm = (g_byte as f32) * (1.0f32 / 255.0f32);
    let b_norm = (b_byte as f32) * (1.0f32 / 255.0f32);

    let r_linear = if r_norm <= 0.04045f32 {
        r_norm * (1.0f32 / 12.92f32)
    } else {
        f32::powf((r_norm + 0.055f32) * (1.0f32 / 1.055f32), 2.4f32)
    };
    let g_linear = if g_norm <= 0.04045f32 {
        g_norm * (1.0f32 / 12.92f32)
    } else {
        f32::powf((g_norm + 0.055f32) * (1.0f32 / 1.055f32), 2.4f32)
    };
    let b_linear = if b_norm <= 0.04045f32 {
        b_norm * (1.0f32 / 12.92f32)
    } else {
        f32::powf((b_norm + 0.055f32) * (1.0f32 / 1.055f32), 2.4f32)
    };

    out_r[idx] = r_linear;
    out_g[idx] = g_linear;
    out_b[idx] = b_linear;
}
