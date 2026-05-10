// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Probe kernel for `Array<u8>` support in cubecl 0.10.
//!
//! Single tiny kernel: read u8 input, cast to f32, write to f32
//! output. If this compiles and runs the round-trip, the u8-input
//! upload optimization for the encode/recon hot path is tractable.
//!
//! Used only by the unit test in this file. Not part of the
//! production pipeline.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn u8_to_f32_kernel(input: &Array<u8>, output: &mut Array<f32>) {
    let idx = ABSOLUTE_POS;
    if idx >= output.len() {
        terminate!();
    }
    output[idx] = input[idx] as f32;
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use cubecl::prelude::*;
    use cubecl::server::Handle;

    type B = cubecl::cuda::CudaRuntime;

    /// Verify cubecl 0.10 + CUDA backend can launch a kernel that
    /// takes `Array<u8>` as input. If this passes, the u8-input
    /// upload optimization for the encode/recon hot path is unblocked.
    #[test]
    fn test_u8_to_f32_roundtrip() {
        let device = <B as cubecl::Runtime>::Device::default();
        let client = <B as cubecl::Runtime>::client(&device);

        let n = 256_usize;
        let input: alloc::vec::Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(7)).collect();
        let h_in: Handle = client.create_from_slice(&input);
        let h_out: Handle = client.empty(n * 4);

        const TPB: u32 = 256;
        unsafe {
            u8_to_f32_kernel::launch_unchecked::<B>(
                &client,
                CubeCount::Static((n as u32).div_ceil(TPB).max(1), 1, 1),
                CubeDim::new_1d(TPB),
                ArrayArg::from_raw_parts(h_in, n),
                ArrayArg::from_raw_parts(h_out.clone(), n),
            );
        }

        let bytes = client.read_one(h_out).expect("read output");
        let out_floats: &[f32] = f32::from_bytes(&bytes);
        assert_eq!(out_floats.len(), n);
        for i in 0..n {
            let expected = input[i] as f32;
            assert_eq!(out_floats[i], expected, "idx {i}: expected {expected}, got {}", out_floats[i]);
        }
    }
}
