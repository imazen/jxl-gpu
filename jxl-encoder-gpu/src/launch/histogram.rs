// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Launcher for the GPU histogram-counting kernel.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::histogram::histogram_count_pow2_kernel;

/// Launch the histogram-count kernel with one thread per token. The
/// `histogram` buffer must be pre-zeroed and sized for at least
/// `bucket_count` u32 entries (treated by the kernel as
/// `Atomic<u32>`).
///
/// `bucket_count` MUST be a power of 2; the kernel masks tokens with
/// `bucket_count - 1`. Callers wanting non-power-of-2 alphabets should
/// round up to the next power of 2 (the extra buckets just stay 0).
pub fn histogram_count_pow2<R: Runtime>(
    client: &ComputeClient<R>,
    tokens: Handle,
    histogram: Handle,
    n_tokens: u32,
    bucket_count: u32,
) {
    assert!(
        bucket_count.is_power_of_two(),
        "bucket_count must be a power of 2 (was {bucket_count})"
    );
    let n = n_tokens as usize;
    if n == 0 {
        return;
    }
    let cubes = n_tokens.div_ceil(256).max(1);
    unsafe {
        histogram_count_pow2_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(tokens, n),
            ArrayArg::from_raw_parts(histogram, bucket_count as usize),
            bucket_count - 1,
        );
    }
}
