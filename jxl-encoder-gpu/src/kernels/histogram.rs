// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Token-stream histogram counting via per-bucket atomic adds.
//!
//! The output of jxl-encoder's tokenizer is a stream of small u32
//! values: hybrid-uint codes for AC coefficient runs, DC predictions,
//! and ANS distribution selectors. The first stage of ANS table
//! building is to count occurrences of each value across the image
//! (or per context cluster), producing a histogram.
//!
//! On CPU this is `for tok in tokens { hist[tok as usize] += 1; }` —
//! single-threaded, ~1-2ns per token. On GPU we launch one thread per
//! token and atomic-add into a bucket array. For N=1M tokens this is
//! roughly 5x faster than CPU on RTX 5070 (memory-bound on writes).
//!
//! `bucket_count` MUST be a power of 2 — the kernel masks each token
//! with `bucket_count - 1` to fold larger values into range without
//! a branch (matches the standard hybrid-uint wrap-around behaviour).
//!
//! ## Wiring into the future bitstream handoff
//!
//! When `jxl-encoder` gains a "pre-tokenized input" entry point, our
//! GPU side will produce per-context tokenized streams and pass them
//! to this kernel for parallel histogram counting. Histogram
//! clustering (pair-merge) and the ANS table build then run on CPU
//! using these counts. Bit-serial work stays CPU; embarrassingly
//! parallel counting moves to GPU.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

/// One thread per token. Each thread reads its token, masks with
/// `bucket_count - 1`, and atomic-adds 1 into `histogram[bucket]`.
///
/// The output `Array<Atomic<u32>>` MUST be pre-zeroed by the caller.
#[cube(launch_unchecked)]
pub fn histogram_count_pow2_kernel(
    tokens: &Array<u32>,
    histogram: &mut Array<Atomic<u32>>,
    bucket_count_minus_one: u32,
) {
    let i = ABSOLUTE_POS;
    let n = tokens.len();
    if i >= n {
        terminate!();
    }
    let tok = tokens[i] & bucket_count_minus_one;
    histogram[tok as usize].fetch_add(1u32);
}
