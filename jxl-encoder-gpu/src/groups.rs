// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JXL group geometry + per-group partitioning of whole-image data.
//!
//! JXL frames are tiled into 256×256-pixel **groups** for entropy
//! coding. Each group is independent: its tokenized output, ANS
//! tables, and bitstream emit can run on a separate CPU thread without
//! coordinating with other groups. This is libjxl's primary
//! parallelism axis on the bitstream side.
//!
//! ## Why this module exists
//!
//! Today the GPU pipeline produces whole-image output:
//! - [`StrategySearchPlan`] (XYB planes, DC grids, all assignments)
//! - per-block aq_field
//! - per-strategy quantized AC coefficients (one batch per strategy)
//!
//! The future bitstream handoff into `jxl-encoder` will want this
//! data **per group**, so the CPU consumer can kick off entropy
//! coding for group N while the GPU is still finishing group N+1.
//! This module provides the partitioning primitives — the conversion
//! from whole-image to per-group slices — without any GPU pipeline
//! changes.
//!
//! ## What's safe to slice at group boundaries
//!
//! Group size = 256 px = **32×32 blocks** of 8×8.
//! Largest selectable transform = DCT64×64 = **8×8 blocks**.
//! 32 % 8 = 0, so no transform spans a group boundary. Slicing is
//! lossless.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use jxl_encoder_gpu::groups::GroupGeometry;
//!
//! let geom = GroupGeometry::for_padded(1024, 1024);
//! assert_eq!(geom.num_groups, 16); // 4 × 4 grid
//!
//! for g in geom.iter() {
//!     // Slice per-block aq_field for this group
//!     let aq_slice = geom.gather_per_block(&aq_field, g);
//!     // Slice strategy assignments for this group
//!     let asn_slice = geom.partition_assignments(&assignments, g);
//!     // …feed into future bitstream handoff…
//! }
//! ```

use alloc::vec::Vec;
extern crate alloc;

use crate::pipeline::StrategyAssignment;

/// Group dimension in pixels (libjxl's `kGroupDim`). Frames are tiled
/// into 256×256 groups for entropy coding.
pub const GROUP_DIM: u32 = 256;

/// Group dimension in 8×8 blocks. `GROUP_DIM / 8 = 32`.
pub const GROUP_DIM_BLOCKS_8: u32 = GROUP_DIM / 8;

/// Group dimension in 64×64 blocks. `GROUP_DIM / 64 = 4`. Used to
/// verify no transform crosses a group boundary.
pub const GROUP_DIM_BLOCKS_64: u32 = GROUP_DIM / 64;

/// Geometry of a frame's group tiling. Constructed from padded image
/// dimensions; supplies group-count math + per-group bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupGeometry {
    /// Padded image width in pixels (multiple of 16 per
    /// `LossyEncoder::new`'s alignment rule).
    pub padded_width: u32,
    /// Padded image height in pixels.
    pub padded_height: u32,
    /// `ceil(padded_width / GROUP_DIM)`. Group columns.
    pub xsize_groups: u32,
    /// `ceil(padded_height / GROUP_DIM)`. Group rows.
    pub ysize_groups: u32,
}

impl GroupGeometry {
    /// Build from padded image dims. Padded dims SHOULD be multiples
    /// of 16 (matching `LossyEncoder::new`'s alignment); the geometry
    /// itself works for any multiple of 8.
    pub fn for_padded(padded_width: u32, padded_height: u32) -> Self {
        Self {
            padded_width,
            padded_height,
            xsize_groups: padded_width.div_ceil(GROUP_DIM),
            ysize_groups: padded_height.div_ceil(GROUP_DIM),
        }
    }

    /// Total group count in raster order.
    #[inline]
    pub fn num_groups(&self) -> u32 {
        self.xsize_groups * self.ysize_groups
    }

    /// Total padded 8×8-block count for the whole image.
    #[inline]
    pub fn num_padded_blocks_8(&self) -> usize {
        ((self.padded_width / 8) * (self.padded_height / 8)) as usize
    }

    /// Bounds for a group by raster index.
    pub fn group_bounds(&self, group_idx: u32) -> GroupBounds {
        let gx = group_idx % self.xsize_groups;
        let gy = group_idx / self.xsize_groups;
        self.group_bounds_xy(gx, gy)
    }

    /// Bounds for a group by (gx, gy).
    pub fn group_bounds_xy(&self, gx: u32, gy: u32) -> GroupBounds {
        debug_assert!(gx < self.xsize_groups, "gx out of range");
        debug_assert!(gy < self.ysize_groups, "gy out of range");
        let px_x = gx * GROUP_DIM;
        let px_y = gy * GROUP_DIM;
        let px_w = (self.padded_width - px_x).min(GROUP_DIM);
        let px_h = (self.padded_height - px_y).min(GROUP_DIM);
        let bx_start = px_x / 8;
        let by_start = px_y / 8;
        let bx_end = bx_start + px_w.div_ceil(8);
        let by_end = by_start + px_h.div_ceil(8);
        GroupBounds {
            group_x: gx,
            group_y: gy,
            pixel_x: px_x,
            pixel_y: px_y,
            pixel_w: px_w,
            pixel_h: px_h,
            block_x_start: bx_start,
            block_y_start: by_start,
            block_x_end: bx_end,
            block_y_end: by_end,
        }
    }

    /// Iterate all groups in raster order.
    pub fn iter(&self) -> GroupIter<'_> {
        GroupIter {
            geom: self,
            i: 0,
            n: self.num_groups(),
        }
    }

    /// Gather a per-padded-8×8-block field into a per-group slice.
    /// The returned `Vec<f32>` is `bounds.num_blocks() * 1` long, in
    /// row-major order over the group's block extent.
    pub fn gather_per_block_f32(&self, field: &[f32], bounds: GroupBounds) -> Vec<f32> {
        debug_assert_eq!(field.len(), self.num_padded_blocks_8());
        let xsize_blocks_8 = self.padded_width / 8;
        let mut out = Vec::with_capacity(bounds.num_blocks());
        for by in bounds.block_y_start..bounds.block_y_end {
            let row_off = (by * xsize_blocks_8) as usize;
            let row = &field[row_off + bounds.block_x_start as usize
                ..row_off + bounds.block_x_end as usize];
            out.extend_from_slice(row);
        }
        out
    }

    /// Strategy-assignment partitioning. Returns the subset of
    /// `assignments` whose upper-left coord falls inside the group.
    /// Order is preserved (raster order if input was).
    ///
    /// **Invariant**: no assignment is split across groups, because
    /// the largest transform (DCT64×64 = 8×8 blocks) divides 32×32
    /// blocks (one group) evenly. We assert this in debug builds.
    pub fn partition_assignments(
        &self,
        assignments: &[StrategyAssignment],
        bounds: GroupBounds,
    ) -> Vec<StrategyAssignment> {
        let mut out = Vec::new();
        for a in assignments {
            let bx = a.bx as u32;
            let by = a.by as u32;
            if bx >= bounds.block_x_start
                && bx < bounds.block_x_end
                && by >= bounds.block_y_start
                && by < bounds.block_y_end
            {
                // Sanity check: the assignment doesn't extend past
                // the group boundary. `coverage_blocks` counts the
                // 8x8 sub-block extent of the strategy.
                #[cfg(debug_assertions)]
                {
                    let (cx, cy) = strategy_coverage_blocks(a.raw_strategy);
                    debug_assert!(
                        bx + cx <= bounds.block_x_end,
                        "assignment at ({bx},{by}) strat={} would cross x boundary",
                        a.raw_strategy
                    );
                    debug_assert!(
                        by + cy <= bounds.block_y_end,
                        "assignment at ({bx},{by}) strat={} would cross y boundary",
                        a.raw_strategy
                    );
                }
                out.push(*a);
            }
        }
        out
    }

    /// Compute the assignment count per group, in one pass over the
    /// whole-image assignment list. Returns a `Vec<u32>` of length
    /// `num_groups()` with raster ordering.
    pub fn assignment_counts_per_group(&self, assignments: &[StrategyAssignment]) -> Vec<u32> {
        let mut counts = alloc::vec![0u32; self.num_groups() as usize];
        for a in assignments {
            let bx = a.bx as u32;
            let by = a.by as u32;
            let gx = bx / GROUP_DIM_BLOCKS_8;
            let gy = by / GROUP_DIM_BLOCKS_8;
            let g = gy * self.xsize_groups + gx;
            counts[g as usize] += 1;
        }
        counts
    }
}

/// Bounds of a single group: pixel-space coords, block-space coords,
/// and raster (gx, gy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupBounds {
    pub group_x: u32,
    pub group_y: u32,
    /// Group's upper-left pixel coordinate (multiple of GROUP_DIM).
    pub pixel_x: u32,
    pub pixel_y: u32,
    /// Group's pixel width (≤ GROUP_DIM, less at right edge).
    pub pixel_w: u32,
    pub pixel_h: u32,
    /// Group's 8×8-block extent (in padded-block coords).
    pub block_x_start: u32,
    pub block_y_start: u32,
    pub block_x_end: u32,
    pub block_y_end: u32,
}

impl GroupBounds {
    /// Number of 8×8 padded blocks in this group.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        ((self.block_x_end - self.block_x_start)
            * (self.block_y_end - self.block_y_start)) as usize
    }

    /// Number of pixels in this group (padded — equals `pixel_w *
    /// pixel_h`).
    #[inline]
    pub fn num_pixels(&self) -> usize {
        (self.pixel_w * self.pixel_h) as usize
    }
}

/// Iterator over a frame's groups in raster order.
pub struct GroupIter<'a> {
    geom: &'a GroupGeometry,
    i: u32,
    n: u32,
}

impl Iterator for GroupIter<'_> {
    type Item = GroupBounds;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.n {
            return None;
        }
        let b = self.geom.group_bounds(self.i);
        self.i += 1;
        Some(b)
    }
}

/// Strategy → (covered_blocks_x, covered_blocks_y) in 8×8 units.
/// Used in debug assertions to verify no assignment crosses a group
/// boundary.
#[cfg(debug_assertions)]
fn strategy_coverage_blocks(raw_strategy: u8) -> (u32, u32) {
    use crate::forks::transform::{
        RAW_STRATEGY_DCT, RAW_STRATEGY_DCT16X16, RAW_STRATEGY_DCT16X32, RAW_STRATEGY_DCT16X8,
        RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64,
        RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8, RAW_STRATEGY_DCT64X32, RAW_STRATEGY_DCT64X64,
        RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT8X4, RAW_STRATEGY_IDENTITY,
    };
    use crate::forks::transform::{
        RAW_STRATEGY_AFV0, RAW_STRATEGY_AFV1, RAW_STRATEGY_AFV2, RAW_STRATEGY_AFV3,
    };
    match raw_strategy {
        RAW_STRATEGY_DCT
        | RAW_STRATEGY_DCT4X4
        | RAW_STRATEGY_DCT4X8
        | RAW_STRATEGY_DCT8X4
        | RAW_STRATEGY_IDENTITY
        | RAW_STRATEGY_DCT2X2
        | RAW_STRATEGY_AFV0
        | RAW_STRATEGY_AFV1
        | RAW_STRATEGY_AFV2
        | RAW_STRATEGY_AFV3 => (1, 1),
        RAW_STRATEGY_DCT16X8 => (1, 2),
        RAW_STRATEGY_DCT8X16 => (2, 1),
        RAW_STRATEGY_DCT16X16 => (2, 2),
        RAW_STRATEGY_DCT32X16 => (2, 4),
        RAW_STRATEGY_DCT16X32 => (4, 2),
        RAW_STRATEGY_DCT32X32 => (4, 4),
        RAW_STRATEGY_DCT64X32 => (4, 8),
        RAW_STRATEGY_DCT32X64 => (8, 4),
        RAW_STRATEGY_DCT64X64 => (8, 8),
        _ => (1, 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_geometry_1024_aligned() {
        let g = GroupGeometry::for_padded(1024, 1024);
        assert_eq!(g.xsize_groups, 4);
        assert_eq!(g.ysize_groups, 4);
        assert_eq!(g.num_groups(), 16);
        assert_eq!(g.num_padded_blocks_8(), 128 * 128);
    }

    #[test]
    fn test_group_geometry_misaligned() {
        // 1056×800 padded — NOT a multiple of 256. 1056/256 = 4.125,
        // so xsize_groups = 5; the rightmost group is 32px wide.
        let g = GroupGeometry::for_padded(1056, 800);
        assert_eq!(g.xsize_groups, 5);
        assert_eq!(g.ysize_groups, 4);
        let last = g.group_bounds_xy(4, 3);
        assert_eq!(last.pixel_x, 1024);
        assert_eq!(last.pixel_w, 32, "rightmost group is the 32px tail");
        assert_eq!(last.pixel_h, 32, "bottom-rightmost group is 32×32");
        assert_eq!(last.block_x_start, 128);
        assert_eq!(last.block_x_end, 132);
    }

    #[test]
    fn test_iter_covers_all_groups() {
        let g = GroupGeometry::for_padded(512, 512);
        let bounds: Vec<_> = g.iter().collect();
        assert_eq!(bounds.len(), 4); // 2×2
        // Every pixel of the padded frame must be covered by exactly
        // one group.
        let mut hits = alloc::vec![0u32; (512 * 512) as usize];
        for b in &bounds {
            for py in b.pixel_y..b.pixel_y + b.pixel_h {
                for px in b.pixel_x..b.pixel_x + b.pixel_w {
                    hits[(py * 512 + px) as usize] += 1;
                }
            }
        }
        assert!(hits.iter().all(|&h| h == 1), "every pixel covered exactly once");
    }

    #[test]
    fn test_gather_per_block_f32() {
        // 256×256 padded → 32×32 = 1024 blocks → exactly 1 group of 32×32 blocks
        let g = GroupGeometry::for_padded(256, 256);
        let field: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let bounds = g.group_bounds(0);
        let slice = g.gather_per_block_f32(&field, bounds);
        assert_eq!(slice.len(), 1024);
        assert_eq!(slice[0], 0.0);
        assert_eq!(slice[1023], 1023.0);
    }

    #[test]
    fn test_gather_per_block_f32_corner() {
        // 512×256: 2 groups horizontally. Group 1 (right) has blocks
        // at x∈[32..64), y∈[0..32). field[0..1024] is group 0, [1024..2048] is group 1.
        let g = GroupGeometry::for_padded(512, 256);
        let xb = 64;  // 512/8
        let mut field = alloc::vec![0.0_f32; xb * 32];
        // Mark group 1's first row distinctively
        for x in 32..64 {
            field[x] = 100.0 + x as f32;
        }
        let bounds = g.group_bounds_xy(1, 0);
        let slice = g.gather_per_block_f32(&field, bounds);
        assert_eq!(slice.len(), 32 * 32);
        // First row of group 1 = x=32..63 of the source
        for x in 0..32 {
            assert_eq!(slice[x], 100.0 + (32 + x) as f32);
        }
    }

    #[test]
    fn test_partition_assignments_basic() {
        // 512×512 padded → 2×2 = 4 groups
        let g = GroupGeometry::for_padded(512, 512);
        let assignments = alloc::vec![
            // Group (0,0): bx=0, by=0
            StrategyAssignment { bx: 0, by: 0, raw_strategy: 0 },
            // Group (1,0): bx=32, by=0
            StrategyAssignment { bx: 32, by: 0, raw_strategy: 0 },
            // Group (0,1): bx=0, by=32
            StrategyAssignment { bx: 0, by: 32, raw_strategy: 0 },
            // Group (1,1): bx=63, by=63
            StrategyAssignment { bx: 63, by: 63, raw_strategy: 0 },
        ];
        let counts = g.assignment_counts_per_group(&assignments);
        assert_eq!(counts, alloc::vec![1u32, 1, 1, 1]);
        let p00 = g.partition_assignments(&assignments, g.group_bounds(0));
        assert_eq!(p00.len(), 1);
        assert_eq!(p00[0].bx, 0);
        let p11 = g.partition_assignments(&assignments, g.group_bounds(3));
        assert_eq!(p11.len(), 1);
        assert_eq!(p11[0].bx, 63);
    }

    #[test]
    fn test_partition_assignments_with_dct64() {
        // 512×512 → 2×2 groups, each 32×32 blocks. DCT64x64 covers
        // 8×8 blocks. Place a DCT64 at block (0,0) — fits inside
        // group (0,0). And one at (32,32) — fits inside group (1,1).
        let g = GroupGeometry::for_padded(512, 512);
        let dct64 = crate::forks::transform::RAW_STRATEGY_DCT64X64;
        let assignments = alloc::vec![
            StrategyAssignment { bx: 0, by: 0, raw_strategy: dct64 },
            StrategyAssignment { bx: 32, by: 32, raw_strategy: dct64 },
        ];
        let p00 = g.partition_assignments(&assignments, g.group_bounds(0));
        let p11 = g.partition_assignments(&assignments, g.group_bounds(3));
        assert_eq!(p00.len(), 1);
        assert_eq!(p11.len(), 1);
        assert_eq!(p00[0].bx, 0);
        assert_eq!(p11[0].bx, 32);
    }
}
