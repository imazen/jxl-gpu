//! Phase 3 Component 2: host-side partition selector test.
//!
//! Pure host-side test (no GPU dependency). Verifies the partition
//! selector picks the lower-cost choice on synthetic cost grids.

// Test fixtures use vec![..; n] for ergonomic readability; allowed in test code.
#![allow(clippy::useless_vec)]

use jxl_encoder_gpu::pipeline::{
    CostGrids16x16, CostGrids32x32, CostGrids64x64, Partition16x16, Partition32x32, Partition64x64,
    select_partitions_16x16, select_partitions_16x16_full, select_partitions_32x32,
    select_partitions_32x32_full, select_partitions_64x64,
};

#[test]
fn picks_dct16x16_when_cheaper() {
    // 4x2 blocks @ 8x8 → 2x1 region @ 16x16
    // DCT16 cheaper than 4*DCT8 → pick DCT16
    let cost_dct8 = vec![10.0; 4 * 2]; // 4*10 = 40 for any 4-block region
    let cost_dct16x16 = vec![30.0; 2]; // < 40

    let p = select_partitions_16x16(&cost_dct8, &cost_dct16x16, 4, 2);
    assert_eq!(p.len(), 2);
    assert!(p.iter().all(|&x| x == Partition16x16::Dct16x16));
}

#[test]
fn picks_four_dct8_when_cheaper() {
    let cost_dct8 = vec![1.0; 4 * 2]; // 4*1 = 4 for any 4-block region
    let cost_dct16x16 = vec![10.0; 2]; // > 4

    let p = select_partitions_16x16(&cost_dct8, &cost_dct16x16, 4, 2);
    assert_eq!(p.len(), 2);
    assert!(p.iter().all(|&x| x == Partition16x16::FourDct8x8));
}

#[test]
fn mixed_per_region() {
    // 4x4 blocks @ 8x8 → 2x2 regions @ 16x16
    // Region (0,0): cheap DCT8s (sum=4), expensive DCT16 (10) → pick DCT8s
    // Region (1,0): expensive DCT8s (sum=40), cheap DCT16 (30) → pick DCT16
    // Region (0,1) and (1,1) tied → exact ties favor DCT16 by `<=`
    let cost_dct8 = vec![
        1.0, 1.0, 10.0, 10.0, // row 0 of 4-wide grid
        1.0, 1.0, 10.0, 10.0, // row 1
        5.0, 5.0, 5.0, 5.0, // row 2
        5.0, 5.0, 5.0, 5.0, // row 3
    ];
    let cost_dct16x16 = vec![
        10.0, 30.0, // top row
        20.0, 20.0, // bottom row
    ];

    let p = select_partitions_16x16(&cost_dct8, &cost_dct16x16, 4, 4);
    assert_eq!(p.len(), 4);
    assert_eq!(p[0], Partition16x16::FourDct8x8); // 4 < 10
    assert_eq!(p[1], Partition16x16::Dct16x16); // 30 < 40
    assert_eq!(p[2], Partition16x16::Dct16x16); // 20 <= 20 (tie → DCT16)
    assert_eq!(p[3], Partition16x16::Dct16x16);
}

#[test]
fn picks_two_dct16x8_when_cheapest() {
    // 4x4 blocks @ 8x8 → 2x2 region @ 16x16. DCT8: 4*10 = 40, DCT16x16: 50,
    // two DCT16x8 (each 16t × 8w pixels = 1w × 2t in 8x8 units): 2*10 = 20.
    // → pick TwoDct16x8 per region.
    let cost_dct8 = vec![10.0; 16];
    let cost_dct16x16 = vec![50.0; 4];
    let cost_16x8 = vec![10.0; 8]; // 4 wide x 2 tall (= ysize_blocks_8/2)
    let extra = CostGrids16x16 {
        dct_16x8: Some(&cost_16x8),
        dct_8x16: None,
    };

    let p = select_partitions_16x16_full(&cost_dct8, &cost_dct16x16, extra, 4, 4);
    assert_eq!(p.len(), 4);
    // For each region: cost_dct16=50, two_16x8=20, four_dct8=40 → TwoDct16x8 wins
    assert!(p.iter().all(|&x| x == Partition16x16::TwoDct16x8Horizontal));
}

#[test]
fn picks_two_dct8x16_when_cheapest() {
    let cost_dct8 = vec![10.0; 16];
    let cost_dct16x16 = vec![50.0; 4];
    let cost_8x16 = vec![10.0; 8]; // 2 wide (xsize_blocks_8/2) x 4 tall
    let extra = CostGrids16x16 {
        dct_16x8: None,
        dct_8x16: Some(&cost_8x16),
    };

    let p = select_partitions_16x16_full(&cost_dct8, &cost_dct16x16, extra, 4, 4);
    assert_eq!(p.len(), 4);
    assert!(p.iter().all(|&x| x == Partition16x16::TwoDct8x16Vertical));
}

#[test]
fn picks_dct64x64_when_cheapest() {
    // 8x8 blocks @ 8x8 = 64x64 = 1 region @ 64x64
    let cost_dct8 = vec![10.0; 64];
    let cost_dct16x16 = vec![30.0; 16];
    let cost_dct32x32 = vec![80.0; 4];
    let cost_dct64x64 = vec![100.0]; // < 4 * 80 = 320, < 16 * 30 = 480
    let p = select_partitions_64x64(
        &cost_dct8,
        &cost_dct16x16,
        &cost_dct32x32,
        &cost_dct64x64,
        CostGrids32x32::default(),
        CostGrids64x64::default(),
        8,
        8,
    );
    assert_eq!(p.len(), 1);
    assert_eq!(p[0], Partition64x64::Dct64x64);
}

#[test]
fn picks_two_dct64x32_horizontal() {
    let cost_dct8 = vec![10.0; 64];
    let cost_dct16x16 = vec![30.0; 16];
    let cost_dct32x32 = vec![80.0; 4];
    let cost_dct64x64 = vec![100.0];
    // 64x32: xsize=2, ysize=1 → 2 cells
    let cost_64x32 = vec![20.0; 2]; // 2*20 = 40 < 100
    let extra64 = CostGrids64x64 {
        dct_64x32: Some(&cost_64x32),
        dct_32x64: None,
    };
    let p = select_partitions_64x64(
        &cost_dct8,
        &cost_dct16x16,
        &cost_dct32x32,
        &cost_dct64x64,
        CostGrids32x32::default(),
        extra64,
        8,
        8,
    );
    assert_eq!(p.len(), 1);
    assert_eq!(p[0], Partition64x64::TwoDct64x32Horizontal);
}

#[test]
fn picks_sub_32x32_when_cheaper() {
    let cost_dct8 = vec![1.0; 64]; // each 16x16 picks 4*1=4; each 32x32 picks 4*4=16
    let cost_dct16x16 = vec![100.0; 16];
    let cost_dct32x32 = vec![100.0; 4]; // each 32x32 sub picks 4*16x16 = 16
    let cost_dct64x64 = vec![1000.0];
    let p = select_partitions_64x64(
        &cost_dct8,
        &cost_dct16x16,
        &cost_dct32x32,
        &cost_dct64x64,
        CostGrids32x32::default(),
        CostGrids64x64::default(),
        8,
        8,
    );
    assert_eq!(p.len(), 1);
    if let Partition64x64::Sub32x32(_) = p[0] {
        // pass
    } else {
        panic!("expected Sub32x32, got {:?}", p[0]);
    }
}

#[test]
fn picks_dct32x32_when_cheapest() {
    // 4x4 blocks @ 8x8 → 1x1 region @ 32x32
    // DCT32 cheaper than any sub-partition → pick DCT32
    let cost_dct8 = vec![10.0; 16];
    let cost_dct16x16 = vec![30.0; 4]; // each 16x16 sub costs min(30, 4*10=40) = 30
    let cost_dct32x32 = vec![100.0]; // < 4*30 = 120

    let p = select_partitions_32x32(&cost_dct8, &cost_dct16x16, &cost_dct32x32, 4, 4);
    assert_eq!(p.len(), 1);
    assert_eq!(p[0], Partition32x32::Dct32x32);
}

#[test]
fn picks_two_dct32x16_horizontal() {
    // 4x4 blocks @ 8x8 = 32x32 pixels = 1x1 region @ 32x32
    // DCT8: 16*1 = 16; DCT16: 4*5 = 20; DCT32: 100; TwoDct32x16: 2*2 = 4
    let cost_dct8 = vec![1.0; 16];
    let cost_dct16x16 = vec![5.0; 4];
    let cost_dct32x32 = vec![100.0];
    // 32x16 grid: xsize=2, ysize=1 → 2 cells
    let cost_32x16 = vec![2.0; 2];
    let extra = CostGrids32x32 {
        dct_32x16: Some(&cost_32x16),
        dct_16x32: None,
    };
    let p = select_partitions_32x32_full(&cost_dct8, &cost_dct16x16, &cost_dct32x32, extra, 4, 4);
    assert_eq!(p.len(), 1);
    assert_eq!(p[0], Partition32x32::TwoDct32x16Horizontal);
}

#[test]
fn picks_two_dct16x32_vertical() {
    let cost_dct8 = vec![1.0; 16];
    let cost_dct16x16 = vec![5.0; 4];
    let cost_dct32x32 = vec![100.0];
    // 16x32 grid: xsize=1, ysize=2 → 2 cells
    let cost_16x32 = vec![2.0; 2];
    let extra = CostGrids32x32 {
        dct_32x16: None,
        dct_16x32: Some(&cost_16x32),
    };
    let p = select_partitions_32x32_full(&cost_dct8, &cost_dct16x16, &cost_dct32x32, extra, 4, 4);
    assert_eq!(p.len(), 1);
    assert_eq!(p[0], Partition32x32::TwoDct16x32Vertical);
}

#[test]
fn picks_sub_partitions_when_cheaper() {
    let cost_dct8 = vec![1.0; 16]; // each 16x16 sub: min(30, 4) = 4
    let cost_dct16x16 = vec![30.0; 4];
    let cost_dct32x32 = vec![100.0]; // > 4*4 = 16 → pick sub

    let p = select_partitions_32x32(&cost_dct8, &cost_dct16x16, &cost_dct32x32, 4, 4);
    assert_eq!(p.len(), 1);
    if let Partition32x32::Sub16x16(subs) = p[0] {
        assert!(subs.iter().all(|&s| s == Partition16x16::FourDct8x8));
    } else {
        panic!("expected Sub16x16, got {:?}", p[0]);
    }
}
