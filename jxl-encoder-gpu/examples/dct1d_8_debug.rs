//! Standalone debug of the in-place dct1d_8 pattern: process a fixed
//! 8-element input through the same operations the GPU kernel does
//! (CPU port of my GPU dct1d_8) and compare to upstream
//! `jxl_encoder::vardct::dct::dct_4x8`'s pass-1 (row 0, indices 0..8
//! of dct1d_8_val output).
//!
//! Runs entirely on CPU — no GPU launch — to isolate algorithmic vs
//! kernel-codegen issues.

fn main() {
    const SQRT2: f32 = core::f32::consts::SQRT_2;
    const WC4_0: f32 = 0.541_196_1;
    const WC4_1: f32 = 1.306_563;
    const WC8_0: f32 = 0.509_795_6;
    const WC8_1: f32 = 0.601_344_9;
    const WC8_2: f32 = 0.899_976_2;
    const WC8_3: f32 = 2.562_915_5;

    // Same input synthesis as dct4_raw_parity, block 0 row 0.
    let mut input = [0.0_f32; 8];
    for i in 0..8 {
        let v = ((0 * 13 + i * 17_usize).wrapping_mul(31) % 251) as f32 / 251.0 - 0.5;
        input[i] = 0.3 + 0.4 * v;
    }
    println!("input row 0: {:?}", input);

    // ── My GPU kernel's dct1d_8 (CPU port, same arithmetic) ──
    let m0 = input[0]; let m1 = input[1]; let m2 = input[2]; let m3 = input[3];
    let m4 = input[4]; let m5 = input[5]; let m6 = input[6]; let m7 = input[7];
    let t0 = m0 + m7; let t1 = m1 + m6; let t2 = m2 + m5; let t3 = m3 + m4;
    let t4 = m0 - m7; let t5 = m1 - m6; let t6 = m2 - m5; let t7 = m3 - m4;
    // Inner DCT-4 on (t0..t3)
    let s0 = t0 + t3; let s1 = t1 + t2; let s2 = t0 - t3; let s3 = t1 - t2;
    let r0_0 = s0 + s1;
    let r0_2 = s0 - s1;
    let v0 = s2 * WC4_0; let v1 = s3 * WC4_1;
    let r0_3 = v0 - v1;
    let r0_1 = SQRT2 * (v0 + v1) + r0_3;
    // WC8 + Inner DCT-4 on (w4..w7)
    let w4 = t4 * WC8_0; let w5 = t5 * WC8_1; let w6 = t6 * WC8_2; let w7 = t7 * WC8_3;
    let q0 = w4 + w7; let q1 = w5 + w6; let q2 = w4 - w7; let q3 = w5 - w6;
    let r1_0 = q0 + q1;
    let r1_2 = q0 - q1;
    let v0b = q2 * WC4_0; let v1b = q3 * WC4_1;
    let r1_3 = v0b - v1b;
    let r1_1 = SQRT2 * (v0b + v1b) + r1_3;
    // Final B-transform
    let b0 = SQRT2 * r1_0 + r1_1;
    let b1 = r1_1 + r1_2;
    let b2 = r1_2 + r1_3;
    let b3 = r1_3;
    let my_out = [r0_0, b0, r0_2, b2, r0_1, b1, r0_3, b3];
    println!("my dct1d_8:  {:?}", my_out);

    // ── Upstream's jxl_encoder::vardct::dct::dct_4x8 ──
    let mut full_in = [0.0_f32; 32];
    full_in[0..8].copy_from_slice(&input);
    let mut full_out = [0.0_f32; 32];
    jxl_encoder::vardct::dct::dct_4x8(&full_in, &mut full_out);

    // dct_4x8 internally does: temp[col*4 + 0] = r[col] * (1/8) for row 0.
    // So temp[col*4] (= temp[0, 4, 8, ..., 28]) for col in 0..8 holds
    // pass-1 row=0's dct1d_8 output (scaled by 1/8). We can recover the
    // un-scaled output by reading those positions from a hand-replicated
    // pass-1, since dct_4x8's `temp` is private. Re-implement pass-1
    // using upstream's primitive.
    //
    // Since dct1d_8_val is private upstream too, the cleanest cross-check
    // is just to call upstream's dct_4x8 on a synthetic input where only
    // row 0 is non-zero and read pass-1's contribution from output via
    // the relation output[col*8 + row=0] for col in 0..4 = pass-2 col-3
    // of temp[col*4..col*4+4]; since rows 1..3 of input are zero, temp[col*4+1..3] = 0,
    // so output reflects only pass-2 of (temp[col*4], 0, 0, 0).
    // In dct1d_4_val(a, 0, 0, 0): t0=a, t1=0, t2=a, t3=0 → u0=a, u1=a, v0=a*WC4_0, v1=0,
    // w0=a*WC4_0, w1=a*WC4_0, b0=SQRT2*a*WC4_0 + a*WC4_0 = a*WC4_0*(SQRT2+1)
    // So output[0*8+0] = a / 4, output[1*8+0] = a*WC4_0*(SQRT2+1)/4, etc.
    //
    // From output[0*8 + 0] = a / 4, we get a = 4 * output[0]. And a is
    // pass-1 output at col=0 (= r0[0]) divided by 8. So upstream r0[0] for row 0 = 8*4*output[0] = 32*output[0].
    println!("upstream dct_4x8 output[0..32]:");
    for chunk in full_out.chunks(8) {
        for v in chunk {
            print!("{:>10.6} ", v);
        }
        println!();
    }

    // Recover upstream pass-1 r[col] for row 0: temp[col*4] only contains row 0
    // contribution since rows 1..3 are zero. After pass-2:
    //   row=0 in pass-2 reads temp[0..4] = [r[0]/8, 0, 0, 0]
    //   (other rows in pass-2 are reading temp[4*k..] for k=1..7; since input rows 1-3
    //    are zero, temp[col*4 + r] for r in 1..4 are zero)
    // dct1d_4(a, 0, 0, 0) returns [a, a*WC4_0*(SQRT2+1), a, a*WC4_0]
    // So output[col*8 + 0] for col=0,1,2,3 = (1/4) * [a, a*WC4_0*(SQRT2+1), a, a*WC4_0]
    // where a = (r[col]/8) for the col-th temp group.
    //
    // We can recover r[col] from output[0*8 + col_of_row_0_in_temp]:
    // wait, pass-2 row corresponds to col of pass-1's temp groups. So pass-2 row k
    // processes temp[k*4..k*4+4] = (r_k_for_row_0/8, 0, 0, 0). Output[0*8 + k] = r_k/8/4 = r_k/32.
    // Therefore r_k = 32 * output[k].
    let mut upstream_r = [0.0_f32; 8];
    for k in 0..8 {
        upstream_r[k] = 32.0 * full_out[k];
    }
    println!("upstream r[col] (recovered): {:?}", upstream_r);

    let mut max_diff = 0.0_f32;
    let mut max_pos = 0usize;
    for i in 0..8 {
        let d = (my_out[i] - upstream_r[i]).abs();
        if d > max_diff {
            max_diff = d;
            max_pos = i;
        }
    }
    println!("\nmax|Δ| at pos {max_pos}: my={}, upstream={}, diff={:.3e}",
        my_out[max_pos], upstream_r[max_pos], max_diff);
}
