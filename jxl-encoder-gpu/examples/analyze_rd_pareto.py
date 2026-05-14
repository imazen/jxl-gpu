#!/usr/bin/env python3
# Copyright (c) Imazen LLC and the JPEG XL Project Authors.
# Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
#
# Analyze TSV output from `examples/rd_pareto_vs_cjxl` and print three
# verdicts:
#  1. RD-pareto: at each measured-quality bin (butteraugli), bytes ratio
#     between our GPU encoder and cjxl. Per-image and aggregated.
#  2. RD-time: at the same RD position, who's faster?
#  3. Quality targeting: at each `--distance` setting, what's the ACTUAL
#     measured butteraugli? Are we hitting the target or under/over?
#
# Usage:
#   python3 analyze_rd_pareto.py path/to/rd_pareto_*.tsv

import csv
import sys
from collections import defaultdict


def load_tsv(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.startswith("image\t"):  # header
                fields = line.split("\t")
                continue
            parts = line.split("\t")
            if len(parts) != len(fields):
                continue
            row = dict(zip(fields, parts))
            # Cast types
            for k in ("width", "height", "effort", "bytes"):
                row[k] = int(row[k])
            for k in ("distance", "measured_bfly", "measured_ssim2", "encode_ms"):
                row[k] = float(row[k])
            rows.append(row)
    return rows


def fmt_pct(p):
    sign = "+" if p > 0 else ""
    return f"{sign}{p:5.1f}%"


def by(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def aggregate(rows, key):
    """Mean of `key` across rows."""
    if not rows:
        return float("nan")
    return sum(r[key] for r in rows) / len(rows)


def main():
    if len(sys.argv) != 2:
        print("usage: analyze_rd_pareto.py PATH.tsv", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    rows = load_tsv(path)
    print(f"# loaded {len(rows)} rows from {path}\n")

    images = sorted({r["image"] for r in rows})
    encoders = sorted({r["encoder"] for r in rows})
    distances = sorted({r["distance"] for r in rows})

    gpu_encoders = [e for e in encoders if e.startswith("gpu_")]
    cjxl_encoders = [e for e in encoders if e.startswith("cjxl_")]

    print("=" * 72)
    print("VERDICT 1: Quality targeting")
    print("=" * 72)
    print(
        "At each `-d` setting, what's the MEASURED butteraugli? Are we hitting target?\n"
    )
    print(
        f"{'Image':<10} {'Dist':>5}  "
        + "  ".join(f"{e:>11}" for e in encoders)
    )
    print("-" * (18 + len(encoders) * 13))
    for img in images:
        for d in distances:
            vals = []
            for e in encoders:
                rs = [r for r in rows if r["image"] == img and r["distance"] == d and r["encoder"] == e]
                if not rs:
                    vals.append("       n/a")
                    continue
                vals.append(f"{rs[0]['measured_bfly']:>11.3f}")
            print(f"{img:<10} {d:>5.2f}  " + "  ".join(vals))
        print()

    # Per-distance averages — measured butteraugli
    print(f"{'Avg':<10} {'Dist':>5}  " + "  ".join(f"{e:>11}" for e in encoders))
    print("-" * (18 + len(encoders) * 13))
    for d in distances:
        vals = []
        for e in encoders:
            rs = [r for r in rows if r["distance"] == d and r["encoder"] == e]
            if not rs:
                vals.append("       n/a")
                continue
            vals.append(f"{aggregate(rs, 'measured_bfly'):>11.3f}")
        print(f"{'avg':<10} {d:>5.2f}  " + "  ".join(vals))

    print()
    print("=" * 72)
    print("VERDICT 2: RD-pareto comparison (bytes vs measured quality)")
    print("=" * 72)
    print(
        "Per-image: at the SAME measured butteraugli (matching quality across encoders\n"
        "by interpolation), who has fewer bytes?\n"
    )

    # We don't yet have an interpolation oracle; the simplest comparable
    # is per-image, per-`-d`-setting, per-encoder pairs (gpu_eN vs cjxl_eN).
    # That's "at the same -d setting, achieved a different (size, quality)
    # pair". We tabulate (Δbytes vs Δbfly vs Δssim2) per pair.
    print(
        f"{'Image':<10} {'Dist':>5}  {'Pair':<24}  {'Δbytes':>9}  {'Δbfly':>9}  {'Δssim2':>9}"
    )
    print("-" * 72)
    pair_aggs = defaultdict(list)  # (eff, pair_kind) → list of triples
    for img in images:
        for d in distances:
            for eff in (7, 8, 9):
                gpu = f"gpu_e{eff}"
                cj = f"cjxl_e{eff}"
                gr = next(
                    (
                        r
                        for r in rows
                        if r["image"] == img and r["distance"] == d and r["encoder"] == gpu
                    ),
                    None,
                )
                cr = next(
                    (
                        r
                        for r in rows
                        if r["image"] == img and r["distance"] == d and r["encoder"] == cj
                    ),
                    None,
                )
                if not gr or not cr:
                    continue
                d_bytes = (gr["bytes"] - cr["bytes"]) / cr["bytes"] * 100
                d_bfly = (gr["measured_bfly"] - cr["measured_bfly"]) / cr["measured_bfly"] * 100
                d_ss2 = gr["measured_ssim2"] - cr["measured_ssim2"]
                pair = f"{gpu} vs {cj}"
                print(
                    f"{img:<10} {d:>5.2f}  {pair:<24}  {fmt_pct(d_bytes):>9}  "
                    f"{fmt_pct(d_bfly):>9}  {d_ss2:>+9.2f}"
                )
                pair_aggs[(eff, d)].append((d_bytes, d_bfly, d_ss2))
        print()

    print()
    print(f"{'Aggregate':<10} {'Dist':>5}  {'Pair':<24}  {'Δbytes':>9}  {'Δbfly':>9}  {'Δssim2':>9}")
    print("-" * 72)
    for (eff, d), triples in sorted(pair_aggs.items()):
        if not triples:
            continue
        n = len(triples)
        ab = sum(t[0] for t in triples) / n
        af = sum(t[1] for t in triples) / n
        ass = sum(t[2] for t in triples) / n
        pair = f"gpu_e{eff} vs cjxl_e{eff}"
        print(
            f"{'avg(' + str(n) + ')':<10} {d:>5.2f}  {pair:<24}  {fmt_pct(ab):>9}  "
            f"{fmt_pct(af):>9}  {ass:>+9.2f}"
        )

    print()
    # Better RD-pareto comparison: linearly interpolate cjxl's (bytes,
    # measured_bfly) curve and ask, "at the same MEASURED bfly as ours,
    # what would cjxl's bytes be?" This is the canonical RD-pareto delta.
    print()
    print("=" * 72)
    print("VERDICT 2b: RD-pareto by interpolated equal-quality")
    print("=" * 72)
    print(
        "For each (image, gpu_encoder), interpolate cjxl's RD curve at the SAME\n"
        "measured butteraugli. Report bytes ratio: ours / cjxl-at-same-quality.\n"
        "(<100% = we're smaller at same quality = WIN; >100% = LOSS)\n"
    )

    print(
        f"{'Image':<10} {'GPU enc':<8} {'Dist':>5}  {'Our bfly':>9}  {'Our bytes':>10}  "
        f"{'C bytes@bfly':>13}  {'Ratio':>7}"
    )
    print("-" * 80)
    interp_aggs = defaultdict(list)
    for img in images:
        # Build cjxl curve(s) — sort by measured_bfly. Use cjxl_e8 if available
        # (best-quality cjxl curve at each bfly) or cjxl_e7 fallback.
        for cj_enc in ("cjxl_e8", "cjxl_e7"):
            cj = sorted(
                [r for r in rows if r["image"] == img and r["encoder"] == cj_enc],
                key=lambda r: r["measured_bfly"],
            )
            if cj:
                cj_curve_bfly = [r["measured_bfly"] for r in cj]
                cj_curve_bytes = [r["bytes"] for r in cj]
                cj_curve_name = cj_enc
                break
        else:
            print(f"  no cjxl curve for {img}, skipping")
            continue

        def cjxl_bytes_at_bfly(target_bfly):
            """Linear-interpolate bytes at target butteraugli (lower bfly = more
            bytes). Returns NaN if outside the curve range."""
            if target_bfly <= cj_curve_bfly[0]:
                return float("nan")  # cjxl can't reach this quality
            if target_bfly >= cj_curve_bfly[-1]:
                return float("nan")  # below cjxl's lowest-bytes point
            for i in range(len(cj_curve_bfly) - 1):
                lo_b, hi_b = cj_curve_bfly[i], cj_curve_bfly[i + 1]
                if lo_b <= target_bfly <= hi_b:
                    lo_y, hi_y = cj_curve_bytes[i], cj_curve_bytes[i + 1]
                    t = (target_bfly - lo_b) / (hi_b - lo_b) if hi_b > lo_b else 0
                    return lo_y + t * (hi_y - lo_y)
            return float("nan")

        for ge in gpu_encoders:
            for d in distances:
                gr = next(
                    (
                        r
                        for r in rows
                        if r["image"] == img and r["distance"] == d and r["encoder"] == ge
                    ),
                    None,
                )
                if not gr:
                    continue
                cj_b = cjxl_bytes_at_bfly(gr["measured_bfly"])
                if cj_b != cj_b:  # NaN
                    print(
                        f"{img:<10} {ge:<8} {d:>5.2f}  {gr['measured_bfly']:>9.3f}  "
                        f"{gr['bytes']:>10}  {'(out)':>13}  {'-':>7}"
                    )
                    continue
                ratio = gr["bytes"] / cj_b * 100
                print(
                    f"{img:<10} {ge:<8} {d:>5.2f}  {gr['measured_bfly']:>9.3f}  "
                    f"{gr['bytes']:>10}  {cj_b:>13.0f}  {ratio:>6.1f}%"
                )
                interp_aggs[(ge, d)].append(ratio)
        print()

    print()
    print("Aggregate RD-pareto ratios (our bytes / cjxl bytes at SAME measured bfly):")
    print(f"{'Aggregate':<10} {'GPU enc':<8} {'Dist':>5}  {'Mean ratio':>12}  {'Median':>9}  {'Worst':>9}  {'n':>3}")
    print("-" * 65)
    for (ge, d), ratios in sorted(interp_aggs.items()):
        if not ratios:
            continue
        ratios.sort()
        m = sum(ratios) / len(ratios)
        med = ratios[len(ratios) // 2]
        worst = max(ratios)
        print(
            f"{'avg':<10} {ge:<8} {d:>5.2f}  {m:>11.1f}%  {med:>8.1f}%  {worst:>8.1f}%  {len(ratios):>3}"
        )

    print()
    # Per-encoder grand-average across all distances
    grand = defaultdict(list)
    for (ge, d), ratios in interp_aggs.items():
        grand[ge].extend(ratios)
    print("Grand average (all distances):")
    for ge, ratios in sorted(grand.items()):
        if not ratios:
            continue
        m = sum(ratios) / len(ratios)
        print(f"  {ge:<10}: mean ratio {m:6.1f}%  (n={len(ratios)})")

    print()
    print("=" * 72)
    print("VERDICT 3: RD-time pareto")
    print("=" * 72)
    print(
        "At the same -d setting, encode-time per encoder. (Same quality target,\n"
        "same encoder effort signal; absolute ms is the comparison.)\n"
    )

    print(f"{'Image':<10} {'Dist':>5}  " + "  ".join(f"{e:>10}" for e in encoders))
    print("-" * (18 + len(encoders) * 12))
    for img in images:
        for d in distances:
            vals = []
            for e in encoders:
                rs = [r for r in rows if r["image"] == img and r["distance"] == d and r["encoder"] == e]
                if not rs:
                    vals.append("       n/a")
                    continue
                vals.append(f"{rs[0]['encode_ms']:>10.1f}")
            print(f"{img:<10} {d:>5.2f}  " + "  ".join(vals))
        print()

    print(f"{'Avg':<10} {'Dist':>5}  " + "  ".join(f"{e:>10}" for e in encoders))
    print("-" * (18 + len(encoders) * 12))
    for d in distances:
        vals = []
        for e in encoders:
            rs = [r for r in rows if r["distance"] == d and r["encoder"] == e]
            if not rs:
                vals.append("       n/a")
                continue
            vals.append(f"{aggregate(rs, 'encode_ms'):>10.1f}")
        print(f"{'avg ms':<10} {d:>5.2f}  " + "  ".join(vals))


if __name__ == "__main__":
    main()
