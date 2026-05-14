#!/usr/bin/env python3
"""Analyze sweep_buttloop_tuning output: rank configs by RD-pareto ratio.

For each config, compute mean ratio across the 4 photos at the test distance,
where ratio = our_bytes / cjxl_bytes (interpolated at our measured bfly).

Usage:
  analyze_sweep.py --ref REF.tsv --sweep SWEEP.tsv [--distance 3.0]
"""
import sys
import argparse
from collections import defaultdict


def load_tsv(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line:
                continue
            parts = line.split("\t")
            if parts[0] == "image":
                continue
            try:
                if len(parts) == 9:
                    # sweep TSV: image w h enc dist config bytes bfly ms
                    rows.append({
                        "image": parts[0],
                        "encoder": parts[3],
                        "distance": float(parts[4]),
                        "config_name": parts[5],
                        "bytes": int(parts[6]),
                        "bfly": float(parts[7]),
                        "ms": float(parts[8]),
                    })
                elif len(parts) == 10:
                    # ref TSV: image w h enc effort dist bytes bfly ssim2 ms
                    rows.append({
                        "image": parts[0],
                        "encoder": parts[3],
                        "distance": float(parts[5]),
                        "config_name": "",
                        "bytes": int(parts[6]),
                        "bfly": float(parts[7]),
                        "ssim2": float(parts[8]),
                        "ms": float(parts[9]),
                    })
            except (ValueError, IndexError):
                continue
    return rows


def linear_interp(x, x0, y0, x1, y1):
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def cjxl_bytes_at_bfly(curve, target_bfly):
    if not curve:
        return None
    sorted_rows = sorted(curve, key=lambda r: r["bfly"])
    if target_bfly <= sorted_rows[0]["bfly"]:
        return None
    if target_bfly >= sorted_rows[-1]["bfly"]:
        return None
    for i in range(len(sorted_rows) - 1):
        a, b = sorted_rows[i], sorted_rows[i + 1]
        if a["bfly"] <= target_bfly <= b["bfly"]:
            return linear_interp(target_bfly, a["bfly"], a["bytes"], b["bfly"], b["bytes"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="reference TSV (cjxl side)")
    parser.add_argument("--sweep", required=True, help="sweep TSV with multiple configs")
    parser.add_argument("--distance", type=float, default=3.0)
    args = parser.parse_args()

    ref_rows = load_tsv(args.ref)
    sweep_rows = load_tsv(args.sweep)

    # Build cjxl curves per image — use cjxl_e8 if available else cjxl_e7
    cjxl_by_image: dict[str, list] = defaultdict(list)
    images = set(r["image"] for r in ref_rows)
    for img in images:
        for cj_enc in ("cjxl_e8", "cjxl_e7"):
            cj_rows = [r for r in ref_rows if r["image"] == img and r["encoder"] == cj_enc]
            if cj_rows:
                cjxl_by_image[img] = cj_rows
                break

    # Filter sweep rows by distance
    sweep = [r for r in sweep_rows if abs(r["distance"] - args.distance) < 1e-6]

    # Group: config_name → image → row
    config_by_name: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in sweep:
        config_by_name[r["config_name"]][r["image"]] = r

    # Per-config analysis
    config_summary = []
    print(f"\n=== {args.sweep} (d={args.distance}) ===\n")
    print(f"{'config':<22} {'image':<10} {'our_bfly':>9} {'our_bytes':>10} {'cjxl_bytes':>11} {'ratio':>7} {'time_ms':>8}")

    for cfg_name in sorted(config_by_name.keys()):
        ratios = []
        for img in sorted(config_by_name[cfg_name].keys()):
            r = config_by_name[cfg_name][img]
            curve = cjxl_by_image.get(img, [])
            cjxl_b = cjxl_bytes_at_bfly(curve, r["bfly"])
            if cjxl_b is None:
                print(f"{cfg_name:<22} {img:<10} {r['bfly']:9.3f} {r['bytes']:10d}   (out of range)   {r['ms']:8.0f}")
                continue
            ratio = 100.0 * r["bytes"] / cjxl_b
            print(f"{cfg_name:<22} {img:<10} {r['bfly']:9.3f} {r['bytes']:10d} {int(cjxl_b):11d} {ratio:6.1f}% {r['ms']:8.0f}")
            ratios.append((img, ratio, r["bytes"], r["bfly"], r["ms"]))
        if ratios:
            mean = sum(x[1] for x in ratios) / len(ratios)
            tot_bytes = sum(x[2] for x in ratios)
            avg_ms = sum(x[4] for x in ratios) / len(ratios)
            mean_bfly = sum(x[3] for x in ratios) / len(ratios)
            config_summary.append((cfg_name, mean, tot_bytes, mean_bfly, avg_ms, len(ratios)))
            print(f"{cfg_name:<22} {'MEAN':<10} {mean_bfly:9.3f} {tot_bytes:10d}             {mean:6.1f}% {avg_ms:8.0f}")
            print()

    print()
    print("="*72)
    print("CONFIG RANKING (by mean RD-pareto ratio across all photos)")
    print("="*72)
    print(f"{'config':<22} {'mean_ratio':>10} {'tot_bytes':>10} {'mean_bfly':>10} {'avg_ms':>8} {'n':>3}")
    print("-"*72)
    for cfg_name, mean, tot_bytes, mean_bfly, avg_ms, n in sorted(config_summary, key=lambda x: x[1]):
        print(f"{cfg_name:<22} {mean:9.1f}% {tot_bytes:10d} {mean_bfly:10.3f} {avg_ms:8.0f} {n:3d}")


if __name__ == "__main__":
    main()
