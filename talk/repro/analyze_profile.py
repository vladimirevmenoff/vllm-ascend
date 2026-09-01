#!/usr/bin/env python3
"""Rank kernels from an Ascend profile, split into prefill and per-decode-step.

    analyze_profile.py <dir containing kernel_details.csv>

Why the splitting is done the way it is: the device is busy ~98% of the time, so
there are no idle gaps between steps to segment on. Instead we cut at the lm_head
GEMM, which runs exactly once per forward pass — first pass is prefill, the rest
are decode steps. Verify the split by checking that the reported per-decode-step
device time is close to the TPOT you measured.
"""
import csv
import statistics
import sys
from collections import defaultdict

# The lm_head GEMM: one per forward pass, and by far the largest single matmul.
# If a future build names it differently, set PASS_MARKER to whatever op has a
# count of exactly (1 + number of decode steps) in the "whole capture" listing.
PASS_MARKER = "MatMulV2"


def load(d):
    rows = []
    with open(d + "/kernel_details.csv") as f:
        for r in csv.DictReader(f):
            try:
                rows.append([
                    float(r["Start Time(us)"]), float(r["Duration(us)"]),
                    r["Name"], r["Type"], r.get("aicore_time(us)", ""),
                    r.get("mac_ratio", ""), r.get("vec_ratio", ""),
                    r.get("scalar_ratio", ""), r.get("mte2_ratio", ""),
                    r.get("mte3_ratio", ""), r.get("memory_bound", ""),
                ])
            except (KeyError, ValueError):
                pass
    rows.sort()
    return rows


def split_passes(rows):
    passes, cur = [], []
    for r in rows:
        cur.append(r)
        if r[3] == PASS_MARKER:
            passes.append(cur)
            cur = []
    if cur:
        passes.append(cur)
    return passes


def top(kernels, label, n=15):
    agg = defaultdict(lambda: [0.0, 0])
    for k in kernels:
        agg[k[3]][0] += k[1]
        agg[k[3]][1] += 1
    total = sum(v[0] for v in agg.values())
    print(f"\n=== {label}  (device-busy {total/1000:.1f} ms, {len(kernels)} kernels)")
    print(f"{'op type':<38} {'total(ms)':>10} {'count':>8} {'avg(us)':>9} {'share':>7}")
    for name, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:n]:
        print(f"{name:<38} {t/1000:>10.2f} {c:>8} {t/c:>9.1f} {100*t/total:>6.1f}%")
    return total


def pipes(kernels, label):
    total = sum(k[1] for k in kernels)
    acc = defaultdict(float)
    aicore = mem_bound = 0.0
    for k in kernels:
        try:
            a = float(k[4]) if k[4] else 0.0
        except ValueError:
            a = 0.0
        aicore += a
        for name, idx in (("mac", 5), ("vec", 6), ("scalar", 7), ("mte2", 8), ("mte3", 9)):
            try:
                acc[name] += a * float(k[idx]) if k[idx] else 0.0
            except ValueError:
                pass
        try:
            if k[10] and float(k[10]) > 0.5:
                mem_bound += k[1]
        except ValueError:
            pass
    mix = "  ".join(f"{n} {100*v/aicore:.0f}%" for n, v in acc.items()) if aicore else "n/a"
    print(f"{label}: aicore {100*aicore/total:.0f}% of busy | of aicore: {mix} "
          f"| memory-bound {100*mem_bound/total:.0f}% of busy")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = load(d)
    if not rows:
        sys.exit(f"no usable rows in {d}/kernel_details.csv")
    passes = split_passes(rows)
    print(f"kernels {len(rows)}, passes found {len(passes)} "
          f"(expect 1 prefill + N decode steps, possibly + a tail)")

    prefill, decode_passes = passes[0], passes[1:]
    if decode_passes and len(decode_passes[-1]) < len(decode_passes[0]) / 2:
        decode_passes = decode_passes[:-1]          # drop the partial tail

    busy = lambda s: sum(k[1] for k in s) / 1000.0
    wall = lambda s: (s[-1][0] + s[-1][1] - s[0][0]) / 1000.0
    print(f"PREFILL : {len(prefill)} kernels, wall {wall(prefill):.1f} ms, "
          f"device-busy {busy(prefill):.1f} ms")
    if decode_passes:
        print(f"DECODE  : {len(decode_passes)} steps, "
              f"{statistics.mean(len(s) for s in decode_passes):.0f} kernels/step, "
              f"busy/step {statistics.mean(busy(s) for s in decode_passes):.2f} ms "
              f"(compare against your measured TPOT)")

    top(prefill, "PREFILL — one pass")
    pipes(prefill, "prefill")
    if decode_passes:
        flat = [k for s in decode_passes for k in s]
        top(flat, f"DECODE — {len(decode_passes)} steps aggregated")
        pipes(flat, "decode")
        print(f"\n--- per decode step (mean of {len(decode_passes)}) ---")
        agg = defaultdict(float)
        for k in flat:
            agg[k[3]] += k[1]
        for name, t in sorted(agg.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {name:<36} {t/1000/len(decode_passes):>7.3f} ms/step")
