#!/usr/bin/env python3
"""Separate MTP draft passes from verify passes in an Ascend kernel trace.

    analyze_mtp_split.py <dir with kernel_details.csv>

Passes are cut at the lm_head GEMM (any op whose type starts with MatMulV2), then
classified: a pass containing RecurrentGatedDeltaRuleV310 ran the target's 24 GDN
layers, so it is a VERIFY pass; one without is a DRAFT pass. That works regardless
of how many passes there are, which the plain pass-counting splitter does not.
"""
import csv
import sys
from collections import defaultdict

VERIFY_MARKER = "RecurrentGatedDeltaRuleV310"


def load(d):
    rows = []
    with open(d + "/kernel_details.csv") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["Start Time(us)"]), float(r["Duration(us)"]), r["Type"]))
            except (KeyError, ValueError):
                pass
    rows.sort()
    return rows


def split(rows):
    passes, cur = [], []
    for r in rows:
        cur.append(r)
        if r[2].startswith("MatMulV2"):
            passes.append(cur)
            cur = []
    if cur:
        passes.append(cur)
    return passes


def report(passes, label):
    if not passes:
        print(f"\n{label}: none found")
        return
    busy = [sum(k[1] for k in p) / 1000.0 for p in passes]
    agg = defaultdict(lambda: [0.0, 0])
    for p in passes:
        for k in p:
            agg[k[2]][0] += k[1]
            agg[k[2]][1] += 1
    total = sum(v[0] for v in agg.values())
    n = len(passes)
    print(f"\n=== {label}: {n} passes, {sum(busy)/n:.2f} ms/pass mean "
          f"({min(busy):.2f}–{max(busy):.2f})")
    print(f"{'op':<46} {'ms/pass':>9} {'calls/pass':>11} {'share':>7}")
    for name, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:10]:
        print(f"{name[:46]:<46} {t/1000/n:>9.3f} {c/n:>11.1f} {100*t/total:>6.1f}%")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = load(d)
    passes = split(rows)
    verify = [p for p in passes if any(k[2] == VERIFY_MARKER for k in p)]
    draft = [p for p in passes if not any(k[2] == VERIFY_MARKER for k in p)]
    # the first pass is prefill (huge); drop it from the verify set
    if verify:
        verify_sorted = sorted(verify, key=lambda p: -sum(k[1] for k in p))
        prefill = verify_sorted[0]
        print(f"prefill pass: {sum(k[1] for k in prefill)/1000:.1f} ms "
              f"({len(prefill)} kernels) — excluded below")
        verify = [p for p in verify if p is not prefill]
    print(f"total passes {len(passes)}: {len(verify)} verify, {len(draft)} draft")
    report(verify, "VERIFY passes (target, 32 layers)")
    report(draft, "DRAFT passes (MTP head, 1 layer + lm_head)")
