#!/usr/bin/env python3
"""Turn `vllm bench serve --save-result` JSONs into a markdown table.

Usage:  summarize.py RESULT_DIR [RESULT_DIR ...]
One table per directory, plus a comparison table when two are given.
Prefill/decode speeds are derived: prefill = input_len / TTFT (BS=1 only),
decode = 1000 / TPOT (per request).
"""
import json
import sys
from pathlib import Path


def load(d):
    runs = []
    for f in sorted(Path(d).glob("*_bs*.json")):
        with open(f) as fh:
            r = json.load(fh)
        bs = r.get("max_concurrency") or 1
        inlen = round(r["total_input_tokens"] / r["completed"]) if r.get("completed") else 0
        runs.append({
            "bs": int(bs),
            "inlen": inlen,
            "ttft": r["mean_ttft_ms"], "ttft99": r["p99_ttft_ms"],
            "tpot": r["mean_tpot_ms"], "tpot99": r["p99_tpot_ms"],
            "itl": r["mean_itl_ms"], "e2el": r["mean_e2el_ms"],
            "out_tput": r["output_throughput"], "tot_tput": r["total_token_throughput"],
            "completed": r["completed"], "failed": r.get("failed", 0),
        })
    return sorted(runs, key=lambda x: x["bs"])


def table(runs, label):
    print(f"\n### {label}\n")
    print("| BS | TTFT mean (ms) | TTFT P99 (ms) | TPOT mean (ms) | TPOT P99 (ms) | "
          "decode/req (tok/s) | output tput (tok/s) | total tput (tok/s) | E2E (ms) | ok |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        print(f"| {r['bs']} | {r['ttft']:.1f} | {r['ttft99']:.1f} | {r['tpot']:.2f} | "
              f"{r['tpot99']:.2f} | {1000 / r['tpot']:.2f} | {r['out_tput']:.2f} | "
              f"{r['tot_tput']:.2f} | {r['e2el']:.0f} | {r['completed']}/"
              f"{r['completed'] + r['failed']} |")
    bs1 = next((r for r in runs if r["bs"] == 1), None)
    if bs1:
        print(f"\nBS=1 prefill speed: **{bs1['inlen'] / (bs1['ttft'] / 1000):.1f} tok/s** "
              f"({bs1['inlen']} tokens / {bs1['ttft']:.1f} ms)  ·  "
              f"decode speed: **{1000 / bs1['tpot']:.2f} tok/s**")
        print("\nPrefill speed is only meaningful at BS=1 — above that TTFT is dominated by "
              "queueing (requests are dispatched at rate `inf`, gated by --max-concurrency).")


def compare(a, b, la, lb):
    ai = {r["bs"]: r for r in a}
    bi = {r["bs"]: r for r in b}
    shared = sorted(set(ai) & set(bi))
    if not shared:
        return
    print(f"\n### {lb} vs {la}\n")
    print(f"| BS | TTFT {la} | TTFT {lb} | TTFT ↑ | TPOT {la} | TPOT {lb} | TPOT ↑ | "
          f"output tput {la} | output tput {lb} | ↑ |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for bs in shared:
        x, y = ai[bs], bi[bs]
        print(f"| {bs} | {x['ttft']:.1f} | {y['ttft']:.1f} | {x['ttft'] / y['ttft']:.2f}× | "
              f"{x['tpot']:.2f} | {y['tpot']:.2f} | {x['tpot'] / y['tpot']:.2f}× | "
              f"{x['out_tput']:.2f} | {y['out_tput']:.2f} | "
              f"{y['out_tput'] / x['out_tput']:.2f}× |")


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["."]
    loaded = []
    for d in dirs:
        runs = load(d)
        if not runs:
            print(f"no *_bs*.json results in {d}", file=sys.stderr)
            continue
        label = Path(d).name
        loaded.append((label, runs))
        table(runs, label)
    if len(loaded) == 2:
        compare(loaded[0][1], loaded[1][1], loaded[0][0], loaded[1][0])
