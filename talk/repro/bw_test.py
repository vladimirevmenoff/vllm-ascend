#!/usr/bin/env python3
"""Measure real achievable memory bandwidth on ONE 310P3 chip (TP=1).

    ASCEND_RT_VISIBLE_DEVICES=<dev> python bw_test.py

Three tests, increasingly close to what decode actually does:
  1. copy          - b.copy_(a). Counts read+write. The classic STREAM-ish number.
  2. GEMV          - y = x @ W with x a single row. Reads all of W, writes almost
                     nothing, reuses nothing: the exact access pattern of BS=1 decode.
                     Effective bandwidth = W's bytes / time.
  3. decode step   - every weight shape in Qwen3.5-9B, in the right multiplicities,
                     as GEMVs. Should reproduce the measured TPOT if the "decode is
                     bandwidth-bound" model is right.

Buffers are far larger than L2, and each is touched once per iteration, so re-reads
go to DRAM.

CRITICAL: weights must be in FRACTAL_NZ layout, which is what vllm-ascend uses on
310P ("Weight layout uses FRACTAL_NZ" in the startup log). A plain ND tensor gives
41-53 GB/s on GEMV; the same tensor cast to NZ gives 166-173 GB/s — a 3-4x
difference that has nothing to do with the memory system. Measuring ND and calling
it "achievable bandwidth" understates the hardware by 3x.
"""
import time

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

GiB = 1024 ** 3
DTYPE = torch.float16


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def bw(nbytes, seconds):
    return nbytes / seconds / 1e9


# ---------------------------------------------------------------- 1. copy
def test_copy():
    print("\n=== 1. copy (read + write) ===")
    for mb in (64, 256, 512, 1024):
        n = mb * 1024 * 1024 // 2
        a = torch.ones(n, dtype=DTYPE, device="npu")
        b = torch.empty_like(a)
        dt = timeit(lambda: b.copy_(a))
        moved = a.numel() * 2 * 2          # read + write
        print(f"  {mb:>5} MB   {dt*1e3:8.3f} ms   {bw(moved, dt):7.1f} GB/s")
        del a, b
        torch.npu.empty_cache()


# ---------------------------------------------------------------- 2. GEMV
# (K, N, label) — the real projections in Qwen3.5-9B
SHAPES = [
    (4096, 12288, "GDN in_proj      "),
    (4096, 4096, "GDN out_proj     "),
    (4096, 8192, "attn q(+gate)    "),
    (4096, 1024, "attn k / v       "),
    (4096, 24576, "MLP gate_up      "),
    (12288, 4096, "MLP down         "),
    (4096, 248320, "lm_head          "),
]


FRACTAL_NZ = 29


def to_nz(w):
    """Cast to the layout vllm-ascend actually uses on 310P. Falls back to ND."""
    try:
        return torch_npu.npu_format_cast(w, FRACTAL_NZ)
    except Exception:
        return w


def test_gemv():
    print("\n=== 2. GEMV — reads the whole matrix, writes one row ===")
    print("     (this is the BS=1 decode access pattern)")
    print(f"  {'shape':<22} {'bytes':>8} {'ND GB/s':>9} {'NZ GB/s':>9} {'ratio':>7}")
    for k, n, label in SHAPES:
        w = torch.randn(k, n, dtype=DTYPE, device="npu")
        x = torch.randn(1, k, dtype=DTYPE, device="npu")
        nbytes = k * n * 2
        nd = timeit(lambda: torch.matmul(x, w), iters=20)
        wnz = to_nz(w)
        nz = timeit(lambda: torch.matmul(x, wnz), iters=20)
        print(f"  {label[:12]:<12}{k:>5}x{n:<6} {nbytes/GiB:6.2f}G "
              f"{bw(nbytes, nd):9.1f} {bw(nbytes, nz):9.1f} {nd/nz:6.2f}x")
        del w, x, wnz
        torch.npu.empty_cache()


# ------------------------------------------------- 3. synthetic decode step
# multiplicity of each projection in one forward pass
LAYERS = [
    (24, [(4096, 12288), (4096, 4096), (4096, 64)]),        # GDN layers
    (8, [(4096, 8192), (4096, 1024), (4096, 1024), (4096, 4096)]),   # attn layers
    (32, [(4096, 24576), (12288, 4096)]),                    # MLP, every layer
    (1, [(4096, 248320)]),                                   # lm_head
]


def test_decode_step(weight_bytes_per_elem=2):
    tag = "fp16" if weight_bytes_per_elem == 2 else "int8"
    print(f"\n=== 3. synthetic decode step, {tag} weights ===")
    # Allocate one buffer per distinct shape; loop over it with the right count.
    # Each buffer is >> L2, so every pass is a fresh DRAM read.
    bufs, total_bytes, plan = {}, 0, []
    for count, shapes in LAYERS:
        for k, n in shapes:
            if (k, n) not in bufs:
                bufs[(k, n)] = (
                    to_nz(torch.randn(k, n, dtype=DTYPE, device="npu")),
                    torch.randn(1, k, dtype=DTYPE, device="npu"),
                )
            plan.append((count, (k, n)))
            total_bytes += count * k * n * weight_bytes_per_elem

    def one_step():
        for count, key in plan:
            w, x = bufs[key]
            for _ in range(count):
                torch.matmul(x, w)

    dt = timeit(one_step, iters=5, warmup=2)
    print(f"  weight traffic : {total_bytes/1e9:7.2f} GB")
    print(f"  time / step    : {dt*1e3:7.2f} ms")
    print(f"  effective BW   : {bw(total_bytes, dt):7.1f} GB/s")
    if weight_bytes_per_elem == 2:
        print("  (measured fp16 TPOT was 101.3 ms)")
    for w, x in bufs.values():
        del w, x
    bufs.clear()
    torch.npu.empty_cache()


if __name__ == "__main__":
    torch.npu.set_device(0)          # 0 = whatever ASCEND_RT_VISIBLE_DEVICES points at
    free, total = torch.npu.mem_get_info(0)
    print(f"device: {free/GiB:.1f} / {total/GiB:.1f} GiB free")
    test_copy()
    test_gemv()
    test_decode_step(2)
    print("\nnote: the synthetic step runs fp16 GEMVs. For the W8A8 model the same")
    print("weights are int8, i.e. half the bytes — scale accordingly.")
