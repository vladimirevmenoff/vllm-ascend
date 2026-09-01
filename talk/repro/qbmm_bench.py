#!/usr/bin/env python3
"""Microbenchmark QuantBatchMatmulV3 (torch_npu.npu_quant_matmul) at the exact
shapes Qwen3.5-9B issues, for both prefill (M=2048) and decode (M=1).

    ASCEND_RT_VISIBLE_DEVICES=<dev> python qbmm_bench.py [M ...]

Weights are cast to FRACTAL_NZ, which is what vllm-ascend uses on 310P. Measuring
in ND understates bandwidth ~3-4x (see bw_test.py).
"""
import sys
import time

import torch
import torch_npu

FRACTAL_NZ = 29
TRY_NZ = False   # NZ -> aclnnQuantMatmulWeightNz, unsupported on 310P

# (K, N, calls_per_forward_pass, label)
SHAPES = [
    (4096, 24576, 32, "MLP gate_up"),
    (12288, 4096, 32, "MLP down"),
    (4096, 4096, 32, "GDN out_proj + attn o_proj"),
    (4096, 12288, 24, "GDN in_proj_qkvz"),
    (4096, 64, 24, "GDN in_proj_ba"),
    (4096, 10240, 8, "attn qkv fused"),
]


def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def bench(m):
    print(f"\n{'='*92}\nM = {m}   ({'prefill, one 2048-token chunk' if m > 1 else 'decode, one token'})")
    print(f"{'shape M x K x N':<26} {'calls':>5} {'ms/call':>9} {'TOPS':>8} "
          f"{'GB/s':>8} {'ms/pass':>9}  what")
    total_ms = 0.0
    for k, n, calls, label in SHAPES:
        # Replicate vllm_ascend/_310p/quantization/methods/w8a8_static.py exactly:
        #   weight is stored [N, K] (nn.Linear layout), cast to NZ, then transposed
        #   deq_scale is int64, length N
        #   x is int8, output fp16
        x = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="npu")
        w_nk = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="npu")
        try:
            scale = torch_npu.npu_trans_quant_param(
                torch.ones(n, dtype=torch.float32, device="npu")
            )
        except Exception:
            scale = torch.ones(n, dtype=torch.int64, device="npu")
        # NZ cast THEN transpose — the order matters; NZ without the transpose
        # routes to aclnnQuantMatmulWeightNz, which 310P does not support.
        try:
            wq = torch_npu.npu_format_cast(w_nk, FRACTAL_NZ).transpose(0, 1)
            layout = "NZ.T"
        except Exception:
            wq = w_nk.transpose(0, 1)
            layout = "ND.T"

        def run():
            return torch_npu.npu_quant_matmul(x, wq, scale, output_dtype=torch.float16)

        try:
            run()
            torch.npu.synchronize()
        except Exception as e:
            print(f"  {m}x{k}x{n:<10} {calls:>5}   FAILED: {str(e)[:52]}")
            del x, w_nk, wq, scale
            torch.npu.empty_cache()
            continue

        iters = 50 if m == 1 else 20
        dt = timeit(run, iters)
        flops = 2 * m * k * n
        wbytes = k * n                      # int8
        total_ms += dt * 1e3 * calls
        print(f"  {m}x{k}x{n:<10} {calls:>5} {dt*1e3:>9.3f} {flops/dt/1e12:>8.1f} "
              f"{wbytes/dt/1e9:>8.1f} {dt*1e3*calls:>9.1f}  {layout} {label}")
        del x, w_nk, wq, scale
        torch.npu.empty_cache()
    print(f"  {'':<26} {'':>5} {'':>9} {'':>8} {'':>8} {total_ms:>9.1f}  TOTAL for 152 calls")
    return total_ms


if __name__ == "__main__":
    torch.npu.set_device(0)
    ms_list = [int(a) for a in sys.argv[1:]] or [2048, 1]
    results = {m: bench(m) for m in ms_list}
    print("\nreference: measured in-model QuantBatchMatmulV3 totals were")
    print("  prefill (M=2048): 251.1 ms over 152 calls")
    print("  decode  (M=1)   :  43.4 ms over 152 calls")
    for m, ms in results.items():
        print(f"  microbench M={m}: {ms:.1f} ms")
