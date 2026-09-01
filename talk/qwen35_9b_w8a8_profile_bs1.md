# Qwen3.5-9B W8A8 — device profile at BS=1 (2026-08-07)

torch-npu profiler via vLLM, one 310P3 chip, 2048 in / 32 out, main vllm-ascend `b2f683ca3`.
Capture is one full request: **1 prefill pass + 31 decode steps**, 62 934 kernels, 3.9 s.

Profiling overhead is negligible — the profiled request measured TTFT 2010.29 ms /
TPOT 62.25 ms against 2000.1 / 62.42 unprofiled.

## How it was captured and split

Server started with `--profiler-config.profiler=torch
--profiler-config.torch_profiler_dir=…` (the old `VLLM_TORCH_PROFILER_DIR` env var is gone in
0.25.1 — it warns `Unknown vLLM environment variable` and the `/start_profile` route 404s).
A warmup run first, then `/start_profile` → one request → `/stop_profile`.

Time-based segmentation does not work: the device is busy 3818 of 3898 ms, so steps have no
idle gaps between them. Instead passes are split at **`MatMulV2`, which occurs exactly 32
times** — once per forward pass (it is the lm_head GEMM). That yields 1 prefill + 31 decode
passes, and the per-pass device time matches the measured latencies.

## Where the time goes

| phase | device-busy | measured | device share |
|---|---|---|---|
| prefill (2048 tokens) | 1 938.1 ms | TTFT 2 010.3 ms | 96.4 % |
| decode (per token) | 60.62 ms | TPOT 62.25 ms | 97.4 % |

Both phases are essentially all device time — host/scheduling is ~41 ms on prefill and
~1.6 ms per decode token. There is nothing to win on the host side.

## Decode — per step (62.25 ms), 48 484 kernels over 31 steps

| op | ms/step | share | count/step | note |
|---|---|---|---|---|
| **QuantBatchMatmulV3** | **43.39** | **71.6 %** | 152 | the W8A8 linear layers |
| **MatMulV2** | **11.48** | **18.9 %** | 1 | **lm_head — still fp16** |
| RecurrentGatedDeltaRuleV310 | 0.92 | 1.5 % | 24 | GDN decode op |
| RmsNorm | 0.67 | 1.1 % | 153 | |
| paged_attention_decoder_mask_0 | 0.64 | 1.1 % | 8 | the 8 full-attention layers |
| Cumsum | 0.57 | 0.9 % | 1 | |
| AscendQuantV2 | 0.49 | 0.8 % | 152 | activation quantization |
| SwiGlu | 0.29 | 0.5 % | 32 | |
| everything else | ~2.2 | 3.6 % | | |

Pipe mix: **mte2 84 %** of aicore time, mac 5 %, scalar 18 %; **96 % of busy time is in
kernels the profiler marks memory-bound.** Decode is pure weight streaming, as the roofline
predicted.

### The actionable finding: lm_head is unquantized and costs 18.9 % of every token

I left `lm_head` in FLOAT during quantization (along with the vision tower). It is
248 k vocab × 4096 × 2 B = **2.03 GB read per token**, and 11.48 ms of that read works out to
177 GB/s — right at this chip's 188 GB/s measured ceiling. Quantizing it to W8A8 halves those
bytes:

| | now | lm_head W8A8 (projected) |
|---|---|---|
| TPOT | 62.25 ms | ~56.5 ms |
| decode speed | 16.0 tok/s | ~17.7 tok/s (+11 %) |

That is the single cheapest remaining decode win, and it costs one line in the quantization
script (drop `lm_head` from `disable_names`, keep `disable_last_linear=False`). Accuracy of a
quantized lm_head should be checked, which is why it was excluded in the first place.

## Prefill — one pass over 2048 tokens (1 938 ms), 14 440 kernels

| op | ms | share | count | note |
|---|---|---|---|---|
| QuantBatchMatmulV3 | 250.70 | 12.9 % | 152 | the actual linear algebra |
| Slice | 178.15 | 9.2 % | 3 219 | |
| TransData | 166.91 | 8.6 % | 521 | ND↔NZ format conversion |
| Tril | 133.44 | 6.9 % | 48 | |
| ViewCopy | 127.58 | 6.6 % | 1 516 | |
| Mul | 119.12 | 6.1 % | 1 712 | |
| ReduceSum | 115.21 | 5.9 % | **1 512** | |
| ChunkFwdO | 106.86 | 5.5 % | 24 | aclnn GDN op |
| ChunkGatedDeltaRuleFwdH | 73.08 | 3.8 % | 24 | aclnn GDN op |
| RmsNorm | 67.54 | 3.5 % | 153 | |
| Cast | 63.53 | 3.3 % | 412 | |
| UnpadFlashAttentionNzEncoderKernel | 61.34 | 3.2 % | 8 | the 8 full-attention layers |
| CausalConv1dV310 | 56.80 | 2.9 % | 24 | |
| SwiGlu | 50.71 | 2.6 % | 32 | |
| RepeatInterleave | 50.08 | 2.6 % | 24 | |

Pipe mix: **scalar 45 %** of aicore time, mte2 54 %, mac only 12 %. Prefill is issue-bound and
format-shuffle-bound, not compute-bound.

### The Python WY loop is visible in the kernel counts

**`ReduceSum` appears exactly 1 512 times = 63 × 24** — the 63-iteration Python triangular
solve in `_compute_kernel_inputs_from_torch_wy`, run once per linear-attention layer (24 of
the 32 layers are linear attention). `ViewCopy` (1 516) and `Mul` (1 712) track the same loop,
and `Slice` (3 219 ≈ 2 × 1 512) is its indexing.

Summing that cluster — Slice + ViewCopy + Mul + ReduceSum + Tril + part of Cast —
**≈ 670–790 ms, or 35–41 % of prefill**, spent shuffling small tensors instead of computing.
Meanwhile the two aclnn GDN ops that do the real chunked work (`ChunkFwdO`,
`ChunkGatedDeltaRuleFwdH`) total only 180 ms, and all the quantized GEMMs together are 251 ms.

This is the same finding as the op-level A/B in
[pr11941_perf_profile.md](pr11941_perf_profile.md), now measured in situ at the model level:
**the torch WY prefix is the largest single item in prefill**, and it is exactly what
`rgdr_optim` replaces. `TransData` (167 ms, 8.6 %) is the next target — ND↔NZ conversions
around the quantized GEMMs.

## Ranked opportunities

1. **Prefill: replace the torch WY loop** — ~670–790 ms of 1 938 ms (35–41 % of TTFT).
   Already the point of `rgdr_optim`; this profile sizes the prize.
2. **Decode: quantize lm_head** — 11.48 ms of 62.25 ms per token; halving it is ~+11 % decode
   throughput for a one-line change (pending an accuracy check).
3. **Prefill: cut TransData** — 167 ms (8.6 %) in pure format conversion.
4. Decode's remaining 71.6 % is `QuantBatchMatmulV3` at 84 % mte2 and 96 % memory-bound —
   there is no kernel-level win there, only fewer weight bytes (W4A8/W4A16) or larger batches.

## Artefacts

`ssh 310p`, `/home/claude_bench`: raw trace `prof_w8a8/rank0_11826_*_ascend_pt/`
(`ASCEND_PROFILER_OUTPUT/kernel_details.csv`, 62 935 rows), analysis script
`analyze_prof3.py`, driver `profile_run2.sh`, server `serve_slim_prof2.sh`.
