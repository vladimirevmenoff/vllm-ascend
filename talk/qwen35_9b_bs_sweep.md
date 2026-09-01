# Qwen3.5-9B on Ascend 310P — fp16 vs W8A8, batch-size sweep (2026-08-07)

2048 input / 1024 output tokens, main vllm-ascend, one 310P3 chip.

## What was done

1. **Baseline, fp16.** Served `/home/models/Qwen3.5-9B` (bf16 checkpoint, engine casts to fp16
   — 310P3 has no bf16) on stock main vllm-ascend, TP=1, one chip.
2. **W8A8 checkpoint had to be made.** No W8A8 Qwen3.5-9B exists in a format 310P can load.
   ModelScope only has llm-compressor / compressed-tensors builds (RedHatAI, iluvatar-corex,
   sogagaga); vllm-ascend registers its compressed-tensors config **only when not 310P**
   (`platform.py:198`), so those die with `KeyError: <PlatformEnum.OOT: 6>`. Forcing the path
   on fails further down (`aclnnQuantMatmulWeightNz` 161002; with NZ off,
   `Tensor scale not implemented for DT_FLOAT16` — that path assumes bf16). Eco-Tech, the
   msModelSlim publisher, ships Qwen3.5 w8a8 for 27B/35B-A3B/122B/397B but not 9B.
3. **Quantized it with msModelSlim** (`/home/claude_bench/quant_w8a8.py`): 248 of 359 Linears
   to W8A8, the 111 `visual.*` + `lm_head` Linears left FLOAT (text-only calibration never
   exercises them). Calib 24 samples × 512 tokens. 157 s end to end. Output
   `/home/models/Qwen3.5-9B-w8a8-modelslim`, 12 GB. Served with `--quantization ascend`.
   Sanity-checked the generations before trusting any timing.
4. **Swept batch size 1 / 2 / 4 / 8** on both, identical protocol: `vllm bench serve`,
   `--dataset-name random --random-input-len 2048 --random-output-len 1024
   --random-range-ratio 0 --ignore-eos --seed 0`, `2×concurrency` prompts (min 4), a discarded
   warmup run before each sweep.

## Configuration

| item | value |
|---|---|
| vllm-ascend | `main` @ `b2f683ca3` (`0.19.1rc2.dev1289+gb2f683ca3`) |
| vllm | `752a3a5` (0.25.1) |
| image | `quay.io/ascend/vllm-ascend:nightly-main-310p-openeuler` |
| CANN / driver | 9.1.0-beta.1 / npu-smi 25.5.0 |
| hardware | 1× Atlas 300I Duo chip (310P3), TP=1 |
| dtype | float16 (both runs) |
| max-model-len | 4096 |
| gpu-memory-utilization | 0.7 (both) |
| max-num-seqs | 16 |
| KV cache | fp16 30,720 tokens · W8A8 99,532 tokens |
| weights on device | fp16 ~18 GB · W8A8 ~12 GB |

## Combined — fp16 vs W8A8

### Latency

| BS | TTFT fp16 (ms) | TTFT W8A8 (ms) | TTFT ↑ | TPOT fp16 (ms) | TPOT W8A8 (ms) | TPOT ↑ |
|---|---|---|---|---|---|---|
| 1 | 2 566.8 | 2 000.1 | **1.28×** | 101.29 | 62.42 | **1.62×** |
| 2 | 4 088.3 | 3 193.9 | 1.28× | 104.70 | 65.63 | 1.60× |
| 4 | 8 584.3 | 6 870.2 | 1.25× | 110.12 | 70.30 | 1.57× |
| 8 | 13 724.9 | 11 021.1 | 1.25× | 124.81 | 82.87 | 1.51× |

### Throughput

| BS | decode/req fp16 | decode/req W8A8 | output fp16 | output W8A8 | total fp16 | total W8A8 | ↑ |
|---|---|---|---|---|---|---|---|
| 1 | 9.87 tok/s | 16.02 tok/s | 9.64 tok/s | 15.55 tok/s | 28.93 tok/s | 46.65 tok/s | **1.61×** |
| 2 | 9.55 | 15.24 | 18.40 | 29.10 | 55.21 | 87.29 | 1.58× |
| 4 | 9.08 | 14.22 | 33.76 | 51.95 | 101.28 | 155.85 | 1.54× |
| 8 | 8.01 | 12.07 | 57.85 | 85.40 | 173.55 | 256.20 | 1.48× |

### Tails and end-to-end

| BS | TTFT P99 fp16 | TTFT P99 W8A8 | TPOT P99 fp16 | TPOT P99 W8A8 | E2E fp16 | E2E W8A8 |
|---|---|---|---|---|---|---|
| 1 | 2 572.6 ms | 2 006.5 ms | 101.30 ms | 62.43 ms | 106.2 s | 65.9 s |
| 2 | 5 363.5 | 4 205.9 | 106.00 | 66.67 | 111.2 s | 70.3 s |
| 4 | 11 399.3 | 9 119.4 | 112.92 | 72.58 | 121.2 s | 78.8 s |
| 8 | 23 787.8 | 19 122.2 | 132.32 | 88.93 | 141.4 s | 95.8 s |

## fp16 (raw)

| BS | TTFT mean (ms) | TTFT P99 (ms) | TPOT mean (ms) | TPOT P99 (ms) | decode/req (tok/s) | output tput (tok/s) | total tput (tok/s) | E2E (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 566.8 | 2 572.6 | 101.29 | 101.30 | 9.87 | 9.64 | 28.93 | 106 182 |
| 2 | 4 088.3 | 5 363.5 | 104.70 | 106.00 | 9.55 | 18.40 | 55.21 | 111 199 |
| 4 | 8 584.3 | 11 399.3 | 110.12 | 112.92 | 9.08 | 33.76 | 101.28 | 121 238 |
| 8 | 13 724.9 | 23 787.8 | 124.81 | 132.32 | 8.01 | 57.85 | 173.55 | 141 407 |

## W8A8 (raw)

| BS | TTFT mean (ms) | TTFT P99 (ms) | TPOT mean (ms) | TPOT P99 (ms) | decode/req (tok/s) | output tput (tok/s) | total tput (tok/s) | E2E (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 000.1 | 2 006.5 | 62.42 | 62.43 | 16.02 | 15.55 | 46.65 | 65 854 |
| 2 | 3 193.9 | 4 205.9 | 65.63 | 66.67 | 15.24 | 29.10 | 87.29 | 70 335 |
| 4 | 6 870.2 | 9 119.4 | 70.30 | 72.58 | 14.22 | 51.95 | 155.85 | 78 788 |
| 8 | 11 021.1 | 19 122.2 | 82.87 | 88.93 | 12.07 | 85.40 | 256.20 | 95 795 |

## W8A8 vs fp16

| BS | TTFT | TPOT | output throughput |
|---|---|---|---|
| 1 | **1.28×** | **1.62×** | **1.61×** |
| 2 | 1.28× | 1.60× | 1.58× |
| 4 | 1.25× | 1.57× | 1.54× |
| 8 | 1.25× | 1.51× | 1.48× |

## Prefill / decode speed

Prefill speed is only meaningful at BS=1 — from BS=2 up, TTFT is dominated by queueing
(requests are dispatched at rate `inf`, gated only by `--max-concurrency`) and by chunked
prefill interleaving with decode.

| | fp16 | W8A8 | ↑ |
|---|---|---|---|
| prefill (2048 / TTFT @ BS=1) | 797.9 tok/s | **1 023.9 tok/s** | 1.28× |
| decode (1000 / TPOT @ BS=1) | 9.87 tok/s | **16.02 tok/s** | 1.62× |
| weights on device | ~18 GB | ~12 GB | 1.5× fewer bytes |
| KV cache | 30 720 tok | 99 532 tok | — |

## Reading it

- **Decode is memory-bandwidth-bound, and W8A8 confirms it.** Weights drop 18 → 12 GB
  (1.5×) and TPOT improves 1.62× at BS=1. Measured achievable traffic on this chip is
  188 GB/s (512 MB device-to-device copy); fp16 decode runs at 18e9/101.3 ms = 178 GB/s,
  i.e. ~95 % of it. There is no headroom left in the decode kernels — only fewer weight bytes
  or bigger batches help.
- **Prefill gains far less (1.25×)** because it is not bandwidth-bound. Main's 310P prefill
  still computes the WY prefix with `_compute_kernel_inputs_from_torch_wy` — a 63-iteration
  Python triangular-solve loop (`_310p/ops/fla/chunk_gated_delta_rule.py:535`); only fwd_h and
  chunk_fwd_o are aclnn ops. That loop is what `rgdr_optim` replaces, and it is where the
  remaining TTFT headroom is.
- **Batching is cheap on the decode side:** BS 1→8 costs 23 % TPOT (fp16) / 33 % (W8A8) and
  returns 6.0× / 5.5× aggregate throughput. The cost of batching shows up in TTFT, not TPOT.
- W8A8 degrades slightly faster with batch size (1.62× → 1.51×): as batch grows, decode
  shifts from weight-bandwidth-bound toward compute/activation-bound, so the weight-size win
  shrinks.

## Caveats

- **The fp16 sweep shared the box with a neighbouring run** (on a different card); the W8A8
  sweep ran clean. Solo fp16 at BS=1 measured 98.96 ms TPOT the previous day vs 101.29 ms
  here (+2.3 %), so the speedup column is likely overstated by roughly that much.
- KV cache sizes differ (30,720 vs 99,532 tokens) because W8A8 weights leave more room at the
  same utilization. Neither run preempted at BS=8 (8 × 3072 = 24,576 tokens fits both), so
  this does not affect TTFT/TPOT.
- An earlier W8A8 attempt at `--gpu-memory-utilization 0.5` gave only 6,963 KV tokens and
  thrashed from BS=2 (0.8 tok/s, one request running and one waiting). Discarded and rerun at
  0.7. If TPOT explodes at low batch size, check `GPU KV cache size` first.
- One measured run per point. Within-run spread is tiny (BS=1 TPOT P99/P50 = 1.0002 fp16,
  1.0002 W8A8) and the discarded warmups agree with the measured runs to <0.5 %.
- The host rebooted three times during this work and repeatedly stopped answering ssh; an
  earlier "hang" attributed to the W8A8 checkpoint turned out to be the box, not the model.
- Text-only. The vision tower is present but never exercised, and is left in FLOAT in the
  quantized checkpoint.

## Artefacts

On `ssh 310p`, `/home/claude_bench` (container `claude_bench_main`):
`quant_w8a8.py`, `sweep2.sh`, `serve_fp16.sh` / `serve_slim.sh`, `bench_fp16.sh` /
`bench_slim2.sh`, `fp16_c{1,2,4,8}.log`, `slimb_c{1,2,4,8}.log`, `result_*.json`, `bw.py`.
