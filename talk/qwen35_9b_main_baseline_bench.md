# Qwen3.5-9B baseline on main vllm-ascend, 310P — 2048 in / 1024 out (2026-08-06)

Baseline for the rgdr_optim work. **main vllm-ascend, unmodified**, nightly image.

## Setup

| item | value |
|---|---|
| host | `ssh 310p` (123.60.231.33:10002), shared box, 8× Atlas 300I Duo 310P3 |
| container | `claude_bench_main` (mine), `--privileged` (device cgroup pass-through is what works on this box) |
| image | `quay.io/ascend/vllm-ascend:nightly-main-310p-openeuler` (7 days old) |
| vllm-ascend | `b2f683ca3` (`0.19.1rc2.dev1289+gb2f683ca3`) |
| vllm | `752a3a5` (0.25.1) |
| CANN | 9.1.0-beta.1, driver npu-smi 25.5.0 |
| model | `/home/models/Qwen3.5-9B`, `Qwen3_5ForConditionalGeneration` |
| dtype | **float16** — checkpoint is bf16, engine casts (`Casting torch.bfloat16 to torch.float16`); 310P3 has no bf16 |
| device | **device 7** only (`ASCEND_RT_VISIBLE_DEVICES=7`), TP=1 |
| max-model-len | 4096 (310P docs: never let it auto-detect, mask is O(n²) fp16) |
| gpu-mem-util | 0.9 → KV cache **118,784 tokens** |
| max-num-seqs | 16 |
| graph mode | default: FULL_AND_PIECEWISE, capture sizes [1,2,4,8,16] |
| chunked prefill | on, `max_num_batched_tokens=8192` |
| client | `vllm bench serve --dataset-name random --random-input-len 2048 --random-output-len 1024 --random-range-ratio 0 --ignore-eos --seed 0` |
| warmup | discarded (separate 2-prompt run before the measured runs) |

Neighbours: devices 0–5 were running other users' vLLM engines throughout. Device 6 had
1.4 GB of someone's python. Device 7 was idle and stayed mine for the whole measurement.

## Results — concurrency 1 (the clean single-stream number)

10 prompts, sequential. Spread is tiny (P99/P50 TPOT = 1.0002).

| metric | value |
|---|---|
| **TTFT** mean / P50 / P90 / P99 (ms) | **2520.0** / 2521.6 / 2527.5 / 2529.3 |
| **TPOT** mean / P50 / P90 / P99 (ms) | **98.96** / 98.97 / 98.98 / 98.99 |
| ITL mean / P99 (ms) | 98.96 / 100.12 |
| E2E latency mean (ms) | 103 757 |
| **Prefill speed** = 2048 / TTFT | **812.7 tok/s** |
| **Decode speed** = 1000 / TPOT | **10.11 tok/s** |
| output throughput (aggregate) | 9.87 tok/s |
| total token throughput | 29.61 tok/s |
| benchmark duration | 1037.6 s, 10/10 successful |

Decode dominates: 1023 × 98.96 ms = 101.2 s of the 103.8 s request, prefill is 2.4 %.

## Results — concurrency 8 (aggregate throughput context)

16 prompts, `--max-concurrency 8`. Per-request latency degrades, aggregate throughput 6×.

| metric | value |
|---|---|
| TTFT mean / P50 / P90 / P99 (ms) | 13 617 / 12 016 / 23 314 / 23 641 |
| TPOT mean / P50 / P90 / P99 (ms) | 122.2 / 124.4 / 129.4 / 129.6 |
| ITL mean / P50 (ms) | 122.2 / 112.1 |
| E2E latency mean (ms) | 138 648 |
| decode speed **per request** = 1000 / TPOT | 8.18 tok/s |
| **output throughput (aggregate)** | **59.0 tok/s** |
| total token throughput | 177.0 tok/s |
| benchmark duration | 277.7 s, 16/16 successful |

TTFT here is queueing, not prefill — 8 concurrent 2048-token prefills share one chip and
chunked prefill interleaves them with decode. Do not read prefill speed off this row.

## Derivations (so nobody re-derives them differently)

- prefill tok/s = 2048 / TTFT — only meaningful at concurrency 1
- decode tok/s per request = 1000 / TPOT(ms)
- aggregate decode tok/s = `Output token throughput`, ≈ concurrency × per-request rate

## Which GDN path these numbers came from (so the branch is diffable)

At `b2f683ca3` the 310P linear-attention layers run `vllm_ascend/_310p/ops/fla/gdn_310.py`:

- **prefill** → `chunk_gated_delta_rule_310`. Inter-chunk state and output are aclnn custom
  ops (`torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h`, `.chunk_fwd_o`), but the WY prefix
  is still **`_compute_kernel_inputs_from_torch_wy`** (`chunk_gated_delta_rule.py:535`) — the
  63-iteration Python triangular-solve loop. That is exactly what `rgdr_optim` / PR 11941
  replaces, and its cost is inside the 2520 ms TTFT here. See
  [pr11941_perf_profile.md](pr11941_perf_profile.md): op-level that swap was 4.29× on the
  WY step (21.97 ms → 5.12 ms per call at 1×1536×8×16×128×128).
- **decode** → `torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310`, already a custom NPU
  op. Conv1d is `npu_causal_conv1d_310`. No torch fallback in the decode path.

So: **TTFT is where the branch should move; TPOT should not.**

## Roofline check on the decode number

Weights 18 GB (fp16, = safetensors on disk). One decode step reads ~all of them:
18e9 / 98.96 ms = **182 GB/s effective**.

Measured achievable traffic on the same chip (512 MB fp16 device-to-device copy, 20 iters,
`bw.py`): **188 GB/s** (read+write combined). So decode runs at ~97 % of measured memory
traffic — **decode is memory-bandwidth-bound at concurrency 1** and 98.96 ms is close to the
floor for this weight footprint on one 310P3 chip. Kernel work on the decode path cannot buy
much; only quantization (fewer weight bytes) or batching (amortize the read) can.
Concurrency 8 confirms it: 8 streams cost only +23 % TPOT and give 6× aggregate throughput.

(The `a.sum()` read-only figure in `bw.py` is 11.3 GB/s — reduction-limited, not a bandwidth
measurement. Ignore it.)

## Notes / gotchas hit on the way

- **Container must be `--privileged`.** Passing `--device=/dev/davinciN` individually gives
  `aclInit ... 507899 / drv devId is invalid` and `torch.npu.device_count() == 0`. Every
  working container on this box (`scc_dflash_dev`, `z00575061-lite-llm`) uses `--privileged`
  with `-v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/`. Copy that.
- **Pin the device or you collide.** First attempt used device 4; `scc_dflash_dev` grabbed
  devices 4+5 (35 GB each) at the same minute and the engine then died with
  `Free memory on device (8.05/43.24 GiB) ... less than desired`. Check
  `npu-smi info` process table immediately before launching.
- **`--disable-log-requests` is gone** in vllm 0.25.1 — `vllm serve` errors out on it.
- Startup ≈ 5 min cold: weights 23 s, then TBE JIT (`kernel_meta` is written into cwd),
  torch.compile 17 s for range (1,2048), then graph capture. Use `PYTHONUNBUFFERED=1` or the
  EngineCore log looks hung for minutes.
- Model loads and generates coherent text in fp16 (`"The capital of France is Paris."`), so
  the linear-attention (GDN) path has no fp16 overflow problem at this length. 310P hybrid
  support is real: `patch_mamba_config_310.py` pads mamba page size to match attention page
  size (attention block size forced to 640 tokens, mamba page padded 22.14 %).

## Caveats

- Text-only. The model is registered as `ForConditionalGeneration` and the engine still
  profiles the vision encoder (16384-token encoder budget) at startup; that costs startup
  memory but not decode time.
- One measured run per concurrency point, but there is an independent second data point at
  c1: the discarded 2-prompt warmup gave TTFT 2519.32 / TPOT 98.85 vs the measured
  2520.01 / 98.96 — **0.03 % / 0.11 % run-to-run**. Within-run P99/P50 TPOT is 1.0002.
- Card-mate not verified idle. Device 6 is the other chip of the same Atlas 300I Duo card and
  had 1.4 GB of someone's python during the run (0 % AICore whenever sampled; both chips were
  free by the end). The two chips share card bandwidth, and decode is bandwidth-bound, so a
  busy card-mate would inflate TPOT. The 0.02 % within-run TPOT spread argues it stayed idle.
  On any comparison run, sample NPU 6 chip 0 AICore % alongside.
- c8 ran at the default request rate `inf` (all 16 dispatched at once, gated only by
  `--max-concurrency 8`) — that is why its TTFT is queueing rather than prefill.
- Image is 7 days old, not today's main.

## W8A8 attempt — BLOCKED on 310P at this commit (2026-08-06)

Asked for the same run on a W8A8 Qwen3.5-9B. **No W8A8 Qwen3.5-9B exists in an
Ascend-consumable format.** What's on ModelScope for this model is llm-compressor /
compressed-tensors only: `RedHatAI/Qwen3.5-9B-quantized.w8a8`, `iluvatar-corex/…`,
`sogagaga/…`. Eco-Tech (the msModelSlim publisher Ascend docs point at) ships Qwen3.5 w8a8
for 27B / 35B-A3B / 122B-A10B / 397B-A17B — **not 9B**. Nothing on the box either
(searched `/home/models`, all `/home/*` depth 4, `/tmp/models_copy`, `/root/.modelscope`).

Downloaded `RedHatAI/Qwen3.5-9B-quantized.w8a8` → `/home/models/Qwen3.5-9B-w8a8` (14 GB,
kept). Three walls, in order:

1. **Stock main refuses it on 310P.** `platform.py:198` registers the Ascend
   compressed-tensors config only `if not is_310p()`; 310P gets `AscendModelSlimConfig310`
   only. So vLLM's own `CompressedTensorsConfig` handles it, picks a CUDA scaled_mm kernel,
   and dies: `KeyError: <PlatformEnum.OOT: 6>` in
   `vllm/model_executor/kernels/linear/__init__.py:549`.
2. Patched that gate to also register `AscendCompressedTensorsConfig` on 310P (reverted
   after). It engaged — "Using the vLLM Ascend llmcompressor Quantization now!", weights
   loaded 14.62 GB — then died in the profile run:
   `npu_quant_matmul → aclnnQuantMatmulWeightNz failed, error code 161002`. The NZ weight
   layout that path assumes is not supported here.
3. With `VLLM_ASCEND_ENABLE_NZ=0`: `Tensor scale not implemented for DT_FLOAT16, should be
   in [DT_UINT64, DT_BFLOAT16, DT_INT64, DT_FLOAT]`. The path builds fp16 scales; 310P must
   run fp16 (no bf16), so it can't satisfy the op.

Read together: **vllm-ascend's compressed-tensors W8A8 is written for bf16 on 910B.** 310P's
supported quant path is msModelSlim (`quant_model_description.json`, `--quantization ascend`),
implemented separately in `vllm_ascend/_310p/quantization/methods/`
(`w8a8_dynamic`, `w8a8_static`, `w8a8s`, `w8a8sc`).

To get a 310P W8A8 number for this model, someone has to **quantize Qwen3.5-9B with
msModelSlim** — `/home/z00575061` has a working recipe for Qwen3.5-2B W8A8 plus
`static_w8a8_fp16_fallback` experiments to crib from. Not a download.

### msModelSlim quantization — DONE; serving it is not (2026-08-07)

Quantized it. `/home/claude_bench/quant_w8a8.py`, adapted from
`/home/z00575061/modelslim_quant_qwen3_reranker4b_repro.py`:

- msModelSlim source at `/home/c00692241/msit/msmodelslim`; needs `pip install accelerate
  easydict` in the vllm-ascend nightly image.
- Load via `AutoModelForImageTextToText` (transformers 5.14 maps `qwen3_5` →
  `Qwen3_5ForConditionalGeneration`; `AutoModelForCausalLM` gives the text-only
  `Qwen3_5ForCausalLM` and would drop the vision tower).
- 359 Linears total; **111 disabled** (all `visual.*` + `lm_head`) because text-only
  calibration never exercises them, **248 quantized**. Calib: 24 samples × 512 tokens.
- `QuantConfig(w_bit=8, a_bit=8, w_sym=True, act_method=1, w_method="MinMax",
  is_dynamic=False, disable_last_linear=True)`, `disable_level="L0"`, dev_type npu.
- Calibration 105 s, save 51 s, **157 s total** on one 310P3. Output
  `/home/models/Qwen3.5-9B-w8a8-modelslim`, 12 GB, 1737 W8A8 / 512 FLOAT entries.
- msModelSlim writes `quant_model_description_w8a8.json`; vllm-ascend looks for
  **`quant_model_description.json`** (`quantization/modelslim_config.py:59`) — copy it.

First serve attempt hung at `Loading safetensors checkpoint shards: 0%` for 35+ min. **That
was the box, not the checkpoint** — it rebooted twice over the next hour (`npu-smi`, `ps`,
`py-spy` were all wedged box-wide at the time). Retried unchanged on a fresh boot and it came
up in ~8 min. No `VLLM_ASCEND_ENABLE_NZ=0` needed, stock config.

Serving it needs `--quantization ascend`; log confirms `Using vLLM Ascend ModelSlim
quantization.` Output is coherent (`"The capital of France is Paris."`), so the W8A8 weights
are sane.

## W8A8 results — same protocol, device 2, gpu-mem-util 0.5

Only differences from the fp16 rows: `--quantization ascend`, `--gpu-memory-utilization 0.5`
(neighbours were filling chips; 0.9 wouldn't start), device 2 instead of 7. KV cache is
therefore smaller, which does not affect TTFT/TPOT at these concurrencies.

| metric (concurrency 1, 10 prompts) | fp16 | **W8A8** | change |
|---|---|---|---|
| TTFT mean (ms) | 2520.0 | **2020.1** | −19.8 % |
| **prefill speed** (2048/TTFT) | 812.7 tok/s | **1013.8 tok/s** | **1.25×** |
| TPOT mean (ms) | 98.96 | **62.51** | −36.8 % |
| **decode speed** (1000/TPOT) | 10.11 tok/s | **16.00 tok/s** | **1.58×** |
| E2E per request (ms) | 103 757 | **65 964** | −36.4 % |
| output throughput (tok/s) | 9.87 | **15.52** | 1.57× |
| total token throughput (tok/s) | 29.61 | **46.57** | 1.57× |

W8A8 percentiles are as tight as fp16's: TTFT P99 2033.8, TPOT P99 62.55 (P99/P50 = 1.0006).
Warmup run agrees with the measured run to 0.4 % / 0.3 % (TTFT 2011.9, TPOT 62.31).

**The decode speedup lands where the roofline predicted.** Weights drop 18 GB → ~12 GB
(1.5×) and decode got 1.58× faster — decode was memory-bandwidth-bound, so cutting weight
bytes converts almost 1:1 into TPOT. Prefill gains much less (1.25×) because it is not
bandwidth-bound; it is still carrying the Python WY loop.

Concurrency 8 (16 prompts) was launched right after c1 and its result was not retrieved — the
box stopped accepting ssh again mid-run. Log is at
`/home/claude_bench/bench_slim_c8.log` (plus `result_slim_c8.json` if it finished); pick it up
when the box is back. The c1 row above is complete and is the number that matters.

## Reproduce

Everything is in `/home/claude_bench` on `310p` (container `claude_bench_main`, left up,
server stopped, device 7 released):
`serve.sh` (DEV/PORT/MAXSEQS env), `bench.sh <tag> <num_prompts> <concurrency>`,
`bw.py`, `result_{c1,c8,warmup}.json`, `bench_*.log`, `serve.log`.
