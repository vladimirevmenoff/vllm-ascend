# Reproducing the Qwen3.5-9B TTFT / TPOT benchmark on Ascend 310P

Two files: `run_bench.sh` (starts a server, warms it, sweeps batch sizes) and
`summarize.py` (turns the result JSONs into a markdown table). Both run **inside** the
vllm-ascend container.

Already on the box at `/home/claude_bench/repro/` (`ssh 310p`, container
`claude_bench_main`).

## 1. Container

`--privileged` is required. Mapping devices individually with `--device=/dev/davinciN`
fails with `aclInit ... error 507899` and `torch.npu.device_count() == 0`.

```bash
docker run -itd --name bench --privileged \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home/models:/home/models:ro \
  -v /home/bench_work:/work \
  --shm-size=32g \
  quay.io/ascend/vllm-ascend:nightly-main-310p-openeuler bash

docker cp run_bench.sh bench:/work/
docker cp summarize.py bench:/work/
docker exec -it bench bash
```

Numbers in `talk/qwen35_9b_bs_sweep.md` came from that image at vllm-ascend `b2f683ca3`
(`0.19.1rc2.dev1289+gb2f683ca3`) / vllm `752a3a5` (0.25.1), CANN 9.1.0-beta.1.
Check what you actually have with `pip show vllm-ascend` and
`git -C /vllm-workspace/vllm-ascend log --oneline -1`.

## 2. Run

```bash
# fp16 baseline
/work/run_bench.sh -m /home/models/Qwen3.5-9B -d 0 -n fp16

# W8A8 (msModelSlim checkpoint; -q adds --quantization ascend)
/work/run_bench.sh -m /home/models/Qwen3.5-9B-w8a8-modelslim -d 1 -p 8200 -q -n w8a8
```

Defaults reproduce the published run: 2048 in / 1024 out, BS 1/2/4/8, `--ignore-eos`,
`--seed 0`, fp16 activations, TP=1, max-model-len 4096, util 0.7, max-num-seqs 16, one
discarded warmup. Options:

| flag | meaning | default |
|---|---|---|
| `-m PATH` | model directory | required |
| `-d N` | NPU device, pinned via `ASCEND_RT_VISIBLE_DEVICES` | 0 |
| `-p N` | server port | 8100 |
| `-q` | quantized checkpoint → `--quantization ascend` | off (fp16) |
| `-b "1 2 4 8"` | batch sizes to sweep | `1 2 4 8` |
| `-i N` / `-o N` | input / output tokens | 2048 / 1024 |
| `-u F` | gpu-memory-utilization | 0.7 |
| `-s N` | max-num-seqs | 16 |
| `-l N` | max-model-len | 4096 |
| `-n NAME` | run label used in filenames | bench |
| `-O DIR` | output directory | `/work/bench_<timestamp>` |

What it does, in order: print free memory per device → start the server → wait for
`Application startup complete` (up to 30 min; cold start is 5–8 min) → **smoke test that the
model emits sensible text**, abort if not → discarded warmup → sweep → kill server → print
the table. Each point writes `<name>_bs<N>.json` plus the raw bench log.

Runtime at defaults: ~8 min startup + ~2 × 1024 × TPOT per point. The fp16 sweep took ~25 min,
W8A8 ~17 min.

## 3. Table

`run_bench.sh` prints it at the end. To regenerate, or to compare two runs:

```bash
python3 /work/summarize.py /work/bench_fp16                     # one table
python3 /work/summarize.py /work/bench_fp16 /work/bench_w8a8    # + speedup table
```

Derived metrics, so nobody re-derives them differently:

- **prefill tok/s = input_len / TTFT** — only meaningful at BS=1. Above that TTFT is queueing:
  requests are dispatched at rate `inf`, gated only by `--max-concurrency`.
- **decode tok/s (per request) = 1000 / TPOT(ms)**
- **aggregate decode = `output_throughput`** ≈ BS × per-request rate

## 4. Things that will bite you

- **Check free memory before launching.** It's a shared box; others grab chips without
  warning. vLLM's own check happens ~4 min into startup and then dies with
  `Free memory on device (X/42.67 GiB) ... less than desired`. `run_bench.sh` prints free
  memory per device up front — read it and pick with `-d`. `npu-smi info` is a second opinion
  but hangs when the box is loaded.
- **Don't starve the KV cache.** At `-u 0.5` the W8A8 model got only 6 963 KV tokens and
  thrashed from BS=2 (0.8 tok/s, one request running and one waiting). At 0.7 it gets 99 532.
  If TPOT explodes at low batch size, grep the server log for `GPU KV cache size` first.
- **`--ignore-eos` is not optional.** Without it generations stop early and TPOT is computed
  over sub-1024-token outputs.
- **Always discard a warmup.** The first request pays TBE JIT and graph capture.
- **310P is fp16.** The checkpoint is bf16; the engine casts (`Casting torch.bfloat16 to
  torch.float16`). Never omit `--max-model-len` on 310P — the attention mask is
  O(max_model_len²) fp16 and auto-detection OOMs.
- **Compare like with like.** KV cache size differs between fp16 and W8A8 at the same
  utilization (smaller weights leave more room). That does not affect TTFT/TPOT as long as
  neither run preempts — check `Waiting: 0 reqs` in the server log.
- This host rebooted three times during the original measurements and repeatedly stopped
  answering ssh. If a run hangs at `Loading safetensors checkpoint shards: 0%`, suspect the
  box before the checkpoint.

## 5. Making the W8A8 checkpoint

Only if you don't have `/home/models/Qwen3.5-9B-w8a8-modelslim`. Script:
`/home/claude_bench/quant_w8a8.py`, ~3 minutes. Note compressed-tensors / llm-compressor
checkpoints (RedHatAI etc.) **do not work on 310P** — vllm-ascend registers that config only
`if not is_310p()` (`platform.py:198`). msModelSlim is the only format 310P loads.

```bash
pip install accelerate easydict      # msModelSlim deps, missing from the image
ASCEND_RT_VISIBLE_DEVICES=0 python /home/claude_bench/quant_w8a8.py
cd /home/models/Qwen3.5-9B-w8a8-modelslim
cp quant_model_description_w8a8.json quant_model_description.json   # the name vllm-ascend looks for
```

Details and the three non-obvious bits (auto-class choice, which Linears to leave FLOAT, the
filename) are in `talk/qwen35_9b_bs_sweep.md`.
