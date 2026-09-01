# Measuring top-k / top-p sampling cost — Qwen3.5-9B on Ascend 310P

Goal: find out how much TTFT / TPOT top-k + top-p sampling costs versus greedy decoding.
Everything below runs on one machine, in one container. Should take ~1.5 h.

---

## 1. Machine

```bash
ssh -p 10002 root@123.60.231.33      # key: ~/.ssh/id_310p
```

8× Atlas 300I Duo (310P3), 2 chips per card, ~43 GB per chip. **Shared box — other teams are
running jobs on it.** Use the `claude_bench_main` container (section 3) and leave every other
container alone.

## 2. Pick a free chip — do this first, every time

```bash
npu-smi info
```

Look at the `Memory-Usage(MB)` column and the process table at the bottom. You need a chip
with **>35 GB free**. Note its device id (0–7); you'll pass it as `-d`.

`npu-smi info` sometimes hangs when the box is loaded. Second opinion, from inside the
container: `python /home/claude_bench/freemem.py`.

## 3. Container

> **Check this first.** As of 2026-08-11 someone pip-installed an in-progress branch as an
> *editable* package into `claude_bench_main`, so it no longer imports the image's stock
> vllm-ascend. Verify before trusting any number as a "main" baseline:
>
> ```bash
> docker exec claude_bench_main pip show vllm_ascend | grep -iE "version|editable"
> ```
>
> If it prints an `Editable project location`, you are running somebody's working copy.
> Make your own container (below) instead of using this one.

Make a clean container from the image:

```bash
docker run -itd --name topk_test --privileged \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home:/home \
  --shm-size=32g \
  quay.io/ascend/vllm-ascend:nightly-main-310p-openeuler bash
docker exec topk_test ln -sfn /home/claude_bench /work
docker exec -it topk_test bash
```

A fresh container gets the image's own site-packages, so it is stock `b2f683ca3` — confirm
with the same `pip show` command; it should report `0.19.1rc2.dev1289+gb2f683ca3` and
**no** editable location.

Image is `quay.io/ascend/vllm-ascend:nightly-main-310p-openeuler` — vllm-ascend main
`b2f683ca3`, vllm 0.25.1, CANN 9.1.0-beta.1. Inside it, `/work` is a symlink to the host's
`/home/claude_bench`, so the scripts are at **`/work/repro/`** and models at `/home/models`.
This container is fine to use — it's ours, not another team's.

If it's stopped: `docker start claude_bench_main`. If it's gone entirely, recreate it:

```bash
docker run -itd --name claude_bench_main --privileged \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home:/home \
  --shm-size=32g \
  quay.io/ascend/vllm-ascend:nightly-main-310p-openeuler bash
docker exec claude_bench_main ln -sfn /home/claude_bench /work
```

`--privileged` is required. Passing `--device=/dev/davinciN` individually fails with
`aclInit ... error 507899` and `torch.npu.device_count() == 0`.

Everything written under `/work` lands on the host at `/home/claude_bench`, so results
survive the container.

## 4. Model

Use the fp16 one so sampling cost isn't mixed up with quantization effects:

```
/home/models/Qwen3.5-9B
```

Vocab is 248 k, which is the whole reason this hypothesis is interesting — top-k/top-p has
to sort/scan a 248 k-wide logit vector every token.

(A W8A8 build exists at `/home/models/Qwen3.5-9B-w8a8-modelslim`; only use it if you also
want the quantized comparison, and add `-q`.)

## 5. Run

The harness is `/work/repro/run_bench.sh`. It starts the server, waits for it, smoke-tests
the output, runs a discarded warmup, sweeps batch sizes, and prints a markdown table.
`-S` passes sampling flags through to the client.

Run these three, one at a time, same device:

```bash
cd /work/repro

# A. greedy baseline
./run_bench.sh -m /home/models/Qwen3.5-9B -d <DEV> -n greedy \
  -S "--temperature 0"

# B. pure top-p
./run_bench.sh -m /home/models/Qwen3.5-9B -d <DEV> -n topp \
  -S "--temperature 1.0 --top-p 0.9"

# C. top-k + top-p
./run_bench.sh -m /home/models/Qwen3.5-9B -d <DEV> -n topk_topp \
  -S "--temperature 1.0 --top-k 50 --top-p 0.9"
```

Defaults: 2048 input / 1024 output tokens, batch sizes 1 2 4 8, `--ignore-eos`, `--seed 0`,
TP=1, max-model-len 4096. Each run is ~25 min (≈8 min server startup + the sweep).

**Check the flag names first** — `vllm bench serve --help | grep -E "top-k|top-p|temperature"`.
If they're absent in this build, send the sampling params per-request instead: hit
`/v1/completions` directly with `{"top_k": 50, "top_p": 0.9, "temperature": 1.0}` in the body
and time it yourself, or patch the `bench()` function inside `run_bench.sh`.

Optional extra points once the three above are in: sweep `--top-k` over 1 / 50 / 1000 /
0 (disabled) at BS=1 to see whether cost depends on k, and try `--top-p 0.99` vs `0.5`.

## 6. Results

Each run writes to `/work/bench_<timestamp>/` and prints its own table. To compare:

```bash
python3 /work/repro/summarize.py /work/bench_greedy /work/bench_topk_topp
```

Report, per batch size: **TTFT mean, TPOT mean, output throughput**, and the delta vs greedy.
TPOT is where sampling shows up — it's per-token work. TTFT should barely move (one sampling
call per request during prefill), so if TTFT moves a lot, something else is going on.

Reference numbers on this box, fp16, greedy, so you know what "normal" looks like:

| BS | TTFT (ms) | TPOT (ms) | output tput (tok/s) |
|---|---|---|---|
| 1 | 2 566.8 | 101.29 | 9.64 |
| 2 | 4 088.3 | 104.70 | 18.40 |
| 4 | 8 584.3 | 110.12 | 33.76 |
| 8 | 13 724.9 | 124.81 | 57.85 |

For scale: at BS=1 the whole decode step is ~62–101 ms, of which the lm_head GEMM alone is
11.5 ms. If sampling costs a few ms/token, that's a few percent — measurable, but you need
the runs to be clean to see it.

## 7. If you want the kernel-level answer too

The end-to-end numbers tell you *how much*; a profile tells you *which op*. There's a working
profiling setup at `/home/claude_bench/profile_run2.sh` + `analyze_prof3.py`. Start the server
with:

```
--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/work/prof_topk
```

(the old `VLLM_TORCH_PROFILER_DIR` env var no longer works in 0.25.1 — it warns
`Unknown vLLM environment variable` and `/start_profile` returns 404), then
`curl -X POST http://127.0.0.1:<PORT>/start_profile`, send one request, `/stop_profile`.

`analyze_prof3.py` splits the trace into prefill and per-decode-step and ranks ops. In the
greedy W8A8 profile the sampling-adjacent ops were small — `Cumsum` 0.57 ms/step,
`SoftmaxV2` 0.16 ms/step. With top-k/top-p you'd expect sort/scan ops on the 248 k vocab to
appear; that's the thing to look for.

## 8. Pitfalls that will waste your day

- **Someone takes your chip mid-startup.** vLLM only checks free memory ~4 min into startup,
  then dies with `Free memory on device (X/42.67 GiB) ... less than desired`. Re-check
  `npu-smi info` and pick another device.
- **Don't lower `-u` below 0.7.** At 0.5 the KV cache was 6 963 tokens and the run thrashed
  from BS=2 (0.8 tok/s). If TPOT explodes at low batch size, grep the server log for
  `GPU KV cache size`.
- **`--ignore-eos` must stay on** (it's already in the script). Sampling changes what gets
  generated; without it, sampled runs stop early and their TPOT is computed over shorter
  outputs than greedy's — you'd be comparing different things.
- **Always run the three configs on the same device, back to back**, and don't run two
  benchmarks at once. Neighbour load moves numbers by ~2 %.
- **The warmup is discarded on purpose** — first request pays TBE JIT and graph capture.
- **Killing a run leaves the chip occupied unless you kill the right process.**
  `pkill -f "vllm serve"` only gets the API server; the engine subprocess is named
  `VLLM::EngineCor` and keeps the memory. After any aborted run, check `npu-smi info`'s
  process table — if your device still shows GB in use at 0 % AICore:

  ```bash
  docker exec claude_bench_main pkill -9 -f EngineCor
  ```

  Confirm the PID is yours first: `head -c 80 /proc/<pid>/cgroup` then
  `docker inspect --format '{{.Name}}' <cgroup-id>`. Never kill another team's process.
- The box has rebooted on its own and stopped answering ssh before. If something hangs at
  `Loading safetensors checkpoint shards: 0%`, suspect the machine, not your config.

## 9. Deliverable

A table: BS × {greedy, top-p, top-k+top-p} × {TTFT, TPOT, output throughput}, plus the
percentage TPOT delta vs greedy. Plus, if profiled, the top ops that appear only in the
sampled runs.
