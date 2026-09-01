# Qwen3.5-9B on Ascend 310P — project handbook

Single source of truth for env, build, test, inference. Written 2026-09-01 before
context compaction. Details of the optimization journey: `qwen9b_optim_20260819_status.md`
and `csrc/moe/chunk_gated_delta_rule_compute_wy/workbench/notes/progress_log.md`.

## Current state (2026-09-01)

- Goal: Qwen3.5-9B decode ≤30 ms/tok, prefill TTFT <1 s. Work anchored to
  https://github.com/vllm-project/vllm-ascend/pull/13122 (head branch `pr-11941`,
  fork `vladimirevmenoff/vllm-ascend`).
- Decode: 61.9 → 37.9 ms/tok done earlier (lm_head W8A8 fix + capture-safe splitfuse
  mask + MTP k=3). 30 ms needs >1 chip — parked.
- Prefill: `chunk_gated_delta_rule_compute_wy` kernel optimized 4691 → ~800 µs
  (787-843 quiet runs, 5.8x) on canonical shape 1,1024,8,16,128,128. All 9 battery
  shapes cos ≥ 0.9999. PR 13122 head = `4faebc035` (single squashed commit).
- Serving TTFT validated 2026-09-01 with the ~800 µs kernel: **1323.5 ms** mean
  (1359 p-high, BS=1, 2048-tok prompts, no MTP) vs 1467 ms with the 4691 µs kernel
  — −144 ms, consistent with per-layer kernel savings. TTFT <1 s NOT reached:
  compute_wy no longer dominates prefill; next win needs a fresh prefill profile.
  Decode in same run: 61.9 ms/tok (bench default = no MTP; MTP k=3 config → 37.9).
- Local branch: `310p-chunk_gated_delta_rule_compute_wy-skill-port` (full commit
  history); PR squash branch: `pr13122-kernel-opt`.

## Boxes and access

- `ssh 310p-a` — dev box (123.60.231.33:10005). Shared with other teams: check
  `npu-smi info` process table before pinning a device or trusting timings
  (a `d2d_memcpy`/python on the same card inflates timings ~2.7x).
- Our container: `claude_13122` (dies sometimes → `docker start claude_13122`).
  Our device: 6 (`ASCEND_RT_VISIBLE_DEVICES=6`); device 7 = same card, avoid.
- `ssh 310p` — older box, container `claude_bench_main` (main-branch baseline work).
- Box network drops sporadically; retry with backoff; always size-verify scp uploads.
- New container: `talk/repro/make_container.sh <name>` on the HOST. `--privileged`
  is mandatory (per-device mapping → aclInit 507899).

## Container layout (claude_13122)

- Repo: `/vllm-workspace/vllm-ascend-combined` (editable pip install of vllm_ascend).
- `/work` = bind mount of host `/home/claude_bench`.
- Kernel harness: `/work/wy_harness/` — see below.
- Models: `/home/models/Qwen3.5-9B` (fp16), `/home/models/Qwen3.5-9B-w8a8-modelslim`
  (W8A8, msModelSlim only — compressed-tensors checkpoints DO NOT run on 310P).
- torch_npu MUST stay 2.10.0.post2 (post4 breaks aclnnMatmulWeightNz 161002).
- CANN 9.1.0-beta.1; toolkit at `/usr/local/Ascend/ascend-toolkit/latest`.

## Building the op (fast iteration)

MANDATORY hygiene every rebuild (stale-cache traps, cost us ~6 h once):
```bash
cd /vllm-workspace/vllm-ascend-combined/csrc
rm -rf build build_out
bash build.sh --pkg --soc=ascend310p --ops=chunk_gated_delta_rule_compute_wy
cd build && ./cann-ops-transformer-custom_linux-aarch64.run --quiet --install-for-all
rm -rf /root/atc_data
```
- Ship op_host + op_kernel TOGETHER (a stale tiling.cpp on the box once paired every
  kernel with a broken 8 KB workspace for a day).
- Targeted `--ops=` install REPLACES the CANN vendor with a single-op package —
  serving breaks until a FULL rebuild (below). Tree vendor
  (`vllm_ascend/_cann_ops_custom/vendors/custom_transformer`) shadows CANN vendor for
  serving; sync kernel .o + libcust_opapi.so after installs, or full-rebuild.
- The kernel .o filename hash is a signature, not a content hash — md5sum to verify.

Full rebuild (regenerates tree vendor with ALL ops — required before serving):
```bash
cd /vllm-workspace/vllm-ascend-combined && pip install -e . --no-build-isolation  # ~20 min
```

## Kernel test harness (no serving needed)

`/work/wy_harness/run_fast.py` — cached torch goldens, cosine + 3 timings per shape:
```bash
docker exec claude_13122 bash -c \
  'ASCEND_RT_VISIBLE_DEVICES=6 python /work/wy_harness/run_fast.py \
   /work/wy_harness/shapes_light.txt /work/wy_harness/out.json'
```
- `shapes_light.txt` = 9-shape battery (gate: min_cos ≥ 0.999 every shape);
  `prod_shape.txt` = canonical only (fast iteration).
- Harness patched so NaN cos → -1.0 (Python `min()` silently swallows NaN — this
  masked broken W/U for a whole day; never trust an unpatched cosine gate).
- Per-output cos: `python /work/wy_harness/pcos2.py` (edit target json inside).
- Long builds: run as a detached lockfile chain (`*_chain.sh` patterns in
  `/work/wy_harness/`) — single chain only, absolute cd paths, log to
  `/work/wy_harness/chainlog`, always echo build/install rc and VERIFY =0 before
  trusting battery numbers (failed installs silently rerun the stale binary).

## Running inference / serving benchmark

Inside the container (after a FULL pip rebuild if any targeted install happened):
```bash
bash /work/repro/run_bench.sh -m /home/models/Qwen3.5-9B-w8a8-modelslim -d 6 -q -b "1"
# -q = --quantization ascend (msModelSlim W8A8); -b batch list; -i/-o token counts
# results: /work/bench_<ts>/; markdown table: python /work/repro/summarize.py <dir>
```
Reference numbers: fp16 TTFT ~3.2 s; W8A8 + old kernel TTFT 1467 ms; W8A8 + new
~800 µs kernel TTFT 1323.5 ms (2026-09-01, bench_20260901_115635-era run); decode
61.9 ms/tok without MTP, 37.9 with MTP k=3.
Full-rebuild trap: `pip install -e .` with cached build finishes in ~2 min and
does NOT regenerate the vendor — wipe `build csrc/build csrc/build_out
vllm_ascend/_cann_ops_custom/vendors` first, then it takes ~15 min and the tree
vendor `binary_info_config.json` must list all 5 ops (incl CausalConv1dV310),
not just ChunkGatedDeltaRuleComputeWy — single-op vendor kills engine init.

Manual server (what run_bench.sh does):
```bash
ASCEND_RT_VISIBLE_DEVICES=6 vllm serve /home/models/Qwen3.5-9B-w8a8-modelslim \
  --quantization ascend --max-model-len 4096 --gpu-memory-utilization 0.7 \
  --port 8100
```

## Key source files

- Kernel: `csrc/moe/chunk_gated_delta_rule_compute_wy/op_kernel/arch20/`
  - `compute_wy_kernel.h` — task flow (loads → βK → g-scan → gram → Λ → gate →
    NZ-resident doubling BuildT → W/U solves → stores)
  - `compute_wy_micro_mm.h` — hand-rolled Mmad path (the 5.8x enabler; the AscendC
    matmul lib costs ~1.8 µs/call vs ~65 ns of math at 64-tile shapes)
  - `compute_wy_cube.h` — legacy lib wrappers (still used: K<128 gram + lib Init,
    which is load-bearing even when unused — skipping it hangs the aicore)
  - `compute_wy_lambda_table.h`, `compute_wy_identity_nz.h` — compile-time GM tables
- Host tiling: `op_host/chunk_gated_delta_rule_compute_wy_tiling.cpp` —
  LOCAL_WORKSPACE_BYTES must stay 32 KB (8 KB overruns at head dim 128 → silent NaN).
- Python: `vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py` (torch reference
  `_compute_kernel_inputs_from_torch_wy` = the harness golden).
- Full trap ledger + micro-mm recipe: auto-memory files
  `310p-{op-build-cache-traps, silent-correctness-traps, micro-mmad-recipe}.md`
  and `mm-handroll-m200-recipe.md` here in talk/.

## Directory map (talk/)

- `HANDBOOK.md` — this file
- `qwen9b_optim_20260819_status.md` — chronological optimization log
- `mm-handroll-m200-recipe.md` — raw-Mmad recipe extracted from toolkit sources
- `repro/` — container/bench/profile scripts (run_bench.sh, make_container.sh, …)
- `mtp_310p_findings.md`, `qwen35_9b_*.md`, `pr11941_perf_profile.md` — decode-side
  and baseline studies
- `archive/` — stale May-Jul era files
