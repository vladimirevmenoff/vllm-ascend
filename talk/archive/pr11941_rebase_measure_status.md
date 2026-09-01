# PR 11941 — rebase + W8A8 BS sweep (2026-08-12)

Dmitry's PR 11941 (`chunk_gated_delta_rule_compute_wy` for 310P), rebased and measured.
**Result: 1.58× faster prefill on W8A8, decode unchanged.**

## Headline — W8A8, 2048 in / 1024 out, one 310P3, TP=1

| BS | TTFT main (ms) | TTFT PR (ms) | **TTFT ↑** | TPOT main | TPOT PR | TPOT ↑ | output tput main | output tput PR |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 000.1 | **1 266.1** | **1.58×** | 62.42 | 62.57 | 1.00× | 15.55 | 15.69 |
| 2 | 3 193.9 | **2 068.7** | **1.54×** | 65.63 | 65.47 | 1.00× | 29.10 | 29.64 |
| 4 | 6 870.2 | **4 292.4** | **1.60×** | 70.30 | 69.63 | 1.01× | 51.95 | 54.20 |
| 8 | 11 021.1 | **6 946.1** | **1.59×** | 82.87 | 80.05 | 1.04× | 85.40 | 92.08 |

Prefill speed at BS=1: **1 023.9 → 1 617.6 tok/s (1.58×)**. Decode 15.98 tok/s, unchanged.

fp16 as a cross-check (BS=1 only): TTFT 2 566.8 → **1 816.0 ms (1.41×)**, prefill
797.9 → 1 127.8 tok/s, decode 9.87 → 9.86 tok/s.

## Why these numbers are credible

- **Decode is untouched, and measures untouched.** The PR only replaces the WY prefix in
  prefill. TPOT moves ≤1% at BS 1–4, and fp16 decode lands at 9.86 vs main's 9.87 tok/s.
  That doubles as a control: source-built vs image-built isn't skewing results.
- **The magnitude matches the profile's prediction.** The WY cluster measured 35–41% of
  prefill, and the op-level A/B in `pr11941_perf_profile.md` was 4.29× on that step.
  1 − 0.38 × (1 − 1/4.29) ≈ 0.71 → **1.41× predicted** for fp16. Measured: 1.41×.
- **W8A8 gains more than fp16 (1.58× vs 1.41×), as it should.** Quantization speeds up the
  GEMMs but not the WY loop, so WY is a *larger* share of W8A8 prefill — removing it helps
  more. The two numbers being different in this direction is a consistency check, not noise.
- Op eligibility verified by reading the guard, not assumed: `_can_use_npu_compute_wy` needs
  fp16 q/k/v/beta, fp32 g, chunk 64, K,V ≤128 and %16, B ≤32, Hv ≤64 — Qwen3.5-9B at
  `--dtype float16` gives K=V=128, Hv=32, B≤8. `enable_custom_op()` is on for 310P.

## What this is worth in practice

At 2048 in / 1024 out the E2E barely moves (65.3 s vs 65.9 s at BS=1) — 1024 decode steps at
62 ms dwarf a 1.3 s prefill. **The win is TTFT**, so it matters for TTFT-sensitive serving,
long prompts, and short outputs. At BS=8 it also shows up as throughput: 92.1 vs 85.4 tok/s
(1.08×), because prefill occupies less of the batch's time.

## Rebase

PR 11941 = 26 commits on base `b2f683ca3`. Rebased onto `upstream/main @ 1aad8745e`
(185 commits ahead): **25 commits, zero conflicts**. Branch `pr11941-rebase`; patch series at
`/home/claude_bench/patches11941/`.

**But the rebased tree cannot run on the 310P nightly image:**
```
AttributeError: module 'vllm.model_executor.layers.fused_moe.layer'
                has no attribute 'FusedMoEFactory'
```
vllm-ascend main has outrun the vLLM the image ships (`752a3a5`, 0.25.1); the two version in
lockstep. Measuring the rebased tree needs a newer paired vLLM — a newer nightly image, or
vLLM from source.

So the measurement above was taken with the PR at **its own base `b2f683ca3`**, which is
exactly the base of the existing baseline numbers. That makes the A/B cleaner than the
rebased version would have been.

## Build recipe

Clean container from the same image (`claude_pr11941`), then:
```bash
cd /vllm-workspace/vllm-ascend
export SOC_VERSION=ascend310p1 MAX_JOBS=16
pip install -e . --no-build-isolation --no-deps -v      # ~22 min
```
- `--no-deps` is **required**: resolution otherwise demands `triton-ascend==3.2.2`, which has
  no aarch64 wheel and which 310P doesn't support anyway.
- Ops load lazily — `import vllm_ascend` is not enough:
  ```python
  import vllm_ascend.vllm_ascend_C
  torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy   # present
  ```
- The rebuilt vendor op package is a superset of the image's: 5 ops vs 4 (adds
  `chunk_gated_delta_rule_compute_wy`). Nothing is dropped.

## Detour worth recording

Three earlier attempts to serve W8A8 on this build failed — two sweeps died ~4.5 min in at
`Loading safetensors shards: 0%`, a direct load hung past a 600 s timeout. **All of it was
the box**, which rebooted twice that evening and was carrying other users' load. The same
build served W8A8 first try once the box was freshly rebooted and idle. The fp16 probe is
what separated "build is broken" from "environment is broken" — worth reaching for early next
time.

## Caveats

- Baseline is the Aug 7 W8A8 sweep (image-built `b2f683ca3`); PR is source-built at the same
  commit. Build method differs; the unchanged decode numbers argue it doesn't matter.
- The fp16 probe overlapped in time with the W8A8 server loading on a different card. Separate
  cards have separate memory, and BS=1 decode is weight-bandwidth-bound per chip, so the
  effect should be nil — but it is not a solo measurement.
- One run per point. Within-run spread is tiny (BS=1 TTFT P99/P50 = 1.003).
- Not measured: the rebased-onto-main version, for the vLLM-pairing reason above.

## Also: `claude_bench_main` is contaminated

At 05:03 on 2026-08-11, user `s60117411` pip-installed **this same PR** (`ec0d40ecd`) as an
editable package into `claude_bench_main`, pointing at `/home/s60117411/vllm_ascend2/vllm-ascend`
— a tree they are actively editing. That container no longer imports the image's stock
`b2f683ca3`.

- The Aug 7 baselines predate this by four days and are unaffected.
- `talk/repro/topk_topp_guide.md` has been fixed to check for an editable install and build a
  fresh container.
- Someone else is working this PR right now — coordinate before duplicating effort.

## Artefacts

`ssh 310p`, container `claude_pr11941` (tree at `ec0d40ecd`, built):
`/home/claude_bench/bench_pr_w8a8/`, `bench_pr_fp16/`, `pr11941_build{,2}.log`,
`patches11941/`. Summarize with `python3 /home/claude_bench/repro/summarize.py <dir>`.
