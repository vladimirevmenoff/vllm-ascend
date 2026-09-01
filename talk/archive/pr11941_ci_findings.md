# PR 11941 (chunk_gated_delta_rule_compute_wy, 310P) — CI failure investigation

## ROOT CAUSE #1 — CONFIRMED AND FIXED: tiling struct packing mismatch

The kernel-side `ChunkGatedDeltaRuleComputeWyTilingData` does not match the layout the
CANN host tiling framework serializes. On real 310P the op dies with

```
retCode=0x26 [aicore exception], error code = 0x800000
errorStr: The DDR address of the MTE instruction is out of range.
```

**Mechanism.** `register/tilingdata_base.h::FieldHandler` is *strictly packed* —
`ret_val = data_size_; data_size_ += typeSize;` — it never inserts alignment padding.
The hand-written kernel struct uses `#pragma pack(push, 8)`, so the `uint64_t
workspaceOffset` that follows three `uint32_t`s gets padded:

| field | host (packed) | kernel `pack(8)` |
|---|---|---|
| localWorkspaceSize | 80 | 80 ok |
| perCoreWorkspaceBytes | 84 | 84 ok |
| usedCoreNum | 88 | 88 ok |
| **workspaceOffset** | **92** | **96** MISMATCH |
| **mmAttn (TCubeTiling)** | **100** | **104** MISMATCH |
| total | 900 | 904 |

(measured on device: `sizeof(TCubeTiling)=200, alignof=4`)

Everything from offset 92 on is read 4 bytes off. The kernel's `workspaceOffset` becomes
`(low 4B = high half of the real value = 0) | (high 4B = first word of mmAttn) << 32`
— i.e. a multi-GB bogus offset — so `aGm_/bGm_/cGm_` point outside the workspace, and all
four `TCubeTiling`s are shifted garbage. Consistent with the observed symptom that
**only `g_kernel` was correct** (it uses only fields below offset 92) while
q/k/w/u_kernel were never written.

**Fix (one line)** in
`op_kernel/chunk_gated_delta_rule_compute_wy_tiling_data.h`:
`#pragma pack(push, 8)` -> `#pragma pack(push, 1)`.
Verified on device: hard aicore exception -> `SIM_RESULT PASS`, cos 1.000000 on all
five outputs.

**Why every simulator test passed.** The CAModel does not bounds-check DDR; the bogus
staging address behaves as ordinary scratch (written then read back), so the op appeared
bit-exact. Same mismatch exists under CANN 9.0.1 and 9.1.0-beta.1 — it is not
version-specific, but whether it faults, silently corrupts, or appears to work depends on
the allocator/memory layout. That is what makes the e2e symptom environment-dependent.

**BUILD TRAP:** editing `tiling_data.h` does NOT trigger a kernel rebuild — the op-compile
step doesn't track it as a dependency and silently reuses the cached `.o`. Must
`rm -rf build/binary build/kernel_meta` first. I lost a full build cycle to this; always
check the `.o` mtime before trusting a result.

## ROOT CAUSE #2 — cube base dim of 128 is mis-driven

**Diagnosis.** The host tiling lets a matmul *base* dimension reach 128, and this kernel's
use of `MatmulImpl` mishandles that on 310P. Two separate symptoms, one cause:

| tiling | symptom |
|---|---|
| `baseN = 128` (apply, N = head dim 128) | runs, **wrong results** |
| `baseK = 128` (gram, inner K = head dim 128) | **aicore timeout (hang)** |

Sharp threshold, measured on device with `DOUBLING_ROUNDS=1` (so round 0 alone is
compared against a 1-round golden — the corruption is already present in round 0):

| V | u_kernel cos |
|---|---|
| 64 / 80 / 96 / 112 | 1.000000 |
| **128** | **0.894222** |

Ruled out along the way — all three of these produced **bit-identical** failures, so the
bug is compute, not memory: local workspace 32 KB vs 4 KB; cube GM staging arena doubled
(`WY_CUBE_MAX_HEAD` 128 -> 256); `mssanitizer memcheck` clean. Also not an L0/UB capacity
limit: 310P3 has `ub_size=262144`, `l0_a/b=65536`, `l0_c=262144`, and the kernel needs
~230 KB UB and 32 KB L0C at head dim 128.

**Fix** (host tiling only, `chunk_gated_delta_rule_compute_wy_tiling.cpp::FillCubeTiling`):
keep `SetOrgShape` at the real sizes but cap the *tiled block* at the 64-wide chunk, so no
cube base dim exceeds 64:
```cpp
mm.SetShape(std::min<int64_t>(m, FIXED_CHUNK),
            std::min<int64_t>(n, FIXED_CHUNK),
            std::min<int64_t>(k, FIXED_CHUNK));
mm.SetFixSplit(FIXED_CHUNK, FIXED_CHUNK, FIXED_CHUNK);
```
The kernel already re-sets `SetOrgShape`/`SetSingleShape` at runtime, so the matmul just
iterates internally (`stepN=2` for N=128, 2 K-iterations for K=128).

Important detail: **`SetFixSplit` alone is not enough.** It forces `baseN` 128->64
(`stepN=2`) but leaves `baseK=128` — which is exactly why it fixed the V=128 wrong-results
case and left the K=128 hang. `SetShape` capped at 64 is what drives `baseK` down.

Measured tiling with both applied: `mmApply N=128 -> baseN=64 stepN=2`,
`mmAttn orgK=128 shapeK=64 -> baseK=64`.

## ROOT CAUSE #2 — original observations: hang at head dim 128

With the packing fix applied, head dim 64 passes but **K=V=128 hangs**:
`retCode=0x25 [aicore timeout]`. Bisect (device):

| shape | result |
|---|---|
| 1 64 1 1 64 64 | PASS |
| 1 64 2 4 64 64 | PASS |
| 1 512 1 1 64 64 (8 tasks) | PASS |
| 1 1024 1 1 64 64 (2 tasks/core) | PASS |
| 1 128 1 1 **128 128** | **HANG** |
| 1 256 2 8 **128 128** | **HANG** |

So it is **head dim, not task count**. Not a plain UB overflow: 310P3 `ub_size=262144`
and the kernel allocates ~230144 B at K=V=128. Prime suspect is the hardcoded
`LOCAL_WORKSPACE_BYTES = 32*1024` matmul local workspace (should come from the
matmul tiling, not be a constant) — only ~32 KB of UB headroom remains, so it cannot
simply be enlarged without shrinking other buffers. **Head dim 128 is the production
Qwen3.5 GDN configuration**, so this alone breaks the e2e path.

### #2b — the gram matmul: inner K > 64 HANGS (this is the real K-side bug)

The "128" framing was wrong on the K side. Measured: **K=80 and K=96 hang too**; only
K=64 works. V (the apply N dim) is fine at 80/96/112 and, after the tiling cap, at 128.
So the K asymmetry is the *gram* matmul's inner-K dimension — i.e. **`GemmATransB` hangs
whenever it needs more than one K iteration**. Capping `baseK` to 64 via the tiling did
NOT help (still hung), so it is the multi-K-iteration path itself that is broken here,
not the base size.

**Fix** (`compute_wy_cube.h::GemmATransB`): split K into <=64 slices in the kernel and
accumulate in UB, so every matmul call is a single K iteration:
```cpp
for (uint32_t k0 = 0; k0 < kDim; k0 += WY_CUBE_CHUNK) {
  const uint32_t kCur = min(WY_CUBE_CHUNK, kDim - k0);
  mmAttn_.SetOrgShape(64, 64, kDim);      // orgK = kDim keeps A/B strides right
  mmAttn_.SetSingleShape(64, 64, kCur);
  mmAttn_.SetTensorA(aGm_[k0], false);
  mmAttn_.SetTensorB(bGm_[k0], true);
  mmAttn_.IterateAll(cGm_);
  // k0 == 0 -> copy C into cUb; else copy into accScratch and Add into cUb
}
```
Caller passes `scratch` (tmpBuf_) as the accumulator. No atomics needed.

## FINAL STATE — all verified on real 310P hardware

Three fixes, all necessary, none of them the thing I originally suspected:

1. `op_kernel/..._tiling_data.h`: `#pragma pack(push, 8)` -> `(push, 1)`
2. `op_host/..._tiling.cpp`: cap `SetShape` at `FIXED_CHUNK` + `SetFixSplit(64,64,64)`
3. `op_kernel/arch20/compute_wy_cube.h`: split `GemmATransB` over K, accumulate in UB
   (+ caller in `compute_wy_kernel.h` passes the scratch tensor)

Device verification (device 3, CANN 9.1.0-beta.1), every output cos = 1.000000:

| B T Hk Hv K V | g | result |
|---|---|---|
| 1 64 1 1 64 64 | 0.5 | PASS |
| 1 64 2 4 64 64 | 2.0 | PASS |
| 1 64 1 1 80 96 | 0.5 | PASS |
| 1 64 1 1 128 64 | 0.5 | PASS (was hang) |
| 1 64 1 1 64 128 | 0.5 | PASS (was cos 0.076) |
| 1 64 1 1 128 128 | 0.5 | PASS (was hang) |
| 1 256 2 8 128 128 | 0.5 | PASS |
| 1 1024 8 16 128 128 | 0.5 | PASS |
| 2 512 4 16 128 128 | 1.0 | PASS |
| **1 1536 8 16 128 128** (production Qwen3.5) | 0.01 | **PASS** |

Determinism stress, production shape, 60 iterations:
`STRESS_NONDETERMINISTIC_ITERS 0/59`, `STRESS_WORST_COS_VS_GOLDEN 1.000000`,
`STRESS_NONFINITE 0`.

## The sync hypothesis is DEAD

The verified build contains **no** `S_V`/`V_S`/`S_MTE3` additions, and it is bitwise
deterministic over 60 runs at the production shape and bit-exact against the fp32 golden
everywhere. The missing-V_S theory I pursued for a long time was wrong. Kept here only as
a record of a dead end: `PipeBarrier<PIPE_V>` really does not order V->S, but it did not
matter for this kernel. Do not re-add those events without evidence.

`gdn_310.py`'s `torch.where` change is retained as NaN-safety hardening — unrelated to
either root cause, no downside, but not required to fix CI.

## CI triage (run 30333756051)

Only ONE real failure. CI's own `analyze_failure_report.py`: 1 failed, 0 unexplained.

| job | verdict |
|---|---|
| `test_qwen3_5_dense_prefix_mamba_cache_tp1_fp16` (310p-1card, vLLM **v0.25.1**) | THE failure |
| a3-4 card (`dcp_full_feature_accuracy`, `qwen3_5_35b_mtp3_flashcomm`) | 910B/A3 4-card, unrelated to a 310P-only op |
| e2e-upstream_pr | pip `uc-manager` OSError — network infra |
| ci-gate, upload-coverage-to-obs | downstream aggregators of the above |

## The decisive CI fact

Same PR commit, same 3 tests in `test_dense_model_310p.py`:
- vLLM leg `fe784ff…` → **3 passed**
- vLLM leg `v0.25.1` → **1 failed, 2 passed**

=> failure is **nondeterministic / environment-dependent**, not a hard regression.

Test asserts ONLY `no_prefix_cache == prefix_cache`. Garbage-but-equal passes.
Both failing outputs were degenerate (`ilkilkilk…` vs `圆圆圆圆…`) but that proves
nothing in either direction — there is no coherent-output baseline in this test.

## Is the test a pre-existing flake? NO — 27/27 elsewhere

Scanned 150 recent E2E runs (workflow 280054652) for completed `310p-1 card` jobs:
28 found. 27 success + 1 failure, and that failure was infra
(`cpToPod failed after 30 attempts`), not the test. All 27 successful jobs ran
`test_qwen3_5_dense_prefix_mamba_cache_tp1_fp16` and passed it — including 2 jobs where
it was the explicitly-selected node, passing on BOTH vLLM legs (v0.25.1 and fe784ff).

=> the test is stable on other branches. **PR 11941 is implicated.**
Combined with "same PR code passes on fe784ff, fails on v0.25.1", the defect is
nondeterministic AND introduced by this PR.

## Simulator setup (WORKING — reusable)

Container `wy310p`, image `ascendai/cann:9.0.1-310p-ubuntu22.04-py3.11`,
host worktree of the PR mounted at `/repo`. No NPU device; CAModel sim only.

```
pip install packaging regex numpy
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /repo/csrc && bash build.sh --pkg --ops="chunk_gated_delta_rule_compute_wy" --soc=ascend310p -j8
cd build && ./cann-ops-transformer-custom_linux-aarch64.run --quiet --install-for-all
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/custom_transformer/op_api/lib/:\
/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend310P3/lib:$LD_LIBRARY_PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /repo/wysim && msprof op simulator --soc-version=Ascend310P3 --output=/tmp/sim_out \
    --timeout=1500 python3 test_compute_wy_sim.py <B> <T> <Hk> <Hv> <K> <V> [g_scale]
```

Build gotchas hit: missing `packaging`, missing `regex`. Nothing else.

Harness `wysim/test_compute_wy_sim.py` (ctypes → `libcust_opapi.so`, numpy golden
mirroring `_compute_kernel_inputs_from_torch_wy`, outputs poisoned with 7.0 so
"never written" is distinguishable from "wrote zeros").

## Simulator results — op is CORRECT everywhere tested

| B T Hk Hv K V | g_scale | tasks/core | worst cos | verdict |
|---|---|---|---|---|
| 1 64 1 1 64 64 | 0.01 | 1 | 1.000000 | PASS |
| 1 64 2 4 64 64 | 0.01 | 1 | 1.000000 | PASS |
| 1 128 1 1 128 128 | 0.01 | 1 | 1.000000 | PASS |
| 1 64 1 1 64 64 | 0.5 | 1 | 1.000000 | PASS |
| 1 64 1 1 64 64 | 2.0 | 1 | 1.000000 | PASS |
| 1 1024 1 1 128 128 | 0.5 | 2 | 1.000000 | PASS |
| 1 256 2 8 128 128 | 0.5 | 4 | 1.000000 | PASS |

All outputs written (no `untouched`), zero non-finite, max abs err ≤ 2.4e-4 (fp16 ulp).

### Faster sim invocation (no msprof, ~10x quicker)

msprof only sets env. To run the CAModel directly:
```
export LD_LIBRARY_PATH=$ASCEND/tools/simulator/Ascend310P3/lib:<vendor>/op_api/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=libruntime_camodel.so
export IS_SIMULATOR_ENV=true CAMODEL_SOC_VERSION=Ascend310P3
export CAMODEL_LOG_PATH=/tmp/camodel CAMODEL_CONFIG_PATH=/tmp/camodel/config
python3 test_compute_wy_sim.py ...
```

### mssanitizer does NOT work under the CAModel — BLOCKED

`mssanitizer --tool=synccheck|racecheck` would directly test the V_S/S_V claim, but it
*overwrites* `LD_PRELOAD` with its own injection libs
(`libmssanitizer_injection.so:libascend_san_stub.so`), dropping `libruntime_camodel.so`,
so `aclrtSetDevice` fails. Re-appending the camodel runtime via an exec wrapper still
fails at `aclrtSetDevice` — the sanitizer stub and the CAModel runtime conflict.
=> sanitizers need a real device. `build.sh` exposes no kernel CPU-debug/ICPU target
(only host-side `--ophost_test`/`--opapi_test`), so the `tikicpulib npuchk` route would
need a bespoke harness.

## Ruled out

1. **fp16 precision of the nilpotent doubling.** `GemmSquare`/`UploadP` cast P to fp16
   each of the 6 rounds. Simulated on host (`numpy`, 20 trials/config): with l2-normed k
   — which the production path guarantees, `_maybe_l2norm` runs before the WY call —
   |A|max ≈ 0.25, powers DECAY, cos = 1.000000, no overflow. Only the no-l2norm case
   explodes (|P|→inf), and that path is unreachable.
2. **Strong-gate overflow in `ApplyLambdaNegStrictLower`.** `Exp` is evaluated on the full
   row incl. upper triangle where `a_i − a_j > 0` can overflow to inf, but
   `Duplicate(rowTmp[i], 0, 64−i)` overwrites exactly those lanes before the `Mul`.
   Confirmed clean at g_scale 0.5 and 2.0.
3. **Math of the doubling recurrence.** 6 rounds of `r += P r; P = P²` yields
   `(I + A + … + A^63) R`, exact for a 64×64 strictly-lower nilpotent A. Matches the
   forward-substitution reference bit-for-bit in the sim.
4. **Precision amplification of the prefix-cache gap.** Hypothesis: the op's extra fp16
   rounding pushes the (inherently nonzero) prefix-ON vs prefix-OFF divergence over the
   greedy token-flip threshold. Tested on host (`scratchpad/prefix_div.py`): full chunked
   GDN in numpy, run A = whole sequence, run B = prefix + fp16 state carry + suffix
   (mirrors `mamba_ssm_cache_dtype="float16"`), with fp32-WY vs fp16-doubling-WY.
   rel_L2(A,B) ratio PR/main over 6 seeds × 3 gate scales: **min 0.98, med 1.00, max 1.04**.
   The A-vs-B gap is dominated entirely by the fp16 state carry; the WY choice does not
   move it. Hypothesis DEAD.

## Still open / untested

- **>1 task per core.** Every sim run so far had ≤1 task/core, so the per-core GM staging
  buffers (`aGm_`/`bGm_`/`cGm_`) were never reused across loop iterations. Running
  T=1024 (16 tasks / 8 cores) now. This is the highest-value remaining sim test.
- **Production shape** B=1, T≈1536, Hv=16–32, K=V=128 — expensive in sim (~384 tasks).
- **The CAModel is a single deterministic schedule.** It cannot reproduce a hardware race;
  a PASS here does not clear the sync concerns below.

## Cross-core races: ruled out by inspection

Task index decodes uniquely to (b, vHead, chunk); u/w writes are disjoint per task;
q_kernel/k_kernel are written by exactly one vHead per (b, kHead, chunk)
(`vHeadIdx % headGroups_ == 0`); the GM staging arena is indexed by
`GetBlockIdx() % usedCoreNum_` with `blockIdx >= blockNum` guarded in `Process()`.
No two cores touch the same bytes. The only remaining concurrency is INTRA-core pipe sync.

## Code concerns (static, NOT reproduced)

1. **Missing V→S sync.** `expGLocal` is written by vector `Exp()` in `BuildCumulativeG`,
   then read by scalar `GetValue()` in `BuildKBetaExpFloat` with only
   `PipeBarrier<PIPE_V>` between. `PipeBarrier<PIPE_V>` orders V→V, not V→S.
   `compute_wy_kernel.h` uses ZERO `HardEvent::V_S`/`S_V` in the whole file, while the
   sibling 310P GDN kernel `csrc/attention/recurrent_gated_delta_rule_v310/op_kernel/…h`
   uses `SetWaitFlag<HardEvent::V_S>` for exactly this pattern (3 sites).
   Nondeterministic by nature → consistent with the leg-dependent pass/fail.
   `ops-transformer/talk/pipe_barrier_pipe_v_findings.md` confirms: cross-pipe deps need
   SetFlag/WaitFlag; PipeBarrier only drains its own pipe.
2. **`_clear_states_without_initial` (gdn_310.py) is NaN-unsafe.** `states * 0` preserves
   NaN/Inf where the replaced `states[~mask] = 0` erased it. Latent, not root cause —
   vLLM allocates the kv/ssm cache with `torch.zeros` (model_runner_v1.py:3553). Fix:
   `torch.where(keep_bool, states, zeros)`, which still avoids the aicpu `IndexPut` the
   author was working around.
3. **`Duplicate(rowTmp[i], …)` starts at a non-32B-aligned element offset** (i not a
   multiple of 8 floats). Predicts DETERMINISTIC corruption, so inconsistent with the
   observed flake, and the sim shows it working. Note only.
4. **The op has zero PR-gated CI coverage.** Its test lives in `tests/e2e/nightly/310p/…`
   and never ran — CI listed it under "Recommended but Not Failed", a bucket that
   includes "not executed". Recommend promoting at least
   `test_compute_wy_qwen35_production_shape_is_deterministic` into the PR-gated suite.

## Real 310P device access (2026-07-29)

`ssh 310p` (123.60.231.33:10002). 8× 310P3, device 0 busy with someone's vLLM engine.
Shared box — do NOT touch `e00927329_310p_latest` or other users' containers.

My container (created fresh, same image as the reference one):
```
docker run -itd --name claude_wy_310p \
  --device=/dev/davinci3 --device=/dev/davinci5 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /home:/home quay.io/ascend/vllm-ascend:nightly-main-310p-openeuler /bin/bash
```
CANN 9.1.0-beta.1, mssanitizer + msprof present, numpy 1.26.4, network works.

Gotchas:
- **`nproc` reports 1** because the image sets `OMP_NUM_THREADS=1`. 128 real CPUs —
  pass `-j16` explicitly or the build is single-threaded.
- **rsync mac->device is ~50 KB/s.** Don't rsync the tree. `git fetch --depth 1 origin
  pull/11941/head` inside the container takes seconds (csrc is 18M without submodules).
  Only scp the small test harnesses.

Work dir: `/home/claude_wy/src` (PR at 68bd26e), harnesses in `/home/claude_wy/wysim`,
unpatched kernel saved at `/home/claude_wy/compute_wy_kernel.ORIG.h` for the A/B.

Planned controlled experiment (A = unpatched PR, B = patched):
1. `stress_device.py` — same input N times, check bitwise run-to-run equality + golden.
   A race shows up as run-to-run divergence; the CAModel could never show this.
2. `mssanitizer --tool=synccheck` / `--tool=racecheck` — names the exact hazard.

## Device results after the packing fix (kernel otherwise UNMODIFIED)

| B T Hk Hv K V | result |
|---|---|---|
| 1 64 1 1 64 64 | PASS cos 1.000000 (was: aicore exception) |
| 1 64 2 4 64 64 g=2.0 | PASS |
| 1 512 1 1 64 64 | PASS |
| 1 1024 1 1 64 64 | PASS |
| 1 64 1 1 64 96 | PASS |
| 1 64 1 1 64 **128** | FAIL — u_kernel cos 0.076 maxabs 8.1e2, w_kernel cos 0.916 |
| 1 64 1 1 **128** 64 | HANG (aicore timeout) |
| 1 128 / 256 / 1024 … **128 128** | HANG |

Note this build does NOT contain the S_V/V_S sync patch, and every head-dim-64 shape is
bit-exact — so those syncs are not required for correctness at the shapes that work.
`mssanitizer --tool=memcheck` on the V=128 case reports **no GM error**, so the V=128
corruption is a UB-aliasing/compute bug, not a GM overrun.

## Sync patch — UNPROVEN, probably NOT the bug

Worktree `/Users/evmenoff/Documents/code/wy-sim-pr11941` (branch pr-11941). Builds clean;
sim still 1.000000 on 3 reruns after the change (proves no regression, proves nothing else).

`compute_wy_kernel.h`:
- `BuildCumulativeG`: `PipeBarrier<PIPE_V>` -> `SyncEvent<HardEvent::S_V>` before
  `Exp(expGLocal, gLocal, …)`; added `SyncEvent<HardEvent::S_MTE3>` before the
  `DataCopy(gKernelGm_[…], gLocal, …)`.
- `BuildKBetaExpFloat`: added `SyncEvent<HardEvent::V_S>` before the loop that reads
  `expGLocal.GetValue(row)`.

`gdn_310.py`: `_clear_states_without_initial` now uses `torch.where(keep, states, 0)`
instead of `states * keep` (NaN/Inf-safe, still no aicpu `IndexPut`).

**NO FAILING TEST WAS EVER REPRODUCED.** The patch rests on inference (test stable
elsewhere + leg-dependent failure + cross-core races excluded + repo convention), not on
a demonstration. It may be wrong. Only a device run of
`mssanitizer --tool=synccheck`, or the v0.25.1 310p leg going green repeatedly,
would confirm it.

## Constraints

No NPU device anywhere in reach — sim only. Cannot push to the PR: it is cross-repo from
`semenishinDmitry/vllm-ascend` (`push:false`); `maintainerCanModify:true` does not help us.
