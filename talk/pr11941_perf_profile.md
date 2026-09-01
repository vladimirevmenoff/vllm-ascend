# chunk_gated_delta_rule_compute_wy — 310P device perf profile (2026-07-30)

NOT a B8 deliverable. No `migration_config.yaml`, no `block_decomposition.yaml` /
`tiling_chosen.yaml`, so no cost-model prediction and no predicted-vs-measured compare.
This is a plain `msprof op` measurement of the built + correctness-verified kernel.

Setup: device 3 of `ssh 310p`, container `claude_wy_run`, CANN 9.1.0-beta.1,
op build = the 3-fix commit (`85906880`). Correctness re-confirmed in the profiled run
(`SIM_COSINE 1.000000`).

## Measured (msprof op, device, blockDim=8)

| shape B×T×Hk×Hv×K×V | task dur | aic cycles | scalar | mte3 | mte2 | vec | cube | mte1 |
|---|---|---|---|---|---|---|---|---|
| 1×256×8×16×128×128 | 1043.4 us | 8,821,182 | **0.694** | 0.371 | 0.254 | 0.146 | 0.018 | 0.031 |
| 1×1024×8×16×128×128 | 3846.5 us | 33,186,086 | **0.730** | 0.385 | 0.228 | 0.155 | 0.019 | 0.033 |
| 1×1536×8×16×128×128 | 5766.9 us | 49,888,373 | **0.728** | 0.386 | 0.232 | 0.154 | 0.019 | 0.033 |

Duration scales linearly with T (~3.75 us/token at Hv=16, K=V=128).

Production shape (1×1536×…) extra counters:

| counter | value |
|---|---|
| L2 cache hit rate | 97.99 % |
| MTE3 instructions | **774,912** |
| MTE2 instructions | 86,032 |
| MTE1 instructions | 46,848 |
| cube instructions | 11,904 |
| main mem read / write BW | 446.8 / 277.1 GB/s |
| UB read/write BW (vector) | 12.26 / 10.89 GB/s |
| wait ratios | vec 0.306, mte2 0.239, mte3 0.239, mte1 0.165, cube 0.143 |
| vec bank-group conflict | 0.013 (negligible) |

## Verdict: SCALAR-bound, cube ~idle

**~73 % scalar, ~1.9 % cube.** The op spends almost none of its time doing the matmuls it
exists to do. L2 hit 98 % — this is *not* a DRAM problem. Two structural causes:

### 1. Every matmul round-trips through GM

`compute_wy_cube.h:17-21` declares A, B and C as `TPosition::GM`:
```cpp
using WyMmAType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using WyMmBType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using WyMmCType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
```
So each matmul stages A/B from UB into GM and reads C back from GM. The doubling loop
(`compute_wy_kernel.h:344-352`) runs 6 rounds × (UploadP + 2 GemmApplyAdd + GemmSquare)
per task, and there are B×Hv×numChunks = 1×16×24 = **384 tasks**. Result: 774,912 MTE3
instructions ≈ **2,018 MTE3 instructions per task**, against only 11,904 cube instructions
in total.

### 2. Per-row scalar loops

Every one of these is a 64-iteration loop issuing one tiny vector op per row, each with
scalar address computation and (in three of them) a scalar `GetValue`:

- `compute_wy_kernel.h:251-267` `ApplyLambdaNegStrictLower` — 64 iterations × {GetValue,
  Duplicate, Sub, Exp, Duplicate, Mul, Muls} + 6 `PipeBarrier<PIPE_V>` per row
- `compute_wy_kernel.h:272-276` `BroadcastMulRowsFloat` — 64 × {GetValue, Muls}
- `compute_wy_kernel.h:280-286` `BuildKBetaExpFloat` — 64 × {GetValue, Muls, GetValue, Muls}
- `compute_wy_kernel.h:235-238` `BuildCumulativeG` — 64-iteration pure-scalar cumsum
  (GetValue + SetValue per element)

11 `GetValue`/`SetValue` sites total in the kernel.

## Optimisation candidates (unranked, unvalidated)

Highest leverage first by measured share:

1. **Keep the cube operands off GM.** Move A/B to `TPosition::VECCALC`/L1 and C to L0C→UB
   instead of the GM ND round-trip. This is the 0.386 MTE3 + much of the scalar issue
   cost. Biggest single win available, and the biggest change.
2. **Vectorise `ApplyLambdaNegStrictLower`.** The Λ = exp(a_i − a_j) strict-lower mask can
   be built as a 64×64 block with broadcast ops instead of row-at-a-time, removing ~64×7
   vector issues + 64 GetValue per task.
3. **Replace the scalar cumsum** at `:235` with a vector prefix-sum (or `Cumsum` if the
   310P intrinsic set has it — check
   `shared/hardware/intrinsic-support-matrix/Ascend310P3.json`, do not assume).
4. **Fold the two per-row `Muls` loops** (`:272`, `:280`) into whole-block ops by
   materialising the beta / exp(a) column vector once and using a broadcast multiply.

Note the doubling algorithm itself is cheap — 6 rounds is only ~11.9 K cube instructions.
The cost is entirely in how the data is shuttled, not in the math.

## Caveats

- Single-run measurement per shape; no 3-run determinism sweep here (the earlier 60-run
  bitwise-determinism check at the production shape passed, so timing variance is the only
  open question).
- No simulator comparison run, so no device-vs-simulator gap figure.
- No cost-model prediction, so no per-pipe error and no B9-consumable recipe list.

---

# Experiment: "cheaply fix the GetValue/SetValue" (2026-07-30)

Two changes were tried together, then isolated. Timing noise floor measured at
**0.7%** (3 runs of 1×1536×8×16×128×128: 5575.4 / 5614.0 / 5586.2 us).

## (a) Scalar-side caches — BIG REGRESSION, reverted

Added `float aScalar_[64]` + `float betaScalar_[64]` as members of `KernelComputeWy`,
filled in `BuildCumulativeG` / `LoadBetaChunk`, so the per-row loops never call
`GetValue`. Correctness held (cos 1.000000, 30-run determinism PASS) but:

| shape | before | with scalar arrays | delta |
|---|---|---|---|
| 1×256×8×16×128×128 | 1 032.7 | 1 563.0 | **+51 %** |
| 1×1024×8×16×128×128 | 3 872.2 | 6 002.6 | **+55 %** |
| 1×1536×8×16×128×128 | 5 731.4 | 9 007.0 | **+57 %** |
| 1×2048×8×16×128×128 | 7 652.1 | 11 961.9 | **+56 %** |
| 1×1536×16×32×128×128 | 11 424.5 | 17 956.7 | **+57 %** |
| 1×1024×8×16×64×64 | 3 064.6 | 4 313.0 | **+41 %** |

Note scalar *ratio* FELL (0.734 → 0.636) while total scalar *cycles* ROSE
(36.3 M → 49.2 M) and MTE3 ratio rose (0.387 → 0.451). Uniform ~55% across every
shape.

Hypothesis (not proven — would need `msobjdump` on the generated code): adding 512 B
of arrays pushed the `KernelComputeWy` object out of scalar registers / fast scratch
into a slower memory space, so *every* member access — including all the hot
GlobalTensor/LocalTensor members — became a load. The rising MTE3 is consistent with
the object being relocated. **Do not cache per-token scalars in member arrays on this
architecture.**

## (b) Op-count reduction — KEPT, ~3%

Pure algebra, no change to what is read from UB:
- hoist the per-row `Muls(…, -1.0f)` out of `ApplyLambdaNegStrictLower` into one
  `Muls` over the whole 64×64 gram (−64 ops, −64 `PipeBarrier`)
- replace per-row `Duplicate` + `Sub` with a single `Adds` against a precomputed
  `-a_j` vector (−64 ops, −64 barriers)
- fuse the two `Muls` in `BuildKBetaExpFloat` into one by multiplying
  `beta * expG` on the scalar unit (−64 ops)

Per row: 6 vector ops + 6 barriers → 4 + 4.

| shape | before | after | delta |
|---|---|---|---|
| 1×256×8×16×128×128 | 1 032.7 | 1 022.3 | −1.0 % |
| 1×1024×8×16×128×128 | 3 872.2 | 3 733.8 | −3.6 % |
| 1×1536×8×16×128×128 | 5 731.4 | 5 569.1 | −2.8 % |
| 1×1536×16×32×128×128 | 11 424.5 | 11 078.6 | −3.0 % |

Above the 0.7 % noise floor, but small. Correctness: cos 1.000000 on all shapes,
30-iteration determinism PASS.

## Conclusion

`GetValue` per se is **not** the cost — removing all of them made things 55% worse.
The scalar pipe is saturated as the *instruction issuer*, so only cutting total
instruction count helps, and the per-row vector loops are only ~450 of the ~2 500
instructions per task. The remaining ~2 018 MTE3 instructions per task come from the
`TPosition::GM` matmul staging (`compute_wy_cube.h:17-21`) — that is where the time
is, and no amount of scalar tidying will reach it.

---

# A/B vs main: torch WY path vs the NPU op (2026-07-30)

Op-level, on device 3. Compares the exact function the PR replaces,
`_compute_kernel_inputs_from_torch_wy` (imported from the PR tree, not a copy — the image's
vLLM lacks `vllm.third_party.flash_linear_attention` so that submodule is stubbed; the WY
prefix function does not use it), against the aclnn op. Both timed as wall clock with an
explicit device sync; harnesses `wysim/ab_torch.py` and `wysim/ab_op.py`.

| shape B×T×Hk×Hv×K×V | main (torch WY) | PR (NPU op) | speedup | saved/call |
|---|---|---|---|---|
| 1×1536×8×16×128×128 | 21 968.6 us | **5 119.4 us** | **4.29×** | 16.85 ms |

Cross-check: the op's wall clock (5 119.4 us) matches its msprof kernel time (5 138.7 us),
so host dispatch is negligible and the op is kernel-bound.

Why main is so slow: `_compute_kernel_inputs_from_torch_wy` does the triangular solve as a
**Python loop of 63 sequential iterations** (`for row_idx in range(1, chunk_size)`), each
issuing several NPU kernels. The op replaces that with 6 rounds of log-depth doubling on the
cube. This is the sequential-substitution cost the design spec attributes to the kernel —
it is in the torch reference, not in the kernel.

## Scope / caveats

- **This is op-level, not a full model prefill.** A model-level e2e needs the PR's torch
  extension rebuilt: the image's `_C_ascend` exposes none of the chunk ops
  (`chunk_gated_delta_rule_compute_wy`, `chunk_gated_delta_rule_fwd_h`, `chunk_fwd_o` all
  absent), and the full GDN layer needs fwd_h + chunk_fwd_o too. `/home/models` has
  Qwen3.5-9B (not the 4B the CI test uses).
- Numerics were verified separately: cos 1.000000 vs the fp32 golden on 7 shapes, plus the
  beta==0 padding case.
