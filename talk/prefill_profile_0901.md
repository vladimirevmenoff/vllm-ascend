# Prefill profile 2026-09-01 — W8A8, new ~800us kernel, dev 6

One request 2048 in / 32 out, profiled TTFT 1322.8 ms (unprofiled bench 1323.5 — overhead nil).
Prefill device-busy 1289 ms. Trace: box `/work/prof_ttft_0901/`, analysis `w8a8new_analysis.txt`.

## Prefill top ops (1289 ms device)

| op | ms | share | count | note |
|---|---|---|---|---|
| **ChunkGatedDeltaRuleComputeWy** | **273.7** | 21.2% | 24 | OUR op — see below |
| QuantBatchMatmulV3 | 259.2 | 20.1% | 152 | memory-bound GEMMs |
| TransData | 127.8 | 9.9% | 305 | ND↔NZ shuffles |
| ChunkFwdO | 107.7 | 8.4% | 24 | aclnn GDN |
| ChunkGatedDeltaRuleFwdH | 73.6 | 5.7% | 24 | aclnn GDN |
| RmsNorm | 68.0 | 5.3% | 153 | |
| UnpadFlashAttentionNz | 61.6 | 4.8% | 8 | full-attn layers |
| CausalConv1dV310 | 57.0 | 4.4% | 24 | |
| SwiGlu + AscendQuantV2 | 101.6 | 7.9% | | |

vs Aug-07 profile (torch WY loop era): prefill 1938→1289 ms; WY cluster 670-790→274 ms.
Decode unchanged: QBMM 43.7 ms/step (71.9%) + fp16 lm_head MatMulV2 11.5 ms (18.9%).

## THE finding: compute_wy runs 7.6x slower in serving than in harness

- Harness same shape (1,2048,8,16,128,128): **1.5 ms**, cos 1.0. Serving: avg **11.4 ms**,
  per-call spread 7.6–15.1 ms across the 24 layers (layer/data-dependent).
- Cause (high confidence): **fp32 fallback gate** — FP32_FS_ROW_SUM_THRESHOLD=2.5 row-sum
  check routes tasks to the slow scalar Fp32ForwardSubstitution. Real model g/βK trips it;
  harness synthetic data stays on the fast micro-Mmad path. Spread = per-layer fallback
  fraction. NOT YET instrumented-confirmed (no counter in kernel).
- If all tasks ran the fast path: 24 × 1.5 = 36 ms → **saves ~238 ms → TTFT ~1.08 s**.

## Ranked TTFT opportunities (target <1 s, need ~330 ms)

1. **Kill/shrink the fp32 fallback** (~238 ms): options (a) confirm fallback fraction with
   a counter first; (b) fp16 doubling + fp32 Mmad accumulation may already be accurate
   enough (precision loss only in casting A/T inputs; L0C accumulates fp32) — re-derive
   threshold or Newton-refine T'=T(2I−(I−A)T) on cube; (c) make fallback itself fast.
2. **TransData 128 ms** — format conversions around QBMMs.
3. **ChunkFwdO+FwdH 181 ms** — aclnn GDN ops, would need same micro-Mmad treatment.
4. QBMM 259 ms — memory-bound (mte2), only fewer bytes helps.

№1 alone gets ~1.08 s; №1 + half of №2 crosses <1 s.
