# chunk_gated_delta_rule precision on 310P — findings

## Setup
- Device: Ascend310P3
- Path: PyTorch fallback `chunk_gated_delta_rule_pytorch` (C++ op NOT loaded on this 310P)
- Comparison: bf16 inputs vs fp32 inputs through same function (fp32 internal compute)

## Key Results (bf16 vs fp32 reference)

| Test | Cosine | MaxAbsErr | RelErr | State Cosine |
|------|--------|-----------|--------|--------------|
| Qwen3 dims T=64 (1 chunk) | 0.999997 | 1.03e-3 | 8.3e+1 | 0.999998 |
| Qwen3 dims T=512 (8 chunks) | 0.999996 | 7.61e-4 | 6.8e+2 | 0.999998 |
| Qwen3 dims T=1024 (16 chunks) | 0.999996 | 9.15e-4 | **3.7e+6** | 0.999998 |
| Qwen3 dims T=2048 (32 chunks) | 0.999995 | 7.11e-4 | **1.1e+4** | 0.999998 |
| Strong gate g=2.0 | 0.999995 | 4.49e-4 | 7.5e+2 | 0.999998 |
| Weak gate g=0.001 | 0.999995 | 1.60e-3 | 5.0e+3 | 0.999998 (mae=5.7e-3) |
| fp16 instead of bf16 | **1.000000** | 1.13e-4 | 1.6e+2 | 1.000000 |
| **No L2 norm** | **NaN** | **NaN** | **NaN** | **NaN** |

## Findings

1. **NaN without L2 norm** — bf16 inputs + no L2-norm → NaN in later chunks. The iterative triangular solve (lines 144-147) diverges with unbounded q/k. Production uses l2norm=True so this doesn't fire, but the numerical instability exists.

2. **bf16 mantissa is the bottleneck, not the algorithm** — fp16 (10-bit mantissa) gives cosine 0.9999999 while bf16 (7-bit) gives 0.99999. The L2 norm computation at line 107-108 `F.normalize(query).to(query.dtype)` happens IN bf16 before fp32 upcasting → 7 bits of mantissa truncation on normalized vectors.

3. **Relative error is enormous on near-zero elements** — up to 3.7e+6 for T=1024. Cosine (direction) is fine but individual elements near zero have wildly wrong magnitudes. This matters for downstream layers that are sensitive to small values.

4. **State drift is minimal** — chunk-to-chunk cosine drift is only ~2e-6 even over 32 chunks. The fp32 internal accumulation handles state well. This would be MUCH worse if the C++ kernel's bf16 state writeback (confirmed in epilogue code) were used.

5. **Weak gate amplifies state error** — g_scale=0.001 gives state mae 5.7e-3 vs 1.2e-3 normal. When state barely decays (exp(g)≈1), bf16 input errors accumulate without being dampened.

## C++ kernel (NOT tested — not loaded on 310P)
The C++ kernel `chunk_gated_delta_rule_fwd_h` writes state as bf16 between chunks (confirmed: `block_epilogue_gdn_fwdh_update.hpp:159` casts fp32→bf16 for non-final chunks). This would cause far worse drift than the Python fallback which keeps state in fp32. The kernel has Ascend950 (310P) arch paths but is not compiled/loaded in this docker.

## Unresolved
- Is the C++ kernel supposed to run on 310P? If so, its bf16 state writeback is the real precision problem.
- The NaN-without-l2norm: is this exploitable in production? (prob not — l2norm always on)
- Multi-layer error compounding: 32 layers × 0.99999 cosine per layer = ?
