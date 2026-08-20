# improvement_audit — chunk_gated_delta_rule_compute_wy (T3, 2026-08-20)
Baseline: workbench/analysis/baseline_light.json (9 shapes, cos≥0.9999998).
Canonical 1,1024,8,16,128,128: 4691µs steady (old pre-two-pass kernel, same shape: 3846µs
→ two-pass tax = +845µs, +22%). Pipe prior (msprof, old kernel, same shape class):
scalar 0.73, mte3 0.385, mte2 0.23, vec 0.15, cube 0.019 — issue-bound, cube idle.
Fresh msprof pending (channel contention, retry armed).

## F1-explicit-T  [predicted −25..30% op time]  ← APPLY FIRST
Math: A strictly-lower ⇒ T=(I−A)⁻¹ computable by doubling ON T (6 in-place rounds,
64×64), then W = T@(γβK), U = T@(βV) — 2 applies total.
Current: doubling chain re-run per RHS ×2 passes (12 applies + 10 squarings + 12
P-uploads + GM SaveSnap/LoadSnap). Removes: second pass, A snapshot, ~14 cube calls,
~14 GM stagings, ~14 PIPE_ALL drains per task. Predicted ≥ the measured +845µs
two-pass tax plus part of the original per-RHS chain: −25..30%.
Risk: none to numerics class (same fp32 accumulate; fp32-substitution fallback kept).
Transform: kernel-only (SolveOneRhs → BuildT + two GemmApply calls; cube.h gains
GemmBuildT; host tilings unchanged — all calls stay 64³).

## F2-strided-cast  [predicted −5..8%]
Stage-1 issues 64 scalar-dispatched Cast ops per K-slice (128/task at K=128) —
feeds the 0.73 scalar-issue prior. Replace with ONE Cast per slice using unary
repeat params (dstRepStride=64/8, srcRepStride=alignK/8, 64 repeats).
Risk: low; repeat-stride limits fine (16 blocks ≤ 255).

## F3-precise-events  [predicted −5..10%]
Every cube helper call is bracketed by PIPE_ALL (~8-22 full drains/task after F1).
Replace with precise event pairs (MTE3→MTE2 for staging, CUBE(V)→V for C-writeback),
freeing MTE2 to prefetch next task's K/β/g into a ~16KB double buffer (UB headroom
~46KB post-F1). Risk: medium (event discipline — validated by 9-shape harness +
3-run determinism).

## F4-overlap-passthrough  [predicted −2..3%]
StoreQKKernel (independent MTE2→MTE3 q/k copies, 1-in-headGroups tasks) serialized
at task end; interleave with solve. Risk: low. May fall below the 5% gate — apply
last, keep only if measured ≥5% or bundled.

Order: F1 → F2 → F3 → (F4). Gate per apply: harness 9-shape cos≥0.999, ≥5% measured
speedup on canonical, no shape regression >2% (variance-aware, 3-run arrays).
