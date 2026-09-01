# progress_log — improve mode, chunk_gated_delta_rule_compute_wy (2026-08-20)
Config: migration_config.yaml (PASS). Branch: 310p-chunk_gated_delta_rule_compute_wy-skill-port
(from combined_13122 @ e888432a6 = validated K=128 two-pass kernel, PR 13122 head 8b4ce7d9a).
- [x] 0a prelude smoke (msprof+numpy+gcc OK in container claude_13122)
- [x] 0b gate: YES + improve_existing → IMPROVE mode (skip B2-B11.5 emission)
- [x] 0c branch + workspace dirs + snapshot sha
- [x] B1 audit: hand-authored block_decomposition.yaml (extractor: 0 matches, bespoke kernel)
- [ ] 10c cosine harness pre-flight: need shapes_light.txt ≥8 shapes + golden + compare
      (extend /work/test_bisect.py: currently 3 shapes; add B/T/Hk/Hv variants)
- [ ] 11 B12 baseline-measure (msprof per-pipe + cost-model gap via tools/baseline_diff.py)
- [ ] 12 B13 improvement-audit (recipes_machine DSL, ≥5% gate)
- [ ] 12.5 B14 loop — PAUSE per finding for --user-authorized=<id> (SP2)
- [ ] 13 B15 device-benchmark; 14 C2 knowledge-capture; 14b final report
Known candidates (user-flagged): single doubling chain for both RHS (two-pass repeats
cube work); per-row scalar-issued Cast loops in K-slice staging; MTE2/V overlap.
Exec facts: build MUST rm -rf csrc/build build_out (stale-cache trap); serving needs pip
editable rebuild after .run install; correctness ref = torch WY, gate cos≥0.999.

## Stage-cut ladder (measured, prod shape, steady us cumulative) 2026-08-20
loads+bK+g 314 | +gram+Lambda 512 | +gate+T-build 590 (!T-build ~78us) |
+W-apply/store 2442 | FULL 4620.
=> ~4030us (87%) in the two solve blocks; only fits the SCALAR fp32 fallback
(2016 serial Axpys/RHS). Gate ||A||inf>=0.75 fires on essentially all data →
every cube-path optimization was invisible. Fix shipped: threshold → 4.0
(a5a25384a) + GM Lambda table (ef2c07ac4) + mmWs 8KB (c58108996); gated on
full 9-shape battery (seed-7 data IS the adversarial case).
npu-pipe-optimizer: hand YAML (wy_hand.yaml) parses; rolled-mode flag synth
FAILS (9 unorderable hazards) — buffer aliasing from the UB diet blocks
overlap; revisit post-gate-fix with 24KB freed.

## thr-1.5 verdict (2026-08-20 ~19:40): ALL 9 PASS cos>=0.9999999; kept (a5a25384a+followup).
Canonical 4312us (-8%). New decomposition: fast path = ~12 IterateAll x ~10us lib
overhead/call. NOTE: the "any 128-dim cube call hangs" lore (drove all <=64 slicing,
64^3 tilings) was established on STALE BINARIES — may be phantom. De-slice probe next:
gram single K=128 call + applies nDim=128 single call. If clean: calls/task halve.
Then: async Iterate pairing (square || apply), early-exit rounds when P underflows
(production Λ-decay win), npu-pipe overlap w/ un-aliased buffers.

## BLOCKED SUBSTITUTION LANDS (2026-08-20 ~20:30): canonical 4691→1979us (2.37x), cos=1.0 x9
4c9590a84: exact fp32 solve, 16-row blocks, off-diag on cube vs solved prefix.
All shapes ~halved (1,256: 598; 1,192: 822; 1,512: 376; min-shape 210).
Remaining to 3x (<=1560): 1.27x. Next: BLK 8 probe → async pairing → overlap.

## 2026-08-21 session state (canonical scoreboard, all cos=1.0 x9)
4691 base → 1979 (blocked substitution 4c9590a84) → 1933 (barrier-free diag cfd1b45a0)
→ 1902 (store/load overlap ae2e68f4d + fix) → 1920±15 plateau after thr2.5 (τ change
bought ~0 on canonical). BUG FIXED en route: halfBuf fp32-C-scratch overflow at
head-dim 64 clobbered rhsBuf (d944aa9a0) — was the REAL cause of both K64 cos-fails
(0.16@τ4, 0.21@τ2.5); numerics of fp16 doubling are exact per host sim (dblsim.py).
K64-shape now 433us (was 1200 at session start).
Current: 2.44x. Target 1560 needs −360. Mini-ladder2 running (cuts 2/4) to split
the 1920 into loads+gram | W-solve | U-tail on the CURRENT kernel.
Process guards active: verify tarball size after scp (0-byte upload happened),
docker start claude_13122 (container died once), stale-log lines in opbuild.log
from prior failed builds — filter by date.

## 2026-08-21: tau=8 probe + U-block bisect
- tau=8 battery: cos=1.0 everywhere, prod shape 1905-1935us — NO CHANGE vs tau=2.5.
  Conclusion: gate isn't the problem; U-phase 1445us happens on doubling path. tau reverted to 2.5.
- (Also confirms doubling numerically exact in-silicon at tau=8, matching dblsim.)
- Bisect: STAGECUT_6 (post beta*V, pre U-apply) / STAGECUT_7 (post U-apply) chain running.
  Expected: cut6~500 if V-load overlapped; cut7 tells if GemmApplyReplace(useU) is the cost.

## 2026-08-21: c67 bisect result (canonical, tau=2.5)
- cut4 (post W-solve+W-store+V-load-kick) = 450; cut6 (post V-cast+betaV) = 1200; cut7 (post U-solve) = 1920; full = 1920.
- => V-cast+betaV segment = 750us (!!), U-solve = 720us, tail = 0. W region literally free; U region 1470us despite identical structure.
- Mechanism suspect: per-call SetOrgShape/SetSingleShape churn (4/task) on scalar pipe + shared mmApplyU_ across BuildT/U-solve.
- Probe in flight: sticky-shape (both solves on mmApplyW_, reshape only on nDim change; mmApplyU_ dedicated 64^3).
- Fallback if flat: fuse W|U into one 64x256 apply (halves calls, batches cast/store); then attack V-load row-copy chain.

## 2026-08-21 afternoon: three flat probes + ablation pivot
- sticky-shape (no per-call SetOrgShape): FLAT ~1990. Reshape churn not the cost.
- half-C UB matmul: AICORE TIMEOUT — half C to VECCALC wedges 310P cube. Documented in cube header.
- uhalf2 (betaV in half, in place; h->f cast + fp32 mul + cast-out deleted): cos=1.0 all 9, FLAT ~1940. KEPT (less work, same speed).
- => cut-ladder attribution unreliable; every local op removal is flat. Pivot: subtractive ablations on real control flow
  (NOSOLVE / NOVLOAD / NOUSTORE / FULL restore) — timing-only, cos expected to fail on ablated builds.

## 2026-08-21 ~14:20 UTC: HARNESS NaN BUG — ALL cos=1.0 SINCE `blocked` INVALID ON CANONICAL
- run_fast.py used worst=min(worst,cos); Python min() ignores NaN -> NaN W/U reported as min_cos=1.000000.
- Saved JSONs: baseline_light w/u GOOD; blocked.json onward w=nan u=nan on canonical (also deslice, t8, t25, sticky, uhalf2, all abl_*).
  descal.json + ub_probe.json eras were GOOD (w=1.0, u=0.99999) - bisect anchors.
- => the whole 1979->1920 plateau + all cut/ablation attribution was measured on a kernel emitting NaN W/U on canonical.
  4691 baseline valid. Everything since must be re-derived after correctness restored.
- Harness fixed (NaN -> cos=-1.0). truth.json battery = current kernel's real damage map.
- Next: tau=0 vs tau=1e30 probes attribute NaN to fp32-substitution vs doubling path; then git bisect local commits (~5 builds).
- Gate-always-true theory WRONG (based on poisoned ablations). Real story: NaN swallowed.

## 2026-08-21 ~15:15: slicefix battery RED — K=128 shapes still NaN
- Pushed candidate (sliced applies + scalar fallback + tau2.5): 8/9 NaN; ONLY 1,256,8,16,64,64 (K=V=64) green.
- NaN correlates with dim=128. Suspects among never-truly-verified post-deslice commits: storeBuf overlap ae2e68f4d,
  uhalf2 half-betaV, sticky shapes, or my 2-slice apply loop.
- PR 13122 head 34d702c58 = this red code (K=128 broken). Fix will be pushed on top.
- Pivot: goodera chain = d95c30a8e arch20 (last device-verified good) + tau2.5 → sanity green + honest optimized timing.
  Then re-add post-deslice pieces stepwise. Canonical timing at slicefix (sliced+scalar): ~4300us (scalar fallback dominating).

## 2026-08-21 ~15:30: goodera (d95c30a8e) ALSO NaN today -> env-vs-code test
- d95c30a8e + tau2.5: 8/9 NaN, K64 shape 0.206 (not even NaN-clean). Same code class was green yesterday 15:09.
- Goldens verified clean (08-20 timestamps, no NaN). No op_host commits post-good. JSON chronology: last good run descal 08-20 15:09;
  first NaN gatefix 08-20 16:19 (cut builds + fusion probe in between).
- Hypotheses: (a) device 6 wedged (aicore hang 12:59 today, maybe earlier ones yesterday), (b) yesterday's good runs were dirty-tree states.
- Running: goodbase chain = PR-head 8b4ce7d9a arch20 (serving-validated code). NaN => environment (npu reset); green => commit bisect.

## 2026-08-21 15:48: ROOT CAUSE FOUND — LOCAL_WORKSPACE 8KB overruns at head dim 128
- Chain of evidence: baseline arch20 NaN on box (env not code) -> dev7 also NaN (not device) -> box op_host md5 = campaign HEAD
  (chains only ever re-ship arch20; tiling.cpp stale since ~yesterday 16:00) -> tiling diff: c58108996 ws 32KB->8KB.
- Mechanism: matmul lib stages VECCALC A/B via localWs; 64x128 half B = 16KB > 8KB -> silent overrun -> NaN.
  64x64 = exactly 8KB -> K=V=64 shapes stayed green, masking it. First NaN run (gatefix 16:19 08-20) right after the ws-8KB ship.
- Fix: ws back to 32KB (commit), chains now ship FULL op source (op_host+op_kernel) - never stale tiling again.
- wsfix chain running: HEAD kernel (sliced+scalar+tau2.5) + ws32. PR 13122 head still carries ws8KB tiling - push fix after green.

## 2026-08-31: flat probes converge on per-call lib overhead; r3 falsification probe
- ubgram (UB-fed gram, VECCALC B-transpose works after decl-order fix): green, FLAT 1530-1590.
- Model: ~17 IterateAll/task x ~2us API latency ~= 1100us of 1530. All plumbing probes flat because staging isn't the cost.
- r3 probe running: DOUBLING_ROUNDS=3 (timing-only) removes 6 calls/task -> predicts ~-380us if model right.
- REOPENED: "direct C into rUb NaNs" was misattributed — NaN era began with ws8KB ship (16:19 08-20), BEFORE deslice (16:54).
  Full-width direct-C solves likely fine with ws32 -> -2 calls + no writebacks/compacts. Also BuildT round1 = I+A (vector, -1 call).
- If r3 confirms: stack (a) full-width direct-C solves, (b) round1 vector shortcut => ~14->13 calls, est ~1250-1350;
  then raw Mmad rewrite of BuildT+solves for the rest of the way to 800.

## 2026-08-31 12:55: RAW MMAD GRIND begins (user-ordered over accepting 3x)
- basic-cfg dead on m200 entirely (copy_cube_in_using_ub rejects it for ALL input positions). Box green-restored (probe7 ~1550).
- Study agent extracted the m200 recipe from toolkit sources (talk/mm-handroll-m200-recipe.md): UB->L1 nd2nz per-16col DataCopy,
  load2d L0A (Zz, stride K1) / L0B (Zn, ifTranspose=true), Mmad(m,n,k,cmatrixInitVal=true), CO1->UB BLOCK_MODE_MATRIX + Muls NZ->ND.
- WyMicroMm implemented (compute_wy_micro_mm.h): A1/B1/A2/B2/CO1 TBufs, event chain V_MTE3/MTE3_MTE1/MTE1_M/M_V/V_M.
- mm1 = micro-mm wired into T-updates only (5 calls/task). Iterating: layout-cos bugs expected before green.
- Unknowns from recipe: UB->L1 executing pipe (assumed MTE3), LoadData3D defaults (avoided: using load2d).

## 2026-08-31 15:10: ALL-GREEN MICRO KERNEL — 4.3x — ON PR
- mm9: AXIS4 PASS all 9 (K<128 shapes on lib gram at 0.9995; all 128-dim shapes exactly 1.0).
- Canonical quiet ~1071-1100us = 4.3x vs 4691 baseline. All 14 cube calls hand-rolled (WyMicroMm).
- Bug ledger for micro path: cross-call M_MTE1+MTE1_MTE3; V_MTE2 WAW before V-load; MTE3_V before storeBuf NZ reuse;
  K<128 gram stays on lib (micro-gram rounding shifts gate boundary on weak-decay data).
- PR 13122 head updated (squash, ~4.3x claim). Next: drop lib for 128-dim path? (lib still used K<128 gram + Init),
  vector cuts (half-domain front, NZ-resident doubling) toward 800.

## 2026-08-31 17:15: TARGET REACHED — ~800us canonical (787-843 quiet, median ~808) = 5.8x
- Final green build (mm16, PR 4faebc035): all 14 cube calls on WyMicroMm; NZ-resident doubling; I_NZ table;
  V-load under doubling + storeBuf C-scratch + qHalf W-B-casts (128-dim path only; K<128 keeps mm13 layout + lib gram).
- Ladder: 4691 -> 1530 (lib-era green) -> 1370 (micro T-updates) -> 1071 (all-micro) -> 812 (NZ-resident) -> ~800 (V overlap).
- Dead ends: GM/UB-half-C (hang), basic MatmulConfig (m200 incompatible), fused rounds (flat), P==0 exit (toxic),
  half-domain betaK (double-rounding), lib-Init skip (hang), early-V+storeBuf-scratch on lib-gram path (K64 corruption).
- NEXT (not done): serving validation — full pip editable rebuild (regenerates tree vendor with ALL ops; box tree vendor
  currently holds only this op from targeted installs) then TTFT bench vs 1467ms.
