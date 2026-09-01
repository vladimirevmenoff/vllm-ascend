# Qwen3.5-9B optim session 2026-08-19 — targets: TTFT <1s, TPOT 30ms

## MEASURED on combined_13122 build (310p-a dev6, W8A8-modelslim, 2048in/1024out BS=1)
| | TTFT | TPOT | note |
|---|---|---|---|
| this build | **1954.9 ms** | **61.88 ms** | compute_wy INELIGIBLE (K=128>64) → torch WY fallback |
| ref Aug-12 (old op, K≤128) | 1266 | 62.6 | PR at ec0d40ecd |
| ref main no-op | 2020 | 62.5 | |
Decode unchanged ✓ (61.9 ≈ 62.6, rgdr #9264 upstream in both). TTFT regression = K-limit, quantified: 689ms lost.

## MTP + int8 lm_head MEASURED (w8a8-lmhead-mtp ckpt, PIECEWISE, dev6, real prompts, greedy)
- **k=1: 47.31 ms/token** (from 61.88 → 1.31×), acceptance 80.9–85.5%, outputs coherent.
- **lm_head W8A8 fix VERIFIED WORKING**: ckpt has only int8 lm_head (+deq_scale); old code
  crashed on load ("no param named lm_head.deq_scale"), now loads + correct text at speed.
- vs 34ms projection: draft still ~25-30ms/step (α≈0.83 ⇒ verify+draft≈86.6ms, verify≈57-62)
  — drafter runs EAGER under PIECEWISE (only target model graph-captured). Draft cost is
  the next decode bottleneck, not lm_head read anymore.
- **k=2 PIECEWISE: 43.33 ms/token**, acceptance 58.4–76.6% (position-2 drafts accepted less;
  code prompt best 36.5). Ladder: 61.9 → 47.3 (k=1) → 43.3 (k=2). Eager draft cost bounds it.
- **FULL_DECODE_ONLY k=2: 40.04 ms/token** (from 43.3 PIECEWISE). Required TWO new fixes
  (committed, in combined_13122):
  - 2a8063f96: get_splitfuse_mask does D2H(.cpu/.tolist)+H2D at forward time → aborts
    capture (107030). Moved mask build into AscendAttentionMetadataBuilder310.build()
    (host-side, outside graph); for ≤64-token batches content copy_ into stable-address
    per-size buffer so replays read fresh values. attention_v1 falls back if attr absent.
  - 0f284f2e0: capture dummy path has seq_lens_cpu=None → D2H fallback in build() (legal
    there — build runs before the captured region).
  - Output text identical to eager runs ⇒ mask replay content correct.
- **torch_npu WARNING**: 2.10.0.post4 (a) still lacks _npu_flash_attention_v3/splitfuse_v2,
  (b) BREAKS aclnnMatmulWeightNz (161002) on CANN 9.1.0-beta.1 → engine dies in profile_run.
  Stay on 2.10.0.post2. Also: a detached pip retry-loop reinstalled post4 AFTER manual revert
  — kill zombie pip loops before pinning versions.
- Decode ladder: 61.9 → 47.3 (k=1 PIECEWISE) → 43.3 (k=2 PIECEWISE) → 40.0 (k=2 FULL) →
  **37.86 (k=3 FULL)**; code prompt 28.2. Acceptance at k=3: 47-54% (deep positions weak).
  Target 30: 1.26× away. Remaining decode levers: profile draft/verify split under FULL
  (residual eager overhead?), W4A8, TP=2. Physical floor ~52ms verify-only ⇒ spec-decode
  is the only single-chip path below it — and it's working.

## Box moves: `ssh 310p-a` (192.168.100.8, port 10005)
- Old box `ssh 310p` (.4, port 10002) also back up. Same LAN, but box-to-box ssh keys
  BLOCKED by policy — pipe small files through laptop: `ssh 310p 'tar czf - x' | ssh 310p-a 'tar xzf -'`.
- 310p-a: 8 chips idle, /home/models has Qwen3.5-9B + Qwen3.5-9B-w8a8-modelslim
  (NO -mtp / -lmhead ckpts — rebuild there, don't copy 11GB).
- Container `claude_13122` created via /home/claude_bench/repro/make_container.sh.
- quay nightly tag is 2 WEEKS STALE (a538cd4b1bc7) — pulls don't help.
- github UNREACHABLE from box; PyPI WORKS (msmodelslim 26.1.0 pip-installable! no need to
  copy /home/c00692241/msit source).

## Branch state (local repo)
- **RGDR decode optim already merged upstream as #9264** — rgdr_optim branch's kernel work
  is redundant; its only real delta vs PR head: 3 example scripts + compute_slot_mapping_draft
  call in model_runner_310p.
- PR 13122 head = origin/pr-11941 @ 9479081 (compute_wy + MTP graph fixes + D2H drop),
  based on 2026-08-18 main, pins vLLM v0.27.1 (image ships 0.25.1 → must source-install).
- Built branch `combined_13122` = PR head + rgdr extras + NEW lm_head quant fix (484d9fd20).
  Pushed to fork (NOT to pr-11941 — don't touch PR branch without asking).

## lm_head W8A8 unblock — patch written (484d9fd20), UNTESTED
`vllm_ascend/_310p/quantization/modelslim_config.py` get_quant_method: every
VocabParallelEmbedding (incl ParallelLMHead) was forced to AscendUnquantizedEmbeddingMethod310.
Fix: ParallelLMHead + not skipped → AscendLinearMethod(create_scheme_for_layer(..., "linear")).
Verified compatible: vLLM 0.27.1 VocabParallelEmbedding.weight_loader handles scalar
(input_scale), 1-D per-channel (deq_scale, narrow+zero-pad), 2-D weight; logits go through
lm_head.quant_method.apply (logits_processor.py:106); nothing in _310p touches lm_head.weight_nz.

## Plan (tasks)
1. Build in claude_13122: vllm 0.27.1 VLLM_TARGET_DEVICE=empty + vllm-ascend combined
   SOC_VERSION=ascend310p1 (~22min). Script: /home/claude_bench/build_13122.sh.
2. Bench W8A8 baseline (ref: TTFT 1266 / TPOT 62.6 on old base b2f683ca3).
3. MTP k=1/k=2 + PIECEWISE (ref proj: k=1→40ms, +int8 lmhead k=2→27ms; PIECEWISE fix in PR).
4. Prefill re-profile (old: TransData 167ms/8.6%, compute_wy scalar 0.73).
5. Rebuild ckpts on 310p-a: mtp shard via add_mtp_to_w8a8.py (mtp.* tensors from base ckpt),
   lmhead via quant_w8a8_lmhead.py + pip msmodelslim. Scripts copied to /home/claude_bench.

## Getting source onto the box — what actually works (learned the hard way)
- **gitee.com/mirrors/vllm WORKS from the box and is FAST** — has tag v0.27.1. Use this,
  never WAN-scp big files. `git clone --depth 1 -b v0.27.1 https://gitee.com/mirrors/vllm.git`.
- PyPI sdist (`pip download vllm --no-binary :all:`) works but crawls (~1.5MB/min).
- WAN scp from laptop ~100KB/s AND box connection drops every few minutes → byte-append
  resume produced a CORRUPT tarball once (append overlap). Box rsync too old for --append-verify.
- Box-side `nohup` scripts + `docker exec -d` survive the drops; anything tied to my ssh dies.
- vllm-ascend tree (12MB git archive) came via scp OK.

## Checkpoints built on 310p-a (all ✓ 2026-08-19)
- Qwen3.5-9B-w8a8-mtp (add_mtp_to_w8a8.py, defaults fit new box)
- Qwen3.5-9B-w8a8-lmhead — quant 95s on dev2, lm_head.weight=W8A8, 2255 entries,
  quant_model_description.json copied
- Qwen3.5-9B-w8a8-lmhead-mtp (combo for the 27ms projection)

## CRITICAL: PR 13122 head can't accelerate Qwen3.5-9B prefill (found 2026-08-19)
- Commit 6fa1b0081 (Aug 17, "limit lowered according to UB size") dropped compute_wy
  K/V limit 128→64: optimized kernel's InitBuffer ~242KB at K=V=128 > 192KB UB.
  **Qwen3.5-9B is K=V=128** → op ineligible → falls back to 63-iter torch WY → TTFT ~2s.
- The 1.58× (TTFT 1266ms) Aug-12 measurement was on OLD kernel (ec0d40ecd) which fit 128.
  The perf-optimization series traded away 128 support. PR headline feature dead for the
  model it targets. Flag to Dmitry.
- Bonus crash bug: python guard still allowed 128 → binding TORCH_CHECK killed engine
  (bench smoke died: "K must be <= 64 (310P UB), got 128"). Fixed guard→fallback: 40fd4ed44.
- Fix direction (task #6): stream K/V in 64-wide slices; A=k@k^T accumulates over K-halves
  in cube k-loop; U/W outputs chunk×K / chunk×V splittable per-slice.
### K=128 RESULT (2026-08-20 ~11:30): **PASSES — cos=1.000000 W/U/g at K=128 and K=64**
Final working combination (branch combined_13122, tip e888432a6):
two-pass solve + GM A-snapshot (SaveSnap/LoadSnap) + per-64-slice K staging + all cube
calls & tilings capped 64³ + **the actual numerics fix: two missing V→MTE2 syncs**
(attnLocal Brcb-scratch → BuildCumulativeG DataCopy; LoadSnap MTE2-write after V reads).

**THE 6-HOUR TRAP — three stacked caches made every device test lie:**
1. `csrc/build` cmake cache served a STALE kernel .o across 6 "rebuilds" → always
   `rm -rf csrc/build build_out` before op builds.
2. vllm_ascend bootstrap prepends the package-tree vendor (_cann_ops_custom) over the
   default CANN vendors dir → per user directive: tree vendor now RENAMED .disabled in
   container; use ONLY default /usr/local/Ascend/.../opp/vendors (--install-for-all).
3. `/root/atc_data` runtime kernel cache → nuke after installs.
Every pre-10:30 "K=64 passes" was the stale ORIGINAL kernel. Trust only md5-verified
fresh binaries (kernel .o hash in filename is a SIGNATURE, not content hash!).
Debug knob via new tiling field: possible host/kernel ABI skew when host+kernel libs
come from different builds — knob stripped from final series.

### K=128 TWO-PASS DESIGN (worked out 2026-08-20, ready to implement)
Kernel per task: load K,V → kBeta=βK → gram A=(βK)@K^T (cube, inner-dim K fine at 128,
staging is GM not UB) → Λ mask → RHS (βV, γβK) → solve (I−A)⁻¹RHS via 6 cube doubling
rounds (UploadP/GemmApplyAdd×2/GemmSquare) or scalar Fp32ForwardSubstitution → store U,W.
**Blocker at 128 = three [64×128] fp32 bufs resident (kFloat/vFloat/kBeta 32KB ea) + tmp32
+ ws32 + lamOff16 + attn16 ≈ 242KB.**
Fix = two passes over the solve, one per RHS:
- kFloat aliases tmpBuf (K loaded→tmp fp32, kBeta built, cast→kHalf, dead before gram
  which uses tmp as scratch).
- Load V only in pass 2 (vFloat aliases same region as kBeta/pass-1 usage).
- Snapshot A after Λ (attnSnap 16KB new buf); pass1 solves W (kBeta), restore A, pass2
  solves U (vFloat). Doubling cube cost ×2 (P squarings repeated), apply cost unchanged.
- Fp32ForwardSubstitution likewise split: row loop twice, once per RHS.
- Budget/pass ≈ ws32+lamOff16+attn16+snap16+RHS32+half16+qHalf16+tmp32+misc2 = 178KB ✓.
- Host side: tiling.cpp MAX_HEAD_DIM→128 + STAGING back to 64×128; binding TORCH_CHECK 128;
  kernel MAX_SAFE_HEAD_DIM/WY_CUBE_MAX_HEAD→128; python guard back to 128 (40fd4ed44 revert).
- Perf estimate: doubling is small fraction of task time; expect ≈ old-kernel TTFT 1266ms,
  minus the later perf-series gains — measure.
- Files: csrc/moe/chunk_gated_delta_rule_compute_wy/{op_host/chunk_gated_delta_rule_compute_wy_tiling.cpp,
  op_kernel/arch20/compute_wy_kernel.h,op_kernel/arch20/compute_wy_cube.h}, csrc/torch_binding.cpp:1909,
  vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py:_can_use_npu_compute_wy.
- Test: examples/python/test_v310.py (sim/device), compare_vllm.py; gate cosine ≥0.999.

- UB budget verified (compute_wy_kernel.h Init): at K=V=128 the K-sized bufs are
  kHalf/vHalf 16KB ea, kFloat/vFloat/kBeta 32KB ea, qHalf 16KB, tmp 32KB, mmLocalWs 32KB,
  attn 16KB → ~242KB. At 64 → ~154KB. Slice plan: only per-slice kFloat/kBeta/vFloat
  (64×64) live at once; A build accumulates kBeta_slice @ k_slice^T (cube L0C accumulate);
  after triangular solve T (64×64, K-independent), W[:,s]=T@kBeta[:,s], U[:,s]=T@vBeta[:,s]
  per slice. Vector stages (beta*k*exp(g)) also per-slice. Est: kernel surgery in
  Init + LoadChunk + BuildAttn + Apply stages; tests exist (examples/python, ICPU).

## Build recipe that finally worked (container claude_13122)
- vllm: gitee mirror clone v0.27.1 + `VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation --no-deps`
  (needs `pip install setuptools_rust setuptools-scm` first).
- vllm-ascend: catlass submodule from **gitcode.com/cann/catlass** @41bf90da (gitcode works
  from box, github doesn't), then `SOC_VERSION=ascend310p1 MAX_JOBS=16 pip install -e . --no-build-isolation --no-deps`.
- Verify: vllm 0.27.1, compute_wy ✓, rgdr310 ✓, 8 NPUs ✓ (BUILD_OK 2026-08-19).
- Host-side `nohup` scripts DIE on box ssh drops; only `docker exec -d` survives. Use it for
  everything long-running.

## Gotchas carried forward
- `echo ===` breaks zsh (=-expansion); quote it.
- Pin ASCEND_RT_VISIBLE_DEVICES after checking npu-smi (shared box).
- Cold start ~5min; PYTHONUNBUFFERED=1.
- pip install vllm-ascend needs --no-deps (triton-ascend has no aarch64 wheel).

## FINAL BENCH with K=128 op live (2026-08-20 12:08, dev6, W8A8, 2048/1024 BS=1)
| | TTFT | TPOT |
|---|---|---|
| **this build (two-pass K=128 op)** | **1466.7 ms** | 61.69 ms |
| fallback (op ineligible) | 1954.9 | 61.88 |
| old-op reference (pre-perf-series) | 1266 | 62.6 |
Op recovers 488ms of the 689ms regression; remaining ~200ms = two-pass overhead
(doubling cube work ×2 + per-row slice casts) — perf polish candidate. Target <1000
still needs prefill work beyond compute_wy (TransData ND↔NZ etc., task #4).

## PR 13122 handoff
Fork branch `pr13122_rewrite` = pr-11941 − 2 MTP commits + guard-fix + squashed
"[Ops][310P] Restore K/V=128 ... two-pass solve" (8b4ce7d9a, byte-identical to
validated e888432a6). USER RUNS: `git push --force-with-lease origin pr13122_rewrite:pr-11941`.
Decode work (lm_head W8A8 484d9fd20, capture-mask 2a8063f96+0f284f2e0, MTP results)
lives on combined_13122 — separate PRs later.

## 2026-08-21 tau=8 probe verdict
- tau=8: all 9 shapes cos=1.0, canonical UNCHANGED (~1905-1935us). Gate-misfire theory DEAD.
- => 1445us U-phase cost is on the DOUBLING path itself; W-phase (same structure) free. Asymmetry unexplained.
- Next: inner cuts 6 (pre-U-apply) / 7 (post-U-apply) running to pin exact lines. tau reverted to 2.5.
- 2026-08-21 PM: sticky-shape flat; half-C UB C hangs aicore (reverted); half-domain betaV kept (cos=1.0, flat).
- Running: subtractive ablation chain to attribute the 45us/task U-region cost for real.
- 2026-08-21 ~15:15 UTC: pushed 74fa81e5b to PR 13122 (squashed kernel-opt series incl. NaN fix). Device battery for this exact code still pending; timing claim TBD.
- 2026-08-31: compute_wy DONE at ~800us canonical (787-843 quiet runs, 5.8x vs 4691). PR 13122 head 4faebc035.
  Hand-rolled Mmad path replaced the AscendC matmul lib (its ~1.8us/call overhead was 60% of runtime).
  Serving TTFT validation pending (needs full pip rebuild for tree vendor).

## 2026-09-01: wrapped for compaction
- Final: canonical ~800us (787-843 quiet, median ~808) = 5.8x; all-green battery; PR 13122 head 4faebc035.
- Everything needed post-compaction: talk/HANDBOOK.md (env, build, harness, inference, traps).
- Open item: serving TTFT validation (full pip rebuild first — box vendor holds single-op package).
