# .claude 310P skill system → management deck (2026-08-17)

Deliverable: `talk/310p_porting_system_howitworks.pptx` — 3 slides, 16:9, "how it works" insert
to sit next to the existing results slide. 2-slide variant: `310p_porting_system_howitworks_2slide.pptx`.
Builder: scratchpad `build_deck.py <3|2>` (rerun to restyle; MODE arg picks the variant).

## What the .claude system actually is

19 skills + shared knowledge library, own git repo inside `.claude/`. Orchestrator A1 reads
`migration_config.yaml`, walks 14 steps → validated 310P operator under
`${WORKSPACE.PATH}/${SOURCE_OPERATOR.NAME}/`.

Counts (verified, not estimated): 35 numbered debug RULES, 17 bug classes, 16 fully-authored
blocks (7 files each) of ~50 planned, 17 decision-rules, 17 transferable-patterns, 21 recipes,
13 open-problems, 9 api-docs. shared/ ~27.5k lines, skills ~9.2k, tools ~3.4k.

## Positioning traps (got these wrong first pass)

- **.claude ≠ npu-pipe-optimizer.** npu-pipe-optimizer = paper §10's *proposed* 3-layer Python
  tool (DSL → graph → codegen). .claude *optionally consumes* it via
  `tools.npu_pipe_optimizer`; correctness-check has Mode A (synthesizer present) / Mode B
  (fallback). `ascend310p-yaml-author` exists only to bridge to it.
- **Empty `learnings/` + schema-only `benchmarks/` ≠ never ran.** Per-op artefacts live at
  `${WORKSPACE.PATH}/<op>/learnings/proposals/` — outside this repo, `workbench/` gitignored
  by design. Absence here is correct behaviour, not evidence of nothing.
- **Don't say "16 of ~50 blocks".** `blocks/INDEX.md`: remaining ~34 "fill in incrementally as
  B5 drives the need" — demand-driven. To management "16/50" reads as 68% unfinished. Invented deficit.
- **Existing `Ascend_review.pdf` (~40 frames, author Feodor Pisnitchenko) already teaches**
  DaVinci arch, 5 pipes, tiling, roofline, DMA cost model, job-shop formulation. Prerequisite
  deck — never re-teach it.

## Gates — exact numbers (use these, they're real)

- Cosine ≥ 0.99 vs golden; raised to 0.999 for dequant prelude / LayerNorm / RMSNorm / final
  logits. **Never lowered to make a test pass** (`shared/recipes/cosine-threshold.md`).
  Cosine necessary-not-sufficient — `max_err` gated alongside (NaN/Inf hides in cosine).
- mssanitizer memcheck + synccheck; 3-run determinism sweep on every multi-core fix (RULE 2).
- Perf change kept only if measured ≥5% faster AND no other shape regresses >2%; else auto-revert
  from snapshot. One git commit per ACCEPTED finding, finding-id in message.

## Enforcement is real, not prose

`runtime-hooks/scope_protect.sh` = PreToolUse hook emitting
`{hookSpecificOutput:{permissionDecision:"deny"}}`. Enforces SP0–SP4: no writes outside project
root, destructive ops (rm/git reset --hard/push --force/pkill/pkg-mgr) hard-stopped, references
read-only, shared/ promotion needs `--user-authorized=<finding-id>`. A1 refuses to start without
hooks; filesystem snapshotted outside workspace at start, re-diffed at end, any drift aborts.

## Best anti-slop specifics (unfakeable)

- Bug class 16: cosine plateau at **exactly 99.33%** ⇒ wrong golden variant, not a kernel bug.
- RULE 5 / blocked-ZN: `actualBlockNG = min(blockNG, N1 - blockIdx·blockNG)` *inside* partial
  last block; fixed `blockNG` only for skipping prior FULL blocks. (commit 939ba50)
- RULE 17/28: graph PASS + alias check PASS but still wrong → route through a V-pipe encode
  (Duplicate+Scatter) that changes no bytes.
- RULE 0 economics: 1 DumpTensor run = ground truth in 5 min; 5 blind fixes = 0 info in 25 min.
- Paper §9 = the *before*: several weeks/operator, 3 sync bugs in ~1000 lines, silent corruption,
  found only via analysis tools. Pair with AscendKernelGen 64% functional correctness on
  multi-pipe kernels (sync errors dominant) → pre-empts "why not just ask an LLM".

## Build environment notes

- No LaTeX (`pdflatex`/`xelatex`/`latexmk` absent) → Beamer not compilable here.
- No python-pptx globally, no LibreOffice → used venv at scratchpad `venv/`; **cannot render
  pptx to verify visually**. Wrote `check_overflow.py` (Calibri 0.53em/1.2em estimator) +
  `bounds.py` instead. Final: 0 overflow, 4 boxes ≥0.88 ratio, nothing off-slide.
- Palette from `Ascend_review.tex`: Accent #1F4E79, Accent2 #0E7490, LightAccent #EAF2F8,
  SoftGray #F5F7FA, DarkGray #4A5568, Good #2F855A, Warn #B7791F, Bad #C53030. Calibri not
  Fira/Metropolis (font not installed on most machines → unpredictable fallback).

## Deliberate editorial choices (don't undo without reason)

- **Slide 2 col 3 compressed to 2 bullets.** Original 4 bullets (deny-hooks, refuses-to-start,
  snapshot/drift-abort, per-finding sign-off) answered "will the agent damage the repo" — wrong
  altitude for management, and plants the idea an autonomous agent is loose in the codebase.
  Highest-risk hostile-question generator in the deck. Detail moved to speaker notes, flagged
  "don't volunteer unless asked".
- **64% AscendKernelGen de-emphasised** to a single hairline-separated line. Two full-width
  "before" bands 20s apart competed; §9 is the stronger one because it's *our* measured baseline,
  not a citation about someone else's LLM.
- **"16 kernel blocks" → "a growing library"**. Bare count invites "out of how many?" — reopens
  the demand-driven-vs-unfinished door for zero management value.
- **VERIFY bullet says "on-device when hardware is available"** — device tests are Tier-2/
  device-conditional; unqualified "device" overclaims to anyone who knows the pipeline.
- Speaker notes on all 3 slides (~1.5k chars each): one-line frame, IF-ASKED depth, and the
  0.99-rationale (8° at 1024 dims) for the inevitable "why that threshold".

## COST MODEL — the differentiator (added 2026-08-18, user: "most important part")

Slide 2 is now the centrepiece. Facts, all from `shared/cost-model/`:

- **Composite lower bound** = `max(L_CP, L_res, L_DDR, L_cyclic)` (paper §5.4 eq. lb_composite).
  Critical path / busiest pipe / off-chip traffic / cyclic steady-state (Hanen-Munier max cycle
  mean via Karp). Within **10–20 % of optimal MILP schedule** (paper §8.4) → usable for choosing
  tilings without building them.
- **DDR 17 B/cyc vs L2 114 B/cyc = 6.7×** (`pipe-bandwidths.yaml`, Ascend310P3).
- **Affine DMA**: `T_xfer = λ_setup·N_calls + bytes/BW_peak`. λ_setup(MTE2)=422 cyc → breakeven
  ~7 KB/call; below that, setup-dominated ⇒ merge. MTE3=15, MTE1=31.
- **L2-aware blended L_DDR** with spill ramp; `L2_eff = 14 MB × 0.7 = 10 MB`. Implemented in
  `analysis-tool-templates/pipe_utilization_model.py::l2_blended_mte2_cycles`.

### THE money evidence — m968 b1 q1 nopt

| | before | after |
|---|---|---|
| grid | NCoreNum=2, mc=4, disjoint cb | NCoreNum=1, mc=8, cores share weight |
| L2 hit | 0.04 % | 74.91 % |
| time | 63 906 µs | 22 097 µs (−65 %, ~3×) |

**`L_DDR_plain` predicts both tilings take similar time. `L_DDR_blended` correctly predicts the
second is 3× faster.** That is the whole "better than a roofline" argument in one number.

### Claim boundaries — do NOT overstate

- Formulas are the **framework paper's** (§3.1/§3.2/§3.3/§5.4). Ours = *operationalized +
  calibrated*: every candidate tiling scored pre-codegen, B8 emits predicted-vs-measured per
  component, >30 % persistent gap raises a calibration finding against the model. Claiming
  invention backfires — paper co-authors may be in the room.
- Cooperative-batch **is** auto-proposed: `coop-batch-predicate.md` ships a heuristic
  (`fracM ≥ aicNum AND weight_shared_across_batch` → NCoreNum=1, restructure). Open problem is
  only the *formal MILP encoding* with a `coop ∈ {0,1}` variable for boundary cases
  (`fracM = aicNum`). So "the system proposes it, the model prices it" is accurate; "the solver
  discovers it" is not.
- Slide says "a bytes ÷ bandwidth estimate rates these identical" — HQ is **not named on the
  slide**. Speaker notes carry the explicit comparison so it can be said aloud. **ASK USER what
  HQ actually uses** before naming them.

## npu-pipe-optimizer (slide 3)

Paper §10, three layers: L1 Python DSL (ops + pipe + read/write sets) → L2 analysis (graph from
RAW/WAR/WAW; cross-pipe ⇒ SetFlag/WaitFlag, same-pipe ⇒ PipeBarrier; back-edges for the loop;
tiling MILP Pyomo+HiGHS; scheduling CP-SAT OR-Tools; flag placement + event-ID allocation +
prime/drain per Commoner) → L3 annotated AscendC + timing + correctness proof.

Framing rules:
- Paper's verb is "**We propose**". It is a dependency the pipeline drives, NOT our system.
- **Layer 1 doesn't exist** — `dsl-to-graph-extractor.md` + `ascendc-cpp-to-yaml-parser.md`
  (bisheng→MLIR doesn't work). That absence is exactly why `ascend310p-yaml-author` exists:
  we compose the synthesizer's input per block from `yaml.tmpl`. **This is the real contribution
  on that slide** — without it the analyser can't run on real kernels.
- We don't trust PASS: `graph-synthesizer-pass-trust.md` needs P1 phase-completeness,
  P2 declared aliasing, P3 loop-body correctness; else downgrade to "PASS within modelled scope
  only". A PASS on an incomplete YAML = silent false negative.

## Open / to confirm with user

- Results slide never seen. Match **vocabulary** (operator name, speedup phrasing, phase words)
  — mismatched terminology between adjacent slides is what reads as a pasted-in AI insert.
- Which operator(s) actually ported by skills + the measured numbers → could harden slide 1.
- **What does HQ actually use for tiling/perf decisions?** Slide 2 currently contrasts against a
  generic bytes÷bandwidth estimate. Naming HQ's approach would sharpen the strongest slide.

## Compaction to 3 slides (2026-08-18) — user: "need it to be compact / maybe even 2"

Went 3 → 5 (adding cost model + analyser) → back to 3. Final arc, and why:

1. **What it does** — §9 baseline, 4 phases, gate band. The framing; can't cut.
2. **How it decides: the cost model** — the differentiator. Untouchable.
3. **The synchronisation half** — analyser + why it needs us + we don't trust PASS.

**Cut: "Why the output can be trusted" and "Each port makes the next cheaper."** Their essence
survives as one strip on slide 3 ("every port leaves the library bigger: 35 debug rules so far,
each written after a real bug, plus the blocks and calibration data"). Lost from the deck (still
in git history + speaker notes): the 99.33 % golden-variant example, the four-310P-variants
point, two-speeds-of-promotion, the 13-open-gaps box, the 64 % AscendKernelGen line.

**2-slide variant** drops slide 3 entirely and folds the analyser into slide 1's SAFETY gate
("races, deadlock and event balance are proved by a dependency analyser the pipeline drives —
and a PASS counts only as far as it modelled"). Tradeoff: npu-pipe-optimizer becomes a *mention*,
not a description — and the "we build its input / we don't trust PASS" contribution disappears
from the slides. Use 3-slide unless time is genuinely the binding constraint.

Both variants verified: 0 estimated overflow, nothing off-slide, exact 16:9, speaker notes intact.

## De-graphomania pass (2026-08-18) — user: "remove graphomania"

Prose → fragments. ~433 words total on-slide across 3 slides (~145/slide). Prose lives in the
speaker notes, which were NOT trimmed — they carry everything the slides no longer say.

Rules applied:
- Full sentences → fragments. `"classify the port: shared tiling, or fully separate"` →
  `"port mode"`. `"generate the tests too: CPU, simulator, and on-device when hardware is
  available"` → `"tests generated: CPU · sim · device"`.
- Gates became numbers on two short lines: `cosine ≥ 0.99 · 0.999 for norms and logits /
  never lowered to pass`; `≥ 5 % or reverted / no other shape worse than 2 %`.
- Middle-dot `·` as separator instead of "and"/commas — reads as spec, not paragraph.
- Titles shortened: "Porting an operator to 310P: what the system actually does" → "What the
  system does"; "The other half of the problem: the synchronisation bugs" → "The synchronisation
  half"; column 3 "And we don't take PASS at face value" → "PASS ≠ correct".
- **Font sizes UP as text came down** (10.5→11.5/12/13, key lines 14–17). Less text at bigger
  type — the point is legibility from the back, not fitting more in.
- Slide 3: three long column footers → one line each; two stacked bottom bars → one split bar.
  N_H 4.10 → 3.70.
- Slide 2 right panel now leads with **"Same bytes ≠ same time."** at 17pt — the one sentence
  that has to survive if they read nothing else.

Geometry after: 0 estimated overflow both variants, nothing off-slide, exact 16:9.
Gate band grew (G_H 1.56 → 1.74) because two-line gates at 11.5pt need the room.

## Results slide added (2026-08-18) — now 4 slides / 3 compact

Placed FIRST (opener). Arc: what shipped → how it works → cost model → sync half.
Rationale: management wants "did it work" before "how". Trivial to move to the end if preferred.

| operator | why | result |
|---|---|---|
| compute_wy | GDN architecture | +40 % e2e prefill |
| TopPTopK | sampling path | +3–4 % decode |
| CGDR_fwd_o · CGDR_fwd_h | GDN architecture | +35 % e2e prefill |
| RGDR | GDN architecture | +25 % on the operator |
| GroupMatMul | AI-PC | 70× on the operator |

Plus a 950PR band: DualMM (>30 % bandwidth-bound), Bessel_io (>95 % vec-core util),
Roll_complex64, Inverse_nd (>95 % vec-core util) — "four operators in five days, to prove a point".

**Spelling fixed: user wrote "RDGR", repo says RGDR** — `csrc/attention/recurrent_gated_delta_rule_v310/`,
`examples/python/profile_rgdr.py`, branch `rgdr_optim`. Recurrent Gated Delta Rule. Slide uses RGDR.
(CGDR = chunked variant, cf. `talk/chunk_gdr_*`. Consistent with user's naming.)

**e2e vs operator-local distinction is load-bearing** and is called out in the speaker notes:
compute_wy and CGDR move an end-to-end metric; RGDR's 25 % and GroupMatMul's 70× are
operator-local. Do not let the room hear the 70× as end-to-end.

MODE 2 now drops slide index 3 (the sync slide), keeping results + mechanism + cost model.
