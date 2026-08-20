# ascend310p-operator-porting

Agent-skills bundle that automates 910B → 310P operator porting
and existing-310P-operator improvement on the Ascend platform.

Consumed by an LLM agent harness (Claude Code, OpenClaw, OpenCode,
…). Given a working source operator + a `migration_config.yaml`,
the skills compose end-to-end to produce a correct, performance-
tuned 310P port — or improve an existing 310P operator measurably.

## Layout

```
ascend310p-operator-porting/
├── README.md                       (this file)
├── skills/                         SKILL.md folders, kebab-case
│   ├── ascend310p-port-orchestrator/      (A1 — entry point)
│   ├── ascend310p-port-analysis/          (B1)
│   ├── ascend310p-def-port/               (B2)
│   ├── ascend310p-config-port/            (B3)
│   ├── ascend310p-tiling-port/            (B4)
│   ├── ascend310p-kernel-port/            (B5)
│   ├── ascend310p-test-gen/               (B6)
│   ├── ascend310p-correctness-check/      (B7)
│   ├── ascend310p-performance-eval/       (B8)
│   ├── ascend310p-performance-tune/       (B9 — deprecated → B14)
│   ├── ascend310p-debug/                  (B10)
│   ├── ascend310p-knowledge-capture/               (B11)
│   ├── ascend310p-baseline-measure/       (B12)
│   ├── ascend310p-improvement-audit/      (B13)
│   ├── ascend310p-improvement-apply/      (B14)
│   ├── ascend310p-device-benchmark/       (B15)
│   ├── ascend310p-yaml-author/                     (S2)
│   ├── ascend310p-config-wizard/                   (S1)
│   ├── ascend310p-port-multipass-review/           (O2)
│   ├── ascend310p-math-contract-author/             (M1 — Mode-C / create-mode entry)
│   ├── ascend310p-algorithm-select/                 (A1 — algorithm + block-list choice)
│   └── ascend310p-reasoning/                        (C3 — structured ranking)
│
├── shared/                         cross-skill resources
│   ├── blocks/                     canonical recurring kernel fragments
│   │                               (k-loop-mmad, vdeq16-dequant, …)
│   ├── transferable-patterns/      910B↔310P porting patterns
│   ├── recipes/                    cross-cutting how-tos
│   ├── decision-rules/             cross-recipe choice rules
│   ├── cost-model/                 composite-lower-bound formula,
│   │                               pipe bandwidths, dma-affine cost
│   ├── api-deltas/                 310P-vs-910B API replacement table
│   ├── api-docs/                   per-API doc references
│   ├── analysis-tool-templates/    parameterised .py analysis tools
│   ├── benchmarks/                 measured hardware data
│   ├── hardware/                   hardware constants from CANN .ini
│   ├── learnings/                  jsonl audit log + proposals
│   ├── literature/                 framework paper excerpts
│   ├── porting-modes/              Mode A/B/C decision tree
│   ├── skill-rules/                runtime SP*+AL* rule docs
│   ├── iterators/                  iterator substrate templates
│   └── templates/                  code-emission .tmpl files
│
├── tools/                          deliverable-side .py utilities
│   ├── apply_recipe.py             B14 transform applier
│   ├── baseline_diff.py            B12 cost-model gap analyser
│   ├── detect_chip_support.py      port-mode vs improve-mode gate
│   ├── extract_blocks_from_kernel.py  block-recovery for improve mode
│   ├── parse_ini.py                CANN .ini → JSON parser
│   ├── promote_learnings.py        SP2-gated C2 promoter
│   ├── validate_config.py          migration_config.yaml validator
│   ├── invariants/                 16 invariant detectors + run_all.py
│   └── README.md                   tools/ inventory + asymmetry note
│
├── runtime-hooks/                  ship with the bundle; install per
│   │                               `runtime-hooks/INSTALL.md`
│   ├── scope_protect.sh            SP1+SP1.5+SP3 enforcement
│   ├── device_literal_grep.sh      SP4 enforcement
│   ├── learnings_promote_gate.sh   SP2 audit trail
│   ├── agent_opus_check.sh         opus model enforcement
│   ├── al_reminders.sh             AL1-AL6 reminders
│   └── closeout_check.sh           end-of-session B11 reminder
│
└── migration_config.yaml.example   end-user template
```

## How an end user runs the skills

1. Author `migration_config.yaml` from `migration_config.yaml.example`
   (or invoke skill **B0 ascend310p-config-wizard** which generates
   it interactively).
2. Install runtime hooks per `runtime-hooks/INSTALL.md`.
3. Invoke skill **A1 ascend310p-port-orchestrator** (Claude
   Code matches by description keywords). A1 dispatches to B1..B15
   per the operator's mode (port vs improve).
4. Authorise individual B14 findings as the improvement loop runs
   (`--user-authorized=<finding-id>` per SP2).

See `skills/ascend310p-port-orchestrator/SKILL.md` for the
full A1 workflow + the migration_config schema.

## Citation policy

This bundle ships to other users on different machines. Skill
content cites only:

- Public main projects `ops-nn` and `ops-transformer` on RELEASED
  branches/tags (relative `<project>/<path>:<line>` form, never
  absolute prefix).
- The CANN toolkit via `${TOOLKIT.CANN_PATH}/...` placeholder.
- The framework paper IF copied into `shared/literature/<file>.md`.

Citations to standalone local projects (QBMv3, qgmmd, GMM,
DynamicQuant as bare names) are FORBIDDEN — they're local-only forks
that other users don't have. Mine them during authoring; rephrase
findings as generic AscendC.

## Reference

- The agent-skills format conventions: `./agent-skills/skills/ascend310p-operator-porting/SKILL.md` (the seed; this bundle expands it). The `./agent-skills/` directory is a checked-in copy at the outer Agent project's root; the user's separate `agent-skills` checkout (if any) is reachable via the path in `talk/memory/reference_projects.md`.
- 19 skill IDs follow the `ascend310p-{operator-,}<role>` kebab-case
  convention.
