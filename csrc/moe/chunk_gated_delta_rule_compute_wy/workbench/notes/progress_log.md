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
