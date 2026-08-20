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
