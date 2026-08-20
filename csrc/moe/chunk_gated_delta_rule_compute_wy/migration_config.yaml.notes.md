# migration_config.yaml — audit trail (wizard, 2026-08-20)
- User-provided: op path/name, improve intent, npu_pipe_optimizer at ./npu-pipe-optimizer,
  device (310p-a container claude_13122, dev 6, CANN 9.1.0-beta.1), build command incl.
  mandatory `rm -rf build build_out` (cmake cache serves stale kernels — talk file).
- Detected: ascend310p support YES (tools/detect_chip_support.py, evidence=3;
  AddConfig("ascend310p") at op_host/chunk_gated_delta_rule_compute_wy_def.cpp:78)
  → improve mode [2]: in_place=true, improve_existing=true.
- Toolkit: remote-only; local mirror .claude/cann-mirror/ holds the REAL Ascend310P3.ini
  (26336 B, fetched from container). set_env.sh/msprof are stubs — all execution goes
  through `ssh 310p-a docker exec claude_13122`.
- Defaults: recipe_min_speedup 0.05, determinism_runs 3, stop_condition tier2_device.
- Correctness gate for improve loop: /work/test_bisect.py (TK/TV) cos≥0.999 on w/u/g at
  K128/K64/K128V64; baseline TTFT 1466.7ms; known perf targets: single-chain doubling for
  both RHS (avoid ×2 cube work), replace per-row scalar-issued casts, MTE2/V overlap.
- Note: user said "mode B" — interpreted as improve flow with npu_pipe_optimizer-backed
  correctness (Mode A checks); port-analysis Mode B (independent tiling) n/a for in-place improve.
