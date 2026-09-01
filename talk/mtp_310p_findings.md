# MTP on 310P for Qwen3.5-9B (2026-08-13)

**Verdict: MTP works on 310P and accepts ~90 % of drafts on real text.** An earlier reading of
1.7 % was a measurement artifact — see "The 1.7 % was my mistake". Two real blockers remain:
graph mode crashes, and the W8A8 checkpoint has no MTP weights.

## What is available

| piece | status |
|---|---|
| MTP weights in the checkpoint | **yes** — 15 `mtp.*` tensors in `/home/models/Qwen3.5-9B` (`mtp.fc.weight`, `mtp.layers.0.*`), matching `mtp_num_hidden_layers: 1` |
| vLLM model class | **yes** — `qwen3_5_mtp` in `MTPModelTypes`, own `Qwen3_5MTPModelArchConfigConvertor`, `Qwen3_5MTP` registered |
| vllm-ascend 310P spec-decode | **yes** — `_310p/spec_decode/llm_base_proposer_310.py` + MTP handling in `model_runner_310p.py` |
| runs on 310P | **yes, `--enforce-eager` only** |
| acceptance on real text | **86–98 %, α ≈ 0.9** |

```
--speculative-config {"method":"mtp","num_speculative_tokens":1} --enforce-eager
```
```
Loading draft model: method=mtp, model=/home/models/Qwen3.5-9B
Detected MTP model. Sharing target model embedding weights with the draft model.
```

## Measured acceptance (real text, greedy, 5 prompts × 128 tokens)

| prompt | acceptance |
|---|---|
| Roman Empire history | 87.5 % |
| gradient descent | 86.4 % |
| photosynthesis | 91.7 % |
| quicksort code | **98.3 %** |
| Treaty of Westphalia | 86.4 % |

Outputs are correct (`' the death of Julius Caesar in 44 BC…'`, working quicksort, `' state
sovereignty. This treaty is often cited as…'`). Code drafts best, which is the usual pattern.

## The 1.7 % was my mistake

The first measurement used `vllm bench serve --dataset-name random`, which generates
**uniformly random token IDs**. There is no linguistic structure to predict, so both models
flail and rarely agree. A DFX probe in the rejection sampler showed the draft predicting
almost exclusively low-ID tokens; decoded, they are `','` `'-'` `'/'` `'>'` `'_'` `' '`
`'ent'` `' of'` — punctuation and subword fragments, i.e. what any LM emits from meaningless
context. The target was equally degenerate (`'@a'` repeatedly).

**Never measure speculative acceptance on synthetic random-token prompts.** TTFT/TPOT are fine
on random data because they are bandwidth-bound; acceptance is a *model quality* metric and
needs real text.

## Revised projections, α = 0.9

Verification is free — measured: QBMM is flat ±3 % from M=1 to M=16 — so a verify pass costs
the same as a normal decode step. The cost is the sequential draft, dominated by the lm_head
read.

| config | draft | verify | E[tokens] | ms/token |
|---|---|---|---|---|
| fp16 + MTP k=1 | 14 | 101 | 1.90 | **60** (from 101.3, 1.67×) |
| W8A8 + MTP k=1 | 14 | 62.6 | 1.90 | **40** (from 62.6, 1.55×) |
| W8A8 + lm_head int8 + k=1 | 8 | 57 | 1.90 | **34** |
| W8A8 + lm_head int8 + k=2 | 16 | 57 | 2.71 | **27** ← under target |

So **30 ms/token looks reachable without W4A8**, via W8A8 + quantized lm_head + k=2 MTP.

## Measured end to end (fp16, eager, identical real prompts, 4 x 128 tokens)

| | ms/token |
|---|---|
| baseline, no MTP | 106.15 |
| MTP k=1 | **90.10** |
| **speedup** | **1.18x** |

Acceptance in this run: 80.0 / 83.3 / 91.4 / 94.6 %.

Far below the 1.67x projection, and the shortfall localises cleanly. At alpha ~ 0.87,
`(verify + draft) / 1.87 = 90.1` implies **draft ~ 62 ms**, against the ~14 ms the lm_head
read alone accounts for. That surplus is eager-mode overhead on the draft pass, paid once per
generated token.

**So graph mode is the unlock for MTP, not a nice-to-have.** Every projection in this document
assumes a cheap draft; eager makes it ~4x too expensive and eats most of the win.

## Blocker 1 — graph mode: two distinct bugs, both root-caused, neither fixed

### FULL and FULL_DECODE_ONLY: sync memcpy inside capture

The drafter is graph-wrapped iff `cudagraph_mode.has_full_cudagraphs()`
(`vllm_ascend/spec_decode/llm_base_proposer.py:495`), which is true for **both** FULL and
FULL_DECODE_ONLY -- hence both crash identically. Wrapped `runtime_mode=FULL`, the whole
drafter forward including attention is inside one capture, and 310P attention does host
round-trips (`vllm_ascend/_310p/attention/attention_v1.py`):
```python
222:  real_tokens = int(attn_metadata.seq_lens.sum().item())   # D2H
262:  qsl_cpu = attn_metadata.query_start_loc.cpu()            # D2H
```
which the runtime forbids during capture:
```
rtMemcpy execution failed, reason=the current capture mode does not support this operation
synchronized memcpy failed, kind = 2, runtime result = 107030
```
The main model escapes this only because `vllm::unified_attention_with_output` is in
`splitting_ops`, so attention runs outside the captured regions.

### PIECEWISE: dummy_run mode assertion

With `--compilation-config {"cudagraph_mode":"PIECEWISE"}` the drafter is correctly left
unwrapped (different error, confirming the analysis), but startup then fails at
`vllm_ascend/worker/model_runner_v1.py:3186`:
```
Cudagraph runtime mode mismatch in dummy_run. Expected NONE, but got PIECEWISE.
```
The spec-decode path passes an explicit `cudagraph_runtime_mode=NONE` while the dispatcher
computes PIECEWISE for that batch descriptor.

### Fix directions

- Cheapest: make the drafter's `dummy_run` not force NONE under PIECEWISE (pass `None` and let
  it adopt the dispatcher mode), then verify a piecewise-captured target + eager drafter runs.
- Proper: remove the two D2H calls from the 310P attention forward so a FULL capture is legal;
  both look precomputable outside the forward.

Note `ASCEND_LAUNCH_BLOCKING=1` cannot be used to localise these -- vllm-ascend rejects it
whenever ACL graph is enabled. Use plog (`$HOME/ascend/log/debug`).

## Blocker 2 — W8A8 checkpoint without an MTP head: FIXED

`/home/models/Qwen3.5-9B-w8a8-mtp` (12 GB). Built by
`/home/claude_bench/add_mtp_to_w8a8.py`:
- hardlinks `quant_model_weight_w8a8.safetensors` from the existing W8A8 build (no extra disk)
- copies the 15 `mtp.*` tensors from the base checkpoint into `mtp_head_fp16.safetensors`
  (0.49 GB, fp16)
- adds 15 `FLOAT` entries to `quant_model_description.json` — the same treatment the vision
  tower already gets

Keeping the draft head in fp16 is deliberate: it is 0.49 GB, and draft cost is dominated by
the lm_head read, not by the MTP layer.

vLLM picks up both shards and the drafter loads from it:
```
Loading draft model: method=mtp, model=/home/models/Qwen3.5-9B-w8a8-mtp
Checkpoint size: 11.56 GiB ... Loading safetensors checkpoint shards: 0/2
```
**Not yet validated end to end** — the validation run stalled during weight loading with the
box at load average 24 (the drafter streams the whole checkpoint a second time, so startup is
roughly doubled). Rerun with
`/home/claude_bench/mtp_w8a8_check.sh` when the box is quiet.

## Blocker 3 — lm_head quantization still blocked

`AscendParallelLMHead310` accepts only `lm_head.weight`; needed for the 27 ms row.

## Upstream bug worth reporting regardless

`vllm/config/speculative.py`, the `qwen3_5` branch:
```python
n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)   # -> None
```
This checkpoint nests `mtp_num_hidden_layers` inside `text_config` (top level: ABSENT,
text_config: 1), so `n_predict` is `None`. The `intern_s2_preview` branch a few lines below
reads it from `text_config` correctly. Benign here — `Qwen3_5MTP` re-reads it from
`hf_text_config` with a default of 1 — but it is wrong and will bite.

## Caveats on α

- Measured greedy (`temperature 0`). Sampling lowers acceptance; re-measure at your serving
  temperature/top-p.
- 5 prompts, English + code, 128 tokens each. Not a traffic-representative sample.
- k=1 only. The k=2 row extrapolates as α + α², which assumes per-position acceptance is
  independent — optimistic.

## Reproduce

`ssh 310p`, container `claude_pr11941`:
```
/home/claude_bench/mtp_real.sh          # real prompts, prints acceptance per request
```
The DFX probe (draft vs target token IDs) is patched into
`vllm_ascend/sample/rejection_sampler.py`, gated by `CLAUDE_SPEC_DFX=1`; backup at
`/tmp/rs.bak`. **Revert it before any timing run.**

---

# MTP in graph mode — SOLVED via PIECEWISE (2026-08-14)

**MTP now runs with graph mode enabled, no `--enforce-eager`. Measured 1.27x.**

| config (real prompts, fp16, BS=1) | ms/token |
|---|---|
| eager, no MTP | 106.15 |
| eager, MTP k=1 | 90.10 (1.18x) |
| **PIECEWISE graph, no MTP** | **102.36** |
| **PIECEWISE graph, MTP k=1** | **80.70 (1.27x)** |

Acceptance 81.8–95.5 %.

## The one-line fix

`vllm_ascend/worker/model_runner_v1.py`, in `_dummy_run` (~line 3186). The assert compared
the dispatcher's mode against the caller's:
```
Cudagraph runtime mode mismatch in dummy_run. Expected NONE, but got PIECEWISE.
```
Message format is `Expected {dispatcher}, but got {caller}` — so the **dispatcher** returns
NONE while `capture_model` deliberately passes PIECEWISE. With spec decode the batch
descriptor carries draft-token slots, so the padded size is not in the dispatcher's capture
set and it declines. The caller is the authority during capture, so the assert becomes a
warning and the caller's mode is used. Backup: `/tmp/mrv1.bak` in container `claude_pr11941`.

Serve with:
```
--speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
--compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```

## Why not FULL — and why it doesn't matter much

FULL / FULL_DECODE_ONLY additionally capture the **drafter** (wrapped iff
`cudagraph_mode.has_full_cudagraphs()`, `spec_decode/llm_base_proposer.py:495`). Both still
fail with:
```
rtMemcpy execution failed, reason=the current capture mode does not support this operation
synchronized memcpy failed, kind = 2, runtime result = 107030
```
i.e. a synchronous D2H inside the capture. I removed one such site —
`_310p/attention/attention_v1.py:222` `int(seq_lens.sum().item())` now reads the host-side
`num_actual_tokens` instead (backup `/tmp/attn310.bak`) — but FULL still fails, because the
310P ops are full of host round-trips: `_310p/ops/causal_conv1d.py` alone has 8+ `.item()`
calls in its forward, and under FULL the target's GDN layers are captured whole too.

**Making FULL work is a refactor across the 310P ops, not a patch.** The payoff is bounded:
PIECEWISE already captures the 32-layer target; FULL would additionally capture the 1-layer
drafter. Implied draft cost is still ~50 ms/token (from `80.70 x 1.88 - 102.36`), so there is
real upside — but it needs the D2H calls hoisted out of every op on the decode path.

## Remaining upside if someone does that work

At the projected cheap draft (~14 ms) and alpha ~ 0.9, k=1 gives `(102 + 14) / 1.9 = 61
ms/token` (1.67x) rather than the 80.70 measured now. That is the prize for the refactor.

---

# MTP + W8A8 — blocked: drafter weight load stalls (2026-08-14)

**Does not work today.** The target model loads the W8A8 checkpoint fine; the *drafter* then
stalls loading the same file, spinning at 101 % CPU for 20+ minutes before I killed it.
Reproduced twice, the second time on a quiet box (uptime 1d16h, load 1.9), so it is not
contention.

| arm (real prompts, PIECEWISE graph, `/home/models/Qwen3.5-9B-w8a8-mtp`) | result |
|---|---|
| baseline, no MTP | **63.80 ms/token** (consistent with the 62.42 TPOT measured on random prompts) |
| MTP k=1 | **never started** — drafter stuck in weight load |

## Evidence

Same run, same checkpoint, consecutive loads:
```
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:10<00:00, 5.01s/it]   <- target, 10 s
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]          <- drafter, 20+ min
```
`py-spy`, sampled three times 20 s apart, identical frame each time — stuck inside one copy,
not iterating:
```
weight_loader (vllm/model_executor/layers/linear.py:541)     # param_data.copy_(loaded_weight)
weight_loader (vllm_ascend/ops/linear.py:498)
_load_param  (vllm/model_executor/models/utils.py:278)
load_weights (vllm/model_executor/models/qwen3_5_mtp.py:178)
_get_model   (vllm_ascend/spec_decode/llm_base_proposer.py:279)
```

## Likely mechanism (not proven)

With `--quantization ascend` the draft model's linear params are constructed NZ-formatted;
`param_data.copy_(loaded_weight)` from an ND source into an NZ destination appears to fall
into a pathological conversion path. The target model does not hit this because its quantized
weights are transformed in `process_weights_after_loading` rather than copied into
pre-formatted params.

## Where that leaves the plan

`W8A8 + MTP` was the row that reached ~40 ms/token and, with lm_head quantization, ~27 ms.
Both are now gated on this loader bug in addition to the lm_head dispatch bug. Measured today:

| config | ms/token | status |
|---|---|---|
| W8A8, no MTP | 63.80 | works |
| fp16 + MTP k=1, graph | 80.70 | works (1.27x over fp16) |
| W8A8 + MTP k=1 | — | **blocked here** |

## Next probes, cheapest first

1. Check whether the draft model can be built with `quant_config=None` — the MTP head is
   fp16 in this checkpoint anyway, so quantizing the drafter buys nothing. If
   `--speculative-config` accepts a per-draft quantization override, that sidesteps the bug
   entirely.
2. Time a single `param_data.copy_` into an NZ-formatted destination in isolation to confirm
   the mechanism.
3. If confirmed, the fix belongs next to `vllm_ascend/ops/linear.py:498`: convert once after
   load rather than copying into a pre-formatted param.

---

# W8A8 + MTP — resolved: works, but buys nothing without drafter capture (2026-08-14)

## The load hang is draft-side, and there is a workaround

`--speculative-config` accepts its own `model`. Pointing the drafter at the **fp16**
checkpoint while the target stays W8A8 loads fine:
```
--quantization ascend                                   # target: /home/models/Qwen3.5-9B-w8a8-mtp
--speculative-config '{"method":"mtp","num_speculative_tokens":1,
                       "model":"/home/models/Qwen3.5-9B"}'
```
So the hang is **loading a drafter from a quantized checkpoint**, not the quantized target
poisoning device state. Script: `/home/claude_bench/mtp_w8a8_mixed.sh`.

## But MTP does not pay at W8A8

| config (real prompts, PIECEWISE graph, BS=1) | ms/token |
|---|---|
| W8A8, no MTP | 63.80 |
| W8A8 + MTP k=1 (drafter from fp16 ckpt) | **63.62** |
| fp16, no MTP | 102.36 |
| fp16 + MTP k=1 | 80.70 (1.27x) |

Acceptance 72–95 %, so speculation works — it just yields nothing. Implied draft cost
`63.62 x 1.84 - 63.80 = 53 ms`, matching the ~50 ms eager draft measured in the fp16 run.

**The reason is structural.** In PIECEWISE the drafter is deliberately *not* captured (that is
what makes graph mode start at all). An uncaptured draft pass costs ~50 ms regardless of
quantization. MTP only pays when verify >> draft:

| | verify | draft | net |
|---|---|---|---|
| fp16 | 102 | ~50 | 1.27x |
| W8A8 | 64 | ~50 | 1.00x |

Quantization makes the target cheaper, which makes the *un-quantizable, uncaptured* draft
proportionally more expensive. The two optimisations fight each other.

## What would make W8A8 + MTP worth having

The drafter must be captured, i.e. FULL graph mode, i.e. the D2H sweep across the 310P ops
(`_310p/ops/causal_conv1d.py` has 8+ `.item()` calls; `_310p/attention/attention_v1.py:222`
already fixed). With a captured draft near its ~14 ms floor:
`(63.8 + 14) / 1.9 = 41 ms/token` — the number the original projection assumed.

Until then: **use MTP at fp16 (1.27x), or use W8A8 without MTP (63.80). Combining them is a
wash.**

## Elimination log for the drafter load hang

Hangs on the drafter's first param (`fc.weight`), operands normal
(`npu fp16 (4096,8192) <- cpu fp16`, contiguous), 101 % CPU, entry logged and exit never
reached. Ruled out by direct measurement:

| hypothesis | test | result |
|---|---|---|
| NZ-formatted destination | copy into NZ param | 6.4 ms vs 5.6 ND |
| int8 <- fp16 cast | copy into int8 param | 63 ms |
| memory pressure | rerun at util 0.5 | identical hang |
| bad tensor | safetensors metadata | `[4096,8192] F16`, correct |
| NZ casts poisoning state | 240 casts then H2D copy | 5.8 ms |

Remaining suspect: the draft-side quant config resolving names against the target's
description entries (draft modules are `mtp.*`/`model.*` after remap). plog
(`$HOME/ascend/log/debug`) is the next tool. Instrumentation left in
`vllm_ascend/ops/linear.py` behind `CLAUDE_WL_DBG`; backup `/tmp/linear.bak`.

---

# FULL capture: root cause reframed, 3 sites fixed, 1 left inside torch_npu (2026-08-14)

## The error is a stream synchronize, not a memcpy

plog (`$HOME/ascend/log/debug/plog/`) shows the *first* failure, which the Python-level
error hides:
```
StreamSynchronize: Not allow to synchronize captured-stream, stream_id=12
rtStreamSynchronize: ErrCode=107027, desc=[stream is captured]
...then, downstream:
MemCopySync: Memory copy sync failed, cnt=36, kind=2
rtMemcpy: ErrCode=107030 [the current capture mode does not support this operation]
```
So the hunt is for **synchronize calls during capture**, not for D2H copies. A D2H tripwire on
`.item()/.cpu()/.tolist()/.numpy()/int()/float()/bool()` fired **zero** times during capture,
which is why the first search came up empty.

## Tooling built (reusable)

`vllm_ascend/utils.py` now carries two debug tripwires, both env-gated and inert otherwise:
- `CLAUDE_D2H_DBG=1` — logs D2H triggers during capture
- `CLAUDE_SYNC_DBG=1` — logs **every** stream/event/device synchronize with a 2-frame stack

The workflow is mechanical: run with `CLAUDE_SYNC_DBG=1`, read the **last** `[SYNC]` line
before the crash, guard that site, repeat. Backup of the original: `/tmp/utils.bak`.

## Sites found and guarded (all real, none sufficient alone)

| # | site | problem | fix |
|---|---|---|---|
| 1 | `_310p/model_runner_310p.py` `update_before_replay` | gated on `forward_context.capturing`, which is not set for the drafter's capture | also check `_EXTRA_CTX.capturing` |
| 2 | same | both context flags unreliable | gate on `torch.npu.is_current_stream_capturing()` — runtime truth |
| 3 | `compilation/acl_graph.py:278` replay-ordering sync | exemption is `_EXTRA_CTX.is_draft_model and use_eagle`, which MTP does not satisfy; the drafter replays from inside the target's capture | add `not is_current_stream_capturing()` |

Backups: `/tmp/mr310.bak`, `/tmp/aclgraph.bak`.

## What remains

After all three, the tripwire's last entry before the crash is a **device-wide
`torch.npu.synchronize()`** reached via `model_runner_310p.py:679 run_model()` with **no
vllm-ascend frame owning it** — the only synchronize in `compilation/` is the one already
guarded. It is inside torch_npu's own graph machinery.

**That puts the remaining fix at vendor level (torch_npu / CANN), not in vllm-ascend.**
Escalate with the plog excerpt above; `torch.npu.is_current_stream_capturing()` exists, so the
same guard is expressible wherever that sync lives.

## Practical position, unchanged

| config | ms/token | ship? |
|---|---|---|
| W8A8, no MTP | **63.80** | **yes — best today** |
| fp16 + MTP k=1, PIECEWISE | 80.70 | no (worse than W8A8) |
| W8A8 + MTP k=1, PIECEWISE | 63.62 | no (wash) |
| W8A8 + MTP, FULL capture | ~42 (projected) | blocked on the above |

MTP stays parked until FULL capture works. Everything needed to resume — tripwires, the three
guards, the plog procedure — is in place.

---

# Can the draft be made cheaper? Answered: not by capturing it (2026-08-17)

The draft costs ~50 ms/token, ~35 ms of which is per-op launch overhead from running eager.
The obvious fix is to graph-capture the drafter. **It cannot be captured on 310P today, in any
mode.**

## What was tried

`spec_decode/llm_base_proposer.py:495` wraps the drafter only when
`cudagraph_mode.has_full_cudagraphs()`. Patched it to also wrap under PIECEWISE, reasoning
that PIECEWISE keeps attention split out and so avoids the FULL sync failure. The wrap took
effect (`Wrapping draft model with ACLGraphWrapper: runtime_mode=PIECEWISE`) and startup died
with the same error as FULL:
```
wait for compute device to finish failed, runtime result = 107027
```
107027 = synchronize on a captured stream — the same class of failure, now from a *device*
synchronize.

**Conclusion: the capture mode is irrelevant.** Capturing the drafter at all puts its graph
work inside the target's capture/replay flow, where a synchronize fires. Three vllm-ascend
sync sites were already guarded (see previous section); the one that remains is inside
torch_npu's own graph machinery. Patch reverted -- leaving it in place breaks the otherwise
working PIECEWISE config.

## So the draft's ~35 ms of launch overhead is not removable at the vllm-ascend level

That caps MTP where the measurements already put it:

| config | ms/token | vs no-MTP |
|---|---|---|
| fp16 + MTP k=1 | 80.70 | 1.27x |
| W8A8 + MTP k=1 | 63.62 | 1.00x |

## Levers that remain, none needing capture

1. **Quantize lm_head** — takes ~6 ms off the draft *and* ~6 ms off verify. Blocked only by the
   `AscendParallelLMHead310` dispatch, a contained PR. Moves W8A8+MTP from break-even to
   roughly `(58 + 44) / 1.85 = 55` ms, i.e. ~1.05x. Modest.
2. **Batching — free today.** The draft pass is weight-bound, so one batched draft serves every
   sequence in flight. At BS=8: `(50 + 83) / 1.85 = 72` vs 83 baseline, ~1.15x. This is the one
   case where MTP pays without any code change, and it was never measured -- worth doing.
3. **Vendor fix for the capture sync** — unlocks the ~35 ms and with it the ~42 ms/token
   projection. Everything else is small by comparison.

## Recommendation unchanged

Ship **W8A8 without MTP (63.80 ms/token)**. MTP is worth revisiting when either the capture
sync is fixed upstream, or you are serving at batch >= 4 where the draft amortises.

---

# Batched MTP measured: 1.06x at BS=8 (2026-08-17)

W8A8, PIECEWISE graph, 8 concurrent real prompts, drafter from the fp16 checkpoint.
Script: `/home/claude_bench/mtp_batch_ab.sh` (thread-pool client; `vllm bench serve` is
unusable here because its random-token data destroys acceptance).

| | per-request ms/token | aggregate tok/s |
|---|---|---|
| W8A8, no MTP | 74.25 | 105.89 |
| W8A8 + MTP k=1 | **69.86** | **109.34** |
| ratio | **1.06x** | 1.03x |

Acceptance 82.7–83.0 %.

## The cost model is now validated

Solving for the draft from the measurement: `1.83 x 69.86 - 74.25 = 54 ms`, matching the
~50 ms inferred at BS=1. Feeding that back: `(54 + 74.25) / 1.83 = 70.1` vs **69.86 measured**
— 0.3 % error. So `ms/token = (draft + verify) / (1 + alpha)` predicts this workload reliably,
and the break-even rule `draft < alpha x verify` holds.

At BS=8 the budget is `0.83 x 74.25 = 62 ms` against a 54 ms draft: it clears, barely. That is
the entire 6 %.

## Full measured picture

| config | BS=1 | BS=8 |
|---|---|---|
| W8A8, no MTP | 63.80 | 74.25 |
| W8A8 + MTP k=1 | 63.62 (1.00x) | 69.86 (1.06x) |
| fp16, no MTP | 102.36 | — |
| fp16 + MTP k=1 | 80.70 (1.27x) | — |

## Verdict

Batching does not rescue MTP either. 6 % is not worth two loaded models, the extra memory, and
the fp16-drafter workaround.

Because the cost model is validated, the projection for a **captured** draft (~14 ms) can be
trusted: `(14 + 74.25) / 1.83 = 48 ms` at BS=8, i.e. **1.55x**. That is what the vendor-level
capture-sync fix is worth, and it remains the only change that makes MTP worth shipping.
