#!/usr/bin/env python3
"""Base the pre-replay sync guard on runtime capture state, not a context flag.

plog: the first failure is `StreamSynchronize: Not allow to synchronize
captured-stream` (107027) from model_runner_310p `_model_forward`'s
`update_before_replay` block; the 107030 memcpy error is downstream.

That block is gated on `forward_context.capturing`, but a sync tripwire shows the
sync still firing during "Capturing CUDA graphs (decode, FULL)" — so neither
`forward_context.capturing` nor `_EXTRA_CTX.capturing` is set for the *drafter's*
capture. torch.npu.is_current_stream_capturing() is the runtime's own answer and
cannot go stale.
"""
import ast

P = "/vllm-workspace/vllm-ascend/vllm_ascend/_310p/model_runner_310p.py"
src = open(P).read()

old = '''def _claude_extra_ctx_capturing() -> bool:
    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        return bool(getattr(_EXTRA_CTX, "capturing", False))
    except Exception:
        return False'''

new = '''def _claude_extra_ctx_capturing() -> bool:
    # Runtime truth first: the context flags are not set during the drafter's
    # capture, so they cannot be relied on here.
    try:
        if torch.npu.is_current_stream_capturing():
            return True
    except Exception:
        pass
    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        return bool(getattr(_EXTRA_CTX, "capturing", False))
    except Exception:
        return False'''

if old not in src:
    raise SystemExit("helper not found -- run patch_sync_guard.py first")

src = src.replace(old, new, 1)
ast.parse(src)
open(P, "w").write(src)
print("guard now uses is_current_stream_capturing()")
