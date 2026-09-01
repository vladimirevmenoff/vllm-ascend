#!/usr/bin/env python3
"""Guard the spec-decode pre-replay stream sync against ACL graph capture.

plog shows the first failure is `StreamSynchronize: Not allow to synchronize
captured-stream` (107027); the 107030 memcpy error is downstream of it. The call is
model_runner_310p.py's `update_before_replay` path, which is gated on
`forward_context.capturing`. The v2 aclgraph path sets the flag on `_EXTRA_CTX`
instead (worker/v2/aclgraph_utils.py:171), so that guard can pass while a capture
is active. Check both.
"""
import ast
import re
import shutil
import sys

P = "/vllm-workspace/vllm-ascend/vllm_ascend/_310p/model_runner_310p.py"
shutil.copy2(P, "/tmp/mr310.bak")
src = open(P).read()

anchor = """            and not forward_context.capturing
            and hasattr(self, "update_stream")"""
replacement = """            and not forward_context.capturing
            # CLAUDE-FIX: the capture flag is set on _EXTRA_CTX in the v2 aclgraph
            # path (worker/v2/aclgraph_utils.py:171). If only that one is set this
            # guard passes and we call stream.synchronize() inside an ACL capture ->
            # "Not allow to synchronize captured-stream" (plog 107027), which surfaces
            # later as the 107030 memcpy failure. Check both flags.
            and not _claude_extra_ctx_capturing()
            and hasattr(self, "update_stream")"""

if anchor not in src:
    sys.exit("anchor not found -- file already patched or changed")

src = src.replace(anchor, replacement, 1)

helper = '''

def _claude_extra_ctx_capturing() -> bool:
    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        return bool(getattr(_EXTRA_CTX, "capturing", False))
    except Exception:
        return False

'''

m = re.search(r"^(class |def )", src, re.M)
src = src[: m.start()] + helper.lstrip("\n") + "\n" + src[m.start() :]

ast.parse(src)
open(P, "w").write(src)
print("patched + syntax ok")
