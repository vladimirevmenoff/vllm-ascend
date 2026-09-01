#!/usr/bin/env python3
"""Never synchronize a capturing stream in the aclgraph replay path.

Root cause of MTP + FULL graph mode failing on 310P. plog's first error is
`StreamSynchronize: Not allow to synchronize captured-stream` (107027); the
107030 memcpy error is downstream of it.

acl_graph.ACLGraphWrapper's replay path syncs before replaying a FULL graph, with
an exemption only for `_EXTRA_CTX.is_draft_model and self.use_eagle`. On the MTP
path `is_draft_model` is not set, so the exemption misses. The drafter's dummy_run
is invoked from inside the target's capture region, so that replay-ordering sync
lands on a capturing stream.

Synchronising a capturing stream is never valid, whatever the mode, so gate on the
runtime's own answer.
"""
import ast

P = "/vllm-workspace/vllm-ascend/vllm_ascend/compilation/acl_graph.py"
src = open(P).read()

old = """        if not self.enable_enpu and need_sync:
            torch.npu.current_stream().synchronize()"""

new = """        # CLAUDE-FIX: never synchronize a stream that is being captured -- ACL
        # rejects it ("Not allow to synchronize captured-stream", plog 107027) and
        # it surfaces later as a 107030 memcpy failure. This replay-ordering
        # barrier is only meaningful outside capture anyway. The existing
        # `is_draft_eagle` exemption does not cover MTP, whose drafter replays
        # from inside the target's capture region.
        if not self.enable_enpu and need_sync and not _claude_stream_capturing():
            torch.npu.current_stream().synchronize()"""

if old not in src:
    raise SystemExit("anchor not found -- already patched or file changed")

src = src.replace(old, new, 1)

helper = '''

def _claude_stream_capturing() -> bool:
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:
        return False

'''

import re

m = re.search(r"^(class |def )", src, re.M)
src = src[: m.start()] + helper.lstrip("\n") + "\n" + src[m.start() :]

ast.parse(src)
open(P, "w").write(src)
print("replay-sync guarded on is_current_stream_capturing()")
