#!/usr/bin/env python3
"""Let the drafter be graph-captured under PIECEWISE, not only under FULL.

Today the drafter is wrapped only when cudagraph_mode.has_full_cudagraphs(), so with
PIECEWISE (the only mode MTP starts in on 310P) the drafter runs eager -- ~50 ms per
token, of which most is per-op launch overhead. FULL would capture it but trips a
synchronize-during-capture bug that bottoms out in torch_npu.

The target model already runs piecewise successfully, and piecewise leaves attention
split out, which is what avoids the sync. So capture the drafter piecewise instead.
"""
import ast

P = "/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py"
src = open(P).read()

old = """        if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs() and self.use_cuda_graph:
            logger.info(
                "[spec_decode/base] Wrapping draft model with ACLGraphWrapper:"
                " runtime_mode=FULL, use_eagle=%s, enable_enpu=%s",
                self.use_eagle,
                self.enable_enpu,
            )
            self.update_stream = torch.npu.Stream()
            self._runnable = ACLGraphWrapper(
                self._run_merged_draft,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                use_eagle=self.use_eagle,
                enable_enpu=self.enable_enpu,
            )"""

new = """        # CLAUDE-CHANGE: also capture the drafter under PIECEWISE. Previously this
        # only fired for has_full_cudagraphs(), so under PIECEWISE -- the only mode
        # MTP starts in on 310P -- the drafter ran eager (~50 ms/token, mostly launch
        # overhead). PIECEWISE keeps attention split out, which is what avoids the
        # synchronize-during-capture failure that FULL hits.
        _cg_mode = self.vllm_config.compilation_config.cudagraph_mode
        _draft_mode = None
        if self.use_cuda_graph:
            if _cg_mode.has_full_cudagraphs():
                _draft_mode = CUDAGraphMode.FULL
            elif _cg_mode == CUDAGraphMode.PIECEWISE:
                _draft_mode = CUDAGraphMode.PIECEWISE
        if _draft_mode is not None:
            logger.info(
                "[spec_decode/base] Wrapping draft model with ACLGraphWrapper:"
                " runtime_mode=%s, use_eagle=%s, enable_enpu=%s",
                _draft_mode,
                self.use_eagle,
                self.enable_enpu,
            )
            self.update_stream = torch.npu.Stream()
            self._runnable = ACLGraphWrapper(
                self._run_merged_draft,
                self.vllm_config,
                runtime_mode=_draft_mode,
                use_eagle=self.use_eagle,
                enable_enpu=self.enable_enpu,
            )"""

if old not in src:
    raise SystemExit("anchor not found -- already patched or file changed")

src = src.replace(old, new, 1)
ast.parse(src)
open(P, "w").write(src)
print("drafter now capturable under PIECEWISE")
