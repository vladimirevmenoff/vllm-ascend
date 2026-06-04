"""ATK custom-API plugin for ChunkFwdO (310P).

Math (matches `_torch_chunk_gated_delta_rule_chunked`):
  for each chunk i (size c):
    q_scaled = q_i * scale
    inter_state = (q_scaled * exp(g_i)) @ h_i        # [B,Hv,c,V]
    attn = q_scaled @ k_i^T * decay_mask              # strict lower triangular
    o_i = inter_state + attn @ v_i
"""

from __future__ import annotations

import math

import torch

from atk.runner.api.base_api import BaseApi
from atk.runner.api.registry import register

CHUNK_SIZE = 64


def _arg(args, kwargs, idx, name, default=None):
    if name in kwargs:
        return kwargs[name]
    if idx < len(args):
        return args[idx]
    return default


def _cpu_reference(q, k, v, h, scale, g, chunk_size):
    """Layouts:
    q,k: [B, Hv, T, K]; v: [B, Hv, T, V]
    h:   [B, Hv, NT, K, V]; g: [B, Hv, T] (cumsum per-chunk, <=0)
    Returns o: [B, Hv, T, V]
    """
    q = q.float() * scale
    k = k.float()
    v = v.float()
    h = h.float()
    g = g.float()

    B, Hv, T, K = q.shape
    V = v.shape[-1]
    NT = T // chunk_size
    assert T % chunk_size == 0
    assert h.shape == (B, Hv, NT, K, V)

    o = torch.zeros(B, Hv, T, V, device=q.device)
    # strict-lower mask (chunk-local causal, excludes diagonal? see ref:
    # mask_upper diag=1 zero'd, decay_mask is .tril().exp().tril() so diag kept).
    # We follow torch ref: decay_mask = tril(exp(g_i - g_j)); attn zeroed above diag.
    tri = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device))

    for i in range(NT):
        t0 = i * chunk_size
        t1 = t0 + chunk_size
        q_i = q[:, :, t0:t1, :]                  # [B,Hv,c,K]
        k_i = k[:, :, t0:t1, :]
        v_i = v[:, :, t0:t1, :]
        g_i = g[:, :, t0:t1]                     # [B,Hv,c]
        h_i = h[:, :, i, :, :]                   # [B,Hv,K,V]

        # decay_mask: exp(g_i[a] - g_i[b]) lower-triangular (rows a, cols b)
        decay = (g_i.unsqueeze(-1) - g_i.unsqueeze(-2)).exp() * tri  # [B,Hv,c,c]

        attn = (q_i @ k_i.transpose(-1, -2)) * decay   # [B,Hv,c,c]
        attn = attn * tri  # mask strict upper (already masked but safe)

        inter = (q_i * g_i.unsqueeze(-1).exp()) @ h_i   # [B,Hv,c,V]
        o[:, :, t0:t1, :] = inter + attn @ v_i

    return o.to(torch.float16)


@register("chunk_fwd_o_v310_custom")
class ChunkFwdOCustomApi(BaseApi):
    def _extract(self, args, kwargs):
        q = _arg(args, kwargs, 0, "q")
        k = _arg(args, kwargs, 1, "k")
        v = _arg(args, kwargs, 2, "v")
        h = _arg(args, kwargs, 3, "h")
        scale = float(_arg(args, kwargs, 4, "scale", 1.0))
        g = _arg(args, kwargs, 5, "g", None)
        chunk_size = int(_arg(args, kwargs, 6, "chunk_size", CHUNK_SIZE))
        return q, k, v, h, scale, g, chunk_size

    def run_npu(self, *args, **kwargs):
        q, k, v, h, scale, g, cs = self._extract(args, kwargs)
        q = q.npu()
        k = k.npu()
        v = v.npu()
        h = h.npu()
        g_npu = g.npu() if g is not None else None

        out = torch.ops._C_ascend.chunk_fwd_o(
            q, k, v, h, scale,
            g=g_npu,
            g_gamma=None,
            cu_seqlens=None,
            chunk_indices=None,
            chunk_size=cs,
            transpose_state_layout=False,
        )
        torch.npu.synchronize()
        self.output = out.cpu().float()
        return self.output

    def run_cpu(self, *args, **kwargs):
        q, k, v, h, scale, g, cs = self._extract(args, kwargs)
        out = _cpu_reference(q.cpu(), k.cpu(), v.cpu(), h.cpu(), scale, g.cpu(), cs)
        self.output = out.float()
        return self.output
