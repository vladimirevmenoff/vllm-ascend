"""ATK custom-API plugin for ChunkGatedDeltaRuleFwdH (310P).

NPU: dispatch to torch.ops._C_ascend.npu_chunk_gated_delta_rule_fwd_h_310
CPU: reference impl mirroring the kernel math (see cpu_reference in
     csrc/moe/chunk_gated_delta_rule_fwd_h/examples/python/test_aclnn_ctypes.py).
"""

from __future__ import annotations

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


def _cpu_reference(k, w, u, g, initial_state, output_final_state, chunk_size):
    """Layouts:
    k,w: [B, Hv, T, K]
    u:   [B, Hv, T, V]
    g:   [B, Hv, T]
    initial_state: [B, Hv, K, V]
    Outputs:
    h_out: [B, Hv, NT, K, V] (state BEFORE each chunk; NT = T // chunk_size)
    v_new_out: [B, Hv, T, V]
    final_state: [B, Hv, K, V] or None
    """
    k = k.float()
    w = w.float()
    u = u.float()
    g = g.float()

    B, Hv, T, K = k.shape
    V = u.shape[-1]
    NT = T // chunk_size
    assert T % chunk_size == 0, f"T={T} must be multiple of chunk_size={chunk_size}"

    h = (
        initial_state.float().clone()
        if initial_state is not None
        else torch.zeros(B, Hv, K, V, device=k.device)
    )

    h_out = torch.zeros(B, Hv, NT, K, V, device=k.device)
    v_new = torch.zeros_like(u)

    for c in range(NT):
        t0 = c * chunk_size
        t1 = t0 + chunk_size
        h_out[:, :, c] = h

        W = w[:, :, t0:t1, :]            # [B, Hv, c, K]
        U = u[:, :, t0:t1, :]            # [B, Hv, c, V]
        K_c = k[:, :, t0:t1, :]
        g_c = g[:, :, t0:t1]             # [B, Hv, c]

        ws = torch.einsum("bhik,bhkv->bhiv", W, h)
        vn = U - ws                       # [B, Hv, c, V]
        v_new[:, :, t0:t1, :] = vn

        # h_new = h * exp(g_last) + sum_i exp(g_last - g_i) * (k_i^T @ vn_i)
        scale = (g_c[:, :, -1:] - g_c).exp().unsqueeze(-1)  # [B, Hv, c, 1]
        v_update = scale * vn
        h_work = torch.einsum("bhik,bhiv->bhkv", K_c, v_update)
        h = h * g_c[:, :, -1:].unsqueeze(-1).exp() + h_work

    final_state = h if output_final_state else None
    return h_out.to(torch.float16), v_new.to(torch.float16), final_state


@register("cgdr_fwd_h_v310_custom")
class CgdrFwdHCustomApi(BaseApi):
    """Bridge: positional/kwarg in -> backend dispatch -> normalized outputs."""

    def _extract(self, args, kwargs):
        k = _arg(args, kwargs, 0, "k")
        w = _arg(args, kwargs, 1, "w")
        u = _arg(args, kwargs, 2, "u")
        g = _arg(args, kwargs, 3, "g")
        initial_state = _arg(args, kwargs, 4, "initial_state", None)
        output_final_state = bool(_arg(args, kwargs, 5, "output_final_state", False))
        chunk_size = int(_arg(args, kwargs, 6, "chunk_size", CHUNK_SIZE))
        return k, w, u, g, initial_state, output_final_state, chunk_size

    def run_npu(self, *args, **kwargs):
        k, w, u, g, init, out_fs, cs = self._extract(args, kwargs)
        # Ensure NPU
        k = k.npu()
        w = w.npu()
        u = u.npu()
        g = g.npu()
        if init is not None:
            init = init.npu()

        # Schema: chunk_gated_delta_rule_fwd_h(k, w, u, g?, *, gk=None,
        #   initial_state=None, output_final_state=False, chunk_size=None,
        #   save_new_value=True, cu_seqlens=None, chunk_indices=None,
        #   use_exp2=False, transpose_state_layout=False)
        out = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
            k, w, u, g,
            initial_state=init,
            output_final_state=out_fs,
            chunk_size=cs,
        )
        # binding returns (h_out, v_new_out, final_state_out)
        h_out, v_new, final = out if isinstance(out, (tuple, list)) else (out, None, None)
        torch.npu.synchronize()
        self.output = (
            h_out.cpu().float(),
            v_new.cpu().float(),
            final.cpu().float() if final is not None else None,
        )
        return self.output

    def run_cpu(self, *args, **kwargs):
        k, w, u, g, init, out_fs, cs = self._extract(args, kwargs)
        h_out, v_new, final = _cpu_reference(
            k.cpu(),
            w.cpu(),
            u.cpu(),
            g.cpu(),
            init.cpu() if init is not None else None,
            out_fs,
            cs,
        )
        self.output = (
            h_out.float(),
            v_new.float(),
            final.float() if final is not None else None,
        )
        return self.output
