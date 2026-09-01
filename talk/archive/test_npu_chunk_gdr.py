"""
Precision test: chunk_gated_delta_rule_fwd_h NPU kernel.
Loads minimal binding .so, calls via torch.ops._C_chunk_test.
"""
import sys, os, math
import torch
import torch_npu

torch.manual_seed(42)
torch_npu.npu.set_device(0)

torch.ops.load_library("/home/e00927329/chunk_gdr_test.so")
print(f"Op registered: {hasattr(torch.ops._C_chunk_test, 'chunk_gated_delta_rule_fwd_h')}", flush=True)

B, T, Hg, HV, K, V = 1, 128, 1, 1, 128, 128
CHUNK = 64
DTYPE = torch.float16


def cpu_reference(k, w, u, g):
    k, w, u, g = k.float(), w.float(), u.float(), g.float()
    if Hg != HV:
        r = HV // Hg
        k = k.repeat_interleave(r, dim=1)
        w = w.repeat_interleave(r, dim=1)

    NT = T // CHUNK
    h = torch.zeros(B, HV, K, V)
    h_chunks = []
    v_new = torch.zeros_like(u)

    for c in range(NT):
        t0 = c * CHUNK
        for i in range(CHUNK):
            gi = g[:, :, t0+i]
            h = h * gi.unsqueeze(-1).unsqueeze(-1).exp()
            wi = w[:, :, t0+i]
            ki = k[:, :, t0+i]
            ui = u[:, :, t0+i]
            bk = wi.exp() * ki
            corr = torch.einsum('bhk,bhkv->bhv', bk, h)
            vn = ui - corr
            v_new[:, :, t0+i] = vn
            h = h + torch.einsum('bhk,bhv->bhkv', bk, vn)
        h_chunks.append(h.clone())
    return h_chunks, v_new


def cosine_ok(a, b, thr=0.99):
    if a.norm() == 0 and b.norm() == 0:
        return 1.0, True
    if a.norm() == 0 or b.norm() == 0:
        return 0.0, False
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return cos, not math.isnan(cos) and cos > thr


def run():
    k = torch.randn(B, Hg, T, K, dtype=DTYPE, device="cpu") * 0.1
    w = -torch.rand(B, Hg, T, K, dtype=DTYPE, device="cpu") * 0.5
    u = torch.randn(B, HV, T, V, dtype=DTYPE, device="cpu") * 0.1
    g_fp32 = (-torch.rand(B, HV, T, device="cpu") * 0.1).float()

    print("CPU reference...", flush=True)
    h_ref, vn_ref = cpu_reference(k, w, u, g_fp32)
    print(f"  h_ref: min={h_ref[0].min():.4f} max={h_ref[0].max():.4f} nan={h_ref[0].isnan().sum()}")

    print("NPU kernel...", flush=True)
    k_npu = k.contiguous().npu()
    w_npu = w.contiguous().npu()
    u_npu = u.contiguous().npu()
    g_npu = g_fp32.contiguous().npu()

    h_out, v_new_out, final_state = torch.ops._C_chunk_test.chunk_gated_delta_rule_fwd_h(
        k_npu, w_npu, u_npu,
        g=g_npu,
        output_final_state=False,
        chunk_size=CHUNK,
        save_new_value=True,
    )
    torch.npu.synchronize()

    h_npu = h_out.cpu().float()
    vn_npu = v_new_out.cpu().float()

    NT = T // CHUNK
    print(f"\nB={B} T={T} K={K} V={V} chunk={CHUNK}")
    print(f"h_npu: shape={list(h_npu.shape)} min={h_npu.min():.4f} max={h_npu.max():.4f} "
          f"zero%={((h_npu==0).sum()*100/h_npu.numel()).item():.0f}")
    print(f"vn_npu: shape={list(vn_npu.shape)} min={vn_npu.min():.4f} max={vn_npu.max():.4f}")

    print(f"\nh_npu[0,0,0,0,:8]  = {h_npu[0,0,0,0,:8].tolist()}")
    print(f"h_ref[0][0,0,0,:8] = {h_ref[0][0,0,0,:8].tolist()}")
    print(f"vn_npu[0,0,0,:8]   = {vn_npu[0,0,0,:8].tolist()}")
    print(f"vn_ref[0,0,0,:8]   = {vn_ref[0,0,0,:8].tolist()}")

    ok = True
    print("\nh (state) per chunk:")
    for c in range(NT):
        ref = h_ref[c].flatten()
        npu = h_npu[0, :, c].flatten()
        cos, passed = cosine_ok(ref, npu)
        mae = (ref - npu).abs().mean().item()
        mx  = (ref - npu).abs().max().item()
        if not passed: ok = False
        print(f"  chunk {c}: cos={cos:.6f}  mae={mae:.6f}  max={mx:.6f}  [{'OK' if passed else 'FAIL'}]")

    print("\nv_new per-chunk:")
    for c in range(NT):
        t0, t1 = c*CHUNK, (c+1)*CHUNK
        rc = vn_ref[:,:,t0:t1].flatten()
        nc = vn_npu[:,:,t0:t1].flatten()
        cos, passed = cosine_ok(rc, nc)
        mae = (rc - nc).abs().mean().item()
        if not passed: ok = False
        print(f"  chunk {c}: cos={cos:.6f}  mae={mae:.6f}  [{'OK' if passed else 'FAIL'}]")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
