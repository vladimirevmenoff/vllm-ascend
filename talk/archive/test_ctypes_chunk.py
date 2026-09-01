"""
Pure-ctypes test for chunk_gated_delta_rule_fwd_h on 310P3.
Bypasses C++ binding entirely — calls aclnn API via ctypes.
T=128 → 2 chunks. With output_final_state=false, chunk 0 should write output.
"""
import sys, os, math, ctypes
import torch
import torch_npu

torch.manual_seed(42)
torch_npu.npu.set_device(0)

# ---- libs ----
CANN = "/usr/local/Ascend/cann-9.0.T3/aarch64-linux"
CUST = "/usr/local/Ascend/cann-9.0.T3/opp/vendors/custom_transformer/op_api/lib"

nnop = ctypes.CDLL(f"{CANN}/lib64/libnnopbase.so")
cust = ctypes.CDLL(f"{CUST}/libcust_opapi.so")
acl  = ctypes.CDLL(f"{CANN}/lib64/libascendcl.so")

create_tensor = nnop.aclCreateTensor
create_tensor.restype = ctypes.c_void_p

ws_fn = cust.aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize
ws_fn.restype = ctypes.c_int

exec_fn = cust.aclnnChunkGatedDeltaRuleFwdH
exec_fn.restype = ctypes.c_int

# ---- dims ----
B, T, Hg, HV, K, V = 1, 128, 1, 1, 128, 128
CHUNK = 64
NT = T // CHUNK

# ---- helpers ----
DTYPE_MAP = {torch.float16: 1, torch.float32: 0, torch.bfloat16: 27}

def make_acl(t):
    """Create aclTensor* from torch tensor via ctypes."""
    if t is None:
        return None
    nd = t.dim()
    shape = (ctypes.c_int64 * nd)(*t.shape)
    strides = (ctypes.c_int64 * nd)(*t.stride())
    stor_dim = (ctypes.c_int64 * 1)(t.storage().nbytes() // t.element_size())
    acl_dtype = DTYPE_MAP[t.dtype]
    ptr = ctypes.c_void_p(t.data_ptr())
    acl_t = create_tensor(shape, nd, acl_dtype, strides, 0, 2, stor_dim, 1, ptr)
    return ctypes.c_void_p(acl_t)


def cpu_reference(k, w, u, g):
    """Scalar loop reference in fp32."""
    k, w, u, g = k.float(), w.float(), u.float(), g.float()
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


def cosine(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    if a.norm() == 0 and b.norm() == 0: return 1.0
    if a.norm() == 0 or b.norm() == 0: return 0.0
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def run_kernel(k_npu, w_npu, u_npu, g_npu, output_final_state=False):
    """Call the aclnn kernel via ctypes. Returns (h, v_new) on CPU."""
    # output tensors
    h_out = torch.full((B, HV, NT, K, V), 42.0, dtype=torch.float16, device="npu:0")
    vn_out = torch.full((B, HV, T, V), 42.0, dtype=torch.float16, device="npu:0")
    if output_final_state:
        fs_out = torch.zeros(B, HV, K, V, dtype=torch.float16, device="npu:0")
    else:
        fs_out = torch.empty(1, dtype=torch.float16, device="npu:0")
    torch.npu.synchronize()

    # create aclTensors
    acl_k  = make_acl(k_npu)
    acl_w  = make_acl(w_npu)
    acl_u  = make_acl(u_npu)
    acl_g  = make_acl(g_npu)
    acl_h  = make_acl(h_out)
    acl_vn = make_acl(vn_out)
    acl_fs = make_acl(fs_out)

    print(f"  acl ptrs: k={acl_k} w={acl_w} u={acl_u} g={acl_g} h={acl_h} vn={acl_vn} fs={acl_fs}")

    # GetWorkspaceSize
    ws_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p(0)

    ret = ws_fn(
        acl_k, acl_w, acl_u,
        acl_g,
        None,  # gk
        None,  # initial_state
        ctypes.c_bool(output_final_state),
        ctypes.c_int64(CHUNK),
        ctypes.c_bool(True),   # save_new_value
        None,  # cu_seqlens
        None,  # chunk_indices
        ctypes.c_bool(False),  # use_exp2
        ctypes.c_bool(False),  # transpose_state_layout
        acl_h, acl_vn, acl_fs,
        ctypes.byref(ws_size),
        ctypes.byref(executor))
    print(f"  GetWS ret={ret} ws_size={ws_size.value} executor={hex(executor.value) if executor.value else 'NULL'}")

    if ret != 0:
        print(f"  ERROR: GetWorkspaceSize failed with {ret}")
        return None, None

    # workspace
    ws_ptr = None
    if ws_size.value > 0:
        ws_t = torch.empty(ws_size.value, dtype=torch.uint8, device="npu:0")
        ws_ptr = ctypes.c_void_p(ws_t.data_ptr())

    # get stream
    stream_ptr = torch_npu.npu.current_stream().npu_stream
    stream = ctypes.c_void_p(stream_ptr)

    # execute
    ret2 = exec_fn(ws_ptr, ws_size, executor, stream)
    print(f"  Exec ret={ret2}")
    torch.npu.synchronize()

    return h_out.cpu().float(), vn_out.cpu().float()


def main():
    # inputs
    k = torch.randn(B, Hg, T, K, dtype=torch.float16) * 0.1
    w = -torch.rand(B, Hg, T, K, dtype=torch.float16) * 0.5
    u = torch.randn(B, HV, T, V, dtype=torch.float16) * 0.1
    g_fp32 = (-torch.rand(B, HV, T) * 0.1).float()

    print("CPU reference...")
    h_ref, vn_ref = cpu_reference(k, w, u, g_fp32)

    # ---- Test 1: g as fp32, output_final_state=false ----
    print("\n=== Test 1: g=fp32, output_final_state=false ===")
    k_npu = k.contiguous().npu()
    w_npu = w.contiguous().npu()
    u_npu = u.contiguous().npu()
    g_npu_fp32 = g_fp32.contiguous().npu()

    h_npu, vn_npu = run_kernel(k_npu, w_npu, u_npu, g_npu_fp32, output_final_state=False)
    if h_npu is not None:
        report(h_npu, vn_npu, h_ref, vn_ref, "g=fp32,ofs=false")

    # ---- Test 2: g as fp16, output_final_state=false ----
    print("\n=== Test 2: g=fp16, output_final_state=false ===")
    g_npu_fp16 = g_fp32.half().contiguous().npu()
    h_npu2, vn_npu2 = run_kernel(k_npu, w_npu, u_npu, g_npu_fp16, output_final_state=False)
    if h_npu2 is not None:
        report(h_npu2, vn_npu2, h_ref, vn_ref, "g=fp16,ofs=false")

    # ---- Test 3: g=fp32, output_final_state=true ----
    print("\n=== Test 3: g=fp32, output_final_state=true ===")
    h_npu3, vn_npu3 = run_kernel(k_npu, w_npu, u_npu, g_npu_fp32, output_final_state=True)
    if h_npu3 is not None:
        report(h_npu3, vn_npu3, h_ref, vn_ref, "g=fp32,ofs=true")

    print("\nDone.")


def report(h_npu, vn_npu, h_ref, vn_ref, label):
    """Print comparison report."""
    print(f"\n  [{label}] h_npu shape={list(h_npu.shape)} range=[{h_npu.min():.4f}, {h_npu.max():.4f}]")
    sentinel_pct = ((h_npu == 42.0).sum() * 100 / h_npu.numel()).item()
    print(f"  sentinel(42.0) remaining: {sentinel_pct:.0f}%")

    print(f"  h_npu[0,0,0,0,:8] = {h_npu[0,0,0,0,:8].tolist()}")
    print(f"  h_ref[0][0,0,0,:8]= {h_ref[0][0,0,0,:8].tolist()}")

    for c in range(NT):
        ref = h_ref[c].flatten()
        npu = h_npu[0, :, c].flatten()
        cos = cosine(ref, npu)
        mae = (ref - npu).abs().mean().item()
        still42 = (npu == 42.0).sum().item()
        tag = "SENTINEL" if still42 > npu.numel() * 0.9 else ("OK" if cos > 0.99 else "FAIL")
        print(f"  h chunk {c}: cos={cos:.6f} mae={mae:.6f} 42s={still42}/{npu.numel()} [{tag}]")

    print(f"\n  vn_npu[0,0,0,:8] = {vn_npu[0,0,0,:8].tolist()}")
    print(f"  vn_ref[0,0,0,:8] = {vn_ref[0,0,0,:8].tolist()}")
    for c in range(NT):
        t0, t1 = c*CHUNK, (c+1)*CHUNK
        rc = vn_ref[:,:,t0:t1].flatten()
        nc = vn_npu[:,:,t0:t1].flatten()
        cos = cosine(rc, nc)
        still42 = (nc == 42.0).sum().item()
        tag = "SENTINEL" if still42 > nc.numel() * 0.9 else ("OK" if cos > 0.99 else "FAIL")
        print(f"  vn chunk {c}: cos={cos:.6f} 42s={still42}/{nc.numel()} [{tag}]")


if __name__ == "__main__":
    main()
