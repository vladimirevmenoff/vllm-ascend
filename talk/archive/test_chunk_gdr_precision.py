"""
Precision test for chunk_gated_delta_rule on Ascend 310P.
Runs with fp16 tensors ON NPU vs fp32 CPU reference.
Targets: L2-norm fp16 truncation, state drift, gate edge cases,
         NPU kernel precision differences.
"""
import sys
import time
import torch
import torch.nn.functional as F

try:
    import torch_npu
    DEVICE = "npu:0"
except ImportError:
    DEVICE = "cpu"

from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_pytorch

CHUNK_SIZE = 64


def metrics(test, ref):
    t = test.cpu().to(torch.float64).flatten()
    r = ref.cpu().to(torch.float64).flatten()
    cos = F.cosine_similarity(t.unsqueeze(0), r.unsqueeze(0)).item()
    mae = (t - r).abs().max().item()
    re = ((t - r).abs() / r.abs().clamp(min=1e-12)).max().item()
    return cos, mae, re


def chunk_cosines(test, ref):
    B, T, H, V = test.shape
    out = []
    for c in range((T + CHUNK_SIZE - 1) // CHUNK_SIZE):
        s, e = c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, T)
        a = test[:, s:e].cpu().to(torch.float64).flatten()
        b = ref[:, s:e].cpu().to(torch.float64).flatten()
        out.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    return out


def make(B, T, Hqk, Hv, K, V, seed=42, g_scale=0.2):
    torch.manual_seed(seed)
    q = torch.randn(B, T, Hqk, K, dtype=torch.float16)
    k = torch.randn(B, T, Hqk, K, dtype=torch.float16)
    v = torch.randn(B, T, Hv, V, dtype=torch.float16)
    g = -g_scale * torch.rand(B, T, Hv, dtype=torch.float32)
    beta = (0.15 + 0.35 * torch.rand(B, T, Hv, dtype=torch.float32)).to(torch.float16)
    init = torch.randn(B, Hv, V, K, dtype=torch.float16)
    return q, k, v, g, beta, init


def run(label, B, T, Hqk, Hv, K, V, g_scale=0.2, l2=True, seed=42):
    nc = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"  B={B} T={T} Hqk={Hqk} Hv={Hv} K={K} V={V} fp16 g={g_scale} l2={l2} chunks={nc}")

    q, k, v, g, beta, init = make(B, T, Hqk, Hv, K, V, seed, g_scale)

    # fp32 CPU reference
    t0 = time.time()
    ref_o, ref_s = chunk_gated_delta_rule_pytorch(
        q=q.float(), k=k.float(), v=v.float(), g=g.float(), beta=beta.float(),
        initial_state=init.float(), output_final_state=True,
        cu_seqlens=None, head_first=False, use_qk_l2norm_in_kernel=l2)
    t_ref = time.time() - t0

    # fp16 on NPU
    qn, kn, vn, gn, bn, inn = [x.to(DEVICE) for x in (q, k, v, g, beta, init)]
    t0 = time.time()
    npu_o, npu_s = chunk_gated_delta_rule_pytorch(
        q=qn, k=kn, v=vn, g=gn, beta=bn,
        initial_state=inn, output_final_state=True,
        cu_seqlens=None, head_first=False, use_qk_l2norm_in_kernel=l2)
    torch.npu.synchronize()
    t_npu = time.time() - t0

    # fp16 on CPU (isolate NPU-specific error from fp16 truncation error)
    t0 = time.time()
    cpu_o, cpu_s = chunk_gated_delta_rule_pytorch(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=init, output_final_state=True,
        cu_seqlens=None, head_first=False, use_qk_l2norm_in_kernel=l2)
    t_cpu = time.time() - t0

    print(f"  time: ref={t_ref:.2f}s npu={t_npu:.2f}s cpu_fp16={t_cpu:.2f}s")

    # NPU fp16 vs fp32 reference (total error)
    cos, mae, re = metrics(npu_o, ref_o)
    cos_s, mae_s, _ = metrics(npu_s, ref_s)
    print(f"  NPU vs REF   out: cos={cos:.8f} mae={mae:.4e} rel={re:.4e}")
    print(f"  NPU vs REF   st:  cos={cos_s:.8f} mae={mae_s:.4e}")

    # CPU fp16 vs fp32 reference (fp16 truncation error only)
    cos2, mae2, re2 = metrics(cpu_o, ref_o)
    print(f"  CPU16 vs REF out: cos={cos2:.8f} mae={mae2:.4e} rel={re2:.4e}")

    # NPU vs CPU fp16 (isolate NPU-specific precision diff)
    cos3, mae3, re3 = metrics(npu_o, cpu_o)
    cos3s, mae3s, _ = metrics(npu_s, cpu_s)
    print(f"  NPU vs CPU16 out: cos={cos3:.8f} mae={mae3:.4e} rel={re3:.4e}")
    print(f"  NPU vs CPU16 st:  cos={cos3s:.8f} mae={mae3s:.4e}")
    if cos3 < 0.999999:
        print(f"  ** NPU DIVERGES FROM CPU (same fp16 inputs) **")

    # Chunk drift
    cc = chunk_cosines(npu_o, ref_o)
    drift = cc[0] - cc[-1] if len(cc) > 1 else 0
    print(f"  CHUNKS first={cc[0]:.8f} last={cc[-1]:.8f} min={min(cc):.8f}@{cc.index(min(cc))}")
    if abs(drift) > 1e-4:
        print(f"  ** DRIFT={drift:.2e} **")

    return {"label": label, "cos": cos, "mae": mae, "npu_cpu_cos": cos3,
            "npu_cpu_mae": mae3, "drift": drift, "state_cos": cos_s}


def main():
    print("CHUNK_GATED_DELTA_RULE 310P PRECISION TEST")
    print(f"Device: {DEVICE}")
    if DEVICE != "cpu":
        print(f"NPU: {torch_npu.npu.get_device_name(0)}")
    print(f"torch: {torch.__version__}")

    R = []

    R.append(run("1. Sanity (small)", B=1, T=64, Hqk=2, Hv=4, K=16, V=12))
    R.append(run("2. Qwen3 1chunk", B=1, T=64, Hqk=4, Hv=8, K=192, V=128))
    R.append(run("3. Qwen3 8chunks", B=1, T=512, Hqk=4, Hv=8, K=192, V=128))
    R.append(run("4. Qwen3 16chunks", B=1, T=1024, Hqk=4, Hv=8, K=192, V=128))
    R.append(run("5. Strong gate g=2", B=1, T=512, Hqk=4, Hv=8, K=192, V=128, g_scale=2.0))
    R.append(run("6. Weak gate g=0.001", B=1, T=512, Hqk=4, Hv=8, K=192, V=128, g_scale=0.001))
    R.append(run("7. No L2norm", B=1, T=512, Hqk=4, Hv=8, K=192, V=128, l2=False))
    R.append(run("8. Long T=2048", B=1, T=2048, Hqk=4, Hv=8, K=192, V=128))

    # Varlen
    print(f"\n{'='*70}")
    print("9. Varlen cu_seqlens")
    torch.manual_seed(42)
    B, T, Hqk, Hv, K, V = 1, 640, 4, 8, 192, 128
    q = torch.randn(B, T, Hqk, K, dtype=torch.float16)
    k = torch.randn(B, T, Hqk, K, dtype=torch.float16)
    v = torch.randn(B, T, Hv, V, dtype=torch.float16)
    g = -0.2 * torch.rand(B, T, Hv, dtype=torch.float32)
    beta = (0.15 + 0.35 * torch.rand(B, T, Hv, dtype=torch.float32)).to(torch.float16)
    cu = torch.tensor([0, 128, 384, 640], dtype=torch.long)
    init = torch.randn(3, Hv, V, K, dtype=torch.float16)

    npu_o, _ = chunk_gated_delta_rule_pytorch(
        q=q.to(DEVICE), k=k.to(DEVICE), v=v.to(DEVICE), g=g.to(DEVICE), beta=beta.to(DEVICE),
        initial_state=init.to(DEVICE), output_final_state=True,
        cu_seqlens=cu.to(DEVICE) if DEVICE != "cpu" else cu, head_first=False, use_qk_l2norm_in_kernel=True)
    ref_o, _ = chunk_gated_delta_rule_pytorch(
        q=q.float(), k=k.float(), v=v.float(), g=g.float(), beta=beta.float(),
        initial_state=init.float(), output_final_state=True,
        cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=True)
    for i, (s, e) in enumerate([(0, 128), (128, 384), (384, 640)]):
        c, m, r = metrics(npu_o[:, s:e], ref_o[:, s:e])
        print(f"  Seq{i} len={e-s}: cos={c:.8f} mae={m:.4e}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'Label':<25} {'NPU/REF':>9} {'NPU/CPU16':>10} {'mae':>10} {'npu_mae':>10} {'drift':>10}")
    for r in R:
        print(f"  {r['label']:<23} {r['cos']:>9.6f} {r['npu_cpu_cos']:>10.6f} {r['mae']:>10.2e} {r['npu_cpu_mae']:>10.2e} {r['drift']:>10.2e}")

    worst = min(R, key=lambda r: r['cos'])
    print(f"\n  Worst NPU/REF cos: {worst['cos']:.6f} ({worst['label']})")
    npu_worst = min(R, key=lambda r: r['npu_cpu_cos'])
    print(f"  Worst NPU/CPU cos: {npu_worst['npu_cpu_cos']:.6f} ({npu_worst['label']})")
    print(f"  ^^^ This isolates NPU-specific precision error from fp16 truncation")


if __name__ == "__main__":
    main()
