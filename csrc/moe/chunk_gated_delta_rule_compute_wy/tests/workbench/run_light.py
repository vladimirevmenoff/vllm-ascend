# Cosine harness (10c): op vs torch WY ref across shapes_light.txt, 3 timed runs each.
import json, sys, time, torch, torch_npu
sys.path.insert(0, "/vllm-workspace/vllm-ascend-combined")
import vllm_ascend.vllm_ascend_C  # noqa
from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import _compute_kernel_inputs_from_torch_wy
torch.npu.set_device(0); torch.manual_seed(7)
results = {}
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#"): continue
    B,T,Hk,Hv,K,Vd = map(int, line.split(","))
    q = torch.randn(B,T,Hk,K, dtype=torch.float16).npu()
    k = torch.nn.functional.normalize(torch.randn(B,T,Hk,K).float(), dim=-1).half().npu()
    v = (torch.randn(B,T,Hv,Vd, dtype=torch.float16)*0.5).npu()
    g = (-torch.rand(B,T,Hv).float()*0.1).npu()
    beta = torch.rand(B,T,Hv, dtype=torch.float16).npu()
    ref = _compute_kernel_inputs_from_torch_wy(q,k,v,g,beta,64)
    durs=[]
    for it in range(3):
        torch.npu.synchronize(); t0=time.time()
        out = torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(q,k,v,g,beta,64)
        torch.npu.synchronize(); durs.append((time.time()-t0)*1e6)
    entry={"task_duration_us_runs":[round(d,1) for d in durs]}
    worst=1.0
    for n,(r,o) in zip(["q","k","w","u","g"], zip(ref,out)):
        r32,o32=r.float().flatten(),o.float().flatten()
        cos=torch.nn.functional.cosine_similarity(r32,o32,dim=0).item()
        entry[n+"_cos"]=round(cos,7); worst=min(worst,cos)
    entry["min_cos"]=round(worst,7)
    results[line]=entry
    print(line, "min_cos=%.6f"%worst, "dur_us=%s"%entry["task_duration_us_runs"], flush=True)
mn=min(e["min_cos"] for e in results.values())
json.dump({"shapes":results,"axis_4_min_cosine":mn,"axis_4_shape_count":len(results),
           "axis_4_status":"PASS" if mn>=0.999 else "FAIL"}, open(sys.argv[2],"w"), indent=1)
print("AXIS4", "PASS" if mn>=0.999 else "FAIL", mn, len(results))
