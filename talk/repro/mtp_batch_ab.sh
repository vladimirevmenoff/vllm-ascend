#!/bin/bash
# Batched A/B for MTP on W8A8: 8 concurrent real prompts, with and without MTP.
#
# Real prompts matter -- acceptance is a model-quality metric and collapses on the
# random-token data `vllm bench serve` generates. Concurrency matters because the
# draft pass is weight-bound, so one batched draft serves every in-flight sequence;
# that is the effect being measured.
#
# MTP arm points the drafter at the fp16 checkpoint: loading a drafter from a
# quantized checkpoint hangs (see findings), and the head is fp16 either way.
set -u
BS=${BS:-8}
MODEL=/home/models/Qwen3.5-9B-w8a8-mtp
PORT=8990

run() {  # <label> <extra serve args...>
  local LABEL=$1; shift
  local DEV
  DEV=$(python /home/claude_bench/repro/freemem.py 2>/dev/null | grep dev \
        | awk '{print $4, $2}' | sort -rn | head -1 | awk '{print $2}')
  echo "=== $LABEL on device $DEV (BS=$BS)"
  PYTHONUNBUFFERED=1 ASCEND_RT_VISIBLE_DEVICES=$DEV vllm serve "$MODEL" \
    --served-model-name q --host 127.0.0.1 --port $PORT --dtype float16 \
    --max-model-len 4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.75 \
    --max-num-seqs 16 --trust-remote-code --quantization ascend \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    "$@" > "/home/claude_bench/bat_${LABEL}.log" 2>&1 &
  local SPID=$!
  local i
  for i in $(seq 1 420); do
    grep -q "Application startup complete" "/home/claude_bench/bat_${LABEL}.log" && break
    kill -0 $SPID 2>/dev/null || { echo "  SERVER_DIED"; grep -a "core.py:1231" "/home/claude_bench/bat_${LABEL}.log" | tail -3; return 1; }
    sleep 5
  done
  grep -q "Application startup complete" "/home/claude_bench/bat_${LABEL}.log" || { echo "  NEVER_READY"; kill $SPID; return 1; }

  BS=$BS PORT=$PORT LABEL=$LABEL python3 - <<'PY'
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BS = int(os.environ["BS"]); PORT = os.environ["PORT"]; LABEL = os.environ["LABEL"]
PROMPTS = [
 "The history of the Roman Empire is often divided into distinct periods. The Republic gave way to the Principate after",
 "In machine learning, gradient descent is an optimization algorithm used to minimize a loss function. The basic idea is",
 "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n",
 "The Treaty of Westphalia, signed in 1648, ended the Thirty Years War and established the principle of",
 "Photosynthesis is the process by which green plants convert light energy into chemical energy. During the light-dependent reactions,",
 "The CAP theorem states that a distributed data store cannot simultaneously provide more than two of three guarantees:",
 "class LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n",
 "Plate tectonics explains the large-scale motion of Earth's lithosphere. Convergent boundaries produce",
]
prompts = (PROMPTS * ((BS // len(PROMPTS)) + 1))[:BS]

def ask(p):
    body = json.dumps({"model": "q", "prompt": p, "max_tokens": 128, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return time.perf_counter() - t0, out["usage"]["completion_tokens"]

with ThreadPoolExecutor(max_workers=BS) as ex:      # warm
    list(ex.map(ask, prompts))

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=BS) as ex:
    res = list(ex.map(ask, prompts))
wall = time.perf_counter() - t0

tokens = sum(n for _, n in res)
per_req = sum(dt / n for dt, n in res) / len(res) * 1000
print(f"RESULT {LABEL} BS={BS}: per-request {per_req:.2f} ms/token | "
      f"aggregate {tokens/wall:.2f} tok/s | wall {wall*1000:.0f} ms | {tokens} tokens")
PY
  grep -aoE "acceptance rate: [0-9.]+%" "/home/claude_bench/bat_${LABEL}.log" | tail -3
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 25
}

run baseline
run mtp1 --speculative-config '{"method":"mtp","num_speculative_tokens":1,"model":"/home/models/Qwen3.5-9B"}'
