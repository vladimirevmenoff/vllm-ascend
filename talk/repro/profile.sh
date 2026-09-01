#!/bin/bash
# Capture a torch-npu device profile of ONE request and rank the kernels,
# split into prefill and per-decode-step. Run INSIDE the container.
#
#   ./profile.sh -m /home/models/Qwen3.5-9B-w8a8-modelslim -d 3 -q -n w8a8
#
# Keep -o small (default 32). Every decode step adds ~1500 kernels to the trace;
# 1024 output tokens would give a multi-GB trace for no extra insight.
set -u

MODEL=""; DEV=0; PORT=8900; QUANT=""; INLEN=2048; OUTLEN=32
UTIL=0.7; MAXSEQS=16; MAXLEN=4096; NAME=prof
OUTDIR=/work/prof_$(date +%Y%m%d_%H%M%S)

usage() {
  cat <<USAGE
Usage: $0 -m MODEL_PATH [-d DEV] [-p PORT] [-q] [-i IN] [-o OUT] [-n NAME] [-O DIR]
  -q  quantized checkpoint (adds --quantization ascend)
  -i  input tokens  (default $INLEN)
  -o  output tokens (default $OUTLEN — keep small, see header)
USAGE
  exit 1
}
while getopts "m:d:p:qi:o:u:s:l:n:O:h" opt; do
  case $opt in
    m) MODEL=$OPTARG ;; d) DEV=$OPTARG ;; p) PORT=$OPTARG ;;
    q) QUANT="--quantization ascend" ;; i) INLEN=$OPTARG ;; o) OUTLEN=$OPTARG ;;
    u) UTIL=$OPTARG ;; s) MAXSEQS=$OPTARG ;; l) MAXLEN=$OPTARG ;;
    n) NAME=$OPTARG ;; O) OUTDIR=$OPTARG ;; *) usage ;;
  esac
done
[ -z "$MODEL" ] && usage
mkdir -p "$OUTDIR/trace"
LOG=$OUTDIR/${NAME}_serve.log
say() { echo "[$(date +%H:%M:%S)] $*"; }

# NOTE: VLLM_TORCH_PROFILER_DIR no longer works in vLLM 0.25.1 — it warns
# "Unknown vLLM environment variable" and /start_profile returns 404.
# The profiler is configured with these two flags instead.
say "starting server (profiler enabled)"
PYTHONUNBUFFERED=1 ASCEND_RT_VISIBLE_DEVICES=$DEV \
vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype float16 --max-model-len "$MAXLEN" \
  --tensor-parallel-size 1 --gpu-memory-utilization "$UTIL" \
  --max-num-seqs "$MAXSEQS" --trust-remote-code $QUANT \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir="$OUTDIR/trace" > "$LOG" 2>&1 &
SPID=$!
cleanup() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; }
trap cleanup EXIT INT TERM

for i in $(seq 1 360); do
  grep -q "Application startup complete" "$LOG" && break
  kill -0 $SPID 2>/dev/null || { say "SERVER DIED"; tail -5 "$LOG"; exit 1; }
  sleep 5
done
grep -q "Application startup complete" "$LOG" || { say "server never became ready"; exit 1; }
say "server ready"

req() { # <output_tokens> <tag>
  vllm bench serve --backend vllm --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --served-model-name "$NAME" --tokenizer "$MODEL" \
    --dataset-name random --random-input-len "$INLEN" --random-output-len "$1" \
    --random-range-ratio 0 --num-prompts 1 --max-concurrency 1 \
    --ignore-eos --seed 0 > "$OUTDIR/${NAME}_$2.log" 2>&1
}

# Warm first, or the trace is dominated by TBE JIT and graph capture.
say "warmup (not profiled)"
req 8 warmup

say "profiling one request: ${INLEN} in / ${OUTLEN} out"
curl -s -X POST "http://127.0.0.1:$PORT/start_profile" > /dev/null
req "$OUTLEN" profiled
curl -s -X POST "http://127.0.0.1:$PORT/stop_profile" > /dev/null

say "waiting for trace flush"      # the CANN parsers run after stop_profile
for i in $(seq 1 60); do
  find "$OUTDIR/trace" -name kernel_details.csv 2>/dev/null | grep -q . && break
  sleep 5
done
cleanup; trap - EXIT

CSVDIR=$(dirname "$(find "$OUTDIR/trace" -name kernel_details.csv | head -1)")
[ -z "$CSVDIR" ] && { say "no kernel_details.csv produced — check $LOG"; exit 1; }
say "trace: $CSVDIR"
python3 "$(dirname "$0")/analyze_profile.py" "$CSVDIR" | tee "$OUTDIR/${NAME}_analysis.txt"
