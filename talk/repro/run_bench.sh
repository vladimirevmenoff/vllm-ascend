#!/bin/bash
# TTFT / TPOT / prefill / decode benchmark for vllm-ascend on Ascend 310P.
# Starts a vLLM server, warms it, sweeps batch sizes, prints a markdown table.
# Run INSIDE the container (see README.md for the docker run line).
set -u

MODEL=""
DEV=0
PORT=8100
QUANT=""
BSLIST="1 2 4 8"
INLEN=2048
OUTLEN=1024
UTIL=0.7
MAXSEQS=16
MAXLEN=4096
OUTDIR=/work/bench_$(date +%Y%m%d_%H%M%S)
NAME=bench
SAMPLING=""

usage() {
  cat <<USAGE
Usage: $0 -m MODEL_PATH [options]
  -m PATH   model directory (required)
  -d N      NPU device id, pinned via ASCEND_RT_VISIBLE_DEVICES (default $DEV)
  -p N      server port (default $PORT)
  -q        quantized checkpoint: adds --quantization ascend (msModelSlim W8A8)
  -b "..."  batch sizes / concurrencies to sweep (default "$BSLIST")
  -i N      input tokens  (default $INLEN)
  -o N      output tokens (default $OUTLEN)
  -u F      gpu-memory-utilization (default $UTIL)
  -s N      max-num-seqs (default $MAXSEQS)
  -l N      max-model-len (default $MAXLEN, must exceed input+output)
  -n NAME   run label used in filenames (default $NAME)
  -O DIR    output directory (default /work/bench_<timestamp>)
  -S "..."  extra args passed verbatim to 'vllm bench serve', e.g. sampling params:
            -S "--temperature 1.0 --top-k 50 --top-p 0.9"
Examples:
  $0 -m /home/models/Qwen3.5-9B -d 0 -n fp16
  $0 -m /home/models/Qwen3.5-9B-w8a8-modelslim -d 1 -p 8200 -q -n w8a8
USAGE
  exit 1
}

while getopts "m:d:p:qb:i:o:u:s:l:n:O:S:h" opt; do
  case $opt in
    m) MODEL=$OPTARG ;;   d) DEV=$OPTARG ;;     p) PORT=$OPTARG ;;
    q) QUANT="--quantization ascend" ;;         b) BSLIST=$OPTARG ;;
    i) INLEN=$OPTARG ;;   o) OUTLEN=$OPTARG ;;  u) UTIL=$OPTARG ;;
    s) MAXSEQS=$OPTARG ;; l) MAXLEN=$OPTARG ;;  n) NAME=$OPTARG ;;
    O) OUTDIR=$OPTARG ;;  S) SAMPLING=$OPTARG ;; *) usage ;;
  esac
done
[ -z "$MODEL" ] && usage
[ -d "$MODEL" ] || { echo "no such model dir: $MODEL"; exit 1; }
[ $((INLEN + OUTLEN)) -gt "$MAXLEN" ] && { echo "max-model-len $MAXLEN too small for $INLEN+$OUTLEN"; exit 1; }

mkdir -p "$OUTDIR"
SERVE_LOG=$OUTDIR/${NAME}_serve.log
RUN_LOG=$OUTDIR/${NAME}_run.log
: > "$RUN_LOG"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN_LOG"; }

# --- free-memory check -------------------------------------------------------
# Other users grab chips without warning; vLLM's own check happens 4 min into
# startup, so look first. This prints free/total for every visible device.
python - <<'PY' 2>/dev/null | tee -a "$RUN_LOG"
import torch, torch_npu
for d in range(torch.npu.device_count()):
    try:
        torch.npu.set_device(d)
        f, t = torch.npu.mem_get_info(d)
        print("  dev %d free %.1f / %.1f GiB" % (d, f/2**30, t/2**30))
    except Exception as e:
        print("  dev %d ERR %s" % (d, str(e)[:50]))
PY

# --- server ------------------------------------------------------------------
log "starting server: model=$MODEL dev=$DEV port=$PORT util=$UTIL ${QUANT:-fp16}"
PYTHONUNBUFFERED=1 ASCEND_RT_VISIBLE_DEVICES=$DEV \
vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype float16 \
  --max-model-len "$MAXLEN" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$UTIL" \
  --max-num-seqs "$MAXSEQS" \
  --trust-remote-code $QUANT ${EXTRA_SERVE:-} > "$SERVE_LOG" 2>&1 &
SPID=$!
cleanup() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; }
trap cleanup EXIT INT TERM

# Cold start is ~5-8 min on 310P: weight load, TBE JIT, torch.compile, graph capture.
for i in $(seq 1 360); do
  grep -q "Application startup complete" "$SERVE_LOG" && break
  if ! kill -0 $SPID 2>/dev/null; then
    log "SERVER DIED — last lines:"; tail -5 "$SERVE_LOG" | tee -a "$RUN_LOG"; exit 1
  fi
  sleep 5
done
grep -q "Application startup complete" "$SERVE_LOG" || { log "server never became ready"; exit 1; }
log "server ready ($(grep -a -m1 'KV cache size' "$SERVE_LOG" | sed 's/.*GPU //'))"

# --- sanity: the model must emit sensible text before any timing counts -------
SMOKE=$(curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$NAME\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"].strip()[:80])' 2>/dev/null)
log "smoke output: ${SMOKE:-<EMPTY — check $SERVE_LOG>}"
[ -z "$SMOKE" ] && { log "smoke test failed, aborting"; exit 1; }

bench() { # <tag> <num_prompts> <concurrency>
  vllm bench serve \
    --backend vllm \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --served-model-name "$NAME" --tokenizer "$MODEL" \
    --dataset-name random \
    --random-input-len "$INLEN" --random-output-len "$OUTLEN" --random-range-ratio 0 \
    --num-prompts "$2" --max-concurrency "$3" \
    --ignore-eos \
    --seed 0 \
    $SAMPLING \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-filename "$OUTDIR/${NAME}_$1.json" > "$OUTDIR/${NAME}_$1.log" 2>&1
}

# --- warmup (discarded): first request pays JIT / graph-capture cost ----------
log "warmup"
bench warmup 2 1 || { log "warmup failed, see $OUTDIR/${NAME}_warmup.log"; exit 1; }

# --- sweep -------------------------------------------------------------------
for C in $BSLIST; do
  NP=$((C * 2)); [ $NP -lt 4 ] && NP=4
  log "BS=$C ($NP prompts)"
  bench "bs$C" "$NP" "$C" || log "BS=$C FAILED, see $OUTDIR/${NAME}_bs$C.log"
done

cleanup; trap - EXIT
log "done — results in $OUTDIR"
python3 "$(dirname "$0")/summarize.py" "$OUTDIR" | tee -a "$RUN_LOG"
