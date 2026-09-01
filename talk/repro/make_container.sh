#!/bin/bash
# Create a vllm-ascend container for benchmarking on the 310P box.
# Run on the HOST (not inside a container).
#
#   ./make_container.sh mybench
#   ./make_container.sh mybench quay.io/ascend/vllm-ascend:v0.23.0rc1-310p
set -eu

NAME=${1:-}
IMAGE=${2:-quay.io/ascend/vllm-ascend:nightly-main-310p-openeuler}
WORK=${WORK:-/home/claude_bench}

[ -z "$NAME" ] && { echo "usage: $0 <container-name> [image]"; exit 1; }

if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "container '$NAME' already exists. Start it with:  docker start $NAME"
  echo "or remove it with:                                 docker rm -f $NAME"
  exit 1
fi

# --privileged is REQUIRED. Mapping devices individually with --device=/dev/davinciN
# gives 'aclInit ... error 507899' and torch.npu.device_count() == 0. Every working
# container on this box uses --privileged plus the driver bind mount below.
#
# Do NOT pin devices here — pin per run with ASCEND_RT_VISIBLE_DEVICES, after checking
# what is free. Other teams grab chips without warning.
docker run -itd --name "$NAME" --privileged \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home:/home \
  --shm-size=32g \
  "$IMAGE" tail -f /dev/null >/dev/null

# tail -f /dev/null as PID 1: the container survives `pkill -f vllm` and friends.
# (An interactive `bash` as PID 1 dies to a stray pattern match and takes the
# container with it.)

docker exec "$NAME" ln -sfn "$WORK" /work
echo "created '$NAME'  (/work -> $WORK, models at /home/models)"

echo
echo "--- sanity ---"
docker exec "$NAME" bash -lc '
  python -c "import torch, torch_npu; print(\"visible NPUs:\", torch.npu.device_count())" 2>/dev/null | tail -1
  pip show vllm_ascend 2>/dev/null | grep -iE "^Version|^Editable"
' || true

cat <<'EOF'

If the sanity block printed an "Editable project location", this container imports
somebody's working copy rather than the image's own vllm-ascend — fine if that is
what you want, wrong for a stock baseline.

Next:
  docker exec -it <name> bash
  python /work/repro/freemem.py                 # pick a chip with >35 GB free
  /work/repro/run_bench.sh -m /home/models/Qwen3.5-9B -d <DEV> -n fp16
EOF
