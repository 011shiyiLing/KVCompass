#!/usr/bin/env bash
# setup_box.sh — run ON a Vast.ai multi-GPU box (inside the container).
# One box = one region: installs deps once, then starts/stops ONE SGLang
# worker per GPU (CUDA_VISIBLE_DEVICES=i, internal port 30000+i).
#
#   N_GPUS=8 MODEL_NAME=... HF_TOKEN=... bash setup_box.sh install
#   N_GPUS=8 EVICTION_POLICY=lfu bash setup_box.sh start
#   bash setup_box.sh stop | restart | status
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
N_GPUS="${N_GPUS:-$(nvidia-smi -L | wc -l)}"
BASE_PORT="${BASE_PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
EVICTION_POLICY="${EVICTION_POLICY:-lru}"
LOGDIR=/root/sglang_logs
mkdir -p "$LOGDIR"

cmd="${1:-start}"

case "$cmd" in
  install)
    pip install --upgrade pip -q
    # RTX 5090 (Blackwell, sm_120) needs a cu128+ torch — recent sglang
    # wheels ship it; pin loosely and let pip resolve.
    pip install -q "sglang[all]>=0.5.10"
    pip install -q httpx aiohttp uvicorn fastapi transformers numpy
    if [[ -n "${HF_TOKEN:-}" ]]; then
      python3 -c "from huggingface_hub import login; login('$HF_TOKEN')"
    fi
    python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("$MODEL_NAME", allow_patterns=["*.json","*.safetensors","tokenizer*"])
print("model cached")
EOF
    echo "install done ($N_GPUS GPUs: $(nvidia-smi -L | head -1) ...)"
    ;;
  start)
    for i in $(seq 0 $((N_GPUS-1))); do
      port=$((BASE_PORT + i))
      if curl -sf -m 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        echo "worker $i already up on :$port"; continue
      fi
      CUDA_VISIBLE_DEVICES=$i nohup python3 -m sglang.launch_server \
          --model-path "$MODEL_NAME" \
          --host 0.0.0.0 --port "$port" \
          --mem-fraction-static "$MEM_FRACTION" \
          --radix-eviction-policy "$EVICTION_POLICY" \
          > "$LOGDIR/worker_$i.log" 2>&1 &
      echo $! > "$LOGDIR/worker_$i.pid"
      echo "worker $i starting on :$port (policy=$EVICTION_POLICY)"
      # Stagger launches: N simultaneous weight loads (~16 GB each into
      # page cache + pinned buffers) can spike past the container's
      # memory cap and get a worker OOM-killed (observed on the 8-GPU
      # box, 2026-07-08).
      sleep "${STAGGER_S:-8}"
    done
    echo "waiting for health ..."
    for i in $(seq 0 $((N_GPUS-1))); do
      port=$((BASE_PORT + i)); ok=0
      for t in $(seq 1 240); do
        curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 \
          && { ok=1; break; }
        sleep 5
      done
      [[ $ok == 1 ]] && echo "worker $i healthy" \
        || { echo "ERROR worker $i: tail $LOGDIR/worker_$i.log"; tail -5 "$LOGDIR/worker_$i.log"; exit 1; }
    done
    ;;
  stop)
    # Kill launchers AND the setproctitle-renamed children
    # (sglang::scheduler etc.) — the schedulers hold the GPU memory and
    # survive a launcher-only kill, OOMing the next start.  (Literal
    # patterns are safe here: this runs from a script file, so the
    # pattern is not part of any process's cmdline.)
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang::" 2>/dev/null || true
    rm -f "$LOGDIR"/worker_*.pid
    for t in $(seq 1 12); do
      nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | grep -q . || break
      sleep 5
    done
    echo "stopped (GPUs freed)"
    ;;
  restart)
    bash "$0" stop; sleep 5
    N_GPUS=$N_GPUS EVICTION_POLICY=$EVICTION_POLICY exec bash "$0" start
    ;;
  status)
    for i in $(seq 0 $((N_GPUS-1))); do
      port=$((BASE_PORT + i))
      curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 \
        && echo "worker $i :$port healthy" || echo "worker $i :$port DOWN"
    done
    ;;
  *) echo "usage: $0 install|start|stop|restart|status"; exit 1;;
esac
