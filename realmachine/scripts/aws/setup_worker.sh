#!/usr/bin/env bash
# setup_worker.sh — run ON each g6.xlarge (1×L4) GPU node.
# Installs SGLang and launches one server on 0.0.0.0:$SGLANG_PORT.
#
#   MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=hf_... \
#   bash setup_worker.sh [install|start|stop|status]
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.82}"
# RadixCache eviction policy — matches the sim's per-workload setting
# (simulate.py EVICT_POLICY): mixed workload → lfu, wildchat/lmsys → lru.
EVICTION_POLICY="${EVICTION_POLICY:-lru}"
LOG=~/sglang_worker.log
PIDFILE=~/sglang_worker.pid

cmd="${1:-start}"

case "$cmd" in
  install)
    # Deep Learning AMI (Ubuntu 22.04, PyTorch) already has CUDA drivers.
    python3 -m pip install --upgrade pip
    # Pin to the version the 8×A100 experiments used.
    python3 -m pip install "sglang[all]==0.5.10.post1"
    if [[ -n "${HF_TOKEN:-}" ]]; then
      python3 -c "from huggingface_hub import login; login('$HF_TOKEN')"
    fi
    echo "install done"
    ;;
  start)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running (pid $(cat "$PIDFILE"))"; exit 0
    fi
    nohup python3 -m sglang.launch_server \
        --model-path "$MODEL_NAME" \
        --host 0.0.0.0 --port "$SGLANG_PORT" \
        --mem-fraction-static "$MEM_FRACTION" \
        --radix-eviction-policy "$EVICTION_POLICY" \
        --enable-metrics \
        > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "started pid $(cat "$PIDFILE"); waiting for /health ..."
    for i in $(seq 1 180); do
      if curl -sf -m 2 "http://127.0.0.1:${SGLANG_PORT}/health" >/dev/null; then
        echo "healthy after ${i}s"; exit 0
      fi
      sleep 5
    done
    echo "ERROR: worker did not become healthy; tail $LOG:"; tail -20 "$LOG"
    exit 1
    ;;
  stop)
    [[ -f "$PIDFILE" ]] && kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"; echo "stopped"
    ;;
  restart)
    "$0" stop; sleep 3; exec "$0" start
    ;;
  status)
    curl -sf -m 2 "http://127.0.0.1:${SGLANG_PORT}/health" \
      && echo "healthy" || echo "not healthy"
    ;;
  *) echo "usage: $0 install|start|stop|restart|status"; exit 1;;
esac
