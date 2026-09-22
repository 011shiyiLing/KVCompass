#!/usr/bin/env bash
# setup_router.sh — run ON each region's c6i.2xlarge router node.
# Clones/installs the repo, builds the Ludo C++ binding, pre-downloads
# the tokenizer.  The router itself is (re)started per experiment cell
# by run_multiregion.sh.
set -euo pipefail

REPO_URL="${REPO_URL:-}"            # or scp the repo over instead
REPO_DIR="${REPO_DIR:-$HOME/KVCompass/realmachine}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"

sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip g++ cmake
if [[ -n "$REPO_URL" && ! -d "$REPO_DIR" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
python3 -m pip install -r requirements.txt
python3 -m pip install httpx aiohttp uvicorn fastapi transformers

# Ludo MinimalPerfectCuckoo binding (falls back to a Python dict if this
# fails, which is fine for correctness but slower).
bash scripts/setup/build_ludo.sh || echo "WARN: ludo binding build failed (dict fallback)"

if [[ -n "${HF_TOKEN:-}" ]]; then
  python3 -c "from huggingface_hub import login; login('$HF_TOKEN')"
fi
python3 - <<EOF
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("$MODEL_NAME")
print("tokenizer cached")
EOF
echo "router node ready"
