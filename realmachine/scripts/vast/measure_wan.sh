#!/usr/bin/env bash
# measure_wan.sh — wrapper: flatten hosts.env arrays and run the TCP
# WAN measurement, then (optionally) push the matrix to every box.
#   bash scripts/vast/measure_wan.sh [--push]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/hosts.env"

BOX_IP_FLAT="${BOX_IP[*]}" \
BOX_SSH_PORT_FLAT="${BOX_SSH_PORT[*]}" \
REGION_NAMES_FLAT="${REGION_NAMES[*]}" \
SSH_USER="$SSH_USER" SSH_KEY="$SSH_KEY" \
python3 "$SCRIPT_DIR/measure_wan_tcp.py" --out wan_matrix_real.json

if [[ "${1:-}" == "--push" ]]; then
  for i in "${!BOX_IP[@]}"; do
    scp -i "$SSH_KEY" -P "${BOX_SSH_PORT[$i]}" -o StrictHostKeyChecking=no \
      wan_matrix_real.json "$SSH_USER@${BOX_IP[$i]}:$REPO_DIR/"
    echo "pushed to ${REGION_NAMES[$i]}"
  done
fi
