#!/usr/bin/env bash
# tunnel_mesh.sh — build the box↔box SSH tunnel mesh (Vast layout).
#
# The rented instances don't map our service ports, so every box opens a
# persistent SSH connection to each other box, forwarding:
#   local 40000+b*100+i  ->  box b's 127.0.0.1:(30000+i)   (workers)
#   local 40000+b*100+99 ->  box b's 127.0.0.1:8001        (router)
# Cross-region requests therefore traverse the real public Internet
# (inside an SSH channel); intra-region stays on plain loopback.
#
#   bash scripts/vast/tunnel_mesh.sh setup    # keys + start tunnels
#   bash scripts/vast/tunnel_mesh.sh start    # (re)start tunnels only
#   bash scripts/vast/tunnel_mesh.sh status   # probe every tunnel
#   bash scripts/vast/tunnel_mesh.sh stop
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/hosts.env"

N=${#BOX_IP[@]}
MESH_KEY=/root/.ssh/kvcompass_mesh

SSHB() { ssh -i "$SSH_KEY" -p "${BOX_SSH_PORT[$1]}" \
             -o StrictHostKeyChecking=no "$SSH_USER@${BOX_IP[$1]}" "${@:2}"; }

cmd="${1:-setup}"

do_keys() {
  # One shared mesh keypair: generate locally (kept for reuse), install
  # on every box.  The pub goes into authorized_keys idempotently (strip
  # any stale kvcompass-mesh line first — vast's own key sync can race
  # plain appends), then every box's stored pub is verified against the
  # private key's fingerprint.
  local keyf="$HOME/.ssh/kvcompass_mesh_local"
  [[ -f "$keyf" ]] || ssh-keygen -t ed25519 -N "" -f "$keyf" -C kvcompass-mesh -q
  local want; want=$(ssh-keygen -y -f "$keyf" | awk '{print $2}')
  for b in $(seq 0 $((N-1))); do
    SSHB "$b" "mkdir -p /root/.ssh && cat > $MESH_KEY && chmod 600 $MESH_KEY" \
      < "$keyf"
    # NB: ensure the existing file ends with a newline BEFORE appending —
    # vast-injected key files can lack one, and appending to such a file
    # merges the two keys into a single (half-broken) line.  The grep -v
    # then treats them as ONE line and would strip the account key with
    # the stale mesh key, locking everyone out (this happened once).
    SSHB "$b" "touch /root/.ssh/authorized_keys && \
      [ -s /root/.ssh/authorized_keys ] && \
        [ -n \"\$(tail -c1 /root/.ssh/authorized_keys)\" ] && \
        echo >> /root/.ssh/authorized_keys; \
      grep -v kvcompass-mesh /root/.ssh/authorized_keys > /tmp/ak.new || true; \
      cat >> /tmp/ak.new && mv /tmp/ak.new /root/.ssh/authorized_keys && \
      chmod 600 /root/.ssh/authorized_keys" < "$keyf.pub"
    local got
    got=$(SSHB "$b" "ssh-keygen -y -f $MESH_KEY | awk '{print \$2}'; \
                     grep kvcompass-mesh /root/.ssh/authorized_keys | awk '{print \$2}'")
    if [[ "$got" != "$want
$want" ]]; then
      echo "ERROR: key verification failed on box$b"; exit 1
    fi
    echo "  box$b: mesh key installed + verified"
  done
}

start_from() {  # $1 = source box index
  local src="$1" b i fwd
  for b in $(seq 0 $((N-1))); do
    [[ "$b" == "$src" ]] && continue
    fwd=""
    for i in $(seq 0 $((${GPUS_PER_BOX[$b]}-1))); do
      fwd+=" -L $((40000+b*100+i)):127.0.0.1:$((30000+i))"
    done
    fwd+=" -L $((40000+b*100+99)):127.0.0.1:8001"
    # A SUPERVISOR loop per (src,dst) pair, not a raw `ssh -N -f`: a WAN
    # blip kills the ssh client and its forward listeners never come
    # back (this 502'd a whole cross-region cell once) — the loop
    # respawns the tunnel within 5 s.  Kill pass stays a separate SSH
    # call with runtime-assembled patterns (self-match hazard).
    SSHB "$src" "P=\$(printf 'kv%s_to_%s.sh' tunnel '$b'); \
      pkill -f \"\$P\" 2>/dev/null; \
      Q=\$(printf '%s -N -i.*root@%s' ssh '${BOX_IP[$b]}'); \
      pkill -f \"\$Q\" 2>/dev/null; \
      R=\$(printf '%s -N -f -i.*root@%s' ssh '${BOX_IP[$b]}'); \
      pkill -f \"\$R\" 2>/dev/null; true" || true
    SSHB "$src" "cat > /root/kvtunnel_to_$b.sh <<'EOS'
#!/usr/bin/env bash
while true; do
  ssh -N -i $MESH_KEY -p ${BOX_SSH_PORT[$b]} \
    -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
    -o ConnectTimeout=15 \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=2 \
    $fwd root@${BOX_IP[$b]}
  sleep 5
done
EOS
      chmod +x /root/kvtunnel_to_$b.sh; \
      nohup /root/kvtunnel_to_$b.sh > /root/kvtunnel_to_$b.log 2>&1 < /dev/null & \
      echo supervisor_pid=\$!"
    echo "  box$src -> box$b tunnel supervisor up (${GPUS_PER_BOX[$b]} workers + router)"
  done
}

case "$cmd" in
  setup)
    do_keys
    for s in $(seq 0 $((N-1))); do start_from "$s"; done
    ;;
  start)
    for s in $(seq 0 $((N-1))); do start_from "$s"; done
    ;;
  status)
    for s in $(seq 0 $((N-1))); do
      for b in $(seq 0 $((N-1))); do
        [[ "$b" == "$s" ]] && continue
        for i in $(seq 0 $((${GPUS_PER_BOX[$b]}-1))); do
          p=$((40000+b*100+i))
          r=$(SSHB "$s" "curl -sf -m 8 http://127.0.0.1:$p/health >/dev/null 2>&1 && echo OK || echo DOWN")
          echo "box$s -> box$b worker$i (:$p): $r"
        done
      done
    done
    ;;
  stop)
    for s in $(seq 0 $((N-1))); do
      SSHB "$s" "P=\$(printf 'kv%s_to_' tunnel); pkill -f \"\$P\" 2>/dev/null; \
        Q=\$(printf '%s -N -i %s' ssh '$MESH_KEY'); pkill -f \"\$Q\" 2>/dev/null; \
        true" || true
    done
    echo "tunnels stopped"
    ;;
  *) echo "usage: $0 setup|start|status|stop"; exit 1;;
esac
