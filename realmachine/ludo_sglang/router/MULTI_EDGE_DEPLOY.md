# Multi-edge deployment (Option B gossip)

This document covers how to bring up N routers across N regions and have
their LudoTables converge via gossip. The router half of "Option B" from
the design discussion.

## TL;DR

Per router process, set:

```bash
export ROUTER_ID=<unique non-negative integer>          # 0..N-1
export PEER_ROUTER_URLS="http://router-r2:8001,http://router-r3:8001,..."
export GOSSIP_PERIOD_S=1.0                              # default; tune as needed
```

Then start the router as usual:

```bash
python -m ludo_sglang.router.ludo_router
```

That's it. The router will print a `[router] gossip enabled: K peer(s)`
banner at startup if everything is wired correctly.

## What gossip does, in one paragraph

Each router's `LudoScheme.advertise()` / `notify_tier()` / `notify_evict()`
produces a `(key, worker_idx, tier, epoch)` delta after applying the
write locally. The `GossipManager` batches these into a per-router
outbound buffer and flushes the buffer to every peer router every
`GOSSIP_PERIOD_S` seconds via `POST /internal/gossip_apply`. Each peer's
endpoint feeds the deltas into `LudoTable.apply_remote()`, which uses
per-`(key, worker_idx)` LWW (`(epoch, source_router_id)` lex-order
tie-break) plus a Lamport-clock bump on the local `_epoch_counter`.
Convergence window is `Δ_gossip ≈ GOSSIP_PERIOD_S + max(WAN to any
peer)`; under that window every peer either applies or LWW-rejects each
delta.

## Required configuration

| Env var                  | Type        | Default | Notes                                                              |
|--------------------------|-------------|---------|---------------------------------------------------------------------|
| `ROUTER_ID`              | int ≥ 0     | `0`     | Must be unique per router process; tie-breaker for LWW              |
| `PEER_ROUTER_URLS`       | CSV of URLs | empty   | Empty = single-edge mode (no gossip overhead)                       |
| `GOSSIP_PERIOD_S`        | float > 0   | `1.0`   | Bigger = lower bandwidth + larger staleness window                  |
| `GOSSIP_HTTP_TIMEOUT_S`  | float > 0   | `1.0`   | Per-POST httpx timeout; failed pushes are silently dropped (LWW recovers) |
| `GOSSIP_MAX_BATCH`       | int > 0     | `1000`  | Max deltas per push; overflow rolls into the next round             |

## Topology guidance

### One router per region

Recommended. Each region's clients hit their nearest router; routers
gossip to each other in the background. Convergence ≈ 1–2 s with a 1 s
gossip period over typical inter-cloud peering links.

### Star / mesh

Gossip is point-to-point: every router POSTs to every peer URL it has
listed. The full mesh is the simplest and the most robust — losing one
peer doesn't isolate any pair.

For O(N) routers you can degrade to a sparse overlay (each router
gossips to log N random peers per round, antientropy-style) but the
push loop in `gossip.py` is hard-coded for full mesh. For workshop /
2026 paper scale (≤ 8 regions) full mesh is fine.

### Single-router (legacy)

Leave `PEER_ROUTER_URLS` unset. Gossip is fully disabled — no buffer
overhead, no background task, no LWW path on the hot side (LWW gate
still runs but always passes since no peers ever write to your table).
Behaves identically to pre-Option-B for backwards compatibility.

## Operational concerns

### Bootstrap

A freshly-started router has an empty table. Existing entries in peer
routers will *not* be pushed to the new joiner — gossip only carries
deltas produced after the joiner started.

In the workshop experiment we always start all routers from a clean
state simultaneously, so this is a non-issue. For production you'd add
a `GET /internal/gossip_snapshot` endpoint that returns the full
`(key, mask, slot_epochs)` state and have new routers pull from one
peer on startup. Not implemented.

### Failure handling

* **Peer down**: pushes to that peer fail and are silently dropped
  (LWW recovers when the peer comes back and someone re-advertises).
* **Local router crash**: in-memory state lost. Same recovery: peers
  push their next deltas; natural advertise traffic re-warms the table.
* **Network partition**: each side accepts local writes; LWW reconciles
  when the partition heals. Slots that diverge during the partition
  are resolved by `(epoch, source_id)`, with the higher pair winning.

### Memory cost

Per active slot adds ~80 bytes (Python tuple + dict overhead) for the
slot-epoch tracking. At workshop scale (≤ 65k entries × ≤ 16 workers ×
typical ≤ 2 active workers per entry) this is ~10 MB extra. At
multi-million-entry scale it would dominate the LudoTable's own
footprint and we'd want a more compact slot-epoch layout (e.g. arrays
indexed by slot rather than dicts), or reduce per-slot epoch storage
to a single 8-byte uint64 packing (epoch_mod_2^48, source_id_mod_2^16).
For the workshop the dict form is fine.

### Bandwidth cost

Per delta on the wire: ~32 bytes JSON. Per push (1 s period, peer
fan-out N): `traffic = (advertise_rate × Δ × N × 32 B/s)`. At 100 req/s
with 8 chunk-size levels and 4 peers: `100 × 8 × 4 × 32 ≈ 100 KB/s` per
router. Not a concern at workshop scale.

## Checking it works

### Convergence smoke test

Run two routers locally on different ports, configure them as peers of
each other:

```bash
# Terminal 1
ROUTER_ID=0 ROUTER_PORT=8001 \
PEER_ROUTER_URLS=http://127.0.0.1:8002 \
python -m ludo_sglang.router.ludo_router

# Terminal 2
ROUTER_ID=1 ROUTER_PORT=8002 \
PEER_ROUTER_URLS=http://127.0.0.1:8001 \
python -m ludo_sglang.router.ludo_router
```

Send a single request to router 0:

```bash
curl -X POST http://127.0.0.1:8001/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "the quick brown fox", "max_new_tokens": 8}'
```

After ≤ `GOSSIP_PERIOD_S` seconds, query router 1's stats:

```bash
curl -s http://127.0.0.1:8002/scheme_stats | jq '.scheme.entries, .gossip'
```

You should see `entries > 0` on router 1, and the gossip stats should
show `pushes_succeeded > 0` on router 0 and `deltas_applied_in > 0` on
router 1.

### Convergence under load

Run the existing trace replayer with the same trace file against each
router (or against just one — the gossip will replicate). After the run,
compare `/scheme_stats` between routers; the `entries` count should be
the same to within `±` the gossip in-flight window.

### LWW semantics

The internal unit tests in `src/ludo/ludo_table.py` (run by hand from
the repo root with `python -c '...'` or wrap them in pytest if you
prefer) cover:

* duplicate gossip is idempotent
* out-of-order delivery converges
* concurrent conflicting writes from different routers tie-break by
  `source_id` and produce identical final state on every router
* Lamport clock bump prevents local-write starvation after seeing
  high-epoch peer writes
