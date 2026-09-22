# SGLang HiRadixCache patch — wiring tier notifications

This document tells you where to hook the four KV-cache lifecycle
events into SGLang's hierarchical RadixCache so the discovery plane
gets a faithful view of every worker's cache state.

The hooks are **fire-and-forget** HTTP POSTs from the worker; they
never block inference.  If neither `LUDO_AUTHORITY_URLS` nor
`LUDO_ROUTER_URL` is set in the environment, every hook is a no-op,
so the patched SGLang is safe to ship even when running without a
discovery plane.

---

## Two deployment modes

`tier_notifier.py` auto-detects which discovery-plane mode is active
from the environment:

| Env var set                     | Mode      | Where the hook POSTs                |
|---------------------------------|-----------|--------------------------------------|
| `LUDO_AUTHORITY_URLS=...`          | central   | the shard's authority, picked by `hash(key)` |
| `LUDO_ROUTER_URL=...`           | gossip    | the local-region edge router         |
| (neither)                       | disabled  | no-op                                |

In **central mode** the worker is the *primary writer*: every event
including the initial advertise after a finished prefill is fired
from the worker.  In **gossip mode** the edge router writes its own
LudoTable on dispatch and the worker only reports tier transitions —
matching the pre-existing behaviour.

---

## The strict three-stage cache pipeline

Under our deployment model, a chunk's KV traverses exactly:

    [VRAM]  ──demote──→  [DRAM]  ──evict──→  [gone]
        ↑                    │
        └────promote─────────┘

There is **no direct VRAM → gone** path.  When VRAM is full and a
chunk needs to leave VRAM, SGLang demotes it to host pinned memory
(DRAM tier).  When DRAM is full, the host eviction path discards the
chunk entirely.  Therefore the four notifier entry points map 1:1
onto the four physical transitions in the pipeline.

If your deployment disables the host DRAM tier, "demote" will
effectively become "evict" (chunk is freed when VRAM kicks it out).
In that case hook `notify_evict` instead of `notify_demote` on the
VRAM eviction path — the authority-side state machine is agnostic to
which tier the chunk left from.

---

## Worker-side client

```python
from ludo_sglang.worker.tier_notifier import (
    notify_advertise,
    notify_demote, notify_promote, notify_evict,
    notify_demote_bulk, notify_evict_bulk,
)
```

Set the appropriate env vars before launching SGLang:

```bash
# Central mode (preferred):
export LUDO_AUTHORITY_URLS=http://authority0:8101,http://authority1:8101
export LUDO_WORKER_RANK=2                    # this worker's index
export LUDO_NOTIFIER_TIMEOUT=1               # seconds; default 1
# Optional; defaults to 256 (the smallest level in LUDO_CHUNK_SIZES).
# Must agree across worker / authority / lookup.
export LUDO_SHARD_HASH_LEVEL_TOKENS=256

# Gossip mode (legacy):
export LUDO_ROUTER_URL=http://edge-router:8001
```

---

## Where to put the four hooks

### §1. Prefill finished — `cache_finished_req(req)` (central mode only)

In gossip mode the edge router advertises VRAM itself after dispatch.
**In central mode** the router is read-only, so the worker emits
the initial VRAM advertise.

```python
def cache_finished_req(self, req):
    # ... existing logic to admit the freshly-prefilled KV into VRAM ...
    notify_advertise(token_ids=req.fill_ids)
```

### §2. VRAM → DRAM — `write_through_to_host_memory(node)`

The canonical SGLang HiRadixCache method that copies a `RadixTreeNode`'s
KV out to host pinned memory and frees its VRAM slot.  At the **end**
of this method (after the copy succeeds), iterate over the chunk-token
sequences the discovery plane indexes and call `notify_demote`:

```python
def write_through_to_host_memory(self, node):
    # ... existing logic to copy node.value to host memory and
    # release node.value (the GPU KV blocks) ...
    for token_ids in self._collect_advertised_chunks(node):
        notify_demote(rank=self.tp_rank, token_ids=token_ids)
```

### §3. DRAM → VRAM — `load_from_host_memory(node)`

Mirror of §2.  After SGLang copies the KV back from host pinned memory
into VRAM:

```python
def load_from_host_memory(self, node):
    # ... existing logic to allocate VRAM and copy from host pinned
    # memory back to GPU ...
    for token_ids in self._collect_advertised_chunks(node):
        notify_promote(rank=self.tp_rank, token_ids=token_ids)
```

### §4. DRAM → gone — host pinned-memory eviction

The strongest event — DRAM tier is the chunk's last stop, and once
the host eviction path drops it, the chunk is gone from this worker
entirely.  The exact method name varies between SGLang versions
(`host_memory_evict`, `evict_host_memory_pool`, the host branch of
`RadixTreeNode.release`, …); look for the call site that **actually
frees the host pinned-memory blocks** — not the VRAM eviction path,
which is just a demote (already covered in §2):

```python
def host_memory_evict(self, num_bytes):
    fully_evicted_host_nodes = []
    # ... existing host-memory eviction logic.  fully_evicted_host_nodes
    # is the list of RadixTreeNodes whose host KV blocks were just
    # released (NOT the ones merely demoted from VRAM into DRAM, which
    # §2 already handled). ...
    for node in fully_evicted_host_nodes:
        for token_ids in self._collect_advertised_chunks(node):
            notify_evict(rank=self.tp_rank, token_ids=token_ids)
```

> **VRAM-tier eviction** (the `evict(num_tokens)` method in older
> SGLang versions) under our pipeline is *always* a demote, not a
> final discard — every freed VRAM slot is first written through to
> DRAM by §2.  Do not also fire `notify_evict` from the VRAM eviction
> path; that would double-count and incorrectly remove a chunk from
> the discovery index while it's still resident in DRAM.

---

## What `_collect_advertised_chunks(node)` should return

The notifier re-derives the per-level Ludo keys by calling
`chunk_to_key(token_ids[:cs])` for every `cs in LUDO_CHUNK_SIZES`,
so the SGLang-side helper just needs to return the **full
prompt-prefix token sequence** the chunk represents:

```python
def _collect_advertised_chunks(self, node):
    """Reconstruct the prompt-prefix token sequence whose discovery-plane
    advertise originally created this RadixTreeNode.  In SGLang's
    RadixCache a node represents a contiguous prefix path from root, so:
    """
    return [self._tokens_from_root_to(node)]
```

Returning multiple sequences is supported and useful when a single
eviction call frees a whole subtree.

---

## What the discovery plane does for you

### Central mode (authority / lookup)

* Every event becomes a slot-update primitive on the authority:
  `set_slot(rank, key, tier)` writes the worker's 2-bit slot in the
  packed 32-bit value.  When the resulting value is zero (last
  holder cleared), the entry is dropped from the authority's
  `AuthorityLudoTable` AND the authority broadcasts a `packed_value=0`
  delta to every lookup so they delete their mirrors.
* Per-worker monotonic `seq` numbers protect the authority from
  out-of-order delivery (a tier event with `seq <= last_seq[rank]`
  is dropped).
* Cross-shard mismatches are silently dropped: if a authority receives
  an event whose key falls into a different shard, it ignores it
  (the worker's `_authority_for(token_ids)` lookup should always pick
  the right shard, but the authority double-checks).

### Gossip mode (edge router)

* Every event becomes a `LudoTable` slot-update with LWW versioning.
* The local router buffers the resulting delta and broadcasts it to
  all peer routers at `GOSSIP_PERIOD_S` cadence.  Peer routers apply
  the delta through the same LWW gate.
* Initial VRAM advertise is fired *by the router itself* after
  dispatch; `notify_advertise` from the worker is a no-op in this
  mode.

---

## Performance characteristics

* Each HTTP POST is ~0.3–1 ms on a localhost router.  The notifier
  schedules them on the asyncio loop and does not block inference.
* Authority-side `/advertise` and `/tier_event` handlers do one
  `AuthorityLudoTable.set_slot` per level (~hundreds of ns each), then
  one parallel HTTP fan-out to all registered lookups.
* Coalescing: if SGLang demotes a chunk and immediately promotes it
  back, the authority sees two events and ends up at TIER_VRAM, which
  is exactly the steady state.  No special handling needed.

---

## Defensive defaults

Even if the patch is buggy or the notifier flakes:

1. The authority / router endpoints validate `rank ∈ [0, N_workers)`,
   `event ∈ {demote, promote, evict}`, and `seq` monotonicity, and
   return 400 / silent drop on bad input.
2. The notifier swallows all HTTP errors — a authority restart or a
   transient timeout never breaks SGLang.
3. The router-side `cache_belief` EMA is still active as
   defence-in-depth: even if a tier notification is lost, repeated
   `cached_tokens < matched_tokens` observations on subsequent
   requests will dampen the worker's `eff_match` score, so traffic
   eventually shifts away from a stale slot.  Tier notifications
   reduce the staleness window from "until belief decays" (~10
   requests at α=0.2) to "one HTTP RTT" (~ms).
