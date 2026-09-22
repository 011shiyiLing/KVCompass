"""
Wire format — Smash-style maintenance protocol.

Two endpoints carry all maintenance traffic:

  Worker → Authority :  POST /events_batch   body = ``EventBatchMsg``
  Authority → Lookup :  POST /internal/apply_delta_batch   body = ``DeltaBatchMsg``

The batched shape is a deliberate departure from Smash's one-message-per-
event protocol: a single SGLang prefill produces L events (one per
chunk-size level) and a single worker eviction call can produce many more.
Batching cuts the per-event HTTP overhead from O(events) to O(batches).
The semantics within a batch are still per-event: authority applies them in
order under one ``update_lock`` and emits one ``DeltaBatchMsg`` to each
lookup with all generated deltas (mixing Insert / Update / Remove /
Snapshot kinds) preserved in apply order.

Event names — worker side:
    advertise   prefill just finished              tier = VRAM
    demote      VRAM → DRAM (write_through)        tier = DRAM
    promote     DRAM → VRAM (load_from_host)       tier = VRAM
    evict       DRAM → gone                        tier = NONE

Delta kinds — authority side (see ``src/ludo/authority_table.py:Delta`` for
the producer; ``src/ludo/dataplane_table.py:apply_delta`` for consumer):
    insert         {value, path}              cuckoo move-chain to apply
    update         {value, bs}                single-slot value change
    fallback       {value, key}               buffered during rebuild
    remove_fallback{key}                      drop key from lookup-side fallback
    snapshot       {snapshot_b64}             full DP rebuild result
"""
from __future__ import annotations

from typing import Literal, TypedDict


# ── Event names ─────────────────────────────────────────────────────────────
TIER_EVENT_NAMES = ("demote", "promote", "evict")
TierEventName = Literal["demote", "promote", "evict"]

EVENT_NAMES = ("advertise", "demote", "promote", "evict")
EventName = Literal["advertise", "demote", "promote", "evict"]


# ── Worker → Authority ──────────────────────────────────────────────────────────
class WorkerEvent(TypedDict):
    event:     EventName
    token_ids: list[int]
    seq:       int               # per-worker monotonic; authority dedups


class EventBatchMsg(TypedDict):
    """One batched POST from worker N to its assigned authority shard.

    All events share the same ``rank``; the authority de-dups out-of-order
    deliveries via the per-event ``seq`` (which is monotone within a worker
    process)."""
    rank:   int
    events: list[WorkerEvent]


# ── Authority → Lookup ──────────────────────────────────────────────────────────
class DeltaWire(TypedDict, total=False):
    """One delta as it appears on the wire. ``kind`` selects which other
    fields are populated; see Delta.to_wire/Delta.from_wire."""
    kind:           str        # one of {insert, update, fallback, remove_fallback, snapshot}
    level:          int        # which chunk-size level this delta applies to
    value:          int        # packed value (insert / update / fallback)
    key:            int        # key (fallback / remove_fallback)
    bs:             int        # bucket-slot index (update)
    path:           list       # cuckoo-move chain (insert)
    snapshot_b64:   str        # base64-encoded DP snapshot bytes


class DeltaBatchMsg(TypedDict):
    """One batched broadcast from authority to each lookup. Deltas appear in
    the order the authority applied them; the lookup must apply them in the
    same order so cuckoo-move-chain semantics are preserved when an Insert
    follows an Update on the same bucket."""
    deltas: list[DeltaWire]


# ── Legacy (kept for back-compat through Step 4 transition) ─────────────────
class AdvertiseMsg(TypedDict):
    rank:      int
    token_ids: list[int]
    seq:       int


class TierEventMsg(TypedDict):
    rank:      int
    token_ids: list[int]
    event:     TierEventName
    seq:       int


class ApplyPackedMsg(TypedDict):
    """Pre-Step-4 wire format (legacy): kept so the existing
    ``/internal/apply_packed`` endpoint continues to work in single-mode
    deployments that haven't migrated to the Delta protocol yet."""
    level:        int
    key:          int
    packed_value: int


# ── NameServer ↔ Everyone (unchanged) ────────────────────────────────────────
class RegisterAuthorityMsg(TypedDict):
    addr:    str
    shard_id: int

class RegisterLookupMsg(TypedDict):
    addr:      str
    lookup_id: int

class RegisterWorkerMsg(TypedDict):
    addr:    str
    rank:    int
    region:  int

class HeartbeatMsg(TypedDict):
    rank: int

class TopologyMsg(TypedDict):
    authorities: list[dict]
    lookups: list[dict]
    workers: list[dict]
    n_shards: int
    lookup_for_shard: list[list[int]]

class LeaveMsg(TypedDict):
    worker_id: int
