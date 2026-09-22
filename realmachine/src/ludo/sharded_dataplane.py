"""
sharded_dataplane.py — edge-side per-shard DP mirror.

Holds ``n_shards`` independent :class:`DataPlaneLudoTable` instances (one
per shard, each with its own cuckoo placement seeded by the corresponding
authority) and demultiplexes by:

  * ``lookup_*(key)``  →  ``key % n_shards``
  * ``apply_delta(d)`` →  ``d.shard_id``

This mirrors Smash's "one Ludo per shard, one shard per master" topology
(see deps/smash/Smash/master_fn.h:40 and ``nShards = nMasters`` in
deps/smash/Smash/node.h:113). With ``n_shards == 1`` the wrapper degrades
to a single underlying DP and is wire-compatible with pre-sharding
deployments.

The wrapper exposes the same lookup/apply surface as
:class:`DataPlaneLudoTable` so :class:`LudoScheme` (which iterates per
level) can swap one in without touching its lookup loop.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from src.ludo.dataplane_table import DataPlaneLudoTable
from src.ludo.authority_table import Delta


class ShardedDataPlaneLudoTable:
    def __init__(self, worker_addrs: List[str], n_shards: int = 1):
        if n_shards < 1:
            raise ValueError(f"n_shards must be ≥ 1, got {n_shards}")
        self.n_shards: int = int(n_shards)
        self._worker_addrs: List[str] = list(worker_addrs)
        self._num_workers: int = len(worker_addrs)
        self._dps: List[DataPlaneLudoTable] = [
            DataPlaneLudoTable(worker_addrs=worker_addrs, shard_id=s)
            for s in range(self.n_shards)
        ]

    # ── Shard routing ──────────────────────────────────────────────────────────
    def shard_of(self, key: int) -> int:
        return key % self.n_shards

    # ── Lookup (per-shard demux) ───────────────────────────────────────────────
    def lookup_packed(self, key: int) -> int:
        return self._dps[self.shard_of(key)].lookup_packed(key)

    def lookup_key(self, key: int) -> List[Tuple[str, int]]:
        return self._dps[self.shard_of(key)].lookup_key(key)

    def unpack_holders(self, packed: int) -> List[Tuple[str, int]]:
        # All sub-DPs share the same worker layout; pick any.
        return self._dps[0].unpack_holders(packed)

    # ── Apply (uses delta.shard_id for demux) ──────────────────────────────────
    def apply_delta(self, delta: Delta) -> bool:
        sid = delta.shard_id
        if not (0 <= sid < self.n_shards):
            return False
        return self._dps[sid].apply_delta(delta)

    def apply_delta_batch(self, deltas: Iterable[Delta]) -> dict:
        received = applied = bad = 0
        for d in deltas:
            received += 1
            if self.apply_delta(d):
                applied += 1
            else:
                bad += 1
        return {"received": received, "applied": applied, "bad": bad}

    def apply_wire_batch(self, wire_deltas: Iterable[dict]) -> dict:
        return self.apply_delta_batch(Delta.from_wire(d) for d in wire_deltas)

    # ── Stats / introspection ──────────────────────────────────────────────────
    def size(self) -> int:
        return sum(dp.size() for dp in self._dps)

    def hit_rate(self) -> float:
        total = sum(dp._total_lookups for dp in self._dps)
        hits  = sum(dp._hit_count    for dp in self._dps)
        return hits / total if total else 0.0

    def stats(self) -> dict:
        per = [dp.stats() for dp in self._dps]
        return {
            "backend":           per[0]["backend"] if per else "empty",
            "n_shards":          self.n_shards,
            "buckets":           sum(p["buckets"] for p in per),
            "fallback_size":     sum(p["fallback_size"] for p in per),
            "total_lookups":     sum(p["total_lookups"] for p in per),
            "hits":              sum(p["hits"] for p in per),
            "hit_rate":          self.hit_rate(),
            "inserts_applied":   sum(p["inserts_applied"] for p in per),
            "updates_applied":   sum(p["updates_applied"] for p in per),
            "fallback_writes":   sum(p["fallback_writes"] for p in per),
            "snapshots_applied": sum(p["snapshots_applied"] for p in per),
            "bad_deltas":        sum(p["bad_deltas"] for p in per),
            "wrong_shard_drops": sum(p["wrong_shard_drops"] for p in per),
            "value_layout":      per[0]["value_layout"] if per else "",
            "num_workers":       self._num_workers,
            "per_shard":         per,
        }
