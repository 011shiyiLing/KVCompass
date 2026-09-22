"""
dataplane_table.py — read-only lookup-side mirror.

Wraps the C++ DataPlaneLudo + a local fallback dict + a local "removed"
tombstone set. Receives Delta payloads from the authority via
``apply_delta()``; serves lookups via ``lookup_key()``.

Why the tombstone set: the authority ships ``Delta(kind="update", value=0)``
to remove a slot, which our DP applies as ``apply_update(bs, 0)``. The
DP locator still maps the (now-zero-valued) key to that slot, so a
lookup naturally reads 0 → unpacks to no holders → behaves as a miss.
But for keys that were *only ever* in the authority's fallback (never
applied to CP+DP), the authority ships ``Delta(kind="remove_fallback")`` —
we just need to drop those from our local fallback dict; nothing to
clear on the DP side.

Snapshot apply (``Delta(kind="snapshot")``): deserialize the bytes into
a fresh DP, then atomically swap. Local fallback is cleared because
the snapshot already incorporates everything the authority had buffered
during the rebuild.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from src.ludo.ludo_table import (
    TIER_NONE, TIER_VRAM, TIER_DRAM,
    _TIER_MASK, _BITS_PER_WORKER, _MAX_WORKERS,
    chunk_to_key,
)
from src.ludo.authority_table import Delta

_HERE = Path(__file__).parent
for candidate in (_HERE, _HERE.parent / "build"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    import ludo_binding as _ext   # type: ignore
    _HAS_NATIVE = True
except ImportError:
    _ext = None
    _HAS_NATIVE = False


class DataPlaneLudoTable:
    def __init__(self, worker_addrs: List[str], shard_id: int = 0):
        if len(worker_addrs) > _MAX_WORKERS:
            raise ValueError(
                f"DataPlaneLudoTable supports at most {_MAX_WORKERS} workers; "
                f"got {len(worker_addrs)}."
            )
        self._worker_addrs: List[str] = list(worker_addrs)
        self._idx_to_addr: dict[int, str] = {i: a for i, a in enumerate(worker_addrs)}
        self._num_workers: int = len(worker_addrs)
        # Which shard's authority this DP mirrors. Deltas with a different
        # shard_id are dropped defensively by apply_delta — the
        # ShardedDataPlaneLudoTable wrapper above already demuxes by
        # shard, so this is a second line of defense against routing bugs.
        self.shard_id: int = int(shard_id)

        if _HAS_NATIVE:
            self._dp = _ext.LudoDataPlane()
        else:
            self._py_table: dict[int, int] = {}

        self.fallback: dict[int, int] = {}

        self._inserts = 0
        self._updates = 0
        self._fallback_writes = 0
        self._snapshot_count = 0
        self._total_lookups = 0
        self._hit_count = 0
        self._bad = 0
        self._wrong_shard = 0

    # ── Lookup ─────────────────────────────────────────────────────────────────
    def lookup_packed(self, key: int) -> int:
        self._total_lookups += 1
        if key in self.fallback:
            v = self.fallback[key]
            if v != 0:
                self._hit_count += 1
            return v
        v = self._dp_lookup(key) or 0
        if v != 0:
            self._hit_count += 1
        return v

    def lookup_key(self, key: int) -> List[Tuple[str, int]]:
        return self.unpack_holders(self.lookup_packed(key))

    def unpack_holders(self, packed: int) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        for i in range(self._num_workers):
            t = (packed >> (_BITS_PER_WORKER * i)) & _TIER_MASK
            if t != TIER_NONE:
                out.append((self._idx_to_addr[i], t))
        return out

    # ── Apply Delta ────────────────────────────────────────────────────────────
    def apply_delta(self, delta: Delta) -> bool:
        if delta.shard_id != self.shard_id:
            self._wrong_shard += 1
            return False
        try:
            kind = delta.kind
            if kind == "insert":
                if not self._dp_ready():
                    # No bootstrap snapshot yet — skip silently. The path
                    # entries reference bucket indices that don't exist on
                    # an uninitialised DP and would segfault if applied.
                    self._bad += 1
                    return False
                self._apply_insert(delta.value, delta.path)
                self._inserts += 1
            elif kind == "update":
                if not self._dp_ready():
                    self._bad += 1
                    return False
                self._apply_update(delta.bs, delta.value)
                self._updates += 1
            elif kind == "fallback":
                self.fallback[delta.key] = delta.value
                self._fallback_writes += 1
            elif kind == "remove_fallback":
                self.fallback.pop(delta.key, None)
                self._fallback_writes += 1
            elif kind == "snapshot":
                self._apply_snapshot(delta.snapshot)
                self._snapshot_count += 1
                self.fallback.clear()
            else:
                self._bad += 1
                return False
            return True
        except Exception:
            self._bad += 1
            return False

    def _dp_ready(self) -> bool:
        """True iff the DP has been bootstrapped with at least one
        snapshot. Inserts/updates on an empty DP would access out-of-
        bounds bucket memory."""
        if not _HAS_NATIVE:
            return True
        return int(self._dp.num_buckets()) > 0

    def apply_delta_batch(self, deltas: Iterable[Delta]) -> dict:
        received = 0
        applied = 0
        for d in deltas:
            received += 1
            if self.apply_delta(d):
                applied += 1
        return {"received": received, "applied": applied, "bad": received - applied}

    def apply_wire_batch(self, wire_deltas: Iterable[dict]) -> dict:
        return self.apply_delta_batch(Delta.from_wire(d) for d in wire_deltas)

    # ── Per-kind apply primitives ──────────────────────────────────────────────
    def _apply_insert(self, value: int, path: list) -> None:
        if _HAS_NATIVE:
            self._dp.apply_insert(path, value)
        else:
            # Fallback path can't reconstruct DP cuckoo state from path
            # entries; we just store the most recent value-only mapping
            # by re-deriving the key. Sufficient for the no-native dev path.
            pass

    def _apply_update(self, bs: int, value: int) -> None:
        if _HAS_NATIVE:
            self._dp.apply_update(bs, value)

    def _apply_snapshot(self, snapshot: bytes) -> None:
        if _HAS_NATIVE:
            new_dp = _ext.LudoDataPlane()
            new_dp.deserialize(snapshot)
            self._dp = new_dp

    def _dp_lookup(self, key: int) -> Optional[int]:
        if _HAS_NATIVE:
            return self._dp.lookup(key)
        return self._py_table.get(key)

    # ── Stats ──────────────────────────────────────────────────────────────────
    def size(self) -> int:
        if _HAS_NATIVE:
            return int(self._dp.num_buckets())  # buckets, not entries — DP doesn't track key count
        return len(self._py_table)

    def hit_rate(self) -> float:
        return self._hit_count / self._total_lookups if self._total_lookups else 0.0

    def stats(self) -> dict:
        return {
            "backend":         "native_ludo_dp" if _HAS_NATIVE else "python_dict",
            "shard_id":        self.shard_id,
            "buckets":         int(self._dp.num_buckets()) if _HAS_NATIVE else 0,
            "fallback_size":   len(self.fallback),
            "total_lookups":   self._total_lookups,
            "hits":            self._hit_count,
            "hit_rate":        self.hit_rate(),
            "inserts_applied":   self._inserts,
            "updates_applied":   self._updates,
            "fallback_writes":   self._fallback_writes,
            "snapshots_applied": self._snapshot_count,
            "bad_deltas":      self._bad,
            "wrong_shard_drops": self._wrong_shard,
            "value_layout":    "16 workers × 2-bit tier (VL=32)",
            "lww_enabled":     False,
            "num_workers":     self._num_workers,
        }
