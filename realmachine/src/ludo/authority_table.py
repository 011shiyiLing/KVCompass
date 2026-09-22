"""
authority_table.py — central-directory single-writer table.

Wraps Smash's ControlPlaneLudo + a fallback dict + a `building` flag to
implement the authority half (Smash's "master" role) of the maintenance protocol:

  * fresh insert (key not yet in CP) → cuckoo path-entry sequence shipped
    to lookups via `Delta.kind == "insert"`;
  * existing-key value change       → single (bs, value) shipped via
    `Delta.kind == "update"`;
  * removal (packed bitmap → 0)     → `Delta.kind == "update"` with value=0
    + CP slot freed via `cp.remove`;
  * CP rebuild required (status<0)  → write parked in `fallback`, broadcast
    as `Delta.kind == "fallback"`; DiscoveryAuthority orchestrates the
    background rebuild and finishes with `merge_fallback_and_snapshot()`
    which yields `Delta.kind == "snapshot"` for lookups to swap in.

Per-slot value layout is unchanged: 16 worker slots × 2-bit tier each
(see ``ludo_table.py`` header).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from src.ludo.ludo_table import (
    TIER_NONE, TIER_VRAM, TIER_DRAM,
    _TIER_MASK, _BITS_PER_WORKER, _MAX_WORKERS,
    chunk_to_key, tier_name,
)

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


# ── Wire-level Delta ───────────────────────────────────────────────────────────
@dataclass
class Delta:
    """One authority→lookup wire payload. Mirrors Smash's Insert / Update /
    Insert(mode=1) / UpdateLudo message kinds.

    ``shard_id`` identifies which shard's mirror this Delta targets. The
    edge-side sharded DP wrapper demuxes on it; an edge whose DP is the
    legacy single-shard one can ignore it (defaults to 0)."""
    kind:  str               # "insert" | "update" | "fallback" | "snapshot" | "remove_fallback"
    value: int = 0           # packed value (0 means "clear" for kind=update)
    key:   int = 0           # only for fallback / remove_fallback
    bs:    int = 0           # only for update; bs = (bid<<2)|sid (DP slot)
    path:  list = field(default_factory=list)   # only for insert
    snapshot: bytes = b""    # only for snapshot
    shard_id: int = 0        # which shard's mirror this targets

    def to_wire(self) -> dict:
        d: dict = {"kind": self.kind, "shard": self.shard_id}
        if self.kind == "insert":
            d["value"] = self.value
            d["path"]  = self.path
        elif self.kind == "update":
            d["value"] = self.value
            d["bs"]    = self.bs
        elif self.kind == "fallback":
            d["value"] = self.value
            d["key"]   = self.key
        elif self.kind == "remove_fallback":
            d["key"]   = self.key
        elif self.kind == "snapshot":
            import base64
            d["snapshot_b64"] = base64.b64encode(self.snapshot).decode("ascii")
        return d

    @classmethod
    def from_wire(cls, d: dict) -> "Delta":
        kind = d["kind"]
        shard = int(d.get("shard", 0))
        if kind == "insert":
            return cls(kind=kind, value=int(d["value"]),
                       path=list(d.get("path", [])), shard_id=shard)
        if kind == "update":
            return cls(kind=kind, value=int(d["value"]),
                       bs=int(d["bs"]), shard_id=shard)
        if kind == "fallback":
            return cls(kind=kind, value=int(d["value"]),
                       key=int(d["key"]), shard_id=shard)
        if kind == "remove_fallback":
            return cls(kind=kind, key=int(d["key"]), shard_id=shard)
        if kind == "snapshot":
            import base64
            return cls(kind=kind,
                       snapshot=base64.b64decode(d["snapshot_b64"]),
                       shard_id=shard)
        raise ValueError(f"unknown Delta kind: {kind!r}")


@dataclass
class ApplyResult:
    """Returned by `AuthorityLudoTable.apply_slot_change`."""
    old_packed:    int
    new_packed:    int
    delta:         Optional[Delta] = None     # None if no-op
    triggered_rebuild: bool = False           # True iff this call flipped `building` to True


# ── AuthorityLudoTable ────────────────────────────────────────────────────────────
class AuthorityLudoTable:
    def __init__(
        self,
        worker_addrs: List[str],
        expected_entries: int = 4096,
        max_entries: Optional[int] = None,
        shard_id: int = 0,
        n_shards: int = 1,
        seed: Optional[int] = None,
    ):
        if len(worker_addrs) > _MAX_WORKERS:
            raise ValueError(
                f"AuthorityLudoTable supports at most {_MAX_WORKERS} workers; "
                f"got {len(worker_addrs)}."
            )
        if not (0 <= shard_id < max(1, n_shards)):
            raise ValueError(
                f"shard_id={shard_id} out of range for n_shards={n_shards}"
            )
        self._worker_addrs: List[str] = list(worker_addrs)
        self._addr_to_idx: dict[str, int] = {a: i for i, a in enumerate(worker_addrs)}
        self._idx_to_addr: dict[int, str] = {i: a for i, a in enumerate(worker_addrs)}
        self._num_workers: int = len(worker_addrs)
        self._max_entries: int = max_entries if max_entries is not None else expected_entries
        self.shard_id: int = shard_id
        self.n_shards: int = max(1, n_shards)

        # Per-shard seed: mirrors Smash's `ludo.setSeed(thisId * 0xe2211)`
        # (deps/smash/Smash/master_fn.h:168). Each shard's CP+DP gets an
        # independent cuckoo placement so that an edge holding N per-shard
        # DPs cannot suffer (bucket, slot) collisions across shards.
        if seed is None:
            seed = ((self.shard_id + 1) * 0xe2211) & 0xFFFFFFFF
        self._seed: int = int(seed) & 0xFFFFFFFF

        if _HAS_NATIVE:
            self._cp = _ext.LudoControlPlane(expected_entries, self._seed)
        else:
            self._fallback_cp: dict[int, int] = {}

        # Fallback table: writes parked here while `building` is True.
        # value==0 entries are tombstones (keys to be removed on merge).
        self.fallback: dict[int, int] = {}
        self.building: bool = False

        # Python-side shadow of all live (key, packed) pairs the CP holds,
        # plus current fallback contents. The C++ CP doesn't expose key
        # iteration, so persistence (snapshot dumps) and Step-4 rebuild
        # (`merge_fallback_and_snapshot`) need this to walk live state.
        self._live_state: dict[int, int] = {}

        self._inserts = 0
        self._updates = 0
        self._removes = 0
        self._fallback_writes = 0
        self._snapshot_count = 0
        self._total_lookups = 0
        self._hit_count = 0

    # ── Sharding helpers ───────────────────────────────────────────────────────
    def owns_key(self, key: int) -> bool:
        """True iff this shard is authoritative for ``key``. Matches the
        ``key % n_shards`` partition used by DiscoveryAuthority."""
        return (key % self.n_shards) == self.shard_id

    # ── Read API ───────────────────────────────────────────────────────────────
    def lookup_packed(self, key: int) -> int:
        self._total_lookups += 1
        if key in self.fallback:
            v = self.fallback[key]
            if v != 0:
                self._hit_count += 1
            return v
        v = self._cp_lookup(key) or 0
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

    # ── Write API ──────────────────────────────────────────────────────────────
    def apply_slot_change(
        self, key: int, worker_idx: int, tier: int,
    ) -> ApplyResult:
        if not (0 <= worker_idx < self._num_workers):
            raise ValueError(f"worker_idx {worker_idx} out of range")
        if tier not in (TIER_NONE, TIER_VRAM, TIER_DRAM):
            raise ValueError(f"invalid tier: {tier}")

        old = self._effective_packed(key)
        shift = _BITS_PER_WORKER * worker_idx
        slot_mask = _TIER_MASK << shift
        new = (old & ~slot_mask) | ((tier & _TIER_MASK) << shift)

        if old == new:
            return ApplyResult(old, new, None, False)

        # During rebuild, every write is parked in fallback regardless of kind.
        if self.building:
            result = self._fallback_write(key, new, old)
        elif old == 0:
            result = self._fresh_insert(key, new)
        elif new == 0:
            result = self._slot_clear_remove(key, old)
        else:
            result = self._value_change(key, new, old)

        if result.new_packed != 0:
            self._live_state[key] = result.new_packed
        else:
            self._live_state.pop(key, None)
        return result

    def iter_live(self) -> Iterator[Tuple[int, int]]:
        """Iterate `(key, packed)` for every live key currently in the
        authority's combined CP+fallback view. Used by the persistence
        snapshot dump and (in Step 4) by the rebuild orchestrator."""
        for k, v in self._live_state.items():
            if v != 0:
                yield k, v

    # ── Rebuild orchestration helpers ──────────────────────────────────────────
    def begin_rebuild(self) -> None:
        """Force `building=True` from the outside (e.g. background task is
        about to start). Idempotent."""
        self.building = True

    def merge_fallback_and_snapshot(
        self, target_capacity: Optional[int] = None,
    ) -> Delta:
        """Resize the CP, merge fallback writes into it, return a snapshot
        Delta to broadcast to lookups, and clear `building`. Called by the
        DiscoveryAuthority on the background rebuild task once it's safe."""
        if target_capacity is None:
            target_capacity = max(self._max_entries, self._cp_capacity() * 2)
        if _HAS_NATIVE:
            self._cp.resize_capacity(target_capacity, True)
            if self.fallback:
                self._cp.merge_dict(self.fallback)
            snapshot = self._cp.serialize_dp()
        else:
            for k, v in self.fallback.items():
                if v == 0:
                    self._fallback_cp.pop(k, None)
                else:
                    self._fallback_cp[k] = v
            snapshot = b""

        self.fallback.clear()
        self.building = False
        self._snapshot_count += 1
        return Delta(kind="snapshot", snapshot=snapshot, shard_id=self.shard_id)

    def serialize_dp(self) -> bytes:
        if _HAS_NATIVE:
            return self._cp.serialize_dp()
        return b""

    # ── Per-kind primitives ────────────────────────────────────────────────────
    def _fresh_insert(self, key: int, new: int) -> ApplyResult:
        if not _HAS_NATIVE:
            self._fallback_cp[key] = new
            self._inserts += 1
            return ApplyResult(0, new,
                               Delta(kind="update", value=new, bs=0,
                                     shard_id=self.shard_id),
                               False)

        result = self._cp.insert(key, new)
        status = int(result["status"])
        if status == 0:
            self._inserts += 1
            return ApplyResult(0, new,
                               Delta(kind="insert", value=new,
                                     path=list(result["path"]),
                                     shard_id=self.shard_id),
                               False)
        # status == -1 (othello rebuild) or -2 (cuckoo enlarge) — fallback the
        # write and signal to the authority that a background rebuild is needed.
        self.building = True
        self.fallback[key] = new
        self._fallback_writes += 1
        return ApplyResult(0, new,
                           Delta(kind="fallback", value=new, key=key,
                                 shard_id=self.shard_id),
                           True)

    def _value_change(self, key: int, new: int, old: int) -> ApplyResult:
        if not _HAS_NATIVE:
            self._fallback_cp[key] = new
            self._updates += 1
            return ApplyResult(old, new,
                               Delta(kind="update", value=new, bs=0,
                                     shard_id=self.shard_id),
                               False)

        cv = self._cp.change_value(key, new)
        if cv is None:
            # Key isn't in CP; treat as fresh insert (e.g. CP was just rebuilt
            # and the key was in fallback before).
            return self._fresh_insert(key, new)
        bs = int(cv["bs"])
        self._updates += 1
        return ApplyResult(old, new,
                           Delta(kind="update", value=new, bs=bs,
                                 shard_id=self.shard_id),
                           False)

    def _slot_clear_remove(self, key: int, old: int) -> ApplyResult:
        if not _HAS_NATIVE:
            self._fallback_cp.pop(key, None)
            self._removes += 1
            return ApplyResult(old, 0,
                               Delta(kind="update", value=0, bs=0,
                                     shard_id=self.shard_id),
                               False)

        cv = self._cp.change_value(key, 0)
        if cv is None:
            return ApplyResult(old, 0, None, False)
        bs = int(cv["bs"])
        self._cp.remove(key)
        self._removes += 1
        return ApplyResult(old, 0,
                           Delta(kind="update", value=0, bs=bs,
                                 shard_id=self.shard_id),
                           False)

    def _fallback_write(self, key: int, new: int, old: int) -> ApplyResult:
        self._fallback_writes += 1
        if new == 0:
            self.fallback[key] = 0
            return ApplyResult(old, 0,
                               Delta(kind="remove_fallback", key=key,
                                     shard_id=self.shard_id),
                               False)
        self.fallback[key] = new
        return ApplyResult(old, new,
                           Delta(kind="fallback", value=new, key=key,
                                 shard_id=self.shard_id),
                           False)

    # ── Internal helpers ───────────────────────────────────────────────────────
    def _effective_packed(self, key: int) -> int:
        if key in self.fallback:
            return self.fallback[key]
        return self._cp_lookup(key) or 0

    def _cp_lookup(self, key: int) -> Optional[int]:
        if _HAS_NATIVE:
            return self._cp.lookup(key)
        return self._fallback_cp.get(key)

    def _cp_capacity(self) -> int:
        if _HAS_NATIVE:
            return self._cp.capacity()
        return max(self._max_entries, len(self._fallback_cp))

    # ── Legacy back-compat (removed in Step 4 protocol cutover) ────────────────
    def set_slot(
        self, key: int, worker_idx: int, tier: int,
    ) -> Tuple[int, int]:
        """Legacy: returns (old_packed, new_packed). Use `apply_slot_change`
        for the new path-entry-aware Delta-returning API."""
        r = self.apply_slot_change(key, worker_idx, tier)
        return r.old_packed, r.new_packed

    def apply_packed(self, key: int, packed_value: int) -> None:
        """Legacy lookup-mirror primitive: authority pushed `(key, packed_value)`,
        mirror it locally without going through the new Delta machinery.
        DataPlaneLudoTable.apply_delta will replace this in Step 4."""
        if not _HAS_NATIVE:
            if packed_value == 0:
                self._fallback_cp.pop(key, None)
            else:
                self._fallback_cp[key] = packed_value
        elif packed_value == 0:
            if self._cp.lookup(key) is not None:
                self._cp.remove(key)
        elif self._cp.lookup(key) is None:
            res = self._cp.insert(key, packed_value)
            if int(res["status"]) < 0:
                # CP needs rebuild; park in fallback so a future lookup still
                # sees the value. The lookup-side never owns rebuild state, so
                # we don't flip `building` — we just rely on the next authority
                # snapshot to clean this up.
                self.fallback[key] = packed_value
        else:
            self._cp.change_value(key, packed_value)

        if packed_value != 0:
            self._live_state[key] = packed_value
        else:
            self._live_state.pop(key, None)

    @property
    def _lru(self) -> dict:
        """Legacy alias used by `discovery_authority.on_leave` to enumerate
        currently-known keys for worker-clear scans. Returns a dict view of
        keys we have records for; values are unused by callers."""
        return {}   # native CP doesn't expose key iteration; on_leave needs Step 4 redesign

    # ── Address ↔ idx ──────────────────────────────────────────────────────────
    def addr_to_idx(self, addr: str) -> int:
        return self._addr_to_idx[addr]

    def idx_to_addr(self, idx: int) -> str:
        return self._idx_to_addr[idx]

    @property
    def num_workers(self) -> int:
        return self._num_workers

    # ── Stats ──────────────────────────────────────────────────────────────────
    def size(self) -> int:
        if _HAS_NATIVE:
            return int(self._cp.size())
        return len(self._fallback_cp)

    def hit_rate(self) -> float:
        return self._hit_count / self._total_lookups if self._total_lookups else 0.0

    def stats(self) -> dict:
        return {
            "backend":          "native_ludo" if _HAS_NATIVE else "python_dict",
            "shard_id":         self.shard_id,
            "n_shards":         self.n_shards,
            "seed":             self._seed,
            "entries":          self.size(),
            "fallback_size":    len(self.fallback),
            "building":         self.building,
            "max_entries":      self._max_entries,
            "total_lookups":    self._total_lookups,
            "hits":             self._hit_count,
            "hit_rate":         self.hit_rate(),
            "inserts":          self._inserts,
            "updates":          self._updates,
            "removes":          self._removes,
            "fallback_writes":  self._fallback_writes,
            "snapshots":        self._snapshot_count,
            "value_layout":     "16 workers × 2-bit tier (VL=32)",
            "lww_enabled":      False,
            "num_workers":      self._num_workers,
        }
