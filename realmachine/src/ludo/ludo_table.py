"""
Python wrapper around the Ludo MinimalPerfectCuckoo C++ extension.

Value encoding — 2 bits per worker (tier-aware)
------------------------------------------------
Each Ludo entry stores a uint32 value packed as 16 worker slots × 2 bits each.
Each 2-bit slot encodes that worker's tier state for the chunk:

    0b00 = TIER_NONE   absent
    0b01 = TIER_VRAM   present in GPU HBM (immediately usable)
    0b10 = TIER_DRAM   present in CPU RAM (PCIe reload before use)
    0b11 = (reserved)  future use — e.g. NVMe tier / in-flight transfer

Bit layout (worker i occupies bits 2i..2i+1):

    bit  31 30 29 28 ...  5  4  3  2  1  0
         ──┬── ──┬──      ──┬── ──┬── ──┬──
          w15   w14         w2   w1   w0

The C++ Ludo table stores a single fixed-width value per entry.  Two
compiled variants exist (same source, -DLUDO_VALUE_BITS):

    ludo_binding    V=uint32, VL=32 → 16 worker slots
    ludo_binding64  V=uint64, VL=64 → 32 worker slots

LudoTable picks the narrowest variant that fits its worker count.  Beyond
32 workers, partition the fleet (e.g. one table per region / per ≤32-worker
shard) and query each partition — do not widen further.

LRU policy
----------
An OrderedDict (_lru) shadows every entry, ordered oldest → newest,
storing the same uint32 value the C++ table holds.  All LRU operations
are O(1) via OrderedDict.move_to_end / popitem.

Public API
----------
  insert(token_ids, worker, tier=TIER_VRAM)
        Mark `worker` as holding `token_ids` at `tier`.  If the worker
        already has a tier in the slot, it is overwritten (not OR'd) — the
        slot is the worker's authoritative current state.

  lookup(token_ids) -> List[(worker_addr, tier_int)]
        Return all workers currently in the entry, paired with their
        tier.  Promotes the entry to MRU on a hit.

  remove(token_ids) -> bool
        Drop the entire entry (all workers forgotten).

  remove_worker(token_ids, worker) -> bool
        Clear `worker`'s 2-bit slot.  If no workers remain (mask becomes
        0), the entry is dropped from both the LRU shadow and the C++
        table.  Returns True if the entry existed.

  update_tier(token_ids, worker, new_tier) -> bool
        Set `worker`'s 2-bit slot to `new_tier`.  Same effect as
        insert(...) — kept as a separate name for code-site clarity when
        the call site is "tier transition" rather than "fresh advertise".
"""

import hashlib
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

# ── Locate the built extension ─────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

try:
    import ludo_binding as _ext
except ImportError:
    _ext = None

try:
    import ludo_binding64 as _ext64
except ImportError:
    _ext64 = None

_HAS_NATIVE = _ext is not None or _ext64 is not None
if not _HAS_NATIVE:
    import warnings
    warnings.warn(
        "ludo_binding native extension not found. "
        "Run scripts/setup/build_ludo.sh to compile it. "
        "Falling back to Python dict (same interface, no performance guarantees).",
        stacklevel=2,
    )


# ── Tier codes ─────────────────────────────────────────────────────────────────
TIER_NONE: int = 0b00
TIER_VRAM: int = 0b01
TIER_DRAM: int = 0b10
# 0b11 is reserved for future tiers (e.g. NVMe, in-flight transfer)

_TIER_MASK: int = 0b11   # mask for one worker's 2-bit slot
_BITS_PER_WORKER: int = 2
_MAX_WORKERS_32: int = 16   # uint32 / 2 bits = 16 slots (ludo_binding)
_MAX_WORKERS_64: int = 32   # uint64 / 2 bits = 32 slots (ludo_binding64)
# Central-mode authority/dataplane tables (authority_table.py,
# dataplane_table.py) pack into the uint32 C++ DataPlane value, so their
# worker cap stays at the 32-bit layout's 16 slots.
_MAX_WORKERS: int = _MAX_WORKERS_32


def chunk_to_key(token_ids: List[int]) -> int:
    """
    Hash a chunk of token IDs to a uint64 Ludo key.

    Lower 63 bits of SHA-256; MSB cleared to avoid the Ludo sentinel value.
    """
    digest = hashlib.sha256(str(tuple(token_ids)).encode()).digest()
    raw = int.from_bytes(digest[:8], "little")
    return raw & 0x7FFF_FFFF_FFFF_FFFF


def tier_name(tier: int) -> str:
    """Stringify a tier code for logs / stats."""
    return {
        TIER_NONE: "NONE",
        TIER_VRAM: "VRAM",
        TIER_DRAM: "DRAM",
    }.get(tier, f"reserved({tier:#04b})")


class LudoTable:
    """
    Routing table: chunk_key → packed (worker_idx, tier) for up to 16 workers.

    Parameters
    ----------
    worker_addrs     : ordered list of worker address strings; position = bit index
    expected_entries : C++ Ludo pre-allocation hint
    max_entries      : LRU capacity; defaults to expected_entries
    router_id        : non-negative integer that uniquely identifies this
                       router process within the multi-edge fleet; used as
                       the second LWW tie-breaker when two routers' deltas
                       carry the same epoch.  Default 0 — fine for a
                       single-router deployment.

    LWW conflict resolution
    -----------------------
    Each per-worker 2-bit slot is also versioned by an ``(epoch, source_id)``
    pair maintained in ``self._slot_epoch[key][worker_idx]``:

        * ``epoch``     — local monotonic counter at the moment of the
                          write (router's own writes) or whatever epoch the
                          incoming gossip delta carried (peer writes).
        * ``source_id`` — the router's own ``router_id`` for local writes,
                          or the peer's ``source_id`` for gossip-applied
                          writes.

    On every write (local or remote) we compare ``(incoming_epoch,
    incoming_source_id)`` lexicographically against the slot's stored
    pair; the larger pair wins.  A write whose pair is ≤ the stored pair
    is dropped.  This makes the table a CRDT-style LWW register at the
    granularity of (chunk_key, worker_idx) — convergent under arbitrary
    delivery order and idempotent under duplicate gossip.
    """

    def __init__(
        self,
        worker_addrs: List[str],
        expected_entries: int = 4096,
        max_entries: Optional[int] = None,
        router_id: int = 0,
    ):
        if len(worker_addrs) > _MAX_WORKERS_64:
            raise ValueError(
                f"LudoTable supports at most {_MAX_WORKERS_64} workers "
                f"(uint64 value, 2 bits per worker); got {len(worker_addrs)}. "
                "Partition the fleet (e.g. one table per region / per "
                "≤32-worker shard) and query each partition."
            )
        if router_id < 0:
            raise ValueError(f"router_id must be >= 0, got {router_id}")
        self._worker_addrs: List[str] = list(worker_addrs)
        self._addr_to_idx: dict[str, int] = {a: i for i, a in enumerate(worker_addrs)}
        self._idx_to_addr: dict[int, str] = {i: a for i, a in enumerate(worker_addrs)}
        self._num_workers: int = len(worker_addrs)
        self._router_id: int = int(router_id)

        # Backend selection: the uint32 module packs ≤16 workers, the
        # uint64 module ≤32.  Prefer the narrowest module that fits (wire-
        # compatible with existing ≤16-worker deployments); fall back to
        # the wider one when only it is built.
        if self._num_workers <= _MAX_WORKERS_32 and _ext is not None:
            backend = _ext
            self._value_bits = 32
        elif _ext64 is not None:
            backend = _ext64
            self._value_bits = 64
        elif _ext is not None:
            raise ValueError(
                f"{len(worker_addrs)} workers need the uint64 binding "
                "(ludo_binding64); only the 16-worker ludo_binding is "
                "built. Re-run src/ludo/setup.py build_ext --inplace."
            )
        else:
            backend = None
            self._value_bits = 64

        self._native = backend is not None
        self._backend = backend
        if self._native:
            self._cp = backend.LudoControlPlane(expected_entries)
        else:
            self._fallback: dict[int, int] = {}   # key → packed value

        # LRU shadow: key → packed value, ordered by recency
        self._lru: OrderedDict[int, int] = OrderedDict()
        self._max_entries: int = (
            max_entries if max_entries is not None else expected_entries
        )

        # Per-slot LWW timestamp: key → {worker_idx → (epoch, source_id)}.
        # Only slots that have ever been written get an entry; cleaned up
        # alongside the LRU shadow when an entry is dropped.
        self._slot_epoch: dict[int, dict[int, Tuple[int, int]]] = {}

        # Local monotonic counter — every local write bumps this and uses
        # (epoch, self._router_id) as the LWW pair.
        self._epoch_counter: int = 0

        # Statistics
        self._total_lookups: int = 0
        self._hit_count: int = 0
        self._evict_count: int = 0
        self._gossip_applied: int = 0
        self._gossip_dropped_stale: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def insert(
        self,
        token_ids: List[int],
        worker_addr: str,
        tier: int = TIER_VRAM,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Record that `worker_addr` holds this chunk at `tier`.

        Overwrites the worker's 2-bit slot (does NOT OR — a tier is the
        worker's authoritative current state, not an additive flag).  If
        the entry is new and the LRU shadow is at capacity, the LRU entry
        is evicted first.

        Returns a delta tuple ``(key, worker_idx, tier, epoch)`` for the
        gossip layer to broadcast, or ``None`` if the write was dropped
        by the LWW check (only happens on `apply_remote`, never here, so
        local writes always return the delta).
        """
        if tier not in (TIER_VRAM, TIER_DRAM):
            raise ValueError(
                f"insert() tier must be TIER_VRAM or TIER_DRAM; got {tier!r}. "
                "Use remove_worker() to clear a worker's slot."
            )
        key = chunk_to_key(token_ids)
        self._epoch_counter += 1
        epoch = self._epoch_counter
        applied = self._set_worker_tier_lww(
            key, worker_addr, tier, epoch, self._router_id,
        )
        if not applied:
            return None
        return key, self._addr_to_idx[worker_addr], tier, epoch

    # Alias for code-site clarity when the call is a tier transition rather
    # than a fresh advertise.  Same implementation either way.
    update_tier = insert

    def insert_key(
        self,
        key: int,
        worker_addr: str,
        tier: int = TIER_VRAM,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Same as insert() but takes a precomputed uint64 Ludo key — for
        callers that already hold content hashes (simulator, eviction
        notifier) and must not pay the SHA-256 step per write.
        """
        if tier not in (TIER_VRAM, TIER_DRAM):
            raise ValueError(
                f"insert_key() tier must be TIER_VRAM or TIER_DRAM; got {tier!r}. "
                "Use remove_worker_key() to clear a worker's slot."
            )
        self._epoch_counter += 1
        epoch = self._epoch_counter
        applied = self._set_worker_tier_lww(
            key, worker_addr, tier, epoch, self._router_id,
        )
        if not applied:
            return None
        return key, self._addr_to_idx[worker_addr], tier, epoch

    update_tier_key = insert_key

    def lookup(self, token_ids: List[int]) -> List[Tuple[str, int]]:
        """
        Return [(worker_addr, tier_int)] for all workers that hold this chunk.

        Returns an empty list on a miss.  Promotes the entry to MRU on a hit.
        """
        self._total_lookups += 1
        key = chunk_to_key(token_ids)
        return self._lookup_key_internal(key)

    def lookup_key(self, key: int) -> List[Tuple[str, int]]:
        """Lookup by raw uint64 key (avoids re-hashing when key is precomputed)."""
        self._total_lookups += 1
        return self._lookup_key_internal(key)

    def remove(self, token_ids: List[int]) -> bool:
        """
        Remove the routing entry entirely (all workers forgotten).

        Call when a chunk has been evicted from every worker.
        """
        key = chunk_to_key(token_ids)
        return self._remove_key(key)

    def remove_worker(
        self, token_ids: List[int], worker_addr: str,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Clear `worker_addr`'s 2-bit slot for this chunk.

        If no workers remain in the entry, the entry is dropped from both
        the LRU shadow and the C++ table.

        Returns a delta tuple ``(key, worker_idx, TIER_NONE, epoch)`` for
        the gossip layer to broadcast, or ``None`` if the entry didn't
        exist (no-op).
        """
        key = chunk_to_key(token_ids)
        if key not in self._lru:
            return None
        self._epoch_counter += 1
        epoch = self._epoch_counter
        applied = self._set_worker_tier_lww(
            key, worker_addr, TIER_NONE, epoch, self._router_id,
        )
        if not applied:
            return None
        return key, self._addr_to_idx[worker_addr], TIER_NONE, epoch

    def remove_worker_key(
        self, key: int, worker_addr: str,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Same as remove_worker but takes a precomputed uint64 Ludo key —
        used by the worker-side eviction notifier (Path B) to skip the
        SHA-256 step on the hot path.
        """
        if key not in self._lru:
            return None
        self._epoch_counter += 1
        epoch = self._epoch_counter
        applied = self._set_worker_tier_lww(
            key, worker_addr, TIER_NONE, epoch, self._router_id,
        )
        if not applied:
            return None
        return key, self._addr_to_idx[worker_addr], TIER_NONE, epoch

    # ── Gossip apply (peer-write inbound) ──────────────────────────────────────

    def apply_remote(
        self,
        key: int,
        worker_idx: int,
        tier: int,
        epoch: int,
        source_id: int,
    ) -> bool:
        """
        Apply a peer-router gossip delta with explicit ``(epoch, source_id)``
        version.  LWW: if the incoming pair beats the locally-stored slot
        version, the slot is updated; otherwise the delta is dropped as
        stale or duplicate.

        Returns True iff the LWW gate accepted the delta (regardless of
        whether the visible LudoTable state changed — e.g. a remove on a
        nonexistent slot accepts the delta but produces no observable
        state change, but it DOES record the slot's epoch so a future
        stale add is correctly rejected).

        Notes
        -----
        * ``source_id`` MUST NOT equal ``self._router_id`` — gossip should
          never echo a router its own writes.  We do not enforce this
          here (it's a soft contract on the gossip layer).
        * Lamport clock: ``apply_remote`` bumps
          ``self._epoch_counter = max(self._epoch_counter, epoch)`` so
          subsequent local writes always carry an epoch greater than
          anything this router has ever seen — guaranteeing local writes
          are never themselves rejected by the LWW gate against
          previously-applied gossip.
        """
        if not (0 <= worker_idx < self._num_workers):
            return False
        if tier not in (TIER_NONE, TIER_VRAM, TIER_DRAM):
            return False

        # Lamport-clock bump: keep local epoch monotonically ahead of
        # everything we've ever seen.
        if epoch > self._epoch_counter:
            self._epoch_counter = epoch

        worker_addr = self._idx_to_addr[worker_idx]
        accepted = self._set_worker_tier_lww(
            key, worker_addr, tier, epoch, source_id,
        )
        if accepted:
            self._gossip_applied += 1
        else:
            self._gossip_dropped_stale += 1
        return accepted

    # ── Statistics ─────────────────────────────────────────────────────────────

    def size(self) -> int:
        return len(self._lru)

    def hit_rate(self) -> float:
        if self._total_lookups == 0:
            return 0.0
        return self._hit_count / self._total_lookups

    def stats(self) -> dict:
        # Count entries with more than one holder.
        multi = sum(
            1 for value in self._lru.values()
            if self._holder_count(value) > 1
        )
        # Count entries currently DRAM-only (every slot is DRAM, no VRAM).
        dram_only = sum(
            1 for value in self._lru.values()
            if self._is_dram_only(value)
        )
        return {
            "backend": "native_ludo" if self._native else "python_dict",
            "entries": self.size(),
            "max_entries": self._max_entries,
            "utilization": self.size() / max(self._max_entries, 1),
            "multi_worker_entries": multi,
            "dram_only_entries": dram_only,
            "total_lookups": self._total_lookups,
            "hits": self._hit_count,
            "hit_rate": self.hit_rate(),
            "lru_evictions": self._evict_count,
            "value_layout": (
                f"{_MAX_WORKERS_32 if self._value_bits == 32 else _MAX_WORKERS_64}"
                f" workers × 2-bit tier (VL={self._value_bits})"
            ),
            "router_id": self._router_id,
            "epoch_counter": self._epoch_counter,
            "slot_epoch_entries": len(self._slot_epoch),
            "gossip_applied": self._gossip_applied,
            "gossip_dropped_stale": self._gossip_dropped_stale,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _worker_shift(self, worker_addr: str) -> int:
        """Bit-shift amount for `worker_addr`'s 2-bit slot."""
        return _BITS_PER_WORKER * self._addr_to_idx[worker_addr]

    def _read_value(self, key: int) -> int:
        """Read packed value for `key`; 0 if absent."""
        if self._native:
            v = self._cp.lookup(key)
            return 0 if v is None else int(v)
        return self._fallback.get(key, 0)

    def _write_value(self, key: int, value: int, *, fresh: bool) -> None:
        """Write packed value into the C++ table (or fallback dict).

        A fresh insert can come back with status < 0 (-1 Othello locator
        needs rebuild, -2 cuckoo needs enlarging).  The authority path
        handles that with fallback + async rebuild; here (gossip-mode,
        single writer) we do the blocking equivalent: rebuild a FRESH C++
        table at double capacity from the LRU shadow (the shadow holds the
        complete key→value state).  We deliberately avoid the CP's in-place
        resize_capacity/powerSize enlarge: its incremental key migration
        corrupts the table (duplicate keys in one bucket → "Cannot generate
        a proper hash seed" on later inserts).  Ignoring the status
        silently drops the write (the pre-fix behaviour).
        """
        if self._native:
            if fresh:
                r = self._cp.insert(key, value)
                if r["status"] < 0:
                    # Rebuild replays the shadow PLUS this pending write, so
                    # after it returns the key is guaranteed present.
                    self._rebuild_native(extra=(key, value))
            else:
                self._cp.update(key, value)
        else:
            self._fallback[key] = value

    def _rebuild_native(self, extra: Optional[Tuple[int, int]] = None) -> None:
        """Rebuild the C++ CP from scratch off the LRU shadow (plus the
        pending write that triggered the rebuild), doubling capacity per
        attempt until every entry replays cleanly.

        Each attempt uses a DIFFERENT locator seed: an Othello online-insert
        failure is deterministic in (key set, seed), so retrying the same
        seed after a same-size rebuild would fail the same key forever."""
        items = list(self._lru.items())
        if extra is not None:
            items.append(extra)
        target = max(2 * len(items) + 1024, 4096)
        for attempt in range(6):
            seed = (0x12345678 ^ (0x9E3779B9 * (attempt + 1))) & 0xFFFFFFFF
            cp = self._backend.LudoControlPlane(target, seed)
            for k, v in items:
                if cp.insert(k, v)["status"] < 0:
                    break
            else:
                self._cp = cp
                return
            target *= 2
        raise RuntimeError(
            f"Ludo rebuild failed at {len(items)} entries "
            f"(last capacity {target // 2})"
        )

    def _set_worker_tier_lww(
        self,
        key: int,
        worker_addr: str,
        tier: int,
        incoming_epoch: int,
        incoming_source_id: int,
    ) -> bool:
        """
        Read-modify-write the worker's 2-bit slot to ``tier``, gated by
        a per-slot LWW comparison against ``(incoming_epoch,
        incoming_source_id)``.

        The slot is updated iff ``(incoming_epoch, incoming_source_id)``
        is lexicographically greater than the slot's stored
        ``(epoch, source_id)`` (or the slot has never been written).
        Otherwise the call is a no-op and returns False.

        Side effects on success:
          * the C++ Ludo + LRU shadow are updated as before
          * ``self._slot_epoch[key][worker_idx]`` is set to the
            incoming pair
          * if the resulting mask is 0 the whole entry (and its slot
            epoch dict) is dropped from both the shadow and the C++
            table

        Returns True iff the local state changed.
        """
        worker_idx = self._addr_to_idx[worker_addr]

        # ── 1. LWW gate ────────────────────────────────────────────────
        slot_epochs = self._slot_epoch.get(key)
        if slot_epochs is not None:
            stored = slot_epochs.get(worker_idx)
            if stored is not None and (incoming_epoch, incoming_source_id) <= stored:
                # Stale / duplicate write — drop.
                return False

        # ── 2. Compute new packed value ───────────────────────────────
        shift = _BITS_PER_WORKER * worker_idx
        slot_mask = _TIER_MASK << shift
        old_value = self._lru.get(key, 0)
        new_value = (old_value & ~slot_mask) | ((tier & _TIER_MASK) << shift)

        if old_value == 0 and new_value == 0:
            # Tombstone-only write: removing a worker from an entry that
            # doesn't exist (yet/anymore).  No observable state change in
            # the LRU/Ludo, but we DO record the slot epoch so a future
            # stale add (re-ordered gossip carrying the original
            # advertise) gets correctly rejected.  LWW gate has already
            # been passed at this point, so this is "accepted, but no-op
            # on the visible table" — return True.
            self._slot_epoch.setdefault(key, {})[worker_idx] = (
                incoming_epoch, incoming_source_id,
            )
            return True

        # ── 3. Apply to LudoTable + LRU shadow ────────────────────────
        if new_value == 0:
            # Last worker just got cleared — drop entry from C++ + LRU.
            # Update slot epoch BEFORE remove so a future stale add
            # gets correctly rejected.
            slot_epochs = self._slot_epoch.setdefault(key, {})
            slot_epochs[worker_idx] = (incoming_epoch, incoming_source_id)
            self._remove_key(key, preserve_slot_epoch_for=worker_idx)
            return True

        if old_value == 0:
            # Fresh entry; respect LRU capacity.
            if len(self._lru) >= self._max_entries:
                self._evict_lru()
            self._write_value(key, new_value, fresh=True)
            self._lru[key] = new_value
        else:
            # Existing entry; update in place + bump to MRU.
            self._write_value(key, new_value, fresh=False)
            self._lru[key] = new_value
            self._lru.move_to_end(key)

        # ── 4. Stamp the slot epoch ──────────────────────────────────
        self._slot_epoch.setdefault(key, {})[worker_idx] = (
            incoming_epoch, incoming_source_id,
        )
        return True

    def _lookup_key_internal(self, key: int) -> List[Tuple[str, int]]:
        value = self._read_value(key)
        if value == 0:
            return []
        self._hit_count += 1
        if key in self._lru:
            self._lru.move_to_end(key)
        return self._unpack_holders(value)

    def _remove_key(
        self,
        key: int,
        *,
        preserve_slot_epoch_for: Optional[int] = None,
    ) -> bool:
        """
        Drop ``key`` from both the LRU shadow and the C++ Ludo table.

        ``preserve_slot_epoch_for`` keeps a single worker's slot-epoch
        entry alive in ``self._slot_epoch[key]`` so that a future stale
        add to that worker (e.g. an out-of-order gossip delta) is
        correctly rejected by the LWW gate.  Pass ``None`` to wipe the
        whole slot_epoch entry (only safe for fresh/manual `remove()`
        calls where no cross-router race is possible).
        """
        existed = key in self._lru
        self._lru.pop(key, None)

        if preserve_slot_epoch_for is None:
            self._slot_epoch.pop(key, None)
        else:
            slot_epochs = self._slot_epoch.get(key)
            if slot_epochs is not None:
                preserved = slot_epochs.get(preserve_slot_epoch_for)
                slot_epochs.clear()
                if preserved is not None:
                    slot_epochs[preserve_slot_epoch_for] = preserved
                else:
                    self._slot_epoch.pop(key, None)

        if self._native:
            # Smash binding's remove() returns a dict; remove_bool keeps the
            # legacy "True iff key was present" contract used here.
            return self._cp.remove_bool(key)
        present = key in self._fallback
        self._fallback.pop(key, None)
        return present or existed

    def _evict_lru(self) -> None:
        """Evict the LRU entry (entire packed value, all workers) in O(1).

        LRU eviction is the LudoTable's local memory-pressure response;
        it does NOT participate in LWW (we forget the slot epochs along
        with the entry).  If a peer router still has the prefix and
        gossips its state to us later, we'll re-learn it with a fresh
        epoch lineage."""
        lru_key, _ = self._lru.popitem(last=False)
        self._slot_epoch.pop(lru_key, None)
        if self._native:
            self._cp.remove(lru_key)
        else:
            self._fallback.pop(lru_key, None)
        self._evict_count += 1

    # ── Bit-pack utilities ─────────────────────────────────────────────────────

    def _unpack_holders(self, value: int) -> List[Tuple[str, int]]:
        """
        Decode packed value → [(worker_addr, tier_int)] for slots != TIER_NONE.
        """
        out: List[Tuple[str, int]] = []
        for i in range(self._num_workers):
            t = (value >> (_BITS_PER_WORKER * i)) & _TIER_MASK
            if t != TIER_NONE:
                out.append((self._idx_to_addr[i], t))
        return out

    def _holder_count(self, value: int) -> int:
        """Count workers with any non-NONE tier in the packed value."""
        n = 0
        for i in range(self._num_workers):
            if (value >> (_BITS_PER_WORKER * i)) & _TIER_MASK:
                n += 1
        return n

    def _is_dram_only(self, value: int) -> bool:
        """True iff every present holder is at TIER_DRAM (no VRAM holders)."""
        any_dram = False
        for i in range(self._num_workers):
            t = (value >> (_BITS_PER_WORKER * i)) & _TIER_MASK
            if t == TIER_VRAM:
                return False
            if t == TIER_DRAM:
                any_dram = True
        return any_dram
