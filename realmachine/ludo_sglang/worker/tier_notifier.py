"""
tier_notifier.py — worker-side client that pushes RadixCache tier
transitions to the discovery plane.

Two modes selected by environment:

* **Gossip mode** (legacy) — set ``LUDO_ROUTER_URL`` to the edge router
  in the same region.  Every notification is a single fire-and-forget
  POST to that URL; the router applies it locally and gossips it to
  its peers.  This mode does NOT take advertise / sequence numbers —
  the router advertises VRAM itself synchronously after dispatch.

* **Central-directory mode** — set ``LUDO_AUTHORITY_URLS`` to a
  comma-separated list of authority base URLs.  Events are coalesced
  for ``LUDO_NOTIFIER_BATCH_MS`` (default 5 ms) into a single batched
  POST ``/events_batch`` per shard, mirroring Smash's master-side batch
  protocol.  Bulk API calls (``notify_*_bulk``) flush immediately so a
  whole subtree eviction lands in one batch instead of forcing a
  coalesce wait.  This mode also carries advertise: the worker emits
  ``notify_advertise()`` itself when prefill is done (the router no
  longer advertises).  Every event carries a per-worker monotonic
  ``seq`` for the authority to dedup re-orderings.

Configuration::

    # Gossip (legacy)
    LUDO_ROUTER_URL=http://router:8001

    # Central (new)
    LUDO_AUTHORITY_URLS=http://m0:8101,http://m1:8101
    LUDO_WORKER_RANK=2                  # this worker's index
    LUDO_NOTIFIER_TIMEOUT=1             # seconds; default 1
    # Optional, used by authority-mode shard hashing.  Defaults to the
    # first chunk size in LUDO_CHUNK_SIZES, which must agree across
    # workers / authorities / lookups.
    LUDO_SHARD_HASH_LEVEL_TOKENS=256

If neither env is set the notifier is a no-op (so a patched SGLang
stays safe to launch without a discovery plane).

Strict VRAM→DRAM→gone pipeline (central mode)
---------------------------------------------
Four entry points exposed.  See :doc:`SGLANG_TIER_PATCH.md` for which
SGLang HiRadixCache method to hook each on:

    notify_advertise(token_ids)   — prefill just finished, KV in VRAM
    notify_demote(token_ids)      — VRAM → DRAM (write_through)
    notify_promote(token_ids)     — DRAM → VRAM (load_from_host)
    notify_evict(token_ids)       — DRAM → gone (host eviction)

Under the strict pipeline there is no "VRAM → gone direct" path; if
your SGLang config has DRAM tier disabled, hook ``notify_evict`` in
the VRAM eviction method instead — the authority-side state machine is
agnostic to which tier the chunk left from.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import List, Optional

import httpx

from src.ludo.ludo_table import chunk_to_key

_log = logging.getLogger("ludo.tier_notifier")


# ── Mode + lazy state ───────────────────────────────────────────────────
_MODE: Optional[str] = None      # "gossip" | "central" | None (disabled)
_client: Optional[httpx.AsyncClient] = None
_router_url: Optional[str] = None     # gossip mode
_authority_urls: List[str] = []          # central mode
_n_shards: int = 0                    # central mode
_my_rank: int = -1                    # central mode (worker idx)
_shard_hash_tokens: int = 256         # central mode (legacy single-target route)
_chunk_sizes: List[int] = []          # central mode (per-level fan-out)
_timeout_s: float = 1.0
_loop: Optional[asyncio.AbstractEventLoop] = None
_initialised: bool = False

# Coalescing window for central-mode batched POSTs (ms).
_coalesce_ms: float = 5.0
_batcher: "Optional[BatchedNotifier]" = None

# Per-worker monotonic counter (central mode).  Bumped by every
# notify_*; serialised by a lock so multiple coroutines on the same
# worker can fire concurrently without collisions.
_seq_counter: int = 0
_seq_lock = threading.Lock()


# ── Initialisation ──────────────────────────────────────────────────────
def _init_once() -> bool:
    """Read env, build the httpx client, return True iff notifier is enabled."""
    global _MODE, _client, _router_url, _authority_urls, _n_shards
    global _my_rank, _shard_hash_tokens, _chunk_sizes, _timeout_s, _loop, _initialised
    global _coalesce_ms, _batcher
    if _initialised:
        return _client is not None
    _initialised = True

    try:
        _timeout_s = float(os.environ.get("LUDO_NOTIFIER_TIMEOUT", "1"))
    except ValueError:
        _timeout_s = 1.0
    try:
        _coalesce_ms = float(os.environ.get("LUDO_NOTIFIER_BATCH_MS", "5"))
    except ValueError:
        _coalesce_ms = 5.0

    authority_env = os.environ.get("LUDO_AUTHORITY_URLS", "").strip()
    router_env = os.environ.get("LUDO_ROUTER_URL", "").strip()

    if authority_env:
        _MODE = "central"
        _authority_urls = [u.strip().rstrip("/") for u in authority_env.split(",") if u.strip()]
        _n_shards = max(1, len(_authority_urls))
        _my_rank = int(os.environ.get("LUDO_WORKER_RANK", "-1"))
        if _my_rank < 0:
            _log.error(
                "LUDO_WORKER_RANK unset (or negative); central-mode notifier "
                "needs to know its rank to populate the right slot.  "
                "Disabling tier notifier.")
            return False
        try:
            _shard_hash_tokens = int(
                os.environ.get("LUDO_SHARD_HASH_LEVEL_TOKENS", "256")
            )
        except ValueError:
            _shard_hash_tokens = 256
        # Per-level fan-out (default keeps all the chunk sizes the router /
        # authority use). Each notify is split into one event per level
        # whose chunk size fits the prefix, and each per-level event is
        # routed to the authority that ``owns`` that level's key — no
        # more cross_shard drops at higher levels. CHUNK_SIZES env mirrors
        # the one read by ``ludo_sglang.config``.
        # An empty-string env var (e.g. fleet script setting `CHUNK_SIZES=""`)
        # should fall back to the default, not be parsed as zero levels —
        # `os.environ.get(k, default)` only returns the default when the key
        # is missing, not when it's an empty string.
        _chunk_sizes_env = (
            os.environ.get("CHUNK_SIZES")
            or os.environ.get("CHUNK_SIZE")
            or "256,512,1024,2048,4096,8192,16384,24576"
        )
        try:
            _chunk_sizes = [int(x.strip()) for x in _chunk_sizes_env.split(",") if x.strip()]
        except ValueError:
            _chunk_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 24576]
        if not _chunk_sizes:
            _chunk_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 24576]
        _log.info(
            "tier-notifier (central): %d authority(ies), my_rank=%d, "
            "shard hash first %d toks, coalesce %.2f ms",
            len(_authority_urls), _my_rank, _shard_hash_tokens, _coalesce_ms,
        )
    elif router_env:
        _MODE = "gossip"
        _router_url = router_env.rstrip("/")
        _log.info("tier-notifier (gossip): → %s (timeout=%.2fs)",
                  _router_url, _timeout_s)
    else:
        _log.info("Neither LUDO_AUTHORITY_URLS nor LUDO_ROUTER_URL set; "
                  "tier-notifier disabled (no-op).")
        return False

    # Dedicated background loop so the notifier never depends on the
    # caller's asyncio state. The SGLang scheduler runs sync code that
    # calls our hooks; there's no running loop at the call site, and the
    # one returned by ``asyncio.get_event_loop()`` is typically idle —
    # ``run_coroutine_threadsafe`` against an idle loop silently buffers
    # forever. Owning the loop guarantees POSTs actually fire.
    _loop = asyncio.new_event_loop()
    _loop_thread = threading.Thread(
        target=_loop.run_forever, name="ludo-notifier-loop", daemon=True,
    )
    _loop_thread.start()

    _client = httpx.AsyncClient(
        timeout=_timeout_s,
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
    )
    if _MODE == "central":
        _batcher = BatchedNotifier(
            authority_urls = _authority_urls,
            my_rank     = _my_rank,
            coalesce_ms = _coalesce_ms,
            client      = _client,
        )
    return True


def _next_seq() -> int:
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


# ── Sharding (central mode) ─────────────────────────────────────────────
def _authority_for(token_ids: List[int]) -> str:
    """Pick the authority URL responsible for the shard this prefix lives on.

    All maintenance traffic for a single ``token_ids`` MUST land on
    the same authority so the authority can write all L levels coherently.
    We hash the first ``_shard_hash_tokens`` tokens (which corresponds
    to the shortest level used in LUDO_CHUNK_SIZES) via the same
    ``chunk_to_key`` the authority uses for level-0 keys — this is what
    ``AuthorityLudoTable.owns_key`` checks against ``key % n_shards``.
    """
    if len(_authority_urls) == 1:
        return _authority_urls[0]
    key = chunk_to_key(token_ids[:_shard_hash_tokens])
    return _authority_urls[key % _n_shards]


# ── BatchedNotifier (central mode) ──────────────────────────────────────
class BatchedNotifier:
    """Per-shard event coalescer for central-mode discovery plane.

    Multiple notify_* calls within ``coalesce_ms`` get bundled into one
    POST ``/events_batch`` per shard. Bulk APIs (``notify_*_bulk``)
    enqueue everything then trigger an immediate flush so a whole
    subtree eviction lands on the authority in one batch.
    """
    def __init__(
        self,
        authority_urls: List[str],
        my_rank: int,
        coalesce_ms: float,
        client: httpx.AsyncClient,
    ):
        self._authority_urls    = list(authority_urls)
        self._n_shards       = max(1, len(authority_urls))
        self._my_rank        = int(my_rank)
        self._coalesce_s     = max(0.0, float(coalesce_ms) / 1000.0)
        self._client         = client
        self._buffers: dict[str, list[dict]] = {u: [] for u in self._authority_urls}
        self._lock           = threading.Lock()
        self._flush_pending  = False
        self._batches_sent   = 0
        self._events_sent    = 0
        self._post_failures  = 0

    def enqueue(self, authority_url: str, event_payload: dict) -> bool:
        """Add one event to the shard's buffer. Returns True iff the
        caller is responsible for scheduling the next flush task (so we
        only have one pending coalesce timer at a time)."""
        with self._lock:
            self._buffers.setdefault(authority_url, []).append(event_payload)
            if self._flush_pending:
                return False
            self._flush_pending = True
            return True

    async def coalesce_then_flush(self) -> None:
        if self._coalesce_s > 0:
            await asyncio.sleep(self._coalesce_s)
        await self.flush_now()

    async def flush_now(self) -> None:
        with self._lock:
            self._flush_pending = False
            snapshot = {u: buf for u, buf in self._buffers.items() if buf}
            for u in snapshot:
                self._buffers[u] = []
        if not snapshot:
            return
        await asyncio.gather(*[
            self._post_one(u, evs) for u, evs in snapshot.items()
        ], return_exceptions=True)

    async def _post_one(self, authority_url: str, events: list[dict]) -> None:
        if self._client is None or not events:
            return
        try:
            await self._client.post(
                f"{authority_url}/events_batch",
                json={"rank": self._my_rank, "events": events},
            )
            self._batches_sent += 1
            self._events_sent  += len(events)
        except Exception as exc:
            self._post_failures += 1
            _log.debug("events_batch %s failed: %s", authority_url, exc)

    def stats(self) -> dict:
        with self._lock:
            queued = sum(len(b) for b in self._buffers.values())
        return {
            "queued":         queued,
            "batches_sent":   self._batches_sent,
            "events_sent":    self._events_sent,
            "post_failures":  self._post_failures,
            "coalesce_ms":    self._coalesce_s * 1000.0,
        }


# ── HTTP fire-and-forget (gossip-mode legacy) ───────────────────────────
async def _post_gossip(event: str, rank: int, token_ids: List[int]) -> None:
    """Original gossip-mode wire format — kept verbatim for back-compat."""
    if _client is None:
        return
    try:
        await _client.post(
            f"{_router_url}/internal/tier_event",
            json={"rank": rank, "token_ids": token_ids, "event": event},
        )
    except Exception as exc:
        _log.debug("gossip tier-notify %s rank=%d failed: %s",
                   event, rank, exc)


def _fire_async(coro) -> None:
    """Schedule a coroutine on the notifier's own background loop. We never
    use the caller's loop because the SGLang scheduler invokes our hooks
    from sync code where ``get_event_loop()`` returns an idle loop that
    nothing drives — ``run_coroutine_threadsafe`` against an idle loop
    silently buffers forever. The background loop set up in
    ``_init_once`` is always running, so submission is reliable."""
    if _loop is None or not _loop.is_running():
        return
    asyncio.run_coroutine_threadsafe(coro, _loop)


def _enqueue_central(event: str, rank: int, token_ids: List[int]) -> None:
    """Fan-out a notify into one event per chunk-size level. Each level's
    event carries:
      * the token slice ``token_ids[:cs]`` so the authority computes the
        same ``chunk_to_key`` the lookup will use at read time;
      * an explicit ``level`` field so the authority skips its
        iterate-all-levels loop and writes only the slot it owns;
      * a fresh per-rank ``seq`` so the authority's stale-seq dedup
        treats each level's event independently.
    Routing: ``authority_urls[chunk_to_key(slice) % n_shards]``. By
    construction the receiving authority always ``owns_key`` for that
    level — cross_shard drops go to zero, and the lookup mirror is
    dense at every level (matching gossip mode's coverage)."""
    if _batcher is None or not _chunk_sizes or _n_shards == 0:
        return
    n = len(token_ids)
    schedule_any = False
    for level, cs in enumerate(_chunk_sizes):
        if cs > n:
            break
        slice_toks = list(token_ids[:cs])
        key = chunk_to_key(slice_toks)
        target = _authority_urls[key % _n_shards]
        payload = {
            "event":     event,
            "token_ids": slice_toks,
            "seq":       _next_seq(),
            "level":     level,
        }
        if _batcher.enqueue(target, payload):
            schedule_any = True
    if schedule_any:
        _fire_async(_batcher.coalesce_then_flush())


def _flush_now() -> None:
    """Force-drain the central-mode batcher (bulk API helper)."""
    if _batcher is None:
        return
    _fire_async(_batcher.flush_now())


def _fire_event(event: str, rank: int, token_ids: List[int]) -> None:
    """Dispatch one event to whichever mode is configured."""
    if not _init_once():
        return
    if not token_ids:
        return
    if _MODE == "central":
        _enqueue_central(event, _my_rank if rank < 0 else rank, list(token_ids))
    elif _MODE == "gossip":
        _fire_async(_post_gossip(event, rank, list(token_ids)))


# ── Public API ──────────────────────────────────────────────────────────

def notify_advertise(token_ids: List[int], rank: int = -1) -> None:
    """
    Worker just finished prefill — KV is in VRAM.  Central mode only;
    in gossip mode the router advertises itself synchronously after
    dispatch and this function is a no-op.

    ``rank`` defaults to the env-configured ``LUDO_WORKER_RANK``;
    pass an explicit value if the worker process serves multiple
    ranks (rare).
    """
    if not _init_once():
        return
    if _MODE != "central":
        return    # gossip mode: router writes VRAM after dispatch
    if not token_ids:
        return
    _enqueue_central("advertise", _my_rank if rank < 0 else rank, list(token_ids))


def notify_demote(token_ids: List[int], rank: int = -1) -> None:
    """VRAM → DRAM (called from ``write_through_to_host_memory``).

    ``rank`` defaults to the env-configured ``LUDO_WORKER_RANK``."""
    _fire_event("demote", _my_rank if rank < 0 else rank, token_ids)


def notify_promote(token_ids: List[int], rank: int = -1) -> None:
    """DRAM → VRAM (called from ``load_from_host_memory``).

    ``rank`` defaults to the env-configured ``LUDO_WORKER_RANK``."""
    _fire_event("promote", _my_rank if rank < 0 else rank, token_ids)


def notify_evict(token_ids: List[int], rank: int = -1) -> None:
    """
    Worker fully evicted this chunk — chunk is gone from both tiers.

    Under the strict VRAM→DRAM→gone pipeline this fires from the
    host-memory eviction path (DRAM is the last tier; once host pinned
    memory pressure forces it out, the chunk leaves the worker
    entirely).  If your deployment disables the DRAM tier, hook this
    on the VRAM-eviction path instead — the authority state machine
    treats both identically.
    """
    _fire_event("evict", _my_rank if rank < 0 else rank, token_ids)


# ── Bulk variants ───────────────────────────────────────────────────────
# In central mode, bulk calls fill the batcher then trigger an immediate
# flush so a whole subtree eviction lands as ONE batched POST per shard
# instead of waiting for the coalesce timer per chunk.  Gossip mode just
# loops because each event is a separate single-event POST anyway.

def notify_demote_bulk(rank: int, chunks: List[List[int]]) -> None:
    if not chunks:
        return
    if _init_once() and _MODE == "central":
        for c in chunks:
            if c:
                _enqueue_central("demote",
                                 _my_rank if rank < 0 else rank, list(c))
        _flush_now()
        return
    for c in chunks:
        notify_demote(c, rank)


def notify_evict_bulk(rank: int, chunks: List[List[int]]) -> None:
    if not chunks:
        return
    if _init_once() and _MODE == "central":
        for c in chunks:
            if c:
                _enqueue_central("evict",
                                 _my_rank if rank < 0 else rank, list(c))
        _flush_now()
        return
    for c in chunks:
        notify_evict(c, rank)


# Back-compat alias for any code that already imports notify_evict_bulk_dram
notify_evict_bulk_dram = notify_evict_bulk


# ── Stats (for /stats endpoints) ────────────────────────────────────────
def stats() -> dict:
    """Snapshot of notifier internals — useful for the SGLang
    worker-side /stats endpoint to verify batching is actually happening."""
    out = {
        "mode":           _MODE,
        "initialised":    _initialised,
        "my_rank":        _my_rank,
        "n_authority_urls":  len(_authority_urls),
        "router_url":     _router_url,
        "coalesce_ms":    _coalesce_ms,
    }
    if _batcher is not None:
        out["batcher"] = _batcher.stats()
    return out
