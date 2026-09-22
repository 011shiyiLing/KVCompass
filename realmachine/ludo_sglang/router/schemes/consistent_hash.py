"""
Consistent Hash scheme — multi-level "hash-placement without migration",
deterministic single-owner per (key, level).

Mirrors §3 simulation's ``ConsistentHashDiscovery`` line for line:

  * State: one ``dict[hash_key → owner_rank]`` per level in
    ``LUDO_CHUNK_SIZES`` (6 levels by default: 128 / 256 / 512 / 1024 /
    2048 / 4096 tokens).
  * advertise(): for each applicable level ``l``, register the serving
    worker ONLY IF that worker happens to be the level-``l``
    hash-owner of the prefix. This is the structural cost of blind
    hash placement — caches don't migrate, so they're only indexed
    at levels where their serving node coincidentally equals the
    designated hash-owner.
  * lookup(): walk levels contiguously from level 0 upwards. If a
    level has no entry, OR has an entry whose owner differs from the
    deepest matched owner, stop. Return the longest prefix length
    along the matched walk.
  * On miss: route to the level-0 hash-owner (so subsequent advertise
    is guaranteed to fill level 0 at minimum).

Hash-owner space:
  We hash into ``NUM_NODES_LOGICAL`` (default 64) logical nodes — the
  planet-scale topology the §3 simulation assumes — then map a logical
  node to a physical worker by ``logical % N_DP_WORKERS``. This keeps
  the advertise success rate identical to sim at any cluster size:
  "level-l owner is one specific worker, any other worker can't
  register at level l".

This scheme is the parent class for the two DHT baselines (Kademlia /
Chord), which add their own ``extra_lookup_overhead_ms`` on top of the
identical placement / lookup behaviour.
"""
from __future__ import annotations

from ludo_sglang import config
from ludo_sglang.router.load_tracker import WorkerLoad
from ludo_sglang.router.schemes.base import RouteDecision, Scheme


class ConsistentHashScheme(Scheme):
    name = "consistent_hash"

    # Subclasses (Kademlia / Chord) override; plain CH still pays a
    # single LAN/WAN hop to consult the consistent-hash ring (= one
    # WAN_AVG sample). Charging zero would be more idealised than the
    # simulation, which already credits CH with no coordinator queue.
    extra_lookup_overhead_ms: float = config.WAN_AVG_MS

    def __init__(self, n_workers: int):
        super().__init__(n_workers)
        self.chunk_sizes: list[int] = list(config.BASELINE_CHUNK_SIZES)
        # state[level][hash_key] = serving worker rank.
        # Only the level-l hash-owner ever writes here, so any non-empty
        # entry at level l is by construction owned by that hash-owner.
        self.state: list[dict[int, int]] = [{} for _ in self.chunk_sizes]

    # ── Hash helpers ────────────────────────────────────────────────────
    @staticmethod
    def _content_key(token_ids: list[int], cs: int) -> int:
        """Hash of the prompt's first ``cs`` tokens."""
        return hash(tuple(token_ids[:cs])) & 0xFFFFFFFF

    @staticmethod
    def _logical_owner(key: int, level: int) -> int:
        """Hash-placement: independent uniform draw over the logical
        topology for each (key, level) pair. Mirrors sim's
        ``hash((key, level)) % NUM_NODES``."""
        return hash((key, level)) % config.NUM_NODES_LOGICAL

    def _physical_owner(self, key: int, level: int) -> int:
        """Map level-l logical owner to a physical worker rank.
        With ``NUM_NODES_LOGICAL = 64`` and ``N_DP_WORKERS = 2``, a
        given (key, level) pair maps deterministically to one of 64
        logical owners, which then bucket into physical workers via
        modulo. Different levels of the same key independently bucket
        — this is what produces sim's per-level coverage rate."""
        return self._logical_owner(key, level) % self.n_workers

    # ── Lookup: contiguous-level walk ───────────────────────────────────
    def _lookup_contig(self, token_ids: list[int]) -> tuple[int, int]:
        """Walk levels 0..L-1 in order, stopping at the first level with
        no entry (or with a different owner from the deepest one).
        Returns (deepest_level, owner_rank); deepest_level is -1 on miss.
        """
        prompt_len = len(token_ids)
        deepest = -1
        owner = -1
        for l, cs in enumerate(self.chunk_sizes):
            if prompt_len < cs:
                break
            key = self._content_key(token_ids, cs)
            entry = self.state[l].get(key)
            if entry is None:
                break                       # non-contiguous → stop
            if deepest == -1:
                deepest, owner = l, entry
            elif entry == owner:
                deepest = l                 # extend the contiguous walk
            else:
                break                       # different owner at this level
        return deepest, owner

    # ── route ───────────────────────────────────────────────────────────
    async def route(self, token_ids, load: WorkerLoad, origin_region: int,
                    ctx: dict | None = None):
        prompt_len = len(token_ids)

        # Sim-faithful routing (simulate.py:1494-1517): if any level has
        # an entry, route to that level's deepest contiguous owner with
        # match_len = min(chunk_size, prompt_len). Otherwise fall back
        # to the level-0 hash-owner so the subsequent advertise() lands
        # at level 0 by construction (matches sim's miss-fallback).
        deepest, owner = self._lookup_contig(token_ids)
        if deepest >= 0:
            best = owner
            matched = min(self.chunk_sizes[deepest], prompt_len)
        else:
            key0 = self._content_key(token_ids, self.chunk_sizes[0])
            best = self._physical_owner(key0, 0)
            matched = 0

        wan_ms = config.wan_ms_to_worker(origin_region, best)
        residual = max(0, prompt_len - matched)
        score_ms = (load.pending_ms[best]
                    + residual * load.prefill_rate[best]
                    + wan_ms
                    + self.extra_lookup_overhead_ms)
        return RouteDecision(
            rank             = best,
            matched_tokens   = matched,
            extra_lookup_ms  = wan_ms + self.extra_lookup_overhead_ms,
            wan_to_worker_ms = wan_ms,
            score_ms         = score_ms,
            info             = {"deepest_level": deepest,
                                "matched_tokens_at_deepest": matched,
                                "registered_owner": owner if deepest>=0 else -1},
        )

    # ── advertise: register only if rank is the level-l hash-owner ──────
    # Hash-placement without migration: a node's cache shows up at level
    # l only if it happens to own the level-l hash. This is the
    # structural cost of blind hash placement that sim charges (≈1/N
    # per-level coverage), and that Ludo's explicit-write multi-level
    # table avoids.
    def advertise(self, token_ids, rank):
        prompt_len = len(token_ids)
        for l, cs in enumerate(self.chunk_sizes):
            if prompt_len < cs:
                break
            key = self._content_key(token_ids, cs)
            if rank == self._physical_owner(key, l):
                self.state[l][key] = rank

    # ── stats ───────────────────────────────────────────────────────────
    def stats(self):
        s = super().stats()
        s["levels"] = len(self.chunk_sizes)
        s["entries_per_level"] = [len(d) for d in self.state]
        return s
