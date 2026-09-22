"""
Central Directory — strongest centralised baseline.

Models a single linearisable coordinator that every request must
consult before dispatch. Two costs are charged:

  (1) RTT: every lookup pays ``DIRECTORY_LOOKUP_OVERHEAD_MS`` — twice
      the worst-case one-way WAN delay (= 2 × max of the CloudPing
      matrix). Calibrated against AWS DynamoDB Global Tables / Spanner
      / WonderNetwork / Facebook TAO production measurements.

  (2) M/D/1 service queue at the coordinator: capped at
      ``DIRECTORY_SERVICE_RATE_RPS`` linearisable reads per second,
      matching published rates for Spanner, etcd v3 readIndex, and
      ZooKeeper sync()+read in cross-region quorum deployments. Each
      arrival is stamped with ``next_free_ms`` and the lookup blocks
      until that point.

Idealised properties (all giving CD an *advantage* over any real
deployable directory):
  - Omniscient cache state: every advertise() is reflected immediately.
  - Infinite per-node memory: no eviction, every hit serves from VRAM.
  - Infinite inter-node bandwidth: only WAN propagation is charged.

These idealisations make CD a *lower-bound on latency* for the entire
class of centralised discovery planes: any real coordinator can only
do worse.
"""
from __future__ import annotations

import asyncio
import time

from ludo_sglang import config
from ludo_sglang.router.load_tracker import WorkerLoad
from ludo_sglang.router.schemes.base import RouteDecision, Scheme


class CentralDirectoryScheme(Scheme):
    name = "central_directory"

    # Uses the published baseline chunk schedule (sim's CHUNK_SIZES),
    # capped at 4096 tokens. Ludo's larger 24K chunk levels are *its*
    # design point — a centralised directory pays per-record coordinator
    # cost so it cannot afford finer granularity at scale.
    def __init__(self, n_workers: int):
        super().__init__(n_workers)
        self.chunk_sizes: list[int] = list(config.BASELINE_CHUNK_SIZES)
        self.rtt_ms        = config.DIRECTORY_LOOKUP_OVERHEAD_MS
        self.service_ms    = config.DIRECTORY_SERVICE_MS
        # Global directory: chunk_key (uint32) → set of worker ranks.
        # Idealised "no eviction": OrderedDict only used here as a fast
        # set-keyed map (no LRU policy).
        self.dir: dict[int, set[int]] = {}
        # Coordinator queue state — an M/D/1 queue with deterministic
        # service. ``next_free_ms`` is the earliest future time at
        # which the coordinator can accept a new arrival.
        self._queue_lock = asyncio.Lock()
        self._next_free_ms: float = 0.0

    @staticmethod
    def _key(token_ids):
        return hash(tuple(token_ids)) & 0xFFFFFFFF

    async def route(self, token_ids, load: WorkerLoad, origin_region: int,
                    ctx: dict | None = None):
        # ── (1) join the coordinator queue (M/D/1) ──────────────────────
        async with self._queue_lock:
            now_ms = time.monotonic() * 1000.0
            start_service = max(now_ms, self._next_free_ms)
            self._next_free_ms = start_service + self.service_ms
            coord_queue_wait_ms = start_service - now_ms

        directory_overhead_ms = (self.rtt_ms
                                 + coord_queue_wait_ms
                                 + self.service_ms)

        # ── (2) consult the directory: pick longest-prefix owner set ────
        prompt_len = len(token_ids)
        owners: set[int] = set()
        matched = 0
        for cs in self.chunk_sizes:
            if cs > prompt_len:
                continue
            cur = self.dir.get(self._key(token_ids[:cs]))
            if cur:
                owners, matched = cur, cs

        # ── (3) least-loaded among the owner set, else globally; ────────
        #         break ties on lower WAN to origin (matches what an
        #         omniscient coordinator would do given equal load).
        candidates = owners if owners else set(range(self.n_workers))
        best = min(
            candidates,
            key=lambda r: (load.pending_ms[r],
                           config.wan_ms_to_worker(origin_region, r)),
        )
        wan_ms = config.wan_ms_to_worker(origin_region, best)
        residual = max(0, prompt_len - matched)
        # Score reflects the full wall-clock TTFT a CD deployment would
        # observe: dir RTT + coord queue + (worker-side) prefill + WAN.
        score_ms = (directory_overhead_ms
                    + load.pending_ms[best]
                    + residual * load.prefill_rate[best]
                    + wan_ms)

        return RouteDecision(
            rank             = best,
            matched_tokens   = matched,
            extra_lookup_ms  = directory_overhead_ms + wan_ms,
            wan_to_worker_ms = wan_ms,
            score_ms         = score_ms,
            info             = {"coord_queue_wait_ms":   coord_queue_wait_ms,
                                "directory_overhead_ms": directory_overhead_ms,
                                "owners_count":          len(owners)},
        )

    def advertise(self, token_ids, rank):
        prompt_len = len(token_ids)
        for cs in self.chunk_sizes:
            if cs <= prompt_len:
                self.dir.setdefault(self._key(token_ids[:cs]), set()).add(rank)

    def stats(self):
        s = super().stats()
        s["directory_entries"] = len(self.dir)
        s["coord_next_free_ms"] = self._next_free_ms
        s["service_ms"] = self.service_ms
        s["rtt_ms"]     = self.rtt_ms
        return s
