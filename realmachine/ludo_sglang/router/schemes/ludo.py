"""
Ludo scheme (KVCompass) — latency-levelled content-addressed routing
table with load-gated joint scoring and explicit popular-KV migration.

Port of the §3 simulation's ``LudoDiscovery`` + ``route()`` ludo branch
(``simulation/simulate.py``) onto the
real-machine router.  Three mechanisms replace the old flat-chunk /
pending-argmin design:

1. **Latency-derived level cuts** — a requester in region S probes
   holders in region X along ``LUDO_PAIR_CUTS[S][X] = [n₁, 1.5·n₁,
   2·n₁, …]`` (α = LUDO_STEP_ALPHA), where n₁(S,X) = WAN[S][X] /
   C_PREFILL is the break-even prefix length that just beats one S↔X
   network latency's worth of prefill.  Level 0 IS the latency floor: a
   cross-region match shorter than n₁ never surfaces.  Same-region
   matching keeps the fine-grained ``LUDO_CHUNK_SIZES`` chain.  A match
   must be *contiguous* along its chain (present at every shallower cut)
   — two prompts colliding at cut c share exactly c tokens, no more.

2. **Joint score over the viewed queue** (milliseconds)::

       score(r) = q_view[r] + (1 + λ)·T_p(r) + T_t(r)

       q_view[r] = own live pending + peer routers' EWMA-reported
                   pending  (see router/lb_view.py — remote regions are
                   seen through a 100 ms load broadcast, own region live)
       T_p(r)    = (prompt_len − eff_match[r]) × prefill_rate[r]
       T_t(r)    = WAN(origin → region[r]) + DRAM-reload if matched
                   holder tier is DRAM
       λ         = min(LUDO_LAMBDA_MAX, min_r q_view[r] / LUDO_LAMBDA_SCALE_MS)

   λ is the capacity-pricing multiplier: with spare capacity λ≈0 and
   routing stays balance-first; at saturation full prefills become
   (1+λ)× expensive and the argmin slides toward cache holders.
   ``eff_match[r] = max_match[r] × cache_belief[r]`` discounts workers
   whose routing-plane claims SGLang's RadixCache has not honoured.

3. **Explicit popular-KV migration** — locality misses are counted per
   (origin region, prefix family, kind) window; when a family is hot
   (≥ MIG_HOT_K misses / MIG_WINDOW_MS and past MIG_COOLDOWN_MS) a
   replica is built on an idle worker in the demand region:
     * 'cross' : requests from S keep matching only REMOTE holders;
     * 'intra' : a LOCAL holder exists but keeps losing the load-gated
                 score, up to MIG_INTRA_RMAX local replicas.
   On real hardware both transports are PROMPT-SHIPPING + re-prefill on
   the destination worker (SGLang exposes no inter-instance KV transfer;
   re-prefill on an idle GPU is also the transport the sim prefers for
   cross-region moves).  The migration runs as a background asyncio task
   (off the request critical path) fired through the router-provided
   ``attach_migrator`` callback; the replica lands in the destination's
   RadixCache and is advertised like any other prefill.  Migrations only
   spend spare capacity: no idle destination (viewed queue ≤
   MIG_IDLE_DST_MS) ⇒ the trigger quiesces and re-arms when load recedes.
"""
from __future__ import annotations

import asyncio
import random
import time

from src.ludo.ludo_table import LudoTable, TIER_VRAM, TIER_DRAM, TIER_NONE, chunk_to_key
from src.ludo.dataplane_table import DataPlaneLudoTable
from src.ludo.sharded_dataplane import ShardedDataPlaneLudoTable
from src.ludo.authority_table import Delta

from ludo_sglang import config
from ludo_sglang.router.load_tracker import WorkerLoad
from ludo_sglang.router.schemes.base import RouteDecision, Scheme


def _now_ms() -> float:
    return time.monotonic() * 1000.0


class LudoScheme(Scheme):
    name = "ludo"

    def __init__(self, n_workers: int):
        super().__init__(n_workers)
        # Union of every cut length any region pair can probe — the
        # holder-side index writes at the region-specific subset
        # (LUDO_REGION_CUTS[region]); central mode allocates one table
        # mirror per union cut.
        self.cuts_all: list[int] = sorted(
            set().union(*[set(c) for c in config.LUDO_REGION_CUTS]))
        self._cut_level = {cs: i for i, cs in enumerate(self.cuts_all)}
        # Kept for legacy /config dumps and stats.
        self.chunk_sizes: list[int] = list(self.cuts_all)
        worker_addrs = [str(i) for i in range(n_workers)]

        if config.MAINTENANCE_MODE == "central":
            # Lookup-side mirror — one ShardedDataPlaneLudoTable per cut.
            self.tables: list[ShardedDataPlaneLudoTable] = [
                ShardedDataPlaneLudoTable(
                    worker_addrs=worker_addrs,
                    n_shards=config.N_SHARDS,
                )
                for _ in self.cuts_all
            ]
            self.table = self.tables[0]
        else:
            # Gossip mode — one shared LudoTable across all cuts (keys
            # are content hashes of token prefixes; the cut length only
            # determines which prefix is hashed).
            self.tables = None  # type: ignore
            self.table = LudoTable(
                worker_addrs     = worker_addrs,
                expected_entries = config.LUDO_INITIAL_CAPACITY,
                max_entries      = config.LUDO_MAX_ENTRIES,
                router_id        = config.ROUTER_ID,
            )

        self.gossip = None    # set via attach_gossip()
        self.migrator = None  # set via attach_migrator()

        # ── Migration state (mirrors sim's LudoDiscovery) ────────────
        self._mig_cnt:  dict[tuple, tuple[int, float]] = {}  # (S,fam,kind)->(n,win_start)
        self._mig_cool: dict[tuple, float] = {}              # (S,fam)->last trigger
        self._mig_inflight: dict[int, int] = {}              # dst rank -> in-flight count
        self.mig_stats = {"cross": 0, "intra": 0, "completed": 0, "failed": 0}

    def attach_gossip(self, manager) -> None:
        self.gossip = manager

    def attach_migrator(self, migrator) -> None:
        """Router wires in ``async migrator(dst_rank, token_ids) -> None``
        that ships the prompt prefix to ``dst_rank`` for re-prefill and
        advertises the resulting replica."""
        self.migrator = migrator

    # ── Lookup ──────────────────────────────────────────────────────────

    def _holders_at(self, cut: int, key: int) -> list[tuple[str, int]]:
        if self.tables is not None:
            return self.tables[self._cut_level[cut]].lookup_key(key)
        return self.table.lookup_key(key)

    def _chain_lookup(self, token_ids: list[int], origin_region: int,
                      ) -> tuple[list[int], list[int]]:
        """Per-holder-region chain walk with contiguous-level guard.

        Returns (max_match, match_tier) per worker.  A worker in region X
        is probed along LUDO_PAIR_CUTS[origin_region][X]; its match
        extends to chain index i only if it was present at every index
        < i.  Tier degrades to DRAM if any level along the chain is DRAM.
        """
        prompt_len = len(token_ids)
        max_match = [0] * self.n_workers
        match_tier = [TIER_NONE] * self.n_workers
        key_cache: dict[int, int] = {}

        for X in range(config.NUM_REGIONS_LOGICAL):
            if config.ABL_LOCAL_ONLY and X != origin_region:
                continue        # A2: remote holders never surface
            chain = config.LUDO_PAIR_CUTS[origin_region][X]
            if not chain:
                continue
            # rank -> (deepest contiguous chain idx, tier)
            contig: dict[int, tuple[int, int]] = {}
            for i, cut in enumerate(chain):
                if prompt_len < cut:
                    break
                key = key_cache.get(cut)
                if key is None:
                    key = chunk_to_key(token_ids[:cut])
                    key_cache[cut] = key
                holders = self._holders_at(cut, key)
                if not holders:
                    continue    # deeper cuts may still extend other chains
                for worker_addr, tier in holders:
                    rank = int(worker_addr)
                    if config.WORKER_REGIONS[rank] != X:
                        continue        # cut length shared across chains
                    if i == 0:
                        contig[rank] = (0, tier)
                    elif rank in contig and contig[rank][0] == i - 1:
                        prev_tier = contig[rank][1]
                        new_tier = (TIER_DRAM
                                    if TIER_DRAM in (prev_tier, tier)
                                    else TIER_VRAM)
                        contig[rank] = (i, new_tier)
                    # else: non-contiguous hit — ignored.
            for rank, (idx, tier) in contig.items():
                ml = min(chain[idx], prompt_len)
                if ml > max_match[rank]:
                    max_match[rank] = ml
                    match_tier[rank] = tier
        return max_match, match_tier

    # ── Routing ─────────────────────────────────────────────────────────

    async def route(self, token_ids, load: WorkerLoad, origin_region: int,
                    ctx: dict | None = None):
        prompt_len = len(token_ids)

        # 1) latency-levelled multi-cut lookup → per-worker (match, tier)
        max_match, match_tier = self._chain_lookup(token_ids, origin_region)

        # 2) joint scoring over the viewed queue.
        q_view = list(load.pending_view_ms)
        lam = min(config.LUDO_LAMBDA_MAX,
                  min(q_view) / config.LUDO_LAMBDA_SCALE_MS)
        if config.ABL_NO_LOADSCORE:
            # A1: pure affinity — no queue term, no capacity pricing.
            # q_score also drives tie-breaks and the busy gate, so zeros
            # keep load information out of every routing decision.
            lam = 0.0
            q_score = [0.0] * self.n_workers
        else:
            q_score = q_view

        wan_per_worker = [
            config.wan_ms_to_worker(origin_region, r)
            for r in range(self.n_workers)
        ]
        scores: list[float] = []
        eff_match: list[float] = []
        reload_ms: list[float] = []
        for r in range(self.n_workers):
            em = max_match[r] * load.cache_belief[r]
            eff_match.append(em)
            residual = max(0.0, prompt_len - em)
            t_p = residual * load.prefill_rate[r]

            # T_t: WAN to the worker's region + PCIe reload for DRAM hits.
            # A5 (tier-blind): tier bits are dropped from the score —
            # DRAM hits priced as VRAM.
            if (match_tier[r] == TIER_DRAM and em > 0
                    and not config.ABL_TIER_BLIND):
                t_reload = em * config.PCIE_RELOAD_MS_PER_TOKEN
            else:
                t_reload = 0.0
            reload_ms.append(t_reload)

            scores.append(q_score[r]
                          + (1.0 + lam) * t_p
                          + wan_per_worker[r] + t_reload)

        # 3) gated / ungated argmin.  A worker whose viewed queue exceeds
        #    the busy threshold is excluded from the gated optimum;
        #    default θ=∞ keeps the gate off (busy avoidance stays soft
        #    through the q_view score term).  Ties break on lower viewed
        #    queue, then random.
        best_g = best_u = None
        best_g_score = best_u_score = float("inf")
        for r in range(self.n_workers):
            if scores[r] < best_u_score:
                best_u_score, best_u = scores[r], r
            elif scores[r] == best_u_score and best_u is not None:
                if (q_score[r] < q_score[best_u]
                        or (q_score[r] == q_score[best_u]
                            and random.random() < 0.5)):
                    best_u = r
            if q_score[r] <= config.LUDO_LB_BUSY_THRESH_MS \
                    and scores[r] < best_g_score:
                best_g_score, best_g = scores[r], r
        best_rank = best_g if best_g is not None else best_u
        best_score = scores[best_rank]

        # 4) popular-KV migration bookkeeping (background, off the
        #    critical path).
        if config.MIG_ENABLE and self.migrator is not None:
            self._mig_note(token_ids, origin_region, max_match,
                           best_rank, q_view)

        return RouteDecision(
            rank             = best_rank,
            matched_tokens   = max_match[best_rank],
            extra_lookup_ms  = wan_per_worker[best_rank],   # WAN only (emulation)
            wan_to_worker_ms = wan_per_worker[best_rank],
            score_ms         = best_score,
            info             = {
                "lambda":               round(lam, 3),
                "max_match_per_worker": list(max_match),
                "match_tier_per_worker": list(match_tier),
                "eff_match_per_worker": [round(e, 1) for e in eff_match],
                "reload_ms_per_worker": [round(t, 2) for t in reload_ms],
                "cache_belief":         [round(b, 3) for b in load.cache_belief],
                "scores_ms":            [round(s, 2) for s in scores],
                "q_view_ms":            [round(q, 1) for q in q_view],
                "wan_per_worker_ms":    [round(w, 1) for w in wan_per_worker],
            },
        )

    # ── Migration ───────────────────────────────────────────────────────

    def _mig_note(self, token_ids, origin_region: int, max_match: list[int],
                  chosen_rank: int, q_view: list[float]) -> None:
        """Count locality-miss events; fire a background migration when a
        prefix family is hot.  Mirrors sim's ``LudoDiscovery.mig_note``."""
        fam_cut = config.LUDO_CHUNK_SIZES[0]
        if len(token_ids) < fam_cut:
            return
        fam = chunk_to_key(token_ids[:fam_cut])
        now = _now_ms()

        best_local = best_remote = 0
        local_holders = 0
        for r in range(self.n_workers):
            ml = max_match[r]
            if ml <= 0:
                continue
            if config.WORKER_REGIONS[r] == origin_region:
                local_holders += 1
                best_local = max(best_local, ml)
            else:
                best_remote = max(best_remote, ml)

        chosen_ml = max_match[chosen_rank]
        if best_local == 0 and best_remote > 0:
            kind = "cross"
        elif best_local > 0 and chosen_ml < best_local:
            if local_holders >= config.MIG_INTRA_RMAX:
                return                      # enough local replicas already
            kind = "intra"
        else:
            return

        key = (origin_region, fam, kind)
        n, start = self._mig_cnt.get(key, (0, now))
        if now - start > config.MIG_WINDOW_MS:
            n, start = 0, now
        n += 1
        self._mig_cnt[key] = (n, start)
        if n < config.MIG_HOT_K:
            return
        if now - self._mig_cool.get((origin_region, fam), -1e18) \
                < config.MIG_COOLDOWN_MS:
            return

        # Destination: an idle non-holder in the demand region with
        # migration capacity — the saturation self-quiescing gate.
        holder_ranks = {r for r in range(self.n_workers) if max_match[r] > 0}
        cands = [r for r in range(self.n_workers)
                 if config.WORKER_REGIONS[r] == origin_region
                 and r not in holder_ranks
                 and q_view[r] <= config.MIG_IDLE_DST_MS
                 and self._mig_inflight.get(r, 0) < config.MIG_MAX_INFLIGHT]
        if not cands:
            return                          # region saturated → quiesce
        dst = min(cands, key=lambda r: q_view[r])

        # Prompt-shipping re-prefill: build the full useful prefix on the
        # destination (not limited by the source's chain-quantised match).
        ml_target = min(len(token_ids), config.LUDO_MAX_CUT)
        self._mig_inflight[dst] = self._mig_inflight.get(dst, 0) + 1
        self._mig_cool[(origin_region, fam)] = now
        self._mig_cnt[key] = (0, now)
        self.mig_stats[kind] += 1
        asyncio.create_task(self._run_migration(dst, token_ids[:ml_target]))

    async def _run_migration(self, dst: int, token_slice: list[int]) -> None:
        try:
            await self.migrator(dst, token_slice)
            self.mig_stats["completed"] += 1
        except Exception:
            self.mig_stats["failed"] += 1
        finally:
            self._mig_inflight[dst] = max(0, self._mig_inflight.get(dst, 0) - 1)

    # ── Advertise / tier notifications ──────────────────────────────────

    def advertise(self, token_ids, rank, tier: int = TIER_VRAM):
        """Record that worker `rank` now holds every indexed prefix cut of
        `token_ids` at `tier`.  The holder-side cut set is the union of
        every requesting region's chain into this worker's region
        (LUDO_REGION_CUTS) — the levels-by-latency write schedule."""
        if config.MAINTENANCE_MODE == "central":
            return
        prompt_len = len(token_ids)
        for cs in config.LUDO_REGION_CUTS[config.WORKER_REGIONS[rank]]:
            if cs > prompt_len:
                break                       # cuts sorted ascending
            delta = self.table.insert(token_ids[:cs], str(rank), tier)
            if self.gossip is not None:
                self.gossip.enqueue(delta)

    def notify_tier(self, token_ids, rank: int, new_tier: int) -> None:
        if config.MAINTENANCE_MODE == "central":
            return
        prompt_len = len(token_ids)
        for cs in config.LUDO_REGION_CUTS[config.WORKER_REGIONS[rank]]:
            if cs > prompt_len:
                break
            if new_tier == TIER_NONE:
                delta = self.table.remove_worker(token_ids[:cs], str(rank))
            else:
                delta = self.table.update_tier(token_ids[:cs], str(rank), new_tier)
            if self.gossip is not None:
                self.gossip.enqueue(delta)

    def notify_evict(self, token_ids, rank: int) -> None:
        self.notify_tier(token_ids, rank, TIER_NONE)

    # ── Central-mode ingress: authority pushed Delta batch ──────────────
    def apply_delta_batch(self, wire_deltas: list[dict]) -> dict:
        if config.MAINTENANCE_MODE != "central" or self.tables is None:
            return {"ok": False, "reason": "not in central mode"}

        applied = bad = 0
        for d in wire_deltas:
            try:
                level = int(d["level"])
                delta = Delta.from_wire(d)
            except (KeyError, ValueError, TypeError):
                bad += 1
                continue
            # Authority levels index config.LUDO_CHUNK_SIZES (the worker
            # fan-out grid); translate to this lookup's cut-indexed table
            # list, which is the union of all latency-derived cuts and is
            # generally larger than the chunk grid.
            if not (0 <= level < len(config.LUDO_CHUNK_SIZES)):
                bad += 1
                continue
            tbl_idx = self._cut_level.get(config.LUDO_CHUNK_SIZES[level])
            if tbl_idx is None:
                bad += 1
                continue
            if self.tables[tbl_idx].apply_delta(delta):
                applied += 1
            else:
                bad += 1
        return {"ok": True, "applied": applied, "bad": bad,
                "received": len(wire_deltas)}

    def stats(self):
        s = super().stats()
        s["cuts_all"] = self.cuts_all
        s["pair_cuts_origin0"] = [
            config.LUDO_PAIR_CUTS[0][X][:4]
            for X in range(config.NUM_REGIONS_LOGICAL)
        ]
        s["mig"] = dict(self.mig_stats,
                        inflight={k: v for k, v in self._mig_inflight.items() if v})
        if self.tables is not None:
            agg = {
                "backend":      self.tables[0].stats()["backend"],
                "entries":      sum(t.size() for t in self.tables),
                "lww_enabled":  False,
                "value_layout": self.tables[0].stats()["value_layout"],
                "n_levels":     len(self.tables),
            }
            s.update(agg)
        else:
            s.update(self.table.stats())
        return s
