"""
workload_multitenant.py — Multi-tenant amplification of §3's mixed workload.

Motivation.  §3's single-run mixed workload has ~7.5 K distinct prefix
families — two to three orders of magnitude below what a planet-scale
inference fleet actually tracks.  We amplify the workload via the same
multi-tenant-replication idea used in §4:

  * n_tenants independent customers, each with a private 30–80-token
    system-prompt prefix (unique random tokens per tenant).
  * Each tenant internally replays §3's mixed workload (50 % TB +
    5 % LG + 45 % AAPS, Zipf α = 1.1).
  * Cross-tenant entries share no content (distinct system prompts).

With n_tenants = 2000 this produces ~10 M distinct prefix families,
matching §4's Internet-scale target.  §3 experiments that care about the
*distribution* of cache behaviour (fig3 / fig3b / fig3c) draw from this
enlarged pool so cumulative cache-hit rate is measured under realistic
prefix sparsity; queue-dynamics experiments (fig1 / fig2 / fig2b / fig4)
also sample from it but use unchanged fixed-duration windows.

Design choice — content-aware hashes that preserve within-tenant sharing.
Discovery-plane schemes consume two opaque IDs per request:
``content_keys[cs]`` (one per Ludo level) and ``radix_path`` (the
RadixTree gate walk).  To preserve the real within-dataset sharing
structure (TB's 6,000 entries share their 320-token system prompt,
AAPS's problem groups share ~244 tokens, etc.) the amplified hashes
must *match* when two (tenant, base_pid) pairs share the same first-cs
tokens, and *differ* when they don't.

We achieve this cheaply by composing per-tenant salt with the base
generator's precomputed content keys:

    content_keys[cs] = hash((tenant_id, base._content_keys[base_pid][cs]))
    radix_path [k]   = hash((tenant_id, base._radix_paths[base_pid][k])) mod RADIX_HASH_MOD

Two requests with the same ``(tenant_id, base_pid)`` get identical keys
(intra-entry reuse → cache hit).  Two requests from the same tenant but
different base_pid get the *same* key at any cs where the base dataset
naturally shares content (preserves TB's system-prompt collision, which
is the mechanism driving hash-scheme DRAM demotion at scale).  Two
requests from different tenants never collide at any level because the
tenant-id salt differs.

Both operations are O(1) dict-lookup + 2-tuple hash; the per-request
cost is a handful of nanoseconds and does not bottleneck the 10 M-
request single-pass even in pure Python.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# We import from the sibling simulate.py module that owns Request,
# CHUNK_SIZES, RADIX_GATE_CHUNKS / RADIX_CHUNK_TOKENS / RADIX_HASH_MOD,
# and the base WorkloadGenerator class.
import sys
HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from simulate import (                                           # type: ignore
    Request, WorkloadGenerator,
    CHUNK_SIZES, RADIX_GATE_CHUNKS, RADIX_CHUNK_TOKENS, RADIX_HASH_MOD,
    NUM_REGIONS, REGION_REQUEST_WEIGHTS,
)


class MultiTenantWorkloadGenerator:
    """Drop-in replacement for §3's WorkloadGenerator with tenant amplification.

    Usage:
        base  = WorkloadGenerator(seed=42)
        multi = MultiTenantWorkloadGenerator(base, n_tenants=2000, seed=42)
        req   = multi.sample(req_id=i, current_time=t, workload='mixed')

    The returned Request has a globally unique ``prefix_id`` =
    ``tenant_id × BASE_POOL + base_pid``; content_keys and radix_path are
    synthetic hashes keyed on ``(tenant_id, base_pid)`` so the discovery
    plane treats every (tenant, base_entry) pair as a distinct content
    family.  ``prefix_len`` adds the tenant prefix length to the base's
    real prefix length so downstream cost / match-length calculations see
    the full pre-tokenised length.
    """

    # Tenant prefix length range — kept short of the RadixTree gate
    # (gate × chunk = 100) so a hypothetical real-token radix walk would
    # still create per-base-entry leaves; chosen to match §4's setting.
    TENANT_PREFIX_MIN_TOKENS = 30
    TENANT_PREFIX_MAX_TOKENS = 80

    def __init__(self, base: WorkloadGenerator,
                 n_tenants: int,
                 seed: int = 42):
        self._base       = base
        self.n_tenants   = max(1, int(n_tenants))

        # Tenant-id shift for constructing globally unique prefix_ids:
        # take the largest base prefix_id and pick the next power-of-10
        # above it as the multiplier so IDs remain human-readable.
        max_base_pid = 0
        for ids, _, _ in base.pools.values():
            if len(ids):
                max_base_pid = max(max_base_pid, int(ids.max()))
        base_span = max_base_pid + 1
        # Round up to 10^k so tenant 7 starts at 7 × 10_000 etc.
        import math
        self._tenant_shift = 10 ** int(math.ceil(math.log10(max(base_span, 10))))

        # Per-tenant prefix length (random in [30, 80]).  Prefix *tokens*
        # are never materialised — we use synthetic hashes that bypass
        # the content, so only the *length* is needed for prefix_len
        # reporting.
        tenant_rng = np.random.default_rng(seed + 101)
        self._tenant_plens = tenant_rng.integers(
            self.TENANT_PREFIX_MIN_TOKENS,
            self.TENANT_PREFIX_MAX_TOKENS + 1,
            size=self.n_tenants, dtype=np.int32)

        # A second RNG drives tenant selection per sample() call; kept
        # independent of the base generator's RNG so tenant sampling
        # doesn't perturb the base's Zipf reproducibility.
        self._tenant_rng = np.random.default_rng(seed + 211)

    # Expose `.rng` so simulator code that does ``wgen.rng`` still works.
    @property
    def rng(self):
        return self._base.rng

    def reset_stream(self, seed: int):
        """Paired-comparison reset: re-seed both the base workload stream and
        the tenant-selection stream (see WorkloadGenerator.reset_stream)."""
        self._base.reset_stream(seed)
        self._tenant_rng = np.random.default_rng(seed + 211)

    # ── Public sampling API ──────────────────────────────────────────────

    def sample(self, request_id: int, current_time: float,
               workload: str = 'mixed') -> Request:
        # Reuse the base generator's Zipf sampling to get a base entry.
        base_req = self._base.sample(request_id, current_time, workload)
        base_pid = int(base_req.prefix_id)

        # Pick a tenant uniformly.  Real deployments have Zipf-skewed
        # tenant popularity too (a handful of customers drive the bulk
        # of traffic), but uniform tenant sampling is the harder test
        # case for discovery — it maximises prefix spread across the
        # pool and puts the most pressure on the cache-hit rate.
        tenant_id = int(self._tenant_rng.integers(0, self.n_tenants))
        global_pid = tenant_id * self._tenant_shift + base_pid
        tenant_plen = int(self._tenant_plens[tenant_id])

        # Content-aware keys — reuse the base generator's precomputed
        # per-entry content hashes so within-tenant dataset sharing is
        # preserved (see module docstring).
        base_keys   = self._base._content_keys.get(base_pid, {})
        base_rpath  = self._base._radix_paths.get(base_pid, [])
        # Salt every base key length — CHUNK_SIZES levels plus Ludo's
        # latency-derived cut lengths (whatever the base precompute emitted).
        content_keys = {
            cs: hash((tenant_id, v)) & 0xFFFFFFFF
            for cs, v in base_keys.items()
        }
        radix_path = [
            hash((tenant_id, h)) % RADIX_HASH_MOD
            for h in base_rpath
        ]

        return Request(
            request_id   = request_id,
            prefix_id    = int(global_pid),
            prefix_len   = int(base_req.prefix_len) + tenant_plen,
            gen_len      = int(base_req.gen_len),
            arrival_time = current_time,
            region_id    = int(base_req.region_id),
            content_keys = content_keys,
            radix_path   = radix_path,
        )


# ── Corpus helper for fig3 / fig3b single-pass 10M replay ────────────────────
#
# fig3 / fig3b in run_section3.py run their own content-aware cache loop
# (`_simulate_fig3`) over a list of request dicts, not via the main
# Simulator.  Those dicts need `prefix_len`, `hashes`, `radix_path`,
# `prefix_blocks`.  For the 10 M multi-tenant replay we pre-generate the
# shuffled request list once (keeps the eventual cache behaviour
# deterministic across schemes).


class MultiTenantFig3Stream:
    """Streaming generator for fig3/fig3b single-pass 10 M replay.

    Yields one request dict per iteration without holding the full list
    in memory — crucial at 10 M × 2-3 KB/req ≈ 20 GB which otherwise
    would not fit.  Iteration is deterministic (re-seeded from
    ``seed + 1`` on each ``__iter__`` call) so the same stream can be
    consumed once per scheme and every scheme observes identical
    requests in identical order.

    Uses the full-depth radix_path / content_keys from the same
    `_build_content_corpus` routine that populated single-tenant fig3,
    so the per-entry prefix_blocks list can reach ~145 chunks for
    ToolBench prompts and the Ludo-level pos values (12/25/51/102/204/
    409) are all within reach.  Each chunk is salted with tenant_id so
    cross-tenant requests never collide while within-tenant
    within-dataset requests (e.g. two TB entries in the same tenant)
    share the base-derived first-N chunks that correspond to their
    shared system prompt.
    """

    def __init__(self, n_tenants: int, n_requests: int, seed: int = 42,
                 zipf_alpha: float = 1.1):
        self.n_tenants  = n_tenants
        self.n_requests = n_requests
        self.seed       = seed
        self.zipf_alpha = zipf_alpha

        # Build the full-depth corpus once (deep radix_path, hashes,
        # prefix_blocks per distinct base entry) via run_section3's
        # helper.  ~30 MB RAM, 7.5 K entries.
        import run_section3                                       # type: ignore
        self._corpus = run_section3._build_content_corpus()

        # Flatten corpus into parallel arrays keyed by (dataset,
        # in-dataset rank) for O(1) Zipf sampling.
        self._ds_names      = ('toolbench', 'loogle', 'aaps')
        self._ds_mix_weight = np.array([0.50, 0.05, 0.45])
        self._ds_mix_cum    = np.cumsum(self._ds_mix_weight)

        self._ds_entries: Dict[str, List[dict]] = {}
        self._ds_zipf_cum: Dict[str, np.ndarray] = {}
        for ds in self._ds_names:
            entries = self._corpus[ds]
            self._ds_entries[ds] = entries
            n = len(entries)
            w = np.arange(1, n + 1, dtype=float) ** (-zipf_alpha)
            w = w / w.sum()
            self._ds_zipf_cum[ds] = np.cumsum(w)

        # Per-tenant prefix length (tokens added to prefix_len only —
        # synthetic, never materialised).
        tenant_rng = np.random.default_rng(seed + 101)
        self._tenant_plens = tenant_rng.integers(
            MultiTenantWorkloadGenerator.TENANT_PREFIX_MIN_TOKENS,
            MultiTenantWorkloadGenerator.TENANT_PREFIX_MAX_TOKENS + 1,
            size=n_tenants, dtype=np.int32)

    def __len__(self) -> int:
        return self.n_requests

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1)
        ARRIVAL_RATE_MS = 1.0        # 1000 req/s nominal — arrival_time is
                                      # only used for tie-breaking in fig3's
                                      # routing loop; fig3 is not queue-dynamics.
        t = 0.0
        tenant_plens = self._tenant_plens
        ds_names     = self._ds_names
        ds_mix_cum   = self._ds_mix_cum
        ds_entries   = self._ds_entries
        ds_zipf_cum  = self._ds_zipf_cum
        n_tenants    = self.n_tenants

        for _ in range(self.n_requests):
            # 1) pick dataset by mix.
            ds_idx = int(np.searchsorted(ds_mix_cum, rng.random()))
            if ds_idx >= len(ds_mix_cum):
                ds_idx = len(ds_mix_cum) - 1
            ds = ds_names[ds_idx]
            entries = ds_entries[ds]

            # 2) pick base entry Zipf within the dataset.
            idx = int(np.searchsorted(ds_zipf_cum[ds], rng.random()))
            if idx >= len(entries):
                idx = len(entries) - 1
            info = entries[idx]

            # 3) pick tenant uniformly.
            tenant_id   = int(rng.integers(0, n_tenants))
            tenant_plen = int(tenant_plens[tenant_id])

            # 4) salt the precomputed full-depth content_keys /
            # radix_path / prefix_blocks with the tenant id.  Salting
            # preserves intra-dataset sharing (two TB entries in same
            # tenant still share their base prefix_blocks[0..k] for
            # k where their first 10(k+1) tokens match on disk) and
            # forces cross-tenant isolation (different tenant_id ⇒
            # different salted hash at every position).
            hashes = {
                cs: hash((tenant_id, v)) & 0xFFFFFFFF
                for cs, v in info['hashes'].items()
            }
            radix_path = [
                hash((tenant_id, h)) % RADIX_HASH_MOD
                for h in info['radix_path']
            ]
            _rh = 0
            prefix_blocks: List[int] = []
            for c in radix_path:
                _rh = (_rh * 2654435761 ^ c) & ((1 << 64) - 1)
                prefix_blocks.append(_rh)

            t += rng.exponential(1.0 / ARRIVAL_RATE_MS)

            yield {
                'prefix_len':    info['prefix_len'] + tenant_plen,
                'hashes':        hashes,
                'radix_path':    radix_path,
                'prefix_blocks': prefix_blocks,
                'region_id':     int(rng.choice(NUM_REGIONS,
                                                p=REGION_REQUEST_WEIGHTS)),
                'arrival_time':  t,
                'dataset':       ds,   # for Ludo's workload-aware eviction
            }


def build_multitenant_fig3_requests(
    n_tenants: int,
    n_requests: int,
    seed: int = 42,
    zipf_alpha: float = 1.1,
):
    """Return a re-iterable stream of fig3 requests (see MultiTenantFig3Stream).

    `_simulate_fig3` iterates the stream once per scheme; the returned
    object supports multiple independent iterations at the cost of
    re-running the per-request generation loop each time (deterministic
    via the fixed seed).  The alternative — materialising 10 M × 2 KB
    requests as a list — exceeds commodity RAM; streaming sidesteps
    that at the cost of ~5× total generation work across 6 schemes.
    """
    return MultiTenantFig3Stream(
        n_tenants=n_tenants, n_requests=n_requests,
        seed=seed, zipf_alpha=zipf_alpha)
