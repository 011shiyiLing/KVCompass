"""
RadixTree scheme — PlanetServe **basic mode**, paper-faithful.

Reproduces PlanetServe's published HashRadixTree design exactly,
without the load-aware enhancements the §3 simulator's
``RadixTreeDiscovery.planetserve_route`` adds for "reproducibility"
(see simulate.py:1364-1365 — *"PlanetServe uses random.choice; we
use least-loaded for reproducibility"*). The simulator's variant is an
**advanced** mode the published code does not implement; using it on
real hardware silently borrows load-balancing the paper does not
claim.

The three PlanetServe-defining mechanisms (this file):

  (1) STRICT gate-depth hit condition
      A request only counts as a cache hit when it walks all
      ``GATE`` (= 10) chunks down the trie (100-token minimum).
      Prompts that match fewer chunks are treated as a miss — even
      if a partial-prefix worker could have served some of the
      prefill. Matches PlanetServe's ``find_match_model``.

  (2) Lossy chunk hash (``mod HASH_MOD = 15``)
      Each 10-token chunk is hashed to an integer in [0, 15).
      Different prompts collide on the same trie path frequently
      — by design — trading cache precision for memory frugality.
      Collisions make trie owner sets noisy: a worker registered for
      prompt A may be picked for prompt B that shares no real content.

  (3) Single-leaf advertise
      A worker is recorded only at depth
      ``min(GATE, cached_len // CHUNK)`` — typically GATE for any
      prompt ≥ 100 tokens. Interior trie nodes never accumulate
      owners, so partial-prefix matches cannot benefit from the
      index even when one exists.

PlanetServe-basic routing (this file, strict 1:1 with
`local_hrt_vllm_wo_lb.py` + `HashRadixTree.py`):

  Hit  : ``owners[0]`` — first-ever owner of this hash slot. The
         published `find_match_model` computes a min(pending_tasks)
         candidate but then `return current.model_list[0]`, so all
         hits funnel to the first registrant deterministically.
  Miss : ``random.choice(candidate_models)`` — published behaviour.
  Hash : polynomial ``(hash * 31 + tid) % 15``, which degenerates to
         ``sum(tids) % 15`` (31 mod 15 = 1) — position-insensitive,
         collision-heavy by design.
  No Phase-2 80th-percentile load-balance climb (advanced-mode only).

The §3 simulator runs the *advanced* variant of PlanetServe with
``min(pending_ms)`` + 80th-percentile climb; that is fine for the
simulator's own internal comparisons but is not the published
baseline. On real hardware we report the basic mode so the
comparison against Ludo is on the design PlanetServe actually
publishes.
"""
from __future__ import annotations

import random

from ludo_sglang import config
from ludo_sglang.router.load_tracker import WorkerLoad
from ludo_sglang.router.schemes.base import RouteDecision, Scheme


# ── Trie node ──────────────────────────────────────────────────────────
def _new_trie_node(parent: dict | None = None, depth: int = 0) -> dict:
    """Trie node mirroring PlanetServe's HashRadixNode: children dict +
    insertion-ordered owner list (model_list) + back-link. Owners is a
    list — not a set — so the first-registered worker stays at index 0,
    which is the deterministic target the published hit path returns
    (HashRadixTree.py:153: `return current.model_list[0], current`)."""
    return {"children": {}, "owners": [], "parent": parent, "depth": depth}


class RadixTreeScheme(Scheme):
    name = "radix_tree"

    # Sim-faithful PlanetServe parameters (simulate.py:180-182,
    # 1167-1169): GATE=10 chunks × 10 tokens = 100-token minimum match,
    # HASH_MOD=15 lossy fingerprint. These are the published
    # PlanetServe defaults — tightening them shifts the scheme away
    # from "hits often, single-leaf advertise concentrates load" toward
    # "rarely hits, falls back to least-loaded", which silently buys
    # PlanetServe better load balance than the simulation ever grants
    # it.
    GATE: int = 10              # 100-token minimum match
    CHUNK_TOKENS: int = 10      # PlanetServe original
    HASH_MOD: int = 15          # PlanetServe original
    # Phase-2 climb is advanced-mode only and is omitted in basic mode
    # (the simulator's planetserve_route adds it; the published
    # PlanetServe code does not).

    def __init__(self, n_workers: int):
        super().__init__(n_workers)
        self.root: dict = _new_trie_node()
        self._n_hits: int = 0
        self._n_routes: int = 0

    # ── Lossy chunk hash (1:1 PlanetServe HashRadixTree.hash_chunk) ─────
    def _h(self, toks: list[int]) -> int:
        # PlanetServe's hash_chunk (HashRadixTree.py:66-76):
        #     hash_val = 0
        #     for tid in chunk:
        #         hash_val = (hash_val * 31 + tid) % 15
        # Because 31 mod 15 = 1, this degenerates to sum(tids) % 15 —
        # position-insensitive and very collision-heavy by design.
        hash_val = 0
        for tid in toks:
            hash_val = (hash_val * 31 + tid) % self.HASH_MOD
        return hash_val

    def _path(self, token_ids: list[int]) -> list[int]:
        """GATE-length chunk-hash path. May be shorter than GATE if the
        prompt has fewer than GATE × CHUNK_TOKENS tokens — that is the
        max_depth cap sim applies."""
        out: list[int] = []
        max_chunks = min(self.GATE, len(token_ids) // self.CHUNK_TOKENS)
        for k in range(max_chunks):
            chunk = token_ids[k * self.CHUNK_TOKENS:(k + 1) * self.CHUNK_TOKENS]
            out.append(self._h(chunk))
        return out

    async def route(self, token_ids, load: WorkerLoad, origin_region: int,
                    ctx: dict | None = None):
        self._n_routes += 1
        path = self._path(token_ids)

        # ── Phase 1: trie traversal ──────────────────────────────────────
        cur = self.root
        depth = 0
        for h in path:
            child = cur["children"].get(h)
            if child is None:
                break
            cur = child
            depth += 1

        # PlanetServe strict-GATE hit: only depth >= GATE counts as a
        # cache hit. Anything less, even with a partial trie walk, is
        # treated as a miss (the trie is *registered*, but no match
        # length is claimed).
        is_hit = (depth >= self.GATE) and bool(cur["owners"])

        if is_hit:
            self._n_hits += 1
            owners = cur["owners"]
            # PlanetServe basic-mode hit (1:1 with
            # HashRadixTree.find_match_model): always return
            # `current.model_list[0]`, the first-ever owner of this
            # hash-slot path. No load distribution among owners — the
            # whole hash slot is permanently bound to its first
            # registrant. This is what concentrates load and ultimately
            # caps PlanetServe-basic's tail latency.
            best = owners[0]

            matched = self.GATE * self.CHUNK_TOKENS
            wan_ms = config.wan_ms_to_worker(origin_region, best)
            residual = max(0, len(token_ids) - matched)
            score_ms = (load.pending_ms[best]
                        + residual * load.prefill_rate[best]
                        + wan_ms)
            return RouteDecision(
                rank             = best,
                matched_tokens   = matched,
                extra_lookup_ms  = wan_ms,
                wan_to_worker_ms = wan_ms,
                score_ms         = score_ms,
                info             = {"path_depth_hit": depth,
                                    "owners_at_leaf": len(owners),
                                    "phase":          "hit"},
            )

        # ── Cache miss: PlanetServe basic-mode picks a random worker.
        # The §3 simulator's note (simulate.py:1364-1365) explicitly
        # states this is what published PlanetServe does, and that the
        # simulator deviates by using least-loaded for reproducibility.
        # Real-machine baseline runs the published behaviour.
        best = random.randrange(self.n_workers)
        wan_ms = config.wan_ms_to_worker(origin_region, best)
        residual = len(token_ids)
        score_ms = (load.pending_ms[best]
                    + residual * load.prefill_rate[best]
                    + wan_ms)
        return RouteDecision(
            rank             = best,
            matched_tokens   = 0,
            extra_lookup_ms  = wan_ms,
            wan_to_worker_ms = wan_ms,
            score_ms         = score_ms,
            info             = {"path_depth_hit": depth,
                                "owners_at_leaf": 0,
                                "phase":          "miss"},
        )

    # ── advertise: insert at LEAF depth only, sim semantics ─────────────
    def advertise(self, token_ids, rank):
        target_depth = min(self.GATE, len(token_ids) // self.CHUNK_TOKENS)
        if target_depth == 0:
            return
        path = self._path(token_ids)[:target_depth]
        cur = self.root
        for h in path:
            child = cur["children"].get(h)
            if child is None:
                child = _new_trie_node(parent=cur, depth=cur["depth"] + 1)
                cur["children"][h] = child
            cur = child
        # Single-leaf insertion mirroring PlanetServe's
        # `assign_workload(modelnode, hrt_node)` (HashRadixTree.py:56-63):
        # only the deepest node records the worker, and duplicates are
        # filtered (`if modelnode not in hrt_node.model_list`). The
        # list-with-dedupe (not set) preserves insertion order, which
        # the hit path depends on.
        if rank not in cur["owners"]:
            cur["owners"].append(rank)

    # ── Stats ──────────────────────────────────────────────────────────
    def stats(self):
        s = super().stats()
        # Walk the tree once to count nodes / owner distribution.
        nodes = 0
        leaves_with_owners = 0
        multi_owner_leaves = 0
        stack = [self.root]
        while stack:
            n = stack.pop()
            nodes += 1
            if n["owners"]:
                leaves_with_owners += 1
                if len(n["owners"]) > 1:
                    multi_owner_leaves += 1
            stack.extend(n["children"].values())
        s["tree_nodes"]          = nodes
        s["leaves_with_owners"]  = leaves_with_owners
        s["multi_owner_leaves"]  = multi_owner_leaves
        s["routing_hit_rate"]    = (self._n_hits / self._n_routes
                                    if self._n_routes else 0.0)
        return s
