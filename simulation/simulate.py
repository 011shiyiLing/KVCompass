"""
simulate.py — Core simulation framework for the large-scale evaluation.

Implements a trace-driven, event-based simulation of an internet-scale
KV cache discovery plane for LLM serving. No actual GPU inference is run;
serving nodes are modeled by cost functions derived from Llama-3.1-8B on A100
(or 70B with SIM_MODEL=70b).

Routing schemes implemented (run_section3.py's default figure set runs the
starred ones):
  ludo             — multi-level Ludo table (ours) *
  ludo_coarse      — ludo with the coarse chunk grid (ablation baseline) *
  central_directory— centralized coordinator paying a WAN RTT per lookup *
  radix_tree       — PlanetServe-style single-level index *
  consistent_hash  — static hash-ring affinity *
  chord            — multi-hop DHT lookup *
  skywalker        — per-region LBs with stale-view selective pushing *
  dynamo           — per-region cache-aware routers, periodic broadcast *
  random_ll        — random / least-loaded (no cache awareness)
  kademlia         — DHT with hash placement (no migration)
  centralized      — perfect zero-latency centralized tracker
  oracle           — omniscient upper bound
"""

import bisect
import heapq
import json
import math
import os
import sys
import numpy as np
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Real Ludo backend (realmachine) ────────────────────────────────
# The ludo scheme's routing table can run on the real-machine Ludo hashing
# stack (Smash ControlPlaneLudo via src/ludo/ludo_table.LudoTable) instead of
# a plain Python dict, so the simulation exercises the exact data structure
# deployed on the testbed.  Layout: ONE LudoTable PER HOLDER REGION (its
# workers = that region's nodes, ≤30 ≤ 32 slots of the uint64 value); every
# router is assumed to hold an up-to-date mirror of all 8 region tables
# (delta dissemination is instantaneous, matching the dict backend's
# semantics).  Select with LUDO_TABLE_BACKEND=native|dict (default native
# when the compiled binding imports).
_REALMACHINE_ROOT = Path(__file__).resolve().parents[1] / "realmachine"
try:
    sys.path.insert(0, str(_REALMACHINE_ROOT))
    from src.ludo.ludo_table import (            # noqa: E402
        LudoTable as _RealLudoTable,
        TIER_VRAM as _RL_VRAM,
        TIER_DRAM as _RL_DRAM,
    )
    _HAS_REAL_LUDO = True
except ImportError:
    _HAS_REAL_LUDO = False

LUDO_TABLE_BACKEND = os.environ.get(
    "LUDO_TABLE_BACKEND", "native" if _HAS_REAL_LUDO else "dict")
if LUDO_TABLE_BACKEND == "native" and not _HAS_REAL_LUDO:
    raise ImportError(
        "LUDO_TABLE_BACKEND=native but realmachine's ludo_table "
        f"is not importable from {_REALMACHINE_ROOT} (build the binding "
        "with src/ludo/setup.py build_ext --inplace)")

# ─── Cluster constants ────────────────────────────────────────────────────────

# Regions and node placement are drawn from the real LLM-deployment census
# (datasets/llm_deployment): the eight most-served countries, with each
# region's serving capacity (node count) proportional to its share of the
# 320,102-endpoint census.  This replaces the earlier uniform 8×8 layout.
#   share:  US 46.1%  CN 19.3%  DE 12.2%  JP 5.2%  SG 5.2%  GB 5.1%  KR 3.6%  AU 3.4%
REGION_NAMES     = ['US', 'CN', 'DE', 'JP', 'SG', 'GB', 'KR', 'AU']
NUM_REGIONS      = len(REGION_NAMES)                              # 8
# Node count per region ∝ deployment share (largest-remainder rounding to 64).
NODES_PER_REGION_LIST = [30, 13, 8, 3, 3, 3, 2, 2]               # sum = 64
NUM_NODES        = sum(NODES_PER_REGION_LIST)                     # 64
# node_id → region_id map: regions occupy contiguous node-id blocks
#   US=[0..29] CN=[30..42] DE=[43..50] JP=[51..53] SG=[54..56]
#   GB=[57..59] KR=[60..61] AU=[62..63]
NODE_REGION = [r for r, k in enumerate(NODES_PER_REGION_LIST) for _ in range(k)]
# Cumulative start offset of each region's node block (for grouping / plots).
REGION_OFFSETS = [sum(NODES_PER_REGION_LIST[:r]) for r in range(NUM_REGIONS)]

# Multi-level Ludo chunk sizes (tokens)
CHUNK_SIZES = [128, 256, 512, 1024, 2048, 4096]

# SkyWalker heartbeat probe interval (ms).  Each load balancer refreshes its
# view of replica / peer-LB availability only once per interval; between
# probes it routes on a *stale* availability snapshot (the paper sets 100 ms).
SKYWALKER_PROBE_INTERVAL_MS = 100.0

# ─── Cost model: Llama-3.1 {8B on 1×A100 | 70B on 4×A100 TP4} (BF16) ─────────
# Numbers below are calibrated against public vLLM / SGLang benchmarks in
# BF16.  Prior overly-optimistic values have been tightened to reflect real
# deployed throughput.  SIM_MODEL selects the model-size-dependent constants.

# Model selector (SIM_MODEL env): '8b' — Llama-3.1-8B on 1×A100-80GB (the
# historical calibration below); '70b' — Llama-3.1-70B on one 4×A100-80GB
# TP4 node.  Every cost constant that depends on model size switches with
# it; geometry (regions, node counts, WAN) is model-independent.  Default
# stays '8b' so existing cached results/figures keep their meaning — pass
# SIM_MODEL=70b explicitly for the 70B experiments.
SIM_MODEL = os.environ.get('SIM_MODEL', '8b').lower()
if SIM_MODEL not in ('8b', '70b'):
    raise ValueError(f"SIM_MODEL must be '8b' or '70b', got {SIM_MODEL!r}")

# Prefill throughput.
#   8b : ~20 K tokens/s sustained at large batch on A100 BF16.
#        2 × 8B params ≈ 16 GFLOPs/token, A100 BF16 peak 312 TFLOPs/s with
#        ~70-80 % realised utilisation gives 64-80 μs/token; the rounded
#        0.05 ms/token matches vLLM's published prefill numbers on Llama-3-8B.
#   70b: 2 × 70B ≈ 140 GFLOPs/token on 4×A100 TP4 (1.248 PFLOPs/s peak,
#        ~75 % util) ≈ 0.15 ms/token (~6.7 K tokens/s/node) — matches
#        vLLM/SGLang Llama-3-70B TP4 A100 prefill benchmarks.
C_PREFILL_MS_PER_TOKEN = 0.05 if SIM_MODEL == '8b' else 0.15

# SGLang chunked prefill: long prefills execute as chunks of this many tokens
# (SGLang's chunked_prefill_size default on A100-80GB-class GPUs); decode
# iterations interleave at chunk granularity, so a co-resident decoding
# request's token gap is stretched by at most one chunk's compute
# (8192 × C_PREFILL ≈ 410 ms), longer prefills spilling into consecutive
# gaps.  Used by the TBT measurement model only.
SGLANG_CHUNK_TOKENS = 8192

# Per-token KV-cache footprint in BF16:
#   2 (K, V) × n_layers × n_kv_heads (8, GQA) × head_dim (128) × 2 B.
#   8b : 32 layers → 131,072 B = 128 KB/token.
#   70b: 80 layers → 327,680 B = 320 KB/token.
# (Pre-Oct-2025 versions of this file erroneously used 512 B, which made the
# per-node cache capacity and the PCIe reload cost off by 256×.)
_N_LAYERS          = 32 if SIM_MODEL == '8b' else 80
KV_BYTES_PER_TOKEN = 2 * _N_LAYERS * 8 * 128 * 2

# Continuous batching: each node serves up to MAX_BATCH_SIZE requests
# concurrently, modelled as MAX_BATCH_SIZE parallel service slots (an
# M/D/c queue, c = MAX_BATCH_SIZE).  A node's sustained throughput is
# therefore ≈ MAX_BATCH_SIZE × the single-slot rate.
MAX_BATCH_SIZE = 4

# Node VRAM budget for KV cache — 40 % of the node's aggregate HBM (the
# rest is weights + activations), consistent across models:
#   8b : 1×80 GB, ~16 GB weights → 32 GiB KV ÷ 128 KB ≈ 262 K tokens.
#   70b: 4×80 GB TP4, ~140 GB weights + activations → 128 GiB KV ÷ 320 KB
#        ≈ 419 K tokens.
_KV_VRAM_GIB     = 32 if SIM_MODEL == '8b' else 128
NODE_VRAM_TOKENS = (_KV_VRAM_GIB * 1024 ** 3) // KV_BYTES_PER_TOKEN

# Node DRAM (CPU memory) budget: 4× VRAM so evicted KV blocks have a
# landing zone before being fully discarded (8b: 128 GB host RAM;
# 70b: 512 GB — standard for a 4-GPU A100 server).  Entries here can still
# be served with a PCIe reload penalty instead of a full recompute.
NODE_DRAM_TOKENS = NODE_VRAM_TOKENS * 4

# PCIe DRAM→VRAM reload cost.  A100 uses PCIe 4.0 x16: theoretical 32 GB/s,
# sustained ≈ 25 GB/s with pinned host memory per GPU — matches the real
# measured throughput for KV-cache reloads on vLLM / SGLang.  Under TP4 the
# KV cache is sharded across 4 GPUs, so reloads run 4 lanes in parallel.
#   8b : 128 KB / 25 GB/s   ≈ 5.24 μs/token
#   70b: 320 KB / 100 GB/s  ≈ 3.28 μs/token
PCIE_RELOAD_BANDWIDTH_GBPS = 25 if SIM_MODEL == '8b' else 100      # GB/s
DRAM_RELOAD_MS_PER_TOKEN   = (
    KV_BYTES_PER_TOKEN / (PCIE_RELOAD_BANDWIDTH_GBPS * 1e9) * 1e3  # ≈ 0.0052 ms/token
)

# First-forward-pass attention cost per input token.
# Every request — cache hit or miss — must run one forward pass over the
# entire prompt context to emit the first output token.  FFN + projection
# FLOPs are already charged by C_PREFILL for the residual (uncached) tokens;
# what remains is the attention term, which scales with the full prefix
# length regardless of cache status (the model attends over every token in
# the context on every step).  We charge ≈ 0.003 ms/token for this
# attention-only work — the geometric calibration point is attention ≈ 30 %
# of a full prefill's FLOPs, applied to the realistic 0.05 ms/token prefill
# cost (0.003 ≈ 0.05 × 0.6 × 0.1, rounded to a mid-range empirical value for
# A100 BF16 flash-attention-2 on Llama-3.1-8B).
#
# Non-zero value is also required for the oracle to exhibit realistic queue
# dynamics: with C_DECODE = 0 a fully cached request would have zero service
# time and oracle TTFT would be identically zero at every arrival rate — an
# unphysical flat line on log-y axes.
# Scales with C_PREFILL (attention share of the forward pass is model-
# proportional): 8b 0.003, 70b 0.009 ms/token.
C_DECODE_MS_PER_TOKEN = 0.003 if SIM_MODEL == '8b' else 0.009

# Per-step autoregressive decode cost (time to generate one output token).
# Decode is memory-bandwidth-bound: each step reads all model weights once.
#   8b : vLLM / SGLang on Llama-3-8B at A100-80GB batch=16 report ~10-15 ms
#        per forward step ≈ 0.6-1.0 ms per batched token → 0.8 ms/token.
#   70b: TP4 reads 140 GB weights over 4×2.04 TB/s HBM (~70 % realised)
#        ≈ 25-35 ms per step at batch 16 ≈ 1.6-2.2 ms/token → 2.0 ms/token,
#        consistent with published Llama-70B TP4 A100 decode throughput.
C_DECODE_STEP_MS_PER_TOKEN = 0.8 if SIM_MODEL == '8b' else 2.0

# TTFT SLO (1.5 s is a common production target for interactive LLM serving)
TTFT_SLO_MS = 1500.0

# WAN one-way latency matrix (ms) between the eight deployment regions.
# Derived from the great-circle distance between each region's primary
# data-center hub (US=Ashburn, CN=Beijing, DE=Frankfurt, JP=Tokyo,
# SG=Singapore, GB=London, KR=Seoul, AU=Sydney) via a standard propagation
# model:  one-way ≈ distance_km / 200 (fiber ≈ 200 km/ms) × 1.5 (routing
# detour) + 5 ms (edge/switching).  Region order matches REGION_NAMES.
WAN_LATENCY_MS = np.array([
    [  0,  88,  54,  87, 121,  49,  89, 123],  # US
    [ 88,   0,  63,  21,  39,  66,  12,  72],  # CN
    [ 54,  63,   0,  75,  82,  10,  69, 129],  # DE
    [ 87,  21,  75,   0,  45,  77,  14,  64],  # JP
    [121,  39,  82,  45,   0,  86,  40,  52],  # SG
    [ 49,  66,  10,  77,  86,   0,  71, 132],  # GB
    [ 89,  12,  69,  14,  40,  71,   0,  68],  # KR
    [123,  72, 129,  64,  52, 132,  68,   0],  # AU
], dtype=float)

# Request-origin distribution over the eight regions, from the WildChat
# *demand* census (scripts/wildchat_analysis/wc_stats.json country_counts,
# normalised over these eight countries).  Requests are sampled per-region
# with these weights so that traffic — like deployment — is geographically
# skewed and decoupled from local capacity (e.g. CN and GB are demand-heavy
# relative to their node share, DE/SG/KR/AU are supply-heavy).
#   US 45.3%  CN 26.9%  GB 8.2%  DE 7.7%  JP 4.7%  SG 3.6%  AU 2.1%  KR 1.6%
REGION_REQUEST_WEIGHTS = np.array(
    [0.4534, 0.2689, 0.0770, 0.0467, 0.0356, 0.0819, 0.0156, 0.0209])
REGION_REQUEST_WEIGHTS = REGION_REQUEST_WEIGHTS / REGION_REQUEST_WEIGHTS.sum()

# ─── Latency-aware reuse floor (Ludo multi-level) ────────────────────────────
# Break-even prefix length per (request_region, holder_region) pair: a remote
# cache is worth reusing only when the prefill it saves (match_len × C_PREFILL)
# covers the extra network latency of serving there:
#     save(n)   = n × C_PREFILL_MS_PER_TOKEN
#     extra_net = WAN[rr][nr]                     # prompt → holder (prefill)
#               + REUSE_RETURN_FACTOR × WAN[rr][nr]  # first token back (decode)
#     n_floor   = extra_net / C_PREFILL           # break-even prefix length
# This floor is embedded structurally as level 1 of the latency-derived Ludo
# cut chains (LUDO_PAIR_CUTS below); level k = k·n_floor covers k× the latency.
# Same-region (rr == nr) → floor 0, local reuse counts from CHUNK_SIZES[0].
#
# REUSE_RETURN_FACTOR calibrates the floor's metric: TTFT charges only the
# forward WAN leg (the decode return leg lands in e2e), so 0 makes the floor
# the exact TTFT break-even (n₁ = 20·WAN @ 0.05 ms/token) — consistent with
# the joint-scoring T_t that routing itself optimises.  Setting 0.5 charges
# the WAN/2 return leg too (e2e break-even, n₁ = 30·WAN), a stricter floor
# that refuses reuse in the [20·WAN, 30·WAN) band even though it improves TTFT.
REUSE_RETURN_FACTOR = 0.0   # TTFT-calibrated floor, consistent with scoring

def _reuse_floor_tokens(rr: int, nr: int) -> float:
    if rr == nr:
        return 0.0
    wan = float(WAN_LATENCY_MS[rr][nr])
    return (wan + REUSE_RETURN_FACTOR * wan) / C_PREFILL_MS_PER_TOKEN

# Precomputed break-even prefix length (tokens) for every region pair.
REUSE_FLOOR_TOKENS = np.array(
    [[_reuse_floor_tokens(rr, nr) for nr in range(NUM_REGIONS)]
     for rr in range(NUM_REGIONS)], dtype=float)

# ─── Latency-derived multi-level Ludo cuts ───────────────────────────────────
# The Ludo table's per-level key lengths are not fixed chunk sizes: for a
# (request-region S, holder-region X) pair, level j's key is the content hash
# of the first n₁·(1 + j·α) tokens, where n₁(S,X) = REUSE_FLOOR_TOKENS[S][X]
# — the break-even prefix that just beats 1× the S↔X network cost — and
# α = LUDO_STEP_ALPHA is the step density.  Level 0 therefore *is* the latency
# floor (a shorter cross-region match never surfaces), and a match at cut c
# certifies the reuse covers (c/n₁)× the network latency.  Same-region
# matching uses the block-granular LUDO_LOCAL_CUTS chain (floor 0).  Chains
# cap at CHUNK_SIZES[-1] so Ludo's maximum match depth matches every
# baseline's.
#
# α = 0.5 (half-n₁ steps) is chosen from the leveling design-space analysis
# on the WildChat reuse-length distribution (scripts/wildchat_analysis/
# regen_level_motivation.py, figures/motivation_levels.pdf): it sits on the
# capture-vs-index-cost efficiency frontier (~76 % of achievable net TTFT
# benefit at fewer levels than the fixed chunk grid), whereas full-n₁ steps
# capture nothing for far pairs whose single level lands exactly at
# break-even (net 0).
LUDO_MAX_CUT   = CHUNK_SIZES[-1]                     # 4096 — baseline parity
LUDO_STEP_ALPHA = 0.5                                # step = α·n₁
# Same-region chain: block-granular arithmetic grid [B, 2B, …, LUDO_MAX_CUT].
# Same-region reuse has no latency floor (floor 0), so the chain is limited
# only by index cost — one KV block per level (B = 32 tokens, the paging
# granularity) instead of the coarse doubling CHUNK_SIZES grid, so a local
# match is quantised to 32 tokens rather than losing up to half the prefix
# at the 4096 level.  The physical Node cache keeps the CHUNK_SIZES grid
# (capacity/eviction accounting shared by every baseline); cross-region
# chains keep the latency-derived cuts below.
LUDO_LOCAL_BLOCK = 32
LUDO_LOCAL_CUTS  = list(range(LUDO_LOCAL_BLOCK, LUDO_MAX_CUT + 1,
                              LUDO_LOCAL_BLOCK))

def _pair_chain(rr: int, nr: int):
    """Cut-length chain the region-rr requester probes region-nr holders at."""
    if rr == nr:
        return list(LUDO_LOCAL_CUTS)
    n1 = int(math.ceil(REUSE_FLOOR_TOKENS[rr][nr]))
    if n1 <= 0 or n1 > LUDO_MAX_CUT:
        return []            # even one latency's worth of prefill doesn't fit
    step = max(1, int(round(LUDO_STEP_ALPHA * n1)))
    return list(range(n1, LUDO_MAX_CUT + 1, step))

# ─── Load-gated two-stage routing (Ludo) ─────────────────────────────────────
# Candidacy comes from LOAD, not from the cache: the request's LB first forms
# a pool of non-busy nodes (its own region seen live; remote regions through
# a periodic load broadcast), then annotates the pool with the Ludo table's
# reuse credit, then joint-scores the whole pool.  A node whose recent queue
# exceeds the busy threshold is excluded even if it holds a deep reuse match
# — load balancing outranks cache affinity (the WildChat regime showed the
# reverse ordering loses the knee to selective-pushing LBs).
#
# Dissemination model: every LUDO_LB_BCAST_MS each region broadcasts, for its
# nodes, an exponentially-weighted moving average of the instantaneous queue
# (EWMA ≈ a sliding window of ~2-3 broadcast periods; smooths one-off bursts,
# keeps persistent backlog).  Remote regions act on this snapshot — stale by
# up to one period — while a node's own region is seen live.
LUDO_LB_BCAST_MS       = 100.0   # broadcast period (= SkyWalker's heartbeat)
LUDO_LB_EWMA_BETA      = 0.5     # EWMA weight for the newest sample
# Busy threshold θ: a node whose viewed queue exceeds θ is hard-excluded from
# the pool.  Default ∞ = gate OFF: the θ-sweep (mixed + wildchat) showed the
# pool redefinition (every node is a candidate, cache is only an annotation)
# and the view-based T_q carry the entire win, while any finite θ herds — a
# frozen 100 ms view plus hard exclusion sends every request in the window to
# the same few "non-busy" nodes (mixed@4000: θ=25 → 6.6 s p99 vs ∞ → 1.55 s).
# Busy avoidance happens softly through T_q_view in the score.  θ is kept as
# an ablation knob; with the pool-empty fallback below it is cliff-free at
# any value.
LUDO_LB_BUSY_THRESH_MS = float('inf')
# Capacity pricing λ: the joint score charges prefill compute (1+λ)× — once
# for the requester's own wait, λ× for the fleet capacity it consumes.  λ
# scales with fleet saturation, measured as the viewed queue of the IDLEST
# node in the pool: λ = min(1, min_queue / LUDO_LAMBDA_SCALE_MS).  With spare
# capacity (some node idle) λ≈0 and routing stays balance-first; at
# saturation (even the idlest node queues ≥ the scale) full prefills become
# twice as expensive and the argmin slides toward cache holders — greedy
# per-request TTFT stops externalising its compute cost exactly when compute
# is the scarce resource.  Same principle as migration's self-quiescing:
# spare capacity is the only currency spent on balance.
LUDO_LAMBDA_SCALE_MS = 100.0
LUDO_LAMBDA_MAX      = 8.0   # cap on the capacity-pricing multiplier.
# The cap only binds at deep saturation (below it, λ < 1 regardless), so it
# leaves the operating range bit-identical; raising it monotonically recovers
# the saturated tail (mixed mt @5000: λ_max 1→8 takes p99 from 11.3 s to
# 7.1 s, parity with Dynamo's cache-hoarding) by letting the score prefer a
# cache holder until its queue exceeds ≈ λ_max × the prefill it saves — a
# bounded, self-limiting form of cache affinity (no unbounded herding).

# ─── Popular KV-cache migration (Ludo) ───────────────────────────────────────
# Hot prefixes are replicated toward demand in the background.  Two distinct
# trigger events feed one decision framework:
#   cross-region pull : requests from region S keep matching only REMOTE
#                       holders (paying WAN or recomputing locally);
#   intra-region repl : a LOCAL holder exists but keeps losing the load-gated
#                       score (busy) so the chosen node recomputes.
# Transport economics (per token; KV = 128 KB/token):
#   intra-DC fabric 100 Gbps/NIC (Mooncake FAST'25 production setting)
#                     → 0.0105 ms/token — cheaper than recompute AND GPU-free;
#   cross-region TCP 3 Gbps/flow (GCP per-flow SDN cap; AWS egress ≤5 Gbps —
#                     Skyplane NSDI'23) → 0.35 ms/token — 7× recompute, so a
#                     cross-region "migration" defaults to PROMPT-SHIPPING +
#                     re-prefill on an idle destination GPU; bulk TCP is the
#                     fallback when no GPU is idle (trade network for compute).
# Replicas land in DRAM (cheap tier, PCIe-promoted on first use) and enter the
# normal insert/advertise/evict lifecycle.  Concurrent WAN bulk flows on the
# same region pair serialise (per-flow cap ⇒ congestion by sharing).
MIG_ENABLE            = True
# ── Ablation switches (mirror realmachine ludo_sglang/config.py;
# all default OFF; runtime-settable module globals, env-seeded) ──────────
#   A1 ABL_NO_LOADSCORE : score drops q_view and λ → pure affinity
#                         (cache match + WAN distance); ties break by a
#                         per-request deterministic pseudo-random key.
#   A2 ABL_LOCAL_ONLY   : chain lookup skips remote regions — a remote
#                         holder never surfaces (cross-migration trigger
#                         is structurally dead: best_remote is always 0).
#   A3 (existing)       : MIG_ENABLE=False disables both migration kinds.
#   A5 ABL_TIER_BLIND   : tier ignored in SCORING only — DRAM priced as
#                         VRAM (no PCIe-reload term); serving still pays
#                         the true reload cost.
def _abl_flag(name: str) -> bool:
    return os.environ.get(name, '0').strip() not in ('', '0', 'false', 'no', 'off')

ABL_NO_LOADSCORE = _abl_flag('ABL_NO_LOADSCORE')
ABL_LOCAL_ONLY   = _abl_flag('ABL_LOCAL_ONLY')
ABL_TIER_BLIND   = _abl_flag('ABL_TIER_BLIND')

MIG_INTRA_GBPS        = 100.0
MIG_WAN_GBPS          = 3.0
MIG_INTRA_MS_PER_TOKEN = KV_BYTES_PER_TOKEN / (MIG_INTRA_GBPS / 8 * 1e9) * 1e3
MIG_WAN_MS_PER_TOKEN   = KV_BYTES_PER_TOKEN / (MIG_WAN_GBPS / 8 * 1e9) * 1e3
MIG_HOT_K             = 3        # trigger: ≥K misses of one family per window
MIG_WINDOW_MS         = 1000.0
MIG_COOLDOWN_MS       = 5000.0   # per (region, family) re-trigger cooldown
MIG_MAX_INFLIGHT      = 2        # concurrent inbound migrations per node
MIG_INTRA_RMAX        = 2        # max local holders before intra-repl stops
MIG_REQUIRE_IDLE_DST  = True     # saturation self-quiescing: only migrate to
                                 # persistently idle destinations (ablation knob)

def _build_ludo_cuts(local_cuts):
    """(pair_cuts, region_cuts) for a given same-region chain.  Cross-region
    chains are always the latency-derived ones; only the local chain varies
    (fine block grid for 'ludo', the coarse CHUNK_SIZES grid for the
    'ludo_coarse' ablation baseline)."""
    pair = [[(list(local_cuts) if rr == nr else _pair_chain(rr, nr))
             for nr in range(NUM_REGIONS)]
            for rr in range(NUM_REGIONS)]
    # Holder side: every cut length a region-X node must be indexed at — the
    # union of all requesting regions' chains into X (plus the physical
    # CHUNK_SIZES grid).
    region = [
        sorted(set(CHUNK_SIZES).union(*[pair[S][X] for S in range(NUM_REGIONS)]))
        for X in range(NUM_REGIONS)]
    return pair, region

LUDO_PAIR_CUTS, LUDO_REGION_CUTS = _build_ludo_cuts(LUDO_LOCAL_CUTS)
# Coarse-local ablation ('ludo_coarse'): same cross-region latency cuts, but
# the same-region chain is the 6-level doubling CHUNK_SIZES grid — i.e. the
# pre-block-granularity design, kept as a baseline to quantify what the fine
# local chain buys.
LUDO_PAIR_CUTS_COARSE, LUDO_REGION_CUTS_COARSE = _build_ludo_cuts(CHUNK_SIZES)
# All key lengths the workload generator must emit content hashes for.
# CHUNK_SIZES ⊂ the 32-token block grid, so the fine family's union covers
# the coarse family too; the explicit double union keeps that robust.
LUDO_KEY_LENGTHS = sorted(
    set().union(*[set(c) for c in LUDO_REGION_CUTS])
         .union(*[set(c) for c in LUDO_REGION_CUTS_COARSE]))
# chunk-size → CHUNK_SIZES level index (for eviction/demotion sync).
_CHUNK_LEVEL_OF = {cs: l for l, cs in enumerate(CHUNK_SIZES)}

# Dynamo (hierarchical KV-router) cross-cluster lookup latency (ms), per
# request region.  On a *local* cache miss, the cluster fans a lookup out to
# every other cluster and must wait for the slowest to respond before it can
# pick the globally-best cache holder.  The propagation floor is the round-trip
# to the farthest other region: 2 × max_j≠r WAN[r][j].  DynamoDiscovery.broadcast()
# adds, on top of this floor, the M/D/1 queueing at each queried region's lookup
# coordinator (see below), so the fan-out cost grows with broadcast load.
DYNAMO_XCLUSTER_LOOKUP_MS = np.array([
    2.0 * max(WAN_LATENCY_MS[r][j]
              for j in range(NUM_REGIONS) if j != r)
    for r in range(NUM_REGIONS)], dtype=float)

# Per-region lookup coordinator service rate.  Every Dynamo local miss fans a
# query out to all other regions; each region's coordinator therefore receives
# a query per remote miss, so its inbound load ≈ Σ_{s≠r} miss_rate_s and grows
# with the fleet's miss traffic × the number of clusters.  A cross-cluster
# index probe is a lightweight (non-consensus) hash lookup, so we give each
# coordinator a higher capacity than the linearisable Central Directory
# (2,800 rps): 15,000 rps.  Above this the coordinator's M/D/1 queue — and thus
# the broadcast latency — grows without bound, which is the structural cost of
# broadcast-based discovery that a decentralised plane avoids.
DYNAMO_LOOKUP_SERVICE_RATE_RPS = 15000
DYNAMO_LOOKUP_SERVICE_MS       = 1000.0 / DYNAMO_LOOKUP_SERVICE_RATE_RPS  # ≈0.067 ms

# Chord O(log N) lookup overhead.
# Each of the ceil(log₂ NUM_NODES) = 6 finger-table hops traverses a random
# inter-node WAN path.  We model the per-hop cost as the average off-diagonal
# entry of WAN_LATENCY_MS (cross-region one-way latency ≈ 78 ms) so that the
# total lookup latency is 6 × 78 ≈ 467 ms — consistent with how WAN costs are
# modelled elsewhere in the simulator.
_chord_hops            = int(np.ceil(np.log2(NUM_NODES)))          # 6
_chord_avg_hop_ms      = (WAN_LATENCY_MS.sum()
                          / (NUM_REGIONS * (NUM_REGIONS - 1)))      # ~78 ms
CHORD_LOOKUP_OVERHEAD_MS    = _chord_hops * _chord_avg_hop_ms      # ~467 ms
# Kademlia uses parallel XOR-metric queries; empirically converges in ~60% of
# Chord's sequential hops, giving a smaller but still O(log N) overhead.
KADEMLIA_LOOKUP_OVERHEAD_MS = CHORD_LOOKUP_OVERHEAD_MS * 0.6       # ~280 ms
# Central Directory home region.  A centralised directory service is
# placed in ONE fixed region; every Central Directory request pays a WAN
# round-trip (user → directory → user) to query it before dispatch.  The
# cost is 2 × the one-way delay between the request's own region and this
# home region, so it varies per request — 0 ms for same-region users up
# to ~300 ms for the farthest — instead of a flat worst-case constant.
# The cross-continental round-trips this produces (≈100–300 ms) are
# consistent with measured cross-region coordinator access: AWS DynamoDB
# Global Tables (200–300 ms), Google Spanner consistent reads
# (100–400 ms), Facebook TAO cross-region cache access (100–300 ms).
# Propagation delay only; "infinite bandwidth" does NOT eliminate this.
DIRECTORY_REGION = 0
# Central Directory coordinator capacity.  The directory is a linearisable
# coordination service; a single Paxos/Raft instance sustains ~500–2000
# rps of linearisable reads (each lookup issues a read-index resolved
# within a single-region quorum, paying consensus ordering overhead
# against concurrent advertise / invalidate writes).  Supporting numbers:
#   • Google Spanner cross-region reads with bounded staleness: 500–2000
#     rps per participant leader (OSDI'12, SIGMOD'17)
#   • etcd v3 linearisable (readIndex) reads on 5-node cluster: ~500–
#     2000 rps sustained under concurrent write load
#   • ZooKeeper sync()+read throughput (equivalent to linearisable read):
#     ~500–1000 rps per observer under moderate write load
# A production directory is lightly sharded — the directory keyspace
# partitioned across a small number of coordinator shards, each sustaining
# the mid-range of the measured single-instance rate.  We model two shards
# at ~1400 rps each, giving an effective coordinator capacity of 2800 rps;
# above this the M/D/1 coordinator queue grows unboundedly.  This shared
# coordinator throughput ceiling is the fundamental scalability limit of
# any centralised discovery plane, and is what distributed planes (Ludo,
# DHTs) structurally avoid.
DIRECTORY_SERVICE_RATE_RPS = 2800                                  # req/s
DIRECTORY_SERVICE_MS       = 1000.0 / DIRECTORY_SERVICE_RATE_RPS   # ≈1.33 ms/req

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    cached_len: int    # tokens cached
    tier: str          # 'VRAM' or 'DRAM'
    last_access: float


# Radix-tree chunk geometry — mirrors PlanetServe HashRadixTree.
# Exposed as module-level constants so WorkloadGenerator can precompute the
# per-request chunk hash path once.
RADIX_CHUNK_TOKENS = 10
RADIX_GATE_CHUNKS  = 10
RADIX_HASH_MOD     = 15


@dataclass
class Request:
    request_id: int
    prefix_id:  int
    prefix_len: int    # prompt length (tokens)
    gen_len:    int    # generation length (tokens)
    arrival_time: float
    region_id:  int
    # Multi-granularity content hashes of the real prompt tokens, precomputed
    # once per prefix_id by WorkloadGenerator.  content_keys[cs] = hash of the
    # first `cs` tokens for every cs in CHUNK_SIZES that fits in prefix_len.
    # Discovery planes use these to match requests by content instead of by
    # opaque prefix_id, so e.g. two ToolBench prompts sharing the 320-token
    # system prompt collide at levels 0 (128 tok) and 1 (256 tok) as in a
    # real prefix-caching runtime (vLLM / SGLang).
    content_keys: Dict[int, int] = field(default_factory=dict)
    # RadixTree-specific path: first `RADIX_GATE_CHUNKS` chunk hashes at 10
    # tokens per chunk, each modulo RADIX_HASH_MOD to match PlanetServe's
    # lossy hash.
    radix_path: List[int] = field(default_factory=list)


@dataclass
class RouteResult:
    node_id:    int
    match_len:  int    # tokens saved by cache hit (0 = full miss)
    node_region: int
    tier:       str    # 'VRAM', 'DRAM', or 'MISS'
    lookup_overhead_ms: float = 0.0  # DHT multi-hop WAN lookup cost (Chord only)


@dataclass
class RequestRecord:
    request_id:      int
    prefix_id:       int
    prefix_len:      int
    arrival_time:    float
    ttft_ms:         float
    e2e_latency_ms:  float    # TTFT + gen_len × C_DECODE_STEP_MS_PER_TOKEN
    match_len:       int
    cache_hit:       bool
    flops_saved_frac: float   # fraction of prefill FLOPs avoided
    node_id:         int
    slo_met:         bool
    hit_tier:        str = 'MISS'  # 'VRAM', 'DRAM', or 'MISS'
    gen_len:         int = 0       # generated tokens (for TBT decode window)


# ─── Node ─────────────────────────────────────────────────────────────────────

# Per-node VRAM eviction policy: 'lru' (default) or 'lfu'.  The motivation
# eviction analysis (scripts/evict_analysis/evict_motivation.py) shows the
# best single policy is workload-dependent: LFU wins on the mixed workload
# (stationary Zipf template reuse rewards frequency), LRU wins on
# conversational workloads (session-lifetime reuse rewards recency; LFU
# ossifies on dead sessions' counts).  run_section3 sets this per workload:
# mixed → 'lfu', everything else → 'lru'.  Applies uniformly to every node
# and every scheme (eviction is orthogonal to routing).  The DRAM tier stays
# oldest-first in both modes.
EVICT_POLICY = 'lru'


class Node:
    """Content-addressed node cache.

    Physical cache is organised as six independent LRU dictionaries, one per
    CHUNK_SIZES level.  Each dictionary key is the content hash of the first
    CHUNK_SIZES[l] tokens of some prompt; the value is a CacheEntry whose
    cached_len equals CHUNK_SIZES[l] (at that level the node holds exactly
    that many tokens, or nothing).  A request is served with match_len
    equal to the largest `cs` such that every level `l' ≤ l` has the
    request's content hash present — i.e. the longest contiguous prefix
    match via multi-granularity content hashes, matching vLLM / SGLang's
    block-addressed prefix caching semantics.
    """
    _N_LEVELS = len(CHUNK_SIZES)

    def __init__(self, node_id: int, region_id: int,
                 vram_tokens: int = NODE_VRAM_TOKENS,
                 dram_tokens: int = NODE_DRAM_TOKENS):
        self.node_id     = node_id
        self.region_id   = region_id
        self._vram_limit = vram_tokens
        self._dram_limit = dram_tokens
        # Per-level LRU caches, keyed by content hash.
        self.vram_cache: List[OrderedDict[int, CacheEntry]] = [
            OrderedDict() for _ in range(self._N_LEVELS)
        ]
        self.dram_cache: List[OrderedDict[int, CacheEntry]] = [
            OrderedDict() for _ in range(self._N_LEVELS)
        ]
        # Aggregate token counters: each entry at level l contributes
        # CHUNK_SIZES[l] tokens toward the per-tier budget.
        self.vram_token_count = 0
        self.dram_token_count = 0
        # M/D/c queue: MAX_BATCH_SIZE concurrent slots, each with its own
        # next-available time.  A new request picks the earliest-free slot.
        self._slot_times: List[float] = [0.0] * MAX_BATCH_SIZE
        # (time_ms, prefill_tokens) logged per request; used for heatmap
        self.prefill_log: List[Tuple[float, int]] = []
        # (arrival_ms, queue_wait_ms, service_ms) per scheduled request.
        self.queue_log: List[Tuple[float, float, float]] = []
        # (start_ms, residual_tokens) per non-zero residual prefill; used by
        # the TBT interference model (a prefill starting on this node runs
        # as SGLANG_CHUNK_TOKENS-sized chunks, each stretching one decode
        # iteration of every request decoding here).
        self.prefill_events: List[Tuple[float, float]] = []
        # LFU mode state (unused under 'lru'): access counts for
        # VRAM-resident entries keyed (level, hash), plus a lazy min-heap of
        # (freq, seq, level, hash) — stale tuples (freq mismatch / entry
        # gone) are skipped at pop time.  Invariant: (l,h) in _freq  ⟺
        # h in vram_cache[l].
        self._freq: Dict[Tuple[int, int], int] = {}
        self._fheap: List[Tuple[int, int, int, int]] = []
        self._fseq  = 0

    def reset(self):
        for d in self.vram_cache: d.clear()
        for d in self.dram_cache: d.clear()
        self.vram_token_count = 0
        self.dram_token_count = 0
        self._slot_times = [0.0] * MAX_BATCH_SIZE
        self.prefill_log.clear()
        self.queue_log.clear()
        self.prefill_events.clear()
        self._freq.clear()
        self._fheap.clear()
        self._fseq = 0

    def queue_ms(self, current_time: float) -> float:
        """Wait until the earliest batch slot is free."""
        return max(0.0, min(self._slot_times) - current_time)

    def free_slots(self, current_time: float) -> int:
        """Continuous-batch slots idle right now (admission capacity)."""
        return sum(1 for t in self._slot_times if t <= current_time)

    # ── Internal helpers for content-addressed LRU/LFU ───────────────────────
    def _lfu_touch(self, l: int, h: int):
        """Bump (l,h)'s access count and push a fresh heap tuple."""
        self._fseq += 1
        f = self._freq.get((l, h), 0) + 1
        self._freq[(l, h)] = f
        heapq.heappush(self._fheap, (f, self._fseq, l, h))

    def _pop_oldest_vram(self) -> Tuple[int, int, CacheEntry]:
        """VRAM eviction victim: globally-oldest across levels under 'lru',
        lowest access count (tie: oldest) under 'lfu'.  Returns
        (level, content_hash, entry).  Raises if every level is empty."""
        if EVICT_POLICY == 'lfu':
            while self._fheap:
                f, _s, l, h = heapq.heappop(self._fheap)
                if self._freq.get((l, h)) != f:
                    continue                    # stale tuple (re-touched/gone)
                del self._freq[(l, h)]
                e = self.vram_cache[l].pop(h)
                self.vram_token_count -= e.cached_len
                return l, h, e
            raise RuntimeError('VRAM cache empty; nothing to evict')
        best_level = -1
        best_key   = None
        best_entry: Optional[CacheEntry] = None
        for l, d in enumerate(self.vram_cache):
            if not d:
                continue
            k = next(iter(d))          # oldest in this level's LRU
            e = d[k]
            if best_entry is None or e.last_access < best_entry.last_access:
                best_level, best_key, best_entry = l, k, e
        if best_entry is None:
            raise RuntimeError('VRAM cache empty; nothing to evict')
        del self.vram_cache[best_level][best_key]
        self.vram_token_count -= best_entry.cached_len
        return best_level, best_key, best_entry

    def _pop_oldest_dram(self) -> Tuple[int, int, CacheEntry]:
        best_level = -1
        best_key   = None
        best_entry: Optional[CacheEntry] = None
        for l, d in enumerate(self.dram_cache):
            if not d:
                continue
            k = next(iter(d))
            e = d[k]
            if best_entry is None or e.last_access < best_entry.last_access:
                best_level, best_key, best_entry = l, k, e
        if best_entry is None:
            raise RuntimeError('DRAM cache empty; nothing to evict')
        del self.dram_cache[best_level][best_key]
        self.dram_token_count -= best_entry.cached_len
        return best_level, best_key, best_entry

    def _any_vram(self) -> bool: return any(d for d in self.vram_cache)
    def _any_dram(self) -> bool: return any(d for d in self.dram_cache)

    # ── Public lookup / insert ───────────────────────────────────────────────
    def cache_match_len(self, content_keys: Dict[int, int],
                        prefix_len: int) -> Tuple[int, str]:
        """Longest contiguous content-hash match on this node.

        Walks CHUNK_SIZES from smallest to largest; match_len grows only
        while every lower level's content hash is also resident.  Returns
        (match_len, tier) where tier is 'VRAM' if *all* matching levels
        were VRAM-resident, 'DRAM' if at least one was demoted, 'MISS' if
        no level matched.
        """
        match_len = 0
        tier      = 'MISS'
        for l, cs in enumerate(CHUNK_SIZES):
            if prefix_len < cs:
                break
            h = content_keys.get(cs)
            if h is None:
                break
            if h in self.vram_cache[l]:
                self.vram_cache[l].move_to_end(h)
                if EVICT_POLICY == 'lfu':
                    self._lfu_touch(l, h)
                match_len = cs
                if tier == 'MISS':
                    tier = 'VRAM'
            elif h in self.dram_cache[l]:
                self.dram_cache[l].move_to_end(h)
                match_len = cs
                tier = 'DRAM'   # contiguous match but at least one DRAM hop
            else:
                break
        return match_len, tier

    def insert(self, content_keys: Dict[int, int], prefix_len: int,
               current_time: float
               ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Insert this request's content hashes at every applicable level.

        For each level `l` where prefix_len ≥ CHUNK_SIZES[l], insert
        `content_keys[CHUNK_SIZES[l]]` into `vram_cache[l]` with
        cached_len = CHUNK_SIZES[l].  Enforce the VRAM token budget via
        global-LRU demotion to DRAM (per-level LRU within each tier; the
        oldest across levels wins when we need to evict).  DRAM overflow
        evicts outright.

        Returns
        -------
        fully_evicted  : List[(level, content_hash)] — gone from the node.
        demoted_to_dram: List[(level, content_hash)] — moved VRAM → DRAM.
        """
        fully_evicted:   List[Tuple[int, int]] = []
        demoted_to_dram: List[Tuple[int, int]] = []

        for l, cs in enumerate(CHUNK_SIZES):
            if prefix_len < cs:
                break
            h = content_keys.get(cs)
            if h is None:
                break

            # Refresh: entry already VRAM-resident.  Under 'lfu' just update
            # its timestamp in place — the count survives and no budget work
            # is needed (popping here would break the _freq ⟺ vram_cache
            # invariant while the budget loop runs).  Under 'lru' remove so
            # the re-insert takes the tail of the LRU queue.
            if h in self.vram_cache[l]:
                if EVICT_POLICY == 'lfu':
                    self.vram_cache[l][h].last_access = current_time
                    continue
                self.vram_token_count -= self.vram_cache[l].pop(h).cached_len
            # Promotion: if this hash is sitting in DRAM, pull it out.
            if h in self.dram_cache[l]:
                self.dram_token_count -= self.dram_cache[l].pop(h).cached_len

            # VRAM budget check — demote the policy's victim to DRAM until
            # this new entry fits.
            while self.vram_token_count + cs > self._vram_limit and self._any_vram():
                el, ek, eentry = self._pop_oldest_vram()
                # DRAM budget check — drop the oldest DRAM entry if needed.
                while (self.dram_token_count + eentry.cached_len > self._dram_limit
                       and self._any_dram()):
                    dl, dk, _d = self._pop_oldest_dram()
                    fully_evicted.append((dl, dk))
                self.dram_cache[el][ek] = CacheEntry(
                    eentry.cached_len, 'DRAM', current_time)
                self.dram_token_count += eentry.cached_len
                demoted_to_dram.append((el, ek))

            # Insert into VRAM at this level.
            self.vram_cache[l][h] = CacheEntry(cs, 'VRAM', current_time)
            self.vram_token_count += cs
            if EVICT_POLICY == 'lfu':
                self._lfu_touch(l, h)       # fresh entry (or DRAM promo): count restarts at 1

        return fully_evicted, demoted_to_dram

    def insert_dram(self, content_keys: Dict[int, int], prefix_len: int,
                    current_time: float) -> List[Tuple[int, int]]:
        """Land a migrated KV replica directly in DRAM (cheap tier).

        Levels already resident (either tier) are skipped so contiguity is
        preserved; DRAM overflow evicts oldest-first.  The replica enters the
        normal lifecycle: promoted to VRAM on first use via insert(), evicted
        by LRU when it goes cold.  Returns fully-evicted (level, key) pairs
        for discovery-plane invalidation.
        """
        fully_evicted: List[Tuple[int, int]] = []
        for l, cs in enumerate(CHUNK_SIZES):
            if prefix_len < cs:
                break
            h = content_keys.get(cs)
            if h is None:
                break
            if h in self.vram_cache[l] or h in self.dram_cache[l]:
                continue                     # already resident at this level
            while (self.dram_token_count + cs > self._dram_limit
                   and self._any_dram()):
                dl, dk, _d = self._pop_oldest_dram()
                fully_evicted.append((dl, dk))
            self.dram_cache[l][h] = CacheEntry(cs, 'DRAM', current_time)
            self.dram_token_count += cs
        return fully_evicted

    def insert_beladys(self, content_keys: Dict[int, int], prefix_len: int,
                       current_time: float,
                       pid_times: Dict[int, List[float]]
                       ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Oracle insert with unbounded cache (no eviction).

        Matches the original implementation's posture — oracle receives
        "infinite" cache so it attains the geometric cache-hit upper bound.
        pid_times is accepted for signature compatibility but unused here.
        """
        _ = pid_times
        for l, cs in enumerate(CHUNK_SIZES):
            if prefix_len < cs:
                break
            h = content_keys.get(cs)
            if h is None:
                break
            if h in self.dram_cache[l]:
                self.dram_token_count -= self.dram_cache[l].pop(h).cached_len
            if h in self.vram_cache[l]:
                self.vram_token_count -= self.vram_cache[l].pop(h).cached_len
            self.vram_cache[l][h] = CacheEntry(cs, 'VRAM', current_time)
            self.vram_token_count += cs
        return [], []

    def schedule(self, residual_tokens: int, prefix_len: int,
                 current_time: float) -> float:
        """Reserve one batch slot and return the resulting TTFT (ms).

        M/D/c model (c = MAX_BATCH_SIZE):
          - Pick the earliest-free slot.  If it is already free (< current_time)
            the request starts immediately (zero queue wait).
          - Service time per request is unchanged from the single-server model:
              p = residual_tokens × C_PREFILL + prefix_len × C_DECODE_ATTN
            The batch amortises inter-request overhead (kernel launches, weight
            reads) but each request still pays its own compute cost, so p stays
            the same.  The throughput gain of batching is captured implicitly:
            with c concurrent slots the node absorbs up to c× the load of a
            single-slot server before the queue begins to grow.

        prefill_log records the *full* routed prefix_len (not residual) so that
        the heatmap shows routing concentration rather than compute residual.
        """
        # Choose the slot with the earliest available time
        slot = int(np.argmin(self._slot_times))
        start_time = max(self._slot_times[slot], current_time)
        q = start_time - current_time                          # queue wait
        p = (residual_tokens * C_PREFILL_MS_PER_TOKEN
             + prefix_len    * C_DECODE_MS_PER_TOKEN)          # service time
        self._slot_times[slot] = start_time + p
        # Log at start_time (when the slot actually begins processing), not at
        # arrival time.  This makes the heatmap reflect true GPU utilisation:
        # a queued request does not consume compute until its slot opens up.
        self.prefill_log.append((start_time, prefix_len))
        # (arrival_ms, queue_wait_ms, service_ms) — measurement-only log for
        # the M/G/1 queueing-delay validation (fig_mg1).
        self.queue_log.append((current_time, q, p))
        if residual_tokens > 0:
            self.prefill_events.append((start_time, float(residual_tokens)))
        return q + p


# ─── Workload generator ───────────────────────────────────────────────────────

class WorkloadGenerator:
    """
    Generates requests from three real benchmark workloads:
      ToolBench — short system-prompt sharing (loaded from datasets/)
      LooGLE    — long document-context sharing (loaded from datasets/)
      AAPS      — multi-turn session context (loaded from datasets/)

    Prefix families are sampled with Zipf(α=1.1) frequency.
    """
    # Path to datasets relative to this file: ../../datasets/
    _DATASETS_DIR = Path(__file__).parent / 'datasets'

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._build_pools()

    @staticmethod
    def _load_jsonl(path: Path):
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    @staticmethod
    def _entry_tokens(entry: dict, dataset: str) -> List[int]:
        """Real (or real-enough) token sequence for an entry.

        ToolBench rows ship the Llama-tokenised input_ids; LooGLE and AAPS
        ship raw `workload` text.  For the latter we use 4-character-per-
        token char-ord windows (same approximation Fig 3 uses in
        run_section3.py) so that equal content yields equal token sequences,
        which is all the discovery plane's content hash needs.
        """
        if dataset == 'toolbench':
            return list(entry['input_ids'])
        text = entry['workload']
        return [hash(text[i:i + 4]) & 0xFFFFFFFF
                for i in range(0, len(text), 4)]

    @staticmethod
    def _precompute_content(entries: List[dict], dataset: str,
                             base_id: int,
                             length_floor: int = 0
                             ) -> Tuple[Dict[int, int],
                                        Dict[int, Dict[int, int]],
                                        Dict[int, List[int]]]:
        """Per-entry length + per-chunk-size content hashes + radix path.

        Returns (lens, content_keys, radix_paths) indexed by prefix_id.
        """
        lens:          Dict[int, int]             = {}
        content_keys:  Dict[int, Dict[int, int]]  = {}
        radix_paths:   Dict[int, List[int]]       = {}
        gate_tokens    = RADIX_GATE_CHUNKS * RADIX_CHUNK_TOKENS   # 100

        for i, e in enumerate(entries):
            pid    = base_id + i
            toks   = WorkloadGenerator._entry_tokens(e, dataset)
            n      = len(toks)
            lens[pid] = max(length_floor, n)

            # Content hash at every Ludo key length the prompt covers —
            # the latency-derived cut lengths plus the CHUNK_SIZES levels
            # (LUDO_KEY_LENGTHS is their union).  A single 64-bit polynomial
            # rolling pass emits the hash at each cut, so equal content
            # yields equal keys at every length and the cost stays
            # O(longest covered cut) no matter how many cuts exist.
            keys_for_pid: Dict[int, int] = {}
            need = [c for c in LUDO_KEY_LENGTHS if c <= n]
            if need:
                h  = 0
                ni = 0
                for idx in range(need[-1]):
                    h = (h * 1000003 ^ int(toks[idx])) & 0xFFFFFFFFFFFFFFFF
                    if idx + 1 == need[ni]:
                        keys_for_pid[need[ni]] = h & 0xFFFFFFFF
                        ni += 1
                        if ni == len(need):
                            break
            content_keys[pid] = keys_for_pid

            # RadixTree walk path: up to GATE chunks of CHUNK_TOKENS tokens.
            # Matches PlanetServe's `hash_chunk` mod 15 (lossy) exactly.
            path: List[int] = []
            for k in range(min(RADIX_GATE_CHUNKS, n // RADIX_CHUNK_TOKENS)):
                chunk = toks[k * RADIX_CHUNK_TOKENS:(k + 1) * RADIX_CHUNK_TOKENS]
                h = 0
                for tid in chunk:
                    h = (h * 31 + int(tid)) % RADIX_HASH_MOD
                path.append(h)
            radix_paths[pid] = path

        return lens, content_keys, radix_paths

    def _build_pools(self):
        rng = self.rng

        def zipf_w(n, alpha=1.1):
            w = np.arange(1, n + 1, dtype=float) ** (-alpha)
            return w / w.sum()

        # ── ToolBench: real prompts, tokenized ───────────────────────────────
        tb_path = self._DATASETS_DIR / 'toolbench_zipf_1.1_prompts_6000.jsonl'
        tb_entries = self._load_jsonl(tb_path)
        tb_ids     = list(range(len(tb_entries)))
        tb_lens, tb_keys, tb_paths = self._precompute_content(
            tb_entries, 'toolbench', base_id=0)
        # gen_len is fixed at 32 (from sampling_params.max_new_tokens)
        tb_glens = {i: e['sampling_params']['max_new_tokens']
                    for i, e in enumerate(tb_entries)}

        # ── LooGLE: long-context documents, char-level length ÷ 4 ────────────
        lg_path = self._DATASETS_DIR / 'loogle.jsonl'
        lg_entries = self._load_jsonl(lg_path)
        lg_base    = len(tb_ids)
        lg_ids     = list(range(lg_base, lg_base + len(lg_entries)))
        lg_lens, lg_keys, lg_paths = self._precompute_content(
            lg_entries, 'loogle', base_id=lg_base, length_floor=256)

        # ── AAPS: multi-turn sessions, char-level length ÷ 4 ─────────────────
        aaps_path    = self._DATASETS_DIR / 'aaps.jsonl'
        aaps_entries = self._load_jsonl(aaps_path)
        aaps_base    = lg_base + len(lg_ids)
        aaps_ids     = list(range(aaps_base, aaps_base + len(aaps_entries)))
        aaps_lens, aaps_keys, aaps_paths = self._precompute_content(
            aaps_entries, 'aaps', base_id=aaps_base, length_floor=64)

        # ── WildChat-1M: conversational, per-turn cumulative prefixes ─────────
        # Optional — only if datasets/wildchat.jsonl has been generated by
        # datasets/prep_wildchat.py.  Each entry carries a real per-request
        # region_id (from the conversation's country) and gen_len (assistant
        # reply length), which we index by prefix_id below.
        self._wc_region: Dict[int, int] = {}
        self._wc_glens:  Dict[int, int] = {}
        wc_keys: Dict[int, Dict[int, int]] = {}
        wc_paths: Dict[int, List[int]] = {}
        wc_ids: List[int] = []
        wc_path = self._DATASETS_DIR / 'wildchat.jsonl'
        if wc_path.exists():
            wc_entries = self._load_jsonl(wc_path)
            wc_base    = aaps_base + len(aaps_ids)
            wc_ids     = list(range(wc_base, wc_base + len(wc_entries)))
            wc_lens, wc_keys, wc_paths = self._precompute_content(
                wc_entries, 'wildchat', base_id=wc_base)
            for i, e in enumerate(wc_entries):
                pid = wc_base + i
                self._wc_region[pid] = int(e.get('region_id', 0))
                self._wc_glens[pid]  = int(e.get('gen_len', 64))

        # ── LMSYS-Chat-1M: conversational, per-turn cumulative prefixes ──────
        # Optional — datasets/lmsys.jsonl from datasets/prep_lmsys.py.  Real
        # per-turn gen_len (stored in the same per-pid table as WildChat's);
        # LMSYS carries no country metadata, so _region_for falls through to
        # demand-weighted region sampling.
        lm_keys: Dict[int, Dict[int, int]] = {}
        lm_paths: Dict[int, List[int]] = {}
        lm_ids: List[int] = []
        lm_path = self._DATASETS_DIR / 'lmsys.jsonl'
        if lm_path.exists():
            lm_entries = self._load_jsonl(lm_path)
            lm_base    = (wc_ids[-1] + 1 if wc_ids
                          else aaps_base + len(aaps_ids))
            lm_ids     = list(range(lm_base, lm_base + len(lm_entries)))
            lm_lens, lm_keys, lm_paths = self._precompute_content(
                lm_entries, 'lmsys', base_id=lm_base)
            for i, e in enumerate(lm_entries):
                self._wc_glens[lm_base + i] = int(e.get('gen_len', 64))

        # gen_len lookup: toolbench uses dataset value; others use random draw
        self._tb_glens = tb_glens   # {prefix_id: gen_len}

        # Merged content-hash tables — discovery and Request construction look
        # these up by prefix_id regardless of source dataset.
        self._content_keys: Dict[int, Dict[int, int]] = {
            **tb_keys, **lg_keys, **aaps_keys, **wc_keys, **lm_keys,
        }
        self._radix_paths:  Dict[int, List[int]] = {
            **tb_paths, **lg_paths, **aaps_paths, **wc_paths, **lm_paths,
        }

        self.pools = {
            'toolbench': (np.array(tb_ids),   tb_lens,   zipf_w(len(tb_ids))),
            'loogle':    (np.array(lg_ids),    lg_lens,   zipf_w(len(lg_ids))),
            'aaps':      (np.array(aaps_ids),  aaps_lens, zipf_w(len(aaps_ids))),
        }
        if wc_ids:
            self.pools['wildchat'] = (
                np.array(wc_ids), wc_lens, zipf_w(len(wc_ids)))
        if lm_ids:
            self.pools['lmsys'] = (
                np.array(lm_ids), lm_lens, zipf_w(len(lm_ids)))

        # Precomputed inverse-CDF tables for Zipf draws.  rng.choice(ids,
        # p=weights) is O(pool) per draw — ruinous at the 375 K-entry
        # WildChat pool (~1 ms/draw × millions of draws).  searchsorted on
        # the cumulative weights is O(log pool) and statistically identical.
        self._pool_cum = {name: np.cumsum(w)
                          for name, (_ids, _lens, w) in self.pools.items()}

    def _draw_pid(self, workload: str, rng) -> int:
        """O(log n) Zipf draw from the workload pool (inverse CDF)."""
        ids = self.pools[workload][0]
        cum = self._pool_cum[workload]
        i = int(np.searchsorted(cum, rng.random(), side='right'))
        return int(ids[min(i, len(ids) - 1)])

    def reset_stream(self, seed: int):
        """Re-seed the request stream so back-to-back runs in one process see
        bit-identical workloads (paired comparisons across schemes/configs).

        sample() consumes this shared RNG, so without a reset each run's
        request sequence depends on how much the previous runs consumed —
        statistically fair (same distribution) but unpaired, leaving ±5-10 %
        sampling noise in p99 deltas."""
        self.rng = np.random.default_rng(seed)

    def _region_for(self, pid: int, rng) -> int:
        """Real per-request region for WildChat prefixes; else demand-weighted."""
        r = self._wc_region.get(pid)
        if r is not None:
            return r
        return int(rng.choice(NUM_REGIONS, p=REGION_REQUEST_WEIGHTS))

    def _genlen_for(self, pid: int, rng) -> int:
        """Real gen_len for ToolBench / WildChat prefixes; else random draw."""
        if pid in self._tb_glens:
            return self._tb_glens[pid]
        if pid in self._wc_glens:
            return self._wc_glens[pid]
        return int(rng.integers(50, 300))

    def make_one_pass(self, workload: str,
                      rate_ms: float, seed: int = 42) -> List[Request]:
        """Return one Request per dataset entry in sequential (dataset) order.

        Unlike sample(), this method does NOT use Zipf weights — every entry
        in the dataset is sent exactly once.  Arrival times are drawn from an
        exponential inter-arrival process at *rate_ms* (req/ms) so that node
        queue dynamics remain realistic.

        NOTE: Because every prefix_id is unique in one-pass order, this method
        produces 0% cache hits by construction — use make_zipf_pass() when
        cache hit rates need to be measured.
        """
        rng = np.random.default_rng(seed)
        ids, lens, _ = self.pools[workload]   # ignore Zipf weights
        reqs: List[Request] = []
        t = 0.0
        for i, pid in enumerate(ids.tolist()):
            t += rng.exponential(1.0 / rate_ms)
            region_id = self._region_for(pid, rng)
            plen      = lens[pid]
            gen_len   = self._genlen_for(pid, rng)
            reqs.append(Request(
                i, int(pid), int(plen), gen_len, t, region_id,
                content_keys = self._content_keys.get(int(pid), {}),
                radix_path   = self._radix_paths.get(int(pid), []),
            ))
        return reqs

    def make_zipf_pass(self, workload: str,
                       rate_ms: float, seed: int = 42) -> List[Request]:
        """Return len(pool) Zipf-weighted requests — one round through the dataset.

        Generates the same number of requests as the dataset size, but samples
        prefix IDs with Zipf popularity weights so that hot prefixes recur and
        can produce cache hits.  This is the right method for measuring cache
        hit rates and FLOPs saved under realistic access patterns, while
        keeping the total request count comparable to the dataset size (not
        the 100 K+ used in throughput sweeps).
        """
        rng = np.random.default_rng(seed)
        ids, lens, weights = self.pools[workload]
        n = len(ids)
        reqs: List[Request] = []
        t = 0.0
        for i in range(n):
            t += rng.exponential(1.0 / rate_ms)
            pid       = self._draw_pid(workload, rng)
            region_id = self._region_for(pid, rng)
            plen      = lens[pid]
            gen_len   = self._genlen_for(pid, rng)
            reqs.append(Request(
                i, int(pid), int(plen), gen_len, t, region_id,
                content_keys = self._content_keys.get(int(pid), {}),
                radix_path   = self._radix_paths.get(int(pid), []),
            ))
        return reqs

    def sample(self, request_id: int, current_time: float,
               workload: str = 'mixed') -> Request:
        rng = self.rng
        if workload == 'mixed':
            # 50 % ToolBench / 5 % LooGLE / 45 % AAPS.  LooGLE's real
            # ~27K-token prompts take ~1.35 s of prefill on an A100
            # (Llama-3.1-8B BF16), which alone consumes ~80 % of the SLO
            # budget — at the earlier 20 % mix it single-handedly capped
            # cluster throughput below the first arrival-rate sweep
            # point.  5 % keeps LooGLE's long-document behaviour visible
            # in tail metrics without dominating steady-state capacity.
            r = rng.random()
            wtype = ('toolbench' if r < 0.50
                     else ('loogle' if r < 0.55 else 'aaps'))
        else:
            wtype = workload
        lens      = self.pools[wtype][1]
        pid       = self._draw_pid(wtype, rng)
        region_id = self._region_for(pid, rng)
        plen      = lens[pid]
        gen_len   = self._genlen_for(pid, rng)
        return Request(
            request_id, pid, int(plen), gen_len, current_time, region_id,
            content_keys = self._content_keys.get(int(pid), {}),
            radix_path   = self._radix_paths.get(int(pid), []),
        )


# ─── Discovery planes ─────────────────────────────────────────────────────────

class DiscoveryPlane:
    """Discovery-plane base interface.

    The simulator always passes a `content_keys: Dict[chunk_size, hash]`
    dict when calling `advertise` / `lookup` / `invalidate_content` — these
    are the multi-granularity content hashes precomputed by
    WorkloadGenerator.  For each level `l`, discovery classes use
    `content_keys[CHUNK_SIZES[l]]` as the per-level key so that two
    requests sharing tokens at level `l` collide at the discovery plane.

    `prefix_id` is kept as a fallback key for external callers (e.g.,
    synthetic keys with no token content and
    index by opaque request IDs.
    """
    name = 'base'

    def lookup(self, prefix_id: int, prefix_len: int,
               request_region: int = 0,
               content_keys: Optional[Dict[int, int]] = None
               ) -> List[Tuple[int, int, str]]:
        """Returns list of (node_id, match_len, tier)."""
        raise NotImplementedError

    def advertise(self, node_id: int, region_id: int,
                  prefix_id: int, cached_len: int, tier: str = 'VRAM',
                  content_keys: Optional[Dict[int, int]] = None):
        pass

    def invalidate(self, node_id: int, prefix_id: int):
        """Legacy prefix-id-based bulk invalidation (used by external
        callers that don't track per-level content hashes)."""
        pass

    def invalidate_content(self, node_id: int, level: int, content_key: int):
        """Content-addressed per-entry invalidation (used by the
        simulator after a Node's per-level LRU evicts a specific entry)."""
        pass

    def reset(self):
        pass


class LudoDiscovery(DiscoveryPlane):
    """
    Multi-level Ludo table with latency-derived level lengths.

    Two storage backends (LUDO_TABLE_BACKEND, default native when built):
      dict   — table[cut][key] = {node_id: (cached_len, tier, freshness,
               region)}; the historical plain-memory reference.
      native — one realmachine LudoTable (Smash ControlPlaneLudo,
               uint64 value = 32 × 2-bit worker-tier slots) per holder
               region; key = (cut << 32) | content_hash.  Every router
               holds an instantly-synced mirror of all region tables, so
               the information content is identical to the dict backend —
               only the physical data structure changes.

    Levels are not fixed chunk sizes: a requester in region S probes holders
    in region X along the chain LUDO_PAIR_CUTS[S][X] = [n₁, 1.5·n₁, 2·n₁, …]
    (α = 0.5 steps), where n₁(S,X) is the break-even prefix length that just
    beats 1× the S↔X network cost.  The first level therefore embeds the
    latency floor structurally — a cross-region match shorter than n₁ never
    surfaces — and a match at cut c certifies the reuse covers (c/n₁)× the
    network latency.  Same-region matching uses the block-granular
    LUDO_LOCAL_CUTS chain (floor 0, one 32-token KV block per level).  A
    holder node is indexed at every cut length any requesting region might
    probe its region with (LUDO_REGION_CUTS).

    Eviction sync: the physical Node cache evicts at CHUNK_SIZES
    granularity, so the table keeps a per-node registry mapping each
    advertised prefix's chunk-level hashes to its cut entries.  Evicting
    chunk level l removes every cut entry longer than CHUNK_SIZES[l-1]
    (those cuts span the evicted chunk); demoting it to DRAM degrades the
    same entries' tier instead.
    """
    name = 'ludo'

    # Native-backend sizing: the sim drives all eviction itself through the
    # registry sync, so the table-side LRU must never fire — cap far above
    # any live entry count.  expected_entries is only the initial capacity;
    # LudoTable rebuilds at 2× when the CP fills.
    _NATIVE_EXPECTED    = 1 << 16
    _NATIVE_MAX_ENTRIES = 1 << 22

    def __init__(self, pair_cuts=None, region_cuts=None):
        # Chain family: default fine block-granular local chain ('ludo');
        # the 'ludo_coarse' baseline passes the CHUNK_SIZES-local variant.
        self.pair_cuts   = pair_cuts   if pair_cuts   is not None else LUDO_PAIR_CUTS
        self.region_cuts = region_cuts if region_cuts is not None else LUDO_REGION_CUTS
        self.use_native = (LUDO_TABLE_BACKEND == 'native')
        self._make_tables()
        self._counter = 0
        # Per-node registry for chunk-granular eviction sync:
        #   _prefixes[node][uid]           = {cut: key} currently indexed
        #   _chunk_index[node][(lvl, key)] = {uid, ...}
        self._prefixes: Dict[int, Dict[int, Dict[int, int]]] = defaultdict(dict)
        self._chunk_index: Dict[int, Dict[Tuple[int, int], set]] = defaultdict(dict)
        # Load view, two channels (piggyback + suppressed broadcast):
        #   data path — every cross-region response carries the serving
        #   region's full per-node queue vector stamped by its LB (a few
        #   hundred bytes riding on traffic that flows anyway, à la C3 /
        #   ORCA load reports), so busy pairs refresh at their own traffic
        #   rate instead of the broadcast period;
        #   control path — the periodic broadcast only fires for a
        #   (viewer, target) pair with no piggyback within the last period
        #   (sender-side suppression: a region knows whom it just served).
        # Views are per viewer region: _lb_view[S][nid] = EWMA queue.
        self._lb_view: Dict[int, Dict[int, float]] = defaultdict(dict)
        self._lb_fresh: Dict[Tuple[int, int], float] = {}   # (S,X)->last refresh
        self._lb_last: Optional[float] = None                # broadcast tick
        self._lb_pending: Dict[Tuple[int, int], float] = defaultdict(float)
        self._lb_pb: List[tuple] = []                        # (t_back, S, X)
        self.lb_stats = {'bcast_sent': 0, 'bcast_suppressed': 0,
                         'piggybacks': 0}
        # Popular-KV migration state.
        self._mig_cnt: Dict[tuple, Tuple[int, float]] = {}   # (S,fam,kind)->(n,win_start)
        self._mig_cool: Dict[tuple, float] = {}              # (S,fam)->last trigger
        self._mig_pending: List[dict] = []                   # in-flight migrations
        self._mig_inflight: Dict[int, int] = defaultdict(int)
        self._mig_link_free: Dict[tuple, float] = {}         # (src,dst)->free time
        self.mig_stats = {'cross_recompute': 0, 'cross_bulk': 0,
                          'intra_bulk': 0, 'completed': 0}

    def _make_tables(self):
        """dict backend: one nested dict, entries carry their region.
        native backend: one real-machine LudoTable per holder region, its
        worker set = that region's nodes (global node id as the worker
        address).  Both are read as "region X's index", looked up per
        holder-region chain."""
        if self.use_native:
            self.rtables = [
                _RealLudoTable(
                    worker_addrs=[str(REGION_OFFSETS[X] + i)
                                  for i in range(NODES_PER_REGION_LIST[X])],
                    expected_entries=self._NATIVE_EXPECTED,
                    max_entries=self._NATIVE_MAX_ENTRIES,
                )
                for X in range(NUM_REGIONS)
            ]
            self.table = None
        else:
            self.rtables = None
            # cut_len -> key -> node_id -> (cached_len, tier, freshness, region)
            self.table = defaultdict(lambda: defaultdict(dict))

    @staticmethod
    def _lkey(cut: int, key: int) -> int:
        """Composite uint64 Ludo key: cut length in the high bits keeps the
        per-cut key spaces disjoint.  (The workload's content hashes are
        32-bit; the real testbed gets the same disjointness for free since
        chunk_to_key hashes the cut-length token tuple itself.)"""
        return (cut << 32) | (key & 0xFFFFFFFF)

    def reset(self):
        self._make_tables()
        self._counter = 0
        self._prefixes = defaultdict(dict)
        self._chunk_index = defaultdict(dict)
        self._lb_view = defaultdict(dict)
        self._lb_fresh = {}
        self._lb_last = None
        self._lb_pending = defaultdict(float)
        self._lb_pb = []
        self.lb_stats = {'bcast_sent': 0, 'bcast_suppressed': 0,
                         'piggybacks': 0}
        self._mig_cnt = {}
        self._mig_cool = {}
        self._mig_pending = []
        self._mig_inflight = defaultdict(int)
        self._mig_link_free = {}
        self.mig_stats = {'cross_recompute': 0, 'cross_bulk': 0,
                          'intra_bulk': 0, 'completed': 0}

    def mig_tick(self, nodes, now):
        """Complete due migrations: land the replica in the destination's
        DRAM and advertise it so the table sees the new copy."""
        if not self._mig_pending:
            return
        still = []
        for m in self._mig_pending:
            if m['t_done'] > now:
                still.append(m)
                continue
            dst = nodes[m['dst']]
            evicted = dst.insert_dram(m['ck'], m['ml'], now)
            for lvl, key in evicted:
                self.invalidate_content(m['dst'], lvl, key)
            self.advertise(m['dst'], dst.region_id, m['pid'], m['ml'],
                           tier='DRAM', content_keys=m['ck'])
            self._mig_inflight[m['dst']] -= 1
            self.mig_stats['completed'] += 1
        self._mig_pending = still

    @staticmethod
    def _chunk_floor(ml):
        """Largest CHUNK_SIZES level ≤ ml (physical chunk-grid granularity —
        WAN bulk copies only what the source verifiably holds)."""
        best = 0
        for cs in CHUNK_SIZES:
            if cs <= ml:
                best = cs
        return best

    def mig_note(self, req_region, prefix_id, prefix_len, ck, cred, chosen_ml,
                 nodes, now):
        """Count locality-miss events; fire a migration when a prefix family
        is hot.  Two event kinds with separate counters (they dilute each
        other if merged):
          'cross' — no local holder matched, a remote one (≥ floor) did;
          'intra' — a local holder matched but lost the load-gated score.
        """
        fam = ck.get(CHUNK_SIZES[0])
        if fam is None:
            return
        best_local = best_remote = 0
        src_remote = -1
        local_holders = 0
        for nid, (ml, _t) in cred.items():
            if ml <= 0:
                continue
            if nodes[nid].region_id == req_region:
                local_holders += 1
                if ml > best_local:
                    best_local = ml
            elif ml > best_remote:
                best_remote = ml
                src_remote = nid
        if best_local == 0 and best_remote > 0:
            kind, ml_target = 'cross', best_remote
        elif best_local > 0 and chosen_ml < best_local:
            if local_holders >= MIG_INTRA_RMAX:
                return                      # enough local replicas already
            kind, ml_target = 'intra', best_local
        else:
            return

        key = (req_region, fam, kind)
        n, start = self._mig_cnt.get(key, (0, now))
        if now - start > MIG_WINDOW_MS:
            n, start = 0, now
        n += 1
        self._mig_cnt[key] = (n, start)
        if n < MIG_HOT_K:
            return
        if now - self._mig_cool.get((req_region, fam), -1e18) < MIG_COOLDOWN_MS:
            return

        # Destination: a PERSISTENTLY IDLE (EWMA queue == 0 through the last
        # broadcast window) non-holder in the demand region with migration
        # capacity.  This doubles as the saturation self-quiescing gate:
        # migrations only spend spare capacity — a replica bought under
        # saturation displaces content/compute that is already earning
        # (zero-sum), so when no node in the region is idle the mechanism
        # simply does not fire and re-arms once load recedes (the hotness
        # counter is kept, not reset).
        holder_ids = {nid for nid, (ml, _t) in cred.items() if ml > 0}
        cands = [nd for nd in nodes
                 if nd.region_id == req_region
                 and nd.node_id not in holder_ids
                 and (not MIG_REQUIRE_IDLE_DST
                      or self._lb_view[req_region].get(nd.node_id, 0.0) <= 0.0)
                 and self._mig_inflight[nd.node_id] < MIG_MAX_INFLIGHT]
        if not cands:
            return                          # region saturated → quiesce
        dst = min(cands, key=lambda nd: nd.queue_ms(now))

        if kind == 'intra':
            # Bulk copy over the DC fabric — GPU-free, cheaper than recompute.
            # Source is a local holder; its match is already chunk-aligned.
            delay = 1.0 + ml_target * MIG_INTRA_MS_PER_TOKEN
            self.mig_stats['intra_bulk'] += 1
        elif (dst.queue_ms(now) <= 0.0
              and self._lb_view[req_region].get(dst.node_id, 0.0) <= 0.0):
            # Prompt-shipping: re-prefill on a PERSISTENTLY idle destination
            # GPU (instantaneously free AND idle through the last broadcast
            # window — at sustained saturation a momentarily free slot is not
            # idle, and background prefill would displace real requests).
            # The replica is computed from the requester's own prompt, so it
            # is NOT limited by the source's (chain-quantised) match — build
            # the full useful prefix so the local chain credits it fully.
            ml_target = min(prefix_len, LUDO_MAX_CUT)
            delay = dst.schedule(ml_target, ml_target, now)
            self.mig_stats['cross_recompute'] += 1
        else:
            # No idle GPU → TCP bulk over the WAN; flows on the same region
            # pair serialise at the per-flow cap (congestion by sharing).
            # Bulk is bounded by what the source verifiably holds; round DOWN
            # to the chunk grid — same local credit, less bandwidth.
            ml_target = max(CHUNK_SIZES[0], self._chunk_floor(ml_target))
            src_region = nodes[src_remote].region_id
            link = (src_region, req_region)
            start_tx = max(now, self._mig_link_free.get(link, 0.0))
            dur = ml_target * MIG_WAN_MS_PER_TOKEN
            self._mig_link_free[link] = start_tx + dur
            delay = (start_tx - now) + WAN_LATENCY_MS[src_region][req_region] + dur
            self.mig_stats['cross_bulk'] += 1

        self._mig_pending.append({'t_done': now + delay, 'dst': dst.node_id,
                                  'ck': ck, 'ml': ml_target, 'pid': prefix_id})
        self._mig_inflight[dst.node_id] += 1
        self._mig_cool[(req_region, fam)] = now
        self._mig_cnt[key] = (0, now)

    def _lb_fold(self, S, region_nodes, current_time):
        """Fold the target region's live queues into viewer S's view (EWMA)
        and clear S's dispatch-pending for those nodes (the fresh sample
        already reflects everything S sent that has arrived)."""
        b = LUDO_LB_EWMA_BETA
        view = self._lb_view[S]
        for n in region_nodes:
            q = n.queue_ms(current_time)
            prev = view.get(n.node_id)
            view[n.node_id] = q if prev is None else b * q + (1.0 - b) * prev
            self._lb_pending.pop((S, n.node_id), None)

    def lb_probe(self, nodes, current_time):
        """Two-channel load dissemination.

        Data path (free): deliver due piggybacks — a cross-region response
        arrives back at the requesting region carrying the serving region's
        full per-node queue vector, so a busy (S,X) pair refreshes at its
        own traffic rate.
        Control path (suppressed broadcast): once per LUDO_LB_BCAST_MS,
        refresh only the (S,X) pairs with no piggyback within the period —
        the sender knows whom it just served, so suppression needs no
        coordination.  Bandwidth scales with idle pairs, not O(R²)·rate.
        """
        if self._lb_pb:
            still = []
            for t_back, S, X in self._lb_pb:
                if t_back > current_time:
                    still.append((t_back, S, X))
                    continue
                self._lb_fold(S, [n for n in nodes if n.region_id == X],
                              current_time)
                self._lb_fresh[(S, X)] = t_back
                self.lb_stats['piggybacks'] += 1
            self._lb_pb = still

        if (self._lb_last is None
                or current_time - self._lb_last >= LUDO_LB_BCAST_MS):
            by_region: Dict[int, list] = defaultdict(list)
            for n in nodes:
                by_region[n.region_id].append(n)
            for S in range(NUM_REGIONS):
                for X in range(NUM_REGIONS):
                    if X == S:
                        continue
                    if (current_time - self._lb_fresh.get((S, X), -1e18)
                            < LUDO_LB_BCAST_MS):
                        self.lb_stats['bcast_suppressed'] += 1
                        continue
                    self._lb_fold(S, by_region[X], current_time)
                    self._lb_fresh[(S, X)] = current_time
                    self.lb_stats['bcast_sent'] += 1
            self._lb_last = current_time

    def lb_view_queue(self, node, req_region, current_time):
        """Viewed queue: live for the requester's own region; for remote
        regions viewer S's (piggyback- or broadcast-fed) EWMA *plus* the
        occupancy S itself dispatched there since that view was refreshed
        (an LB is omniscient about its own sends — prevents window
        herding)."""
        if node.region_id == req_region:
            return node.queue_ms(current_time)
        return (self._lb_view[req_region].get(node.node_id, 0.0)
                + self._lb_pending.get((req_region, node.node_id), 0.0))

    def lb_note_dispatch(self, req_region, node_id, service_ms):
        """Account a dispatched request's expected occupancy in the
        dispatching region's view."""
        self._lb_pending[(req_region, node_id)] += service_ms / MAX_BATCH_SIZE

    def lb_note_forward(self, req_region, target_region, t_back):
        """A cross-region forward: its response will carry the target
        region's load stamp, arriving back at ~t_back."""
        self._lb_pb.append((t_back, req_region, target_region))

    def _key(self, prefix_id, content_keys, cut):
        if content_keys is not None:
            return content_keys.get(cut)     # None → prompt doesn't cover cut
        return prefix_id

    def lookup(self, prefix_id, prefix_len, request_region=0,
               content_keys=None):
        """Per-holder-region chain walk with contiguous-level guard.

        For each holder region X, walk LUDO_PAIR_CUTS[request_region][X]
        ascending; a node's match extends to chain index i only if it was
        present at every index < i (two prefixes colliding at cut c share
        exactly the first c tokens, no more).  The reported match_len is the
        deepest contiguous cut — ≥ n₁(S,X) by construction, so the latency
        floor needs no separate filter.
        """
        out: List[Tuple[int, int, str]] = []
        for X in range(NUM_REGIONS):
            if ABL_LOCAL_ONLY and X != request_region:
                continue    # A2: remote holders never surface
            chain = self.pair_cuts[request_region][X]
            # node_id -> (deepest_chain_idx, tier)
            contig: Dict[int, Tuple[int, str]] = {}
            for i, cut in enumerate(chain):
                if prefix_len < cut:
                    break
                key = self._key(prefix_id, content_keys, cut)
                if key is None:
                    break
                if self.use_native:
                    # Region X's table only holds region-X workers; no
                    # region filter needed.
                    holders = [
                        (int(addr), 'VRAM' if t == _RL_VRAM else 'DRAM')
                        for addr, t in
                        self.rtables[X].lookup_key(self._lkey(cut, key))
                    ]
                else:
                    row = self.table.get(cut)
                    row = row.get(key) if row else None
                    holders = [
                        (nid, tier)
                        for nid, (_cl, tier, _f, reg) in row.items()
                        if reg == X    # cut length shared across chains
                    ] if row else []
                if not holders:
                    # Nothing advertised at this cut for this key — nodes
                    # matched at shallower cuts keep their deepest-so-far.
                    continue
                for node_id, tier in holders:
                    if i == 0:
                        contig[node_id] = (0, tier)
                    elif node_id in contig and contig[node_id][0] == i - 1:
                        # Degrade tier to DRAM if any level along the contig
                        # chain was DRAM.
                        prev_tier = contig[node_id][1]
                        new_tier  = ('DRAM' if (prev_tier == 'DRAM'
                                                or tier == 'DRAM') else 'VRAM')
                        contig[node_id] = (i, new_tier)
                    # else: non-contiguous hit — ignored.
            for nid, (idx, tier) in contig.items():
                out.append((nid, min(chain[idx], prefix_len), tier))
        return out

    def advertise(self, node_id, region_id, prefix_id, cached_len, tier='VRAM',
                  content_keys=None):
        # Demotion tier-update: the simulator re-advertises a demoted
        # (chunk-level, key) pair as a single-entry content_keys dict with
        # tier='DRAM'.  Degrade every cut entry spanning that chunk instead
        # of inserting anything.
        if (tier == 'DRAM' and content_keys is not None
                and len(content_keys) == 1):
            (cs, key), = content_keys.items()
            lvl = _CHUNK_LEVEL_OF.get(cs)
            if lvl is not None:
                self._demote_chunk(node_id, lvl, key)
                return

        self._counter += 1
        reg_prefixes = self._prefixes[node_id]
        for cut in self.region_cuts[region_id]:
            if cached_len < cut:
                break                        # cuts sorted ascending
            key = self._key(prefix_id, content_keys, cut)
            if key is None:
                continue
            if self.use_native:
                self.rtables[region_id].insert_key(
                    self._lkey(cut, key), str(node_id),
                    _RL_VRAM if tier == 'VRAM' else _RL_DRAM)
            else:
                self.table[cut][key][node_id] = (cached_len, tier,
                                                 self._counter, region_id)
            if content_keys is not None:
                reg_prefixes.setdefault(prefix_id, {})[cut] = key
        # Register chunk-level hashes so Node evictions/demotions
        # (CHUNK_SIZES-granular) map back to this prefix's cut entries.
        if content_keys is not None:
            cidx = self._chunk_index[node_id]
            for lvl, cs in enumerate(CHUNK_SIZES):
                if cached_len < cs:
                    break
                k = content_keys.get(cs)
                if k is None:
                    break
                cidx.setdefault((lvl, k), set()).add(prefix_id)

    def _demote_chunk(self, node_id, lvl, chunk_key):
        """Chunk level `lvl` moved VRAM→DRAM: every cut entry spanning it
        (cut > CHUNK_SIZES[lvl-1]) now requires a PCIe reload → tier DRAM."""
        thresh = CHUNK_SIZES[lvl - 1] if lvl > 0 else 0
        for uid in self._chunk_index[node_id].get((lvl, chunk_key), ()):
            cuts = self._prefixes[node_id].get(uid)
            if not cuts:
                continue
            for cut, key in cuts.items():
                if cut <= thresh:
                    continue
                if self.use_native:
                    self.rtables[NODE_REGION[node_id]].update_tier_key(
                        self._lkey(cut, key), str(node_id), _RL_DRAM)
                    continue
                row = self.table.get(cut)
                inner = row.get(key) if row else None
                ent = inner.get(node_id) if inner else None
                if ent is not None:
                    inner[node_id] = (ent[0], 'DRAM', ent[2], ent[3])

    def invalidate(self, node_id, prefix_id):
        """Legacy prefix-id bulk invalidation (also cleans the registry)."""
        cuts = self._prefixes[node_id].pop(prefix_id, None)
        if cuts:
            for cut, key in cuts.items():
                if self.use_native:
                    self.rtables[NODE_REGION[node_id]].remove_worker_key(
                        self._lkey(cut, key), str(node_id))
                    continue
                tab = self.table.get(cut)
                inner = tab.get(key) if tab else None
                if inner is not None:
                    inner.pop(node_id, None)
                    if not inner:
                        del tab[key]
            return
        # prefix_id-key mode (no content hashes): rows keyed by prefix_id.
        if self.use_native:
            X = NODE_REGION[node_id]
            for cut in self.region_cuts[X]:
                self.rtables[X].remove_worker_key(
                    self._lkey(cut, prefix_id), str(node_id))
            return
        for cut, tab in list(self.table.items()):
            inner = tab.get(prefix_id)
            if inner is not None:
                inner.pop(node_id, None)
                if not inner:
                    del tab[prefix_id]

    def invalidate_content(self, node_id, level, content_key):
        """Node evicted chunk `level`: contiguity beyond CHUNK_SIZES[level-1]
        is broken, so remove every cut entry longer than that for each
        prefix on this node whose level-`level` hash is `content_key`."""
        uids = self._chunk_index[node_id].pop((level, content_key), None)
        if not uids:
            return
        thresh = CHUNK_SIZES[level - 1] if level > 0 else 0
        for uid in uids:
            cuts = self._prefixes[node_id].get(uid)
            if not cuts:
                continue
            for cut in [c for c in cuts if c > thresh]:
                key = cuts.pop(cut)
                if self.use_native:
                    self.rtables[NODE_REGION[node_id]].remove_worker_key(
                        self._lkey(cut, key), str(node_id))
                    continue
                tab = self.table.get(cut)
                inner = tab.get(key) if tab else None
                if inner is not None:
                    inner.pop(node_id, None)
                    if not inner:
                        del tab[key]
            if not cuts:
                del self._prefixes[node_id][uid]


class DynamoDiscovery(DiscoveryPlane):
    """NVIDIA-Dynamo-style KV-cache-aware router, hierarchical across clusters.

    Baseline.  Each cluster (region) keeps an *accurate* per-cluster
    index of which local replica holds which prefix, kept in sync with the
    real caches via advertise/invalidate — exactly like Dynamo's KvIndexer fed
    by worker KV-cache events.  A request first consults its local cluster
    index; on a *local cache miss* it fans a lookup out to every other cluster
    (paying DYNAMO_XCLUSTER_LOOKUP_MS, the round-trip to the farthest region).
    Candidate scoring is Dynamo's cache-overlap + load cost, realised as the
    joint T_q + T_p + T_t scoring in route().

    The index layout mirrors LudoDiscovery (multi-level, content-addressed,
    contiguous longest-prefix match) but WITHOUT the latency-aware floor —
    Dynamo has no notion that a farther cluster needs a longer match.  The
    local-first / cross-cluster fan-out control flow lives in the
    `scheme == 'dynamo'` branch of route(); this class owns the accurate index
    and its (uncapped) lookup.
    """
    name = 'dynamo'

    def __init__(self):
        # table[level][content_key][node_id] = (cached_len, tier)
        self.table = [defaultdict(dict) for _ in CHUNK_SIZES]
        # Per-region lookup coordinator: next-free time (M/D/1 queue) for the
        # inbound cross-cluster broadcast queries this region answers.
        self.region_lookup_free = [0.0] * NUM_REGIONS
        # Broadcast load view — SAME dissemination realism as Ludo's: a
        # hierarchical gateway sees its own cluster live but learns remote
        # clusters' load only through a periodic (100 ms) EWMA broadcast.
        # Without this, Dynamo scored remote nodes with the omniscient
        # instantaneous queue — an information asymmetry that mattered most
        # past saturation, where queues move faster than any real
        # dissemination could track.
        self._lb_ewma: Dict[int, float] = {}
        self._lb_last: Optional[float] = None
        self._lb_pending: Dict[int, float] = {}

    def reset(self):
        self.table = [defaultdict(dict) for _ in CHUNK_SIZES]
        self.region_lookup_free = [0.0] * NUM_REGIONS
        self._lb_ewma = {}
        self._lb_last = None
        self._lb_pending = {}

    def lb_probe(self, nodes, current_time):
        """Refresh the broadcast load view (see LudoDiscovery.lb_probe)."""
        if (self._lb_last is None
                or current_time - self._lb_last >= LUDO_LB_BCAST_MS):
            b = LUDO_LB_EWMA_BETA
            for n in nodes:
                q = n.queue_ms(current_time)
                prev = self._lb_ewma.get(n.node_id)
                self._lb_ewma[n.node_id] = (q if prev is None
                                            else b * q + (1.0 - b) * prev)
            self._lb_last = current_time
            self._lb_pending = {}

    def lb_view_queue(self, node, req_region, current_time):
        """Viewed queue: live for the requester's own cluster; remote =
        broadcast EWMA + own dispatches since the broadcast (see
        LudoDiscovery.lb_view_queue — prevents window herding)."""
        if node.region_id == req_region:
            return node.queue_ms(current_time)
        return (self._lb_ewma.get(node.node_id, 0.0)
                + self._lb_pending.get(node.node_id, 0.0))

    def lb_note_dispatch(self, node_id, service_ms):
        self._lb_pending[node_id] = (self._lb_pending.get(node_id, 0.0)
                                     + service_ms / MAX_BATCH_SIZE)

    def broadcast(self, req_region: int, current_time: float) -> float:
        """Cross-cluster fan-out latency on a local miss (wait for all).

        The requesting region sends a lookup query to every other region in
        parallel.  Each query queues at that region's lookup coordinator
        (M/D/1, DYNAMO_LOOKUP_SERVICE_MS service time), so under heavy
        broadcast load the coordinators congest.  We charge the *slowest*
        response, since the requester must hear from all regions to pick the
        globally-best cache holder:

            latency = max_{r≠s} ( 2·WAN[s][r] + queue_r + service )

        The coordinator queue is advanced at `current_time` (request-routing
        time); the one-way WAN legs are added only to the latency so the
        per-region next-free times stay monotonic in processing order.
        """
        worst = 0.0
        for r in range(NUM_REGIONS):
            if r == req_region:
                continue
            start = max(self.region_lookup_free[r], current_time)
            queue = start - current_time
            self.region_lookup_free[r] = start + DYNAMO_LOOKUP_SERVICE_MS
            rt = (2.0 * float(WAN_LATENCY_MS[req_region][r])
                  + queue + DYNAMO_LOOKUP_SERVICE_MS)
            if rt > worst:
                worst = rt
        return worst

    def _key(self, prefix_id, content_keys, cs):
        if content_keys is not None:
            k = content_keys.get(cs)
            if k is not None:
                return k
        return prefix_id

    def lookup(self, prefix_id, prefix_len, request_region=0, content_keys=None):
        """Accurate contiguous longest-prefix match per node (no floor)."""
        contig: Dict[int, Tuple[int, str]] = {}
        for l, cs in enumerate(CHUNK_SIZES):
            if prefix_len < cs:
                break
            key = self._key(prefix_id, content_keys, cs)
            row = self.table[l].get(key, {})
            if not row:
                continue
            for node_id, (_cl, tier) in row.items():
                if l == 0:
                    contig[node_id] = (0, tier)
                elif node_id in contig and contig[node_id][0] == l - 1:
                    prev_tier = contig[node_id][1]
                    new_tier  = 'DRAM' if (prev_tier == 'DRAM' or tier == 'DRAM') else 'VRAM'
                    contig[node_id] = (l, new_tier)
        return [(nid, min(CHUNK_SIZES[lvl], prefix_len), tier)
                for nid, (lvl, tier) in contig.items()]

    def advertise(self, node_id, region_id, prefix_id, cached_len, tier='VRAM',
                  content_keys=None):
        for l, cs in enumerate(CHUNK_SIZES):
            if cached_len >= cs:
                key = self._key(prefix_id, content_keys, cs)
                self.table[l][key][node_id] = (cached_len, tier)

    def invalidate(self, node_id, prefix_id):
        for l in range(len(CHUNK_SIZES)):
            row = self.table[l].get(prefix_id)
            if row:
                row.pop(node_id, None)
                if not row:
                    del self.table[l][prefix_id]

    def invalidate_content(self, node_id, level, content_key):
        row = self.table[level].get(content_key)
        if row:
            row.pop(node_id, None)
            if not row:
                del self.table[level][content_key]


class SkyWalkerDiscovery(DiscoveryPlane):
    """SkyWalker (EuroSys'26) — two-layer, locality-first, prefix-tree LB.

    Reproduces the paper's prefix-tree variant (its headline system).  Each
    region hosts one logical load balancer holding two *approximate* prefix
    tries built incrementally from the requests it routed (bounded, FIFO):

      replica_trie[region][level][content_key]  -> {local node_ids served}
      snapshot_trie[region][level][content_key] -> {remote regions forwarded to}

    The tries only *guide routing*; they are NOT kept in sync with the
    replicas' real KV caches (the paper's "regional snapshot" is deliberately
    approximate).  The actual served match is therefore recomputed against the
    chosen replica's real cache in route().  Routing itself (two-layer,
    locality-first, availability-gated, longest-prefix tie-break) lives in the
    `scheme == 'skywalker'` branch of route(); this class only owns the trie
    state and the longest-available-prefix queries.

    advertise()/invalidate*() are intentionally the base-class no-ops: the
    tries are updated via record() from route(), not from the simulator's
    cache-insert/evict hooks.
    """
    name = 'skywalker'

    def __init__(self):
        # Bound each per-level trie to ≈ a single node's cache scale (in
        # entries): NODE_VRAM_TOKENS / smallest chunk.  FIFO eviction.
        self._max = max(1, NODE_VRAM_TOKENS // CHUNK_SIZES[0])
        self.replica_trie: List[List[OrderedDict]] = [
            [OrderedDict() for _ in CHUNK_SIZES] for _ in range(NUM_REGIONS)]
        self.snapshot_trie: List[List[OrderedDict]] = [
            [OrderedDict() for _ in CHUNK_SIZES] for _ in range(NUM_REGIONS)]
        # Heartbeat availability snapshot: free-slot credits per node,
        # refreshed every probe interval and consumed per push in between.
        self._credits: Dict[int, int] = {}
        self._last_probe: Optional[float] = None

    def reset(self):
        self.replica_trie = [
            [OrderedDict() for _ in CHUNK_SIZES] for _ in range(NUM_REGIONS)]
        self.snapshot_trie = [
            [OrderedDict() for _ in CHUNK_SIZES] for _ in range(NUM_REGIONS)]
        self._credits = {}
        self._last_probe = None

    def probe(self, nodes, current_time):
        """Refresh the availability snapshot at most once per probe interval.

        The snapshot records each replica's *free batch slots* at probe time
        (its admission capacity as the LB last saw it).  Between probes the
        LB plays the paper's selective pushing: every request it routes to a
        replica consumes one credit (the LB is omniscient about its own
        pushes), so a replica stops looking available the moment the LB has
        filled the capacity it saw — no pile-on beyond the probed slots.
        Staleness still costs: capacity freed mid-window is invisible until
        the next heartbeat, so the LB under-utilises between probes.
        """
        if (self._last_probe is None
                or current_time - self._last_probe >= SKYWALKER_PROBE_INTERVAL_MS):
            self._credits = {n.node_id: n.free_slots(current_time)
                             for n in nodes}
            self._last_probe = current_time

    def is_avail(self, node_id: int) -> bool:
        return self._credits.get(node_id, 0) > 0

    def consume(self, node_id: int):
        """One selective push to `node_id` — burn a free-slot credit."""
        c = self._credits.get(node_id, 0)
        if c > 0:
            self._credits[node_id] = c - 1

    def lookup(self, prefix_id, prefix_len, request_region=0, content_keys=None):
        # SkyWalker routing is handled entirely in route(); the generic
        # candidate lookup is unused.
        return []

    @staticmethod
    def _insert(level_tries, content_keys, prefix_len, target, max_entries):
        """Record `target` at every content-hash level this prefix reaches."""
        if content_keys is None:
            return
        for l, cs in enumerate(CHUNK_SIZES):
            if prefix_len < cs:
                break
            key = content_keys.get(cs)
            if key is None:
                break
            d = level_tries[l]
            if key in d:
                d[key].add(target)
                d.move_to_end(key)
            else:
                d[key] = {target}
                if len(d) > max_entries:
                    d.popitem(last=False)          # FIFO evict earliest

    @staticmethod
    def _match(level_tries, content_keys, prefix_len, avail):
        """Longest contiguous prefix whose holders intersect `avail`.

        Returns the set of tied longest-match targets (empty if none).  Walks
        levels low→high; stops at the first level with no *available* holder
        (child target sets ⊆ parent's, so no deeper match can exist).
        """
        best: set = set()
        if content_keys is None:
            return best
        for l, cs in enumerate(CHUNK_SIZES):
            if prefix_len < cs:
                break
            key = content_keys.get(cs)
            if key is None:
                break
            holders = level_tries[l].get(key)
            if not holders:
                break
            inter = holders & avail
            if not inter:
                break
            best = inter
        return best

    def local_match(self, region, content_keys, prefix_len, avail_nodes):
        """Available local replicas with the longest prefix match."""
        return self._match(self.replica_trie[region], content_keys,
                           prefix_len, avail_nodes)

    def snapshot_match(self, origin_region, content_keys, prefix_len,
                       avail_regions):
        """Available remote regions whose snapshot has the longest match."""
        return self._match(self.snapshot_trie[origin_region], content_keys,
                           prefix_len, avail_regions)

    def record(self, origin_region, node_id, node_region, content_keys,
               prefix_len):
        """Update tries after a routing decision.

        The serving region's replica trie records that `node_id` holds this
        prefix; a cross-region forward also updates the origin region's
        snapshot trie with the target region.
        """
        self._insert(self.replica_trie[node_region], content_keys,
                     prefix_len, node_id, self._max)
        if node_region != origin_region:
            self._insert(self.snapshot_trie[origin_region], content_keys,
                         prefix_len, node_region, self._max)


class RandomLLDiscovery(DiscoveryPlane):
    """No cache awareness; always returns empty list → fallback to least-loaded."""
    name = 'random_ll'

    def lookup(self, prefix_id, prefix_len, request_region=0,
               content_keys=None):
        return []


class ConsistentHashDiscovery(DiscoveryPlane):
    """
    Multi-level consistent hashing.

    Mirrors Ludo's CHUNK_SIZES levels so all schemes now share the same
    multi-granularity matching capability — the only remaining structural
    difference is how placement is derived.  For consistent hashing the
    rule is hash-placement without migration: at level l a cached entry
    is indexed only if the publisher node happens to equal the level-l
    hash-owner, hash((prefix_id, l)) % N.  Lookup queries every level with
    prefix_len ≥ chunk_size and returns each hit as a separate candidate
    so the scheduler can pick the longest match across aligned levels.

    Cache entries typically land at different nodes across levels (the
    level-l owner rarely equals the level-(l+k) owner), so on average a
    given cache appears in only ≈ 1 out of N levels — this is the cost of
    hash-derived placement that Ludo's explicit-write value field avoids.
    """
    name = 'consistent_hash'

    def __init__(self):
        # state[level][prefix_id] = (owner_node_id, cached_len, tier)
        self.state: List[Dict[int, Tuple[int, int, str]]] = [
            {} for _ in CHUNK_SIZES
        ]

    def reset(self):
        self.state = [{} for _ in CHUNK_SIZES]

    @staticmethod
    def _owner(key: int, level: int) -> int:
        """Hash-owner at a given level.  Independent hash per level."""
        return hash((key, level)) % NUM_NODES

    def _key_at(self, prefix_id, content_keys, cs):
        if content_keys is not None:
            k = content_keys.get(cs)
            if k is not None:
                return k
        return prefix_id

    def lookup(self, prefix_id, prefix_len, request_region=0,
               content_keys=None):
        # Contiguous-level walk: same pattern as LudoDiscovery.lookup.  At
        # most one node per level (hash-placement), so contig becomes a
        # simple (deepest_level, node, tier) triple.
        deepest = -1
        node_id = -1
        tier    = 'MISS'
        for l, cs in enumerate(CHUNK_SIZES):
            if prefix_len < cs:
                break
            key = self._key_at(prefix_id, content_keys, cs)
            entry = self.state[l].get(key)
            if entry is None:
                break                        # non-contiguous → stop
            nid, _cl, t = entry
            if deepest == -1:
                deepest, node_id, tier = l, nid, t
            elif nid == node_id:
                deepest = l
                if t == 'DRAM':
                    tier = 'DRAM'
            else:
                # Different hash-owner at this level → can't extend the
                # contig run for the level-0 owner.  Stop.
                break
        if deepest < 0:
            return []
        return [(node_id, min(CHUNK_SIZES[deepest], prefix_len), tier)]

    def advertise(self, node_id, region_id, prefix_id, cached_len, tier='VRAM',
                  content_keys=None):
        # Hash-placement without migration: a node's cache shows up at level
        # l only if it happens to own the level-l hash.  Probability ≈ 1/N
        # per level, independently.
        for l, cs in enumerate(CHUNK_SIZES):
            if cached_len < cs:
                continue
            key = self._key_at(prefix_id, content_keys, cs)
            if node_id == self._owner(key, l):
                self.state[l][key] = (node_id, cached_len, tier)

    def invalidate(self, node_id, prefix_id):
        for l in range(len(CHUNK_SIZES)):
            entry = self.state[l].get(prefix_id)
            if entry and entry[0] == node_id:
                del self.state[l][prefix_id]

    def invalidate_content(self, node_id, level, content_key):
        entry = self.state[level].get(content_key)
        if entry and entry[0] == node_id:
            del self.state[level][content_key]


class KademliaDHTDiscovery(DiscoveryPlane):
    """
    Multi-level Kademlia DHT (hash-placement, no migration).

    Same multi-level index layout as the Ludo table, but each (prefix_id,
    level) key is owned by the XOR-closest node — a locally-generated KV
    cache is only indexed at level l iff the computing node happens to be
    that key's hash-owner.  No migration → per-level index coverage stays
    at ≈ 1/N even though the matching granularity now extends up to
    CHUNK_SIZES[-1] tokens.

    Per-decision WAN overhead (KADEMLIA_LOOKUP_OVERHEAD_MS ≈ 280 ms) is
    charged once per routing decision: the six level queries are
    dispatched as parallel α = 3 probes in a single DHT session, so
    wall-clock latency is bounded by the slowest level lookup, not the
    sum.
    """
    name = 'kademlia'

    def __init__(self):
        # state[level][prefix_id] = (owner_node_id, cached_len, tier)
        self.state: List[Dict[int, Tuple[int, int, str]]] = [
            {} for _ in CHUNK_SIZES
        ]

    def reset(self):
        self.state = [{} for _ in CHUNK_SIZES]

    @staticmethod
    def _owner(key: int, level: int) -> int:
        return hash((key, level)) % NUM_NODES

    def _key_at(self, prefix_id, content_keys, cs):
        if content_keys is not None:
            k = content_keys.get(cs)
            if k is not None:
                return k
        return prefix_id

    def lookup(self, prefix_id, prefix_len, request_region=0,
               content_keys=None):
        deepest = -1
        node_id = -1
        tier    = 'MISS'
        for l, cs in enumerate(CHUNK_SIZES):
            if prefix_len < cs:
                break
            key = self._key_at(prefix_id, content_keys, cs)
            entry = self.state[l].get(key)
            if entry is None:
                break
            nid, _cl, t = entry
            if deepest == -1:
                deepest, node_id, tier = l, nid, t
            elif nid == node_id:
                deepest = l
                if t == 'DRAM':
                    tier = 'DRAM'
            else:
                break
        if deepest < 0:
            return []
        return [(node_id, min(CHUNK_SIZES[deepest], prefix_len), tier)]

    def advertise(self, node_id, region_id, prefix_id, cached_len, tier='VRAM',
                  content_keys=None):
        for l, cs in enumerate(CHUNK_SIZES):
            if cached_len < cs:
                continue
            key = self._key_at(prefix_id, content_keys, cs)
            if node_id == self._owner(key, l):
                self.state[l][key] = (node_id, cached_len, tier)

    def invalidate(self, node_id, prefix_id):
        for l in range(len(CHUNK_SIZES)):
            entry = self.state[l].get(prefix_id)
            if entry and entry[0] == node_id:
                del self.state[l][prefix_id]

    def invalidate_content(self, node_id, level, content_key):
        entry = self.state[level].get(content_key)
        if entry and entry[0] == node_id:
            del self.state[level][content_key]


class ChordDHTDiscovery(DiscoveryPlane):
    """
    Multi-level Chord DHT (hash-placement, no migration).

    Structural twin of the multi-level Kademlia DHT: six independent ring
    mappings, one per CHUNK_SIZES level, keyed by hash((prefix_id, l)) and
    populated only when the publishing node happens to be the level-l
    ring successor.  The matching-granularity improvement is identical to
    Kademlia's; the remaining per-decision cost is Chord's characteristic
    sequential finger-table walk — charged as a single CHORD_LOOKUP_OVER-
    HEAD_MS (~467 ms) on the critical path per decision (the six level
    queries share one ring traversal).
    """
    name = 'chord'

    def __init__(self):
        # state[level][prefix_id] = (successor_node_id, cached_len, tier)
        self.state: List[Dict[int, Tuple[int, int, str]]] = [
            {} for _ in CHUNK_SIZES
        ]

    def reset(self):
        self.state = [{} for _ in CHUNK_SIZES]

    @staticmethod
    def _successor(key: int, level: int) -> int:
        return hash((key, level)) % NUM_NODES

    def _key_at(self, prefix_id, content_keys, cs):
        if content_keys is not None:
            k = content_keys.get(cs)
            if k is not None:
                return k
        return prefix_id

    def lookup(self, prefix_id: int, prefix_len: int,
               request_region: int = 0,
               content_keys: Optional[Dict[int, int]] = None
               ) -> List[Tuple[int, int, str]]:
        deepest = -1
        node_id = -1
        tier    = 'MISS'
        for l, cs in enumerate(CHUNK_SIZES):
            if prefix_len < cs:
                break
            key = self._key_at(prefix_id, content_keys, cs)
            entry = self.state[l].get(key)
            if entry is None:
                break
            nid, _cl, t = entry
            if deepest == -1:
                deepest, node_id, tier = l, nid, t
            elif nid == node_id:
                deepest = l
                if t == 'DRAM':
                    tier = 'DRAM'
            else:
                break
        if deepest < 0:
            return []
        return [(node_id, min(CHUNK_SIZES[deepest], prefix_len), tier)]

    def advertise(self, node_id: int, region_id: int,
                  prefix_id: int, cached_len: int, tier: str = 'VRAM',
                  content_keys: Optional[Dict[int, int]] = None):
        for l, cs in enumerate(CHUNK_SIZES):
            if cached_len < cs:
                continue
            key = self._key_at(prefix_id, content_keys, cs)
            if node_id == self._successor(key, l):
                self.state[l][key] = (node_id, cached_len, tier)

    def invalidate(self, node_id: int, prefix_id: int):
        for l in range(len(CHUNK_SIZES)):
            entry = self.state[l].get(prefix_id)
            if entry and entry[0] == node_id:
                del self.state[l][prefix_id]

    def invalidate_content(self, node_id: int, level: int, content_key: int):
        entry = self.state[level].get(content_key)
        if entry and entry[0] == node_id:
            del self.state[level][content_key]


class CentralizedTrackerDiscovery(DiscoveryPlane):
    """
    Perfect centralized index: exact cache state of every node.
    Placement–identity decoupled; a single bottleneck at scale.
    """
    name = 'centralized'

    def __init__(self):
        self.index: Dict[int, Dict[int, Tuple[int, str]]] = defaultdict(dict)

    def reset(self):
        self.index = defaultdict(dict)

    def lookup(self, prefix_id, prefix_len, request_region=0,
               content_keys=None):
        # CentralizedTracker still uses prefix_id as its bookkeeping key —
        # it's not content-addressed by design.  content_keys is accepted
        # for interface uniformity but ignored.
        return [(nid, min(cl, prefix_len), tier)
                for nid, (cl, tier) in self.index.get(prefix_id, {}).items()]

    def advertise(self, node_id, region_id, prefix_id, cached_len, tier='VRAM',
                  content_keys=None):
        self.index[prefix_id][node_id] = (cached_len, tier)

    def invalidate(self, node_id, prefix_id):
        row = self.index.get(prefix_id)
        if row:
            row.pop(node_id, None)
            if not row:
                del self.index[prefix_id]


class RadixTreeDiscovery(DiscoveryPlane):
    """
    Faithful re-implementation of PlanetServe's HashRadixTree
    (deps/planetserve/eval/hrt+lb/HashRadixTree/HashRadixTree.py).

    Routing logic mirrors the original exactly:
      - chunk_size = 10 tokens, gate = 10 chunks (100-token minimum match)
      - Lossy hash mod=15 per chunk (same as PlanetServe's hash_chunk)
      - Cache hit  (depth ≥ gate): select least-loaded node from model_list;
        if that node's load exceeds the 80th-percentile cutoff, climb the
        trie upward (load_balance) to find a less-loaded ancestor node.
      - Cache miss (depth < gate): route to a random candidate node and
        register it at the newly inserted trie leaf (PlanetServe default).
      - Parent pointers stored in every trie node to support tree-climb.
      - advertise() stores each entry at depth = min(gate, cached_len //
        chunk_size), so short prefixes occupy shallower nodes and are
        discoverable during tree-climb fallback.
    """
    name       = 'radix_tree'
    CHUNK_SIZE = 10   # tokens per chunk (PlanetServe original)
    GATE       = 10   # chunks for a match (100-token minimum match)
    HASH_MOD   = 15   # lossy hash modulus (same as PlanetServe)

    # ── Internal trie node factory ────────────────────────────────────────
    @staticmethod
    def _new_node(parent=None, depth: int = 0) -> dict:
        # node_cache: {node_id: (cached_len, tier)}
        # parent / depth needed for tree-climb load balancing.
        return {'children': {}, 'node_cache': {}, 'parent': parent, 'depth': depth}

    def __init__(self):
        self.root = self._new_node()
        # (node_id, prefix_id) → trie node where the entry lives (O(1) invalidation)
        self._reg: Dict[Tuple[int, int], dict] = {}
        # prefix_id → list of GATE hash values (computed once, reused)
        self._path_cache: Dict[int, List[int]] = {}

    def reset(self):
        self.root = self._new_node()
        self._reg = {}
        self._path_cache = {}

    # ── Hash helper ── mirrors PlanetServe's hash_chunk(chunk, mod=15) ───
    # Legacy fallback used only when the caller does NOT supply a real
    # radix_path (i.e. from §4 baselines.py which has no tokens).  For the
    # §3 simulator path, WorkloadGenerator precomputes the real chunk path
    # from actual prompt tokens and the simulator passes it through.
    @staticmethod
    def _hash_chunk(prefix_id: int, chunk_idx: int) -> int:
        """Deterministic lossy hash for (prefix_id, chunk_idx), mod 15."""
        return ((prefix_id * 2654435761) ^ (chunk_idx * 2246822519)) % RadixTreeDiscovery.HASH_MOD

    def _get_path(self, prefix_id: int,
                  radix_path: Optional[List[int]] = None) -> List[int]:
        """GATE-length chunk hash path.

        Prefers the caller-supplied `radix_path` (real 10-token content
        hashes built by WorkloadGenerator); falls back to the pseudo hash
        over (prefix_id, chunk_idx) for callers that lack token content.
        """
        if radix_path is not None and radix_path:
            # Truncate or pad to exactly GATE length so trie walks never
            # run off the end.  PlanetServe already caps at GATE anyway.
            return list(radix_path[:self.GATE])
        if prefix_id not in self._path_cache:
            self._path_cache[prefix_id] = [
                self._hash_chunk(prefix_id, i) for i in range(self.GATE)
            ]
        return self._path_cache[prefix_id]

    # ── DiscoveryPlane interface ──────────────────────────────────────────
    def lookup(self, prefix_id: int, prefix_len: int,
               request_region: int = 0,
               content_keys: Optional[Dict[int, int]] = None,
               radix_path: Optional[List[int]] = None
               ) -> List[Tuple[int, int, str]]:
        """Standard lookup used by non-radix_tree routing paths.
        Returns candidates only when depth ≥ GATE (strict hit condition).
        match_len is capped at GATE × CHUNK_SIZE = 100 tokens — the true
        content-derived match granularity of PlanetServe's design."""
        max_depth = min(self.GATE, prefix_len // self.CHUNK_SIZE)
        if max_depth == 0:
            return []
        path  = self._get_path(prefix_id, radix_path)
        current = self.root
        depth = 0
        for hval in path[:max_depth]:
            if hval not in current['children']:
                break
            current = current['children'][hval]
            depth += 1
            if depth >= self.GATE:
                break
        if depth < self.GATE or not current['node_cache']:
            return []
        match_len = min(depth * self.CHUNK_SIZE, prefix_len)
        return [(nid, match_len, tier)
                for nid, (_cl, tier) in current['node_cache'].items()]

    def advertise(self, node_id: int, region_id: int,
                  prefix_id: int, cached_len: int, tier: str = 'VRAM',
                  content_keys: Optional[Dict[int, int]] = None,
                  radix_path: Optional[List[int]] = None):
        """Insert/update an entry at depth = min(GATE, cached_len // CHUNK_SIZE).
        Removes any stale entry at a different depth first (handles tier updates
        and cached_len growth correctly).  When the caller supplies a real
        `radix_path` (10-token content hashes), two different prefix_ids that
        share real tokens will walk to the same trie leaf — real prefix
        sharing gets indexed here the way PlanetServe's production code
        would.
        """
        target_depth = min(self.GATE, cached_len // self.CHUNK_SIZE)
        if target_depth == 0:
            return

        # Remove stale registration if it points to a different node
        old_trie_node = self._reg.get((node_id, prefix_id))
        if old_trie_node is not None:
            old_trie_node['node_cache'].pop(node_id, None)

        path = self._get_path(prefix_id, radix_path)
        current = self.root
        for i, hval in enumerate(path[:target_depth]):
            if hval not in current['children']:
                child = self._new_node(parent=current, depth=i + 1)
                current['children'][hval] = child
            current = current['children'][hval]

        current['node_cache'][node_id] = (cached_len, tier)
        self._reg[(node_id, prefix_id)] = current

    def invalidate(self, node_id: int, prefix_id: int):
        key = (node_id, prefix_id)
        trie_node = self._reg.pop(key, None)
        if trie_node is not None:
            trie_node['node_cache'].pop(node_id, None)

    def invalidate_content(self, node_id: int, level: int, content_key: int):
        # RadixTree does not index by (level, content_hash) — its granularity
        # is a full GATE-chunk trie walk, not per-Ludo-level buckets.  The
        # simulator falls back to per-(node_id, prefix_id) invalidation via
        # the Simulator's own bookkeeping (pid_in_use tracking).
        _ = (level, content_key)

    # ── PlanetServe routing ───────────────────────────────────────────────
    def planetserve_route(self, prefix_id: int, prefix_len: int,
                          request_region: int,
                          nodes: List['Node'], current_time: float,
                          radix_path: Optional[List[int]] = None
                          ) -> Tuple[int, int, str]:
        """Full PlanetServe routing pipeline (HashRadixTree.find_match_model +
        the external load-balance check from the scheduler loop).

        Phase 1 — trie traversal (find_match_model):
          Traverse up to min(GATE, prefix_len // CHUNK_SIZE) chunks.
          Hit  (depth ≥ GATE): pick least-loaded node from matched node_cache.
          Miss (depth  < GATE): insert trie path, route to random candidate.

        Phase 2 — load_balance (scheduler-side check):
          If the matched node's queue_ms exceeds the 80th-percentile cutoff
          of all active nodes, climb the trie upward (load_balance) to find
          an ancestor node whose models fall below the cutoff.
          Falls back to global least-loaded if the root is reached.

        Returns (node_id, match_len, tier).
        """
        max_depth = min(self.GATE, prefix_len // self.CHUNK_SIZE)

        # ── Phase 1: trie traversal ───────────────────────────────────────
        path    = self._get_path(prefix_id, radix_path)
        current = self.root
        depth   = 0
        for hval in path[:max_depth]:
            if hval not in current['children']:
                break
            current = current['children'][hval]
            depth  += 1
            if depth >= self.GATE:
                break

        if depth >= self.GATE and current['node_cache']:
            # ── Cache hit: least-loaded from model_list ───────────────────
            best_nid  = min(current['node_cache'],
                            key=lambda nid: nodes[nid].queue_ms(current_time))
            _cl, tier  = current['node_cache'][best_nid]
            # Content-addressed cap: PlanetServe's gate commits to exactly
            # `depth × CHUNK_SIZE` tokens of matched content; the
            # cached_len field (metadata about the source prompt) cannot be
            # used as the reported match for cross-request collisions.
            match_len = min(depth * self.CHUNK_SIZE, prefix_len)
            matched_trie_node = current

            # ── Phase 2: load_balance check (80th-percentile cutoff) ──────
            active_loads = sorted(
                n.queue_ms(current_time) for n in nodes
                if n.queue_ms(current_time) > 0)
            if active_loads and len(active_loads) > 1:
                cutoff_idx = max(int(math.ceil(0.8 * len(active_loads))) - 1, 0)
                cutoff     = active_loads[cutoff_idx]
                if nodes[best_nid].queue_ms(current_time) > cutoff:
                    climbed = self._load_balance(matched_trie_node, nodes,
                                                 current_time, cutoff)
                    if climbed is not None:
                        climb_nid, climb_depth = climbed
                        _cl2, tier2 = (current['node_cache'].get(climb_nid)
                                       or (0, 'MISS'))
                        best_nid  = climb_nid
                        match_len = (min(climb_depth * self.CHUNK_SIZE, prefix_len)
                                     if tier2 != 'MISS' else 0)
                        tier      = tier2

            return best_nid, match_len, tier

        else:
            # ── Cache miss: insert trie path, route to least-loaded node ──
            # PlanetServe uses random.choice(candidate_models); we use
            # least-loaded (PlanetServe's advanced mode) for reproducibility.
            miss_nid = _nearest_least_loaded(nodes, request_region, current_time)
            # Register this node at the newly reached trie leaf so future
            # requests for the same prefix will find it (assign_workload).
            current['node_cache'][miss_nid] = (prefix_len, 'MISS')
            self._reg[(miss_nid, prefix_id)] = current
            return miss_nid, 0, 'MISS'

    def _load_balance(self, trie_node: dict, nodes: List['Node'],
                      current_time: float, cutoff: float
                      ) -> Optional[Tuple[int, int]]:
        """Climb the trie toward the root looking for a model whose queue_ms
        is below *cutoff*.  Mirrors PlanetServe's HashRadixTree.load_balance.

        Returns (node_id, trie_depth) of the first acceptable ancestor model,
        or None if the root is reached without finding one (the caller falls
        back to global least-loaded in that case).
        """
        parent = trie_node.get('parent')
        if parent is None:
            return None           # reached root — no ancestor found
        for nid in parent['node_cache']:
            if nodes[nid].queue_ms(current_time) < cutoff:
                return nid, parent['depth']
        return self._load_balance(parent, nodes, current_time, cutoff)


class OracleDiscovery(DiscoveryPlane):
    """
    Offline omniscient oracle: perfect knowledge of all node cache states for
    routing (joint scoring), combined with Bélády's optimal cache replacement
    which requires knowing the full future request sequence in advance.
    True theoretical lower-bound on TTFT / upper-bound on cache hit rate.
    """
    name = 'oracle'

    def __init__(self):
        self.index: Dict[int, Dict[int, Tuple[int, str]]] = defaultdict(dict)

    def reset(self):
        self.index = defaultdict(dict)

    def lookup(self, prefix_id, prefix_len, request_region=0,
               content_keys=None):
        return [(nid, min(cl, prefix_len), tier)
                for nid, (cl, tier) in self.index.get(prefix_id, {}).items()]

    def advertise(self, node_id, region_id, prefix_id, cached_len, tier='VRAM',
                  content_keys=None, radix_path=None):
        self.index[prefix_id][node_id] = (cached_len, tier)

    def invalidate(self, node_id, prefix_id):
        row = self.index.get(prefix_id)
        if row:
            row.pop(node_id, None)


# ─── Routing / scoring ────────────────────────────────────────────────────────

SIM_WAN_JITTER      = float(os.environ.get('SIM_WAN_JITTER', '0') or 0)
SIM_UPDATE_LOSS     = float(os.environ.get('SIM_UPDATE_LOSS', '0') or 0)
SIM_UPDATE_DELAY_MS = float(os.environ.get('SIM_UPDATE_DELAY_MS', '0') or 0)
_ROBUST_ACTIVE = (SIM_WAN_JITTER > 0 or SIM_UPDATE_LOSS > 0
                  or SIM_UPDATE_DELAY_MS > 0)


def _transfer_ms(tier: str, node_region: int, req_region: int,
                 match_len: int, wan_jitter: float = 1.0) -> float:
    """Transfer cost to serve a cache hit.

    Two independent cost components, both applied when relevant:
      - WAN latency  : incurred whenever node and request are in different regions,
                       regardless of tier (the KV block must still travel the wire).
      - DRAM reload  : incurred whenever tier == 'DRAM', regardless of region
                       (PCIe DRAM→VRAM copy must happen before prefill can start).
    """
    wan  = (float(WAN_LATENCY_MS[req_region][node_region]) * wan_jitter
            if node_region != req_region else 0.0)
    dram = match_len * DRAM_RELOAD_MS_PER_TOKEN if tier == 'DRAM' else 0.0
    return wan + dram


def route(request: Request, discovery: DiscoveryPlane,
          nodes: List[Node], current_time: float,
          scheme: str,
          oracle_routing: Optional[Dict[int, int]] = None) -> RouteResult:
    """Select best node for the request under the given scheme."""
    candidates = discovery.lookup(
        request.prefix_id, request.prefix_len, request.region_id,
        content_keys=request.content_keys or None,
    )

    # ── Oracle: pre-computed optimal home-node routing + Bélády eviction ────────
    # oracle_routing maps each prefix_id to its pre-assigned home node, computed
    # offline with full knowledge of all future requests (greedy load-balanced,
    # region-affinity-aware).  Every request for prefix X always goes to the same
    # home node, maximising cache reuse.  Bélády's eviction (insert_beladys) then
    # ensures that when the home node's VRAM fills up, the eviction decisions are
    # globally optimal.  Together these give the tightest achievable upper bound.
    if scheme == 'oracle':
        # Oracle uses dynamic joint-scoring (load-aware) routing with offline scheduling foresight.
        best_per_node: Dict[int, Tuple[int, str]] = {
            n.node_id: (0, 'MISS') for n in nodes
        }
        for nid, ml, tier in candidates:
            if ml > best_per_node[nid][0]:
                best_per_node[nid] = (ml, tier)
        best_score = float('inf')
        best = None
        for nid, (ml, tier) in best_per_node.items():
            node = nodes[nid]
            # Offline Oracle schedules requests optimally to interleave tasks and bypass micro-burst
            # queue delays. Greedy online routing feels the full T_q, Oracle simulates perfect
            # packing by paying a fraction (0.1x) of the single-server queue penalty.
            T_q = node.queue_ms(current_time) * 0.1
            T_p = (request.prefix_len - ml) * C_PREFILL_MS_PER_TOKEN
            T_t = _transfer_ms(tier, node.region_id, request.region_id, ml)
            score = T_q + T_p + T_t
            if score < best_score:
                best_score = score
                best = (nid, ml, node.region_id, tier)
        nid, ml, nr, tier = best
        return RouteResult(nid, ml, nr, tier)

    # ── Random / least-loaded: no cache awareness ─────────────────────────────
    if scheme == 'random_ll':
        nid = _nearest_least_loaded(nodes, request.region_id, current_time)
        return RouteResult(nid, 0, nodes[nid].region_id, 'MISS')

    # ── Multi-level hash-placement schemes ───────────────────────────────────
    # Consistent Hash, Kademlia DHT, Chord DHT all share the same
    # multi-granularity lookup format now: `candidates` is a list of hits
    # (one per CHUNK_SIZES level the prefix is eligible for), each pointing
    # at that level's hash-owner.  Hash-placement without migration means
    # the same cache rarely aligns with the hash-owner at more than one
    # level, so `candidates` is usually 0 or 1 long — but when two levels
    # happen to align we pick the longer match.  Routing stays "always to
    # hash-owner"; there is no load-aware scoring (that is Ludo's
    # distinguishing feature, not these schemes').
    if scheme in ('consistent_hash', 'kademlia', 'chord'):
        if candidates:
            nid, ml, tier = max(candidates, key=lambda c: c[1])
        else:
            # Fallback: level-0 hash-owner — must match the owner function
            # the discovery plane uses internally so that *the routed node
            # is the level-0 hash-owner by construction*, which is what
            # lets the subsequent advertise() populate the level-0 slot.
            # Prefer the content-hash key; fall back to prefix_id only if
            # content_keys is unavailable (e.g., §4 baselines.py).
            if request.content_keys and CHUNK_SIZES[0] in request.content_keys:
                fb_key = request.content_keys[CHUNK_SIZES[0]]
            else:
                fb_key = request.prefix_id
            nid = hash((fb_key, 0)) % NUM_NODES
            ml, tier = 0, 'MISS'
        overhead = 0.0
        if scheme == 'kademlia':
            overhead = KADEMLIA_LOOKUP_OVERHEAD_MS
        elif scheme == 'chord':
            overhead = CHORD_LOOKUP_OVERHEAD_MS
        return RouteResult(nid, ml if tier != 'MISS' else 0,
                           nodes[nid].region_id, tier,
                           lookup_overhead_ms=overhead)

    # ── PlanetServe RadixTree: full HRT routing pipeline ─────────────────────
    if scheme == 'radix_tree':
        nid, ml, tier = discovery.planetserve_route(   # type: ignore[attr-defined]
            request.prefix_id, request.prefix_len,
            request.region_id, nodes, current_time,
            radix_path=request.radix_path or None)
        return RouteResult(nid, ml, nodes[nid].region_id, tier)

    # ── Dynamo: hierarchical KV-cache-aware routing ──────────────────────────
    # Accurate per-cluster index (candidates already computed above).  Consult
    # the local cluster first; only on a *local cache miss* fan a lookup out to
    # every other cluster, paying the round-trip probe latency.  Among the
    # candidate set we minimise Dynamo's cache-overlap + load cost = the joint
    # T_q + T_p + T_t.  No latency-aware floor (Dynamo has none).
    if scheme == 'dynamo':
        req_region = request.region_id
        discovery.lb_probe(nodes, current_time)
        local_cands = [(nid, ml, tier) for nid, ml, tier in candidates
                       if nodes[nid].region_id == req_region]
        local_hit   = any(ml > 0 for _, ml, _ in local_cands)
        local_ll    = _nearest_least_loaded(nodes, req_region, current_time)
        pool        = list(local_cands) + [(local_ll, 0, 'MISS')]
        lookup_ovh  = 0.0
        if not local_hit:
            # Local miss → broadcast a lookup to every other cluster; pay the
            # slowest coordinator's round-trip (propagation + M/D/1 queueing).
            lookup_ovh = discovery.broadcast(req_region, current_time)
            pool += [(nid, ml, tier) for nid, ml, tier in candidates
                     if nodes[nid].region_id != req_region]
        best_score = float('inf')
        best = None
        for nid, ml, tier in pool:
            node = nodes[nid]
            # Same information realism as Ludo: own cluster seen live, remote
            # clusters through the periodic EWMA load broadcast.
            T_q = discovery.lb_view_queue(node, req_region, current_time)
            T_p = (request.prefix_len - ml) * C_PREFILL_MS_PER_TOKEN
            T_t = _transfer_ms(tier, node.region_id, req_region, ml)
            score = T_q + T_p + T_t
            if score < best_score:
                best_score = score
                best = (nid, ml, node.region_id, tier)
        nid, ml, nr, tier = best
        discovery.lb_note_dispatch(
            nid, (request.prefix_len - ml) * C_PREFILL_MS_PER_TOKEN
                 + request.prefix_len * C_DECODE_MS_PER_TOKEN)
        return RouteResult(nid, ml, nr, tier, lookup_overhead_ms=lookup_ovh)

    # ── SkyWalker: two-layer, locality-first, prefix-tree routing ─────────────
    # (1) Availability comes from the 100 ms heartbeat snapshot (stale between
    #     probes): a replica is "available" iff it had a free continuous-batch
    #     slot (pending==0) at the last probe — so the LB can misroute to a
    #     replica that has since filled up (the modelled cost of probe lag).
    # (2) Locality-first: if ANY local replica is available, serve locally and
    #     never pay cross-region latency; only when the whole local region is
    #     full do we forward to an available remote region.
    # (3) Prefix-aware pick: among the available candidate set, choose the
    #     target with the longest matching prefix in the guiding trie
    #     (replica trie locally, snapshot trie for the region hop); ties and
    #     no-match fall back to least-loaded.
    # The trie only guides routing — the *served* match is recomputed against
    # the chosen replica's real cache (the snapshot is deliberately stale).
    if scheme == 'skywalker':
        req_region = request.region_id
        ck         = request.content_keys or None
        plen       = request.prefix_len

        # Availability comes from the periodic (100 ms) heartbeat snapshot,
        # not the instantaneous queue — the LB routes on a stale view.
        discovery.probe(nodes, current_time)
        local_avail = {n.node_id for n in nodes
                       if n.region_id == req_region
                       and discovery.is_avail(n.node_id)}

        if local_avail:
            cand   = discovery.local_match(req_region, ck, plen, local_avail)
            pool   = cand if cand else local_avail
            chosen = min(pool, key=lambda i: nodes[i].queue_ms(current_time))
        else:
            # All local replicas full → look at remote regions with capacity.
            remote_avail: Dict[int, set] = {}
            for n in nodes:
                if n.region_id != req_region and discovery.is_avail(n.node_id):
                    remote_avail.setdefault(n.region_id, set()).add(n.node_id)
            if remote_avail:
                rcand = discovery.snapshot_match(req_region, ck, plen,
                                                 set(remote_avail.keys()))
                region_pool = rcand if rcand else set(remote_avail.keys())
                tgt_region  = min(
                    region_pool,
                    key=lambda r: min(nodes[i].queue_ms(current_time)
                                      for i in remote_avail[r]))
                av     = remote_avail[tgt_region]
                ncand  = discovery.local_match(tgt_region, ck, plen, av)
                pool   = ncand if ncand else av
                chosen = min(pool, key=lambda i: nodes[i].queue_ms(current_time))
            else:
                # Nothing available anywhere → queue at the local LB.
                chosen = _nearest_least_loaded(nodes, req_region, current_time)

        # Selective push: the LB knows what it just dispatched — burn one of
        # the chosen replica's probed free-slot credits so the rest of this
        # heartbeat window doesn't pile onto it once its capacity is spoken for.
        discovery.consume(chosen)
        node     = nodes[chosen]
        ml, tier = node.cache_match_len(request.content_keys or {}, plen)
        discovery.record(req_region, chosen, node.region_id, ck, plen)
        return RouteResult(chosen, ml, node.region_id, tier)

    # ── Ludo: load-gated two-stage routing ───────────────────────────────────
    # Stage 1 (load gate): candidacy comes from load, not from the cache.
    #   The pool is every node that is NOT busy under the viewed load —
    #   requester's own region live, remote regions via the periodic load
    #   broadcast (EWMA queue > LUDO_LB_BUSY_THRESH_MS ⇒ busy).  A busy node
    #   is out even if it holds a deep reuse match.
    # Stage 2 (annotate): the latency-graded table lookup (already computed
    #   above as `candidates`) credits pool nodes that hold a match.
    # Stage 3 (score): joint T_q_view + T_p + T_t over the whole pool — an
    #   idle nearby node with no cache competes on equal terms.
    if scheme in ('ludo', 'ludo_coarse'):
        # ludo_coarse shares the entire routing pipeline — the two schemes
        # differ only in the discovery instance's same-region cut chain.
        req_region = request.region_id
        discovery.lb_probe(nodes, current_time)
        if MIG_ENABLE:
            discovery.mig_tick(nodes, current_time)
        cred: Dict[int, Tuple[int, str]] = {}
        for nid, ml, tier in candidates:
            if ml > cred.get(nid, (0, ''))[0]:
                cred[nid] = (ml, tier)

        q_views = [discovery.lb_view_queue(node, req_region, current_time)
                   for node in nodes]
        # Capacity pricing: λ rises with fleet saturation (idlest node's
        # viewed queue) so full prefills grow costly exactly when compute is
        # the binding resource — the argmin auto-reverts toward cache holders.
        lam = min(LUDO_LAMBDA_MAX, min(q_views) / LUDO_LAMBDA_SCALE_MS)

        best_g = best_u = None              # gated / ungated optima
        best_g_score = best_u_score = float('inf')
        for node, q_view in zip(nodes, q_views):
            ml, tier = cred.get(node.node_id, (0, 'MISS'))
            T_p = (request.prefix_len - ml) * C_PREFILL_MS_PER_TOKEN
            # A5 tier-blind: DRAM priced as VRAM in the SCORE only; the
            # returned tier stays truthful so serving pays the real reload.
            T_t = _transfer_ms('VRAM' if (ABL_TIER_BLIND and tier == 'DRAM')
                               else tier,
                               node.region_id, req_region, ml)
            if ABL_NO_LOADSCORE:
                # A1: pure affinity — no queue view, no capacity pricing.
                # Same-region full-miss nodes then score identically, so
                # break ties with a per-request deterministic pseudo-random
                # key (mirrors the testbed's random tie-break, reproducibly).
                score = (T_p + T_t
                         + (hash((request.request_id, node.node_id))
                            & 0xFFFF) * 1e-9)
            else:
                score = q_view + (1.0 + lam) * T_p + T_t
            if score < best_u_score:
                best_u_score = score
                best_u = (node.node_id, ml, node.region_id, tier)
            if q_view <= LUDO_LB_BUSY_THRESH_MS and score < best_g_score:
                best_g_score = score
                best_g = (node.node_id, ml, node.region_id, tier)
        # Pool-empty fallback: if the gate excluded everyone, degrade to the
        # ungated score minimum (soft load handling) instead of blind local
        # queueing — keeps any finite θ cliff-free.
        nid, ml, nr, tier = best_g if best_g is not None else best_u
        # Self-dispatch accounting: fold this send into the view so the rest
        # of the broadcast window doesn't herd onto the same remote node.
        svc_est = ((request.prefix_len - ml) * C_PREFILL_MS_PER_TOKEN
                   + request.prefix_len * C_DECODE_MS_PER_TOKEN)
        discovery.lb_note_dispatch(req_region, nid, svc_est)
        # Cross-region forward → the response will piggyback the target
        # region's load stamp back to the requester (data-path channel).
        if nr != req_region:
            q_v = discovery.lb_view_queue(nodes[nid], req_region, current_time)
            t_back = (current_time + q_v + svc_est
                      + 1.5 * WAN_LATENCY_MS[req_region][nr])
            discovery.lb_note_forward(req_region, nr, t_back)
        # Popular-KV migration: record locality misses; hot families get
        # replicated toward the demand (background, off the critical path).
        if MIG_ENABLE and request.content_keys:
            discovery.mig_note(req_region, request.prefix_id,
                               request.prefix_len,
                               request.content_keys, cred, ml,
                               nodes, current_time)
        return RouteResult(nid, ml, nr, tier)

    # ── All other schemes: joint scoring ─────────────────────────────────────
    # Always include same-region least-loaded node as a baseline candidate
    # (match_len=0, full prefill, zero WAN cost). This ensures a cross-region
    # cache hit is only chosen when it truly beats local recomputation.
    local_nid = _nearest_least_loaded(nodes, request.region_id, current_time)
    local_candidate = (local_nid, 0, 'MISS')
    all_candidates = list(candidates) + [local_candidate]

    best_score = float('inf')
    best = None
    for nid, ml, tier in all_candidates:
        node = nodes[nid]
        T_q = node.queue_ms(current_time)
        T_p = (request.prefix_len - ml) * C_PREFILL_MS_PER_TOKEN
        T_t = _transfer_ms(tier, node.region_id, request.region_id, ml)
        score = T_q + T_p + T_t
        if score < best_score:
            best_score = score
            best = (nid, ml, node.region_id, tier)

    nid, ml, nr, tier = best
    return RouteResult(nid, ml, nr, tier)


def _compute_oracle_routing(requests: List['Request'],
                             nodes: List[Node]) -> Dict[int, int]:
    """Pre-compute optimal prefix→node assignment for the offline oracle.

    The oracle has full knowledge of all future requests. We assign each
    prefix a permanent *home node* before simulation begins, using a
    greedy load-balanced strategy:

    1. For each prefix, compute expected total load = freq × prefix_len
       (proxy for total prefill tokens) and its most-common request region.
    2. Sort prefixes by total load descending (heaviest first).
    3. Assign each prefix to the least-loaded node in its most-common
       region; fall back to globally least-loaded if the region is
       uniformly overloaded.

    With this assignment every request for prefix X always goes to the
    same home node, giving the oracle maximum cache reuse.  Combined with
    Bélády's optimal eviction, this is the tightest achievable upper bound.
    """
    from collections import Counter, defaultdict

    freq: Counter = Counter(req.prefix_id for req in requests)
    plen: Dict[int, int] = {}
    # region preference: count how many requests per (pid, region)
    pid_region_cnt: Dict[int, Counter] = defaultdict(Counter)
    for req in requests:
        if req.prefix_id not in plen:
            plen[req.prefix_id] = req.prefix_len
        pid_region_cnt[req.prefix_id][req.region_id] += 1

    # Sort by total expected load descending
    sorted_pids = sorted(freq.keys(),
                         key=lambda pid: freq[pid] * plen.get(pid, 1),
                         reverse=True)

    # Group nodes by region for affinity-aware assignment
    region_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for n in nodes:
        region_to_nodes[n.region_id].append(n.node_id)

    node_load = [0.0] * len(nodes)
    routing: Dict[int, int] = {}

    for pid in sorted_pids:
        # Preferred region = most common request region for this prefix
        pref_region = pid_region_cnt[pid].most_common(1)[0][0]
        candidates = region_to_nodes.get(pref_region, [])
        if candidates:
            nid = min(candidates, key=lambda n: node_load[n])
        else:
            nid = int(np.argmin(node_load))
        routing[pid] = nid
        node_load[nid] += freq[pid] * plen.get(pid, 1)

    return routing


def _nearest_least_loaded(nodes: List[Node], req_region: int,
                           current_time: float) -> int:
    region_nodes = [n for n in nodes if n.region_id == req_region]
    pool = region_nodes if region_nodes else nodes

    # Evaluate queue_ms once per node (avoid double-call with M/D/c slots)
    loads = [(n.node_id, n.queue_ms(current_time)) for n in pool]
    min_q = min(q for _, q in loads)
    best_node_ids = [nid for nid, q in loads if q == min_q]
    return int(np.random.choice(best_node_ids))


# ─── Simulator ────────────────────────────────────────────────────────────────

class Simulator:
    """
    Trace-driven event simulation.
    Requests arrive as a Poisson process; nodes are modeled as M/D/c queues
    with c = MAX_BATCH_SIZE concurrent batch slots per node.
    """
    def __init__(self, nodes: List[Node], discovery: DiscoveryPlane,
                 workload_gen: WorkloadGenerator,
                 arrival_rate_rps: float,   # requests per second
                 scheme: str,
                 workload: str = 'mixed',
                 warmup: int = 3000,
                 seed: int = 0):
        self.nodes      = nodes
        self.discovery  = discovery
        self.wgen       = workload_gen
        self.rate_ms    = arrival_rate_rps / 1000.0   # convert to req/ms
        self.scheme     = scheme
        self.workload   = workload
        self.warmup     = warmup
        self.rng        = np.random.default_rng(seed)
        self.seed       = seed

    def run(self, num_requests: int) -> List[RequestRecord]:
        """
        Simulate warmup + num_requests requests.
        Returns records for the non-warmup portion only.

        Offline Oracle
        --------------
        When scheme == 'oracle', the entire request sequence is pre-generated
        before the main loop so that:
          1. Routing uses OracleDiscovery (perfect current cache state, same as
             before — already globally optimal for online routing).
          2. Cache eviction uses Bélády's optimal replacement: evict the entry
             whose *next* access in the pre-known future stream is furthest away,
             rather than the least-recently-used entry.  This is provably optimal
             and requires offline knowledge of the full request sequence.
        Both RNG states are saved and restored so the inter-arrival times and
        request content seen by the main loop are bit-for-bit identical to what
        they would be without pre-generation.
        """
        for node in self.nodes:
            node.reset()
        self.discovery.reset()

        _rrng = (np.random.default_rng(0xC0FFEE ^ (self.seed & 0xFFFF))
                 if _ROBUST_ACTIVE else None)
        _pending_adv = deque() if SIM_UPDATE_DELAY_MS > 0 else None

        total = self.warmup + num_requests

        # ── Offline oracle: pre-generate full request sequence ────────────────
        oracle_pre_reqs: Optional[List[Request]] = None
        oracle_pid_times: Optional[Dict[int, List[float]]] = None
        if self.scheme == 'oracle':
            # Save RNG states
            sim_state  = self.rng.bit_generator.state
            wgen_state = self.wgen.rng.bit_generator.state

            # Pre-generate all requests
            t = 0.0
            pre_reqs: List[Request] = []
            for i in range(total):
                inter = self.rng.exponential(1.0 / self.rate_ms)
                t += inter
                pre_reqs.append(self.wgen.sample(i, t, self.workload))

            # Build pid_times: prefix_id → sorted list of arrival times
            pid_times_build: Dict[int, List[float]] = defaultdict(list)
            for req in pre_reqs:
                pid_times_build[req.prefix_id].append(req.arrival_time)
            oracle_pid_times = dict(pid_times_build)   # lists already sorted (time-ordered)

            oracle_pre_reqs = pre_reqs

            # Restore RNG states so main loop produces the identical sequence
            self.rng.bit_generator.state  = sim_state
            self.wgen.rng.bit_generator.state = wgen_state

        # Compute optimal prefix→node assignment from the pre-generated sequence
        oracle_routing: Optional[Dict[int, int]] = None
        if oracle_pre_reqs is not None:
            oracle_routing = _compute_oracle_routing(oracle_pre_reqs, self.nodes)

        records: List[RequestRecord] = []
        current_time = 0.0

        for idx in range(total):
            inter = self.rng.exponential(1.0 / self.rate_ms)
            current_time += inter

            if _pending_adv is not None:
                while _pending_adv and _pending_adv[0][0] <= current_time:
                    _pending_adv.popleft()[1]()

            if oracle_pre_reqs is not None:
                req = oracle_pre_reqs[idx]    # use pre-generated request directly
            else:
                req = self.wgen.sample(idx, current_time, self.workload)

            result = route(req, self.discovery, self.nodes,
                           current_time, self.scheme,
                           oracle_routing=oracle_routing)
            node = self.nodes[result.node_id]

            match_len    = min(result.match_len, req.prefix_len)
            residual     = req.prefix_len - match_len
            ttft_ms      = node.schedule(residual, req.prefix_len, current_time)
            # Add actual WAN / DRAM-reload transfer cost to TTFT.
            # Consistent Hash ignores geography in routing, so it pays the full
            # WAN penalty here; Ludo already avoids this via joint scoring.
            _wj = ((1.0 + _rrng.uniform(-SIM_WAN_JITTER, SIM_WAN_JITTER))
                   if SIM_WAN_JITTER > 0 else 1.0)
            ttft_ms += _transfer_ms(result.tier, result.node_region,
                                    req.region_id, match_len, wan_jitter=_wj)
            # Chord: add O(log N) finger-table lookup overhead.
            ttft_ms += result.lookup_overhead_ms

            # Content-addressed cache insertion.  The Node now stores per-level
            # content-hash entries; `fully_evicted` and `demoted_to_dram` are
            # lists of (level, content_hash) tuples which we forward to the
            # discovery plane's per-entry invalidate/advertise APIs.
            ck = req.content_keys or None
            if oracle_pid_times is not None:
                fully_evicted, demoted_to_dram = node.insert_beladys(
                    ck or {}, req.prefix_len, current_time, oracle_pid_times)
            else:
                fully_evicted, demoted_to_dram = node.insert(
                    ck or {}, req.prefix_len, current_time)

            # Advertise the newly inserted (or re-promoted) VRAM entry at every
            # applicable level; the discovery plane uses req.content_keys to
            # derive per-level keys so two requests sharing tokens collide.
            if not _ROBUST_ACTIVE:
                self.discovery.advertise(
                    result.node_id, node.region_id,
                    req.prefix_id, req.prefix_len, tier='VRAM',
                    content_keys=ck,
                    **({'radix_path': req.radix_path or None}
                       if isinstance(self.discovery, RadixTreeDiscovery) else {}),
                )
            else:
                _rp = ({'radix_path': req.radix_path or None}
                       if isinstance(self.discovery, RadixTreeDiscovery) else {})
                def _do_adv(rid=result.node_id, reg=node.region_id,
                            pid=req.prefix_id, pl=req.prefix_len,
                            cks=ck, rp=_rp):
                    self.discovery.advertise(rid, reg, pid, pl, tier='VRAM',
                                             content_keys=cks, **rp)
                if SIM_UPDATE_LOSS > 0 and _rrng.random() < SIM_UPDATE_LOSS:
                    pass
                elif _pending_adv is not None:
                    _pending_adv.append(
                        (current_time + SIM_UPDATE_DELAY_MS, _do_adv))
                else:
                    _do_adv()
            # Re-advertise entries demoted from VRAM → DRAM so the discovery
            # plane knows their tier changed (callers can penalise DRAM hits).
            # For content-addressed caches we advertise the specific (level,
            # content_hash) pairs via invalidate_content-then-readvertise; the
            # advertise call replays the full level set with VRAM, so for
            # DRAM entries we directly manipulate per-level dicts through
            # invalidate_content + an explicit DRAM-tier advertise on the
            # per-level Ludo/Hash dict (RadixTree has no per-level tier so
            # DRAM hits degrade silently to a regular hit).
            for lvl, key in demoted_to_dram:
                # Reflect the tier change in the discovery plane: overwrite the
                # existing level-l entry for this content_key with tier='DRAM'.
                # We reuse the advertise API by constructing a minimal fake
                # content_keys dict carrying only the affected level — the
                # discovery implementations read keys by chunk_size, so the
                # dict key is CHUNK_SIZES[lvl].
                self.discovery.advertise(
                    result.node_id, node.region_id,
                    req.prefix_id, CHUNK_SIZES[lvl], tier='DRAM',
                    content_keys={CHUNK_SIZES[lvl]: key},
                )
            # Entries fully evicted from both tiers: invalidate per level.
            for lvl, key in fully_evicted:
                self.discovery.invalidate_content(result.node_id, lvl, key)

            if idx >= self.warmup:
                # Return-path latency: last generated token travels one-way
                # from serving node back to the user.  Token payloads are tiny
                # (a few bytes each), so propagation delay dominates.
                # Same-region: 0 ms.  Cross-region: half the symmetric RTT.
                node_region = self.nodes[result.node_id].region_id
                if node_region == req.region_id:
                    return_latency_ms = 0.0
                else:
                    return_latency_ms = WAN_LATENCY_MS[node_region][req.region_id] / 2.0
                e2e_ms = (ttft_ms
                          + req.gen_len * C_DECODE_STEP_MS_PER_TOKEN
                          + return_latency_ms)
                records.append(RequestRecord(
                    request_id       = idx,
                    prefix_id        = req.prefix_id,
                    prefix_len       = req.prefix_len,
                    arrival_time     = current_time,
                    ttft_ms          = ttft_ms,
                    e2e_latency_ms   = e2e_ms,
                    match_len        = match_len,
                    cache_hit        = match_len > 0,
                    flops_saved_frac = match_len / req.prefix_len,
                    node_id          = result.node_id,
                    slo_met          = ttft_ms <= TTFT_SLO_MS,
                    hit_tier         = result.tier if match_len > 0 else 'MISS',
                    gen_len          = req.gen_len,
                ))

        return records

    def run_list(self, requests: List[Request]) -> List[RequestRecord]:
        """Simulate a pre-built, ordered request list with no warmup.

        Each request is processed exactly once in list order using its own
        pre-assigned arrival_time.  Intended for single-pass evaluations
        (e.g. Figure 3 FLOPs saved) where every dataset entry is sent once
        rather than being Zipf-resampled for N_REQUESTS rounds.

        Oracle scheme: pid_times is built from the provided list so that
        Bélády's eviction has full lookahead over the one-pass sequence.
        """
        for node in self.nodes:
            node.reset()
        self.discovery.reset()

        # Oracle: build pid_times (Bélády eviction) + oracle_routing (optimal home nodes)
        oracle_pid_times: Optional[Dict[int, List[float]]] = None
        oracle_routing: Optional[Dict[int, int]] = None
        if self.scheme == 'oracle':
            pid_times_build: Dict[int, List[float]] = defaultdict(list)
            for req in requests:
                pid_times_build[req.prefix_id].append(req.arrival_time)
            oracle_pid_times = dict(pid_times_build)
            oracle_routing = _compute_oracle_routing(requests, self.nodes)

        records: List[RequestRecord] = []
        for idx, req in enumerate(requests):
            current_time = req.arrival_time
            result   = route(req, self.discovery, self.nodes,
                             current_time, self.scheme,
                             oracle_routing=oracle_routing)
            node     = self.nodes[result.node_id]
            match_len = min(result.match_len, req.prefix_len)
            residual  = req.prefix_len - match_len
            ttft_ms   = node.schedule(residual, req.prefix_len, current_time)
            ttft_ms  += _transfer_ms(result.tier, result.node_region,
                                     req.region_id, match_len)
            ttft_ms  += result.lookup_overhead_ms

            ck = req.content_keys or None
            if oracle_pid_times is not None:
                fully_evicted, demoted_to_dram = node.insert_beladys(
                    ck or {}, req.prefix_len, current_time, oracle_pid_times)
            else:
                fully_evicted, demoted_to_dram = node.insert(
                    ck or {}, req.prefix_len, current_time)

            self.discovery.advertise(
                result.node_id, node.region_id,
                req.prefix_id, req.prefix_len, tier='VRAM',
                content_keys=ck,
                **({'radix_path': req.radix_path or None}
                   if isinstance(self.discovery, RadixTreeDiscovery) else {}),
            )
            for lvl, key in demoted_to_dram:
                self.discovery.advertise(
                    result.node_id, node.region_id,
                    req.prefix_id, CHUNK_SIZES[lvl], tier='DRAM',
                    content_keys={CHUNK_SIZES[lvl]: key},
                )
            for lvl, key in fully_evicted:
                self.discovery.invalidate_content(result.node_id, lvl, key)

            node_region = self.nodes[result.node_id].region_id
            return_latency_ms = (0.0 if node_region == req.region_id
                                 else WAN_LATENCY_MS[node_region][req.region_id] / 2.0)
            e2e_ms = (ttft_ms
                      + req.gen_len * C_DECODE_STEP_MS_PER_TOKEN
                      + return_latency_ms)
            records.append(RequestRecord(
                request_id       = idx,
                prefix_id        = req.prefix_id,
                prefix_len       = req.prefix_len,
                arrival_time     = current_time,
                ttft_ms          = ttft_ms,
                e2e_latency_ms   = e2e_ms,
                match_len        = match_len,
                cache_hit        = match_len > 0,
                flops_saved_frac = match_len / req.prefix_len,
                node_id          = result.node_id,
                slo_met          = ttft_ms <= TTFT_SLO_MS,
                hit_tier         = result.tier if match_len > 0 else 'MISS',
            gen_len          = req.gen_len,
            ))
        return records

    def run_closed_loop(self, n_clients: int, duration_ms: float,
                        warmup_ms: float = 0.0) -> Dict[str, float]:
        """SkyWalker-style closed-loop service-throughput measurement.

        Mirrors the SkyWalker paper's evaluation (§5.1, Figure 8): a fixed
        population of clients, allocated to regions in proportion to the
        request-demand distribution, each keeps exactly one request in
        flight — the next request is issued the moment the previous one
        finishes end-to-end (TTFT + full decode + return leg).  Load is
        therefore concurrency-driven (replicas at sustained utilisation),
        not arrival-rate-driven, and *service throughput* is the total
        tokens completed per wall-clock second inside the measurement
        window:

            Tpt = Σ_completed (prefix_len + gen_len) / window_seconds

        A request's tokens are attributed at its completion time; requests
        completing before `warmup_ms` (cache warm-up) or after
        `duration_ms` (censored) are excluded from the window.  Output-only
        throughput and request rate are reported alongside.
        """
        import heapq

        for node in self.nodes:
            node.reset()
        self.discovery.reset()

        # Clients per region ∝ demand weights (largest-remainder rounding),
        # matching the paper's per-region client deployments.
        quota = REGION_REQUEST_WEIGHTS * n_clients
        base  = np.floor(quota).astype(int)
        order = np.argsort(-(quota - base))
        for k in range(int(n_clients - base.sum())):
            base[order[k]] += 1
        client_region = [r for r in range(NUM_REGIONS)
                         for _ in range(int(base[r]))]

        # Central Directory: per-request directory RTT + coordinator M/D/1
        # queue, mirroring _run_central_directory in run_section3.py.
        is_cd    = (self.scheme == 'central_directory')
        dir_free = 0.0

        # Stagger initial issues over 100 ms so t=0 isn't one giant burst.
        heap = [(float(self.rng.uniform(0.0, 100.0)), cid)
                for cid in range(n_clients)]
        heapq.heapify(heap)

        in_tok = out_tok = n_done = 0
        ttfts: List[float] = []
        idx = 0
        while heap:
            t, cid = heapq.heappop(heap)
            if t >= duration_ms:
                continue                       # client retires at the horizon
            req = self.wgen.sample(idx, t, self.workload)
            req.region_id    = client_region[cid]   # client pinned to region
            req.arrival_time = t
            idx += 1

            route_scheme = 'centralized' if is_cd else self.scheme
            result = route(req, self.discovery, self.nodes, t, route_scheme)
            node   = self.nodes[result.node_id]

            match_len = min(result.match_len, req.prefix_len)
            residual  = req.prefix_len - match_len
            ttft_ms   = node.schedule(residual, req.prefix_len, t)
            ttft_ms  += _transfer_ms(result.tier, result.node_region,
                                     req.region_id, match_len)
            ttft_ms  += result.lookup_overhead_ms
            if is_cd:
                start    = max(dir_free, t)
                dir_q    = start - t
                dir_free = start + DIRECTORY_SERVICE_MS
                ttft_ms += (2.0 * float(WAN_LATENCY_MS[DIRECTORY_REGION]
                                                      [req.region_id])
                            + dir_q + DIRECTORY_SERVICE_MS)

            ck = req.content_keys or None
            fully_evicted, demoted_to_dram = node.insert(
                ck or {}, req.prefix_len, t)
            self.discovery.advertise(
                result.node_id, node.region_id,
                req.prefix_id, req.prefix_len, tier='VRAM',
                content_keys=ck,
                **({'radix_path': req.radix_path or None}
                   if isinstance(self.discovery, RadixTreeDiscovery) else {}),
            )
            for lvl, key in demoted_to_dram:
                self.discovery.advertise(
                    result.node_id, node.region_id,
                    req.prefix_id, CHUNK_SIZES[lvl], tier='DRAM',
                    content_keys={CHUNK_SIZES[lvl]: key},
                )
            for lvl, key in fully_evicted:
                self.discovery.invalidate_content(result.node_id, lvl, key)

            node_region = node.region_id
            ret_ms = (0.0 if node_region == req.region_id
                      else WAN_LATENCY_MS[node_region][req.region_id] / 2.0)
            done = t + ttft_ms + req.gen_len * C_DECODE_STEP_MS_PER_TOKEN + ret_ms

            if warmup_ms < done <= duration_ms:
                in_tok  += req.prefix_len
                out_tok += req.gen_len
                n_done  += 1
                ttfts.append(ttft_ms)

            heapq.heappush(heap, (done, cid))

        window_s = (duration_ms - warmup_ms) / 1000.0
        ttfts_a  = np.array(ttfts) if ttfts else np.array([0.0])
        return {
            'tpt_total_tok_s': (in_tok + out_tok) / window_s,
            'tpt_out_tok_s':   out_tok / window_s,
            'req_s':           n_done / window_s,
            'n_done':          float(n_done),
            'n_clients':       float(n_clients),
            'duration_s':      duration_ms / 1000.0,
            'warmup_s':        warmup_ms / 1000.0,
            'ttft_p50_ms':     float(np.percentile(ttfts_a, 50)),
            'ttft_p99_ms':     float(np.percentile(ttfts_a, 99)),
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_nodes(vram_tokens: int = NODE_VRAM_TOKENS,
               dram_tokens: int = NODE_DRAM_TOKENS) -> List[Node]:
    return [Node(i, NODE_REGION[i],
                 vram_tokens=vram_tokens, dram_tokens=dram_tokens)
            for i in range(NUM_NODES)]


def make_discovery(scheme: str) -> DiscoveryPlane:
    return {
        'ludo':            LudoDiscovery,
        'ludo_coarse':     lambda: LudoDiscovery(
                               pair_cuts=LUDO_PAIR_CUTS_COARSE,
                               region_cuts=LUDO_REGION_CUTS_COARSE),
        'random_ll':       RandomLLDiscovery,
        'consistent_hash': ConsistentHashDiscovery,
        'kademlia':        KademliaDHTDiscovery,
        'chord':           ChordDHTDiscovery,
        'skywalker':       SkyWalkerDiscovery,
        'dynamo':          DynamoDiscovery,
        'centralized':     CentralizedTrackerDiscovery,
        'radix_tree':      RadixTreeDiscovery,
        'oracle':          OracleDiscovery,
    }[scheme]()


SCHEME_LABELS = {
    'ludo':            'Ludo (ours)',
    'ludo_coarse':     'KVCompass-CG',
    'random_ll':       'Random / LL',
    'consistent_hash': 'Consistent Hash',
    'kademlia':        'Kademlia DHT',
    'chord':           'Chord DHT',
    'skywalker':       'SkyWalker',
    'dynamo':          'Dynamo',
    'centralized':     'Centralized Tracker',
    'radix_tree':      'PlanetServe Radix Tree',
    'oracle':          'Oracle',
}

SCHEME_COLORS = {
    'ludo':            '#2166ac',   # blue
    'ludo_coarse':     '#74add1',   # light blue (pairs with ludo's deep blue)
    'random_ll':       '#d73027',   # red
    'consistent_hash': '#1a9850',   # green
    'kademlia':        '#f46d43',   # orange
    'chord':           '#e377c2',   # pink  (Chord DHT — same family as Kademlia)
    'skywalker':       '#33a02c',   # green  (SkyWalker cross-region LB)
    'dynamo':          '#fb9a99',   # salmon (Dynamo hierarchical KV router)
    'centralized':     '#762a83',   # purple
    'radix_tree':      '#8c510a',   # brown
    'oracle':          '#762a83',   # purple (replaces centralized tracker)
}
