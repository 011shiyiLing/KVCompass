"""
Global configuration for the Ludo-SGLang real-machine framework.

Most settings can be overridden via environment variables. The constants
that govern routing-scheme behaviour (DHT lookup overhead, central
directory RTT/throughput) are derived from a CloudPing.co WAN-latency
snapshot — see ``ludo_sglang/wan_matrix_p*.py`` for the snapshot data
and provenance, and ``scripts/build_wan_matrix.py`` for the regenerator.
"""
import importlib
import os

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# 1. Routing scheme selector
# ════════════════════════════════════════════════════════════════════════════
# Picks which routing strategy the FastAPI router runs. Each request is
# dispatched through the matching ``ludo_sglang.router.schemes.*`` module.
# Used to switch between baselines without restarting the SGLang fleet.
SCHEMES_AVAILABLE = (
    "ludo",
    "radix_tree",
    "consistent_hash",
    "kademlia",
    "chord",
    "central_directory",
    "skywalker",
    "dynamo",
)
SCHEME: str = os.environ.get("SCHEME", "ludo").strip().lower()
if SCHEME not in SCHEMES_AVAILABLE:
    raise ValueError(
        f"SCHEME={SCHEME!r} not recognised; pick one of {SCHEMES_AVAILABLE}"
    )


# ════════════════════════════════════════════════════════════════════════════
# 2. Ludo / Radix routing-table sizing
# ════════════════════════════════════════════════════════════════════════════
def _parse_chunk_sizes(env_val: str) -> list[int]:
    return [int(x.strip()) for x in env_val.split(",") if x.strip()]


# Multi-level Ludo: hash the first L tokens at each L in CHUNK_SIZES.
LUDO_CHUNK_SIZES: list[int] = _parse_chunk_sizes(
    os.environ.get(
        "CHUNK_SIZES",
        os.environ.get("CHUNK_SIZE", "256, 512, 1024, 2048, 4096, 8192, 16384, 24576"),
    )
)
LUDO_INITIAL_CAPACITY: int = 65536
LUDO_MAX_ENTRIES: int = 65536

# Baselines (ConsistentHash, CentralDirectory, Kademlia, Chord) use the §3
# simulation's chunk-size schedule [128..4096]. Keeping baselines at this
# coarser, published configuration — while letting Ludo's explicit-write
# multi-level table reach 24K — is the design difference Ludo claims:
# Ludo can afford fine-grained advertising up to the model context length;
# blind-hash schemes don't, because they pay 1/N coverage at every level.
BASELINE_CHUNK_SIZES: list[int] = _parse_chunk_sizes(
    os.environ.get(
        "BASELINE_CHUNK_SIZES",
        "128, 256, 512, 1024, 2048, 4096",
    )
)

# PlanetServe-style RadixTree gate (only used when SCHEME=radix_tree).
RADIX_GATE_CHUNKS: int = int(os.environ.get("RADIX_GATE_CHUNKS", "10"))
RADIX_CHUNK_TOKENS: int = int(os.environ.get("RADIX_CHUNK_TOKENS", "10"))


# ════════════════════════════════════════════════════════════════════════════
# 3. SGLang backend fleet
# ════════════════════════════════════════════════════════════════════════════
# The fleet is a list of independent SGLang servers, one per worker, each
# addressed by its own URL. No SGLang --dp-size / data_parallel_rank.
# Same router code works in both deployments:
#   - 2×A6000 smoke test : 2 processes, 1 per GPU, distinct ports
#   - 8×A100 production  : 8 processes, 1 per GPU, distinct ports
# Set via SGLANG_WORKER_URLS="http://host:30000,http://host:30001,...".
# If unset, defaults to N_PREFILL_WORKERS workers on 127.0.0.1 starting
# at SGLANG_BASE_PORT — matches what scripts/sglang_fleet.sh produces.
_SGLANG_BASE_PORT: int = int(os.environ.get("SGLANG_BASE_PORT", "30000"))
_N_PREFILL_WORKERS_DEFAULT: int = int(os.environ.get("N_PREFILL_WORKERS", "8"))


def _parse_worker_urls(env_val: str | None) -> list[str]:
    if env_val:
        return [u.strip() for u in env_val.split(",") if u.strip()]
    return [
        f"http://127.0.0.1:{_SGLANG_BASE_PORT + i}"
        for i in range(_N_PREFILL_WORKERS_DEFAULT)
    ]


SGLANG_WORKER_URLS: list[str] = _parse_worker_urls(os.environ.get("SGLANG_WORKER_URLS"))
N_DP_WORKERS: int = len(SGLANG_WORKER_URLS)


# ════════════════════════════════════════════════════════════════════════════
# 4. Model / sampling
# ════════════════════════════════════════════════════════════════════════════
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
MAX_NEW_TOKENS: int = int(os.environ.get("MAX_NEW_TOKENS", "128"))

# ── Prefill cost model ─────────────────────────────────────────────────
# Joint scoring estimates each worker's TTFT in milliseconds; the scoring
# math needs a per-token prefill rate. We start from a prior calibrated
# to ~7-8B BF16 dense models on A100 (≈ 0.05 ms/token, matches §3 sim;
# Qwen2.5-7B / Llama-3.1-8B sit within ±20% of this) and update
# per-worker rates online via EMA over observed TTFT minus the
# router's queue-wait estimate at dispatch. Until each worker has
# accumulated PREFILL_RATE_MIN_SAMPLES observations, the prior is used
# directly (avoids feedback from one or two outlier requests).
C_PREFILL_MS_PER_TOKEN_PRIOR: float = float(os.environ.get(
    "C_PREFILL_MS_PER_TOKEN_PRIOR", "0.05"))
PREFILL_RATE_EMA_ALPHA: float = float(os.environ.get(
    "PREFILL_RATE_EMA_ALPHA", "0.2"))
PREFILL_RATE_MIN_SAMPLES: int = int(os.environ.get(
    "PREFILL_RATE_MIN_SAMPLES", "5"))

# ── PCIe reload cost (DRAM-tier hit) ─────────────────────────────────
# When a Ludo entry's holder is at TIER_DRAM, the worker has the chunk
# in CPU RAM and must DMA it to GPU HBM before attention can run.  The
# joint scoring formula adds T_t_reload = match_len × this constant for
# DRAM hits, so the router can compare a same-region VRAM hit against a
# same-region DRAM hit on the same time axis.
#
# Calibration (default = 0.0052 ms/token):
#   per-token KV bytes (Qwen2.5-7B BF16, GQA 4:1)
#     = 2 (K,V) × 28 layers × 4 KV heads × 128 head_dim × 2 bytes BF16
#     = 57,344 B/token
#   PCIe Gen4 x16 sustained ~25 GB/s
#     = 57,344 / 25e9 ≈ 2.3 μs/token theoretical
#   Empirically we observe ~5 μs/token because of pinned-host-memory
#   overhead, kernel-launch dispatch, and partial overlap with attn-1.
#   For Llama-3.1-8B (1.5× more KV) the empirical value is ~5.2 μs/tok.
# Override via env if running on a different model / accelerator:
PCIE_RELOAD_MS_PER_TOKEN: float = float(os.environ.get(
    "PCIE_RELOAD_MS_PER_TOKEN", "0.0052"))


# ════════════════════════════════════════════════════════════════════════════
# 5. Router server
# ════════════════════════════════════════════════════════════════════════════
ROUTER_HOST: str = os.environ.get("ROUTER_HOST", "127.0.0.1")
ROUTER_PORT: int = int(os.environ.get("ROUTER_PORT", "8001"))
HTTP_TIMEOUT: float = float(os.environ.get("HTTP_TIMEOUT", "120.0"))

# ── Multi-edge gossip (Option B replication) ───────────────────────────
# Each router process is one "edge node" in the design-doc sense; each
# region runs one.  Set ROUTER_ID to a unique non-negative integer per
# router so LWW tie-breaking on equal-epoch concurrent writes is
# deterministic.  Set PEER_ROUTER_URLS to the comma-separated list of
# peer routers' base URLs; gossip is push-based to every entry.
#
# When PEER_ROUTER_URLS is empty the gossip layer is disabled — the
# router runs as a single edge with no replication overhead, identical
# to pre-Option-B behaviour.
ROUTER_ID: int = int(os.environ.get("ROUTER_ID", "0"))


def _parse_peer_urls(env_val: str | None) -> list[str]:
    if not env_val:
        return []
    return [u.strip().rstrip("/") for u in env_val.split(",") if u.strip()]


PEER_ROUTER_URLS: list[str] = _parse_peer_urls(
    os.environ.get("PEER_ROUTER_URLS"),
)
GOSSIP_PERIOD_S: float = float(os.environ.get("GOSSIP_PERIOD_S", "1.0"))
GOSSIP_HTTP_TIMEOUT_S: float = float(
    os.environ.get("GOSSIP_HTTP_TIMEOUT_S", "1.0"),
)
GOSSIP_MAX_BATCH: int = int(os.environ.get("GOSSIP_MAX_BATCH", "1000"))


# ── Maintenance mode (gossip vs central directory) ─────────────────────
# 'gossip'  : every router is a full replica + LWW-merged gossip layer.
#             This is the legacy multi-edge mode; PEER_ROUTER_URLS gates
#             the gossip layer within that mode.
# 'central' : Smash-style central directory.  One (or a few sharded)
#             discovery authority(ies) own the authoritative LudoTable; the
#             routers run as read-only Lookup nodes that receive pushes
#             from the authority.  Workers fire advertise + tier events
#             directly to their authority (sharded by hash(prefix_key) %
#             N_SHARDS).
MAINTENANCE_MODE: str = os.environ.get(
    "MAINTENANCE_MODE", "gossip",
).strip().lower()
if MAINTENANCE_MODE not in ("gossip", "central"):
    raise ValueError(
        f"MAINTENANCE_MODE={MAINTENANCE_MODE!r} not recognised; "
        "expected 'gossip' or 'central'."
    )

# ── Central-directory topology (only used when MAINTENANCE_MODE=central) ──
# Single registry that everyone (authority / lookup / worker) registers with
# at startup; also runs the worker heartbeat watcher.
NAMESERVER_URL: str = os.environ.get(
    "NAMESERVER_URL", "http://127.0.0.1:9999",
).rstrip("/")

# Comma-separated list of authority base URLs.  The shard a prefix belongs
# to is determined by hash(prefix_key) % len(AUTHORITY_URLS).  For v1 a
# single authority is sufficient (set AUTHORITY_URLS to one URL).
AUTHORITY_URLS: list[str] = _parse_peer_urls(os.environ.get("AUTHORITY_URLS"))
N_SHARDS: int = max(1, len(AUTHORITY_URLS))

# Set on each authority process; lets the authority decide whether an incoming
# (key, ...) tuple actually belongs to its shard.  Ignored on lookups /
# workers / name server.
SHARD_ID: int = int(os.environ.get("SHARD_ID", "0"))

# Set on each lookup (edge router) process so the authority knows which
# lookup to push updates to and so the name server can broadcast
# topology changes coherently.
LOOKUP_ID: int = int(os.environ.get("LOOKUP_ID", "0"))

# How the name-server partitions lookups across shards.
#   'full'      : every edge mirrors every shard.  Each authority broadcasts
#                 to every lookup, so an edge router can answer queries
#                 for chunks landing in any shard locally — required when
#                 the edge IS the request-routing gateway (our case),
#                 because a single inference request hits L chunk levels
#                 whose hashes can land in different shards.
#   'partition' : Smash-style client-side sharding — lookup i mirrors
#                 shard ``i*nShards/nLookups``.  Smaller per-edge memory
#                 + bandwidth, but assumes the client routes by shard
#                 (which our gateway architecture doesn't do).  Kept as
#                 an opt-in for ablation experiments.
LOOKUP_SHARDING_MODE: str = os.environ.get(
    "LOOKUP_SHARDING_MODE", "full",
).strip().lower()
if LOOKUP_SHARDING_MODE not in ("full", "partition"):
    raise ValueError(
        f"LOOKUP_SHARDING_MODE={LOOKUP_SHARDING_MODE!r} not recognised; "
        "expected 'full' or 'partition'."
    )

# Worker heartbeat: workers POST /heartbeat every HEARTBEAT_INTERVAL_S;
# name server's daemon checks every HEARTBEAT_CHECK_INTERVAL_S, declares
# a worker dead after HEARTBEAT_TIMEOUT_S of silence and broadcasts
# Leave to all authorities (authority then clears that worker's slot in every
# entry it owns).
HEARTBEAT_INTERVAL_S: float = float(
    os.environ.get("HEARTBEAT_INTERVAL_S", "5.0"),
)
HEARTBEAT_CHECK_INTERVAL_S: float = float(
    os.environ.get("HEARTBEAT_CHECK_INTERVAL_S", "10.0"),
)
HEARTBEAT_TIMEOUT_S: float = float(
    os.environ.get("HEARTBEAT_TIMEOUT_S", "30.0"),
)

# HTTP timeout for authority <-> lookup pushes and worker -> authority events.
MAINTENANCE_HTTP_TIMEOUT_S: float = float(
    os.environ.get("MAINTENANCE_HTTP_TIMEOUT_S", "2.0"),
)

# Authority-side log/snapshot directory (v2; v1 keeps it for forward compat).
AUTHORITY_LOG_DIR: str = os.environ.get("AUTHORITY_LOG_DIR", "./back")


# ════════════════════════════════════════════════════════════════════════════
# 6. WAN topology — CloudPing.co snapshot
# ════════════════════════════════════════════════════════════════════════════
# Inter-region one-way propagation delay (ms) between 8 AWS regions,
# measured by CloudPing.co (TCP RTT / 2, then symmetrised). The exact
# snapshot file (and percentile / timeframe) is selected by the
# WAN_PERCENTILE env var; available choices are p50 / p90 / p99 — all
# generated by scripts/build_wan_matrix.py from the same dated snapshot.
WAN_PERCENTILE: str = os.environ.get("WAN_PERCENTILE", "p50").strip().lower()
if WAN_PERCENTILE not in ("p50", "p90", "p99"):
    raise ValueError(
        f"WAN_PERCENTILE={WAN_PERCENTILE!r} not recognised; "
        "pick one of p50, p90, p99 (re-run scripts/build_wan_matrix.py for others)"
    )

# ── WAN mode: emulated (single-host) vs real (multi-region deployment) ──
# WAN_EMULATION=1 (default, legacy 8×A100-single-host behaviour): the
# router injects `asyncio.sleep(wan_ms)` before dispatch to emulate the
# origin→worker propagation delay taken from the matrix below.
# WAN_EMULATION=0 (real multi-region deployment): the network itself
# provides the propagation delay — the router lives in one region and
# dispatches to workers over real WAN links, so injecting the matrix
# value again would double-count it.  The matrix is still loaded because
# the *scoring* math (joint score's T_t term, Ludo's latency-derived
# level cuts, Dynamo's fan-out floor) needs the predicted WAN cost.
# For real deployments point WAN_MATRIX_FILE at a measured matrix (see
# scripts/aws/measure_wan_matrix.py) so predictions match the actual
# links instead of the CloudPing 8-region snapshot.
WAN_EMULATION: bool = os.environ.get("WAN_EMULATION", "1").strip() not in (
    "0", "false", "no", "off",
)

# Optional measured-matrix override.  JSON layout:
#   {"regions": ["us-east-1", "eu-central-1", "ap-northeast-1"],
#    "one_way_ms": [[0.0, 45.1, 82.3], [45.1, 0.0, 121.7], [82.3, 121.7, 0.0]]}
WAN_MATRIX_FILE: str = os.environ.get("WAN_MATRIX_FILE", "").strip()
if WAN_MATRIX_FILE:
    import json as _json
    with open(WAN_MATRIX_FILE) as _f:
        _wan_json = _json.load(_f)
    WAN_LATENCY_MS: np.ndarray = np.asarray(_wan_json["one_way_ms"], dtype=float)
    DEFAULT_REGIONS: list[str] = list(_wan_json["regions"])
    if WAN_LATENCY_MS.shape != (len(DEFAULT_REGIONS), len(DEFAULT_REGIONS)):
        raise ValueError(
            f"WAN_MATRIX_FILE={WAN_MATRIX_FILE}: one_way_ms shape "
            f"{WAN_LATENCY_MS.shape} does not match {len(DEFAULT_REGIONS)} regions")
else:
    _wan_module = importlib.import_module(f"ludo_sglang.wan_matrix_{WAN_PERCENTILE}")
    WAN_LATENCY_MS = _wan_module.WAN_LATENCY_MS
    DEFAULT_REGIONS = _wan_module.DEFAULT_REGIONS
NUM_REGIONS_LOGICAL: int = WAN_LATENCY_MS.shape[0]

# The region THIS router process runs in (real multi-region mode runs one
# router per region; requests that arrive without an origin_region are
# assumed to originate here, matching "client sends to nearest LB").
ROUTER_REGION: int = int(os.environ.get("ROUTER_REGION", "0"))

# region index -> peer router base URL, for the schemes that talk
# router-to-router in real mode (SkyWalker two-layer forwarding, Dynamo
# cross-cluster lookup fan-out).  Format: "0:http://a:8001,1:http://b:8001".
# The entry for ROUTER_REGION (if present) is ignored.
def _parse_region_routers(env_val: str | None) -> dict[int, str]:
    out: dict[int, str] = {}
    if not env_val:
        return out
    for part in env_val.split(","):
        part = part.strip()
        if not part:
            continue
        idx, url = part.split(":", 1)
        out[int(idx)] = url.strip().rstrip("/")
    return out


REGION_ROUTER_URLS: dict[int, str] = _parse_region_routers(
    os.environ.get("REGION_ROUTER_URLS"),
)

# Mean off-diagonal one-way latency — the per-hop / per-RTT cost used in
# the DHT overhead derivations below.
_off_diag_mask = ~np.eye(NUM_REGIONS_LOGICAL, dtype=bool)
_off_diag_samples = WAN_LATENCY_MS[_off_diag_mask]
WAN_AVG_MS: float = float(_off_diag_samples.mean())
WAN_MAX_MS: float = float(WAN_LATENCY_MS.max())


# ════════════════════════════════════════════════════════════════════════════
# 7. Cluster topology assumed by overhead models
# ════════════════════════════════════════════════════════════════════════════
# DHT and Central-Directory baselines pay overhead determined by the
# *intended deployment topology* (planet-scale node count + WAN matrix),
# NOT by the physical testbed size. We charge the same planet-scale
# overhead the §3 simulation charges — parameterised on a 64-node
# 8-region topology — so real-machine and simulation results stay
# directly comparable even when the testbed has only 2 or 8 GPUs.
NUM_NODES_LOGICAL: int = int(os.environ.get("NUM_NODES_LOGICAL", "64"))


# ── Per-physical-worker region assignment ──────────────────────────────
# Each SGLang worker is mapped to one of the WAN regions so we can charge
# the geographic propagation cost of routing a request from its origin
# region to the worker's region. Without this every scheme runs
# WAN-free on localhost — the exact failure mode that lets blind hash
# placement (consistent_hash, DHT) look unrealistically competitive.
#
# Default placement spreads workers as far apart as possible across
# NUM_REGIONS_LOGICAL so the routing decision space has the widest
# possible WAN spread. With N workers ≤ N regions we use one region
# per worker; with N > regions we cycle.
def _default_worker_regions(n_workers: int, n_regions: int) -> list[int]:
    if n_workers <= 0:
        return []
    if n_workers <= n_regions:
        # Pick every (n_regions / n_workers)-th region — spreads them
        # maximally. Examples:
        #   n=2, regions=8 → step=4 → [0, 4]
        #   n=4, regions=8 → step=2 → [0, 2, 4, 6]
        #   n=8, regions=8 → step=1 → [0, 1, 2, 3, 4, 5, 6, 7]
        step = n_regions / n_workers
        return [int(round(i * step)) % n_regions for i in range(n_workers)]
    # More workers than regions → cycle (e.g. 2 workers per region).
    return [i % n_regions for i in range(n_workers)]


def _parse_worker_regions(env_val: str | None,
                          n_workers: int, n_regions: int) -> list[int]:
    if not env_val:
        return _default_worker_regions(n_workers, n_regions)
    out = [int(x.strip()) for x in env_val.split(",") if x.strip() != ""]
    if len(out) != n_workers:
        raise ValueError(
            f"WORKER_REGIONS env has {len(out)} entries but N_DP_WORKERS="
            f"{n_workers}; lengths must match")
    if any(r < 0 or r >= n_regions for r in out):
        raise ValueError(
            f"WORKER_REGIONS contains an out-of-range index "
            f"(must be in [0, {n_regions})); got {out}")
    return out


WORKER_REGIONS: list[int] = _parse_worker_regions(
    os.environ.get("WORKER_REGIONS"), N_DP_WORKERS, NUM_REGIONS_LOGICAL,
)


def request_origin_region(req_id: int) -> int:
    """Deterministic, well-spread origin-region assignment per request.

    Using a multiplicative hash (Knuth's golden-ratio constant) so
    adjacent req_ids don't cluster into the same region — a clustered
    distribution would produce artificial bursts of same-region load
    that make blind hash schemes look better than they really are.
    """
    h = (req_id * 2654435761) & 0xFFFFFFFF
    return h % NUM_REGIONS_LOGICAL


def wan_ms_to_worker(origin_region: int, worker_rank: int) -> float:
    """One-way WAN propagation from a request's origin region to the
    region of the chosen physical worker, in milliseconds."""
    return float(WAN_LATENCY_MS[origin_region][WORKER_REGIONS[worker_rank]])


# ════════════════════════════════════════════════════════════════════════════
# 7b. New-design constants (ported from simulation/simulate.py)
# ════════════════════════════════════════════════════════════════════════════

# ── Latency-derived multi-level Ludo cuts ───────────────────────────────
# Per-(requester-region S, holder-region X) level chains: level j's key is
# the content hash of the first n₁·(1 + j·α) tokens, where
# n₁(S,X) = WAN[S][X] / C_PREFILL is the break-even prefix length that
# just beats one S↔X network latency's worth of prefill.  Level 0 IS the
# latency floor — a cross-region match shorter than n₁ never surfaces.
# Same-region matching keeps the fine-grained LUDO_CHUNK_SIZES chain.
LUDO_STEP_ALPHA: float = float(os.environ.get("LUDO_STEP_ALPHA", "0.5"))
LUDO_MAX_CUT: int = int(os.environ.get("LUDO_MAX_CUT",
                                       str(max(LUDO_CHUNK_SIZES))))


# ── Ablation switches (SIGMETRICS supplement) — all default OFF ─────────
# Each removes exactly one mechanism from the full system (A0); no forked
# code paths, only guarded branches at the mechanism's single site.
#   A1 ABL_NO_LOADSCORE : score drops q_view and λ → pure affinity
#                         (cache match + WAN distance); ties break random.
#   A2 ABL_LOCAL_ONLY   : chain lookup skips remote regions — a remote
#                         holder never surfaces (cross-migration trigger
#                         is structurally dead: best_remote is always 0).
#   A3 (existing)       : MIG_ENABLE=0 disables both migration kinds.
#   A4 ABL_FIXED_GRID   : every region pair uses the intra-region
#                         LUDO_CHUNK_SIZES grid — no latency-derived
#                         floors/levels.
#   A5 ABL_TIER_BLIND   : tier bits ignored in scoring — DRAM hits are
#                         priced as VRAM (no PCIe-reload term).
def _abl_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip() not in ("", "0", "false", "no", "off")


ABL_NO_LOADSCORE: bool = _abl_flag("ABL_NO_LOADSCORE")
ABL_LOCAL_ONLY: bool = _abl_flag("ABL_LOCAL_ONLY")
ABL_FIXED_GRID: bool = _abl_flag("ABL_FIXED_GRID")
ABL_TIER_BLIND: bool = _abl_flag("ABL_TIER_BLIND")


def _reuse_floor_tokens(rr: int, nr: int) -> float:
    if rr == nr:
        return 0.0
    return float(WAN_LATENCY_MS[rr][nr]) / C_PREFILL_MS_PER_TOKEN_PRIOR


REUSE_FLOOR_TOKENS: np.ndarray = np.array(
    [[_reuse_floor_tokens(rr, nr) for nr in range(NUM_REGIONS_LOGICAL)]
     for rr in range(NUM_REGIONS_LOGICAL)], dtype=float)


def _pair_chain(rr: int, nr: int) -> list[int]:
    """Cut-length chain a region-rr requester probes region-nr holders at."""
    if rr == nr or ABL_FIXED_GRID:      # A4: intra grid applied globally
        return list(LUDO_CHUNK_SIZES)
    n1 = int(np.ceil(REUSE_FLOOR_TOKENS[rr][nr]))
    if n1 <= 0 or n1 > LUDO_MAX_CUT:
        return []            # even one latency's worth of prefill doesn't fit
    step = max(1, int(round(LUDO_STEP_ALPHA * n1)))
    return list(range(n1, LUDO_MAX_CUT + 1, step))


LUDO_PAIR_CUTS: list[list[list[int]]] = [
    [_pair_chain(rr, nr) for nr in range(NUM_REGIONS_LOGICAL)]
    for rr in range(NUM_REGIONS_LOGICAL)]
# Holder side: every cut a region-X worker must be indexed at — the union of
# all requesting regions' chains into X plus the local fine-grained chain.
LUDO_REGION_CUTS: list[list[int]] = [
    sorted(set(LUDO_CHUNK_SIZES).union(
        *[LUDO_PAIR_CUTS[S][X] for S in range(NUM_REGIONS_LOGICAL)]))
    for X in range(NUM_REGIONS_LOGICAL)]

# ── Load-gated routing / cross-region load view (Ludo + Dynamo) ─────────
# Each region's router sees its own dispatches live and learns the load
# other routers put on the fleet through a periodic (LB_BCAST_MS) load
# report, EWMA-folded — mirrors the sim's piggyback+suppressed-broadcast
# dissemination at the fidelity a real HTTP deployment supports.
LUDO_LB_BCAST_MS: float = float(os.environ.get("LUDO_LB_BCAST_MS", "100"))
LUDO_LB_EWMA_BETA: float = float(os.environ.get("LUDO_LB_EWMA_BETA", "0.5"))
# Busy threshold θ (ms). Default ∞ = gate OFF: candidacy stays soft via
# the T_q_view score term (the θ-sweep in the sim showed any finite θ herds).
LUDO_LB_BUSY_THRESH_MS: float = float(os.environ.get(
    "LUDO_LB_BUSY_THRESH_MS", "inf"))
# Capacity pricing λ: prefill compute is charged (1+λ)× in the joint
# score; λ = min(LUDO_LAMBDA_MAX, min_view_queue / LUDO_LAMBDA_SCALE_MS)
# rises with fleet saturation so the argmin slides toward cache holders
# exactly when compute is the scarce resource.
LUDO_LAMBDA_SCALE_MS: float = float(os.environ.get("LUDO_LAMBDA_SCALE_MS", "100"))
LUDO_LAMBDA_MAX: float = float(os.environ.get("LUDO_LAMBDA_MAX", "8"))
# Piggyback channel (data path) — the sim's second dissemination channel.
# When ON (default) and running multi-router real mode, a Ludo cross-region
# dispatch is relayed through the serving region's router
# (/internal/ludo_relay), whose response carries that router's live
# pending vector back to the requester (a few hundred bytes riding on
# traffic that flows anyway).  Busy region pairs therefore refresh their
# load view at their own traffic rate instead of the broadcast period,
# and the periodic broadcast is sender-side suppressed for pairs with a
# recent piggyback.  Set LUDO_PIGGYBACK=0 to ablate (broadcast-only).
LUDO_PIGGYBACK: bool = os.environ.get("LUDO_PIGGYBACK", "1").strip() not in (
    "0", "false", "no", "off")

# ── Decode-rate prior (per output token, ms) — used only for the
# dispatch-occupancy estimate folded into the load view.
C_DECODE_MS_PER_TOKEN_PRIOR: float = float(os.environ.get(
    "C_DECODE_MS_PER_TOKEN_PRIOR", "25.0"))

# ── Explicit popular-KV migration (Ludo) ────────────────────────────────
# Hot prefixes are replicated toward demand in the background:
#   cross-region pull : requests from region S keep matching only REMOTE
#                       holders → replicate into S;
#   intra-region repl : a LOCAL holder exists but keeps losing the
#                       load-gated score (busy) → add a local replica.
# On real hardware both are realised as PROMPT-SHIPPING + re-prefill on
# an idle destination worker (SGLang has no inter-instance KV transfer;
# re-prefill on an idle GPU is the transport the sim itself prefers for
# cross-region moves).  The replica lands in the destination's RadixCache
# (VRAM tier) and is advertised like any other prefill.
MIG_ENABLE: bool = os.environ.get("MIG_ENABLE", "1").strip() not in (
    "0", "false", "no", "off")
MIG_HOT_K: int = int(os.environ.get("MIG_HOT_K", "3"))
MIG_WINDOW_MS: float = float(os.environ.get("MIG_WINDOW_MS", "1000"))
MIG_COOLDOWN_MS: float = float(os.environ.get("MIG_COOLDOWN_MS", "5000"))
MIG_MAX_INFLIGHT: int = int(os.environ.get("MIG_MAX_INFLIGHT", "2"))
MIG_INTRA_RMAX: int = int(os.environ.get("MIG_INTRA_RMAX", "2"))
# Saturation self-quiescing: only migrate to a destination whose viewed
# queue is below this (ms).  0 would demand a perfectly idle GPU, which a
# real fleet under Poisson load almost never shows — a small allowance
# keeps the sim's "persistently idle" semantics workable on real noise.
MIG_IDLE_DST_MS: float = float(os.environ.get("MIG_IDLE_DST_MS", "50"))

# ── SkyWalker baseline (EuroSys'26, two-layer locality-first LB) ────────
SKYWALKER_PROBE_INTERVAL_MS: float = float(os.environ.get(
    "SKYWALKER_PROBE_INTERVAL_MS", "100"))
# Free-slot credits granted per probe: a worker is "available" while the
# router has pushed fewer than this many requests to it since the last
# probe saw it non-busy (selective pushing by pending requests).
SKYWALKER_SLOTS: int = int(os.environ.get("SKYWALKER_SLOTS", "4"))
# A worker counts as busy at probe time when its pending queue estimate
# exceeds this (ms) — the real-machine proxy for "has pending requests".
SKYWALKER_BUSY_MS: float = float(os.environ.get("SKYWALKER_BUSY_MS", "200"))

# ── Dynamo baseline (hierarchical KV-aware router) ──────────────────────
# On a local-cluster cache miss the router fans a lookup out to every
# other cluster and waits for the slowest.  In real multi-router mode the
# fan-out is a real HTTP round-trip to the peer routers; in emulation
# mode the propagation floor below is injected instead (M/D/1 queueing at
# each region's lookup coordinator added on top, as in the sim).
DYNAMO_LOOKUP_SERVICE_RATE_RPS: float = float(os.environ.get(
    "DYNAMO_LOOKUP_SERVICE_RATE_RPS", "15000"))
DYNAMO_LOOKUP_SERVICE_MS: float = 1000.0 / DYNAMO_LOOKUP_SERVICE_RATE_RPS
DYNAMO_XCLUSTER_LOOKUP_MS: np.ndarray = np.array([
    2.0 * max(float(WAN_LATENCY_MS[r][j])
              for j in range(NUM_REGIONS_LOGICAL) if j != r)
    if NUM_REGIONS_LOGICAL > 1 else 0.0
    for r in range(NUM_REGIONS_LOGICAL)], dtype=float)


# ════════════════════════════════════════════════════════════════════════════
# 8. Per-baseline lookup-overhead constants (with citations)
# ════════════════════════════════════════════════════════════════════════════
# These constants are injected as ``await asyncio.sleep(overhead/1000)``
# inside each baseline's route() method. The injection is what makes
# real-machine wall-clock measurements faithful to a planet-scale
# deployment (which the localhost testbed cannot reproduce physically).
# Every value below is either (a) directly measured (WAN matrix), or
# (b) derived in code from such measurements — no free parameters.

# ── (a) Chord — Stoica et al., "Chord: A Scalable P2P Lookup Service",
#         ACM SIGCOMM 2001. ──────────────────────────────────────────────
# A Chord lookup walks O(log₂ N) finger-table hops. Each hop crosses a
# WAN link to the next-responsible node; per-hop cost is WAN_AVG_MS.
_CHORD_HOPS: int = int(np.ceil(np.log2(NUM_NODES_LOGICAL)))
CHORD_LOOKUP_OVERHEAD_MS: float = _CHORD_HOPS * WAN_AVG_MS

# ── (b) Kademlia — Maymounkov & Mazières, "Kademlia: A P2P Information
#         System Based on the XOR Metric", IPTPS 2002. ──────────────────
# Iterative parallel α-probe lookup: at each iteration α nodes are
# queried in parallel and the closest result drives the next iteration.
# Two consequences vs. Chord:
#   (i) α-fold parallel exploration ⇒ iterations = ceil(log_α N), not
#       log_2 N.
#   (ii) per-iteration wall-clock = MAX of α concurrent WAN samples,
#        not the mean. We compute E[max of α i.i.d. draws from the
#        empirical off-diag WAN distribution] analytically below
#        (no Monte-Carlo, fully deterministic).
KADEMLIA_ALPHA: int = int(os.environ.get("KADEMLIA_ALPHA", "3"))


def _expected_max_of_alpha(samples: np.ndarray, alpha: int) -> float:
    """E[max of α i.i.d. draws (with replacement) from the empirical
    distribution of *samples*]. With-replacement is the standard model
    used in DHT analysis when the destination set is large compared to
    α; the difference vs. without-replacement is < 5% for our N=56
    off-diag cells and α=3."""
    s = np.sort(samples)             # ascending
    n = len(s)
    rk = np.arange(1, n + 1)
    p_max = (rk / n) ** alpha - ((rk - 1) / n) ** alpha
    return float(np.dot(s, p_max))


_KADEMLIA_ITERS: float = float(np.ceil(
    np.log(NUM_NODES_LOGICAL) / np.log(KADEMLIA_ALPHA)
))
_KADEMLIA_PER_ITER_MS: float = _expected_max_of_alpha(_off_diag_samples, KADEMLIA_ALPHA)
KADEMLIA_LOOKUP_OVERHEAD_MS: float = _KADEMLIA_ITERS * _KADEMLIA_PER_ITER_MS

# ── (c) Central Directory — production cross-region coordinator. ───────
# RTT: every request pays one user → coordinator → user round-trip. The
# directory lives in ONE fixed region; worst-case users sit in the
# geographically farthest region and pay 2 × WAN_MAX_MS. Calibrated to
# real production measurements:
#     AWS DynamoDB Global Tables : 200–300 ms
#     Google Spanner cross-region: 100–400 ms (OSDI'12, SIGMOD'17)
#     WonderNetwork US ↔ Asia    : 210–250 ms
#     Facebook TAO cross-region  : 100–300 ms
DIRECTORY_LOOKUP_OVERHEAD_MS: float = 2.0 * WAN_MAX_MS

# Service-rate cap of a single linearisable-read coordinator. Conservative
# lower-middle of published sustained throughput:
#     Spanner cross-region reads : 500–2000 rps  (OSDI'12, SIGMOD'17)
#     etcd v3 readIndex          : 500–2000 rps  (etcd benchmarks)
#     ZooKeeper sync()+read      : 500–1000 rps  (cross-region quorum)
# Past this rate the coordinator's M/D/1 queue grows unboundedly — the
# textbook scalability bound of any centralised discovery plane, which
# distributed planes (Ludo, DHTs) structurally avoid.
DIRECTORY_SERVICE_RATE_RPS: float = float(os.environ.get(
    "DIRECTORY_SERVICE_RATE_RPS", "750"
))
DIRECTORY_SERVICE_MS: float = 1000.0 / DIRECTORY_SERVICE_RATE_RPS


# ════════════════════════════════════════════════════════════════════════════
# 9. Diagnostics — print effective config when imported by the router
# ════════════════════════════════════════════════════════════════════════════
def summary() -> str:
    return (
        f"[config] SCHEME={SCHEME}  workers={N_DP_WORKERS}  model={MODEL_NAME}\n"
        f"[config] WAN snapshot={WAN_PERCENTILE}  "
        f"avg={WAN_AVG_MS:.1f}ms  max={WAN_MAX_MS:.1f}ms  "
        f"({NUM_REGIONS_LOGICAL} regions)\n"
        f"[config] Logical topology: NUM_NODES_LOGICAL={NUM_NODES_LOGICAL}\n"
        f"[config] Overheads (ms): "
        f"Chord={CHORD_LOOKUP_OVERHEAD_MS:.1f} "
        f"(={_CHORD_HOPS}×{WAN_AVG_MS:.1f})  "
        f"Kademlia={KADEMLIA_LOOKUP_OVERHEAD_MS:.1f} "
        f"(={_KADEMLIA_ITERS:.0f}×{_KADEMLIA_PER_ITER_MS:.1f})  "
        f"CentralDir RTT={DIRECTORY_LOOKUP_OVERHEAD_MS:.1f}  "
        f"μ_coord={DIRECTORY_SERVICE_RATE_RPS}rps"
    )
