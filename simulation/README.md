# Simulation

Trace-driven, event-based simulator of the KVCompass discovery plane for the
paper's large-scale evaluation (§5.3), robustness study (§6.2), and appendix
experiments. No GPU inference is run; serving nodes are modeled by cost
functions calibrated on Llama-3.1-8B (1×A100) or Llama-3.1-70B (4×A100 TP4,
`SIM_MODEL=70b`).

The topology is 64 workers across 8 regions, with per-region capacity
proportional to a real LLM-deployment census (US 46%, CN 19%, DE 12%, …) and
a WAN-latency matrix derived from inter-region great-circle distances via a
propagation model.

## Routing schemes

The default figure set compares KVCompass against the paper's six baselines:

| Scheme | What it models |
|--------|----------------|
| `ludo` | KVCompass: multi-level Ludo index + joint score (ours) |
| `ludo_coarse` | KVCompass on the coarse chunk grid (ablation baseline) |
| `central_directory` | centralized coordinator paying a WAN RTT per lookup |
| `radix_tree` | PlanetServe-style single-level index |
| `consistent_hash` | static hash-ring affinity |
| `chord` | multi-hop DHT lookup |
| `skywalker` | per-region load balancers with stale-view selective pushing |
| `dynamo` | per-region cache-aware routers with periodic global broadcast |

`simulate.py` additionally implements `random_ll`, `kademlia`, a
zero-latency `centralized` tracker, and an omniscient `oracle` upper bound,
which the default driver does not run.

## Layout

```
simulate.py              # Core framework: topology, cost model, schemes, events
run_section3.py          # Main driver: figure generation + CLI knobs
workload_multitenant.py  # Multi-tenant prefix-pool amplification (--n-tenants)
fig_mg1.py               # M/G/1 queueing analysis (§3.5)
scripts/
  wildchat_analysis/     # Motivation: cross-region reuse on WildChat/LMSYS (§2)
  evict_analysis/        # Motivation: eviction-policy study (§2)
  trace_analysis/        # Appendix: Bailian/Mooncake production-trace replay
datasets/                # Workload files (not shipped — see datasets/README.md)
results/                 # Figure and JSON-cache output (created on first run)
```

## Setup

```bash
pip install -r requirements.txt
```

The simulator reuses the real prototype's Ludo table when the native binding
has been built under `../realmachine/` (see the top-level README). Without it,
a semantically identical Python dict backend is used and a warning is printed.
`LUDO_TABLE_BACKEND=native|dict` selects explicitly.

Place the workload datasets under `datasets/` first — see
[`datasets/README.md`](datasets/README.md) for the expected files and how each
is derived from its public release.

## Running

Fast debug pass (fewer requests and rate points):

```bash
python3 run_section3.py --quick --workload mixed
```

Paper-scale run used in §5.3:

```bash
python3 run_section3.py --figs 125 --workload mixed --n-tenants 50
python3 run_section3.py --figs 125 --workload wildchat --n-tenants 50
python3 run_section3.py --figs 125 --workload lmsys --n-tenants 50
```

Key flags (`python3 run_section3.py --help` for the full list):

| Flag | Meaning |
|------|---------|
| `--workload` | `mixed` (ToolBench 50% / APPS 45% / LooGLE 5%), `toolbench`, `loogle`, `aaps`, `wildchat`, `lmsys` |
| `--figs` | which figures to generate: `1` p99-TTFT-vs-rate sweep, `2` TTFT/E2E CDFs + box plots, `3` normalized cache hit rate (FLOPs-saved stats cached as JSON), `4` per-node load heatmap, `5` closed-loop throughput |
| `--n-tenants` | multi-tenant prefix-pool amplification (1 = single tenant; 50 used in §5.3) |
| `--fig1-rates` / `--fig1-seeds` | arrival-rate sweep override / multi-seed mean±std band |
| `--tpt-clients` / `--tpt-duration` | closed-loop client population and duration for `--figs 5` |
| `--seed`, `--results-dir` | reproducibility and output location |

Figures and JSON result caches land in `results/`; re-plotting an existing
cache does not re-run the simulation.

## Environment knobs

| Variable | Effect |
|----------|--------|
| `SIM_MODEL=8b\|70b` | cost-model constants (Llama-3.1-8B default) |
| `SIM_WAN_JITTER` | multiplicative WAN-latency jitter (fraction) — §6.2 |
| `SIM_UPDATE_LOSS` | probability an index-update advertisement is dropped — §6.2 |
| `SIM_UPDATE_DELAY_MS` | added advertisement propagation delay — §6.2 |
| `ABL_NO_LOADSCORE`, `ABL_LOCAL_ONLY`, `ABL_TIER_BLIND` | ablation switches mirroring `realmachine/ludo_sglang/config.py` |
| `LUDO_TABLE_BACKEND=native\|dict` | discovery-index backend selection |

## Motivation and appendix analyses

- `scripts/wildchat_analysis/` — extracts WildChat-1M / LMSYS-Chat-1M
  conversations and quantifies cross-region prefix reuse (§2).
- `scripts/evict_analysis/evict_motivation.py` — eviction-policy comparison (§2).
- `scripts/trace_analysis/replay_traces.py` — standalone replay of Bailian /
  Mooncake production traces measuring cross-region reuse (appendix);
  `extract_na16.py` is the matching WildChat analysis it mirrors.
