# KVCompass

KVCompass is a KV cache discovery plane for Internet-scale prefix cache
sharing in LLM serving. It tracks where reusable prefix KV caches reside
across geographic regions with a compact keyless Ludo index replicated at
every edge node, and forwards each request with a joint score over cache
reuse, worker load, network latency, and storage tier.

This repository contains the two artifacts evaluated in the paper.

## Layout

```
simulation/    Event-driven simulator (large-scale evaluation)
  simulate.py, run_section3.py   main simulation driver and robustness knobs
  scripts/                   motivation analyses (WildChat/LMSYS extraction,
                             eviction study, production-trace replay)
  datasets/                  dataset placement (see its README; not shipped)
realmachine/   Real-machine prototype (SGLang integration)
  src/ludo/                  Ludo index bindings (C++ Smash Ludo + Python)
  deps/smash/                vendored Ludo hashing implementation
  ludo_sglang/               router, authority, gossip, workers, workloads
  experiments/p1_micro/      index operation microbenchmarks
  experiments/p2_mem/        index memory footprint measurement
  scripts/                   fleet setup and experiment drivers (aws/, vast/)
```

The simulator reuses the real prototype's Ludo table: build the native
binding once under `realmachine/` and the simulation picks it up
automatically (it falls back to a semantically identical Python dict with a
warning if the native module is absent).

## Build

```
cd realmachine
pip install -r requirements.txt
bash scripts/setup/build_ludo.sh        # compiles src/ludo against deps/smash
```

## Reproducing the paper's experiments

| Paper experiment | Entry point |
|---|---|
| Large-scale simulation (64 workers / 8 regions) | `simulation/run_section3.py --figs 125 --workload {mixed,wildchat,lmsys} --n-tenants 50` |
| Robustness to WAN jitter and update loss/delay | same driver with `SIM_WAN_JITTER` / `SIM_UPDATE_LOSS` / `SIM_UPDATE_DELAY_MS` |
| 16-region / 128-worker simulation (appendix) | same driver with the 16-region configuration in `simulate.py` |
| M/G/1 queueing analysis | `simulation/fig_mg1.py` |
| Real-machine testbed (3 regions) | `realmachine/scripts/vast/run_multiregion_vast.sh` (or `scripts/aws/run_multiregion.sh`) |
| Ablation study (real machine) | `realmachine/scripts/vast/run_ablation.sh` |
| Index operation microbenchmarks | `realmachine/experiments/p1_micro/build.sh && ./p1_bench` |
| Index memory footprint | `realmachine/experiments/p2_mem/run_p2_full.sh` |
| Motivation: cross-region reuse on WildChat | `simulation/scripts/wildchat_analysis/` |
| Motivation: eviction policies | `simulation/scripts/evict_analysis/evict_motivation.py` |
| Appendix: production-trace replay | `simulation/scripts/trace_analysis/replay_traces.py` |

Datasets are not shipped; see `simulation/datasets/README.md`.
