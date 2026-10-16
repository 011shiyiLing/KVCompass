# Real-machine prototype

The deployed KVCompass prototype evaluated in §4–§5 of the paper: a discovery
plane in front of an SGLang fleet spread across geographic regions. Each
region runs an edge router (FastAPI) holding the full multi-level Ludo index;
workers advertise their SGLang HiRadixCache contents (including VRAM/DRAM
tier) into the index via a discovery authority, and routers exchange load
views by gossip.

Per request, the router tokenizes the prompt with the model's chat template
(so its lookup keys live in the same key space the worker advertises into),
walks the multi-level index from the longest prefix level down, and scores
candidate workers jointly over cache match, load, WAN latency, and storage
tier.

## Routing schemes

The same router runs 8 schemes, selected by the `SCHEME` environment variable
(`ludo_sglang/router/schemes/`): `ludo` (ours), `radix_tree`,
`consistent_hash`, `kademlia`, `chord`, `central_directory`, `skywalker`,
`dynamo`. Baselines are driven by WAN-latency constants derived from a
CloudPing.co snapshot (`ludo_sglang/wan_matrix_p*.py`;
`scripts/setup/build_wan_matrix.py` regenerates them).

## Layout

```
src/ludo/                  # Discovery index: pybind11 C++ binding + Python tables
  ludo_binding.cpp         #   Smash ControlPlaneLudo/DataPlaneLudo wrapper
  ludo_table.py            #   multi-level table used by router & simulator
deps/                      # Vendored Smash / Ludo hashing (pins: ../DEPS_MANIFEST.txt)
ludo_sglang/
  config.py                # All knobs (env-overridable): SCHEME, CHUNK_SIZES, ablations
  router/                  # Edge router, gossip, load tracker, scheme registry
  authority/               # Discovery authority, name server, persistence
  worker/                  # SGLang HiRadixCache tier-advertisement patch
  workload/                # Mixed-trace construction and open-loop replay
  client/                  # Async SGLang HTTP client
experiments/
  p1_micro/                # Index operation microbenchmarks (§5.5)
  p2_mem/                  # Index memory footprint (§6.1)
  p3_falsehit/             # False-hit micro- and end-to-end benchmarks (§6.3)
scripts/
  setup/                   # build_ludo.sh, SGLang tier patch, WAN-matrix tools
  experiment/              # sglang_fleet.sh, discovery_fleet.sh, run_experiment*.sh
  vast/                    # vast.ai fleet drivers (testbed used in the paper)
  aws/                     # AWS multi-region drivers
  plotting/                # plot_fig1_fig2.py, plot_eval_grid.py (result CSVs → figures)
```

`router/MULTI_EDGE_DEPLOY.md` and `worker/SGLANG_TIER_PATCH.md` document the
deployment topology and the worker-side patch in detail.

## Build

```bash
pip install -r requirements.txt
bash scripts/setup/build_ludo.sh                  # compiles src/ludo against deps/smash
python scripts/setup/apply_sglang_tier_patch.py   # on worker machines, after installing sglang
```

`sglang` is installed per machine to match the host CUDA version (it is not in
`requirements.txt` on purpose; the vast.ai "SGLang" template has it
pre-installed).

## Running the testbed experiments

Fleet-level drivers fill in a `hosts.env` (copy the `hosts.env.example` next to
each driver) describing the boxes per region, then bring up workers, authority,
and routers, replay the workload, and collect per-scheme results.

- **vast.ai fleet (paper testbed):** `scripts/vast/run_multiregion_vast.sh`
  end-to-end; `scripts/vast/run_ablation.sh` for the §5.4 ablations;
  `scripts/vast/run_preflight.sh` checks the fleet, `overnight.sh` queues the
  full scheme sweep, and `measure_wan.sh` / `tunnel_mesh.sh` handle WAN
  measurement and connectivity.
- **AWS multi-region:** `scripts/aws/launch_instances.sh`,
  `setup_worker.sh` / `setup_router.sh`, then `run_multiregion.sh`;
  `merge_region_results.py` aggregates.
- **Lower-level pieces** (used by the drivers): `scripts/experiment/sglang_fleet.sh`
  and `discovery_fleet.sh` start workers and the discovery plane;
  `run_experiment.sh` replays one workload against one scheme.

## Index benchmarks (CPU-only)

```bash
# §5.5 — operation latency/throughput microbenchmarks
cd experiments/p1_micro && ./build.sh && ./p1_bench

# §6.1 — memory footprint vs entries and levels.
# Needs a token dump first: experiments/p2_mem/export_tokens.py turns a
# replayed trace into results_tmp/p2_tokens.bin (see the script headers).
cd experiments/p2_mem && ./run_p2_full.sh

# §6.3 — false-hit rate microbenchmark (standalone)
cd experiments/p3_falsehit && python3 fp_microbench.py
# e2e_false_lookup.py quantifies the end-to-end effect from testbed
# result logs: python3 e2e_false_lookup.py <results_dir> ...
```

After a testbed run, `scripts/plotting/plot_fig1_fig2.py --results-dir <dir>`
renders the per-scheme latency figures and `scripts/plotting/plot_eval_grid.py`
the combined grid.
