# Datasets

The simulation expects five workload files in this directory. **All five are
derived files produced by us from public dataset releases — none of them can
be downloaded under these names elsewhere.** They are not committed to the
repository because of size.

| File | Derived from | Records | How to obtain |
|------|--------------|---------|---------------|
| `aaps.jsonl` | [APPS](https://huggingface.co/datasets/codeparrot/apps) competitive-programming problems | 1,000 | release asset (below) |
| `loogle.jsonl` | [LooGLE](https://huggingface.co/datasets/bigainlco/LooGLE) long-document QA | 500 | release asset (below) |
| `toolbench_zipf_1.1_prompts_6000.jsonl` | [ToolBench](https://github.com/OpenBMB/ToolBench) agent tool-calling prompts, sampled under a Zipf(1.1) template popularity distribution | 6,000 | release asset (below) |
| `wildchat.jsonl` | [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) real user conversations | — | build locally with `prep_wildchat.py` |
| `lmsys.jsonl` | [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) conversations | — | build locally with `prep_lmsys.py` |

The Mixed workload samples ToolBench (50%), APPS (45%), and LooGLE (5%).

## Mixed-workload files (release assets)

The exact `aaps.jsonl`, `loogle.jsonl`, and
`toolbench_zipf_1.1_prompts_6000.jsonl` used in the paper are attached to the
GitHub release of this repository — download them into this directory:

```bash
gh release download v1.0 --repo 011shiyiLing/KVCompass \
   --pattern '*.jsonl' --dir simulation/datasets/
```

Record schema: `aaps.jsonl` / `loogle.jsonl` are
`{"workload_id": int, "workload": "<problem / document text>"}` per line;
the ToolBench file is `{"text": "<full agent prompt incl. system prefix>"}`
per line.

## `wildchat.jsonl` / `lmsys.jsonl` (build locally)

Both upstream releases are gated and license-restricted, so we do not
redistribute derived copies. Accept the terms on Hugging Face, download the
parquet shards into `datasets/wildchat-1m/data/` (resp.
`datasets/lmsys-1m/data/`), then:

```bash
python datasets/prep_wildchat.py   # → datasets/wildchat.jsonl
python datasets/prep_lmsys.py      # → datasets/lmsys.jsonl
```

The prep scripts need `pyarrow` and (for WildChat's country→region mapping)
`pycountry-convert`, which are not in `requirements.txt`:

```bash
pip install pyarrow pycountry-convert
```

The motivation analyses in `../scripts/wildchat_analysis/` additionally use
`tiktoken` for tokenized prefix statistics.
