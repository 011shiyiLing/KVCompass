# Datasets

The simulation expects the following files in this directory (not shipped
with the repository due to size). Each is derived from its public release.

- `aaps.jsonl` — APPS competitive-programming problems
- `toolbench_zipf_1.1_prompts_6000.jsonl` — ToolBench agent tool-calling
  traces, replayed under a Zipf(1.1) template popularity distribution
- `loogle.jsonl` — LooGLE long-document QA
- `wildchat.jsonl` — WildChat real user conversations (WildChat-1M)
- `lmsys.jsonl` — LMSYS Chatbot Arena conversations (LMSYS-Chat-1M)

The Mixed workload samples ToolBench (50%), APPS (45%), and LooGLE (5%).

## Building `wildchat.jsonl` / `lmsys.jsonl`

Download the Hugging Face parquet shards of WildChat-1M into
`datasets/wildchat-1m/data/` (and LMSYS-Chat-1M into `datasets/lmsys-1m/data/`),
then run the prep scripts in this directory:

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
