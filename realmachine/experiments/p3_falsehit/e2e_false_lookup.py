"""e2e_false_lookup.py — end-to-end false-lookup rate from real-machine
per-request CSVs (paper Table 5, right half).

A request is a claimed hit when the discovery plane reported a match of at
least 64 tokens at dispatch (matched_tokens >= 64). It is a false lookup
when the chosen worker did not actually hold the claimed match, judged by
the official criterion of scripts/vast/summarize_cells.py: a real hit needs
cached_tokens > TEMPLATE_PREFIX + 20 = 50 (the ~30-token chat-template
prefix is always cached).

Usage: python3 e2e_false_lookup.py <results_dir> [<results_dir> ...]
  where each results_dir holds <scheme>_r<rate>.csv merged request logs.
"""
import csv, glob, re, sys
from pathlib import Path

SCHEMES = ['ludo', 'central_directory', 'radix_tree', 'consistent_hash',
           'chord', 'skywalker', 'dynamo']
CLAIM_MIN = 64      # matched_tokens >= 64 counts as a claimed hit
HIT_MIN = 50        # cached_tokens > 50 counts as a real hit (official)

for rd in sys.argv[1:]:
    print(f"\n#### {Path(rd).name} ####")
    print(f"{'scheme':<18}{'claimed':>10}{'false':>8}{'FP-rate%':>10}")
    for s in SCHEMES:
        claimed = fp = 0
        for f in glob.glob(f"{rd}/{s}_r*.csv"):
            if re.search(r'\.region\d+\.csv$', f):
                continue
            for row in csv.DictReader(open(f)):
                if row.get('ok') != '1':
                    continue
                try:
                    m = int(row['matched_tokens']); c = int(row['cached_tokens'])
                except (KeyError, ValueError):
                    continue
                if m >= CLAIM_MIN:
                    claimed += 1
                    if c <= HIT_MIN:
                        fp += 1
        r = 100 * fp / claimed if claimed else 0
        print(f"{s:<18}{claimed:>10,}{fp:>8,}{r:>10.3f}")
