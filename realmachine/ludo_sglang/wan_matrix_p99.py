# ── WAN_LATENCY_MS — auto-generated, do not hand-edit. ─────────────────
# Source : CloudPing.co inter-region TCP RTT
#          https://api.cloudping.co/latencies
#          percentile = 'p_99', timeframe = '1M', unit = 'milliseconds'
# Snapshot file : cloudping_2026-04-27_p_99_1M.json
# Conversion    : RTT / 2 → one-way ms, then symmetrised (A→B + B→A) / 2
# Region order  : ['us-east-1', 'us-west-2', 'eu-west-1', 'eu-central-1', 'ap-northeast-1', 'ap-southeast-1', 'ap-south-1', 'sa-east-1']

import numpy as np

DEFAULT_REGIONS = [
    'us-east-1',
    'us-west-2',
    'eu-west-1',
    'eu-central-1',
    'ap-northeast-1',
    'ap-southeast-1',
    'ap-south-1',
    'sa-east-1',
]

# Column / row order:
#   [0] us-east-1           [4] ap-northeast-1
#   [1] us-west-2           [5] ap-southeast-1
#   [2] eu-west-1           [6] ap-south-1
#   [3] eu-central-1        [7] sa-east-1
WAN_LATENCY_MS = np.array([
    [   0.0,   38.3,   35.6,   47.3,   85.3,  115.7,  111.5,   63.7],   # [0] from us-east-1
    [  38.3,    0.0,   66.7,   79.0,   58.1,   84.4,  137.4,   94.9],   # [1] from us-west-2
    [  35.6,   66.7,    0.0,   11.8,  107.3,   96.9,   77.2,   90.8],   # [2] from eu-west-1
    [  47.3,   79.0,   11.8,    0.0,  127.9,   95.2,   66.1,  104.1],   # [3] from eu-central-1
    [  85.3,   58.1,  107.3,  127.9,    0.0,   36.0,   67.6,  140.8],   # [4] from ap-northeast-1
    [ 115.7,   84.4,   96.9,   95.2,   36.0,    0.0,   32.7,  171.6],   # [5] from ap-southeast-1
    [ 111.5,  137.4,   77.2,   66.1,   67.6,   32.7,    0.0,  167.4],   # [6] from ap-south-1
    [  63.7,   94.9,   90.8,  104.1,  140.8,  171.6,  167.4,    0.0],   # [7] from sa-east-1
], dtype=float)
