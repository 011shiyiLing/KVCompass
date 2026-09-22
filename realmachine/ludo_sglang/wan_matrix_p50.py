# ── WAN_LATENCY_MS — auto-generated, do not hand-edit. ─────────────────
# Source : CloudPing.co inter-region TCP RTT
#          https://api.cloudping.co/latencies
#          percentile = 'p_50', timeframe = '1M', unit = 'milliseconds'
# Snapshot file : cloudping_2026-04-27_p_50_1M.json
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
    [   0.0,   32.3,   34.5,   46.3,   74.4,  110.1,  107.9,   57.3],   # [0] from us-east-1
    [  32.3,    0.0,   61.0,   73.2,   51.1,   82.0,  115.4,   89.3],   # [1] from us-west-2
    [  34.5,   61.0,    0.0,   10.8,  101.5,   86.5,   75.4,   89.0],   # [2] from eu-west-1
    [  46.3,   73.2,   10.8,    0.0,  115.3,   82.7,   63.5,  101.6],   # [3] from eu-central-1
    [  74.4,   51.1,  101.5,  115.3,    0.0,   34.9,   64.3,  129.9],   # [4] from ap-northeast-1
    [ 110.1,   82.0,   86.5,   82.7,   34.9,    0.0,   31.4,  164.9],   # [5] from ap-southeast-1
    [ 107.9,  115.4,   75.4,   63.5,   64.3,   31.4,    0.0,  163.2],   # [6] from ap-south-1
    [  57.3,   89.3,   89.0,  101.6,  129.9,  164.9,  163.2,    0.0],   # [7] from sa-east-1
], dtype=float)
