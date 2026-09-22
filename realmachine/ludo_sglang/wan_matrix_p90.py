# ── WAN_LATENCY_MS — auto-generated, do not hand-edit. ─────────────────
# Source : CloudPing.co inter-region TCP RTT
#          https://api.cloudping.co/latencies
#          percentile = 'p_90', timeframe = '1M', unit = 'milliseconds'
# Snapshot file : cloudping_2026-04-27_p_90_1M.json
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
    [   0.0,   36.3,   34.9,   46.7,   75.6,  112.2,  108.8,   59.3],   # [0] from us-east-1
    [  36.3,    0.0,   62.8,   74.9,   52.5,   83.5,  119.3,   91.9],   # [1] from us-west-2
    [  34.9,   62.8,    0.0,   11.1,  102.4,   88.7,   76.4,   89.8],   # [2] from eu-west-1
    [  46.7,   74.9,   11.1,    0.0,  126.9,   94.1,   64.5,  102.4],   # [3] from eu-central-1
    [  75.6,   52.5,  102.4,  126.9,    0.0,   35.5,   64.8,  131.0],   # [4] from ap-northeast-1
    [ 112.2,   83.5,   88.7,   94.1,   35.5,    0.0,   31.9,  167.0],   # [5] from ap-southeast-1
    [ 108.8,  119.3,   76.4,   64.5,   64.8,   31.9,    0.0,  164.9],   # [6] from ap-south-1
    [  59.3,   91.9,   89.8,  102.4,  131.0,  167.0,  164.9,    0.0],   # [7] from sa-east-1
], dtype=float)
