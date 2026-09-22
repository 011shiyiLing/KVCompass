"""
Chord DHT — same hash placement as ConsistentHash, plus an injected
``CHORD_LOOKUP_OVERHEAD_MS`` sleep that models the wall-clock cost
a Chord finger-table walk would pay in a real planet-scale deployment.

The overhead value is computed in ``ludo_sglang.config`` from:
   ceil(log_2(NUM_NODES_LOGICAL)) sequential hops × WAN_AVG_MS

drawn from the empirical CloudPing WAN matrix. The hop count is the
Chord paper's published O(log N) finger-table search bound (Stoica
et al., SIGCOMM'01).
"""
from __future__ import annotations

from ludo_sglang import config
from ludo_sglang.router.schemes.consistent_hash import ConsistentHashScheme


class ChordScheme(ConsistentHashScheme):
    name = "chord"
    extra_lookup_overhead_ms = config.CHORD_LOOKUP_OVERHEAD_MS
