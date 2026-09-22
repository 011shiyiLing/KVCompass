"""
Kademlia DHT — same hash placement as ConsistentHash, plus an injected
``KADEMLIA_LOOKUP_OVERHEAD_MS`` sleep that models the wall-clock cost
a Kademlia lookup would pay in a real planet-scale deployment.

The overhead value is computed in ``ludo_sglang.config`` from:
   ceil(log_α(NUM_NODES_LOGICAL)) iterations × E[max of α=3 i.i.d.
   draws from the empirical CloudPing WAN matrix]

so this scheme has zero free parameters: every constant either comes
from the WAN snapshot or is the published Kademlia paper's
specification (Maymounkov & Mazières, IPTPS'02).
"""
from __future__ import annotations

from ludo_sglang import config
from ludo_sglang.router.schemes.consistent_hash import ConsistentHashScheme


class KademliaScheme(ConsistentHashScheme):
    name = "kademlia"
    extra_lookup_overhead_ms = config.KADEMLIA_LOOKUP_OVERHEAD_MS
