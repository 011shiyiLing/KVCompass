"""Routing-scheme registry — maps the ``SCHEME`` env var to a class."""
from ludo_sglang.router.schemes.base import RouteDecision, Scheme
from ludo_sglang.router.schemes.central_directory import CentralDirectoryScheme
from ludo_sglang.router.schemes.chord import ChordScheme
from ludo_sglang.router.schemes.consistent_hash import ConsistentHashScheme
from ludo_sglang.router.schemes.dynamo import DynamoScheme
from ludo_sglang.router.schemes.kademlia import KademliaScheme
from ludo_sglang.router.schemes.ludo import LudoScheme
from ludo_sglang.router.schemes.radix_tree import RadixTreeScheme
from ludo_sglang.router.schemes.skywalker import SkyWalkerScheme

SCHEME_REGISTRY: dict[str, type[Scheme]] = {
    LudoScheme.name:              LudoScheme,
    RadixTreeScheme.name:         RadixTreeScheme,
    ConsistentHashScheme.name:    ConsistentHashScheme,
    KademliaScheme.name:          KademliaScheme,
    ChordScheme.name:             ChordScheme,
    CentralDirectoryScheme.name:  CentralDirectoryScheme,
    SkyWalkerScheme.name:         SkyWalkerScheme,
    DynamoScheme.name:            DynamoScheme,
}

__all__ = ["RouteDecision", "Scheme", "SCHEME_REGISTRY"]
