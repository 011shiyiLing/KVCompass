"""
ludo_sglang.authority — central-directory maintenance plane.

This package implements the Smash-style discovery plane introduced when
``config.MAINTENANCE_MODE == 'central'``.

Components::

    name_server          registry + heartbeat watcher (one per fleet)
    discovery_authority  single-shard authoritative writer (one or more)
    authority_main       CLI entry: python -m ludo_sglang.authority.authority_main
    messages           wire format (TypedDicts + event names)

The legacy gossip mode (``MAINTENANCE_MODE == 'gossip'``) is implemented
in ``ludo_sglang.router.gossip`` and continues to work alongside this
package; the two are selected by feature flag in ``config.py``.
"""
