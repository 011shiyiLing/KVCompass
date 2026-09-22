"""
authority_main.py — CLI entry for a DiscoveryAuthority process.

Usage::

    SHARD_ID=0 \
    AUTHORITY_URLS=http://10.0.0.1:8101,http://10.0.0.1:8102 \
    NAMESERVER_URL=http://10.0.0.1:9999 \
    N_DP_WORKERS=8 \
    python -m ludo_sglang.authority.authority_main

By default the authority binds on the host:port extracted from
``AUTHORITY_URLS[SHARD_ID]``; if that is not set, falls back to
``ROUTER_HOST`` from config + ``AUTHORITY_PORT`` env (default 8101).

The matching NameServer entry::

    NAMESERVER_URL=http://10.0.0.1:9999 \
    python -m ludo_sglang.authority.name_server
"""
from __future__ import annotations

import os

import uvicorn

from ludo_sglang import config


def _self_bind() -> tuple[str, int]:
    """Return (host, port) the authority should bind on."""
    if config.SHARD_ID < len(config.AUTHORITY_URLS):
        url = config.AUTHORITY_URLS[config.SHARD_ID]
        # Strip scheme.
        no_scheme = url.split("://", 1)[-1].rstrip("/")
        host, _, port = no_scheme.partition(":")
        return host or "0.0.0.0", int(port or "8101")
    return config.ROUTER_HOST, int(os.environ.get("AUTHORITY_PORT", "8101"))


def main():
    if config.MAINTENANCE_MODE != "central":
        print("[authority_main] WARNING: MAINTENANCE_MODE != 'central'; "
              "authority will start but workers / lookups in gossip mode "
              "won't talk to it.")
    host, port = _self_bind()
    print(f"[authority_main] shard_id={config.SHARD_ID}  bind={host}:{port}")
    uvicorn.run(
        "ludo_sglang.authority.discovery_authority:app",
        host=host, port=port, log_level="info",
    )


if __name__ == "__main__":
    main()
