"""
persistence.py — append log + snapshot rotation for one (shard, level).

On authority startup, ``replay(callback)`` reads the latest snapshot then the
tail of the log and invokes ``callback(key, packed)`` for each record;
the authority uses this to re-populate its in-memory AuthorityLudoTable. After
replay, every successful write must call ``append(key, packed)`` to
journal the change. When the log file grows past ``log_max_bytes``, the
authority calls ``dump(state_iter)`` to atomically rewrite the snapshot
from current live state and truncate the log.

Record format: 12 bytes little-endian per record::

    u64 key  ||  u32 packed_value

``packed_value == 0`` is a tombstone (key was removed). The replay path
applies records in file order, so a later tombstone correctly drops a
key inserted by an earlier record.

Durability: writes are buffered in a 4KB in-memory buffer and flushed to
the OS on overflow / explicit flush / close.  No fsync — matches the
Smash master_fn.h behaviour and is appropriate for the experimental
setting (crash recovery is best-effort, not transactional).
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Callable, Iterable, Tuple


REC_FMT = "<QI"                 # u64 key, u32 packed (little-endian)
REC_SIZE = struct.calcsize(REC_FMT)
assert REC_SIZE == 12

DEFAULT_LOG_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_BUFFER_BYTES  = 4096


def _snapshot_path(log_dir: Path, shard_id: int, level: int) -> Path:
    return log_dir / f"authority_s{shard_id}_l{level}.snapshot"


def _log_path(log_dir: Path, shard_id: int, level: int) -> Path:
    return log_dir / f"authority_s{shard_id}_l{level}.log"


def _new_snapshot_path(log_dir: Path, shard_id: int, level: int) -> Path:
    return log_dir / f"authority_s{shard_id}_l{level}.snapshot.new"


class AuthorityPersistence:
    def __init__(
        self,
        log_dir: Path | str,
        shard_id: int,
        level: int,
        log_max_bytes: int = DEFAULT_LOG_MAX_BYTES,
        buffer_bytes: int = DEFAULT_BUFFER_BYTES,
    ):
        self.log_dir = Path(log_dir)
        self.shard_id = shard_id
        self.level = level
        self.log_max_bytes = int(log_max_bytes)
        self.buffer_bytes  = int(buffer_bytes)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = _snapshot_path(self.log_dir, shard_id, level)
        self.log_path      = _log_path(self.log_dir, shard_id, level)
        self.new_snapshot_path = _new_snapshot_path(self.log_dir, shard_id, level)

        self._buffer: bytearray = bytearray()
        self._log_fd: int = -1
        self._log_size: int = 0       # bytes written to log file (excludes buffer)
        self._records_written: int = 0
        self._snapshots_written: int = 0
        self._closed = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    def replay(self, callback: Callable[[int, int], None]) -> int:
        """Apply snapshot then log records to `callback(key, packed)`. Returns
        the number of records applied (post-tombstone collapse not done; the
        callback sees raw record sequence)."""
        applied = 0
        for path in (self.snapshot_path, self.log_path):
            if not path.exists():
                continue
            with open(path, "rb") as f:
                while True:
                    buf = f.read(REC_SIZE)
                    if not buf:
                        break
                    if len(buf) < REC_SIZE:
                        # Torn write at the end of the log — stop here, this
                        # record can't be trusted. (No fsync = no durability
                        # guarantee on the last record.)
                        break
                    key, packed = struct.unpack(REC_FMT, buf)
                    callback(int(key), int(packed))
                    applied += 1

        # Open log fd in append mode after replay.
        self._log_fd = os.open(
            self.log_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        self._log_size = self.log_path.stat().st_size
        return applied

    def append(self, key: int, packed: int) -> None:
        """Buffer one (key, packed) record. Flushes to log file when the
        in-memory buffer reaches `buffer_bytes`."""
        self._buffer.extend(struct.pack(REC_FMT, key & 0xFFFFFFFFFFFFFFFF,
                                        packed & 0xFFFFFFFF))
        self._records_written += 1
        if len(self._buffer) >= self.buffer_bytes:
            self.flush()

    def flush(self) -> None:
        if not self._buffer or self._log_fd < 0:
            return
        os.write(self._log_fd, bytes(self._buffer))
        self._log_size += len(self._buffer)
        self._buffer.clear()

    def needs_dump(self) -> bool:
        return (self._log_size + len(self._buffer)) >= self.log_max_bytes

    def dump(self, state_iter: Iterable[Tuple[int, int]]) -> None:
        """Atomic snapshot rewrite from `state_iter` then log truncate.

        `state_iter` should yield every live (key, packed) pair currently
        in the authority's combined view (see ``AuthorityLudoTable.iter_live``)."""
        # Drop in-memory buffer first; its records are already represented
        # in `state_iter` if it came from the same AuthorityLudoTable.
        self._buffer.clear()

        with open(self.new_snapshot_path, "wb") as f:
            for key, packed in state_iter:
                if packed == 0:
                    continue
                f.write(struct.pack(REC_FMT,
                                    int(key) & 0xFFFFFFFFFFFFFFFF,
                                    int(packed) & 0xFFFFFFFF))
            f.flush()
            os.fsync(f.fileno())  # the snapshot ITSELF needs to be durable
                                  # before we truncate the log; a crash
                                  # between os.replace and log truncate
                                  # would otherwise lose history.
        os.replace(self.new_snapshot_path, self.snapshot_path)

        # Truncate the log file. Re-open the fd so subsequent appends start
        # at offset 0 cleanly.
        if self._log_fd >= 0:
            os.close(self._log_fd)
        # Re-create the log file empty.
        with open(self.log_path, "wb"):
            pass
        self._log_fd = os.open(
            self.log_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        self._log_size = 0
        self._snapshots_written += 1

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        except Exception:
            pass
        if self._log_fd >= 0:
            try:
                os.close(self._log_fd)
            except OSError:
                pass
            self._log_fd = -1
        self._closed = True

    # ── Stats ──────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "shard_id":         self.shard_id,
            "level":            self.level,
            "snapshot_path":    str(self.snapshot_path),
            "log_path":         str(self.log_path),
            "log_size_bytes":   self._log_size,
            "buffer_bytes":     len(self._buffer),
            "log_max_bytes":    self.log_max_bytes,
            "records_written":  self._records_written,
            "snapshots_written": self._snapshots_written,
        }

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
