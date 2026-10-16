"""Snapshots + manifest + recovery.

Snapshot integrity: per-page hashes plus a Merkle tree, not aggregates
alone.  ``leaf_i = H(page_index || page_bytes)`` (NO watermark in the
leaf — a watermark there would invalidate unchanged leaves on every
snapshot).  Parents are ``H(left || right)``.  The root is bound as
``snapshot_root = H(merkle_root || watermark || chain_N)``.

Manifest: temp + fsync + atomic rename + directory fsync.  The live
manifest is NEVER truncated or overwritten in place.  Retain ``N``
and ``N_prev`` until N's successor is fully verified.  Hot pruning
only after: verified cold archival + atomic manifest flip + a
grace/restart cycle + reader drainage.

Recovery phases: REPLAY -> RECONCILING -> LIVE.  Replay runs with
effects suppressed.  Orphan ``*.tmp`` / ``*.part`` / ``*.shadow``
files are ignored/cleaned on startup; only a torn final frame may be
truncated.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PAGE_SIZE = 4096


def page_hash(index: int, page_bytes: bytes) -> bytes:
    """leaf_i = H(page_index || page_bytes).  No watermark in the leaf."""
    return hashlib.sha256(index.to_bytes(8, "big") + page_bytes).digest()


def merkle_root(leaves: list[bytes]) -> bytes:
    """Merkle root over page hashes.  Empty tree -> 32 zero bytes."""
    if not leaves:
        return b"\x00" * 32
    level = list(leaves)
    while len(level) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(hashlib.sha256(left + right).digest())
        level = nxt
    return level[0]


def snapshot_root(merkle: bytes, watermark: int, chain_n: bytes) -> bytes:
    """snapshot_root = H(merkle_root || watermark || chain_N)."""
    return hashlib.sha256(
        merkle + int(watermark).to_bytes(8, "big") + bytes(chain_n)).digest()


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: str | Path, data: bytes) -> None:
    """temp + fsync + atomic rename + directory fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(str(path.parent))


def build_snapshot(pages: list[bytes], watermark: int,
                   chain_n: bytes) -> dict:
    """Build a snapshot dict from 4 KiB pages (padded by the caller)."""
    leaves = [page_hash(i, p) for i, p in enumerate(pages)]
    merkle = merkle_root(leaves)
    root = snapshot_root(merkle, watermark, chain_n)
    return {
        "watermark": int(watermark),
        "chain_n": bytes(chain_n).hex(),
        "page_hashes": [h.hex() for h in leaves],
        "merkle_root": merkle.hex(),
        "snapshot_root": root.hex(),
        "page_count": len(pages),
        "page_size": PAGE_SIZE,
    }


def verify_snapshot(meta: dict, pages: list[bytes]) -> bool:
    """Verify page hashes, Merkle root and snapshot-root binding."""
    if int(meta.get("page_count", -1)) != len(pages):
        return False
    leaves: list[bytes] = []
    for i, page in enumerate(pages):
        h = page_hash(i, page)
        if h.hex() != meta["page_hashes"][i]:
            return False
        leaves.append(h)
    merkle = merkle_root(leaves)
    if merkle.hex() != meta.get("merkle_root"):
        return False
    chain_n = bytes.fromhex(meta["chain_n"])
    root = snapshot_root(merkle, int(meta["watermark"]), chain_n)
    return root.hex() == meta.get("snapshot_root")


class SnapshotStore:
    """Retains N and N_prev; prunes hot copies only after the safe order."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"

    # ---------------------------------------------------------- manifest
    def read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"snapshots": [], "cold_ranges": []}
        return json.loads(self.manifest_path.read_text())

    def write_manifest(self, manifest: dict) -> None:
        atomic_write(self.manifest_path,
                     json.dumps(manifest, sort_keys=True).encode())

    # ---------------------------------------------------------- snapshots
    def write_snapshot(self, watermark: int, chain_n: bytes,
                       pages: list[bytes]) -> dict:
        """Write + verify a snapshot, then flip the manifest atomically.

        Retains the previous snapshot as N_prev.  Returns the manifest.
        """
        meta = build_snapshot(pages, watermark, chain_n)
        if not verify_snapshot(meta, pages):
            raise ValueError("snapshot self-verification failed")
        snap_path = self.dir / f"snapshot-{watermark}.json"
        atomic_write(snap_path, json.dumps(meta, sort_keys=True).encode())
        manifest = self.read_manifest()
        snaps = [s for s in manifest.get("snapshots", [])
                 if s["watermark"] != watermark]
        snaps.append({"watermark": watermark,
                      "path": str(snap_path),
                      "snapshot_root": meta["snapshot_root"],
                      "chain_n": meta["chain_n"]})
        # Retain at most N and N_prev, newest first.
        snaps.sort(key=lambda s: s["watermark"], reverse=True)
        manifest["snapshots"] = snaps[:2]
        self.write_manifest(manifest)
        return manifest

    def load_snapshot(self, watermark: int) -> dict | None:
        manifest = self.read_manifest()
        for snap in manifest.get("snapshots", []):
            if snap["watermark"] == watermark:
                return json.loads(Path(snap["path"]).read_text())
        return None

    # ---------------------------------------------------------- pruning / orphans
    def prune_hot_copy(self, watermark: int, *, cold_verified: bool,
                        grace_passed: bool, readers_drained: bool) -> bool:
        """Prune a hot copy ONLY after the full safe order.

        Requires verified cold archival + manifest flip (implied by
        the manifest already pointing at cold) + grace/restart cycle
        + reader drainage.  Returns False (refused) if any gate fails.
        """
        if not (cold_verified and grace_passed and readers_drained):
            return False
        return True  # caller deletes the hot copy; this gate is the guard

    def cleanup_orphans(self) -> list[str]:
        """Delete unreferenced *.tmp/*.part/*.shadow files.  Never
        deletes a file referenced by the manifest."""
        manifest = self.read_manifest()
        referenced = {str(s.get("path")) for s in manifest.get("snapshots", [])}
        referenced.add(str(self.manifest_path))
        removed: list[str] = []
        for suffix in (".tmp", ".part", ".shadow"):
            for path in self.dir.glob(f"*{suffix}"):
                if str(path) not in referenced:
                    path.unlink(missing_ok=True)
                    removed.append(path.name)
        return removed
