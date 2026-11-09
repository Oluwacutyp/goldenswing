"""The shared bundle contract — GoldenSwing side (WS1-P7).

Devon's Archivist (nomorals/archives.py, WS1-P5) defines ONE bundle
format for every family that packs files into a transportable unit:
a zip or tar-family archive carrying a manifest member,
``bundle.manifest.json``, written LAST, whose JSON shape is the
cross-repo contract::

    {"format": "nomorals-bundle", "version": 1,
     "created_by": <str>, "created_at": <epoch float>,
     "kind": <str>,
     "members": [{"name": str, "size": int, "sha256": str,
                  "is_dir": bool}, ...]}

GoldenSwing must not import Devon code (separate repo, separate
host process), so this module mirrors the contract with stdlib
only — the same member name, the same field names, the same JSON
encoding (``json.dumps(..., indent=2, ensure_ascii=False)``) — so
a bundle written here opens and verifies natively in Devon's
reader (``open_bundle`` / ``verify_bundle``), and Devon's reader
treats a family's own ``manifest.json`` as a ride-along, never as
the contract.

Verify semantics on the Devon side, which the writer must satisfy:
every recorded member is checked for presence, size and SHA-256,
and any archive member the manifest does not record (other than
the bundle manifest itself) is itself a problem — so the member
list must cover EVERY other member of the archive, generated
members (RESULTS.md, the family manifest.json) included.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

#: Contract constants — mirrored from Devon's BUNDLE_FORMAT /
#: BUNDLE_FORMAT_VERSION / BUNDLE_MANIFEST_NAME (archives.py).
BUNDLE_FORMAT = "nomorals-bundle"
BUNDLE_FORMAT_VERSION = 1
BUNDLE_MANIFEST_NAME = "bundle.manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file — the contract's one hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def member_entry(name: str, *, size: int, sha256: str,
                 is_dir: bool = False) -> dict:
    """One manifest member dict, in the contract's field shape."""
    return {"name": str(name), "size": int(size),
            "sha256": str(sha256), "is_dir": bool(is_dir)}


def bytes_member(name: str, data: bytes) -> dict:
    """Member entry for bytes about to be written to the archive."""
    return member_entry(name, size=len(data), sha256=sha256_bytes(data))


def file_member(name: str, path: str | Path) -> dict:
    """Member entry for a file about to be written to the archive."""
    p = Path(path)
    return member_entry(name, size=p.stat().st_size,
                        sha256=sha256_file(p))


def bundle_manifest(members, *, created_by: str, kind: str = "",
                    created_at: float | None = None) -> dict:
    """Build the standard bundle manifest dict.

    ``members`` is an iterable of dicts with ``name`` / ``size`` /
    ``sha256`` (``is_dir`` optional) — the same input shape Devon's
    ``bundle_manifest`` accepts, producing the same output shape.
    """
    entries = [member_entry(m["name"], size=m.get("size", 0),
                            sha256=m.get("sha256", ""),
                            is_dir=m.get("is_dir", False))
               for m in members]
    return {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_FORMAT_VERSION,
        "created_by": str(created_by),
        "created_at": float(created_at if created_at is not None
                            else time.time()),
        "kind": str(kind),
        "members": entries,
    }


def bundle_manifest_bytes(members, *, created_by: str, kind: str = "",
                          created_at: float | None = None) -> bytes:
    """The manifest as archive-ready bytes (Devon's encoding:
    ``json.dumps(indent=2, ensure_ascii=False)`` as UTF-8)."""
    manifest = bundle_manifest(members, created_by=created_by,
                               kind=kind, created_at=created_at)
    return json.dumps(manifest, indent=2,
                      ensure_ascii=False).encode("utf-8")
