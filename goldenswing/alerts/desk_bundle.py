"""The desk draft bundle — GoldenSwing side (WS3-P7).

The /desk card is the day's public face, and its outward step
belongs to Devon's content cooperative: the owner approves and
posts from Devon's review queue, never from the trading bot.
So the desk delivery gains exactly one affordance — this bundle.
The rendered card (PNG + its caption, the same bytes and words
/desk delivers) is packed as a nomorals-bundle (alerts/bundle.py
contract, kind ``gs-desk-card``) the owner hands to Devon, where
it lands as ONE draft per GS day with the PNG attached.

Nothing else crosses: no social code, no queue client, no
network call to Devon — a file out, the same shape /statebundle
already proves.

Daily identity: the bundle for a day is ONE file,
``goldenswing-desk-<YYYY-MM-DD>.zip``.  Asking again the same
day rebuilds it from the day's current card and replaces the
file — a named supersede by rebuild, never a second identity —
and Devon's landing dedupes by the family manifest's day, so a
day can only ever produce one draft.
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .bundle import (BUNDLE_MANIFEST_NAME, bundle_manifest_bytes,
                     bytes_member)

#: The bundle kind recorded in the contract manifest — one
#: string, both repos (Devon: agents/desk_drafts.py
#: GS_DESK_BUNDLE_KIND).
DESK_BUNDLE_KIND = "gs-desk-card"

#: Member names inside the desk bundle.
DESK_PNG_MEMBER = "desk.png"
DESK_CAPTION_MEMBER = "caption.txt"
DESK_FAMILY_MEMBER = "manifest.json"


def desk_bundle_name(day: str) -> str:
    """The one filename a day's desk bundle ever has."""
    return f"goldenswing-desk-{day}.zip"


def build_desk_bundle(*, out_dir: Path, day: str, png: bytes,
                      caption: str, stats: dict | None = None
                      ) -> tuple[Path, dict]:
    """Pack one day's desk card as a desk bundle.

    Returns (zip_path, family_manifest).  Members: the card PNG
    and the caption as rendered for /desk, then the family
    manifest (day, caption, the card's headline numbers), then
    the contract manifest LAST — the order Devon's verify_bundle
    contract requires (every other member recorded in it).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / desk_bundle_name(day)
    stats = dict(stats or {})
    family = {
        "bundle": "goldenswing desk card",
        "day": str(day),
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "caption": str(caption),
        "stats": {
            "pnl": stats.get("pnl"),
            "currency": stats.get("currency"),
            "trades": stats.get("trades"),
            "win_rate": stats.get("win_rate"),
            "equity": stats.get("equity"),
            "halted": bool(stats.get("halted") or False),
        },
    }
    caption_bytes = str(caption).encode("utf-8")
    family_bytes = json.dumps(
        family, indent=2, ensure_ascii=False).encode("utf-8")
    members = [
        bytes_member(DESK_PNG_MEMBER, bytes(png)),
        bytes_member(DESK_CAPTION_MEMBER, caption_bytes),
        bytes_member(DESK_FAMILY_MEMBER, family_bytes),
    ]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(DESK_PNG_MEMBER, bytes(png))
        zf.writestr(DESK_CAPTION_MEMBER, caption_bytes)
        zf.writestr(DESK_FAMILY_MEMBER, family_bytes)
        zf.writestr(BUNDLE_MANIFEST_NAME, bundle_manifest_bytes(
            members, created_by="goldenswing",
            kind=DESK_BUNDLE_KIND))
    family["zip_bytes"] = zip_path.stat().st_size
    family["zip_name"] = zip_path.name
    return zip_path, family
