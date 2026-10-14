"""Offline outbox: where alerts go when a channel has no credentials.

Writes, per alert and channel: ``<alert_id>.<channel>.png`` (chart),
``<alert_id>.<channel>.txt`` (thesis text) and ``<alert_id>.json``
(shared metadata, written once).  Files state plainly in their header
that they were NOT sent.
"""
from __future__ import annotations

import json
from pathlib import Path

from .model import Alert

DEFAULT_OUTBOX = Path(__file__).resolve().parents[2] / "data" / "alert_outbox"


def resolve_outbox(explicit: str | Path | None = None) -> Path:
    import os
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("GS_ALERT_OUTBOX")
    return Path(env) if env else DEFAULT_OUTBOX


def write_alert(outbox_dir: str | Path, alert: Alert, channel: str) -> list[Path]:
    out = Path(outbox_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if alert.chart_png:
        p = out / f"{alert.alert_id}.{channel}.png"
        p.write_bytes(alert.chart_png)
        written.append(p)
    p = out / f"{alert.alert_id}.{channel}.txt"
    p.write_text(
        f"[OFFLINE OUTBOX — {channel} alert NOT sent: no credentials "
        f"configured]\n\n{alert.text}\n", encoding="utf-8")
    written.append(p)
    meta = out / f"{alert.alert_id}.json"
    if not meta.exists():
        meta.write_text(json.dumps({
            "alert_id": alert.alert_id,
            "instrument": alert.instrument,
            "mode": alert.mode,
            "direction": alert.direction,
            "kind": alert.kind,
            "score": alert.score,
            "is_golden": alert.is_golden,
            "entry_ref": alert.entry_ref,
            "entry_zone": list(alert.entry_zone),
            "invalidation": alert.invalidation,
            "targets": alert.targets,
            "factors": alert.factors,
            "bar_index": alert.bar_index,
            "timestamp": str(alert.timestamp),
            "forecast_note": alert.forecast_note,
        }, indent=2), encoding="utf-8")
        written.append(meta)
    return written
