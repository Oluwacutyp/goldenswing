"""Bill Williams — the family's OWN journal stream
(Pass 8, SPEC §6.6).  An append-only JSONL file the
research line owns outright: it is not the live
journal, feeds no census, and nothing on the live
path reads it.  Record types:

- ``bw_read`` — a sparse annotation (alligator state
  change or a fractal confirmation bar), the full
  research read from ``indicators.annotate``;
- ``bw_signal`` — one strategy signal (any ablation
  variant, or the full rule's abstain arm) with its
  paired outcome;
- ``bw_evaluation`` — one snapshot of the family's
  own promotion evaluation over the stream.

Damaged lines are counted on read, never guessed at
(the live journal's discipline, kept here too).
"""
from __future__ import annotations

import json
from pathlib import Path

__all__ = ["append_records", "read_records"]


def append_records(path, records: list[dict]) -> int:
    """Append records as JSON lines; returns the count
    written.  Creates parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
            n += 1
    return n


def read_records(path) -> tuple[list[dict], int]:
    """Read the stream back: (records, damaged_lines).
    A missing file is an empty stream, not an error."""
    p = Path(path)
    if not p.exists():
        return [], 0
    out: list[dict] = []
    damaged = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            damaged += 1
            continue
        if isinstance(rec, dict):
            out.append(rec)
        else:
            damaged += 1
    return out, damaged
