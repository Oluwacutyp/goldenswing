"""Bill Williams — the research harness (Pass 8):
drives one frame's history through the organs and
produces the family's journal records plus its
evaluation snapshot.  This is the research line's
only driver — nothing in the live system calls it;
it exists so the family can be exercised (tests,
research runs) and journal its own evidence.

Record production, in order:

- ``bw_read`` annotations, sparse: a record on the
  first formed bar, on every bar where the Alligator
  state changes, and on every fractal confirmation
  bar — the annotation stream a reader needs to
  reconstruct what the research line saw, without a
  record per bar of chop.
- ``bw_signal`` for every ablation variant's signals
  and the full rule's abstain arm, each with its
  paired outcome and the entry read (Gator phase /
  BW-MFI class ride along for the lift table).
- one ``bw_evaluation`` snapshot folding the signals
  just produced.

Every record carries the timeframe and, when the
caller threads one, the situation snapshot id — as
context on the research record only; the situation
never reads it back.
"""
from __future__ import annotations

import pandas as pd

from . import evaluation as _evaluation
from .indicators import PreparedWilliams, annotate
from .strategy import run_strategy

__all__ = ["drive_history"]


def _ts(frame, k) -> str | None:
    try:
        return str(frame.index[k])
    except Exception:
        return None


def drive_history(frame: pd.DataFrame, *, timeframe: str,
                  situation_id: str | None = None) -> dict:
    """Drive ``frame`` and return ``{"records": [...],
    "evaluation": snapshot, "signals": {...}}`` — the
    records are journal-ready (append them with
    ``journal.append_records``); the snapshot is the
    same dict the trailing ``bw_evaluation`` record
    carries."""
    prep = PreparedWilliams(frame)
    records: list[dict] = []

    prev_state = None
    for j in range(prep.n):
        alli = prep.alligator_at(j)
        confirmed = prep.fractals.confirmed_at_bar(j)
        if alli["state"] == "unformed" and not confirmed:
            continue
        if alli["state"] != prev_state or confirmed:
            rec = {"type": "bw_read",
                   **annotate(prep, j, timeframe=timeframe,
                              situation_id=situation_id)}
            records.append(rec)
        if alli["state"] != "unformed":
            prev_state = alli["state"]

    signals = run_strategy(prep)
    for sigs in signals.values():
        for sig in sigs:
            records.append({
                "type": "bw_signal",
                "timeframe": timeframe,
                "situation_id": situation_id,
                "ts": sig["entry_ts"],
                "volume_basis": prep.volume_basis,
                **sig,
            })

    snapshot = _evaluation.evaluate_records(records)
    records.append({
        "type": _evaluation.RECORD_TYPE,
        "timeframe": timeframe,
        "situation_id": situation_id,
        "ts": _ts(frame, prep.n - 1) if prep.n else None,
        **snapshot,
    })
    return {"records": records, "evaluation": snapshot,
            "signals": signals}
