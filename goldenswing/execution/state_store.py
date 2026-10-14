"""Warm-resume state store — restart mid-stream, don't replay 11k bars.

The live loop keeps a rolling bar buffer (default 2000 H1 bars — more
than the reader's deepest warmup: 800-bar pools, 252-bar ATR
percentile, ~1440-bar daily EMA).  The store persists:

- ``bars.pkl`` — the rolling OHLCV buffer (pickle; corruption is
  reported, never silently used),
- ``meta.json`` — last processed bar time, last signal bar, instrument,
- ``playbooks/<name>.json`` — playbook working state (pending setups,
  cursors), via the ``Playbook.export_state / import_state`` hook.

On restart: load the buffer, fetch only the missing bars from the
feed, re-read.  The tail states match a full-history run to within
indicator warmup — the cold-start path (full fetch + full read)
remains the fallback and the reference.

The store never touches risk state (``RiskGuards`` owns its file) or
the journal (append-only) — those are separate concerns with their
own durability contracts.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


#: Bars kept in the rolling buffer.  Must exceed the reader's deepest
#: lookback (800-bar pools, 252-bar ATR percentile, daily EMA ~1440).
BUFFER_BARS = 2000


class StateStore:
    def __init__(self, path: str | Path,
                 buffer_bars: int = BUFFER_BARS) -> None:
        self.path = Path(path)
        self.buffer_bars = int(buffer_bars)
        self.load_error: str | None = None

    # ---------------------------------------------------------------- paths
    @property
    def _bars_path(self) -> Path:
        return self.path / "bars.pkl"

    @property
    def _meta_path(self) -> Path:
        return self.path / "meta.json"

    def _playbook_path(self, name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in name)
        return self.path / "playbooks" / f"{safe}.json"

    # ----------------------------------------------------------------- bars
    def save_bars(self, df: pd.DataFrame) -> None:
        """Persist the rolling buffer (tail of ``df``)."""
        self.path.mkdir(parents=True, exist_ok=True)
        tail = df.tail(self.buffer_bars)
        tmp = self.path / "bars.pkl.tmp"
        with tmp.open("wb") as f:
            pickle.dump(tail, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._bars_path)

    def load_bars(self) -> pd.DataFrame | None:
        """The persisted buffer, or None when there is nothing (or it
        is corrupt — corruption is reported, never silently used)."""
        if not self._bars_path.exists():
            return None
        try:
            with self._bars_path.open("rb") as f:
                df = pickle.load(f)
            df = df.sort_index()
            # defensive: drop any duplicated timestamps, keep the last
            df = df[~df.index.duplicated(keep="last")]
            return df
        except Exception as exc:
            self.load_error = f"bar buffer unreadable ({exc}); cold start"
            return None

    # ----------------------------------------------------------------- meta
    def save_meta(self, meta: dict) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        tmp = self._meta_path.with_suffix(".json.tmp")
        # timestamps -> isoformat for JSON safety
        clean = {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                 for k, v in meta.items()}
        tmp.write_text(json.dumps(clean, indent=1))
        tmp.replace(self._meta_path)

    def load_meta(self) -> dict:
        if not self._meta_path.exists():
            return {}
        try:
            return json.loads(self._meta_path.read_text())
        except Exception as exc:
            self.load_error = f"meta unreadable ({exc}); cold start"
            return {}

    # ------------------------------------------------------------- playbooks
    def save_playbook(self, name: str, state: dict) -> None:
        p = self._playbook_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(p)

    def load_playbook(self, name: str) -> dict:
        p = self._playbook_path(name)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    # -------------------------------------------------------- paper state
    @property
    def _paper_path(self) -> Path:
        return self.path / "paper.json"

    def save_paper(self, state: dict) -> None:
        """Persist the paper executor snapshot (cron --once resume)."""
        self.path.mkdir(parents=True, exist_ok=True)
        tmp = self._paper_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=1, default=str))
        tmp.replace(self._paper_path)

    def load_paper(self) -> dict:
        p = self._paper_path
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception as exc:
            self.load_error = f"paper state unreadable ({exc}); cold start"
            return {}

    # ------------------------------------------- multi-TF stack (MTF)
    @property
    def _stack_path(self) -> Path:
        return self.path / "stack.pkl"

    def save_stack(self, frames: dict) -> None:
        """Persist the all-timeframe stack (one frame per TF).

        Tailed per TF so the file stays bounded (D1 800, H4 1500,
        H1 2000, M15/M30/M5 4000 bars).  Cursors live in ``meta.json``
        under ``stack_cursors`` (see ``save_meta`` callers).  A store
        without ``stack.pkl`` is a pre-MTF store: callers cold-start
        the stack from the feed (forward-compatible, never an error).
        """
        caps = {"D1": 800, "H4": 1500, "H1": 2000,
                "M15": 4000, "M30": 4000, "M5": 4000}
        self.path.mkdir(parents=True, exist_ok=True)
        tailed = {tf: df.tail(caps.get(tf, 2000))
                  for tf, df in frames.items()
                  if df is not None and len(df)}
        tmp = self.path / "stack.pkl.tmp"
        with tmp.open("wb") as f:
            pickle.dump(tailed, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._stack_path)

    def load_stack(self) -> dict | None:
        """The persisted stack frames, or None when there is nothing
        (pre-MTF store) or it is corrupt — corruption is reported,
        never silently used."""
        if not self._stack_path.exists():
            return None
        try:
            with self._stack_path.open("rb") as f:
                frames = pickle.load(f)
            out = {}
            for tf, df in frames.items():
                df = df.sort_index()
                df = df[~df.index.duplicated(keep="last")]
                out[tf] = df
            return out
        except Exception as exc:
            self.load_error = (f"MTF stack unreadable ({exc}); "
                               f"cold start")
            return None

    # ----------------------------------------------------------------- misc
    def clear(self) -> None:
        """Wipe the store (fresh cold start)."""
        import shutil
        shutil.rmtree(self.path, ignore_errors=True)
