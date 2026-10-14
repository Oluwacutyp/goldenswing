"""Per-day tick CSV logging + tick-history export for the MT5 engine.

Ticks are logged EXACTLY as seen from the terminal — no resampling,
no fabrication, no filling.  One file per UTC day::

    data/mt5_ticks_YYYYMMDD.csv   (time,bid,ask,tick_volume,real_volume,spread)

``export_tick_history`` pulls N days of terminal tick history via
``copy_ticks_range`` (``feed.ticks_range``) into the same CSV shape for
backtesting.  History rows carry the terminal's own tick volume;
``real_volume`` is 0 when the terminal does not report it (most
metals) — callers must check, never assume.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

TICK_CSV_COLUMNS = ("time", "bid", "ask", "tick_volume", "real_volume",
                    "spread")


def tick_csv_name(day: str) -> str:
    """File name for one UTC day (``day`` = ``YYYYMMDD``)."""
    return f"mt5_ticks_{day}.csv"


class TickLogger:
    """Append every processed tick to its UTC-day CSV.

    One open handle per day, rotated on date change, flushed per
    write.  Logging never raises into the tick loop — callers wrap
    in try/except and the logger itself guards the write.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._day: str | None = None
        self._fh = None
        self._writer = None
        self.rows = 0

    def _open_day(self, day: str) -> None:
        if self._day == day and self._fh is not None:
            return
        self.close()
        path = self.directory / tick_csv_name(day)
        new_file = not path.exists()
        self._fh = path.open("a", newline="")
        self._writer = csv.writer(self._fh)
        if new_file:
            self._writer.writerow(TICK_CSV_COLUMNS)
        self._day = day

    def log(self, tick: dict) -> bool:
        """Append one tick dict.  Returns False (never raises) on bad input."""
        try:
            ts = tick.get("time")
            if ts is None:
                return False
            if isinstance(ts, datetime):
                day = ts.astimezone(timezone.utc).strftime("%Y%m%d")
                ts_txt = ts.astimezone(timezone.utc).isoformat()
            else:
                ts_txt = str(ts)
                day = ts_txt[:10].replace("-", "")
            bid = float(tick.get("bid", 0.0))
            ask = float(tick.get("ask", 0.0))
            spread = tick.get("spread")
            if spread is None:
                spread = ask - bid
            self._open_day(day)
            assert self._writer is not None
            self._writer.writerow([
                ts_txt, f"{bid:.5f}", f"{ask:.5f}",
                tick.get("tick_volume", 0),
                tick.get("real_volume", 0),
                f"{float(spread):.5f}",
            ])
            self._fh.flush()
            self.rows += 1
            return True
        except Exception:
            return False

    def log_many(self, ticks: Iterable[dict]) -> int:
        """Log an iterable of ticks; returns rows written."""
        return sum(1 for t in ticks if self.log(t))

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = None
        self._writer = None
        self._day = None


def normalize_tick_frame(df):
    """Terminal tick frame -> the canonical CSV columns.

    ``copy_ticks_range`` rows carry ``time, bid, ask, volume, flags``
    (and ``volume_real`` when the terminal reports it).  ``volume``
    is tick volume; ``spread`` is recomputed as ask - bid.
    """
    import pandas as pd

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=list(TICK_CSV_COLUMNS))
    out = pd.DataFrame()
    times = pd.to_datetime(df["time"], utc=True)
    out["time"] = times.dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    out["bid"] = df["bid"].astype(float)
    out["ask"] = df["ask"].astype(float)
    out["tick_volume"] = df["volume"].astype(float) if "volume" in df else 0
    out["real_volume"] = df["volume_real"].astype(float) \
        if "volume_real" in df else 0
    out["spread"] = out["ask"] - out["bid"]
    return out[list(TICK_CSV_COLUMNS)]


def export_tick_history(feed, start: datetime, end: datetime,
                        out_path: str | Path) -> tuple[Path, int]:
    """Pull terminal tick history into one CSV. Returns (path, rows)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = feed.ticks_range(start, end)
    norm = normalize_tick_frame(df)
    norm.to_csv(out_path, index=False)
    return out_path, len(norm)
