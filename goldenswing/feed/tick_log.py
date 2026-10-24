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

#: Default tick-CSV retention: 30 UTC days of tick files.  A month
#: of replay/backtest material; positions held longer are protected
#: separately by TickLogger.prune's keep_since_day (their open day
#: onward is never evicted while they are open).
TICK_CSV_RETAIN_DAYS = 30


def tick_csv_day(name: str) -> str | None:
    """UTC day (``YYYYMMDD``) of a tick-CSV filename, else None.

    Only the logger's own ``mt5_ticks_YYYYMMDD.csv`` shape matches;
    anything else is never a prune candidate.
    """
    if not name.startswith("mt5_ticks_") or not name.endswith(".csv"):
        return None
    day = name[len("mt5_ticks_"):-len(".csv")]
    if len(day) == 8 and day.isdigit():
        try:
            datetime.strptime(day, "%Y%m%d")
        except ValueError:
            return None
        return day
    return None


def _shift_day(day: str, delta_days: int) -> str | None:
    """``YYYYMMDD`` shifted by ``delta_days`` (negative = earlier)."""
    try:
        d = datetime.strptime(day, "%Y%m%d").date()
    except ValueError:
        return None
    from datetime import timedelta
    return (d + timedelta(days=delta_days)).strftime("%Y%m%d")


def tick_csv_name(day: str) -> str:
    """File name for one UTC day (``day`` = ``YYYYMMDD``)."""
    return f"mt5_ticks_{day}.csv"


class TickLogger:
    """Append every processed tick to its UTC-day CSV.

    One open handle per day, rotated on date change, flushed per
    write.  Logging never raises into the tick loop — callers wrap
    in try/except and the logger itself guards the write.

    Retention: the warehouse is BOUNDED, not append-forever.  The
    runner calls :meth:`prune` on UTC day rollover and at startup
    (never implicitly here, so the runner's open-position guard is
    always applied): the newest ``retain_days`` days are kept
    (today included), except today's file and every file from
    ``keep_since_day`` onward (the earliest open position's open
    day — a live position's ticks are never evicted while it is
    open).
    """

    def __init__(self, directory: str | Path, *,
                 retain_days: int = TICK_CSV_RETAIN_DAYS):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.retain_days = max(0, int(retain_days))
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

    def prune(self, *, keep_since_day: str | None = None,
              today: str | None = None) -> list[str]:
        """Delete tick-CSV day files older than retention.

        Keeps: today's file (always — the live record, and the
        currently open handle), every file from ``keep_since_day``
        onward (a live position's open day onward), and the newest
        ``retain_days`` days (today included) otherwise.  A malformed
        ``keep_since_day`` skips pruning entirely (fail-closed:
        never evict on a guessed span).  Never raises; returns the
        deleted filenames for journaling.  Non-tick files are never
        touched.
        """
        try:
            if today is None:
                today = datetime.now(timezone.utc).strftime("%Y%m%d")
            if tick_csv_day(f"mt5_ticks_{today}.csv") is None:
                return []
            if keep_since_day is not None and \
                    tick_csv_day(f"mt5_ticks_{keep_since_day}.csv") \
                    is None:
                return []
            cutoff = _shift_day(today, -(self.retain_days - 1))
            if cutoff is None:
                return []
            protected = {today}
            if self._day is not None:
                protected.add(self._day)
            deleted = []
            for path in sorted(self.directory.glob("mt5_ticks_*.csv")):
                day = tick_csv_day(path.name)
                if day is None or day in protected:
                    continue
                # Either protection keeps the file: within
                # retention, or at/after the earliest open
                # position's day.  keep_since_day only ever
                # EXTENDS retention backward, never shrinks it.
                if day >= cutoff:
                    continue
                if keep_since_day is not None \
                        and day >= keep_since_day:
                    continue
                try:
                    path.unlink()
                except OSError:
                    continue
                deleted.append(path.name)
            return deleted
        except Exception:
            return []

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
