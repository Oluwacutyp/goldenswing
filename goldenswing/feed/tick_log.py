"""Per-day tick CSV logging + tick-history export for the MT5 engine.

Ticks are logged EXACTLY as seen from the terminal — no resampling,
no fabrication, no filling.  One file per UTC day::

    data/mt5_ticks_YYYYMMDD.csv   (symbol,time,bid,ask,tick_volume,real_volume,spread)

Warehouse integrity (§3.11, from the 2026-10-27 finding): the day
file must be ONE LABELED, MONOTONIC series.  That day's file mixed
two price series with no symbol column (four rows stamped 19:40:00
carrying bids 4002.79–4003.02 AND 4062.72, the live-trade price),
and concatenated an evening segment with a full day starting at
midnight (one inversion) — not replay-grade.  The logger therefore
enforces, at the write path:

- every row is stamped with the logger's symbol, and a tick
  carrying a DIFFERENT symbol is refused (never mixed in);
- a tick older than the day's last written tick is refused
  (monotonic append — the tail is seeded from the existing file
  on open, so a restart cannot append an inversion either);
- a same-timestamp print farther than
  :data:`DIVERGENT_DUPLICATE_FRACTION` from the last print at that
  timestamp is refused (a divergent duplicate — one series does
  not jump $60 inside one timestamp while its neighbours move
  $0.23).

Refusals are counted on the logger (``dropped_foreign`` /
``dropped_out_of_order`` / ``dropped_divergent``), never silent,
and never raise into the tick loop.  A day file keeps the column
shape it was created with for its whole life (legacy files retire
via the retention bound); rows are written in the file's own
column order and the guards apply to legacy files identically.

``export_tick_history`` pulls N days of terminal tick history via
``copy_ticks_range`` (``feed.ticks_range``) into the same CSV shape for
backtesting.  History rows carry the terminal's own tick volume;
``real_volume`` is 0 when the terminal does not report it (most
metals) — callers must check, never assume.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

TICK_CSV_COLUMNS = ("symbol", "time", "bid", "ask", "tick_volume",
                    "real_volume", "spread")

#: The pre-integrity column layout (no symbol label).  Files
#: created before the fix keep this shape until retention retires
#: them; the logger detects it from the header on open.
LEGACY_TICK_CSV_COLUMNS = ("time", "bid", "ask", "tick_volume",
                           "real_volume", "spread")

#: Same-timestamp divergence bound: a print at the same timestamp
#: as the previous row whose bid is farther than this fraction of
#: price (floored absolutely) is not the same series.  An integrity
#: bound for the warehouse, not a market claim — live management
#: never reads these CSVs, so a refused print costs nothing
#: operationally.
DIVERGENT_DUPLICATE_FRACTION = 0.0025
DIVERGENT_DUPLICATE_FLOOR = 1.0

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


def _tick_ts_key(ts, ts_txt: str):
    """Comparable UTC instant for a tick timestamp, or None when
    it cannot be parsed (the row is still logged as seen; its
    order simply cannot be judged, so the guards skip it)."""
    try:
        if isinstance(ts, datetime):
            dt = ts
        else:
            dt = datetime.fromisoformat(
                str(ts_txt).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


class TickLogger:
    """Append every processed tick to its UTC-day CSV.

    One open handle per day, rotated on date change, flushed per
    write.  Logging never raises into the tick loop — callers wrap
    in try/except and the logger itself guards the write.

    Integrity (see the module docstring): rows are labeled with
    the logger's ``symbol``; foreign-symbol ticks, out-of-order
    ticks and divergent same-timestamp prints are refused and
    counted (``dropped_foreign`` / ``dropped_out_of_order`` /
    ``dropped_divergent``).  Per-day tail state (last timestamp +
    bid, and the file's own column layout) is seeded from the
    existing file on first open, so the guards hold across
    restarts and on legacy-shaped files.

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
                 symbol: str = "",
                 retain_days: int = TICK_CSV_RETAIN_DAYS):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.symbol = str(symbol or "")
        self.retain_days = max(0, int(retain_days))
        self._day: str | None = None
        self._fh = None
        self._writer = None
        self.rows = 0
        self.dropped_foreign = 0
        self.dropped_out_of_order = 0
        self.dropped_divergent = 0
        # Per-day write state, seeded from the file on first open:
        # {"cols": the file's column layout, "key": last written
        # timestamp (aware datetime) or None, "bid": its bid}.
        self._tails: dict[str, dict] = {}

    def _seed_tail(self, day: str, path: Path) -> dict:
        """Read an existing day file's layout + last row so the
        guards continue seamlessly across a restart.  An unreadable
        or empty file starts fresh with the current layout."""
        tail = {"cols": TICK_CSV_COLUMNS, "key": None, "bid": None}
        try:
            if path.exists() and path.stat().st_size > 0:
                rows = [r for r in csv.reader(io.StringIO(
                    path.read_text())) if r]
                if rows:
                    header = tuple(rows[0])
                    if header in (TICK_CSV_COLUMNS,
                                  LEGACY_TICK_CSV_COLUMNS):
                        tail["cols"] = header
                    if len(rows) > 1:
                        last = dict(zip(tail["cols"], rows[-1]))
                        tail["key"] = _tick_ts_key(
                            None, last.get("time", ""))
                        try:
                            tail["bid"] = float(last.get("bid", ""))
                        except (TypeError, ValueError):
                            tail["bid"] = None
        except OSError:
            pass
        self._tails[day] = tail
        return tail

    def _open_day(self, day: str) -> None:
        if self._day == day and self._fh is not None:
            return
        self.close()
        path = self.directory / tick_csv_name(day)
        if day not in self._tails:
            self._seed_tail(day, path)
        new_file = not path.exists() or path.stat().st_size == 0
        self._fh = path.open("a", newline="")
        self._writer = csv.writer(self._fh)
        if new_file:
            self._writer.writerow(TICK_CSV_COLUMNS)
            self._tails[day]["cols"] = TICK_CSV_COLUMNS
        self._day = day

    def log(self, tick: dict) -> bool:
        """Append one tick dict.  Returns False (never raises) on
        bad input or an integrity refusal (foreign symbol,
        out-of-order, divergent duplicate — each counted)."""
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
            tick_symbol = tick.get("symbol")
            if self.symbol and tick_symbol \
                    and str(tick_symbol) != self.symbol:
                self.dropped_foreign += 1
                return False
            self._open_day(day)
            assert self._writer is not None
            tail = self._tails[day]
            key = _tick_ts_key(ts, ts_txt)
            if key is not None and tail["key"] is not None:
                if key < tail["key"]:
                    self.dropped_out_of_order += 1
                    return False
                if key == tail["key"] and tail["bid"] is not None:
                    bound = max(
                        DIVERGENT_DUPLICATE_FLOOR,
                        DIVERGENT_DUPLICATE_FRACTION
                        * abs(tail["bid"]))
                    if abs(bid - tail["bid"]) > bound:
                        self.dropped_divergent += 1
                        return False
            values = {
                "symbol": self.symbol,
                "time": ts_txt,
                "bid": f"{bid:.5f}",
                "ask": f"{ask:.5f}",
                "tick_volume": tick.get("tick_volume", 0),
                "real_volume": tick.get("real_volume", 0),
                "spread": f"{float(spread):.5f}",
            }
            self._writer.writerow(
                [values.get(col, "") for col in tail["cols"]])
            self._fh.flush()
            if key is not None:
                tail["key"] = key
                tail["bid"] = bid
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


def normalize_tick_frame(df, symbol: str = ""):
    """Terminal tick frame -> the canonical CSV columns.

    ``copy_ticks_range`` rows carry ``time, bid, ask, volume, flags``
    (and ``volume_real`` when the terminal reports it).  ``volume``
    is tick volume; ``spread`` is recomputed as ask - bid.  Every
    row is labeled with ``symbol`` (the feed's own symbol when the
    caller passes it) — the same labeled shape the live warehouse
    writes.
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
    # Assigned last, once the frame's index exists: a scalar
    # assigned to an empty frame creates a zero-length column
    # that the Series assignments then NaN out.
    out["symbol"] = str(symbol or "")
    return out[list(TICK_CSV_COLUMNS)]


def export_tick_history(feed, start: datetime, end: datetime,
                        out_path: str | Path) -> tuple[Path, int]:
    """Pull terminal tick history into one CSV. Returns (path, rows)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = feed.ticks_range(start, end)
    symbol = getattr(getattr(feed, "cfg", None), "symbol", "") \
        or getattr(feed, "symbol", "") or ""
    norm = normalize_tick_frame(df, symbol=symbol)
    norm.to_csv(out_path, index=False)
    return out_path, len(norm)
