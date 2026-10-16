"""Native per-timeframe data loader for the MT5 full-engine backtest.

OWNER RULE (2026-10-16): no resampling. Every timeframe's bars must
come from that timeframe's own file — MT5 ``copy_rates`` output for
that TF, or that vendor's own per-TF file (OANDA's H4 file is OANDA's
own H4 bars, NOT H1 bars aggregated by us) — or be aggregated from
the MT5 tick stream (flagged ``built_from_ticks``). A timeframe with
no file AND no ticks is MISSING, and loading raises
:class:`MissingNativeData` with auto-fetch diagnostics. There is no
manual CSV step: the backtest fetches from the running MT5 terminal
itself. This module contains no resampling call of any kind.

Why this matters: resampling H1 -> H4/D1/W1/MN1 fabricates TF
structure. Real H4/D1 bars have their own opens/closes (broker
session boundaries, Sunday candles, gaps) that H1 aggregation does
not reproduce. A backtest on resampled HTF is a backtest of a
simplified model, not of the engine.

Tick provenance: MT5 ticks are a SINGLE stream (``copy_ticks_range``),
not one stream per TF. TF bars are MT5's own per-TF ``copy_rates``.
``load_native_ticks`` reads an MT5 tick CSV when present; otherwise
it raises. ``ticks_from_m1`` is a labelled APPROXIMATION (M1
OHLC -> 4 ticks) for smoke runs only — it must be opted into
explicitly and is always flagged ``ticks: m1_proxy`` in any report.
It is never presented as real MT5 ticks.

Tick-aggregation fallback (2026-10-16): the terminal often has no
cached bars for a window (M1 3y is ~1M bars; manual history
download did not fix it and H1 STILL came back empty), even though it
HAS the tick stream. When ANY TF's bars are missing but ticks are
present, ``build_bars_from_ticks`` aggregates the tick stream
itself into that TF's OHLC bars (open = first bid, high = max bid,
low = min bid, close = last bid, per bucket). This is NOT
resampling from H1: it aggregates the finest-grained source
(ticks — the same stream MT5 itself builds bars from), never a
coarser bar series. The provenance flags every such TF as
``built_from_ticks``. If ticks are missing too, loading still
blocks. HTF buckets are aligned to the broker server timezone
when the terminal reports an offset (``server_utc_offset_hours``);
without an offset they fall back to UTC buckets, flagged the same
way — the provenance never hides which TFs were tick-built.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

#: Timeframes the full-engine backtest requires, finest first for the
#: tick proxy. M30 is an MT5 timeframe (the live feed serves it); it
#: has no per-TF file in the repo either.
REQUIRED_TFS = ("MN1", "W1", "D1", "H4", "H1", "M30", "M15", "M5", "M1")

#: Gap that splits a TF into segments (per TF, in bar units is wrong
#: across TFs — use wall-clock multiples of the TF length).
GAP_MULTIPLE = 3.0

#: TF length in minutes (MN1 approximated as 43200 = 30d).
TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60,
              "H4": 240, "D1": 1440, "W1": 10080, "MN1": 43200}

#: Default broker symbol for native files. The owner's terminal
#: is Exness, where gold is listed as ``XAUUSDm`` — plain ``XAUUSD``
#: does not exist there. ``--symbol`` overrides; the loader takes
#: the same symbol so fetch cache and loader never disagree.
DEFAULT_SYMBOL = "XAUUSDm"

#: EVERY timeframe may be aggregated from the MT5 tick stream
#: when the terminal has no cached bars for it. Ticks are the
#: source truth and they contain the session boundaries; blocking
#: on missing native HTF history (the owner tried manual history
#: download and H1 still came back empty) was the bug, not missing data.
#: Tick-built TFs are always flagged ``built_from_ticks`` in the
#: provenance — never hidden.
BUILDABLE_FROM_TICKS = ("M1", "M5", "M15", "M30",
                        "H1", "H4", "D1", "W1", "MN1")


def _bucket_times(times: pd.Series, tf: str,
                  offset_hours: float = 0.0) -> pd.Series:
    """Bucket boundaries for ``tf`` in broker server time.

    ``times`` are absolute (UTC) tick timestamps. MT5 bars are
    stamped in server local time, so the bucket grid is computed
    on ``times + offset`` and shifted back. Without an offset the
    grid is UTC. H4 buckets are 00/04/08/... server time, D1 is
    server midnight, W1 is Monday 00:00 server time, MN1 is the
    1st 00:00 server time.
    """
    shifted = times + pd.Timedelta(hours=float(offset_hours))
    if tf == "W1":
        bucket = (shifted - pd.to_timedelta(
            shifted.dt.weekday, unit="D")).dt.floor("D")
    elif tf == "MN1":
        bucket = (shifted - pd.to_timedelta(
            shifted.dt.day - 1, unit="D")).dt.floor("D")
    elif tf == "D1":
        bucket = shifted.dt.floor("D")
    else:
        bucket = shifted.dt.floor(f"{TF_MINUTES[tf]}min")
    return bucket - pd.Timedelta(hours=float(offset_hours))


def build_bars_from_ticks(ticks_df: pd.DataFrame, tf: str,
                          server_utc_offset_hours: float = 0.0
                          ) -> pd.DataFrame:
    """Aggregate an MT5 tick stream into OHLC bars for one TF.

    ``ticks_df`` needs ``time`` (datetime), ``bid``, ``ask`` and
    optionally ``volume`` / ``real_volume``. Bars are Bid-based,
    like MT5 chart bars: open = first bid in the bucket, high =
    max bid, low = min bid, close = last bid. ``volume`` is the
    summed tick volume; ``spread`` is the mean ask-bid in the
    bucket. Buckets are aligned to broker server time (see
    :func:`_bucket_times`) — a bucket contains only ticks inside
    it, so aggregation is causal with no lookahead. Time buckets
    with no ticks produce no bar (the terminal likewise has no
    bar without ticks).
    This aggregates the tick stream itself — it never touches a
    coarser bar series, so it is not H1 resampling by another
    name. Any TF may be built this way; the provenance flags it
    ``built_from_ticks``. Building H4/W1/MN1 from ticks with a
    UTC fallback grid (no server offset reported) can shift
    bucket boundaries vs the terminal's own bars — that is why
    the flag exists and why native bars are preferred whenever
    the terminal has them.
    """
    if tf not in BUILDABLE_FROM_TICKS:
        raise ValueError(
            f"{tf} is not a known timeframe — buildable TFs are "
            f"{BUILDABLE_FROM_TICKS}.")
    if ticks_df is None or len(ticks_df) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close",
                                     "volume", "real_volume", "spread"])
    df = ticks_df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time")
    if "bid" not in df.columns:
        if "mid" not in df.columns:
            raise ValueError(
                "build_bars_from_ticks needs bid (or mid)")
        df["bid"] = df["mid"]
    if "ask" not in df.columns:
        df["ask"] = df["bid"]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    df["spread_tick"] = df["ask"].astype(float) - df["bid"].astype(float)
    df["bucket"] = _bucket_times(df["time"], tf,
                                 server_utc_offset_hours)
    grouped = df.groupby("bucket")
    bars = pd.DataFrame({
        "open": grouped["bid"].first(),
        "high": grouped["bid"].max(),
        "low": grouped["bid"].min(),
        "close": grouped["bid"].last(),
        "volume": grouped["volume"].sum(),
        "spread": grouped["spread_tick"].mean(),
    })
    if "real_volume" in df.columns:
        bars["real_volume"] = grouped["real_volume"].sum()
    else:
        bars["real_volume"] = 0.0
    bars.index.name = "timestamp"
    bars = bars[["open", "high", "low", "close", "volume",
                 "real_volume", "spread"]].sort_index()
    return bars


def _tf_candidates(symbol: str) -> dict[str, tuple[str, ...]]:
    """Candidate files per TF for ``symbol``, best first.

    MT5 native first, then the vendor's own per-TF file. NEVER a
    file of another TF. W1 / MN1 / M30 have no vendor file: no
    file exists in the repo.
    """
    return {
        "D1": (f"MT5_{symbol}_D1_native*.csv",
               f"MT5_{symbol}_D1_from_ticks*.csv",),
        "H4": (f"MT5_{symbol}_H4_native*.csv",
               f"MT5_{symbol}_H4_from_ticks*.csv",
               "OANDA_XAU_USD_H4_*.csv"),
        "H1": (f"MT5_{symbol}_H1_native*.csv",
               f"MT5_{symbol}_H1_from_ticks*.csv",
               "OANDA_XAU_USD_H1_*.csv"),
        "M15": (f"MT5_{symbol}_M15_native*.csv",
                f"MT5_{symbol}_M15_from_ticks*.csv",
                "OANDA_XAU_USD_M15_*.csv"),
        "M5": (f"MT5_{symbol}_M5_native*.csv",
               f"MT5_{symbol}_M5_from_ticks*.csv",
               "OANDA_XAU_USD_M5_*.csv"),
        "M1": (f"MT5_{symbol}_M1_native*.csv",
               f"MT5_{symbol}_M1_from_ticks*.csv",
               "OANDA_XAU_USD_M1_*.csv"),
        "W1": (f"MT5_{symbol}_W1_native*.csv",
               f"MT5_{symbol}_W1_from_ticks*.csv",),
        "MN1": (f"MT5_{symbol}_MN1_native*.csv",
                f"MT5_{symbol}_MN1_from_ticks*.csv",),
        "M30": (f"MT5_{symbol}_M30_native*.csv",
                f"MT5_{symbol}_M30_from_ticks*.csv"),
    }


def _tick_candidates(symbol: str) -> tuple[str, ...]:
    """MT5 tick export candidates (copy_ticks_range CSV)."""
    return (f"MT5_{symbol}_ticks*.csv",)

#: Short note used in missing-data errors. The backtest
#: auto-fetches from the running MT5 terminal
#: (``copy_rates_range`` per TF + ``copy_ticks_range``) — there is
#: no manual CSV step. If data is missing, start MT5, log in, and
#: re-run; the fetch pulls the data itself and reports
#: diagnostics (terminal path / account / probe) when the
#: terminal returns nothing.
AUTO_FETCH_NOTE = (
    "The backtest auto-fetches from the running MT5 terminal "
    "(copy_rates_range per TF + copy_ticks_range) — no manual "
    "CSV step. Start MT5, log in, and re-run; the fetch reports "
    "diagnostics when the terminal returns nothing.")


class MissingNativeData(RuntimeError):
    """A required native TF / tick file is missing — no fallback."""

    def __init__(self, missing: list[str], coverage: dict | None = None):
        self.missing = list(missing)
        self.coverage = coverage or {}
        msg = ("Native backtest data missing for: "
               + ", ".join(self.missing)
               + ". " + AUTO_FETCH_NOTE)
        super().__init__(msg)


@dataclass
class TFCoverage:
    tf: str
    source: str | None = None
    rows: int = 0
    start: str = ""
    end: str = ""
    segments: list[tuple[str, str, int]] = field(default_factory=list)
    gap_days: float = 0.0
    missing: bool = False
    built_from_ticks: bool = False


def _normalize_bars(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize one vendor file to timestamp-indexed OHLCV.

    Accepts OANDA (``ts``) and MT5 (``time``) column names. Drops
    duplicate timestamps (keeps first), sorts, validates OHLC sanity.
    """
    ts_col = "ts" if "ts" in df.columns else (
        "time" if "time" in df.columns else "Datetime")
    if ts_col not in df.columns:
        raise ValueError(f"{source}: no timestamp column "
                         f"(have {list(df.columns)})")
    out = df.rename(columns={ts_col: "timestamp"}).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True,
                                      format="mixed")
    out = out.set_index("timestamp").sort_index()
    out = out[~out.index.duplicated(keep="first")]
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype(float)
    if "volume" not in out.columns:
        out["volume"] = 0.0
    bad = (out["high"] < out[["open", "close"]].max(axis=1) - 1e-9) | (
        out["low"] > out[["open", "close"]].min(axis=1) + 1e-9)
    if bool(bad.any()):
        raise ValueError(f"{source}: {int(bad.sum())} OHLC-violating rows")
    out["real_volume"] = out.get("real_volume", 0.0)
    out["spread"] = out.get("spread", 0.20)
    return out[["open", "high", "low", "close", "volume",
                "real_volume", "spread"]]


def _segments(index: pd.DatetimeIndex, tf: str) -> list[tuple[str, str, int]]:
    """Contiguous segments: a gap > 3x the TF length breaks a segment."""
    if len(index) == 0:
        return []
    step = pd.Timedelta(minutes=TF_MINUTES[tf] * GAP_MULTIPLE)
    breaks = index[1:] - index[:-1] > step
    starts = [0] + [i + 1 for i, b in enumerate(breaks) if b]
    ends = [i for i, b in enumerate(breaks) if b] + [len(index) - 1]
    return [(str(index[s]), str(index[e]), int(e - s + 1))
            for s, e in zip(starts, ends)]


def coverage_report(data_dir: Path | None = None,
                    symbol: str = DEFAULT_SYMBOL) -> dict[str, TFCoverage]:
    """Per-TF coverage from native files only. Never loads a fallback."""
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    candidates = _tf_candidates(symbol)
    out: dict[str, TFCoverage] = {}
    for tf in REQUIRED_TFS:
        cov = TFCoverage(tf=tf, missing=True)
        for pattern in candidates.get(tf, ()):  # pragma: no branch
            matches = sorted(base.glob(pattern))
            if not matches:
                continue
            # Concatenate all matching files for the TF (same TF only).
            frames = [_normalize_bars(pd.read_csv(p), p.name)
                      for p in matches]
            frame = pd.concat(frames)
            frame = frame[~frame.index.duplicated(keep="first")].sort_index()
            cov.source = ", ".join(p.name for p in matches)
            cov.built_from_ticks = "_from_ticks" in pattern
            cov.rows = len(frame)
            cov.start = str(frame.index[0])
            cov.end = str(frame.index[-1])
            cov.segments = _segments(frame.index, tf)
            if len(frame) > 1:
                diffs = (frame.index[1:] - frame.index[:-1])
                cov.gap_days = float(diffs.max().total_seconds() / 86400)
            cov.missing = False
            cov._frame = frame  # type: ignore[attr-defined]
            break
        out[tf] = cov
    return out


def _bar_window_covers(frame: pd.DataFrame, tf: str,
                       start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True iff ``frame``'s bars cover [``start``, ``end``].

    Bars are stamped by OPEN time, so the last bar covers up to
    ``last_open + TF length``. A TF whose native file ends before
    the tick window ends (e.g. a stale file ending Dec 2025 while
    ticks run to Oct 2026) does NOT cover the window — its bars
    are rebuilt from ticks instead of blocking the run.
    """
    if frame is None or len(frame) == 0:
        return False
    first = frame.index[0]
    last = frame.index[-1]
    last_covers_to = last + pd.Timedelta(minutes=TF_MINUTES[tf])
    return bool(first <= start
                and last_covers_to >= end - pd.Timedelta(seconds=1))


def _tick_window_coverage(frames: dict[str, pd.DataFrame],
                          ticks_df: pd.DataFrame
                          ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(tick_start, tick_end) iff every frame covers the tick window.

    Used by the fetch cache check: a cache is sufficient for a
    tick-window run when ticks exist and every TF's bars cover
    [first tick, last tick] — even when the tick window is
    narrower than the originally requested window (broker tick
    history is limited: bars go back to 2023, ticks only from
    Jan 2026).
    """
    if ticks_df is None or len(ticks_df) == 0:
        return None
    tick_start = ticks_df["time"].min()
    tick_end = ticks_df["time"].max()
    if tick_end <= tick_start:
        return None
    for tf, frame in frames.items():
        if not _bar_window_covers(frame, tf, tick_start, tick_end):
            return None
    return (tick_start, tick_end)


def load_native_frames(data_dir: Path | None = None,
                       symbol: str = DEFAULT_SYMBOL,
                       ) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load every required TF from its own native file.

    Run window = the TICK window. Broker tick history is LIMITED
    (bars go back to 2023, ticks only from Jan 2026), so requiring
    every TF to share one identical 3-year window blocks forever.
    Instead: ticks define the run window
    [first tick, last tick]; every TF must cover that window —
    native bars when they cover it, tick-built bars (flagged
    ``built_from_ticks``) when the native file is missing or does
    not cover the tick window (e.g. a stale file ending Dec 2025
    while ticks run to Oct 2026). No TF is ever aggregated from
    H1 bars; tick-built TFs aggregate the tick stream itself.
    Raises MissingNativeData if ticks are missing AND any TF has
    no file — with auto-fetch diagnostics, never a manual
    export instruction.
    """
    cov = coverage_report(data_dir, symbol)
    missing = [tf for tf, c in cov.items() if c.missing]
    provenance = {tf: {"source": c.source, "rows": c.rows,
                       "start": c.start, "end": c.end,
                       "segments": c.segments,
                       "gap_days": round(c.gap_days, 1),
                       "built_from_ticks": bool(c.built_from_ticks)}
                  for tf, c in cov.items()}
    # Ticks once: they define the run window when present.
    try:
        tick_list = load_native_ticks(data_dir, symbol)
    except MissingNativeData:
        tick_list = None
    ticks_df: pd.DataFrame | None = None
    tick_start: pd.Timestamp | None = None
    tick_end: pd.Timestamp | None = None
    if tick_list:
        ticks_df = pd.DataFrame(tick_list).rename(
            columns={"tick_volume": "volume"})
        ticks_df["time"] = pd.to_datetime(ticks_df["time"], utc=True)
        ticks_df = ticks_df.sort_values("time").drop_duplicates("time")
        if len(ticks_df) >= 2:
            tick_start = ticks_df["time"].min()
            tick_end = ticks_df["time"].max()

    if ticks_df is not None and tick_start is not None \
            and tick_end is not None:
        frames: dict[str, pd.DataFrame] = {}
        still_missing: list[str] = []
        for tf in REQUIRED_TFS:
            native = None if cov[tf].missing else \
                cov[tf]._frame.copy()  # type: ignore[attr-defined]
            if native is not None and _bar_window_covers(
                    native, tf, tick_start, tick_end):
                frames[tf] = native
                continue
            # Native missing OR native does not cover the tick
            # window (stale / short file): rebuild from ticks.
            bars = build_bars_from_ticks(ticks_df, tf)
            if len(bars) == 0:
                still_missing.append(tf)
                continue
            frames[tf] = bars
            note = ""
            if native is not None:
                note = (f" — native file existed but did not cover "
                        f"the tick window (native "
                        f"{native.index[0]}..{native.index[-1]})")
            if len(bars) > 1:
                diffs = bars.index[1:] - bars.index[:-1]
                gap = float(diffs.max().total_seconds() / 86400)
            else:
                gap = 0.0
            provenance[tf] = {
                "source": f"built_from_ticks "
                          f"(MT5_{symbol}_ticks.csv){note}",
                "rows": len(bars),
                "start": str(bars.index[0]),
                "end": str(bars.index[-1]),
                "segments": _segments(bars.index, tf),
                "gap_days": round(gap, 1),
                "built_from_ticks": True,
            }
        if still_missing:
            raise MissingNativeData(still_missing, provenance)
        provenance["window"] = {"start": str(tick_start),
                                "end": str(tick_end)}
        provenance["tick_window"] = {
            "start": str(tick_start), "end": str(tick_end),
            "rows": int(len(ticks_df)),
            "source": f"MT5_{symbol}_ticks.csv (copy_ticks_range — "
                      "broker tick history is limited; bars run "
                      "on the tick window)"}
        provenance["tick_coverage"] = (
            f"ticks available {tick_start}..{tick_end}; "
            f"full-engine tick run on the tick window")
        provenance["built_from_ticks"] = sorted(
            tf for tf in REQUIRED_TFS
            if provenance.get(tf, {}).get("built_from_ticks"))
        provenance["resampling"] = (
            "NONE — every TF from its own native file or aggregated "
            "from the MT5 tick stream (never from H1 bars)")
        return frames, provenance

    # No usable ticks: fall back to native files only. A TF with
    # no file cannot be built — raise, with auto-fetch diagnostics.
    if missing:
        raise MissingNativeData(missing, provenance)
    frames = {tf: cov[tf]._frame.copy()  # type: ignore[attr-defined]
              for tf in REQUIRED_TFS}
    # Native-only intersection: the run window when no ticks exist.
    start = max(f.index[0] for f in frames.values())
    end = min(f.index[-1] for f in frames.values())
    if start >= end:
        raise MissingNativeData(
            [f"native TF files do not share a run window "
             f"(latest first bar {start} >= earliest last bar "
             f"{end})"],
            provenance)
    provenance["window"] = {"start": str(start), "end": str(end)}
    provenance["built_from_ticks"] = sorted(
        tf for tf in REQUIRED_TFS
        if provenance.get(tf, {}).get("built_from_ticks"))
    provenance["resampling"] = (
        "NONE — every TF from its own native file or aggregated "
        "from the MT5 tick stream (never from H1 bars)")
    return frames, provenance


def load_native_ticks(data_dir: Path | None = None,
                      symbol: str = DEFAULT_SYMBOL) -> list[dict]:
    """Load real MT5 ticks (copy_ticks_range CSV). Raises if absent."""
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    for pattern in _tick_candidates(symbol):
        matches = sorted(base.glob(pattern))
        if not matches:
            continue
        df = pd.concat([pd.read_csv(p) for p in matches],
                       ignore_index=True)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").drop_duplicates("time")
        ticks = []
        for row in df.itertuples(index=False):
            d = row._asdict()
            bid, ask = float(d["bid"]), float(d["ask"])
            ticks.append({"bid": bid, "ask": ask, "mid": (bid + ask) / 2,
                          "spread": ask - bid, "time": d["time"],
                          "tick_volume": float(d.get("volume", 1.0)),
                          "real_volume": float(d.get("volume", 0.0))})
        return ticks
    raise MissingNativeData([f"ticks (MT5_{symbol}_ticks_*.csv)"])


def ticks_from_m1(m1: pd.DataFrame) -> list[dict]:
    """M1-OHLC -> 4-tick APPROXIMATION. Smoke runs only, never ticks.

    One tick per M1 bar at +0s (open), +15s (high), +30s (low),
    +45s (close). Intra-minute path is invented. Any report using
    these MUST carry ``ticks: m1_proxy`` — never "MT5 ticks".
    """
    ticks: list[dict] = []
    for ts, row in m1.iterrows():
        for off_s, price in ((0, row["open"]), (15, row["high"]),
                             (30, row["low"]), (45, row["close"])):
            t = ts + pd.Timedelta(seconds=int(off_s))
            ticks.append({"bid": float(price) - 0.10,
                          "ask": float(price) + 0.10,
                          "mid": float(price), "spread": 0.20,
                          "time": t, "tick_volume": 1.0,
                          "real_volume": 0.0})
    return ticks
