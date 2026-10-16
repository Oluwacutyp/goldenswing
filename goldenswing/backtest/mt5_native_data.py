"""Native per-timeframe data loader for the MT5 full-engine backtest.

OWNER RULE (2026-10-16): no resampling. Every timeframe's bars must
come from that timeframe's own file — MT5 ``copy_rates`` output for
that TF, or that vendor's own per-TF file (OANDA's H4 file is OANDA's
own H4 bars, NOT H1 bars aggregated by us). A timeframe with no file
is MISSING, and loading raises :class:`MissingNativeData` with the
exact MT5 export needed. There is no fallback, no silent aggregation,
and this module contains no resampling call of any kind.

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


def _tf_candidates(symbol: str) -> dict[str, tuple[str, ...]]:
    """Candidate files per TF for ``symbol``, best first.

    MT5 native first, then the vendor's own per-TF file. NEVER a
    file of another TF. W1 / MN1 / M30 have no vendor file: no
    file exists in the repo.
    """
    return {
        "D1": (f"MT5_{symbol}_D1_native*.csv",),
        "H4": (f"MT5_{symbol}_H4_native*.csv",
               "OANDA_XAU_USD_H4_*.csv"),
        "H1": (f"MT5_{symbol}_H1_native*.csv",
               "OANDA_XAU_USD_H1_*.csv"),
        "M15": (f"MT5_{symbol}_M15_native*.csv",
                "OANDA_XAU_USD_M15_*.csv"),
        "M5": (f"MT5_{symbol}_M5_native*.csv",
               "OANDA_XAU_USD_M5_*.csv"),
        "M1": (f"MT5_{symbol}_M1_native*.csv",
               "OANDA_XAU_USD_M1_*.csv"),
        "W1": (f"MT5_{symbol}_W1_native*.csv",),
        "MN1": (f"MT5_{symbol}_MN1_native*.csv",),
        "M30": (f"MT5_{symbol}_M30_native*.csv",),
    }


def _tick_candidates(symbol: str) -> tuple[str, ...]:
    """MT5 tick export candidates (copy_ticks_range CSV)."""
    return (f"MT5_{symbol}_ticks*.csv",)

MT5_EXPORT_SPEC = """\
MT5 EXPORT NEEDED — one common window, every timeframe native.

On the EC2 MT5 terminal (or any terminal with full history), run
MetaTrader5 Python (pip install MetaTrader5) with this script:

    import MetaTrader5 as mt5
    import pandas as pd
    mt5.initialize()
    SYMBOL = "XAUUSDm"  # broker symbol on that terminal (Exness)
    FROM = pd.Timestamp("2024-10-01", tz="UTC")
    TO = pd.Timestamp("2026-10-12", tz="UTC")
    TFS = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
           "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
           "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
           "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
           "MN1": mt5.TIMEFRAME_MN1}
    for name, tf in TFS.items():
        rates = mt5.copy_rates_range(SYMBOL, tf, FROM.to_pydatetime(),
                                     TO.to_pydatetime())
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.to_csv(f"MT5_XAUUSDm_{name}_native.csv", index=False)
    # Ticks (single stream, NOT per-TF):
    ticks = mt5.copy_ticks_range(SYMBOL, FROM.to_pydatetime(),
                                 TO.to_pydatetime(), mt5.COPY_TICKS_ALL)
    tdf = pd.DataFrame(ticks)
    tdf["time"] = pd.to_datetime(tdf["time"], unit="s", utc=True)
    tdf.to_csv("MT5_XAUUSDm_ticks.csv", index=False)

Copy the 10 CSVs to goldenswing/data/ and re-run. Requirements:
- Same window for every TF (common intersection must be non-empty).
- Closed bars only (copy_rates_range returns closed bars).
- Do NOT stitch across history holes: if the terminal returns
  gapped history, the loader reports segments; re-download history
  first (F2 -> download) or export a window inside one segment.
- Ticks: COPY_TICKS_ALL. Without ticks the harness can smoke-run on
  the labelled M1 proxy, but that is NOT a tick backtest.
"""


class MissingNativeData(RuntimeError):
    """A required native TF / tick file is missing — no fallback."""

    def __init__(self, missing: list[str], coverage: dict | None = None):
        self.missing = list(missing)
        self.coverage = coverage or {}
        msg = ("Native backtest data missing for: "
               + ", ".join(self.missing)
               + ". No resampling fallback — export native MT5 bars.\n"
               + MT5_EXPORT_SPEC)
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


def load_native_frames(data_dir: Path | None = None,
                       symbol: str = DEFAULT_SYMBOL,
                       ) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load every required TF from its own native file.

    Returns (frames, provenance). Raises MissingNativeData if ANY
    required TF has no file — no resampling, no partial run.
    """
    cov = coverage_report(data_dir, symbol)
    missing = [tf for tf, c in cov.items() if c.missing]
    provenance = {tf: {"source": c.source, "rows": c.rows,
                       "start": c.start, "end": c.end,
                       "segments": c.segments,
                       "gap_days": round(c.gap_days, 1)}
                  for tf, c in cov.items()}
    if missing:
        raise MissingNativeData(missing, provenance)
    frames = {tf: cov[tf]._frame.copy()  # type: ignore[attr-defined]
              for tf in REQUIRED_TFS}
    # Common intersection must be non-empty, else the run is vacuous.
    start = max(f.index[0] for f in frames.values())
    end = min(f.index[-1] for f in frames.values())
    if start >= end:
        raise MissingNativeData(
            [f"common window empty (latest start {start} >= "
             f"earliest end {end}) — export one common window"],
            provenance)
    provenance["window"] = {"start": str(start), "end": str(end)}
    provenance["resampling"] = "NONE — every TF from its own file"
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
