#!/usr/bin/env python3
"""No-lookahead tick replay for GoldenSwing Hot Scalp (research only).

Input candle CSVs must be stamped with candle OPEN times in UTC and contain
open/high/low/close; volume is optional. Tick CSV must contain timestamp/time,
bid and ask. This runner never connects to MT5 and never sends orders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goldenswing.live.hot_scalp import HotScalpConfig, HotScalpResearchEngine
from goldenswing.live.setup_store import SetupStore

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15}


def _time_column(columns) -> str:
    aliases = ("timestamp", "time", "datetime", "date", "ts")
    lowered = {str(c).strip().lower(): str(c) for c in columns}
    for name in aliases:
        if name in lowered:
            return lowered[name]
    raise ValueError("CSV needs a timestamp/time/datetime/date/ts column")


def _read_frame(path: Path, tf: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    tcol = _time_column(df.columns)
    df.index = pd.to_datetime(df.pop(tcol), utc=True, errors="raise")
    df.index.name = "time"
    lowered = {str(c).strip().lower(): str(c) for c in df.columns}
    missing = [c for c in ("open", "high", "low", "close") if c not in lowered]
    if missing:
        raise ValueError(f"{path}: missing OHLC columns {missing}")
    df = df.rename(columns={lowered[k]: k for k in ("open", "high", "low", "close")})
    if "volume" not in df.columns:
        df["volume"] = 0.0
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if not len(df):
        raise ValueError(f"{path}: contains no candles")
    # Explicitly reject impossible or malformed bars instead of training on them.
    if ((df["high"] < df[["open", "close", "low"]].max(axis=1)) |
            (df["low"] > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError(f"{path}: contains invalid OHLC ranges")
    return df


def _read_ticks(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    tcol = _time_column(df.columns)
    lower = {str(c).strip().lower(): str(c) for c in df.columns}
    for col in ("bid", "ask"):
        if col not in lower:
            raise ValueError(f"{path}: missing {col} column")
    df["time"] = pd.to_datetime(df.pop(tcol), utc=True, errors="raise")
    df["bid"] = pd.to_numeric(df[lower["bid"]], errors="raise")
    df["ask"] = pd.to_numeric(df[lower["ask"]], errors="raise")
    df = df.sort_values("time", kind="stable")
    df = df[(df["bid"] > 0) & (df["ask"] >= df["bid"])]
    if not len(df):
        raise ValueError(f"{path}: no valid bid/ask ticks")
    return df[["time", "bid", "ask"]].reset_index(drop=True)


def _closed_asof(frame: pd.DataFrame, tf: str, now: pd.Timestamp,
                 max_rows: int = 3000) -> pd.DataFrame:
    """Only expose candles whose open + timeframe <= current tick time."""
    cutoff = now - pd.Timedelta(minutes=TF_MINUTES[tf])
    end = int(frame.index.searchsorted(cutoff, side="right"))
    return frame.iloc[max(0, end - max_rows):end]


def replay(*, symbol: str, ticks: pd.DataFrame, frames: dict[str, pd.DataFrame],
           store_path: Path, output_path: Path, max_ticks: int | None = None) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    # Step 3 (clock provenance): ticks stamped in the future are
    # impossible history.  The 2026-10-26 harvests carried tick
    # files stamped AFTER their harvest times — quarantined from
    # training/replay.  Fail closed instead of replaying them; a
    # small tolerance covers broker/local clock skew.
    _future_cutoff = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5)
    future = ticks[ticks["time"] > _future_cutoff]
    if len(future):
        raise ValueError(
            f"refusing {len(future)} future-dated tick(s) "
            f"(first {future['time'].iloc[0]}, "
            f"last {future['time'].iloc[-1]}; wall now "
            f"{pd.Timestamp.now(tz='UTC')}) — ticks stamped after "
            f"the present cannot be history; check clock provenance")
    engine = HotScalpResearchEngine(
        SetupStore(store_path), HotScalpConfig(enabled=True),
        instance_id="historical-shadow-replay")
    counts: dict[str, int] = {}
    processed = 0
    with output_path.open("a", encoding="utf-8") as out:
        for row in ticks.itertuples(index=False):
            now = pd.Timestamp(row.time)
            available = {tf: _closed_asof(frame, tf, now)
                         for tf, frame in frames.items()}
            tick = {"time": now.isoformat(), "bid": float(row.bid),
                    "ask": float(row.ask), "mid": (float(row.bid) + float(row.ask)) / 2.0}
            events = engine.on_tick(symbol=symbol, tick=tick, frames=available, now=now)
            for event in events:
                event_name = str(event.get("event", "unknown"))
                counts[event_name] = counts.get(event_name, 0) + 1
                out.write(json.dumps({"timestamp": now.isoformat(),
                                      "symbol": symbol,
                                      "execution_authority": "shadow_only",
                                      "event": event}, sort_keys=True,
                                     default=str) + "\n")
            processed += 1
            if max_ticks and processed >= max_ticks:
                break
    return {"symbol": symbol, "ticks_processed": processed,
            "event_counts": counts, "output_path": str(output_path),
            "store_path": str(store_path), "execution_authority": "shadow_only",
            "tick_first": (str(ticks["time"].iloc[0])
                           if len(ticks) else None),
            "tick_last": (str(ticks["time"].iloc[-1])
                          if len(ticks) else None)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--m5", type=Path, required=True)
    parser.add_argument("--m15", type=Path, required=True)
    parser.add_argument("--ticks", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True,
                        help="SQLite state path; use a fresh file for a clean replay")
    parser.add_argument("--out", type=Path, required=True,
                        help="JSONL event output (appended, not overwritten)")
    parser.add_argument("--max-ticks", type=int)
    args = parser.parse_args(argv)
    try:
        frames = {"M1": _read_frame(args.m1, "M1"),
                  "M5": _read_frame(args.m5, "M5"),
                  "M15": _read_frame(args.m15, "M15")}
        ticks = _read_ticks(args.ticks)
        result = replay(symbol=args.symbol, ticks=ticks, frames=frames,
                        store_path=args.store, output_path=args.out,
                        max_ticks=args.max_ticks)
    except (OSError, ValueError, KeyError) as exc:
        print(f"replay refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
