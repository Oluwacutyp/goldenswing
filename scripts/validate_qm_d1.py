"""Validate the Daily QM playbook (qm_d1) on resampled daily bars.

No D1 CSV exists in this repo, so daily bars are built by resampling
the OANDA XAUUSD H1 files (open=first, high=max, low=min,
close=last, volume=sum) — the same source the other playbooks are
validated on, one timeframe up.  Resampling is calendar-true: each
daily bar aggregates exactly that UTC day's H1 bars.

Primary window (like the other playbooks): H1 2024-10 -> 2026-10
(~2y, ~500 daily bars).  Extended window: H1 2022-01 -> 2026-10
(~4y, ~1000 daily bars) — reported alongside, because a D1
pattern is rare and the 2y sample is thin.

Backtests map WHERE a setup works, never binary kill/adopt: the
script reports the iron-rule verdict (PF >= 1.10 AND total R > 0 AND
>= 30 trades) AND breakdowns by direction, tags (sweep_head /
displacement_break), drift alignment (QM direction vs the 50-bar
drift into the touch — a reversal that runs WITH the longer drift
vs one that fights it), ATR regime (touch-bar ATR vs its median),
and first/second half.  qm_d1 stays SHADOW regardless: promotion is
a separate owner decision.

Usage:  python3 scripts/validate_qm_d1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from goldenswing.playbooks.base import EngineContext
from goldenswing.playbooks.common import (
    MIN_PF, MIN_TRADES, promotion_verdict, summarize)
from goldenswing.playbooks.qm_d1 import QMD1Playbook

ROOT = Path(__file__).resolve().parents[1]
H1_PRIMARY = ROOT / "data" / "OANDA_XAU_USD_H1_20241001_20261012.csv"
H1_EARLY = ROOT / "data" / "OANDA_XAU_USD_H1_20220101_20241001.csv"


def load_h1(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, parse_dates=["ts"])
        df = df.set_index("ts").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        frames.append(df[["open", "high", "low", "close", "volume"]])
    return pd.concat(frames).sort_index()


def to_daily(h1: pd.DataFrame) -> pd.DataFrame:
    """H1 -> D1 (UTC calendar days).  Days with no H1 bars (weekends /
    holidays) produce no bar — the D1 frame is trading days only."""
    d1 = h1.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum"}).dropna()
    return d1


def _atr_series(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(),
                    (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _breakdown(label: str, cands) -> None:
    if not cands:
        print(f"  {label}: no trades")
        return
    s = summarize([c.outcome.r_net for c in cands])
    print(f"  {label}: trades={s['trades']} win={s['win_rate']:.0%} "
          f"total_R={s['total_r']:+.1f} PF={s['pf']:.2f} "
          f"avg_R={s['avg_r']:+.2f}")


def validate(label: str, h1: pd.DataFrame) -> dict:
    d1 = to_daily(h1)
    years = (d1.index[-1] - d1.index[0]).total_seconds() / 86400 / 365.25
    ctx = EngineContext(df=d1, states=[], years=years,
                        instrument="XAUUSD")
    pb = QMD1Playbook()
    cands = pb.candidates(ctx)
    rs = [c.outcome.r_net for c in cands]
    s = summarize(rs)
    mid = len(d1) // 2
    first = [c.outcome.r_net for c in cands if c.signal_index < mid]
    second = [c.outcome.r_net for c in cands if c.signal_index >= mid]
    ok, reason = promotion_verdict(s)
    print(f"{label}: daily bars={len(d1)} window: "
          f"{d1.index[0].date()} -> {d1.index[-1].date()} "
          f"({years:.2f} years)")
    print(f"  qm_d1: trades={s['trades']} win={s['win_rate']:.0%} "
          f"total_R={s['total_r']:+.1f} PF={s['pf']:.2f} "
          f"avg_R={s['avg_r']:+.2f} maxDD={s['max_dd_r']:.1f}R "
          f"first_half_R={sum(first):+.1f} "
          f"second_half_R={sum(second):+.1f} -> "
          f"{'PASS iron rule' if ok else 'FAIL iron rule'} ({reason})")
    print("  breakdowns (where it works / doesn't):")
    _breakdown("long", [c for c in cands if c.direction == 1])
    _breakdown("short", [c for c in cands if c.direction == -1])
    _breakdown("sweep_head",
               [c for c in cands if "sweep_head" in c.context["tags"]])
    _breakdown("no sweep_head",
               [c for c in cands if "sweep_head" not in c.context["tags"]])
    _breakdown("displacement_break",
               [c for c in cands
                if "displacement_break" in c.context["tags"]])
    _breakdown("no displacement_break",
               [c for c in cands
                if "displacement_break" not in c.context["tags"]])
    # Drift alignment: QM direction vs the 50-bar drift into the
    # touch.  A QM is a reversal — "with drift" means the post-QM
    # move runs WITH the longer drift (the failed rally was the
    # counter-drift leg); "against drift" means it fights it.
    closes = d1["close"].to_numpy(dtype=float)
    with_drift, against_drift = [], []
    for c in cands:
        t = c.signal_index
        if t < 50:
            continue
        drift = np.sign(closes[t] - closes[t - 50])
        (with_drift if drift == c.direction else against_drift).append(c)
    _breakdown("with 50-bar drift", with_drift)
    _breakdown("against 50-bar drift", against_drift)
    # ATR regime: touch-bar ATR vs the median touch-bar ATR.
    atrs = _atr_series(d1).to_numpy(dtype=float)
    touch_atrs = [atrs[c.signal_index] for c in cands]
    med = float(np.median(touch_atrs)) if touch_atrs else 0.0
    _breakdown("ATR >= median (expanded)",
               [c for c in cands if atrs[c.signal_index] >= med])
    _breakdown("ATR < median (compressed)",
               [c for c in cands if atrs[c.signal_index] < med])
    # Exit reasons: how trades actually end (target / stop /
    # invalidation / timeout).
    reasons: dict[str, list[float]] = {}
    for c in cands:
        reasons.setdefault(c.outcome.exit_reason, []).append(
            c.outcome.r_net)
    for reason_name, rlist in sorted(reasons.items()):
        print(f"  exit {reason_name}: n={len(rlist)} "
              f"total_R={sum(rlist):+.1f}")
    print()
    return s


def main() -> int:
    print("Daily QM (qm_d1) validation — H1 resampled to D1")
    print(f"promotion rule: PF >= {MIN_PF} AND total R > 0 AND "
          f"trades >= {MIN_TRADES} (iron rule)")
    print("qm_d1 stays SHADOW regardless — promotion is a separate "
          "owner decision.")
    print()
    validate("PRIMARY 2024-10 -> 2026-10 (~2y)",
             load_h1([H1_PRIMARY]))
    validate("EXTENDED 2022-01 -> 2026-10 (~4y)",
             load_h1([H1_EARLY, H1_PRIMARY]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
