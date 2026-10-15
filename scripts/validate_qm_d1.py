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

NATIVE D1 MODE (--csv PATH): validate on a native MT5 D1 export
instead of resampled bars (see data/MT5_XAUUSD_D1_native_*.NOTES.md).
Native exports can be FRAGMENTED (MT5 history holes): pattern
detection must never run across a hole, so native mode splits the
frame into contiguous segments (a gap > 3 days breaks a segment)
and validates the LONGEST segment only — EXTENDED = the whole
longest segment, PRIMARY = its last 2 years.  Short fragments are
reported as skipped, never validated.  --replay-oct14 additionally
attempts the owner's Oct 14 2026 setup replay on the native file
and reports exactly what the playbook finds (or why it is blocked).

Backtests map WHERE a setup works, never binary kill/adopt: the
script reports the iron-rule verdict (PF >= 1.10 AND total R > 0 AND
>= 30 trades) AND breakdowns by direction, tags (sweep_head /
displacement_break), drift alignment (QM direction vs the 50-bar
drift into the touch — a reversal that runs WITH the longer drift
vs one that fights it), ATR regime (touch-bar ATR vs its median),
and first/second half.  qm_d1 stays SHADOW regardless: promotion is
a separate owner decision.

Usage:  python3 scripts/validate_qm_d1.py [--csv PATH] [--replay-oct14]
"""
from __future__ import annotations

import argparse
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
    return report(label, to_daily(h1))


def report(label: str, d1: pd.DataFrame) -> dict:
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


def load_native_d1(path: Path) -> pd.DataFrame:
    """Native MT5 D1 export: time,open,high,low,close,volume,..."""
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["open", "high", "low", "close", "volume"]]


def contiguous_segments(d1: pd.DataFrame,
                        max_gap_days: int = 3) -> list[pd.DataFrame]:
    """Split a D1 frame at holes.  A gap > ``max_gap_days`` between
    consecutive bars breaks a segment — pattern detection must never
    run across a hole (a 60-bar lookback spanning a 275-day hole is
    meaningless).  Normal weekend gaps are 1-3 days and stay inside
    a segment."""
    gaps = d1.index.to_series().diff().dt.total_seconds() / 86400
    breaks = gaps > max_gap_days
    groups = breaks.cumsum()
    return [g for _, g in d1.groupby(groups)]


def replay_oct14(d1: pd.DataFrame) -> None:
    """Replay the owner's Oct 14 2026 setup on native D1.

    The owner spotted a Daily QM at ~4,133-4,140 that held as
    resistance and dropped to ~4,104.  This runs the playbook on
    the native frame and reports exactly what it finds: pattern
    found? POI in the 4133-4140 zone? limit triggered? filled?
    A pattern needs up to 60 bars of formation history — if that
    history is inside a data hole, the replay is BLOCKED and this
    says so instead of claiming anything about the setup."""
    print("REPLAY Oct 14 2026 setup (owner: QM ~4,133-4,140, drop to ~4,104)")
    target = d1[d1.index.date == pd.Timestamp("2026-10-14").date()]
    if len(target):
        row = target.iloc[0]
        print(f"  Oct 14 D1 bar: O {row['open']:.3f} H {row['high']:.3f} "
              f"L {row['low']:.3f} C {row['close']:.3f} "
              f"(range ${row['high'] - row['low']:.2f})")
    else:
        print("  Oct 14 bar ABSENT from the native file — replay impossible.")
        return
    # Formation history available before Oct 14 within its segment.
    segs = contiguous_segments(d1)
    seg = next((s for s in segs
                if s.index[-1].date() == pd.Timestamp("2026-10-14").date()),
               None)
    if seg is None:
        print("  Oct 14 bar is not in any contiguous segment — replay blocked.")
        return
    n_before = len(seg) - 1  # bars before Oct 14 in the same segment
    print(f"  Bars before Oct 14 in the same contiguous segment: {n_before}")
    if n_before < 60:
        print("  REPLAY BLOCKED: the QM formation history (up to 60 bars) "
              "lies inside a data hole before Oct 11.  The 4-bar "
              "Oct 11-14 fragment cannot support QM detection.  "
              "No claim is made for or against the owner's Oct 14 QM.")
        return
    ctx = EngineContext(df=seg, states=[], years=1.0, instrument="XAUUSD")
    cands = QMD1Playbook().candidates(ctx)
    print(f"  Playbook candidates on the Oct-14 segment: {len(cands)}")
    for c in cands:
        print(f"    dir={c.direction:+d} signal_idx={c.signal_index} "
              f"tags={sorted(c.context.get('tags', []))} "
              f"R={c.outcome.r_net:+.2f} exit={c.outcome.exit_reason}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None,
                    help="native MT5 D1 CSV (skips H1 resampling)")
    ap.add_argument("--replay-oct14", action="store_true",
                    help="attempt the Oct 14 2026 setup replay")
    args = ap.parse_args()
    print(f"promotion rule: PF >= {MIN_PF} AND total R > 0 AND "
          f"trades >= {MIN_TRADES} (iron rule)")
    print("qm_d1 stays SHADOW regardless — promotion is a separate "
          "owner decision.")
    print()
    if args.csv is not None:
        print(f"Daily QM (qm_d1) validation — NATIVE MT5 D1 ({args.csv.name})")
        d1 = load_native_d1(args.csv)
        print(f"  native file: {len(d1)} bars "
              f"{d1.index[0].date()} -> {d1.index[-1].date()}")
        segs = contiguous_segments(d1)
        print(f"  contiguous segments: {len(segs)}")
        for i, s in enumerate(segs):
            print(f"    seg {i}: {len(s)} bars "
                  f"{s.index[0].date()} -> {s.index[-1].date()}")
        main_seg = max(segs, key=len)
        skipped = [s for s in segs if s is not main_seg]
        for s in skipped:
            print(f"  SKIPPED fragment ({len(s)} bars "
                  f"{s.index[0].date()} -> {s.index[-1].date()}): "
                  "too short for QM validation, never run across holes")
        print()
        two_y_ago = main_seg.index[-1] - pd.Timedelta(days=730)
        primary = main_seg[main_seg.index >= two_y_ago]
        report("PRIMARY native D1 (last 2y of longest segment)", primary)
        report("EXTENDED native D1 (longest contiguous segment)", main_seg)
        if args.replay_oct14:
            replay_oct14(d1)
        return 0
    print("Daily QM (qm_d1) validation — H1 resampled to D1")
    validate("PRIMARY 2024-10 -> 2026-10 (~2y)",
             load_h1([H1_PRIMARY]))
    validate("EXTENDED 2022-01 -> 2026-10 (~4y)",
             load_h1([H1_EARLY, H1_PRIMARY]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
