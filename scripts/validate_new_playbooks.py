"""Validate the new MT5 playbooks on OANDA XAUUSD H1 (2024-10 -> 2026-10).

Each new setup (V-reversal, retracement trap, inversion, order block,
killzone open) runs its own ``candidates()`` over the full window with
its own entry rules and the shared exit discipline (1.5R target, stop
= invalidation +/- 0.25 ATR, invalidation exit at close, 48-bar max
hold).  The iron rule is applied mechanically:

    PASS (executes):  PF >= 1.10 AND total R > 0 AND trades >= 30
    FAIL (stays shadow): anything else

Halves (first/second half R) are printed for stability: a setup
positive in only one half is suspect even when it passes.

Usage:  python scripts/validate_new_playbooks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from goldenswing.playbooks.base import EngineContext
from goldenswing.playbooks.common import (
    MIN_PF, MIN_TRADES, promotion_verdict, summarize)
from goldenswing.playbooks.inversion import InversionPlaybook
from goldenswing.playbooks.killzone_open import KillzoneOpenPlaybook
from goldenswing.playbooks.order_block import OrderBlockPlaybook
from goldenswing.playbooks.retracement_trap import RetracementTrapPlaybook
from goldenswing.playbooks.v_reversal import VReversalPlaybook

DATA = Path(__file__).resolve().parents[1] / "data" \
    / "OANDA_XAU_USD_H1_20241001_20261012.csv"

PLAYBOOKS = [
    VReversalPlaybook,
    RetracementTrapPlaybook,
    InversionPlaybook,
    OrderBlockPlaybook,
    KillzoneOpenPlaybook,
]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["ts"])
    df = df.set_index("ts").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["open", "high", "low", "close", "volume"]]


def main() -> int:
    df = load()
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400 / 365.25
    ctx = EngineContext(df=df, states=[], years=years,
                        instrument="XAUUSD")
    mid = len(df) // 2
    print(f"bars: {len(df)}  window: {df.index[0].date()} -> "
          f"{df.index[-1].date()}  ({years:.2f} years)")
    print(f"promotion rule: PF >= {MIN_PF} AND total R > 0 AND "
          f"trades >= {MIN_TRADES}")
    print()
    verdicts: dict[str, bool] = {}
    for cls in PLAYBOOKS:
        pb = cls()
        cands = pb.candidates(ctx)
        rs = [c.outcome.r_net for c in cands]
        s = summarize(rs)
        # Halves (by signal bar): stability check, not part of the rule.
        first = [c.outcome.r_net for c in cands
                 if c.signal_index < mid]
        second = [c.outcome.r_net for c in cands
                  if c.signal_index >= mid]
        ok, reason = promotion_verdict(s)
        verdicts[pb.name] = ok
        print(f"{pb.name}: trades={s['trades']} win={s['win_rate']:.0%} "
              f"total_R={s['total_r']:+.1f} PF={s['pf']:.2f} "
              f"avg_R={s['avg_r']:+.2f} maxDD={s['max_dd_r']:.1f}R "
              f"first_half_R={sum(first):+.1f} "
              f"second_half_R={sum(second):+.1f} -> "
              f"{'PASS executes' if ok else 'FAIL stays shadow'} "
              f"({reason})")
    print()
    executing = [n for n, ok in verdicts.items() if ok]
    shadow = [n for n, ok in verdicts.items() if not ok]
    print(f"executing: {executing or 'none'}")
    print(f"shadow: {shadow or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
