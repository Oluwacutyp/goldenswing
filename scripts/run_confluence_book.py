#!/usr/bin/env python3
"""Phase 9: run the multi-playbook confluence engine on 2y GC=F H1.

- The live book is the SweepReversal anchor and MUST reproduce the
  validated anchor exactly (+28.92%, 199 trades, PF 1.25) — the canary.
  If it doesn't, the run aborts: something changed in the machinery.
- The FVG-midline and trendline-retest sleeves run in shadow/paper
  mode: scored, logged, graded in their own books, never in live P&L.
  Their books must reproduce the experiment numbers (+25.88%/97 and
  +15.40%/61).
- Risk runs in SHADOW mode: every risk decision is computed and
  logged, but the historical sequence is never altered.

Usage: python scripts/run_confluence_book.py [--journal PATH]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.data import (REPO_ROOT, load_gold_h1, load_peers)
from goldenswing.confluence import ConfluenceEngine
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.risk import RiskGuards
from goldenswing.features.engine import read_and_featurize
from goldenswing.playbooks import (FVGMidlinePlaybook, SweepReversalPlaybook,
                                   TrendlineRetestPlaybook)
from goldenswing.playbooks.base import EngineContext

#: The canary — validated anchor numbers (AOC.md / TRENDLINE_CLAIMS.md).
CANARY = {"trades": 199, "return_pct": 28.92, "pf": 1.25}
#: Shadow-book targets under the CURRENT firing rules (hunting grounds
#: applied).  fvg_midline's experiment (CHoCH_FVG_MID_OB.md) measured
#: 97 / +25.88% UNFILTERED; the shadow book applies the playbook's
#: hunting ground (trend=down excluded — measured context map, and the
#: sequential book improves +25.88% -> +34.61%, so this is not the
#: Phase-5 routing trap).  The experiment report keeps the unfiltered
#: numbers as the validation record; this target is how the sleeve
#: fires NOW.
SHADOW_TARGETS = {
    "fvg_midline": {"trades": 48, "return_pct": 34.61},
    "trendline_retest": {"trades": 61, "return_pct": 15.40},
}
#: Machinery check: raw scanned candidate counts (BEFORE any context
#: gate).  If the scanner drifts, these move even when the filtered
#: books don't — the canary for the machinery itself.
MACHINERY_COUNTS = {
    "sweep_reversal": 467,   # detector emissions at floor 35
    "fvg_midline": 393,      # fills (setups that got filled)
    "trendline_retest": 67,  # retest signals
}


def _close(a: float, b: float, tol: float = 0.005) -> bool:
    return abs(a - b) <= tol


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase-9 confluence book")
    ap.add_argument("--journal", type=str,
                    default=str(REPO_ROOT / "data" / "confluence_journal.jsonl"))
    args = ap.parse_args(argv)

    df = load_gold_h1()
    peers = load_peers()  # the canary lesson: relstr needs its peers
    states = read_and_featurize(df, peers)
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    ctx = EngineContext(df=df, states=states, years=years)

    Path(args.journal).unlink(missing_ok=True)
    journal = TradeJournal(args.journal)
    # Backtest risk state lives in /tmp — production state is untouched.
    risk = RiskGuards(Path(tempfile.mkdtemp()) / "risk_state.json")

    engine = ConfluenceEngine(
        [SweepReversalPlaybook(), FVGMidlinePlaybook(),
         TrendlineRetestPlaybook()],
        risk, journal, risk_mode="shadow")

    print("== MACHINERY (raw scanned candidates, before context gates) ==")
    playbooks = {"sweep_reversal": SweepReversalPlaybook(),
                 "fvg_midline": FVGMidlinePlaybook(),
                 "trendline_retest": TrendlineRetestPlaybook()}
    for name, pb in playbooks.items():
        n = len(pb.candidates(ctx))
        want = MACHINERY_COUNTS[name]
        status = "OK" if n == want else f"MISMATCH (want {want})"
        print(f"  {name}: {n} scanned — {status}")
        assert n == want, f"MACHINERY DRIFT {name}: {n} != {want}"

    rep = engine.run(ctx)

    m = rep.live_metrics
    print("== LIVE BOOK (sweep_reversal anchor) ==")
    print(f"trades {m['trades']} | win {m['win_rate']:.1f}% | "
          f"PF {m['pf']:.2f} | return {m['return_pct']:+.2f}% | "
          f"MaxDD {m['max_dd_pct']:.2f}% | Sharpe {m['sharpe']:.2f} | "
          f"avgR {m['avg_r']:+.3f}")
    assert m["trades"] == CANARY["trades"], \
        f"CANARY FAILED: trades {m['trades']} != {CANARY['trades']}"
    assert _close(m["return_pct"], CANARY["return_pct"]), \
        f"CANARY FAILED: return {m['return_pct']:.2f} != {CANARY['return_pct']}"
    assert _close(m["pf"], CANARY["pf"]), \
        f"CANARY FAILED: PF {m['pf']:.2f} != {CANARY['pf']}"
    print("canary OK: anchor reproduces exactly")

    print("\n== SHADOW BOOKS (paper only) ==")
    for name, tgt in SHADOW_TARGETS.items():
        b = rep.shadow[name]["book"]
        n = rep.shadow[name]["n_candidates"]
        print(f"{name}: {n} candidates -> book trades {b['trades']} | "
              f"win {b['win_rate']:.1f}% | PF {b['pf']:.2f} | "
              f"return {b['return_pct']:+.2f}% | MaxDD {b['max_dd_pct']:.2f}% "
              f"| Sharpe {b['sharpe']:.2f}")
        assert b["trades"] == tgt["trades"], \
            f"SHADOW MISMATCH {name}: trades {b['trades']} != {tgt['trades']}"
        assert _close(b["return_pct"], tgt["return_pct"], tol=0.01), \
            f"SHADOW MISMATCH {name}: return {b['return_pct']:.2f} != " \
            f"{tgt['return_pct']}"
    print("shadow books match the current firing-rule targets "
          "(hunting grounds applied)")

    print("\n== RISK (shadow mode) ==")
    blocked = [r for r in rep.risk_log if not r["risk_would_allow"]]
    print(f"{len(rep.risk_log)} live entries risk-checked, "
          f"{len(blocked)} would have been blocked in enforce mode")
    seen_halt = False
    for r in blocked:
        if "latched" in r["reason"]:
            if seen_halt:
                continue  # one line for the latched halt, not one per bar
            seen_halt = True
        print(f"  bar {r['bar']} {r['playbook']}: {r['reason']}")

    print("\n== SKIPS ==")
    for reason, n in sorted(rep.skips.items()):
        print(f"  {reason}: {n}")
    print(f"\njournal: {rep.journal_path}")


if __name__ == "__main__":
    main()
