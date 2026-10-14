#!/usr/bin/env python3
"""Phase 10 methodology: measure each playbook's context-performance map.

For every playbook: where does the R concentrate across session,
killzone, volatility regime, trend, HTF-bias alignment, AMD phase and
day-of-week?  The map is the playbook's hunting ground — data, not
prose.  Prints per-dimension tables and saves the raw maps to
goldenswing/playbooks/context_maps.json.

Usage: python scripts/measure_context_maps.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.data import REPO_ROOT, load_gold_h1, load_peers
from goldenswing.confluence import ConfluenceEngine
from goldenswing.contextmap import (DIMENSIONS, derive_hunting_ground,
                                    measure_map)
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.risk import RiskGuards
from goldenswing.features.engine import read_and_featurize
from goldenswing.playbooks import (FVGMidlinePlaybook, SweepReversalPlaybook,
                                   TrendlineRetestPlaybook)
from goldenswing.playbooks.base import EngineContext


def _book_outcomes(engine: ConfluenceEngine, ctx: EngineContext,
                   pb) -> list:
    """The playbook's traded book: live-book path for the anchor,
    greedy shadow path for sleeves (their live_threshold is 0)."""
    cands = [c for c in pb.candidates(ctx)
             if c.base_score >= pb.live_threshold]
    return ConfluenceEngine._greedy(
        [c.outcome for c in cands], pb.live_threshold)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="context-performance maps")
    ap.add_argument("--out", type=str, default=str(
        REPO_ROOT / "goldenswing" / "playbooks" / "context_maps.json"))
    args = ap.parse_args(argv)

    df = load_gold_h1()
    peers = load_peers()
    states = read_and_featurize(df, peers)
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    ctx = EngineContext(df=df, states=states, years=years,
                        instrument="GC=F")

    playbooks = [SweepReversalPlaybook(), FVGMidlinePlaybook(),
                 TrendlineRetestPlaybook()]
    # engine only needed for the greedy book-builder; risk/journal unused
    tmp = Path(tempfile.mkdtemp())
    engine = ConfluenceEngine(playbooks, RiskGuards(tmp / "r.json"),
                              TradeJournal(tmp / "j.jsonl"))

    all_maps: dict[str, dict] = {}
    for pb in playbooks:
        outs = _book_outcomes(engine, ctx, pb)
        overall = sum(o.r_net for o in outs) / len(outs) if outs else 0.0
        m = measure_map(outs, states)
        hg = derive_hunting_ground(m, overall)
        all_maps[pb.name] = {
            "n_trades": len(outs), "overall_avg_r": overall,
            "map": m, "hunting_ground": hg.to_dict(),
        }
        print(f"\n{'=' * 70}\n{pb.name}: {len(outs)} trades, "
              f"overall avg R {overall:+.3f}")
        for dim in DIMENSIONS:
            print(f"  [{dim}]")
            levels = sorted(m[dim].items(),
                            key=lambda kv: kv[1]["avg_r"])
            for lvl, c in levels:
                flag = ""
                if (dim, lvl) in hg.excluded:
                    flag = "  <-- EXCLUDE (decisively bad)"
                elif (dim, lvl) in hg.bonuses:
                    flag = (f"  <-- PROVEN (+{hg.bonuses[(dim, lvl)]:.0f} "
                            f"bonus)")
                h1 = c["half1_avg_r"]
                h2 = c["half2_avg_r"]
                htxt = (f"h1 {h1:+.3f} h2 {h2:+.3f}"
                        if h1 is not None and h2 is not None else "halves n/a")
                print(f"    {lvl:>14}: n={c['n']:>4} win={c['win_rate']:.0%} "
                      f"avgR={c['avg_r']:+.3f} totR={c['total_r']:+.1f} "
                      f"{htxt}{flag}")
        if not hg.excluded and not hg.bonuses:
            print("  no exclusions, no proven cells — fires everywhere, "
                  "no map bonuses")

    Path(args.out).write_text(json.dumps(all_maps, indent=1))
    print(f"\nmaps saved to {args.out}")


if __name__ == "__main__":
    main()
