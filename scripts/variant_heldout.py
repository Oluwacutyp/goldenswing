#!/usr/bin/env python3
"""Held-out validation of the sweep_reversal anchor variant.

Variant definition (from experiments/ANCHOR_AUTOPSY.md hunting ground):
  - LONG ONLY (signal.direction == 1)
  - HTF bias alignment == "with-bias" (daily_bias == direction)
  - vol_regime == "expanded"
Everything else is byte-identical to the validated anchor: same scanner
(SweepReversalPlaybook.candidates), same gates (session_scope,
check_context, hunting_ground exclusions), same greedy book at the
validated threshold 60.0, same exits (simulate_signal_v2).

Control: the full anchor, unchanged, on the same held-out window.

Data: XAU/USD H1 2022-01-01 -> 2024-10-01 from Dukascopy (yfinance cannot
serve H1 before ~Oct 2024: 730-day intraday wall). Silver peer XAG/USD
from Dukascopy; dollar peer unavailable (graceful degradation).
This window is SACRED held-out data: no tuning, no parameter changes.

Writes experiments/VARIANT_HELDOUT.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.harness import book_metrics
from goldenswing.confluence import ConfluenceEngine
from goldenswing.contextmap import context_levels
from goldenswing.features.engine import read_and_featurize
from goldenswing.playbooks import SweepReversalPlaybook
from goldenswing.playbooks.base import EngineContext

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "VARIANT_HELDOUT.md"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def build_book(df: pd.DataFrame, peers: dict, variant: bool):
    states = read_and_featurize(df, peers)
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    ctx = EngineContext(df=df, states=states, years=years,
                        instrument="XAUUSD")
    pb = SweepReversalPlaybook()
    cands = pb.candidates(ctx)
    scoped = []
    for c in cands:
        if pb.session_scope != "any":
            t = c.signal_index
            if t >= len(states) or states[t].session != pb.session_scope:
                continue
        ok, _ = pb.check_context(c, ctx)
        if not ok:
            continue
        if pb.hunting_ground.excluded:
            lv = context_levels(states, c.signal_index, c.direction)
            if any(lv.get(d) == l
                   for d, l in pb.hunting_ground.excluded):
                continue
        if variant:
            lv = context_levels(states, c.signal_index, c.direction)
            if c.direction != 1:
                continue
            if lv.get("htf") != "with-bias":
                continue
            if lv.get("vol_regime") != "expanded":
                continue
        scoped.append(c)
    outs = [c.outcome for c in scoped]
    book = ConfluenceEngine._greedy(outs, pb.live_threshold)
    return book, states, years, len(cands), len(scoped)


def frame(book, df, states):
    rows = []
    for o in sorted(book, key=lambda x: x.signal.index):
        t = o.signal.index
        lv = context_levels(states, t, o.signal.direction)
        rows.append({
            "time": df.index[t],
            "year": df.index[t].year,
            "r": o.r_net,
            "win": o.r_net > 0,
            "regime": lv.get("vol_regime"),
            "session": lv.get("session"),
            "exit": getattr(o, "exit_reason", "?"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_csv(REPO / "data" / "XAU_H1_2022_2024.csv")
    silver = load_csv(REPO / "data" / "XAG_H1_2022_2024.csv")
    peers = {"silver": silver, "dollar": None}
    print(f"gold: {df.shape} {df.index[0]} -> {df.index[-1]}", flush=True)
    print(f"silver: {silver.shape}", flush=True)

    # sanity: OHLC integrity + gap check
    bad = df[(df["high"] < df["low"]) | (df["high"] < df[["open", "close"]].max(axis=1)) |
             (df["low"] > df[["open", "close"]].min(axis=1))]
    print(f"bad OHLC bars: {len(bad)}", flush=True)
    gaps = df.index.to_series().diff().dropna()
    big = gaps[gaps > pd.Timedelta(hours=72)]
    print(f"gaps > 72h (beyond weekends): {len(big)}", flush=True)
    if len(big):
        print(big.head(10).to_string())

    results = {}
    for name, variant in [("control", False), ("variant", True)]:
        book, states, years, n_cands, n_scoped = build_book(df, peers, variant)
        m = book_metrics(book, years)
        fr = frame(book, df, states)
        results[name] = (m, fr)
        print(f"{name}: cands={n_cands} scoped={n_scoped} "
              f"trades={m['trades']} net={m['return_pct']:+.2f}% "
              f"PF={m['pf']:.2f} win={m['win_rate']:.1f}% "
              f"avgR={m['avg_r']:+.3f}", flush=True)

    lines = []
    a = lines.append
    a("# Variant Held-Out Validation — sweep_reversal long/with-bias/expanded")
    a("")
    a("**Data:** XAU/USD H1, Dukascopy, 2022-01-01 → 2024-10-01 "
      f"({len(df)} bars, {(df.index[-1]-df.index[0]).days/365.25:.2f}y). "
      "yfinance cannot serve H1 before ~Oct 2024 (730-day intraday wall), "
      "so the task's 2022–2024 GC=F request was fulfilled with the same "
      "underlying market (spot XAU/USD vs GC=F futures — spreads/rolls "
      "differ slightly, price action tracks). Silver peer XAG/USD from "
      "Dukascopy; dollar peer unavailable (graceful degradation, "
      "documented). No overlap with the 2024-10→2026-10 development window.")
    a("")
    a("**Variant definition** (from ANCHOR_AUTOPSY.md hunting ground): "
      "LONG ONLY + HTF bias alignment == 'with-bias' + vol_regime == "
      "'expanded'. Scanner, gates, greedy book at validated threshold 60.0, "
      "and exits are byte-identical to the validated anchor. The ONLY "
      "changes are the three context restrictions.")
    a("")
    a("## Headline")
    a("")
    a("| Book | Trades | Net % | PF | Win% | Avg R | MaxDD% |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for name in ("control", "variant"):
        m, _ = results[name]
        a(f"| {name} | {m['trades']} | {m['return_pct']:+.2f} | "
          f"{m['pf']:.2f} | {m['win_rate']:.1f} | {m['avg_r']:+.3f} | "
          f"{m['max_dd_pct']:.1f} |")
    a("")
    for name in ("control", "variant"):
        m, fr = results[name]
        a(f"## {name} — by year")
        a("")
        a("| Year | Trades | Net R | Avg R | Win% |")
        a("|---|---:|---:|---:|---:|")
        for yr, g in fr.groupby("year"):
            a(f"| {yr} | {len(g)} | {g['r'].sum():+.1f} | "
              f"{g['r'].mean():+.3f} | {g['win'].mean()*100:.0f} |")
        a("")
        a(f"## {name} — by regime")
        a("")
        a("| Regime | Trades | Net R | Avg R | Win% |")
        a("|---|---:|---:|---:|---:|")
        for rg, g in fr.groupby("regime"):
            a(f"| {rg} | {len(g)} | {g['r'].sum():+.1f} | "
              f"{g['r'].mean():+.3f} | {g['win'].mean()*100:.0f} |")
        a("")
        a(f"## {name} — by session")
        a("")
        a("| Session | Trades | Net R | Avg R | Win% |")
        a("|---|---:|---:|---:|---:|")
        for ss, g in fr.groupby("session"):
            a(f"| {ss} | {len(g)} | {g['r'].sum():+.1f} | "
              f"{g['r'].mean():+.3f} | {g['win'].mean()*100:.0f} |")
        a("")
    m_v, fr_v = results["variant"]
    m_c, fr_c = results["control"]
    a("## Verdict")
    a("")
    if m_v["trades"] >= 50 and m_v["avg_r"] > 0 and m_v["pf"] > 1.2:
        a(f"**VARIANT HOLDS on held-out data.** {m_v['trades']} trades "
          f"(≥50-trade bar cleared), avg {m_v['avg_r']:+.3f}R/trade, "
          f"PF {m_v['pf']:.2f}. The hunting ground replicates outside the "
          "development window. Next: paper-loop sleeve candidacy per the "
          "promotion rules (live_signal parity, 50-trade bar already met "
          "here — paper OOS still required before demo).")
    elif m_v["trades"] >= 50:
        a(f"**VARIANT KILLED AS TESTED.** {m_v['trades']} trades clear the "
          f"count bar, but the edge did not hold: avg {m_v['avg_r']:+.3f}R, "
          f"PF {m_v['pf']:.2f}. The hunting ground does not replicate on "
          "2022–2024 data. No variant, no promotion.")
    else:
        a(f"**VARIANT INCONCLUSIVE — below the 50-trade bar.** Only "
          f"{m_v['trades']} trades on 2.75y of held-out data "
          f"(avg {m_v['avg_r']:+.3f}R, PF {m_v['pf']:.2f}). Cannot promote "
          "or kill on this count; the hunting ground is too thin to "
          "validate here.")
    a("")
    a("Control (full anchor, unchanged) on the same window: "
      f"{m_c['trades']} trades, {m_c['return_pct']:+.2f}%, PF {m_c['pf']:.2f}, "
      f"avg {m_c['avg_r']:+.3f}R — for the 'does the whole thing work here' "
      "question.")
    a("")
    a("**Method notes:** no tuning was performed against this window; all "
      "parameters are the validated anchor's. Spot-vs-futures and "
      "missing-dollar-peer are the documented deviations from the "
      "development setup.")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
