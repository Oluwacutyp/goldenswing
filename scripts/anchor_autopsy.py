#!/usr/bin/env python3
"""Anchor autopsy: where did sweep_reversal's edge live and die?

Diagnosis only — no tuning, no parameter changes.  Rebuilds the anchor's
full-sample traded book at the validated threshold (60.0) and breaks it
down by time, session, regime, HTF, AMD phase, side, weekday, with
direction-vs-timing (MFE) and trap-intelligence overlap.

Writes experiments/ANCHOR_AUTOPSY.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.data import load_gold_h1, load_peers
from goldenswing.backtest.harness import book_metrics
from goldenswing.confluence import ConfluenceEngine
from goldenswing.contextmap import context_levels, DIMENSIONS
from goldenswing.features.engine import read_and_featurize
from goldenswing.features.traps import compute_trap_features
from goldenswing.playbooks import SweepReversalPlaybook
from goldenswing.playbooks.base import EngineContext

OUT = Path(__file__).resolve().parents[1] / "experiments" / "ANCHOR_AUTOPSY.md"


def main() -> None:
    df = load_gold_h1()
    peers = load_peers()
    states = read_and_featurize(df, peers)
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    ctx = EngineContext(df=df, states=states, years=years, instrument="GC=F")
    pb = SweepReversalPlaybook()

    # ---- candidates + the same three gates walkforward applies
    cands = pb.candidates(ctx)
    hg = pb.hunting_ground
    scoped = []
    for c in cands:
        if pb.session_scope != "any":
            t = c.signal_index
            if t >= len(states) or states[t].session != pb.session_scope:
                continue
        ok, _ = pb.check_context(c, ctx)
        if not ok:
            continue
        if hg.excluded:
            lv = context_levels(states, c.signal_index, c.direction)
            if any(lv.get(d) == l for d, l in hg.excluded):
                continue
        scoped.append(c)
    print(f"candidates: {len(cands)}, after gates: {len(scoped)}")

    # ---- traded book at the validated threshold (full sample)
    outs = [c.outcome for c in scoped]
    book = ConfluenceEngine._greedy(outs, pb.live_threshold)
    m = book_metrics(book, years)
    print(f"traded: {m['trades']} trades, return {m['return_pct']:+.2f}%, "
          f"PF {m['pf']:.2f}")

    # ---- per-trade frame
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float)
    L = df["low"].to_numpy(float)
    t0 = df.index[0]
    months = (df.index - t0).days / 30.44
    trap = compute_trap_features(df)
    bait_cols = [c for c in trap.columns if c.startswith("bait_")]
    sfp_cols = [c for c in trap.columns if "sfp" in c.lower()]

    rows = []
    for o in sorted(book, key=lambda x: x.signal.index):
        t = o.signal.index
        d = o.signal.direction
        entry_px = O[o.entry_index]
        risk = abs(entry_px - o.signal.invalidation)
        # MFE in R over the holding period
        if d == 1:
            mfe = float(np.max(d * (H[o.entry_index:o.exit_index + 1] - entry_px) / risk)) if risk > 0 else 0.0
        else:
            mfe = float(np.max(d * (entry_px - L[o.entry_index:o.exit_index + 1]) / risk)) if risk > 0 else 0.0
        ctx_lv = context_levels(states, t, d)
        ts = df.index[t]
        bait = bool(trap[bait_cols[0]].iloc[t] or trap[bait_cols[1]].iloc[t]) if bait_cols else False
        sfp = bool(any(trap[c].iloc[t] for c in sfp_cols)) if sfp_cols else False
        rows.append({
            "month": float(months[t]),
            "month_idx": int(months[t]),
            "ts": ts,
            "side": "long" if d == 1 else "short",
            "r": o.r_net,
            "win": o.r_net > 0,
            "exit": o.exit_reason,
            "hold": o.hold_bars,
            "mfe": mfe,
            "partial": o.partial_taken,
            "wicked": o.wicked_inv,
            "bait": bait,
            "sfp": sfp,
            **{f"ctx_{k}": v for k, v in ctx_lv.items()},
        })
    T = pd.DataFrame(rows)
    T["half"] = np.where(T["month"] < 12, "H1 (mo 0-12)", "H2 (mo 12-24)")

    L = []
    A = L.append
    A("# Anchor Autopsy — sweep_reversal\n")
    A(f"_Data: {len(df)} H1 bars, {df.index[0].date()} → {df.index[-1].date()} "
      f"({years:.2f}y). Traded book at validated threshold "
      f"{pb.live_threshold:.0f}: **{m['trades']} trades, "
      f"{m['return_pct']:+.2f}%, PF {m['pf']:.2f}**._\n")

    # ---- 1. monthly equity
    A("## 1. Where it broke — monthly\n")
    A("| Month | n | R | Win% |")
    A("|---|---:|---:|---:|")
    mg = T.groupby("month_idx").agg(n=("r", "size"), R=("r", "sum"),
                                    w=("win", "mean"))
    for i, r in mg.iterrows():
        A(f"| {i} ({(t0 + pd.offsets.MonthBegin(i)).date() if False else ''}"
          f"{(df.index[0] + pd.DateOffset(months=i)).strftime('%Y-%m')}) "
          f"| {int(r['n'])} | {r['R']:+.1f} | {r['w'] * 100:.0f} |")
    h1 = T[T["half"].str.startswith("H1")]["r"]
    h2 = T[T["half"].str.startswith("H2")]["r"]
    A(f"\n**H1 (mo 0–12): {len(h1)} trades, {h1.sum():+.1f}R, "
      f"avg {h1.mean():+.3f}R/trade, win {h1.gt(0).mean() * 100:.0f}%**")
    A(f"**H2 (mo 12–24): {len(h2)} trades, {h2.sum():+.1f}R, "
      f"avg {h2.mean():+.3f}R/trade, win {h2.gt(0).mean() * 100:.0f}%**\n")

    # cliff vs decay: cumulative R by month + first negative quarter
    A("Monthly cumulative R: " + ", ".join(
        f"{int(i)}:{v:+.0f}" for i, v in mg["R"].cumsum().items()) + "\n")

    # ---- 2..7 dimension tables with half splits
    def cell_table(col: str, title: str):
        A(f"## {title}\n")
        A("| Level | n | avgR | Win% | H1 n/avgR | H2 n/avgR |")
        A("|---|---:|---:|---:|---|---|")
        for lvl, g in T.groupby(col):
            g1 = g[g["half"].str.startswith("H1")]
            g2 = g[g["half"].str.startswith("H2")]
            A(f"| {lvl} | {len(g)} | {g['r'].mean():+.3f} | "
              f"{g['win'].mean() * 100:.0f} | "
              f"{len(g1)}/{g1['r'].mean():+.3f} | "
              f"{len(g2)}/{g2['r'].mean():+.3f} |")
        A("")

    cell_table("ctx_session", "2. Session")
    cell_table("ctx_vol_regime", "3. Volatility regime")
    cell_table("ctx_trend", "4. Trend state")
    cell_table("ctx_htf", "5. HTF bias alignment")
    cell_table("ctx_phase", "6. AMD phase")
    cell_table("side", "7. Side")
    cell_table("ctx_dow", "8. Weekday")
    cell_table("ctx_killzone", "9. Killzone")

    # ---- cross-cuts
    A("## 10. Cross-cuts\n")
    A("### Side × half")
    A("| Side | Half | n | avgR | Win% |")
    A("|---|---|---:|---:|---:|")
    for (s, h), g in T.groupby(["side", "half"]):
        A(f"| {s} | {h} | {len(g)} | {g['r'].mean():+.3f} | "
          f"{g['win'].mean() * 100:.0f} |")
    A("\n### Session × half (top sessions)")
    A("| Session | Half | n | avgR | Win% |")
    A("|---|---|---:|---:|---:|")
    for (s, h), g in T.groupby(["ctx_session", "half"]):
        A(f"| {s} | {h} | {len(g)} | {g['r'].mean():+.3f} | "
          f"{g['win'].mean() * 100:.0f} |")
    A("\n### Exit reason × half")
    A("| Exit | Half | n | avgR |")
    A("|---|---|---:|---:|")
    for (e, h), g in T.groupby(["exit", "half"]):
        A(f"| {e} | {h} | {len(g)} | {g['r'].mean():+.3f} |")

    # ---- direction vs timing
    A("\n## 11. Direction vs timing (MFE analysis)\n")
    A("| Bucket | H1 n | H1 avgR | H2 n | H2 avgR |")
    A("|---|---:|---:|---:|---:|")
    for label, cond in [
        ("MFE < 0.25R (never went anywhere — direction wrong)",
         T["mfe"] < 0.25),
        ("0.25R ≤ MFE < 1R (moved, failed to extend — timing/exit)",
         (T["mfe"] >= 0.25) & (T["mfe"] < 1.0)),
        ("MFE ≥ 1R (had a real move)", T["mfe"] >= 1.0),
    ]:
        g = T[cond]
        g1 = g[g["half"].str.startswith("H1")]
        g2 = g[g["half"].str.startswith("H2")]
        A(f"| {label} | {len(g1)} | "
          f"{g1['r'].mean():+.3f} | {len(g2)} | {g2['r'].mean():+.3f} |")
    los = T[~T["win"]]
    A(f"\nOf {len(los)} losers: MFE<0.25R: {(los['mfe'] < 0.25).sum()} "
      f"({(los['mfe'] < 0.25).mean() * 100:.0f}% never moved); "
      f"MFE≥1R: {(los['mfe'] >= 1).sum()} (had ≥1R then gave it back).")
    A(f"Wicked invalidation (crossed raw stop, survived): "
      f"{int(T['wicked'].sum())} trades, avg {T[T['wicked']]['r'].mean():+.3f}R; "
      f"non-wicked: avg {T[~T['wicked']]['r'].mean():+.3f}R.\n")

    # ---- trap overlap
    A("## 12. Trap-intelligence overlap\n")
    for flag in ["bait", "sfp"]:
        A(f"### {flag}")
        A("| Flag | Half | n | avgR | Win% |")
        A("|---|---|---:|---:|---:|")
        for (f, h), g in T.groupby([flag, "half"]):
            A(f"| {f} | {h} | {len(g)} | {g['r'].mean():+.3f} | "
              f"{g['win'].mean() * 100:.0f} |")
        A("")

    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
