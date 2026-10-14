#!/usr/bin/env python3
"""Held-out validation of the two surviving sleeves on 2022-2024 data.

  - fvg_midline: exact validated scanner, gates (incl. the trend-down
    hunting-ground exclusion), fixed-2R exits. Unfiltered book.
  - trendline_retest: exact validated scanner + gates, Phase-2 exits.
    Base book (unfiltered) is primary. The promoted +4 trap confluence
    bonus is a RANKING input (same convention as the salvage: it affects
    same-bar ranking, not book membership), so its held-out test is the
    trap subset map + the boost_only filter book as a labeled secondary.

Data: XAU/USD H1 2022-01-01 -> 2024-10-01 from Dukascopy (yfinance cannot
serve H1 before ~Oct 2024: 730-day intraday wall). Silver peer XAG/USD
from Dukascopy; dollar peer unavailable (graceful degradation).
This window is SACRED held-out data: no tuning, no parameter changes.
Validated baselines (development window 2024-10->2026-10):
  fvg_midline      48 trades / +34.61% / PF 2.47
  trendline_retest 61 trades / +15.40% / PF 1.43

Writes experiments/SLEEVES_HELDOUT.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.harness import book_metrics
from goldenswing.contextmap import context_levels
from goldenswing.features.engine import read_and_featurize
from goldenswing.features.traps import compute_trap_features
from goldenswing.playbooks import FVGMidlinePlaybook, TrendlineRetestPlaybook
from goldenswing.playbooks.base import EngineContext

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "SLEEVES_HELDOUT.md"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def gated_candidates(pb, ctx):
    """Validated scanner + the engine's own context gates (session scope,
    check_context tripwire, hunting-ground exclusions). Mirrors
    scripts/variant_heldout.py::build_book exactly."""
    states = ctx.states
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
        scoped.append(c)
    return cands, scoped


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
            "side": "long" if o.signal.direction == 1 else "short",
            "exit": getattr(o, "exit_reason", "?"),
        })
    return pd.DataFrame(rows)


def _dir_flags(row: pd.Series, direction: int) -> dict:
    long = direction > 0
    return {
        "boost": bool(row["trap_boost_long"] if long
                      else row["trap_boost_short"]),
        "oppose": bool(row["trap_boost_short"] if long
                       else row["trap_boost_long"]),
    }


def trap_subset_map(book, trap_df) -> dict:
    """Baseline book trades grouped by trap state at the ENTRY bar
    (salvage §3a convention)."""
    groups: dict[str, list[float]] = {"boost": [], "oppose": [],
                                      "neutral": []}
    for o in book:
        if o.entry_index >= len(trap_df):
            continue
        f = _dir_flags(trap_df.iloc[o.entry_index], o.signal.direction)
        if f["boost"]:
            groups["boost"].append(o.r_net)
        elif f["oppose"]:
            groups["oppose"].append(o.r_net)
        else:
            groups["neutral"].append(o.r_net)

    def summ(rs):
        rs = np.array(rs, dtype=float)
        return {"n": int(len(rs)),
                "avg_r": float(rs.mean()) if len(rs) else 0.0,
                "win_rate": float(np.mean(rs > 0) * 100) if len(rs) else 0.0,
                "total_r": float(rs.sum()) if len(rs) else 0.0}

    return {k: summ(v) for k, v in groups.items()}


def table(fr: pd.DataFrame, col: str) -> list[str]:
    out = []
    for key, g in fr.groupby(col):
        out.append(f"| {key} | {len(g)} | {g['r'].sum():+.1f} | "
                   f"{g['r'].mean():+.3f} | {g['win'].mean()*100:.0f} |")
    return out


def main() -> None:
    df = load_csv(REPO / "data" / "XAU_H1_2022_2024.csv")
    silver = load_csv(REPO / "data" / "XAG_H1_2022_2024.csv")
    peers = {"silver": silver, "dollar": None}
    print(f"gold: {df.shape} {df.index[0]} -> {df.index[-1]}", flush=True)
    bad = df[(df["high"] < df["low"])]
    print(f"bad OHLC bars: {len(bad)}", flush=True)

    states = read_and_featurize(df, peers)
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    ctx = EngineContext(df=df, states=states, years=years,
                        instrument="XAUUSD")

    results = {}
    for pb in (FVGMidlinePlaybook(), TrendlineRetestPlaybook()):
        raw, scoped = gated_candidates(pb, ctx)
        book = [c.outcome for c in scoped]  # sleeves unfiltered (thr 0)
        m = book_metrics(book, years)
        fr = frame(book, df, states)
        results[pb.name] = {"m": m, "fr": fr, "n_raw": len(raw),
                            "n_scoped": len(scoped), "book": book}
        print(f"{pb.name}: raw={len(raw)} scoped={len(scoped)} "
              f"trades={m['trades']} net={m['return_pct']:+.2f}% "
              f"PF={m['pf']:.2f} win={m['win_rate']:.1f}% "
              f"avgR={m['avg_r']:+.3f}", flush=True)

    # trap treatment for trendline_retest (held-out test of the bonus premise)
    print("computing trap features ...", flush=True)
    trap_df = compute_trap_features(df)
    assert len(trap_df) == len(df)
    tl_book = results["trendline_retest"]["book"]
    tl_map = trap_subset_map(tl_book, trap_df)
    boost_book = [o for o in tl_book
                  if o.entry_index < len(trap_df) and
                  _dir_flags(trap_df.iloc[o.entry_index],
                             o.signal.direction)["boost"]]
    m_boost = book_metrics(boost_book, years)
    print(f"trendline boost_only: trades={m_boost['trades']} "
          f"net={m_boost['return_pct']:+.2f}% PF={m_boost['pf']:.2f} "
          f"avgR={m_boost['avg_r']:+.3f}", flush=True)

    lines = []
    a = lines.append
    a("# Sleeves Held-Out Validation — fvg_midline + trendline_retest")
    a("")
    a("**Data:** XAU/USD H1, Dukascopy, 2022-01-01 → 2024-10-01 "
      f"({len(df)} bars, {(df.index[-1]-df.index[0]).days/365.25:.2f}y). "
      "yfinance cannot serve H1 before ~Oct 2024 (730-day intraday wall), "
      "so the 2022–2024 window uses the same underlying market as spot "
      "(GC=F futures vs spot XAU/USD — spreads/rolls differ slightly, "
      "price action tracks). Silver peer XAG/USD from Dukascopy; dollar "
      "peer unavailable (graceful degradation, documented). Zero overlap "
      "with the 2024-10→2026-10 development window. "
      "**No tuning was performed against this window** — all parameters "
      "are the validated ones, byte-identical.")
    a("")
    a("## Headline")
    a("")
    a("| Sleeve | Trades | Net % | PF | Win% | Avg R | MaxDD% |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for name in ("fvg_midline", "trendline_retest"):
        m = results[name]["m"]
        a(f"| {name} | {m['trades']} | {m['return_pct']:+.2f} | "
          f"{m['pf']:.2f} | {m['win_rate']:.1f} | {m['avg_r']:+.3f} | "
          f"{m['max_dd_pct']:.1f} |")
    a("")
    a("Development-window baselines for comparison: fvg_midline 48 / "
      "+34.61% / PF 2.47 · trendline_retest 61 / +15.40% / PF 1.43.")
    a("")
    for name in ("fvg_midline", "trendline_retest"):
        m = results[name]["m"]
        fr = results[name]["fr"]
        a(f"## {name} — by year")
        a("")
        a("| Year | Trades | Net R | Avg R | Win% |")
        a("|---|---:|---:|---:|---:|")
        a("\n".join(table(fr, "year")) if len(fr) else "| — | 0 | — | — | — |")
        a("")
        a(f"## {name} — by regime")
        a("")
        a("| Regime | Trades | Net R | Avg R | Win% |")
        a("|---|---:|---:|---:|---:|")
        a("\n".join(table(fr, "regime")) if len(fr)
          else "| — | 0 | — | — | — |")
        a("")
        a(f"## {name} — by session")
        a("")
        a("| Session | Trades | Net R | Avg R | Win% |")
        a("|---|---:|---:|---:|---:|")
        a("\n".join(table(fr, "session")) if len(fr)
          else "| — | 0 | — | — | — |")
        a("")
        a(f"## {name} — by side")
        a("")
        a("| Side | Trades | Net R | Avg R | Win% |")
        a("|---|---:|---:|---:|---:|")
        a("\n".join(table(fr, "side")) if len(fr)
          else "| — | 0 | — | — | — |")
        a("")

    a("## trendline_retest — trap treatment (held-out test of the +4 bonus premise)")
    a("")
    a("The promoted +4 bonus is a RANKING input (same-bar conflicts), not "
      "a filter — on a standalone book it cannot change membership. Its "
      "held-out test is the trap subset map (salvage §3a convention, "
      "trap state at the entry bar):")
    a("")
    a("| Trap state | n | Avg R | Win% | Total R |")
    a("|---|---:|---:|---:|---:|")
    for k in ("boost", "oppose", "neutral"):
        s = tl_map[k]
        a(f"| {k} | {s['n']} | {s['avg_r']:+.3f} | {s['win_rate']:.0f} | "
          f"{s['total_r']:+.1f} |")
    a("")
    a("Secondary (labeled, NOT the promoted convention): boost_only filter "
      f"book → {m_boost['trades']} trades, {m_boost['return_pct']:+.2f}%, "
      f"PF {m_boost['pf']:.2f}, avg {m_boost['avg_r']:+.3f}R.")
    a("")

    a("## Verdicts")
    a("")
    for name, dev_trades, dev_note in (
            ("fvg_midline", 48, "+34.61% / PF 2.47"),
            ("trendline_retest", 61, "+15.40% / PF 1.43")):
        m = results[name]["m"]
        if m["trades"] >= 50 and m["avg_r"] > 0 and m["pf"] > 1.2:
            a(f"**{name}: HOLDS on held-out data.** {m['trades']} trades, "
              f"avg {m['avg_r']:+.3f}R/trade, PF {m['pf']:.2f} "
              f"(dev baseline {dev_trades} / {dev_note}). The edge "
              "replicates outside the development window.")
        elif m["trades"] >= 50:
            a(f"**{name}: KILLED AS TESTED.** {m['trades']} trades clear "
              f"the count bar, but the edge did not replicate: avg "
              f"{m['avg_r']:+.3f}R, PF {m['pf']:.2f} (dev baseline "
              f"{dev_trades} / {dev_note}).")
        elif m["trades"] > 0:
            a(f"**{name}: INCONCLUSIVE — below the 50-trade bar.** Only "
              f"{m['trades']} trades on 2.75y of held-out data "
              f"(avg {m['avg_r']:+.3f}R, PF {m['pf']:.2f}; dev baseline "
              f"{dev_trades} / {dev_note}). Cannot promote or kill on "
              "this count.")
        else:
            a(f"**{name}: NO TRADES on held-out data.** The scanner fired "
              f"zero validated setups in 2.75y (dev baseline {dev_trades} "
              f"/ {dev_note}). The sleeve's contexts did not occur here — "
              "neither a kill nor a hold, but no evidence either.")
    a("")
    a("**Method notes:** scanner, gates, exits, and booking are the "
      "validated ones, unchanged. Spot-vs-futures and missing-dollar-peer "
      "are the documented deviations from the development setup. The "
      "paper loop is unaffected by these verdicts — it keeps earning "
      "live OOS evidence either way.")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
