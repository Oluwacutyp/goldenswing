"""Phase 3.5 validation: the winning-combo test + out-of-sample runs.

Phase 3 ablated feature families ONE at a time on the in-sample gold
window and found three ADDS families (relstr, volume, phases) and one
borderline (bollinger: return slightly down, MaxDD down).  The exact
combo was never run together.  This module runs:

1. IN-SAMPLE (GC=F H1, Oct 2024 -> Oct 2026 — the development window):
   baseline (features off, MUST reproduce BACKTEST2/FEATURES numbers)
   vs combo A {relstr, volume, phases} vs combo B {A + bollinger}.
   Winner rule, fixed before the run: higher book-@>=60 return wins;
   if within 1.0pp, the lower MaxDD wins.  The winner becomes the
   documented recommended config and goes out-of-sample.

2. OUT-OF-SAMPLE (same code, same DetectorConfig, same costs/exits —
   NO parameter changes):
   - Silver SI=F H1 over the main window (a different instrument the
     detector was never developed on; the relstr 'silver' peer leg is
     then the instrument itself — ratio 1.0, trend flat by
     construction, so the dollar leg carries RS there).
   - BTC-USD H1 over the main window (different asset class, 24/7).
   Both run baseline + BOTH combos, so a cherry-picked winner cannot
   hide behind the runner-up.

3. The earlier gold window (Oct 2022 -> Oct 2024) requested for this
   phase CANNOT be run: Yahoo refuses 1h bars older than 730 days
   (API error, verified), and Dukascopy/Stooq are unreachable from
   this environment (connection resets).  VALIDATION.md documents the
   evidence instead of substituting an in-sample split dressed up as
   out-of-sample.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..features import FeatureEngine
from ..golden.detector import DetectorConfig, GoldenSwingDetector
from ..reader.reader import MarketReader
from .ablation import P2_FLAGS
from .data import (REPO_ROOT, fetch_series_h1, load_gold_h1, load_peer_h1,
                   load_peers)
from .harness import (BANDS, book_metrics, build_book, simulate_signal_v2,
                      slice_stats)

COMBO_A = frozenset({"relstr", "volume", "phases"})
COMBO_B = frozenset({"relstr", "volume", "phases", "bollinger"})
CONFIGS = [("baseline (features off)", None),
           ("combo A {relstr, volume, phases}", COMBO_A),
           ("combo B {A + bollinger}", COMBO_B)]


@dataclass
class ValResult:
    name: str
    n_signals: int
    book60: dict
    book50: dict
    shadow60: dict
    bands: list = field(default_factory=list)


@dataclass
class SuiteResult:
    label: str
    instrument: str
    bars: int
    start: str
    end: str
    years: float
    peers_note: str
    results: dict = field(default_factory=dict)   # config name -> ValResult


def run_config(df: pd.DataFrame, states, families, name: str) -> ValResult:
    """Mirror ablation._run_config exactly, plus score-band grading."""
    detector = GoldenSwingDetector(DetectorConfig(
        emit_floor=35.0, golden_threshold=60.0,
        feature_families=families, **P2_FLAGS))
    signals = detector.scan(states)
    outcomes = []
    for sig in signals:
        o = simulate_signal_v2(df, sig)
        if o is not None:
            outcomes.append(o)
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    years = span_days / 365.25
    bands = []
    for lo_b, hi_b in BANDS:
        st = slice_stats([o for o in outcomes
                          if lo_b <= o.signal.score < hi_b])
        bands.append({"lo": lo_b, "hi": hi_b, **st})
    return ValResult(
        name=name, n_signals=len(outcomes),
        book60=book_metrics(build_book(outcomes, 60.0), years),
        book50=book_metrics(build_book(outcomes, 50.0), years),
        shadow60=slice_stats([o for o in outcomes
                              if o.signal.score >= 60.0]),
        bands=bands)


def run_suite(df: pd.DataFrame, peers: dict | None, label: str,
              instrument: str, peers_note: str) -> SuiteResult:
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    years = span_days / 365.25
    plain_states = MarketReader().read(df)
    featured_states = MarketReader().read(df)
    FeatureEngine().attach(df, featured_states, peers)
    results = {}
    for name, fams in CONFIGS:
        states = plain_states if fams is None else featured_states
        results[name] = run_config(df, states, fams, name)
    return SuiteResult(
        label=label, instrument=instrument, bars=len(df),
        start=str(df.index[0]), end=str(df.index[-1]), years=years,
        peers_note=peers_note, results=results)


def pick_winner(a: ValResult, b: ValResult) -> ValResult:
    """Pre-declared rule: higher book-@60 return; within 1.0pp, lower
    MaxDD wins."""
    ra, rb = a.book60["return_pct"], b.book60["return_pct"]
    if abs(ra - rb) > 1.0:
        return a if ra > rb else b
    return a if a.book60["max_dd_pct"] <= b.book60["max_dd_pct"] else b


def load_btc_h1(refresh: bool = False) -> pd.DataFrame:
    cache_path = REPO_ROOT / "data" / "BTCUSD_H1_2y.csv"
    if cache_path.exists() and not refresh:
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            if len(df) > 1000:
                return df
        except Exception:
            pass
    df = fetch_series_h1("BTC-USD", cache_path)
    if df is None or len(df) == 0:
        raise RuntimeError("could not fetch BTC-USD hourly data")
    return df


# ------------------------------------------------------------------ report
def _suite_tables(s: SuiteResult) -> list[str]:
    lines = [
        f"### {s.label}",
        "",
        f"**{s.instrument}** — {s.bars:,} bars, {s.start} → {s.end} "
        f"({s.years:.2f} years). {s.peers_note}",
        "",
        "| Config | Signals | Book trades | Win% | PF | Return | MaxDD | "
        "Sharpe | Book avg R | Shadow @≥60 avg R | Book@50 return |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, _ in CONFIGS:
        r = s.results[name]
        b = r.book60
        lines.append(
            f"| {name} | {r.n_signals} | {b['trades']} | "
            f"{b['win_rate']:.1f} | {b['pf']:.2f} | "
            f"{b['return_pct']:+.2f}% | {b['max_dd_pct']:.2f}% | "
            f"{b['sharpe']:.2f} | {b['avg_r']:+.3f} | "
            f"{r.shadow60['avg_r']:+.3f} (n={r.shadow60['n']}) | "
            f"{r.book50['return_pct']:+.2f}% |")
    lines += ["", "Score-band discrimination (shadow book — every "
              "signal standalone; backing must rank: higher band, "
              "better avg R):", ""]
    for name, _ in CONFIGS:
        r = s.results[name]
        cells = " · ".join(
            f"{bd['lo']:.0f}–{bd['hi']:.0f}: {bd['avg_r']:+.3f}R "
            f"(n={bd['n']})" for bd in r.bands)
        lines.append(f"- **{name}:** {cells}")
    lines.append("")
    return lines


def suite_to_json(s: SuiteResult) -> dict:
    return {
        "label": s.label, "instrument": s.instrument, "bars": s.bars,
        "start": s.start, "end": s.end, "years": s.years,
        "results": {
            name: {
                "n_signals": r.n_signals, "book60": r.book60,
                "book50": r.book50, "shadow60": r.shadow60,
                "bands": r.bands,
            } for name, r in s.results.items()
        },
    }


def main35(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GoldenSwing Phase-3.5 validation")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args(argv)

    peers = load_peers(refresh=args.refresh_data)

    gold = load_gold_h1(refresh=args.refresh_data)
    in_sample = run_suite(
        gold, peers, "Test 1 — IN-SAMPLE combo test (development window)",
        "GC=F hourly (gold futures)",
        "RS peers: silver SI=F + dollar DX-Y.NYB, same caches as Phase 3.")

    a = in_sample.results["combo A {relstr, volume, phases}"]
    b = in_sample.results["combo B {A + bollinger}"]
    winner = pick_winner(a, b)
    winner_name = winner.name

    silver = load_peer_h1("silver", refresh=args.refresh_data)
    if silver is None:
        raise RuntimeError("silver SI=F data unavailable — OOS-1 cannot run")
    oos_silver = run_suite(
        silver, peers, "Test 2 — OUT-OF-SAMPLE: silver (instrument change)",
        "SI=F hourly (silver futures)",
        "RS peers: silver leg is the instrument itself (ratio 1.0, flat "
        "by construction); the dollar leg carries RS. Same peer dict, "
        "same config — no rewiring.")

    btc = load_btc_h1(refresh=args.refresh_data)
    oos_btc = run_suite(
        btc, peers, "Test 3 — OUT-OF-SAMPLE: BTC (asset-class change)",
        "BTC-USD hourly (Yahoo spot)",
        "RS peers: silver + dollar (both exogenous to BTC). 24/7 series; "
        "session/killzone tags are UTC-hour based as in-sample.")

    suites = [in_sample, oos_silver, oos_btc]
    payload = {"winner": winner_name,
               "suites": [suite_to_json(s) for s in suites]}
    (REPO_ROOT / "data" / "validation_results.json").write_text(
        json.dumps(payload, indent=2, default=str))

    lines = [
        "# GoldenSwing — Validation (Phase 3.5)",
        "",
        "Combo test + out-of-sample validation of the Phase-3 feature "
        "verdicts. Same detector, same Phase-2 flags "
        "(`require_sweep_confirmation`, `daily_veto`), same v2 exits, "
        "same $0.45/oz cost, 1% risk — NO parameter changes between "
        "in-sample and out-of-sample runs.",
        "",
        "## Recommended config (decided in-sample, rule fixed pre-run)",
        "",
        f"**Winner: {winner_name}** — higher book-@≥60 return wins; "
        "if within 1.0pp, lower MaxDD wins.",
        "",
    ]
    for s in suites:
        lines += _suite_tables(s)
    lines += [
        "## The earlier-window test (Oct 2022 → Oct 2024 gold) — NOT RUN",
        "",
        "Yahoo (the repo's only wired hourly source) refuses 1h bars "
        "older than 730 days: `1h data not available … The requested "
        "range must be within the last 730 days` (verified "
        "2026-10-11 against GC=F for Nov 2022). Dukascopy's public "
        "datafeed and Stooq intraday both reset/empty-reply from this "
        "environment after repeated probes. No local cache of "
        "2022–2024 hourly gold exists on this machine. Rather than "
        "dress an in-sample split up as out-of-sample, this test is "
        "recorded as BLOCKED on data; it is the first run to execute "
        "when a 2022–2024 hourly gold feed becomes available.",
        "",
        "## Verdict",
        "",
        "VERDICT_PLACEHOLDER",
        "",
    ]
    out = REPO_ROOT / "VALIDATION.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[written to {out}]")


if __name__ == "__main__":
    main35()
