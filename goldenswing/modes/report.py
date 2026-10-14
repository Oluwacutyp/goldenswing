"""Phase-5 report: MODES.md + data/modes_results.json.

Runs the three modes independently, then the swing routing study
(OFF → measured matrix → ON), and writes the numbers.  The verdict
prose in MODES.md is hand-written after the run, against these
numbers — the generator only guarantees the numbers are all here.
"""
from __future__ import annotations

import argparse
import json

from ..backtest.data import REPO_ROOT
from .config import FAST_SCALP, SCALP, SWING
from .routing import PHASE_GROUPS, matrix_rows
from .runner import ModeRun, RoutingStudy, run_mode, run_swing_routing
from ..golden.detector import PLAYBOOKS


def _book_row(label: str, b: dict) -> str:
    return (f"| {label} | {b['trades']} | {b['win_rate']:.1f} | "
            f"{b['pf']:.2f} | {b['return_pct']:+.2f}% | "
            f"{b['max_dd_pct']:.2f}% | {b['sharpe']:.2f} | "
            f"{b['avg_r']:+.3f} | {b['avg_hold']:.1f} |")


def _mode_section(r: ModeRun) -> list[str]:
    m = r.mode
    lines = [
        f"## Mode: {m.name} ({m.interval})",
        "",
        f"**Data:** GC=F {m.interval}, {r.bars:,} bars, {r.start} → "
        f"{r.end} ({r.years:.3f} years). RS peers at the same "
        f"interval: " + ", ".join(
            f"{k} {'available' if v else 'UNAVAILABLE'}"
            for k, v in sorted(r.peers_available.items())) + ".",
        "",
        f"Filters: killzone_only={m.killzone_only}, "
        f"allowed_kinds={sorted(m.allowed_kinds) if m.allowed_kinds else 'all'}, "
        f"min_displacement={m.min_displacement}, exit={m.exit_style}, "
        f"horizon={m.horizon} bars, cost=${m.cost_per_oz:.2f}/oz.",
        "",
        f"Signals: {r.n_scanned} scanned → {r.n_filtered} after mode "
        f"filters → {r.n_simulated} simulated "
        f"({r.n_filtered - r.n_simulated} skipped by the simulator).",
        "",
        "| Book @ ≥60 | Trades | Win% | PF | Return | MaxDD | Sharpe | "
        "Avg R | Avg hold (bars) |",
        "|---|---|---|---|---|---|---|---|---|",
        _book_row("real cost ($0.45/oz)", r.book),
        _book_row("zero cost (signal quality only)", r.book_zero_cost),
        "",
        f"Shadow (all simulated signals, standalone): n="
        f"{r.shadow['n']}, win {r.shadow['win_rate']:.1f}%, avg R "
        f"{r.shadow['avg_r']:+.3f}, PF {r.shadow['pf']:.2f}. "
        f"Shadow @≥60: n={r.shadow60['n']}, avg R "
        f"{r.shadow60['avg_r']:+.3f}.",
    ]
    if r.cost_r_mean is not None:
        lines += [
            "",
            f"**Cost in R** (per simulated trade, cost / risk at the "
            f"buffered stop): mean {r.cost_r_mean:.3f}R, median "
            f"{r.cost_r_median:.3f}R, p90 {r.cost_r_p90:.3f}R. "
            f"For scale, the same $0.45 costs 0.022R on the swing "
            f"mode's H1 stops (VALIDATION.md).",
        ]
        if r.exit_mix:
            lines.append(
                "Book exit mix: " + ", ".join(
                    f"{k} {v:.1f}%" for k, v in sorted(r.exit_mix.items()))
                + ".")
    lines.append("")
    return lines


def _routing_section(st: RoutingStudy) -> list[str]:
    lines = [
        "## Phase→playbook routing (swing mode)",
        "",
        f"Matrix measured from the routing-OFF shadow book "
        f"({st.n_signals_off} simulated signals, GC=F H1, "
        f"{st.start} → {st.end}). Playbooks: **sweep_reversal** "
        f"(detector kind), **breakout** (continuation riding a fresh "
        f"BOS, age ≤ 5), **trend_continuation** (any other "
        f"continuation). Muting rule: avg R < 0 AND n ≥ "
        f"{st.table.min_n if st.table else 30} → the playbook is "
        f"capped at {st.table.cap if st.table else 59.0:.0f} in that "
        f"phase.",
        "",
        "| Phase | " + " | ".join(
            f"{pb} n / win% / avg R" for pb in PLAYBOOKS) + " |",
        "|---|" + "---|" * len(PLAYBOOKS),
    ]
    cells = st.cells
    for ph in PHASE_GROUPS:
        row = [f"| {ph}"]
        for pb in PLAYBOOKS:
            c = cells.get((ph, pb))
            if c is None or c.n == 0:
                row.append("— (n=0)")
            else:
                muted = st.table is not None and (ph, pb) in st.table.muted
                row.append(
                    f"{c.n} / {c.win_rate:.1f} / {c.avg_r:+.3f}"
                    + (" **MUTED**" if muted else ""))
        lines.append(" | ".join(row) + " |")
    lines += [
        "",
        f"Signals emitted: OFF {st.n_signals_off} simulated, ON "
        f"{st.n_signals_on} simulated (routing caps scores; it does "
        f"not change the emitted set).",
        "",
        "| Book | Trades | Win% | PF | Return | MaxDD | Sharpe | "
        "Avg R | Avg hold |",
        "|---|---|---|---|---|---|---|---|---|",
        _book_row("Routing OFF @ ≥60", st.off_book60),
        _book_row("Routing ON @ ≥60", st.on_book60),
        _book_row("Routing OFF @ ≥50", st.off_book50),
        _book_row("Routing ON @ ≥50", st.on_book50),
        "",
        f"Shadow @≥60 avg R: OFF {st.off_shadow60['avg_r']:+.3f} "
        f"(n={st.off_shadow60['n']}) → ON "
        f"{st.on_shadow60['avg_r']:+.3f} (n={st.on_shadow60['n']}) "
        f"— the ON shadow grade mechanically excludes the muted "
        f"cells from the ≥60 region, so it is not independent "
        f"evidence; the book comparison is the test.",
        "",
    ]
    return lines


def build_modes_report(swing: ModeRun, scalp: ModeRun, fast: ModeRun,
                       routing: RoutingStudy) -> str:
    lines = [
        "# GoldenSwing — Modes + Phase Routing (Phase 5)",
        "",
        "**Method:** one market read (MarketReader + FeatureEngine), "
        "one detector (Phase-2 flags + Combo A {relstr, volume, "
        "phases}), three execution modes differing only in timeframe, "
        "entry filters, exit style and horizon.  Costs are NOT "
        "rescaled per mode: $0.45/oz round trip everywhere, so the "
        "cost-as-R at scalp timeframes is measured, not assumed.  "
        "Every mode run is independent (own data, own reader pass, "
        "own detector).",
        "",
        "**Reproduction anchors:** swing book @ ≥60 must equal "
        "VALIDATION.md combo A (+25.62% / PF 1.22 / 199 trades); the "
        "routing-OFF book is the same run by construction and must "
        "match it too.",
        "",
    ]
    lines += _mode_section(swing)
    lines += _routing_section(routing)
    lines += _mode_section(scalp)
    lines += _mode_section(fast)
    lines += [
        "## Verdicts",
        "",
        "VERDICTS_PLACEHOLDER",
        "",
        "## Caveats",
        "",
        "- The routing matrix is computed from the SAME window it "
        "routes on: the ON/OFF comparison tests the routing "
        "mechanism in-sample, it is not out-of-sample proof.  A "
        "walk-forward split (derive the table on window A, route on "
        "window B) is the honest next test and has NOT been run.",
        "- Scalp samples are ~60 days (yfinance intraday cap) against "
        "the swing mode's 2 years, inside the same bullish gold "
        "regime.  Scalp verdicts are weaker claims and are labelled "
        "as such.",
        "- Bar-count parameters of the shared reader/features rescale "
        "with the timeframe (48-bar AMD base = 48h on H1, 12h on "
        "M15, 4h on M5).  That is what 'same read, faster clock' "
        "means here; per-timeframe reparametrisation was NOT done "
        "and would be a different experiment.",
        "- The MTF feature family's HTF mapping assumes hourly bars "
        "(it keys HTF states to bar start + 1h).  MTF is computed but "
        "NOT scored in any mode (not in Combo A); on sub-hourly "
        "frames its values can be up to one bar early and must not "
        "be enabled there before that mapping is fixed.",
        "- GC=F 5m/15m volume is real futures volume; yfinance "
        "intraday bars can contain partial/odd-lot prints the live "
        "feed would differ on.  No slippage beyond the flat cost.",
        "",
    ]
    return "\n".join(lines)


def _run_to_json(r: ModeRun) -> dict:
    return {
        "mode": r.mode.name, "interval": r.mode.interval,
        "bars": r.bars, "start": r.start, "end": r.end,
        "years": r.years, "peers_available": r.peers_available,
        "n_scanned": r.n_scanned, "n_filtered": r.n_filtered,
        "n_simulated": r.n_simulated, "book": r.book,
        "book_zero_cost": r.book_zero_cost, "shadow": r.shadow,
        "shadow60": r.shadow60, "cost_r_mean": r.cost_r_mean,
        "cost_r_median": r.cost_r_median, "cost_r_p90": r.cost_r_p90,
        "exit_mix": r.exit_mix,
    }


def main5(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GoldenSwing Phase-5 modes")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args(argv)

    swing = run_mode(SWING, refresh=args.refresh_data)
    routing = run_swing_routing(refresh=args.refresh_data)
    scalp = run_mode(SCALP, refresh=args.refresh_data)
    fast = run_mode(FAST_SCALP, refresh=args.refresh_data)

    payload = {
        "swing": _run_to_json(swing),
        "scalp": _run_to_json(scalp),
        "fast_scalp": _run_to_json(fast),
        "routing": {
            "cells": {f"{ph}|{pb}": {"n": c.n, "win_rate": c.win_rate,
                                     "avg_r": c.avg_r, "total_r": c.total_r}
                      for (ph, pb), c in routing.cells.items()},
            "muted": sorted(f"{ph}|{pb}"
                            for ph, pb in (routing.table.muted
                                           if routing.table else [])),
            "off_book60": routing.off_book60,
            "on_book60": routing.on_book60,
            "off_book50": routing.off_book50,
            "on_book50": routing.on_book50,
            "off_shadow60": routing.off_shadow60,
            "on_shadow60": routing.on_shadow60,
        },
    }
    (REPO_ROOT / "data" / "modes_results.json").write_text(
        json.dumps(payload, indent=2, default=str))

    report = build_modes_report(swing, scalp, fast, routing)
    out = REPO_ROOT / "MODES.md"
    out.write_text(report)
    print(report)
    print(f"\n[written to {out}]")


if __name__ == "__main__":
    main5()
