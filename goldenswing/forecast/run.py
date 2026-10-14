"""End-to-end forecaster run: data -> states -> features -> labels ->
purged walk-forward -> evaluation -> trading tests -> FORECAST.md.

The trading anchor is VALIDATION.md's Combo-A book (+25.62%, PF 1.22,
199 trades @ >=60): the gate test's full-window plain book must
reproduce it, or the comparison means nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.data import REPO_ROOT, load_gold_h1, load_peers
from ..features import FeatureEngine
from ..golden.detector import GoldenSwingDetector
from ..reader.reader import MarketReader
from . import evaluate, trading
from .features import build_feature_frame, combo_a_config, detector_opinions
from .labels import HORIZON, build_labels
from .walkforward import run_walk_forward


@dataclass
class ForecastRun:
    bars: int
    start: str
    end: str
    oof_bars: int
    oof_start_bar: int
    base_rate_up: float
    base_rate_dn: float
    auc: pd.DataFrame
    quality_up: pd.DataFrame
    quality_dn: pd.DataFrame
    calib_up: pd.DataFrame
    calib_dn: pd.DataFrame
    gate: dict
    standalone: dict
    control: dict = field(default_factory=dict)
    importance: list = field(default_factory=list)


def run(df: pd.DataFrame | None = None,
        peers: dict | None = None) -> ForecastRun:
    if df is None:
        df = load_gold_h1()
    if peers is None:
        peers = load_peers()

    states = MarketReader().read(df)
    FeatureEngine().attach(df, states, peers)
    atr = np.array([s.atr for s in states], dtype=float)

    opinions = detector_opinions(states)
    X = build_feature_frame(states, opinions)
    labels = build_labels(df, atr, horizon=HORIZON)
    labels.index = pd.RangeIndex(len(df))

    wf = run_walk_forward(X, labels, horizon=HORIZON)
    probs = wf.probs

    det_signed = (opinions["det_score"] * opinions["det_dir"])
    det_signed.index = pd.RangeIndex(len(df))
    auc = evaluate.auc_table(probs, labels, det_signed)
    quality_up = evaluate.alert_quality(probs, labels, "ens_up", "y_up")
    quality_dn = evaluate.alert_quality(probs, labels, "ens_dn", "y_dn")
    calib_up = evaluate.calibration(probs, labels, "ens_up", "y_up")
    calib_dn = evaluate.calibration(probs, labels, "ens_dn", "y_dn")

    # ---- trading tests -------------------------------------------------
    detector = GoldenSwingDetector(combo_a_config())
    signals = detector.scan(states)
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    full_years = span_days / 365.25
    oof_span = (df.index[probs.index[-1]] - df.index[probs.index[0]]) \
        .total_seconds() / 86400.0
    oof_years = max(oof_span / 365.25, 1e-9)
    gate = trading.gate_test(df, signals, probs, full_years, oof_years)
    standalone = trading.standalone_test(df, probs, atr, oof_years)
    control = trading.permutation_control(df, probs, atr, oof_years)
    probs.to_csv(REPO_ROOT / "data" / "forecast_oof_probs.csv")

    oof_labels = labels.loc[probs.index]
    return ForecastRun(
        bars=len(df), start=str(df.index[0]), end=str(df.index[-1]),
        oof_bars=len(probs), oof_start_bar=int(probs.index[0]),
        base_rate_up=float(oof_labels["y_up"].mean()),
        base_rate_dn=float(oof_labels["y_dn"].mean()),
        auc=auc, quality_up=quality_up, quality_dn=quality_dn,
        calib_up=calib_up, calib_dn=calib_dn,
        gate=gate, standalone=standalone, control=control)


def _book_row(name: str, b: dict) -> str:
    return (f"| {name} | {b['trades']} | {b['win_rate']:.1f} | "
            f"{b['pf']:.2f} | {b['return_pct']:+.2f}% | "
            f"{b['max_dd_pct']:.2f}% | {b['sharpe']:.2f} | "
            f"{b['avg_r']:+.3f} |")


def build_report(r: ForecastRun) -> str:
    L: list[str] = []
    L += ["# GoldenSwing — Ensemble ML Forecaster (Phase 6)", "",
          "## What was asked", "",
          "The owner's core want: *an ML that can forecast a big move "
          "in 3 days and alert me.*  This phase builds that forecaster "
          "and grades it the house way — walk-forward, embargoed, and "
          "against the detector it would sit next to.", "",
          "## Definition", "",
          "At each H1 bar t: P(big move up) / P(big move down), where a "
          "big move = favourable excursion reaching **2x ATR(t)** "
          "within the next **72 bars** BEFORE adverse excursion "
          "reaches **1x ATR(t)** (same-bar double touch = adverse "
          "wins; labels mutually exclusive; bars without a full "
          "72-bar window are unlabelled, never zero-filled).", "",
          "## Method", "",
          f"- Data: GC=F H1, {r.bars:,} bars, {r.start} → {r.end} "
          "(same window as Phases 1-5).",
          "- Features: reader state + Combo-A feature families + "
          "detector opinion per bar, all causal (pure functions of "
          "the bar's MarketState).",
          "- Models: HistGradientBoosting, logistic regression "
          "(median-imputed, scaled, class-weighted), ExtraTrees — "
          "trained separately per direction; ensemble = probability "
          "mean.",
          "- Walk-forward: expanding window, first test block after "
          "5,000 bars, retrain every 500 bars. PURGE: a training row "
          "is admitted only if its 72-bar label window ends before "
          "the test block opens. No shuffled splits anywhere.",
          f"- Out-of-fold coverage: {r.oof_bars:,} bars "
          f"(from bar {r.oof_start_bar}).  Base rates on OOF rows: "
          f"big-up {100 * r.base_rate_up:.1f}%, big-down "
          f"{100 * r.base_rate_dn:.1f}% (so always-predict-base-rate "
          "has AUC 0.500 by construction).", "",
          "## Discrimination (AUC, out-of-fold)", "",
          "| Direction | Model | AUC | n |",
          "|---|---|---|---|"]
    for _, row in r.auc.iterrows():
        L.append(f"| {row['direction']} | {row['model']} | "
                 f"{row['auc']:.3f} | {row['n']} |")
    L += ["", "## Alert quality (ensemble, out-of-fold)", "",
          "### Big move UP", "",
          "| P threshold | Alerts | Alert rate | Precision | Recall | Base rate |",
          "|---|---|---|---|---|---|"]
    for _, q in r.quality_up.iterrows():
        L.append(f"| {q['threshold']:.1f} | {q['alerts']} | "
                 f"{q['alert_rate_pct']:.1f}% | {q['precision']:.3f} | "
                 f"{q['recall']:.3f} | {q['base_rate']:.3f} |")
    L += ["", "### Big move DOWN", "",
          "| P threshold | Alerts | Alert rate | Precision | Recall | Base rate |",
          "|---|---|---|---|---|---|"]
    for _, q in r.quality_dn.iterrows():
        L.append(f"| {q['threshold']:.1f} | {q['alerts']} | "
                 f"{q['alert_rate_pct']:.1f}% | {q['precision']:.3f} | "
                 f"{q['recall']:.3f} | {q['base_rate']:.3f} |")
    L += ["", "## Calibration (ensemble, by predicted-probability decile)", "",
          "### UP — mean predicted vs empirical rate", "",
          "| n | Mean predicted | Empirical |", "|---|---|---|"]
    for _, c in r.calib_up.iterrows():
        L.append(f"| {c['n']} | {c['mean_pred']:.3f} | {c['empirical']:.3f} |")
    L += ["", "### DOWN", "", "| n | Mean predicted | Empirical |",
          "|---|---|---|"]
    for _, c in r.calib_dn.iterrows():
        L.append(f"| {c['n']} | {c['mean_pred']:.3f} | {c['empirical']:.3f} |")

    g = r.gate
    L += ["", "## Trading test 1 — forecaster as a GATE on the detector", "",
          "Full-window plain Combo-A book (reproduction anchor; "
          "VALIDATION.md says 199 trades, PF 1.22, +25.62%):", "",
          "| Book | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |",
          "|---|---|---|---|---|---|---|---|",
          _book_row("plain detector, full window", g["full_book"]), "",
          f"Restricted to the {g['oof_signals']} detector signals that "
          "fall in the out-of-fold region (the only fair arena for the "
          "gate):", "",
          "| Book | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |",
          "|---|---|---|---|---|---|---|---|",
          _book_row("plain detector, OOF signals only",
                    g["plain_oof_book"])]
    for th in sorted(g["gated"]):
        gt = g["gated"][th]
        L.append(_book_row(
            f"gated @ P >= {th:.1f} (kept {gt['kept']}, dropped "
            f"{gt['dropped']})", gt["book"]))
        L.append(
            f"Shadow grades @ {th:.1f}: kept signals avg R "
            f"{gt['shadow']['avg_r']:+.3f} (n={gt['shadow']['n']}) vs "
            f"dropped signals avg R {gt['dropped_shadow']['avg_r']:+.3f} "
            f"(n={gt['dropped_shadow']['n']}) — the gate ranks signals "
            "if kept > dropped.")
    L += ["", "## Trading test 2 — standalone alerts (no detector)", "",
          "Every ensemble alert (24-bar cooldown) traded directly on "
          "the label's geometry (stop 1x ATR, target 2x ATR) through "
          "the Phase-2 v2 exit machinery:", "",
          "| Threshold | Alerts | Simulated | Shadow avg R | Book trades | Book return | Book PF |",
          "|---|---|---|---|---|---|---|"]
    for th in sorted(r.standalone):
        st = r.standalone[th]
        L.append(f"| {th:.1f} | {st['alerts']} | {st['simulated']} | "
                 f"{st['shadow']['avg_r']:+.3f} (n={st['shadow']['n']}) | "
                 f"{st['book']['trades']} | {st['book']['return_pct']:+.2f}% | "
                 f"{st['book']['pf']:.2f} |")
    c = r.control
    if c:
        L += ["", "## Trading test 3 — permutation control (the null)", "",
              "The standalone book trades the label's own barrier "
              "geometry, so a control is mandatory: the (P_up, P_dn) "
              "pairs are permuted across out-of-fold bars (same score "
              "distribution, timing destroyed) and the standalone "
              f"book @ {c['threshold']:.1f} is re-run over "
              f"{c['seeds']} seeds.", "",
              f"- Shuffled-alert book return: mean "
              f"{c['book_return_mean']:+.2f}% (range "
              f"{c['book_return_min']:+.2f}% … "
              f"{c['book_return_max']:+.2f}%), mean PF "
              f"{c['book_pf_mean']:.2f}, shadow avg R "
              f"{c['shadow_avg_r_mean']:+.3f}.",
              f"- Real-alert book @ {c['threshold']:.1f} (from test "
              f"2): {r.standalone[c['threshold']]['book']['return_pct']:+.2f}%, "
              f"shadow avg R {r.standalone[c['threshold']]['shadow']['avg_r']:+.3f}.",
              "- Real minus shuffled is the part of the standalone "
              "result attributable to WHEN the model fires, as opposed "
              "to the geometry it fires into.  The verdict below "
              "reads them together."]
    return "\n".join(L) + "\n"


def main() -> None:
    r = run()
    report = build_report(r)
    out = REPO_ROOT / "FORECAST.md"
    out.write_text(report)
    print(report)
    print(f"[written to {out}]")


if __name__ == "__main__":
    main()
