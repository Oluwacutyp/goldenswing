"""Multi-horizon forecaster evaluation (Phase 6.5).

The owner correction: *"Not limited to 3 days though."*  Phase 6
hard-coded the 72-bar (3-day) label.  This module runs the SAME
purged walk-forward machinery at every registered horizon
(``labels.HORIZON_SPECS``: 12 / 24 / 72 / 168 / 360 H1 bars), each
with its own barrier geometry, its own purge (embargo = that
horizon's window) and its own trading clock (standalone trades exit
through the v2 machinery with the time stop set to the horizon).

Discipline notes:

- Labels at adjacent bars overlap heavily, and the overlap grows
  with the horizon: a 360-bar label shares its outcome window with
  ~359 neighbours.  Raw row counts therefore flatter long horizons,
  so every horizon reports an EFFECTIVE sample size —
  ``labelled_rows // horizon``, the count of non-overlapping outcome
  windows — and verdicts shrink to fit it.  A horizon with ~17
  effective out-of-fold windows cannot support a strong claim
  whatever its AUC prints.
- The 72-bar row is the Phase-6 regression anchor: same data, same
  folds, same geometry, same seeds — it must reproduce FORECAST.md's
  numbers, and the run checks that it does.
- The multi-horizon AGREEMENT test asks whether requiring the 24h,
  72h and 168h ensembles to fire the same direction at the same bar
  makes a better filter on the detector book than the 72h gate
  alone — same arena (the intersection of the three OOF regions),
  same signals, three books: plain / gated@72 / gated@agreement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.data import REPO_ROOT, load_gold_h1, load_peers
from ..backtest.harness import (ExitConfig, book_metrics, build_book,
                                simulate_signal_v2, slice_stats)
from ..features import FeatureEngine
from ..golden.detector import GoldenSignal, GoldenSwingDetector
from ..reader.reader import MarketReader
from . import evaluate, trading
from .features import build_feature_frame, combo_a_config, detector_opinions
from .labels import HORIZONS, HorizonSpec, build_labels_for_horizon, spec_for
from .walkforward import run_walk_forward

#: Horizons that must agree for the multi-horizon alert (1d + 3d + 1w).
AGREEMENT_HORIZONS: tuple[int, ...] = (24, 72, 168)
AGREEMENT_THRESHOLD = 0.5

#: Phase-6 anchors (FORECAST.md) the 72-bar row must reproduce.
PHASE6_ANCHORS = {
    "base_rate_up": (0.347, 0.002),
    "base_rate_dn": (0.334, 0.002),
    "ens_auc_up": (0.517, 0.002),
    "ens_auc_dn": (0.526, 0.002),
    "standalone_05_return": (31.96, 0.05),
    "standalone_05_pf": (1.24, 0.02),
    "standalone_05_shadow_r": (0.107, 0.002),
    "standalone_06_return": (12.71, 0.05),
    "control_return_mean": (-3.32, 0.05),
}


def effective_samples(labelled_rows: int, horizon: int) -> int:
    """Crude independence count: how many NON-overlapping outcome
    windows fit in the labelled rows.  Deliberately conservative —
    adjacent labels overlap by construction, so this, not the raw
    row count, is the honest denominator for a horizon's claims."""
    return int(labelled_rows // horizon)


def cooldown_for(horizon: int) -> int:
    """Standalone-alert cooldown per horizon: the Phase-6 flat 24
    bars at/below the 3-day horizon, a third of the window above it
    (a 2-week forecast that re-fires every day is one alert, not
    fourteen)."""
    return max(trading.ALERT_COOLDOWN, horizon // 3)


def calibration_summary(calib: pd.DataFrame) -> dict:
    """Compact reliability readout from a decile table: what the
    bottom and top deciles actually did, and the mean absolute
    predicted-vs-empirical gap across deciles."""
    if calib.empty:
        return {"bottom_emp": float("nan"), "top_emp": float("nan"),
                "mae": float("nan"), "pred_spread": float("nan")}
    w = calib["n"].to_numpy(dtype=float)
    mae = float(np.average(
        np.abs(calib["mean_pred"] - calib["empirical"]), weights=w))
    return {
        "bottom_emp": float(calib["empirical"].iloc[0]),
        "top_emp": float(calib["empirical"].iloc[-1]),
        "mae": mae,
        "pred_spread": float(calib["mean_pred"].iloc[-1]
                             - calib["mean_pred"].iloc[0]),
    }


def agreement_directions(
        probs_by_horizon: dict[int, pd.DataFrame],
        horizons: tuple[int, ...] = AGREEMENT_HORIZONS,
        threshold: float = AGREEMENT_THRESHOLD) -> dict[int, int]:
    """Bars where EVERY listed horizon's ensemble fires the same
    direction at ``threshold``.  Returns {bar_position: direction}.
    If both directions qualify at a bar (possible — the per-direction
    models are not complements), the direction whose WEAKEST horizon
    score is higher wins; an exact tie is skipped, never guessed."""
    frames = [probs_by_horizon[h] for h in horizons]
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    out: dict[int, int] = {}
    for t in common:
        mins = {}
        for dname in ("up", "dn"):
            col = f"ens_{dname}"
            mins[dname] = min(float(f.loc[t, col]) for f in frames)
        up_ok = mins["up"] >= threshold
        dn_ok = mins["dn"] >= threshold
        if up_ok and dn_ok:
            if mins["up"] == mins["dn"]:
                continue
            out[int(t)] = 1 if mins["up"] > mins["dn"] else -1
        elif up_ok:
            out[int(t)] = 1
        elif dn_ok:
            out[int(t)] = -1
    return out


@dataclass
class HorizonResult:
    spec: HorizonSpec
    folds: int
    oof_bars: int
    labelled_bars: int
    n_eff_labelled: int
    n_eff_oof: int
    base_rate_up: float
    base_rate_dn: float
    auc: pd.DataFrame
    quality_up: pd.DataFrame
    quality_dn: pd.DataFrame
    calib_up: dict
    calib_dn: dict
    standalone: dict
    control: dict

    def ens_auc(self, direction: str) -> float:
        row = self.auc[(self.auc["direction"] == direction)
                       & (self.auc["model"] == "ensemble")]
        return float(row["auc"].iloc[0]) if len(row) else float("nan")


@dataclass
class AgreementResult:
    arena_signals: int
    plain_book: dict
    gate72_book: dict
    gate72_kept: int
    gate72_shadow: dict
    gate72_dropped_shadow: dict
    agree_book: dict
    agree_kept: int
    agree_shadow: dict
    agree_dropped_shadow: dict
    standalone_alerts: int
    standalone_book: dict
    standalone_shadow: dict


@dataclass
class HorizonsRun:
    bars: int
    start: str
    end: str
    results: dict[int, HorizonResult]
    agreement: AgreementResult
    reproduction: dict = field(default_factory=dict)


def _standalone_for_horizon(df, probs, atr, spec: HorizonSpec,
                            years: float) -> tuple[dict, dict]:
    kw = dict(cooldown=cooldown_for(spec.horizon),
              fav_mult=spec.fav_mult, adv_mult=spec.adv_mult,
              exits=ExitConfig(horizon=spec.horizon))
    standalone = trading.standalone_test(df, probs, atr, years, **kw)
    control = trading.permutation_control(df, probs, atr, years, **kw)
    return standalone, control


def _agreement(df, signals, probs_by_h, atr, years) -> AgreementResult:
    agree = agreement_directions(probs_by_h)
    p72 = probs_by_h[72]
    common = p72.index
    for h in AGREEMENT_HORIZONS:
        common = common.intersection(probs_by_h[h].index)

    outcomes = []
    for sig in signals:
        if sig.index not in common:
            continue
        o = simulate_signal_v2(df, sig)
        if o is not None:
            outcomes.append(o)

    def gate(keep_fn):
        kept = [o for o in outcomes if keep_fn(o)]
        dropped = [o for o in outcomes if not keep_fn(o)]
        return kept, dropped

    kept72, dropped72 = gate(
        lambda o: float(p72.loc[o.signal.index,
                                "ens_up" if o.signal.direction == 1
                                else "ens_dn"]) >= AGREEMENT_THRESHOLD)
    kept_ag, dropped_ag = gate(
        lambda o: agree.get(o.signal.index) == o.signal.direction)

    # standalone agreement alerts on the 72-bar trade clock/geometry
    spec72 = spec_for(72)
    close = df["close"].to_numpy(dtype=float)
    sigs: list[GoldenSignal] = []
    last = -10**9
    for t in sorted(agree):
        if t - last < trading.ALERT_COOLDOWN:
            continue
        d = agree[t]
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        binding = min(float(probs_by_h[h].loc[
            t, "ens_up" if d == 1 else "ens_dn"])
            for h in AGREEMENT_HORIZONS)
        sigs.append(GoldenSignal(
            index=int(t), timestamp=df.index[t], direction=d,
            score=float(100.0 * binding), kind="forecast_agreement",
            factors=[f"multi-horizon agreement "
                     f"{'/'.join(str(h) for h in AGREEMENT_HORIZONS)} "
                     f"({'up' if d == 1 else 'down'}), weakest link "
                     f"P = {binding:.2f}"],
            entry_ref=float(close[t]),
            invalidation=float(close[t] - d * spec72.adv_mult * a),
            targets=[float(close[t] + d * spec72.fav_mult * a)],
            atr=float(a)))
        last = t
    sa_outcomes = []
    for sig in sigs:
        o = simulate_signal_v2(df, sig, ExitConfig(horizon=72))
        if o is not None:
            sa_outcomes.append(o)

    return AgreementResult(
        arena_signals=len(outcomes),
        plain_book=book_metrics(build_book(outcomes, 60.0), years),
        gate72_book=book_metrics(build_book(kept72, 60.0), years),
        gate72_kept=len(kept72),
        gate72_shadow=slice_stats(kept72),
        gate72_dropped_shadow=slice_stats(dropped72),
        agree_book=book_metrics(build_book(kept_ag, 60.0), years),
        agree_kept=len(kept_ag),
        agree_shadow=slice_stats(kept_ag),
        agree_dropped_shadow=slice_stats(dropped_ag),
        standalone_alerts=len(sigs),
        standalone_book=book_metrics(build_book(sa_outcomes, 0.0), years),
        standalone_shadow=slice_stats(sa_outcomes))


def run_horizons(df: pd.DataFrame | None = None,
                 peers: dict | None = None,
                 horizons: tuple[int, ...] = HORIZONS) -> HorizonsRun:
    if df is None:
        df = load_gold_h1()
    if peers is None:
        peers = load_peers()

    states = MarketReader().read(df)
    FeatureEngine().attach(df, states, peers)
    atr = np.array([s.atr for s in states], dtype=float)
    opinions = detector_opinions(states)
    X = build_feature_frame(states, opinions)
    det_signed = (opinions["det_score"] * opinions["det_dir"])
    det_signed.index = pd.RangeIndex(len(df))

    results: dict[int, HorizonResult] = {}
    probs_by_h: dict[int, pd.DataFrame] = {}
    for h in horizons:
        spec = spec_for(h)
        labels = build_labels_for_horizon(df, atr, h)
        labels.index = pd.RangeIndex(len(df))
        wf = run_walk_forward(X, labels, horizon=h)
        probs = wf.probs
        probs_by_h[h] = probs
        probs.to_csv(REPO_ROOT / "data" / f"forecast_oof_probs_h{h}.csv")

        oof_labels = labels.loc[probs.index]
        oof_span = (df.index[probs.index[-1]] - df.index[probs.index[0]]) \
            .total_seconds() / 86400.0
        years = max(oof_span / 365.25, 1e-9)
        standalone, control = _standalone_for_horizon(
            df, probs, atr, spec, years)
        labelled = int(labels["y_up"].notna().sum())
        results[h] = HorizonResult(
            spec=spec, folds=len(wf.folds), oof_bars=len(probs),
            labelled_bars=labelled,
            n_eff_labelled=effective_samples(labelled, h),
            n_eff_oof=effective_samples(len(probs), h),
            base_rate_up=float(oof_labels["y_up"].mean()),
            base_rate_dn=float(oof_labels["y_dn"].mean()),
            auc=evaluate.auc_table(probs, labels, det_signed),
            quality_up=evaluate.alert_quality(probs, labels,
                                              "ens_up", "y_up"),
            quality_dn=evaluate.alert_quality(probs, labels,
                                              "ens_dn", "y_dn"),
            calib_up=calibration_summary(
                evaluate.calibration(probs, labels, "ens_up", "y_up")),
            calib_dn=calibration_summary(
                evaluate.calibration(probs, labels, "ens_dn", "y_dn")),
            standalone=standalone, control=control)

    # ---- multi-horizon agreement on the detector book ------------------
    missing = [h for h in AGREEMENT_HORIZONS if h not in probs_by_h]
    if missing:
        raise ValueError(f"agreement horizons not evaluated: {missing}")
    detector = GoldenSwingDetector(combo_a_config())
    signals = detector.scan(states)
    agree_probs = {h: probs_by_h[h] for h in AGREEMENT_HORIZONS}
    common = agree_probs[72].index
    for h in AGREEMENT_HORIZONS:
        common = common.intersection(agree_probs[h].index)
    span = (df.index[common[-1]] - df.index[common[0]]) \
        .total_seconds() / 86400.0
    agreement = _agreement(df, signals, agree_probs, atr,
                           max(span / 365.25, 1e-9))

    # ---- Phase-6 reproduction check (72-bar row) ----------------------
    reproduction: dict = {}
    if 72 in results:
        r72 = results[72]
        got = {
            "base_rate_up": r72.base_rate_up,
            "base_rate_dn": r72.base_rate_dn,
            "ens_auc_up": r72.ens_auc("up"),
            "ens_auc_dn": r72.ens_auc("dn"),
            "standalone_05_return":
                r72.standalone[0.5]["book"]["return_pct"],
            "standalone_05_pf": r72.standalone[0.5]["book"]["pf"],
            "standalone_05_shadow_r": r72.standalone[0.5]["shadow"]["avg_r"],
            "standalone_06_return":
                r72.standalone[0.6]["book"]["return_pct"],
            "control_return_mean": r72.control["book_return_mean"],
        }
        for name, (want, tol) in PHASE6_ANCHORS.items():
            val = got[name]
            reproduction[name] = {"expected": want, "got": float(val),
                                  "ok": bool(abs(val - want) <= tol)}

    return HorizonsRun(
        bars=len(df), start=str(df.index[0]), end=str(df.index[-1]),
        results=results, agreement=agreement, reproduction=reproduction)


# ------------------------------------------------------------- verdicts
def verdict_for(r: HorizonResult) -> str:
    """One honest line per horizon, derived from its own numbers:
    ensemble discrimination (worse direction governs), whether the
    standalone book beats its timing-destroyed null, and whether the
    effective sample can carry the claim at all."""
    auc_min = min(r.ens_auc("up"), r.ens_auc("dn"))
    book = r.standalone[0.5]["book"]
    shadow = r.standalone[0.5]["shadow"]
    ctrl = r.control
    beats_null = (book["return_pct"] > ctrl["book_return_mean"]
                  and shadow["avg_r"] > ctrl["shadow_avg_r_mean"])
    if auc_min >= 0.53 and beats_null:
        v = ("REAL SIGNAL — ensemble discriminates in both directions "
             "and its alerts beat the shuffled null as a book.")
    elif auc_min >= 0.515 and beats_null:
        v = ("WEAK BUT REAL — discrimination is a few points over a "
             "coin flip, yet the alerts still beat the null; a "
             "ranking/sizing input, not an oracle.")
    elif max(r.ens_auc("up"), r.ens_auc("dn")) >= 0.515:
        v = ("PARTIAL — only one direction carries anything, or the "
             "alerts do not survive the null; not a standalone input.")
    else:
        v = ("NOISE — ensemble discrimination is at coin-flip level "
             "in both directions; this horizon's forecast is a "
             "lottery ticket and must not be quoted.")
    if r.n_eff_oof < 30:
        v += (f" SAMPLE WARNING: only ~{r.n_eff_oof} effective "
              f"(non-overlapping) out-of-fold windows — too thin to "
              f"grade; treat the line above as provisional at best.")
    return v


def _auc_cell(r: HorizonResult, direction: str) -> str:
    return f"{r.ens_auc(direction):.3f}"


def build_horizons_report(run: HorizonsRun) -> str:
    L: list[str] = []
    L += ["# GoldenSwing — Multi-Horizon Forecaster (Phase 6.5)", "",
          "## What changed", "",
          "The owner's correction: *\"Not limited to 3 days though.\"*  "
          "Phase 6 hard-coded the 72-bar (3-day) label.  This phase "
          "generalises the forecaster to every registered horizon and "
          "grades each one the house way — purged walk-forward, "
          "standalone trading test, permutation null — on the same "
          "data window as Phases 1-6 (GC=F H1, "
          f"{run.bars:,} bars, {run.start} → {run.end}).", "",
          "## Horizons and their geometry", "",
          "A big move at bar t = favourable excursion reaching "
          "fav x ATR(t) BEFORE adverse excursion reaches adv x ATR(t) "
          "inside the window (same-bar double touch = adverse wins; "
          "labels mutually exclusive; incomplete windows unlabelled).  "
          "Barriers widen with the window (price wanders further "
          "given more time); the fav:adv ratio stays the house 2:1.  "
          "Standalone trades use the matching geometry and a time "
          "stop equal to the horizon.", "",
          "| Horizon | Tag | Window | Fav | Adv | Alert cooldown |",
          "|---|---|---|---|---|---|"]
    for h in sorted(run.results):
        s = run.results[h].spec
        L.append(f"| {s.name} | {s.tag} | {s.horizon} bars | "
                 f"{s.fav_mult:.2f} ATR | {s.adv_mult:.2f} ATR | "
                 f"{cooldown_for(h)} bars |")
    L += ["", "## Method notes", "",
          "- Walk-forward identical to Phase 6 (expanding window, "
          "first test block after 5,000 bars, retrain every 500), "
          "except the PURGE scales with the horizon: a training row "
          "is admitted only if its whole label window ends before "
          "the test block opens.",
          "- EFFECTIVE SAMPLES: labels at neighbouring bars share "
          "outcome windows, so raw row counts lie — worst at long "
          "horizons.  n_eff = labelled rows / horizon (non-overlapping "
          "windows) is the honest denominator and is printed per "
          "horizon; verdicts shrink to fit it.",
          "- Standalone test + permutation control per horizon, same "
          "as Phase 6: the control permutes (P_up, P_dn) pairs across "
          "bars — same distribution, timing destroyed — and re-books "
          "the alerts.  Real-minus-shuffled is the part attributable "
          "to WHEN the model fires.", "",
          "## Comparison — all horizons, out-of-fold", "",
          "| Horizon | Base up | Base dn | AUC up | AUC dn | OOF bars | n_eff (OOF) | Standalone @0.5: return (PF, trades) | Null: return | Verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for h in sorted(run.results):
        r = run.results[h]
        b = r.standalone[0.5]["book"]
        L.append(
            f"| {r.spec.tag} ({h}) | {r.base_rate_up:.3f} | "
            f"{r.base_rate_dn:.3f} | {_auc_cell(r, 'up')} | "
            f"{_auc_cell(r, 'dn')} | {r.oof_bars:,} | {r.n_eff_oof} | "
            f"{b['return_pct']:+.2f}% ({b['pf']:.2f}, {b['trades']}) | "
            f"{r.control['book_return_mean']:+.2f}% | "
            f"{verdict_for(r).split(' — ')[0]} |")
    L += ["", "## Per-horizon detail", ""]
    for h in sorted(run.results):
        r = run.results[h]
        s = r.spec
        L += [f"### {s.name} — {h} bars ({s.tag}), fav "
              f"{s.fav_mult:.2f} / adv {s.adv_mult:.2f} ATR", "",
              f"Labelled bars: {r.labelled_bars:,} "
              f"(n_eff ~{r.n_eff_labelled}); OOF bars: {r.oof_bars:,} "
              f"(n_eff ~{r.n_eff_oof}); folds: {r.folds}; base rates "
              f"up {r.base_rate_up:.3f} / dn {r.base_rate_dn:.3f}.", "",
              "| Direction | Model | AUC | n |", "|---|---|---|---|"]
        for _, row in r.auc.iterrows():
            L.append(f"| {row['direction']} | {row['model']} | "
                     f"{row['auc']:.3f} | {row['n']} |")
        L += ["", "Alert quality (ensemble):", "",
              "| Direction | P thr | Alerts | Precision | Recall | Base |",
              "|---|---|---|---|---|---|"]
        for dname, q in (("up", r.quality_up), ("dn", r.quality_dn)):
            for _, row in q.iterrows():
                L.append(f"| {dname} | {row['threshold']:.1f} | "
                         f"{int(row['alerts'])} | "
                         f"{row['precision']:.3f} | {row['recall']:.3f} | "
                         f"{row['base_rate']:.3f} |")
        cu, cd = r.calib_up, r.calib_dn
        L += ["",
              f"Calibration (deciles): up — bottom-decile empirical "
              f"{cu['bottom_emp']:.3f}, top-decile {cu['top_emp']:.3f}, "
              f"mean |pred-emp| {cu['mae']:.3f}; down — bottom "
              f"{cd['bottom_emp']:.3f}, top {cd['top_emp']:.3f}, MAE "
              f"{cd['mae']:.3f}. As in Phase 6 these are ranking "
              "scores, not probabilities — read the decile SPREAD, "
              "not the level.", "",
              "Standalone alerts (label geometry, horizon time stop):",
              "",
              "| P thr | Alerts | Shadow avg R | Book return | Book PF |",
              "|---|---|---|---|---|"]
        for th in sorted(r.standalone):
            st = r.standalone[th]
            L.append(f"| {th:.1f} | {st['alerts']} | "
                     f"{st['shadow']['avg_r']:+.3f} "
                     f"(n={st['shadow']['n']}) | "
                     f"{st['book']['return_pct']:+.2f}% | "
                     f"{st['book']['pf']:.2f} |")
        c = r.control
        L += [f"Permutation null @ 0.5: mean book return "
              f"{c['book_return_mean']:+.2f}% (range "
              f"{c['book_return_min']:+.2f}% … "
              f"{c['book_return_max']:+.2f}%), mean shadow avg R "
              f"{c['shadow_avg_r_mean']:+.3f}.", "",
              f"**Verdict: {verdict_for(r)}**", ""]
    a = run.agreement
    L += ["## Multi-horizon agreement — 1d + 3d + 1w firing together", "",
          "Detector (Combo A) signals inside the intersection of the "
          f"three horizons' OOF regions: {a.arena_signals}.  Three "
          "books on the SAME arena outcomes; a gate keeps a signal "
          "only when its own direction qualifies.", "",
          "| Book | Kept | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, book, kept in (
            ("plain detector (arena)", a.plain_book, a.arena_signals),
            ("gated @ 72h P>=0.5", a.gate72_book, a.gate72_kept),
            ("gated @ agreement 24+72+168", a.agree_book, a.agree_kept)):
        L.append(f"| {name} | {kept} | {book['trades']} | "
                 f"{book['win_rate']:.1f} | {book['pf']:.2f} | "
                 f"{book['return_pct']:+.2f}% | "
                 f"{book['max_dd_pct']:.2f}% | {book['sharpe']:.2f} | "
                 f"{book['avg_r']:+.3f} |")
    L += ["",
          f"Shadow grades — agreement gate: kept avg R "
          f"{a.agree_shadow['avg_r']:+.3f} (n={a.agree_shadow['n']}) "
          f"vs dropped {a.agree_dropped_shadow['avg_r']:+.3f} "
          f"(n={a.agree_dropped_shadow['n']}); 72h gate: kept "
          f"{a.gate72_shadow['avg_r']:+.3f} (n={a.gate72_shadow['n']}) "
          f"vs dropped {a.gate72_dropped_shadow['avg_r']:+.3f} "
          f"(n={a.gate72_dropped_shadow['n']}).", "",
          f"Agreement alerts traded standalone (72-bar clock): "
          f"{a.standalone_alerts} alerts, book return "
          f"{a.standalone_book['return_pct']:+.2f}%, PF "
          f"{a.standalone_book['pf']:.2f}, shadow avg R "
          f"{a.standalone_shadow['avg_r']:+.3f} "
          f"(n={a.standalone_shadow['n']}).", ""]
    better = (a.agree_book["return_pct"] > a.gate72_book["return_pct"]
              and a.agree_shadow["avg_r"] > a.gate72_shadow["avg_r"])
    quality_only = (not better
                    and a.agree_shadow["avg_r"] > a.gate72_shadow["avg_r"])
    if better:
        agree_verdict = ("AGREEMENT WINS on this window: it keeps "
                         "fewer signals than the 72h gate, grades "
                         "them higher, AND books more — the three "
                         "clocks genuinely vote, they do not echo.")
    elif quality_only:
        agree_verdict = ("AGREEMENT IS A QUALITY FILTER ONLY: kept "
                         "signals grade higher than the 72h gate's, "
                         "but the book books less — agreement throws "
                         "away profitable trades along with the bad "
                         "ones.  A sizing input, not a switch.")
    else:
        agree_verdict = ("AGREEMENT DOES NOT BEAT THE SINGLE 72h "
                         "GATE on this window: requiring three "
                         "clocks to agree adds no ranking power over "
                         "the 3-day ensemble alone.  The horizons "
                         "mostly echo each other.")
    L += [f"**Agreement verdict: {agree_verdict}**", "",
          "## Phase-6 reproduction check (72-bar row vs FORECAST.md)",
          "",
          "| Quantity | FORECAST.md | This run | Match |",
          "|---|---|---|---|"]
    for name, chk in run.reproduction.items():
        L.append(f"| {name} | {chk['expected']:+.3f} | "
                 f"{chk['got']:+.3f} | "
                 f"{'YES' if chk['ok'] else 'NO'} |")
    L += ["", "## Bottom line", ""]
    for h in sorted(run.results):
        r = run.results[h]
        L.append(f"- **{r.spec.tag} ({h} bars):** {verdict_for(r)}")
    L += ["",
          "Caveats that stand for every horizon: one instrument, one "
          "bullish 2-year window, OOF coverage only from bar 5,000; "
          "the temporal/regime out-of-sample question is still open. "
          "Long-horizon rows rest on a handful of effective windows "
          "and their numbers will move the most when new data "
          "arrives.  Outputs remain ranking scores, NOT calibrated "
          "probabilities, at every horizon."]
    return "\n".join(L) + "\n"
