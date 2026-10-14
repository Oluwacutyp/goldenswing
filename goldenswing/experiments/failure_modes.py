"""Failure-mode diagnosis on losing trades: direction failure vs timing failure.

The owner's distinction (refining the inverse work): when a strategy's
buy loses, there are TWO different failures demanding opposite
responses:

1. DIRECTION failure: price goes down and STAYS down (the adverse
   direction prevails) -> the detection found something real but the
   label is wrong -> flip candidate.
2. TIMING failure: price dips, hits the buy's stop, THEN rallies the
   predicted way -> the direction was right, the entry was early ->
   do NOT flip; fix the entry timing instead.

Method: for every losing trade that exited via its initial structural
stop (clean failures only -- partial-banked, trailed, breakeven and
timeout exits are counted separately, not diagnosed), measure the NET
drift after the stop over the next 12/24 bars, in units of the trade's
own risk R:

    drift_w = direction * (close[exit+w] - stop_price) / risk

Net drift (not max excursion): the owner's scenarios are about what
PREVAILS -- "goes down and STAYS down" vs "rallies the predicted way".
A wick that touches back and fades is not a rally.  Net drift is
symmetric under the null (a random walk after the stop has no
preferred side), so the classifier is unbiased by construction --
verified against the random-entry control in run_random_control().

  TIMING:    drift_24 >= +1.0R  (direction prevailed; entry was early)
  DIRECTION: drift_24 <= -1.0R  (adverse prevailed; label was wrong)
  NEITHER:   everything else (chop)

Max favorable/adverse excursions are still recorded (they inform the
entry-refinement suggestion: how far back did it come at best), but
they do not classify.

This is diagnosis, not a signal: drift is measured from completed bars
after the trade died.  The strategies' own signals remain causal by
construction (reused runners, untouched).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from ..backtest.data import load_gold_h1
from ..backtest.harness import Outcome
from ..features.engine import read_and_featurize
from ..golden.detector import DetectorConfig, GoldenSwingDetector
from ..reader.sessions import session_of
from .placement import PlacementOutcome, _dedupe, _simulate
from .placement import (_t1_signals, _t2_signals, _t3_signals, _t4_signals,
                        FAIL_BARS)

#: Post-stop drift windows, in bars.
DRIFT_WINDOWS = (12, 24)
#: Minimum post-exit bars required to classify a trade.
MIN_POST_BARS = 12
#: Net drift >= this (in R) after the stop -> the direction was right.
TIMING_DRIFT_R = 1.0
#: Net drift <= this (in R) after the stop -> the direction was wrong.
DIRECTION_DRIFT_R = -1.0


class FailureKind(Enum):
    DIRECTION = "direction"   # adverse prevailed -> flip candidate
    TIMING = "timing"        # came back -> entry-refinement candidate
    NEITHER = "neither"      # chop / unclear -> stays buried
    INSUFFICIENT = "insufficient"  # < MIN_POST_BARS of post-exit data


@dataclass
class LoserRecord:
    strategy: str
    signal_index: int
    direction: int
    session: str
    half: str               # "H1" | "H2"
    r_net: float
    drift_12: float | None
    drift_24: float | None
    max_recovery_24: float | None  # best excursion back toward signal dir
    max_continued_24: float | None  # worst excursion against signal dir
    kind: FailureKind
    window_bars: int        # post-exit bars actually available


@dataclass
class StrategyDiagnosis:
    strategy: str
    n_signals: int
    n_trades: int
    n_losers: int
    n_other_exits: int      # losers not via initial stop (be/trail/timeout/partial)
    records: list[LoserRecord] = field(default_factory=list)

    def split(self) -> dict[str, int]:
        out = {k.value: 0 for k in FailureKind}
        for r in self.records:
            out[r.kind.value] += 1
        return out

    def split_by_half(self) -> dict[str, dict[str, int]]:
        halves: dict[str, dict[str, int]] = {
            "H1": {k.value: 0 for k in FailureKind},
            "H2": {k.value: 0 for k in FailureKind},
        }
        for r in self.records:
            halves[r.half][r.kind.value] += 1
        return halves


def classify(drift_24: float) -> FailureKind:
    """Pure classification rule on NET post-stop drift, unit-tested."""
    if drift_24 >= TIMING_DRIFT_R:
        return FailureKind.TIMING
    if drift_24 <= DIRECTION_DRIFT_R:
        return FailureKind.DIRECTION
    return FailureKind.NEITHER


def _drift(df_close: np.ndarray, df_high: np.ndarray, df_low: np.ndarray,
           exit_idx: int, d: int, ref: float, risk: float, window: int,
           n: int) -> tuple[float, float, float, int] | None:
    """(net_drift_R, max_recovery_R, max_continued_R, bars_used) after exit.

    Net drift uses the close ``window`` bars after the exit bar vs the
    stop price.  Max excursions use wicks over the same span.  Returns
    None when fewer than MIN_POST_BARS are available.
    """
    lo = exit_idx + 1
    hi = min(exit_idx + window, n - 1)
    used = hi - lo + 1
    if used < MIN_POST_BARS:
        return None
    drift = d * (float(df_close[hi]) - ref) / risk
    seg_h = df_high[lo:hi + 1]
    seg_l = df_low[lo:hi + 1]
    if d == 1:
        max_recovery = (float(seg_h.max()) - ref) / risk
        max_continued = (ref - float(seg_l.min())) / risk
    else:
        max_recovery = (ref - float(seg_l.min())) / risk
        max_continued = (float(seg_h.max()) - ref) / risk
    return drift, max_recovery, max_continued, used


def diagnose_trades(df: pd.DataFrame,
                    trades: list[dict]) -> list[LoserRecord]:
    """Diagnose a list of normalized trade dicts.

    Each trade dict: strategy, signal_index, direction, entry, stop,
    risk, exit_index, exit_reason, r_net, partial_taken.
    Only clean initial-stop losers are classified; the rest are
    counted by the caller.
    """
    n = len(df)
    df_close = df["close"].to_numpy(dtype=float)
    df_high = df["high"].to_numpy(dtype=float)
    df_low = df["low"].to_numpy(dtype=float)
    half = n // 2
    records: list[LoserRecord] = []
    for t in trades:
        if not (t["exit_reason"] == "stop" and t["r_net"] < 0
                and not t["partial_taken"]):
            continue
        d = t["direction"]
        ref = t["stop"]
        risk = t["risk"]
        if risk <= 0:
            continue
        w12 = _drift(df_close, df_high, df_low, t["exit_index"], d,
                     ref, risk, 12, n)
        w24 = _drift(df_close, df_high, df_low, t["exit_index"], d,
                     ref, risk, 24, n)
        if w24 is None:
            kind = FailureKind.INSUFFICIENT
            dr12 = dr24 = mr24 = mc24 = None
            used = 0
        else:
            dr12 = w12[0] if w12 is not None else None
            dr24, mr24, mc24, used = w24
            kind = classify(dr24)
        ts = df.index[t["signal_index"]]
        session, _ = session_of(ts)
        records.append(LoserRecord(
            strategy=t["strategy"],
            signal_index=t["signal_index"],
            direction=d,
            session=session,
            half="H1" if t["signal_index"] < half else "H2",
            r_net=t["r_net"],
            drift_12=dr12, drift_24=dr24,
            max_recovery_24=mr24, max_continued_24=mc24,
            kind=kind, window_bars=used))
    return records


# ------------------------------------------------- trade normalization
def _norm_placement(oc: PlacementOutcome, strategy: str) -> dict:
    d = oc.direction
    risk = (oc.entry - oc.stop) * d
    return {
        "strategy": strategy, "signal_index": oc.signal_index,
        "direction": d, "entry": oc.entry, "stop": oc.stop, "risk": risk,
        # entry at signal+1; hold_bars measured from the entry bar.
        "exit_index": oc.signal_index + 1 + oc.hold_bars,
        "exit_reason": oc.exit_reason, "r_net": oc.r_net,
        "partial_taken": False,
    }


def _norm_outcome(df: pd.DataFrame, oc: Outcome, strategy: str,
                  stop_buffer_atr: float = 0.25) -> dict | None:
    o = df["open"].to_numpy(dtype=float)
    d = oc.signal.direction
    entry = float(o[oc.entry_index])
    atr = oc.signal.atr
    if not atr or atr <= 0:
        return None
    stop = oc.signal.invalidation - d * stop_buffer_atr * atr
    risk = (entry - stop) * d
    if risk <= 0:
        return None
    return {
        "strategy": strategy, "signal_index": oc.signal.index,
        "direction": d, "entry": entry, "stop": stop, "risk": risk,
        "exit_index": oc.exit_index, "exit_reason": oc.exit_reason,
        "r_net": oc.r_net, "partial_taken": oc.partial_taken,
    }


# ------------------------------------------------------- strategy runners
def _run_placement_probe(df: pd.DataFrame, states, scanner,
                         strategy: str) -> tuple[list[dict], int, int]:
    """Bias-variant outcomes for a placement probe.  Returns
    (normalized trades, n_signals, n_trades)."""
    trades: list[dict] = []
    n_signals = 0
    for sig_bar, bias_dir, stop_bias, _ in _dedupe(
            scanner(df, states, fail_bars=FAIL_BARS)):
        n_signals += 1
        oc = _simulate(df, sig_bar, bias_dir, stop_bias, strategy)
        if oc is not None:
            trades.append(_norm_placement(oc, strategy))
    return trades, n_signals, len(trades)


def _run_placement_inverse(df: pd.DataFrame, states, scanner,
                          strategy: str) -> tuple[list[dict], int, int]:
    trades: list[dict] = []
    n_signals = 0
    for sig_bar, bias_dir, _, stop_inv in _dedupe(
            scanner(df, states, fail_bars=FAIL_BARS)):
        n_signals += 1
        oc = _simulate(df, sig_bar, -bias_dir, stop_inv, strategy)
        if oc is not None:
            trades.append(_norm_placement(oc, strategy))
    return trades, n_signals, len(trades)


def run_all(df: pd.DataFrame | None = None) -> list[StrategyDiagnosis]:
    """Run every strategy's losers through the diagnosis.

    Reuses the experiments' own scanners/simulators; nothing is
    rebuilt, only re-driven to keep per-trade outcomes.
    """
    from .engulf_trigger import scan_variant as engulf_scan
    from .engulf_trigger import simulate_hits as engulf_sim
    from .marco_sweep import scan_variant as marco_scan
    from .marco_sweep import simulate_hits as marco_sim

    if df is None:
        df = load_gold_h1()
    states = read_and_featurize(df)
    det = GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True))

    jobs: list[tuple[str, list[dict], int, int]] = []

    for name, scanner in (("T1_failed_pattern", _t1_signals),
                          ("T2_pin_violation", _t2_signals),
                          ("T3_zone_breaker", _t3_signals),
                          ("T4_doji_expansion", _t4_signals)):
        trades, n_sig, n_tr = _run_placement_probe(df, states, scanner, name)
        jobs.append((name, trades, n_sig, n_tr))

    # T2 inverse: the catastrophic -1.32R variant.  If its losers are
    # mostly DIRECTION failures, the bias direction was right all along
    # (asymmetry confirmation, not a flip).
    trades, n_sig, n_tr = _run_placement_inverse(
        df, states, _t2_signals, "T2_pin_violation_inv")
    jobs.append(("T2_pin_violation_inv", trades, n_sig, n_tr))

    hits = engulf_scan(df, states, det, "B")
    eng_trades = []
    for oc, _ in engulf_sim(df, hits):
        norm = _norm_outcome(df, oc, "engulf_trigger_B")
        if norm is not None:
            eng_trades.append(norm)
    jobs.append(("engulf_trigger_B", eng_trades, len(hits), len(eng_trades)))

    mhits = marco_scan(df, states, det, "B")
    mar_trades = []
    for oc in marco_sim(df, mhits, "B"):
        norm = _norm_outcome(df, oc, "marco_sweep_B")
        if norm is not None:
            mar_trades.append(norm)
    jobs.append(("marco_sweep_B", mar_trades, len(mhits), len(mar_trades)))

    diagnoses: list[StrategyDiagnosis] = []
    for name, trades, n_sig, n_tr in jobs:
        records = diagnose_trades(df, trades)
        n_losers = sum(1 for t in trades if t["r_net"] < 0)
        n_diag = len(records)
        diagnoses.append(StrategyDiagnosis(
            strategy=name, n_signals=n_sig, n_trades=n_tr,
            n_losers=n_losers, n_other_exits=n_losers - n_diag,
            records=records))
    return diagnoses


def verdict(split: dict[str, int]) -> str:
    """Flip / refine / buried from a classified split."""
    classified = (split["direction"] + split["timing"] + split["neither"])
    if classified == 0:
        return "no-data"
    d_share = split["direction"] / classified
    t_share = split["timing"] / classified
    if d_share >= 0.60:
        return "flip-candidate"
    if t_share >= 0.60:
        return "refine-entry"
    return "stays-buried"


def run_random_control(df: pd.DataFrame, n: int = 2000, seed: int = 7,
                       stop_atr_mults: tuple[float, ...] = (0.5, 1.0, 2.0)
                       ) -> list[StrategyDiagnosis]:
    """Classifier calibration: random bars, random direction, generic
    structural stop (``stop_atr_mult`` x ATR), fixed 2R target -- the
    same simulation conventions as the placement probes.

    If the classifier labels random trades as TIMING at a high rate,
    the strategy verdicts above are miscalibrated.  Run at several
    stop widths to check stability.
    """
    from .placement import _simulate as placement_sim

    rng = np.random.default_rng(seed)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    # Wilder ATR(14) for the generic structural stop.
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(h - low,
                    np.maximum(np.abs(h - prev_close),
                               np.abs(low - prev_close)))
    atr = pd.Series(tr).ewm(alpha=1.0 / 14.0, min_periods=14,
                            adjust=False).mean().to_numpy()
    diags: list[StrategyDiagnosis] = []
    for mult in stop_atr_mults:
        name = f"control_random_{mult}atr"
        trades: list[dict] = []
        n_sig = 0
        # random signal bars with room for entry + full horizon + drift
        lo, hi = 50, len(df) - 72 - 24 - 2
        bars = rng.integers(lo, hi, size=n)
        for t in sorted(set(int(b) for b in bars)):
            a = atr[t]
            if not np.isfinite(a) or a <= 0:
                continue
            d = 1 if rng.random() < 0.5 else -1
            entry_ref = closes[t]
            stop = entry_ref - d * mult * a
            n_sig += 1
            oc = placement_sim(df, t, d, stop, name)
            if oc is not None:
                trades.append(_norm_placement(oc, name))
        records = diagnose_trades(df, trades)
        n_losers = sum(1 for t in trades if t["r_net"] < 0)
        diags.append(StrategyDiagnosis(
            strategy=name, n_signals=n_sig, n_trades=len(trades),
            n_losers=n_losers, n_other_exits=n_losers - len(records),
            records=records))
    return diags
