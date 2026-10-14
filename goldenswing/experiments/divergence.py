"""RSI divergence as a feature — experiment measurements.

Mined claim (TradingLab video, user-mined 2026-10-11): RSI divergence
(price makes a new extreme, RSI doesn't) is an early sign of a new
move.  Regular divergence = reversal; hidden divergence = continuation.

Two roles under test (2y GC=F H1, repo conventions, $0.45/oz):

  1. SCORED FAMILY — ``COMBO_A | {aoc}`` (anchor) vs
     ``COMBO_A | {aoc, divergence}``: does the family add return/R to
     the traded book?  Tested at RSI(14) and RSI(9) (the video's length).
  2. ENTRY CONTEXT — anchor signals split by the divergence state at
     the signal bar: aligned (any type), opposing (any type), none —
     and per divergence type.  Shadow R comparison: does divergence
     carry signal, and do regular/hidden do opposite jobs as claimed?

Causality: pivots are fractals confirmed with a 2-bar lag; a divergence
is known only from its second pivot's confirmation bar.  Nothing at bar
t uses information past t.  See features/divergence.py for the
pre-committed rule set.
"""
from __future__ import annotations

import numpy as np

from ..backtest.ablation import P2_FLAGS
from ..backtest.harness import book_metrics, build_book, simulate_signal_v2
from ..backtest.validation import COMBO_A, run_config
from ..features import FeatureEngine
from ..features.config import FeatureConfig
from ..golden.detector import DetectorConfig, GoldenSwingDetector
from ..reader.reader import MarketReader

ANCHOR_FAMS = COMBO_A | {"aoc"}
TEST_FAMS = COMBO_A | {"aoc", "divergence"}

KINDS = ("bull_reg", "bear_reg", "bull_hid", "bear_hid")


def featurize(df, peers=None, rsi_period: int = 14,
              min_drsi: float = 2.0):
    """Reader + feature states with divergence configured."""
    cfg = FeatureConfig(div_rsi_period=rsi_period, div_min_drsi=min_drsi)
    states = MarketReader().read(df)
    FeatureEngine(cfg).attach(df, states, peers)
    return states


def run_family_test(df, states):
    """Role 1: anchor book vs anchor + divergence family."""
    anchor = run_config(df, states, ANCHOR_FAMS, "anchor")
    test = run_config(df, states, TEST_FAMS, "divergence")
    return anchor, test


def anchor_outcomes(df, states):
    """All anchor signals simulated (shadow universe for role 2)."""
    detector = GoldenSwingDetector(DetectorConfig(
        emit_floor=35.0, golden_threshold=60.0,
        feature_families=ANCHOR_FAMS, **P2_FLAGS))
    signals = detector.scan(states)
    outcomes = []
    for sig in signals:
        o = simulate_signal_v2(df, sig)
        if o is not None:
            outcomes.append(o)
    return outcomes


def _pf(rs: list[float]) -> float:
    wins = sum(r for r in rs if r > 0)
    losses = -sum(r for r in rs if r < 0)
    return wins / losses if losses > 0 else float("inf")


def _bucket_stats(rs: list[float]) -> dict:
    n = len(rs)
    if n == 0:
        return {"n": 0, "avg_r": float("nan"), "win": float("nan"),
                "pf": float("nan")}
    return {"n": n, "avg_r": float(np.mean(rs)),
            "win": float(np.mean([r > 0 for r in rs])), "pf": _pf(rs)}


def context_table(states, outcomes):
    """Role 2: anchor-signal shadow R by divergence state at the signal
    bar.  Buckets: aligned (any type), opposing (any type), none, plus
    per-type aligned buckets.  A signal may sit in several per-type
    buckets; the three top-level buckets are exclusive
    (aligned > opposing > none)."""
    buckets: dict[str, list[float]] = {
        "aligned_any": [], "opposing_any": [], "none": [],
        **{f"aligned_{k}": [] for k in KINDS},
    }
    for o in outcomes:
        d = o.signal.direction
        st = states[o.signal.index].features.divergence
        al = st.aligned(d)
        op = st.opposing(d)
        if al is not None:
            buckets["aligned_any"].append(o.r_net)
        elif op is not None:
            buckets["opposing_any"].append(o.r_net)
        else:
            buckets["none"].append(o.r_net)
        for k in KINDS:
            dv = st.of_kind(k)
            if dv is not None and dv.direction == d:
                buckets[f"aligned_{k}"].append(o.r_net)
    return {k: _bucket_stats(v) for k, v in buckets.items()}


def divergence_census(states):
    """How many divergences of each type fired (confirmed), and the
    confirmation-lag / age profile."""
    counts = {k: 0 for k in KINDS}
    # count distinct (kind, confirmed_at) pairs across all bars
    seen = set()
    for s in states:
        for dv in s.features.divergence.divergences:
            key = (dv.kind, dv.confirmed_at)
            if key not in seen:
                seen.add(key)
                counts[dv.kind] += 1
    return counts


def gate_firing_rate(df, states):
    """Diagnostic: how many emitted divergence-family signals carried a
    divergence factor (the gate actually spoke)."""
    detector = GoldenSwingDetector(DetectorConfig(
        emit_floor=35.0, golden_threshold=60.0,
        feature_families=TEST_FAMS, **P2_FLAGS))
    signals = detector.scan(states)
    hits = sum(1 for s in signals
               if any("divergence" in f for f in s.factors))
    return hits, len(signals)
