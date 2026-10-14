"""Trading tests for the forecaster.

Two questions, answered with the Phase-2 exit machinery
(``simulate_signal_v2``) and the harness's sequential one-position
book — the same machinery that produced the +25.62% Combo-A book:

1. GATE test — does the forecaster ADD anything on top of the
   detector?  Take the Combo-A detector's signals; keep only those
   whose bar carries an out-of-fold ensemble probability (for the
   signal's own direction) at/above a threshold.  Compare the gated
   book against the plain book RESTRICTED TO THE SAME out-of-fold
   signal set (the fair comparison), and report the full-window
   plain book as the reproduction anchor.

2. STANDALONE test — if you traded every high-probability alert
   directly (no detector), what happens?  Alerts are fabricated
   signals built on the label's own geometry (stop 1x ATR beyond the
   signal close, target 2x ATR) and run through the same v2 exits;
   the buffered stop / partial / trail machinery means this is the
   label trade as the house would actually manage it, not the raw
   label.  A 24-bar cooldown between alerts limits overlap spam.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.harness import (ExitConfig, book_metrics, build_book,
                                simulate_signal_v2, slice_stats)
from ..golden.detector import GoldenSignal
from .labels import ADV_MULT, FAV_MULT

GATE_THRESHOLDS = (0.5, 0.6)
ALERT_COOLDOWN = 24      # bars between standalone alerts


def gate_test(df: pd.DataFrame, signals: list[GoldenSignal],
              probs: pd.DataFrame, full_years: float,
              oof_years: float) -> dict:
    """``signals`` are the Combo-A detector's scanned signals over the
    full window; ``probs`` the out-of-fold ensemble frame.  Sharpe for
    the full-window book uses ``full_years``; the OOF-restricted books
    use ``oof_years`` (their own span) — returns/DD are unaffected."""
    outcomes = []
    for sig in signals:
        o = simulate_signal_v2(df, sig)
        if o is not None:
            outcomes.append(o)

    full_book = book_metrics(build_book(outcomes, 60.0), full_years)

    oof = [o for o in outcomes if o.signal.index in probs.index]
    plain_oof_book = book_metrics(build_book(oof, 60.0), oof_years)
    plain_oof_shadow = slice_stats(oof)

    gated = {}
    for th in GATE_THRESHOLDS:
        kept, dropped = [], []
        for o in oof:
            col = "ens_up" if o.signal.direction == 1 else "ens_dn"
            (kept if probs.loc[o.signal.index, col] >= th
             else dropped).append(o)
        gated[th] = {
            "kept": len(kept),
            "dropped": len(dropped),
            "book": book_metrics(build_book(kept, 60.0), oof_years),
            "shadow": slice_stats(kept),
            "dropped_shadow": slice_stats(dropped),
        }
    return {"full_book": full_book, "oof_signals": len(oof),
            "plain_oof_book": plain_oof_book,
            "plain_oof_shadow": plain_oof_shadow,
            "gated": gated}


def standalone_alerts(df: pd.DataFrame, probs: pd.DataFrame,
                      atr: np.ndarray, threshold: float,
                      cooldown: int = ALERT_COOLDOWN,
                      fav_mult: float = FAV_MULT,
                      adv_mult: float = ADV_MULT) -> list[GoldenSignal]:
    """Fabricated signals from ensemble probabilities alone.  The
    barrier geometry defaults to the Phase-6 label geometry (stop
    ``adv_mult`` x ATR, target ``fav_mult`` x ATR); other horizons
    pass their own registered spec (see labels.HORIZON_SPECS)."""
    close = df["close"].to_numpy(dtype=float)
    out: list[GoldenSignal] = []
    last = -10**9
    for t in probs.index:
        if t - last < cooldown:
            continue
        pu = probs.loc[t, "ens_up"]
        pdn = probs.loc[t, "ens_dn"]
        if max(pu, pdn) < threshold:
            continue
        d = 1 if pu >= pdn else -1
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        out.append(GoldenSignal(
            index=int(t), timestamp=df.index[t], direction=d,
            score=float(100.0 * max(pu, pdn)), kind="forecast",
            factors=[f"ensemble P(big move {'up' if d == 1 else 'down'}) "
                     f"= {max(pu, pdn):.2f}"],
            entry_ref=float(close[t]),
            invalidation=float(close[t] - d * adv_mult * a),
            targets=[float(close[t] + d * fav_mult * a)],
            atr=float(a)))
        last = t
    return out


def standalone_test(df: pd.DataFrame, probs: pd.DataFrame,
                    atr: np.ndarray, years: float,
                    thresholds: tuple[float, ...] = GATE_THRESHOLDS,
                    cooldown: int = ALERT_COOLDOWN,
                    fav_mult: float = FAV_MULT,
                    adv_mult: float = ADV_MULT,
                    exits: ExitConfig | None = None
                    ) -> dict:
    """Trade the standalone alerts through the v2 exit machinery.
    ``exits`` defaults to the harness default (72-bar time stop —
    the Phase-6 behaviour); a horizon test passes
    ``ExitConfig(horizon=<that horizon>)`` so the trade's clock
    matches the forecast's clock."""
    res = {}
    for th in thresholds:
        sigs = standalone_alerts(df, probs, atr, th, cooldown=cooldown,
                                 fav_mult=fav_mult, adv_mult=adv_mult)
        outcomes = []
        for sig in sigs:
            o = simulate_signal_v2(df, sig, exits)
            if o is not None:
                outcomes.append(o)
        res[th] = {
            "alerts": len(sigs),
            "simulated": len(outcomes),
            "shadow": slice_stats(outcomes),
            "book": book_metrics(build_book(outcomes, 0.0), years),
        }
    return res


def permutation_control(df: pd.DataFrame, probs: pd.DataFrame,
                        atr: np.ndarray, years: float,
                        threshold: float = 0.5,
                        seeds: tuple[int, ...] = (1, 2, 3),
                        cooldown: int = ALERT_COOLDOWN,
                        fav_mult: float = FAV_MULT,
                        adv_mult: float = ADV_MULT,
                        exits: ExitConfig | None = None) -> dict:
    """The null hypothesis for the standalone test, tested directly:
    permute the (P_up, P_dn) PAIRS across out-of-fold bars — the
    score distribution and the pair's joint structure survive, the
    TIMING information is destroyed — and re-run the standalone book.
    If shuffled alerts make what the real alerts make, the barrier
    geometry (2:1 at a ~35% base rate) plus the v2 exits are doing
    the work, not forecast skill."""
    books, shadows = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = probs.copy()
        perm = rng.permutation(len(shuffled))
        for col in ("ens_up", "ens_dn"):
            shuffled[col] = shuffled[col].to_numpy()[perm]
        st = standalone_test(df, shuffled, atr, years,
                             thresholds=(threshold,), cooldown=cooldown,
                             fav_mult=fav_mult, adv_mult=adv_mult,
                             exits=exits)[threshold]
        books.append(st["book"])
        shadows.append(st["shadow"])
    return {
        "threshold": threshold,
        "seeds": len(seeds),
        "book_return_mean": float(np.mean([b["return_pct"] for b in books])),
        "book_return_min": float(np.min([b["return_pct"] for b in books])),
        "book_return_max": float(np.max([b["return_pct"] for b in books])),
        "book_pf_mean": float(np.mean([b["pf"] for b in books])),
        "shadow_avg_r_mean": float(np.mean([s["avg_r"] for s in shadows])),
    }
