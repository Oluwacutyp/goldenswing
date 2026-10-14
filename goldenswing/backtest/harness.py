"""Backtest harness for the GoldenSwing detector.

Method (deliberately conservative):
- Signal at close of bar t -> enter at the OPEN of bar t+1 (no lookahead).
- Stop = the signal's structural invalidation.
- Target = nearest opposing liquidity pool beyond entry, with a 2R floor
  (if the nearest pool is closer than 2R, the target is 2R).
- If both stop and target are touched in the same bar, the STOP wins.
- Time stop: position closed at the close of bar t+72 (~3 trading days
  of H1 bars) if neither level is hit.
- Cost: $0.45 per ounce round-trip (spread + slippage), expressed in R as
  0.45 / risk-per-ounce.  Position size = 1% of equity at risk.
- Traded book: one position at a time, signals skipped while in a trade.
- Shadow book: EVERY signal simulated standalone (ignoring the position
  lock) purely to grade the score — this is what validates or refutes
  the "backed move" concept: high scores must outperform low scores.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..golden.detector import DetectorConfig, GoldenSignal, GoldenSwingDetector
from ..reader.reader import MarketReader
from .data import DEFAULT_CACHE, REPO_ROOT, load_gold_h1

HORIZON = 72          # bars (~3 days of H1)
COST_PER_OZ = 0.45    # round trip, dollars per ounce
RISK_PCT = 0.01       # 1% of equity per trade


# --------------------------------------------------------------- simulation
@dataclass
class Outcome:
    signal: GoldenSignal
    entry_index: int
    exit_index: int
    exit_reason: str    # v1: "target"|"stop"|"timeout"; v2 adds "trail"
    r_net: float
    hold_bars: int
    partial_taken: bool = False   # v2: partial exit filled at first pool
    wicked_inv: bool = False      # v2: price crossed the *raw* invalidation


def simulate_signal(df: pd.DataFrame, sig: GoldenSignal,
                    horizon: int = HORIZON, cost: float = COST_PER_OZ
                    ) -> Outcome | None:
    t = sig.index
    n = len(df)
    if t + 1 >= n:
        return None
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    d = sig.direction
    entry = open_[t + 1]
    stop = sig.invalidation
    risk = (entry - stop) * d
    if risk <= 0:
        return None
    # target: nearest pool beyond entry, 2R floor
    ahead = [p for p in sig.targets if (p - entry) * d > 0]
    target = entry + d * 2.0 * risk
    if ahead:
        nearest = min(ahead, key=lambda p: abs(p - entry))
        if abs(nearest - entry) >= 2.0 * risk:
            target = nearest
    if (target - entry) * d <= 0 or (open_[t + 1] - target) * d >= 0:
        return None  # already ran to target before entry

    exit_price, exit_idx, reason = close[min(t + horizon, n - 1)], min(t + horizon, n - 1), "timeout"
    last = min(t + horizon, n - 1)
    for k in range(t + 1, last + 1):
        if d == 1:
            if low[k] <= stop:
                exit_price = stop if open_[k] > stop else open_[k]
                exit_idx, reason = k, "stop"
                break
            if high[k] >= target:
                exit_price = target if open_[k] < target else open_[k]
                exit_idx, reason = k, "target"
                break
        else:
            if high[k] >= stop:
                exit_price = stop if open_[k] < stop else open_[k]
                exit_idx, reason = k, "stop"
                break
            if low[k] <= target:
                exit_price = target if open_[k] > target else open_[k]
                exit_idx, reason = k, "target"
                break
    else:
        exit_price, exit_idx, reason = close[last], last, "timeout"

    r_gross = d * (exit_price - entry) / risk
    r_net = r_gross - cost / risk
    return Outcome(signal=sig, entry_index=t + 1, exit_index=exit_idx,
                   exit_reason=reason, r_net=float(r_net),
                   hold_bars=exit_idx - t)


# ------------------------------------------------------- Phase-2 simulation
@dataclass
class ExitConfig:
    """Swing-horizon exits.  Phase 1 stopped trades AT the structural
    invalidation and let them scalp out in ~11 bars; Phase 2 gives the
    3-day swing room to breathe:

    - the stop sits beyond the structural invalidation by an ATR buffer;
    - half the position is banked at the first liquidity pool ahead
      (only when that pool is at least ``partial_min_r`` away);
    - once price has run +``trail_trigger_r``, the remainder trails
      ``trail_dist_r`` behind the best excursion (levels set from
      completed bars only — causal);
    - the 72-bar time stop still closes whatever is left.
    R is measured on the *buffered* stop, so 1% risk sizing automatically
    shrinks the position to pay for the wider stop.
    """

    horizon: int = HORIZON
    cost: float = COST_PER_OZ
    stop_buffer_atr: float = 0.25
    partial_frac: float = 0.5
    partial_min_r: float = 1.0
    trail_trigger_r: float = 1.5
    trail_dist_r: float = 1.0


def simulate_signal_v2(df: pd.DataFrame, sig: GoldenSignal,
                       exits: ExitConfig | None = None) -> Outcome | None:
    exits = exits or ExitConfig()
    t = sig.index
    n = len(df)
    if t + 1 >= n:
        return None
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    d = sig.direction
    entry = open_[t + 1]
    inv = sig.invalidation
    buffer = exits.stop_buffer_atr * sig.atr if sig.atr > 0 else 0.0
    stop0 = inv - d * buffer
    risk = (entry - stop0) * d
    if risk <= 0:
        return None

    # partial target: nearest pool ahead at least partial_min_r away
    partial_price: float | None = None
    ahead = sorted((p for p in sig.targets if (p - entry) * d >= exits.partial_min_r * risk),
                   key=lambda p: abs(p - entry))
    if ahead and exits.partial_frac > 0:
        partial_price = float(ahead[0])

    last = min(t + exits.horizon, n - 1)
    r_banked = 0.0
    rest_frac = 1.0
    partial_done = False
    wicked = False
    best_r = 0.0
    trail_level: float | None = None
    exit_price, exit_idx, reason = float(close[last]), last, "timeout"

    for k in range(t + 1, last + 1):
        # did price cross the raw invalidation this bar? (fact, recorded
        # regardless of what the buffered stop does with it)
        if (d == 1 and low[k] < inv) or (d == -1 and high[k] > inv):
            wicked = True
        # 1. stops first (conservative, as Phase 1): buffered structural
        #    stop, tightened by any trailing level set from bars < k.
        active_stop = stop0
        stop_is_trail = False
        if trail_level is not None:
            if (d == 1 and trail_level > active_stop) or \
                    (d == -1 and trail_level < active_stop):
                active_stop, stop_is_trail = trail_level, True
        stop_hit = low[k] <= active_stop if d == 1 else high[k] >= active_stop
        if stop_hit:
            if d == 1:
                exit_price = active_stop if open_[k] > active_stop else open_[k]
            else:
                exit_price = active_stop if open_[k] < active_stop else open_[k]
            exit_idx = k
            reason = "trail" if stop_is_trail else "stop"
            break
        # 2. partial bank at the first liquidity pool
        if not partial_done and partial_price is not None:
            crossed = high[k] >= partial_price if d == 1 else low[k] <= partial_price
            if crossed:
                fill = partial_price
                if (d == 1 and open_[k] > partial_price) or \
                        (d == -1 and open_[k] < partial_price):
                    fill = open_[k]  # gapped through: take the open
                r_banked += exits.partial_frac * d * (fill - entry) / risk
                rest_frac -= exits.partial_frac
                partial_done = True
        # 3. update best excursion / trailing level for the NEXT bar
        fav = high[k] if d == 1 else low[k]
        best_r = max(best_r, d * (fav - entry) / risk)
        if best_r >= exits.trail_trigger_r:
            lvl = entry + d * (best_r - exits.trail_dist_r) * risk
            if trail_level is None or (d == 1 and lvl > trail_level) or \
                    (d == -1 and lvl < trail_level):
                trail_level = lvl
    else:
        exit_price, exit_idx, reason = float(close[last]), last, "timeout"

    r_net = r_banked + rest_frac * d * (exit_price - entry) / risk \
        - exits.cost / risk
    return Outcome(signal=sig, entry_index=t + 1, exit_index=exit_idx,
                   exit_reason=reason, r_net=float(r_net),
                   hold_bars=exit_idx - t, partial_taken=partial_done,
                   wicked_inv=wicked)


# ----------------------------------------------------------------- metrics
def book_metrics(outcomes: list[Outcome], years: float) -> dict:
    if not outcomes:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "return_pct": 0.0,
                "max_dd_pct": 0.0, "sharpe": 0.0, "avg_r": 0.0, "avg_hold": 0.0}
    rs = np.array([o.r_net for o in outcomes])
    equity = np.cumprod(1.0 + RISK_PCT * rs)
    curve = np.concatenate([[1.0], equity])
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 \
        else float("inf")
    tpy = len(rs) / years if years > 0 else 0.0
    sharpe = float(rs.mean() / rs.std(ddof=1) * np.sqrt(tpy)) \
        if len(rs) > 1 and rs.std(ddof=1) > 0 else 0.0
    return {
        "trades": int(len(rs)),
        "win_rate": float(np.mean(rs > 0) * 100.0),
        "pf": pf,
        "return_pct": float((curve[-1] - 1.0) * 100.0),
        "max_dd_pct": float(-dd.min() * 100.0),
        "sharpe": sharpe,
        "avg_r": float(rs.mean()),
        "avg_hold": float(np.mean([o.hold_bars for o in outcomes])),
    }


def build_book(outcomes: list[Outcome], threshold: float) -> list[Outcome]:
    """Greedy sequential book: one position at a time."""
    book: list[Outcome] = []
    free_from = -1
    for o in sorted(outcomes, key=lambda x: x.signal.index):
        if o.signal.score < threshold:
            continue
        if o.signal.index < free_from:
            continue
        book.append(o)
        free_from = o.exit_index
    return book


@dataclass
class BacktestResult:
    bars: int
    start: str
    end: str
    years: float
    n_signals: int
    books: dict[float, dict] = field(default_factory=dict)
    bands: list[dict] = field(default_factory=list)
    primary_threshold: float = 60.0


BANDS = [(35.0, 50.0), (50.0, 60.0), (60.0, 75.0), (75.0, 101.0)]


def run_backtest(df: pd.DataFrame, primary_threshold: float = 60.0,
                 thresholds: tuple[float, ...] = (50.0, 60.0, 70.0)
                 ) -> BacktestResult:
    reader = MarketReader()
    states = reader.read(df)
    detector = GoldenSwingDetector(
        DetectorConfig(emit_floor=35.0, golden_threshold=primary_threshold))
    signals = detector.scan(states)

    outcomes: list[Outcome] = []
    for sig in signals:
        o = simulate_signal(df, sig)
        if o is not None:
            outcomes.append(o)

    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    years = span_days / 365.25

    books = {}
    for th in thresholds:
        books[th] = book_metrics(build_book(outcomes, th), years)
    if primary_threshold not in books:
        books[primary_threshold] = book_metrics(
            build_book(outcomes, primary_threshold), years)

    bands: list[dict] = []
    for lo_b, hi_b in BANDS:
        sub = [o for o in outcomes if lo_b <= o.signal.score < hi_b]
        if sub:
            rs = np.array([o.r_net for o in sub])
            wins = rs[rs > 0]
            losses = rs[rs <= 0]
            pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf")
            bands.append({"lo": lo_b, "hi": hi_b, "n": len(sub),
                          "win_rate": float(np.mean(rs > 0) * 100.0),
                          "avg_r": float(rs.mean()), "pf": pf})
        else:
            bands.append({"lo": lo_b, "hi": hi_b, "n": 0,
                          "win_rate": 0.0, "avg_r": 0.0, "pf": 0.0})

    return BacktestResult(
        bars=len(df), start=str(df.index[0]), end=str(df.index[-1]),
        years=years, n_signals=len(outcomes), books=books, bands=bands,
        primary_threshold=primary_threshold,
    )


# ------------------------------------------------------------------ report
def build_report(r: BacktestResult) -> str:
    p = r.books[r.primary_threshold]
    lines = [
        "# GoldenSwing — Backtest Report (Phase 1)",
        "",
        f"**Data:** GC=F hourly (yfinance), {r.bars:,} bars, {r.start} → {r.end} "
        f"({r.years:.2f} years). GC=F is a futures proxy for spot XAUUSD.",
        "",
        "**Method:** signal at close of bar t, entry at open of t+1, stop = "
        "structural invalidation, target = nearest opposing liquidity pool "
        "(2R floor), stop wins same-bar ties, 72-bar time stop, $0.45/oz "
        "round-trip cost, 1% risk, one position at a time (traded book).",
        "",
        f"**Signals simulated (shadow, score ≥ 35):** {r.n_signals}",
        "",
        f"## Traded book @ score ≥ {r.primary_threshold:.0f}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Trades | {p['trades']} |",
        f"| Win rate | {p['win_rate']:.1f}% |",
        f"| Profit factor | {p['pf']:.2f} |",
        f"| Total return | {p['return_pct']:+.2f}% |",
        f"| Max drawdown | {p['max_dd_pct']:.2f}% |",
        f"| Sharpe (per-trade, annualised) | {p['sharpe']:.2f} |",
        f"| Avg R | {p['avg_r']:+.3f} |",
        f"| Avg hold | {p['avg_hold']:.0f} bars |",
        "",
        "## Threshold comparison (traded book)",
        "",
        "| Threshold | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for th in sorted(r.books):
        b = r.books[th]
        lines.append(
            f"| ≥{th:.0f} | {b['trades']} | {b['win_rate']:.1f} | {b['pf']:.2f} | "
            f"{b['return_pct']:+.2f}% | {b['max_dd_pct']:.2f}% | {b['sharpe']:.2f} | "
            f"{b['avg_r']:+.3f} |")
    lines += [
        "",
        "## Score discrimination (shadow book — every signal, standalone)",
        "",
        "This is the test of the core concept: if 'backing' means anything, "
        "higher scores must produce better outcomes than lower scores.",
        "",
        "| Score band | Signals | Win% | Avg R | PF |",
        "|---|---|---|---|---|",
    ]
    for b in r.bands:
        lines.append(
            f"| {b['lo']:.0f}–{b['hi']:.0f} | {b['n']} | {b['win_rate']:.1f} | "
            f"{b['avg_r']:+.3f} | {b['pf']:.2f} |")
    lines += ["", "## Caveats", "",
              "- One instrument (gold), one timeframe (H1), one 2-year window "
              "that was strongly bullish for gold — long-heavy systems get a tailwind.",
              "- GC=F futures ≠ spot XAUUSD: spreads, roll and session gaps differ.",
              "- No slippage modelling beyond the flat $0.45/oz cost; news spikes "
              "would be worse in live trading.",
              "- Score weights are hand-set priors, NOT fitted to this data — "
              "which keeps the test honest but leaves performance on the table.",
              "- The detector fires on H1 bars; the 3-day horizon is the "
              "72-bar time stop, most trades exit earlier on target/stop.",
              ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GoldenSwing backtest")
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--cache", type=str, default=str(DEFAULT_CACHE))
    args = ap.parse_args(argv)

    df = load_gold_h1(Path(args.cache), refresh=args.refresh_data)
    result = run_backtest(df, primary_threshold=args.threshold)
    report = build_report(result)
    out = REPO_ROOT / "BACKTEST.md"
    out.write_text(report)
    print(report)
    print(f"\n[written to {out}]")


# ------------------------------------------------------------- Phase 2 run
def slice_stats(outcomes: list[Outcome]) -> dict:
    if not outcomes:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "pf": 0.0}
    rs = np.array([o.r_net for o in outcomes])
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 \
        else float("inf")
    return {"n": int(len(rs)), "win_rate": float(np.mean(rs > 0) * 100.0),
            "avg_r": float(rs.mean()), "pf": pf}


@dataclass
class BacktestResult2:
    bars: int
    start: str
    end: str
    years: float
    confirm_window: int
    n_signals: int
    books: dict[float, dict] = field(default_factory=dict)
    bands: list[dict] = field(default_factory=list)
    shadow_slices: dict[str, dict] = field(default_factory=dict)
    book_slices: dict[str, dict] = field(default_factory=dict)
    exit_mix: dict[str, float] = field(default_factory=dict)
    partial_rate: float = 0.0
    wicked_survivors: int = 0
    long_cont: dict = field(default_factory=dict)
    primary_threshold: float = 60.0


def run_backtest2(df: pd.DataFrame, primary_threshold: float = 60.0,
                  thresholds: tuple[float, ...] = (50.0, 60.0, 70.0),
                  confirm_window: int = 8,
                  exits: ExitConfig | None = None) -> BacktestResult2:
    reader = MarketReader()
    states = reader.read(df)
    detector = GoldenSwingDetector(DetectorConfig(
        emit_floor=35.0, golden_threshold=primary_threshold,
        require_sweep_confirmation=True, confirm_window=confirm_window,
        daily_veto=True))
    signals = detector.scan(states)

    outcomes: list[Outcome] = []
    for sig in signals:
        o = simulate_signal_v2(df, sig, exits)
        if o is not None:
            outcomes.append(o)

    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    years = span_days / 365.25

    books = {}
    for th in thresholds:
        books[th] = book_metrics(build_book(outcomes, th), years)
    if primary_threshold not in books:
        books[primary_threshold] = book_metrics(
            build_book(outcomes, primary_threshold), years)

    bands: list[dict] = []
    for lo_b, hi_b in BANDS:
        sub = [o for o in outcomes if lo_b <= o.signal.score < hi_b]
        st = slice_stats(sub)
        bands.append({"lo": lo_b, "hi": hi_b, **st})

    def sl(pred) -> dict:
        return slice_stats([o for o in outcomes if pred(o)])

    shadow_slices = {
        "all": sl(lambda o: True),
        "sweep_reversal": sl(lambda o: o.signal.kind == "sweep_reversal"),
        "continuation": sl(lambda o: o.signal.kind == "continuation"),
        "long": sl(lambda o: o.signal.direction == 1),
        "short": sl(lambda o: o.signal.direction == -1),
        "long_sweep": sl(lambda o: o.signal.kind == "sweep_reversal" and o.signal.direction == 1),
        "short_sweep": sl(lambda o: o.signal.kind == "sweep_reversal" and o.signal.direction == -1),
        "long_cont": sl(lambda o: o.signal.kind == "continuation" and o.signal.direction == 1),
        "short_cont": sl(lambda o: o.signal.kind == "continuation" and o.signal.direction == -1),
        "sweep_gte60": sl(lambda o: o.signal.kind == "sweep_reversal" and o.signal.score >= 60.0),
        "cont_gte60": sl(lambda o: o.signal.kind == "continuation" and o.signal.score >= 60.0),
    }

    book = build_book(outcomes, primary_threshold)
    book_slices = {
        "sweep_reversal": slice_stats([o for o in book if o.signal.kind == "sweep_reversal"]),
        "continuation": slice_stats([o for o in book if o.signal.kind == "continuation"]),
        "long": slice_stats([o for o in book if o.signal.direction == 1]),
        "short": slice_stats([o for o in book if o.signal.direction == -1]),
    }
    if book:
        reasons = [o.exit_reason for o in book]
        exit_mix = {r: 100.0 * reasons.count(r) / len(reasons)
                    for r in ("stop", "trail", "timeout", "target")}
        partial_rate = 100.0 * float(np.mean([o.partial_taken for o in book]))
        wicked_survivors = int(sum(
            1 for o in book if o.wicked_inv and o.exit_reason != "stop"))
    else:
        exit_mix, partial_rate, wicked_survivors = {}, 0.0, 0

    # ---- long-continuation standalone: real sleeve or bull artifact? ----
    lc = [o for o in outcomes
          if o.signal.kind == "continuation" and o.signal.direction == 1]
    mid = len(df) // 2
    lc_h1 = [o for o in lc if o.signal.index < mid]
    lc_h2 = [o for o in lc if o.signal.index >= mid]
    long_cont = {
        "shadow_all": slice_stats(lc),
        "shadow_h1": slice_stats(lc_h1),
        "shadow_h2": slice_stats(lc_h2),
        "book60_all": book_metrics(build_book(lc, 60.0), years),
        "book60_h1": book_metrics(build_book(lc_h1, 60.0), years / 2.0),
        "book60_h2": book_metrics(build_book(lc_h2, 60.0), years / 2.0),
        "book35_all": book_metrics(build_book(lc, 35.0), years),
    }

    return BacktestResult2(
        bars=len(df), start=str(df.index[0]), end=str(df.index[-1]),
        years=years, confirm_window=confirm_window,
        n_signals=len(outcomes), books=books, bands=bands,
        shadow_slices=shadow_slices, book_slices=book_slices,
        exit_mix=exit_mix, partial_rate=partial_rate,
        wicked_survivors=wicked_survivors, long_cont=long_cont,
        primary_threshold=primary_threshold,
    )


# Phase 1 reference numbers (from BACKTEST.md, same data/costs) for the
# side-by-side comparison in the Phase 2 report.
P1_BOOK60 = {"trades": 584, "win_rate": 25.2, "pf": 0.87, "return_pct": -49.82,
             "max_dd_pct": 61.09, "sharpe": -0.91, "avg_r": -0.101, "avg_hold": 11.0}
# Phase 1 diagnostics reported n / win% / avg R only (no PF) — kept exact.
P1_SHADOW = {
    "all": {"n": 1836, "win_rate": 26.5, "avg_r": -0.086},
    "sweep_reversal": {"n": 1551, "win_rate": 25.0, "avg_r": -0.113},
    "continuation": {"n": 285, "win_rate": 34.7, "avg_r": 0.062},
    "long": {"n": 949, "win_rate": 28.9, "avg_r": -0.026},
    "short": {"n": 887, "win_rate": 23.9, "avg_r": -0.151},
    "long_cont": {"n": 167, "win_rate": 38.3, "avg_r": 0.185},
}


def _fmt_slice(name: str, st: dict, p1: dict | None = None) -> str:
    base = (f"| {name} | {st['n']} | {st['win_rate']:.1f} | "
            f"{st['avg_r']:+.3f} | {st['pf']:.2f} |")
    if p1 is not None:
        base += f" {p1['n']} | {p1['win_rate']:.1f} | {p1['avg_r']:+.3f} | — |"
    return base


def build_report2(r: BacktestResult2) -> str:
    p = r.books[r.primary_threshold]
    q = P1_BOOK60
    lines = [
        "# GoldenSwing — Backtest Report (Phase 2)",
        "",
        f"**Data:** GC=F hourly (yfinance), {r.bars:,} bars, {r.start} → {r.end} "
        f"({r.years:.2f} years). Same data, same $0.45/oz cost, same 72-bar "
        "horizon, same 1% risk as Phase 1.",
        "",
        "**What changed (the Phase-1 diagnoses, applied):**",
        "1. Sweep reversals now require CONFIRMATION — displacement (body "
        ">= 1.0x ATR or a displacement FVG) AND a structure shift (CHoCH/BOS "
        "or trend flip) in the reversal direction, both at/after the sweep "
        f"bar, within {r.confirm_window} bars — otherwise the setup expires.",
        "2. Daily-bias veto: trades against the D1 trend state (daily "
        "fractal structure, EMA50 tilt while ranging) are blocked, except "
        "a confirmed sweep reversal at a major daily level.",
        "3. Swing exits: stop buffered 0.25x ATR beyond the structural "
        "invalidation, 50% banked at the first liquidity pool (>= 1R away), "
        "remainder trails 1.0R behind the best excursion once +1.5R is "
        "reached, 72-bar time stop kept. R is measured on the buffered "
        "stop.",
        "",
        f"**Signals simulated (shadow, score ≥ 35):** {r.n_signals} "
        f"(Phase 1: 1836)",
        "",
        f"## Traded book @ score ≥ {r.primary_threshold:.0f} — Phase 2 vs Phase 1",
        "",
        "| Metric | Phase 2 | Phase 1 |",
        "|---|---|---|",
        f"| Trades | {p['trades']} | {q['trades']} |",
        f"| Win rate | {p['win_rate']:.1f}% | {q['win_rate']:.1f}% |",
        f"| Profit factor | {p['pf']:.2f} | {q['pf']:.2f} |",
        f"| Total return | {p['return_pct']:+.2f}% | {q['return_pct']:+.2f}% |",
        f"| Max drawdown | {p['max_dd_pct']:.2f}% | {q['max_dd_pct']:.2f}% |",
        f"| Sharpe | {p['sharpe']:.2f} | {q['sharpe']:.2f} |",
        f"| Avg R | {p['avg_r']:+.3f} | {q['avg_r']:+.3f} |",
        f"| Avg hold (bars) | {p['avg_hold']:.0f} | {q['avg_hold']:.0f} |",
        "",
        "## Threshold comparison (traded book, Phase 2)",
        "",
        "| Threshold | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for th in sorted(r.books):
        b = r.books[th]
        lines.append(
            f"| ≥{th:.0f} | {b['trades']} | {b['win_rate']:.1f} | {b['pf']:.2f} | "
            f"{b['return_pct']:+.2f}% | {b['max_dd_pct']:.2f}% | {b['sharpe']:.2f} | "
            f"{b['avg_r']:+.3f} |")
    lines += [
        "",
        "## Per-slice results (shadow book — every signal standalone)",
        "",
        "| Slice | P2 n | P2 Win% | P2 Avg R | P2 PF | P1 n | P1 Win% | P1 Avg R | P1 PF |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, label in (("all", "All signals"), ("sweep_reversal", "Sweep reversals"),
                       ("continuation", "Continuations"), ("long", "Longs"),
                       ("short", "Shorts"), ("long_cont", "Long continuation")):
        lines.append(_fmt_slice(label, r.shadow_slices[key],
                                P1_SHADOW.get(key)))
    lines += [
        "",
        "Phase-2-only shadow slices:",
        "",
        "| Slice | n | Win% | Avg R | PF |",
        "|---|---|---|---|---|",
    ]
    for key, label in (("long_sweep", "Long sweep reversals"),
                       ("short_sweep", "Short sweep reversals"),
                       ("short_cont", "Short continuations"),
                       ("sweep_gte60", "Sweep reversals @ ≥60"),
                       ("cont_gte60", "Continuations @ ≥60")):
        lines.append(_fmt_slice(label, r.shadow_slices[key]))
    lines += [
        "",
        f"## Traded-book slices @ ≥{r.primary_threshold:.0f} (Phase 2)",
        "",
        "| Slice | n | Win% | Avg R | PF |",
        "|---|---|---|---|---|",
    ]
    for key, label in (("sweep_reversal", "Sweep reversals"),
                       ("continuation", "Continuations"),
                       ("long", "Longs"), ("short", "Shorts")):
        lines.append(_fmt_slice(label, r.book_slices[key]))
    lines += [
        "",
        "## Exit behaviour (traded book, Phase 2)",
        "",
        f"- Exit mix: " + ", ".join(
            f"{k} {v:.1f}%" for k, v in sorted(r.exit_mix.items()) if v > 0)
        + " (Phase 1: stop 73%, target 25%, timeout 2%)",
        f"- Partial banked at first pool on {r.partial_rate:.1f}% of trades",
        f"- Trades that wicked through the raw invalidation but were NOT "
        f"stopped out (the ATR buffer doing its job): {r.wicked_survivors}",
        "",
        "## Score discrimination (shadow book — the core concept test)",
        "",
        "| Score band | Signals | Win% | Avg R | PF |",
        "|---|---|---|---|---|",
    ]
    for b in r.bands:
        lines.append(
            f"| {b['lo']:.0f}–{b['hi']:.0f} | {b['n']} | {b['win_rate']:.1f} | "
            f"{b['avg_r']:+.3f} | {b['pf']:.2f} |")
    lines += [
        "",
        "Phase 1 bands for reference: 35–50 −0.260R · 50–60 −0.084R · "
        "60–75 −0.042R · 75+ −0.120R.",
        "",
        "## Long-continuation standalone — real sleeve or bull artifact?",
        "",
        "The one Phase-1 slice with a pulse (+0.185R), run as its OWN book "
        "(one position at a time among long continuations only), with the "
        "Phase-2 exits. Both window halves reported separately — a sleeve "
        "that only works in one half is a regime artifact until proven "
        "otherwise.",
        "",
        "| Book | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lc = r.long_cont
    for key, label in (("book60_all", "≥60 full window"), ("book60_h1", "≥60 first half"),
                       ("book60_h2", "≥60 second half"), ("book35_all", "≥35 full window")):
        b = lc[key]
        lines.append(
            f"| {label} | {b['trades']} | {b['win_rate']:.1f} | {b['pf']:.2f} | "
            f"{b['return_pct']:+.2f}% | {b['max_dd_pct']:.2f}% | {b['sharpe']:.2f} | "
            f"{b['avg_r']:+.3f} |")
    lines += [
        "",
        "Shadow (all long-continuation signals, standalone): "
        f"full {lc['shadow_all']['n']} signals, avg R {lc['shadow_all']['avg_r']:+.3f}; "
        f"first half {lc['shadow_h1']['n']}, avg R {lc['shadow_h1']['avg_r']:+.3f}; "
        f"second half {lc['shadow_h2']['n']}, avg R {lc['shadow_h2']['avg_r']:+.3f}.",
        "",
        "## Caveats",
        "",
        "- Same single window as Phase 1: one instrument, one strongly "
        "bullish 2-year stretch for gold. Nothing here is validated out of "
        "sample.",
        "- Daily bars are UTC-calendar aggregates of hourly GC=F bars — "
        "an approximation of the futures daily candle.",
        "- Exit redesign changes the meaning of R vs Phase 1 (R is now "
        "measured on a wider, buffered stop), so avg-R comparisons across "
        "phases are indicative, not exact.",
        "- Score weights remain hand-set priors, not fitted.",
        "",
    ]
    return "\n".join(lines)


def main2(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GoldenSwing Phase-2 backtest")
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--confirm-window", type=int, default=8)
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--cache", type=str, default=str(DEFAULT_CACHE))
    args = ap.parse_args(argv)

    df = load_gold_h1(Path(args.cache), refresh=args.refresh_data)
    result = run_backtest2(df, primary_threshold=args.threshold,
                           confirm_window=args.confirm_window)
    report = build_report2(result)
    out = REPO_ROOT / "BACKTEST2.md"
    out.write_text(report)
    print(report)
    print(f"\n[written to {out}]")


if __name__ == "__main__":
    main()
