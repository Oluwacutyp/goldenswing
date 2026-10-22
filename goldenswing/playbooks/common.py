"""Shared exit discipline + validation summary for the new playbooks.

The new setups (V-reversal, retracement trap, inversion, order block,
killzone open) differ in ENTRY only.  Their exits are held constant,
exactly the exit discipline the FVG-limit A/B validation measured
(``scripts/test_fvg_entry_ab.py``): the entry variable is isolated, the
exit policy is not re-invented per setup.

- Stop: structural invalidation level +/- 0.25 ATR buffer.
- Target: 1.5R fixed.
- Invalidation: a close through the structural level exits
  immediately at that close (before the buffered stop — the
  invalidation level sits inside the stop, as in fvg_limit).
- Max hold: 48 H1 bars; unresolved trades are dropped, never booked.

Promotion doctrine (owner directive 2026-10-22 — this supersedes
the former backtest gate): backtests MAP where a setup historically
worked; they are evidence, never a binary kill/adopt gate.  A
playbook executes on the live DEMO when the owner says it executes,
and forward live-demo performance is the verdict.  The historical
bar below (full-window PF >= 1.10 AND total R > 0 AND >= 30 trades,
measured by ``scripts/validate_new_playbooks.py``) stays recorded
as the map's reference line: a playbook measured under it carries
that reading in its own file, honestly — failures included, never
relabeled.  The reading informs the owner; it does not gate the
``live`` flag.
"""
from __future__ import annotations

import numpy as np

#: Fixed target, in R (fvg_limit's validated exit).
TARGET_R = 1.5
#: Catastrophic backstop buffer beyond the structural level.
STOP_BUFFER_ATR = 0.25
#: Maximum hold after entry (dropped, as in the fvg_limit validation).
MAX_HOLD_BARS = 48
#: 14-bar mean true range.
ATR_N = 14
#: Historical bar: profit factor reference line for validation maps.
MIN_PF = 1.10
#: Historical bar: minimum trade count (enough sample to mean anything).
MIN_TRADES = 30


def atr(high, low, close, i: int, n: int = ATR_N) -> float:
    """14-bar mean true range at closed bar i (needs i >= n)."""
    if i < n:
        return 0.0
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    trs = [max(h[k] - l[k], abs(h[k] - c[k - 1]), abs(l[k] - c[k - 1]))
           for k in range(i - n + 1, i + 1)]
    return float(sum(trs) / n)


def avg_body(close, open_, i: int, n: int = 20) -> float:
    """Mean absolute body over the n bars ending at i (exclusive of i)."""
    if i < 1:
        return 0.0
    c = np.asarray(close, dtype=float)
    o = np.asarray(open_, dtype=float)
    lo = max(0, i - n)
    return float(np.abs(c[lo:i] - o[lo:i]).mean()) if i > lo else 0.0


def simulate_exit(direction: int, entry: float, stop: float,
                  invalidation: float, target: float,
                  o, h, l, c, start_idx: int,
                  max_hold: int = MAX_HOLD_BARS) -> tuple | None:
    """Walk-forward exit machine (fvg_limit semantics, verbatim).

    From the bar AFTER ``start_idx``: invalidation first (a close
    through the structural level exits at that close), then stop,
    then target (stop wins a same-bar stop+target, as in fvg_limit).
    Returns ``(r_multiple, exit_idx, reason)`` or None when the trade
    is unresolved after ``max_hold`` bars (dropped, never booked).
    """
    risk = (entry - stop) * direction
    if risk <= 0:
        return None
    n = len(c)
    for k in range(start_idx + 1, min(start_idx + max_hold + 1, n)):
        if direction == 1 and c[k] < invalidation:
            return (c[k] - entry) / risk, k, "invalidation"
        if direction == -1 and c[k] > invalidation:
            return (entry - c[k]) / risk, k, "invalidation"
        if direction == 1:
            if l[k] <= stop:
                return (stop - entry) / risk, k, "stop"
            if h[k] >= target:
                return TARGET_R, k, "target"
        else:
            if h[k] >= stop:
                return (entry - stop) / risk, k, "stop"
            if l[k] <= target:
                return TARGET_R, k, "target"
    return None


def summarize(r_multiples: list[float]) -> dict:
    """Backtest summary over a list of per-trade R multiples."""
    rs = [float(r) for r in r_multiples]
    n = len(rs)
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    # Max drawdown in R (running-peak-to-trough on the equity curve).
    peak = 0.0
    dd = 0.0
    running = 0.0
    for r in rs:
        running += r
        peak = max(peak, running)
        dd = min(dd, running - peak)
    return {
        "trades": n,
        "wins": len(wins),
        "win_rate": len(wins) / n if n else 0.0,
        "total_r": sum(rs),
        "avg_r": sum(rs) / n if n else 0.0,
        "pf": pf,
        "max_dd_r": dd,
        "best_r": max(rs) if rs else 0.0,
        "worst_r": min(rs) if rs else 0.0,
    }


def confirmed_swings(high, low, up_to: int, n: int = 3,
                     lookback: int = 200
                     ) -> tuple[list[tuple[int, float]],
                                list[tuple[int, float]]]:
    """Confirmed fractal swings with last bar index <= ``up_to``.

    A bar ``i`` is a swing high iff ``high[i]`` is strictly greater
    than every other bar in ``[i-n, i+n]``; swing low is the mirror.
    A swing is CONFIRMED only once ``i + n`` bars have closed, so the
    caller passes ``up_to`` = the last bar that may contain a
    confirmed swing (``t - n`` for a live bar ``t``).  Pure function
    of closed bars — no state, no repainting: a confirmed swing never
    moves.  Returns ``(highs, lows)`` as ``(index, price)`` pairs.

    Only the last ``lookback`` bars are scanned: callers (trap, OB)
    only ever need recent swings, and a full rescan from bar 0 on
    every call would be O(n^2) over a 2-year frame.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    lo = max(n, up_to - lookback)
    hi = min(up_to, len(h) - n - 1)
    for i in range(lo, hi + 1):
        if float(h[i]) > float(np.max(h[i - n:i])) \
                and float(h[i]) > float(np.max(h[i + 1:i + n + 1])):
            highs.append((i, float(h[i])))
        if float(l[i]) < float(np.min(l[i - n:i])) \
                and float(l[i]) < float(np.min(l[i + 1:i + n + 1])):
            lows.append((i, float(l[i])))
    return highs, lows


def promotion_verdict(summary: dict) -> tuple[bool, str]:
    """Historical-bar reading of a validation summary: does the
    measured window clear the reference bar (PF >= 1.10 AND total
    R > 0 AND >= 30 trades)?  A MAP reading, not a promotion gate —
    under the owner directive of 2026-10-22 the verdict on a
    playbook is its forward live-demo performance, decided by the
    owner; this function only reports where the historical
    measurement sits against the bar."""
    if summary["trades"] < MIN_TRADES:
        return False, (f"only {summary['trades']} trades "
                       f"(historical bar >= {MIN_TRADES})")
    if summary["total_r"] <= 0:
        return False, f"total R {summary['total_r']:+.1f} (bar > 0)"
    if summary["pf"] < MIN_PF:
        return False, f"PF {summary['pf']:.2f} (bar >= {MIN_PF})"
    return True, "clears the historical bar"
