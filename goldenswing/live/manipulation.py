"""Manipulation-leg / liquidity-test filter — entry selection, NOT a veto.

A real trader reads the market, not just the setup.  When price
approaches a prior high/low (liquidity — the Asia range edge, a
confirmed swing level), the FIRST interpretation is a sweep /
manipulation candidate, NOT a breakout to chase:

- Test FAILS (wick beyond the level, close back inside, no
  follow-through) -> the push was manipulation.  The engine does
  NOT enter in the push direction by chasing past the level.  It
  waits for the post-manipulation retest (entry AT the level) or the
  real leg, and reversal setups in the opposite direction stay open.
- Test HOLDS (close beyond the level with displacement, and the
  next bar holds beyond it) -> the break is real.  Breakout entries
  are allowed again.

This is ENTRY SELECTION, not a veto and not an AMD veto: AMD stays a
tag, never a gate.  The filter blocks only the CHASE — an entry
beyond a level whose most recent test failed.  Retest entries (at
the level) and opposite-direction reversals always pass.  Once the
test resolves (real break holds, or the failed test ages out), the
block lifts on its own.

Pure function of closed bars: no persistent state, nothing to
desync.  The caller (the MT5 runner's signal path) passes the closed
H1 frame, the signal's direction / entry level / ATR, and gets
``(blocked, reason)``.
"""
from __future__ import annotations

import numpy as np

from ..reader.session_timeframes import session_of
from .hierarchy import TF_MINUTES  # noqa: F401  (re-exported for callers)
from ..playbooks.common import confirmed_swings

#: How far back a level test still counts (H1 bars).  A failed test
#: older than this is stale — the market has moved on.
LOOKBACK_BARS = 8
#: A sweep must wick beyond the level by this x ATR to count (dust
#: wicks are not manipulation).
SWEEP_MARGIN_ATR = 0.10
#: A confirmed break needs this body (displacement, not a drift).
BREAK_BODY_ATR = 0.5
#: An entry beyond the level by more than this x ATR is a chase.
CHASE_TOLERANCE_ATR = 0.25


def _asia_range(df) -> tuple[float, float] | None:
    """High/low of the most recent Asia session block in ``df``.

    Today's Asia block if present, else yesterday's (the frame may
    start after Asia opened).  None when no Asia bar exists at all.
    """
    idx = df.index
    sessions = [session_of(ts)[0] for ts in idx]
    asia_idx = [i for i, s in enumerate(sessions) if s == "asia"]
    if not asia_idx:
        return None
    # Most recent Asia block = the trailing run of Asia bars ending
    # at the last Asia bar (the block is contiguous by construction).
    last_asia = asia_idx[-1]
    start = last_asia
    while start - 1 >= 0 and sessions[start - 1] == "asia":
        start -= 1
    block = df.iloc[start:last_asia + 1]
    return float(block["high"].max()), float(block["low"].min())


def _key_levels(df) -> list[dict]:
    """Liquidity levels: Asia range edges + most recent swing high/low.

    Each level carries its push direction (beyond the level = the
    manipulation push direction for a high, the mirror for a low).
    """
    levels: list[dict] = []
    rng = _asia_range(df)
    if rng is not None:
        hi, lo = rng
        levels.append({"level": hi, "push": 1, "label": "asia_high"})
        levels.append({"level": lo, "push": -1, "label": "asia_low"})
    n = len(df)
    if n >= 2 * 3 + 5:
        highs, lows = confirmed_swings(
            df["high"].to_numpy(), df["low"].to_numpy(),
            n - 1 - 3, n=3)
        if highs:
            levels.append({"level": highs[-1][1], "push": 1,
                           "label": "swing_high"})
        if lows:
            levels.append({"level": lows[-1][1], "push": -1,
                           "label": "swing_low"})
    return levels


def _latest_test(df, level: float, push: int, atr: float,
                 lookback: int = LOOKBACK_BARS) -> str | None:
    """Most recent test of ``level`` in the last ``lookback`` bars.

    Returns ``"sweep_fail"`` (wick beyond, close back inside — the
    push was manipulation), ``"confirmed_break"`` (close beyond with
    displacement, next bar holds beyond — the break is real), or
    None (no test in the window).
    """
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    n = len(df)
    # One extra bar back: a confirmed break is a (break, hold) PAIR,
    # so a break whose hold bar is the first in-window bar still counts.
    start = max(0, n - lookback - 1)
    latest: tuple[int, str] | None = None
    for i in range(start, n):
        if push == 1:  # testing a high
            if h[i] > level + SWEEP_MARGIN_ATR * atr \
                    and c[i] < level:
                latest = (i, "sweep_fail")
            if i + 1 < n and c[i] > level \
                    and abs(c[i] - o[i]) >= BREAK_BODY_ATR * atr \
                    and c[i + 1] > level:
                latest = (i + 1, "confirmed_break")
        else:  # testing a low
            if l[i] < level - SWEEP_MARGIN_ATR * atr \
                    and c[i] > level:
                latest = (i, "sweep_fail")
            if i + 1 < n and c[i] < level \
                    and abs(c[i] - o[i]) >= BREAK_BODY_ATR * atr \
                    and c[i + 1] < level:
                latest = (i + 1, "confirmed_break")
    return latest[1] if latest is not None else None


def chase_blocked(df, direction: int, entry_ref: float,
                  atr_value: float) -> tuple[bool, str]:
    """Is this signal chasing a manipulation push past a failed test?

    ``df`` is the CLOSED H1 frame.  A signal is a chase when its
    direction matches a level's push direction AND its entry sits
    beyond that level (past the chase tolerance) while the level's
    most recent test FAILED.  Retest entries (at the level) and
    opposite-direction reversals are never blocked.
    """
    if df is None or len(df) < 2:
        return False, ""
    atr = float(atr_value) if atr_value and atr_value > 0 else 0.0
    tol = CHASE_TOLERANCE_ATR * atr
    for lv in _key_levels(df):
        level, push, label = lv["level"], lv["push"], lv["label"]
        if direction != push:
            continue  # opposite direction: reversal, always allowed
        test = _latest_test(df, level, push, atr)
        if test != "sweep_fail":
            continue  # no failed test: nothing to protect against
        if push == 1 and entry_ref > level + tol:
            return True, (
                f"manipulation: {label} {level:.1f} swept and failed, "
                f"entry {entry_ref:.1f} chases past it — waiting for "
                f"the retest / real leg")
        if push == -1 and entry_ref < level - tol:
            return True, (
                f"manipulation: {label} {level:.1f} swept and failed, "
                f"entry {entry_ref:.1f} chases past it — waiting for "
                f"the retest / real leg")
    return False, ""
