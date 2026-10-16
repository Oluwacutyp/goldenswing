"""STATE 7 Exit / Journal — pure functions.

Exit reasons (canonical): TP1 / TP2 / TP3 (a partial target took the
last of the position), SL (structural stop hit), SL_BREAKEVEN (the
stop hit after breakeven moved it to entry — risk-off, not a loss),
INVALIDATION (close back through the CHoCH extreme + confirm FVG
mitigated), TIME_STOP (the plan's H1 horizon elapsed), MANUAL
(owner / Telegram close), PANIC, SCALP_CAP.  Aliases from the
runner / TickManager (``timeout``, ``stop``, ``stop_hit``,
``breakeven``) normalize here — one canonical vocabulary, so the
journal never records the same exit two ways.

Realized R: partials bank R in fractions.  Total realized R is
the volume-weighted sum over every partial plus the final exit
on the remainder::

    total = sum(frac_i * R_i) + remaining_frac * R_final

``finalize_exit`` (in ``live/narrative.py``) is the canonical
runner path: it journals the full narrative chain and resets the
machine to STATE 0 IDLE.  The reset clears the setup (zone, sweep,
displacement, confirm, entry, management, opposing, TP
adjustment) but KEEPS the regime and the zone cache (the cache
lives in the ZoneStore, not in the narrative state — nothing to
clear).  SL discipline from STATE 5/6 carries over: the SL
locked at fill is never recalculated from a later ATR — the
exit R is measured against that locked SL, never a moved one.
"""

from __future__ import annotations

EXIT_REASONS = (
    "tp1", "tp2", "tp3", "sl", "sl_breakeven", "invalidation",
    "time_stop", "manual", "panic", "scalp_cap",
)

_ALIASES = {
    "timeout": "time_stop",
    "time-stop": "time_stop",
    "stop": "sl",
    "stop_hit": "sl",
    "stop-hit": "sl",
    "breakeven": "sl_breakeven",
    "breakeven_sl": "sl_breakeven",
    "tp": "tp3",
    "take_profit": "tp3",
    "invalid": "invalidation",
}


def normalize_exit_reason(reason: str) -> str:
    """Canonical exit reason.  Unknown reasons pass through
    lower-cased (never silently re-labelled as something else)."""
    r = str(reason or "").strip().lower().replace(" ", "_")
    if r in EXIT_REASONS:
        return r
    return _ALIASES.get(r, r)


def partial_r(*, entry_price: float, stop: float, direction: int,
              exit_price: float) -> float:
    """R of one exit price against the locked entry/stop."""
    risk = abs(entry_price - stop)
    if risk <= 0 or direction not in (1, -1):
        return 0.0
    return (exit_price - entry_price) * direction / risk


def realized_r(partials, final_r: float) -> float:
    """Volume-weighted realized R.

    ``partials`` is a list of ``{"fraction": f, "r": r}`` (fraction
    of the original position banked at R ``r``).  ``final_r`` is
    the R of the final exit on the remainder.  Fractions must sum
    to <= 1.0 — over-banking is a bug, never silently clipped.
    """
    total_frac = 0.0
    total = 0.0
    for p in partials or []:
        frac = float(p.get("fraction", 0.0))
        r = float(p.get("r", 0.0))
        if frac < 0.0 or frac > 1.0:
            raise ValueError(f"partial fraction out of range: {frac}")
        total_frac += frac
        total += frac * r
    if total_frac > 1.0 + 1e-9:
        raise ValueError(
            f"partial fractions sum to {total_frac:.3f} > 1.0")
    return total + max(0.0, 1.0 - total_frac) * float(final_r)


def build_journal_entry(state_payload: dict, *, reason: str,
                        realized: float,
                        pnl_dollars: float | None = None,
                        exit_price: float | None = None,
                        partials: list | None = None) -> dict:
    """Journal entry for one exited setup.

    ``state_payload`` is ``NarrativeState.payload()`` (duck-typed:
    anything with a ``payload()`` method or a plain dict works —
    this module must not import ``live/narrative.py``).
    Carries the FULL narrative chain (zone -> sweep ->
    displacement -> confirm -> entry -> management), footprint
    inputs (late_confirm / high_vol / ATR regime), tier +
    opposing + TP adjustment, and the exit.  JSON-safe: the
    payload is already JSON-safe, so the entry is too.
    """
    if hasattr(state_payload, "payload"):
        payload = state_payload.payload()
    else:
        payload = dict(state_payload or {})
    entry = dict(payload.get("entry") or {})
    return {
        "kind": "narrative_exit",
        "exit_reason": normalize_exit_reason(reason),
        "realized_r": float(realized),
        "pnl_dollars": pnl_dollars,
        "exit_price": exit_price,
        "partials": list(partials or []),
        "direction": int(payload.get("direction", 0)),
        "regime": payload.get("regime", "unknown"),
        "entry": entry,
        "zone": payload.get("zone") or {},
        "sweep": payload.get("sweep") or {},
        "displacement": payload.get("displacement") or {},
        "confirm": payload.get("confirm") or {},
        "management": payload.get("management") or {},
        "opposing": payload.get("opposing") or {},
        "entry_tier": payload.get("entry_tier", ""),
        "tp_adjustment": payload.get("tp_adjustment") or {},
        "atr_regime": payload.get("atr_regime", "NORMAL"),
        "atr_ratio": payload.get("atr_ratio", 0.0),
        "late_confirm": bool(payload.get("late_confirm", False)),
        "high_vol": bool(payload.get("high_vol", False)),
        "complex_pullback": bool(
            payload.get("complex_pullback", False)),
        "narrative": list(payload.get("narrative") or []),
    }
