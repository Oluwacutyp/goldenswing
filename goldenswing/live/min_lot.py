"""Min-lot floor: the smallest order is 0.01 lots, and the
SL is invalidation — never shrink the SL to fit a risk
target, never use fake fractional lots below 0.01.

XAUUSD: 0.01 lot = 1 oz, so $1 of gold move = $1 P&L.
Min risk = 0.01 lots x SL distance x $1.

Decision tree when min-lot risk > target risk:
- min_risk_pct = min_risk / equity.
- min_risk_pct > 50% equity AND mode NOT in
  {FAST_GROWTH, FIXED_LOTS}: SKIP — "min lot risk $X >
  50% equity ($Y), skip, would blow account".  No trade,
  no exception.
- min_risk_pct > 10% equity AND min_risk > target AND
  mode == FIXED_LOTS (user explicitly enabled fixed lots,
  $12 fast-growth mode, timing-based, margin-aware, NOT
  1% R grinding): ACCEPT WITH WARNING + guards — max 1
  trade open, no layering, no scaling in, margin shield ON
  (margin < 50% equity, free margin > SL risk).
- min_risk_pct <= 10% equity: ACCEPT at 0.01 lots — target
  risk is a guideline, equity % is the hard guard.
- min_risk <= target_risk: normal sizing, lots =
  target / (SL distance x 100), rounded down to the 0.01
  step, min 0.01.

$12 account special: a $10 target risk = 83% equity is
NOT risk management, it is gambling.  The correct target
for $12 is fixed-lots mode, margin-aware, not % risk.  If
the user sets target $10 on $12 equity, log the WARNING
"target risk $10 = 83% equity, unrealistic, using
fixed-lots 0.01 + margin shield, max 1 trade,
timing-based".  Never allow 2 trades on $12, never
allow layering, never allow risk > equity.

Margin shield: before placing 0.01 lot, free margin must
exceed min_risk (and cover the margin required).  If
free margin < min_risk -> SKIP, "margin shield: free
margin insufficient for min lot risk".
"""

from __future__ import annotations

MIN_LOT = 0.01
XAUUSD_OZ_PER_LOT = 100.0
SKIP_EQUITY_PCT = 0.50
FIXED_LOTS_EQUITY_PCT = 0.10
UNREALISTIC_TARGET_PCT = 0.50
FAST_GROWTH_MODES = {"FAST_GROWTH", "FIXED_LOTS"}


def min_lot_risk(sl_distance: float) -> float:
    """$ risk of the minimum 0.01-lot order at this SL."""
    return MIN_LOT * abs(sl_distance) * XAUUSD_OZ_PER_LOT


def evaluate_min_lot(*, sl_distance: float, target_risk: float,
                     equity: float, mode: str = "PERCENT",
                     free_margin: float | None = None,
                     margin_required: float | None = None,
                     open_trades: int = 0) -> dict:
    """The min-lot floor decision.  Pure — no I/O.

    Returns a dict: decision (accept | skip), lots,
    min_risk, min_risk_pct, target_risk, mode, guards,
    warnings, log.  SL is never adjusted here — the caller
    keeps the invalidation SL exactly as computed.
    """
    min_risk = min_lot_risk(sl_distance)
    pct = (min_risk / equity) if equity > 0 else float("inf")
    target_pct = (target_risk / equity) if equity > 0 else 0.0
    warnings = []
    if target_pct > UNREALISTIC_TARGET_PCT:
        warnings.append(
            f"target risk ${target_risk:.2f} = "
            f"{target_pct * 100:.0f}% equity, unrealistic, "
            f"using fixed-lots 0.01 + margin shield, max 1 "
            f"trade, timing-based")
    if min_risk > target_risk:
        warnings.append("SL is invalidation, never shrink to "
                        "fit size")
    guards = {"max_trades": 1, "no_layering": True,
              "no_scaling_in": True, "margin_shield": True}
    base = {"min_risk": min_risk, "min_risk_pct": pct,
            "target_risk": target_risk, "mode": mode,
            "guards": guards, "warnings": warnings}
    # Margin shield first: cannot even open the min lot.
    if free_margin is not None and free_margin < min_risk:
        return {**base, "decision": "skip", "lots": 0.0,
                "log": ("margin shield: free margin "
                        f"insufficient for min lot risk "
                        f"(${free_margin:.2f} < "
                        f"${min_risk:.2f})")}
    if margin_required is not None and free_margin is not None \
            and free_margin < margin_required:
        return {**base, "decision": "skip", "lots": 0.0,
                "log": ("margin shield: free margin "
                        f"insufficient for margin required "
                        f"(${free_margin:.2f} < "
                        f"${margin_required:.2f})")}
    # One trade at a time on the fixed-lots floor.
    if open_trades >= 1 and mode in FAST_GROWTH_MODES:
        return {**base, "decision": "skip", "lots": 0.0,
                "log": ("min lot floor: max 1 trade open in "
                        "fixed-lots mode, already "
                        f"{open_trades} open")}
    if min_risk <= target_risk:
        return {**base, "decision": "accept", "lots": None,
                "log": (f"min lot risk ${min_risk:.2f} <= "
                        f"target ${target_risk:.2f}, normal "
                        "sizing")}
    if pct > SKIP_EQUITY_PCT and mode not in FAST_GROWTH_MODES:
        return {**base, "decision": "skip", "lots": 0.0,
                "log": (f"min lot risk ${min_risk:.2f} > 50% "
                        f"equity (${equity:.2f}), skip, would "
                        "blow account")}
    if pct > FIXED_LOTS_EQUITY_PCT \
            and mode in FAST_GROWTH_MODES:
        return {**base, "decision": "accept",
                "lots": MIN_LOT,
                "log": (f"min lot floor, risk ${min_risk:.2f} "
                        f"exceeds ${target_risk:.2f} target, "
                        f"fixed-lots mode, accepting with "
                        f"margin shield, max 1 trade")}
    return {**base, "decision": "accept", "lots": MIN_LOT,
            "log": (f"min lot floor, risk ${min_risk:.2f} "
                    f"exceeds ${target_risk:.2f} target but "
                    f"<={FIXED_LOTS_EQUITY_PCT * 100:.0f}% "
                    f"equity, accepting 0.01 lots")}


def describe_min_lot(decision: dict, *, equity: float) -> str:
    """One /status line for the min-lot decision."""
    if not decision:
        return "Risk: min lot n/a"
    if decision.get("decision") == "skip":
        return (f"SKIP: min lot risk "
                f"${decision['min_risk']:.2f} — "
                f"{decision['log']}")
    return (f"Risk: min lot 0.01 risks "
            f"${decision['min_risk']:.2f} "
            f"({decision['min_risk_pct'] * 100:.0f}% equity "
            f"${equity:.2f}), target "
            f"${decision['target_risk']:.2f}, mode "
            f"{decision['mode']}, margin shield ON, max 1 trade")
