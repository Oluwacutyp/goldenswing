"""Differentiated management: clean vs late confirm get
DIFFERENT rules, not the same rules for less conviction.

Clean first-bar institutional confirm (1.0x size, full
conviction):
- Risk = full risk, SL = sweep extreme - buffer.
- TP1 = +1R, 50% partial, move SL to breakeven on remaining.
- TP2 = +2R, 25% partial, trail SL behind structure (last
  opposing swing / FVG).
- TP3 = +3R, 25% runner, trail behind structure, let run to
  the HTF opposite zone / liquidity pool, time stop 12 H1.
- Management: breakeven at +1R, trailing after +2R, runner
  allowed.  Scaling in on a retest at 0.5x is allowed.

Late second-bar confirm (0.5x size, less conviction, high
volatility):
- Risk = 0.5x normal risk (size halved, SL same distance,
  so $ risk halved, R in $ is half).  Log "late confirm,
  0.5x size, $ risk halved".
- TP1 = +0.75R (earlier, not +1R), 50% partial, move SL to
  breakeven IMMEDIATELY on remaining (not waiting for +1R —
  high vol = protect fast).
- TP2 = +1.5R (not +2R), 30% partial, trail SL behind M5
  structure (tighter, not H1), not H1 structure.
- TP3 = +2R (not +3R), 20% runner OR exit all at +2R if
  high-vol persists (ATR still > 1.5x average).  NO 3R
  runner on late confirm, max +2R, log "late confirm, no
  3R runner, max +2R, high vol".
- Time stop: 6 H1 bars (not 12).  High vol = faster
  invalidation; if not at TP1 in 6 H1 bars, exit, log
  "late confirm, time stop 6 H1, exit".
- Trailing: tighter, trail behind the M5 opposing swing /
  M5 FVG, not H1, step = 0.25 ATR (not 0.5 ATR).
- Invalidation: stricter — if price closes back through
  the CHoCH extreme + mitigates the FVG, exit FULL
  immediately (not partial), log "late confirm,
  invalidation, full exit".
- Scaling IN: NO scaling in / adding on late confirm
  (clean confirm can scale in on retest at 0.5x, late
  confirm never adds, one entry only).
- If TP1 is hit, breakeven, then price returns to
  breakeven, exit with small profit (50% at +0.75R), not
  loss.  Clean confirm runner can give back to breakeven
  on the 25% runner; late confirm cannot afford that,
  tighter.

UPGRADE: if a late-confirm position moves to +1R quickly
(within 2 M15 bars, strong momentum after late confirm),
UPGRADE — log "late confirm, momentum strong, upgrading
to clean management", TP2/TP3 extend to +2R/+3R, runner
allowed, size stays 0.5x (never add), TPs extend.
Upgrade only if +1R in <= 2 M15 bars + volume >= 1.5x,
else stay conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

UPGRADE_MIN_R = 1.0
UPGRADE_MAX_BARS_M15 = 2
UPGRADE_VOLUME_MIN = 1.5


@dataclass
class ManagementPlan:
    """Management targets for one filled setup."""

    confirm_kind: str = "clean"  # clean | late
    size_mult: float = 1.0
    tp1_r: float = 1.0
    tp1_pct: float = 0.50
    tp2_r: float = 2.0
    tp2_pct: float = 0.25
    tp3_r: float = 3.0
    tp3_pct: float = 0.20
    tp1: float | None = None
    tp2: float | None = None
    tp3: float | None = None
    breakeven_at_r: float = 1.0
    trail_tf: str = "H1"
    trail_step_atr: float = 0.5
    time_stop_h1: int = 12
    allow_scale_in: bool = True
    allow_runner: bool = True
    max_r: float = 3.0
    tp3_exit_all_if_high_vol: bool = False
    invalidation: str = "standard"
    upgraded: bool = False
    log: str = ""
    # P3 (S9/F13): HOW breakeven is earned.  "earned" = a funding/
    # structure event or TP1 banking arms it; the price touch is
    # demoted to a backstop at be_backstop_r.  "touch" = the
    # legacy pure price-touch rule — plan_management selects it
    # for the LATE profile only (its entries are already late;
    # for them the touch IS the discipline, per S9).
    be_mode: str = "earned"
    be_backstop_r: float = 2.0


def plan_management(*, direction: int, entry_price: float,
                    stop: float, late_confirm: bool = False,
                    high_vol: bool = False) -> ManagementPlan:
    """Build the management plan for a filled setup.

    ``direction`` +1 / -1, ``entry_price`` / ``stop`` from
    the entry plan.  ``late_confirm`` selects the 0.5x
    conservative ladder; ``high_vol`` marks the TP3
    exit-all condition on a late confirm.
    """
    plan = ManagementPlan()
    r = abs(entry_price - stop)
    if r <= 0 or direction not in (1, -1):
        plan.log = "management plan invalid: bad entry/stop"
        return plan
    if late_confirm:
        plan.confirm_kind = "late"
        plan.size_mult = 0.5
        plan.tp1_r, plan.tp1_pct = 0.75, 0.50
        plan.tp2_r, plan.tp2_pct = 1.5, 0.30
        plan.tp3_r, plan.tp3_pct = 2.0, 0.20
        plan.breakeven_at_r = 0.0  # breakeven IMMEDIATELY
        plan.trail_tf = "M5"
        plan.trail_step_atr = 0.25
        plan.time_stop_h1 = 6
        plan.allow_scale_in = False
        plan.allow_runner = False
        plan.max_r = 2.0
        plan.tp3_exit_all_if_high_vol = bool(high_vol)
        plan.invalidation = "full_exit"
        plan.be_mode = "touch"      # S9: late profile keeps the
        plan.be_backstop_r = 0.0    # touch (breakeven immediate)
        plan.log = ("late confirm, 0.5x size, $ risk halved — "
                    "TP1 +0.75R / TP2 +1.5R / TP3 +2R, "
                    "breakeven immediate, M5 trail, time stop "
                    "6 H1, no 3R runner, no scaling in")
    else:
        plan.confirm_kind = "clean"
        plan.size_mult = 1.0
        plan.tp1_r, plan.tp1_pct = 1.0, 0.50
        plan.tp2_r, plan.tp2_pct = 2.0, 0.25
        plan.tp3_r, plan.tp3_pct = 3.0, 0.25
        plan.breakeven_at_r = 1.0
        plan.trail_tf = "H1"
        plan.trail_step_atr = 0.5
        plan.time_stop_h1 = 12
        plan.allow_scale_in = True
        plan.allow_runner = True
        plan.max_r = 3.0
        plan.tp3_exit_all_if_high_vol = False
        plan.invalidation = "standard"
        plan.log = ("clean confirm, 1.0x size — TP1 +1R / "
                    "TP2 +2R / TP3 +3R, breakeven at +1R, H1 "
                    "trail, time stop 12 H1, runner allowed")
    plan.tp1 = entry_price + direction * r * plan.tp1_r
    plan.tp2 = entry_price + direction * r * plan.tp2_r
    plan.tp3 = entry_price + direction * r * plan.tp3_r
    return plan


def check_upgrade(*, bars_since_entry_m15: int, current_r: float,
                  volume_ratio: float | None) -> bool:
    """Upgrade gate: +1R within <= 2 M15 bars + volume >= 1.5x.
    No volume data -> no upgrade (never assume momentum)."""
    if current_r < UPGRADE_MIN_R:
        return False
    if bars_since_entry_m15 > UPGRADE_MAX_BARS_M15:
        return False
    return volume_ratio is not None and \
        volume_ratio >= UPGRADE_VOLUME_MIN


def upgrade_plan(plan: ManagementPlan, *, entry_price: float,
                 stop: float, direction: int) -> ManagementPlan:
    """Upgrade a late plan to clean TPs: TP2/TP3 extend to
    +2R/+3R, runner allowed.  Size stays 0.5x (never add),
    time stop stays 6 H1, trailing stays M5 — only the TP
    ladder and the runner change."""
    if plan.confirm_kind != "late" or plan.upgraded:
        return plan
    r = abs(entry_price - stop)
    if r <= 0:
        return plan
    plan.tp2_r, plan.tp3_r = 2.0, 3.0
    plan.max_r = 3.0
    plan.allow_runner = True
    plan.upgraded = True
    plan.tp2 = entry_price + direction * r * plan.tp2_r
    plan.tp3 = entry_price + direction * r * plan.tp3_r
    plan.log = ("late confirm, momentum strong, upgrading to "
                "clean management — TP2 +2R / TP3 +3R, runner "
                "allowed, size stays 0.5x")
    return plan


# ------------------------------------------------- STATE 6 RUNTIME
# The plan above is the ladder.  These pure functions are the
# per-tick / per-bar-close decisions the runner applies to a
# filled position: breakeven, partials, trailing, time-stop,
# invalidation, upgrade.  They never move the SL against the
# trade, never widen it, and never recalculate it from a new
# ATR — the SL locked at fill only moves in favour (breakeven
# / trailing) or exits (time-stop / invalidation).

# ATR HIGH threshold (live/atr_regime.py: HIGH is ratio > 1.5).
HIGH_ATR_RATIO = 1.5
# ATR HIGH early breakeven: +0.5R, not waiting for +1R.
EARLY_BREAKEVEN_R = 0.5


@dataclass
class ManagementAction:
    """One evaluation of STATE 6 for a filled position."""

    breakeven: bool = False
    partials: list = field(default_factory=list)  # newly hit TPs
    new_sl: float | None = None  # in-favour SL move, else None
    exit: str | None = None  # None | "time_stop" | "invalidation"
    upgraded: bool = False
    plan: dict = field(default_factory=dict)  # updated plan copy
    remaining_pct: float = 1.0
    log: list = field(default_factory=list)


def breakeven_threshold_r(plan: dict, atr_ratio: float) -> float:
    """Effective breakeven trigger: the plan's own threshold,
    pulled early to +0.5R when ATR is HIGH (ratio > 1.5).
    Late plans already carry 0.0 (breakeven immediately)."""
    base = float(plan.get("breakeven_at_r", 1.0))
    if atr_ratio > HIGH_ATR_RATIO:
        return min(base, EARLY_BREAKEVEN_R)
    return base


def should_breakeven(plan: dict, current_r: float,
                     atr_ratio: float = 1.0) -> bool:
    """Breakeven due when current R reaches the effective
    threshold.  Late confirm: threshold 0.0, due at fill."""
    return current_r >= breakeven_threshold_r(plan, atr_ratio)


def partials_hit(plan: dict, current_r: float) -> list:
    """TPs reached at current R.  TP3 None (opposing GREEN,
    exit all at TP2) is skipped — never a phantom TP."""
    hit = []
    for key in ("tp1", "tp2", "tp3"):
        if plan.get(key) is None:
            continue
        if current_r >= float(plan.get(f"{key}_r", 0.0)):
            hit.append(key)
    return hit


def remaining_after_partials(plan: dict, taken: list) -> float:
    """Position fraction left after the taken TPs (0.0 when
    TP2 was exit-all: its pct already includes the runner)."""
    left = 1.0
    for key in taken:
        left -= float(plan.get(f"{key}_pct", 0.0))
    return max(0.0, left)


def compute_trailing_stop(*, direction: int, current_sl: float,
                          structure_level: float, atr: float,
                          trail_step_atr: float,
                          trail_cap_price: float | None = None) -> float:
    """Trailing SL behind structure, never against the trade.

    Long: candidate = structure low - step*ATR; the new SL is
    max(current, candidate), capped at trail_cap_price
    (opposing zone - 0.25 ATR: trail up to just below the
    barrier, never through it).  Short mirrored.  A cap
    that would move the SL against the trade is ignored."""
    if direction not in (1, -1) or atr <= 0:
        return current_sl
    candidate = structure_level - direction * trail_step_atr * atr
    if direction > 0:
        new_sl = max(current_sl, candidate)
        if trail_cap_price is not None and \
                trail_cap_price >= current_sl:
            new_sl = min(new_sl, trail_cap_price)
    else:
        new_sl = min(current_sl, candidate)
        if trail_cap_price is not None and \
                trail_cap_price <= current_sl:
            new_sl = max(new_sl, trail_cap_price)
    return new_sl


def check_time_stop(elapsed_h1: int, time_stop_h1: int) -> bool:
    """Time-stop due when elapsed H1 bars reach the plan's
    time-stop (tier-scaled, ATR-scaled at fill)."""
    return elapsed_h1 >= int(time_stop_h1)


def check_invalidation(*, through_choch: bool,
                       fvg_mitigated: bool) -> tuple[bool, str]:
    """Invalidation needs BOTH: close back through the CHoCH
    extreme AND the confirm FVG mitigated.  One alone is
    noise, not invalidation.  When it fires, the exit is
    the FULL remaining position (late confirm is stricter
    about the exit, not about the trigger)."""
    if through_choch and fvg_mitigated:
        return True, ("close back through CHoCH extreme + "
                       "confirm FVG mitigated")
    return False, ""


def evaluate_management(*, plan: dict, direction: int,
                        entry_price: float, stop: float,
                        current_price: float, elapsed_h1: int,
                        atr: float, atr_ratio: float = 1.0,
                        choch_extreme: float | None = None,
                        fvg_mitigated: bool = False,
                        bars_since_entry_m15: int | None = None,
                        volume_ratio: float | None = None,
                        structure_level: float | None = None,
                        already_taken: tuple = ()) -> ManagementAction:
    """STATE 6, one evaluation.  Pure: never mutates the
    input plan, never touches the broker.

    Precedence: invalidation > time-stop > breakeven /
    partials / trailing / upgrade.  The returned plan is a
    copy with any upgrade applied; bookkeeping (partials
    taken, breakeven moved) is the caller's job — the action
    reports newly-hit TPs, not cumulative state."""
    action = ManagementAction(plan=dict(plan))
    r = abs(entry_price - stop)
    if r <= 0 or direction not in (1, -1):
        action.log.append("STATE 6: invalid entry/stop — no action")
        return action
    current_r = (current_price - entry_price) * direction / r

    # Invalidation first: close back through CHoCH extreme
    # AND confirm FVG mitigated.
    through = False
    if choch_extreme is not None:
        through = (direction > 0 and current_price < choch_extreme) \
            or (direction < 0 and current_price > choch_extreme)
    invalid, reason = check_invalidation(
        through_choch=through, fvg_mitigated=fvg_mitigated)
    if invalid:
        action.exit = "invalidation"
        action.log.append(f"STATE 6: invalidation — {reason} — "
                          "full exit")
        return action

    # Time-stop: exit when elapsed H1 reaches the plan.
    if check_time_stop(elapsed_h1,
                       int(plan.get("time_stop_h1", 12))):
        action.exit = "time_stop"
        action.log.append(
            f"STATE 6: time stop — {elapsed_h1} H1 elapsed, "
            f"plan {int(plan.get('time_stop_h1', 12))} H1 — exit")
        return action

    # Upgrade (late confirm only): +1R within <= 2 M15 bars
    # + volume >= 1.5x extends TP2/TP3 to +2R/+3R.  Size
    # stays 0.5x (never add).
    if plan.get("confirm_kind") == "late" and not plan.get("upgraded"):
        if bars_since_entry_m15 is not None and check_upgrade(
                bars_since_entry_m15=bars_since_entry_m15,
                current_r=current_r, volume_ratio=volume_ratio):
            action.plan["tp2_r"] = 2.0
            action.plan["tp3_r"] = 3.0
            action.plan["max_r"] = 3.0
            action.plan["allow_runner"] = True
            action.plan["upgraded"] = True
            action.plan["tp2"] = entry_price + direction * r * 2.0
            action.plan["tp3"] = entry_price + direction * r * 3.0
            action.upgraded = True
            action.log.append(
                "STATE 6: late confirm, momentum strong, "
                "upgrading to clean management — TP2 +2R / "
                "TP3 +3R, runner allowed, size stays 0.5x")

    # Breakeven: clean +1R (or +0.5R early when ATR HIGH),
    # late immediate.
    action.breakeven = should_breakeven(
        action.plan, current_r, atr_ratio)
    if action.breakeven:
        action.log.append(
            f"STATE 6: breakeven due at +{current_r:.2f}R")

    # Partials: newly hit TPs only (caller tracks taken).
    hit = partials_hit(action.plan, current_r)
    action.partials = [t for t in hit if t not in already_taken]
    if action.partials:
        action.log.append(
            "STATE 6: partials hit: " + ", ".join(action.partials))
    action.remaining_pct = remaining_after_partials(
        action.plan, hit)

    # Trailing / breakeven SL: candidates in favour only —
    # entry (breakeven) and structure trailing — capped at
    # the opposing barrier.  SL never moves against.
    candidates = [stop]
    if action.breakeven:
        candidates.append(entry_price)
    if structure_level is not None and atr > 0:
        trail = compute_trailing_stop(
            direction=direction, current_sl=stop,
            structure_level=structure_level, atr=atr,
            trail_step_atr=float(
                action.plan.get("trail_step_atr", 0.5)),
            trail_cap_price=action.plan.get("trail_cap_price"))
        candidates.append(trail)
    if direction > 0:
        new_sl = max(candidates)
    else:
        new_sl = min(candidates)
    if new_sl != stop:
        action.new_sl = new_sl
        action.log.append(f"STATE 6: SL -> {new_sl:.2f}")
    return action
