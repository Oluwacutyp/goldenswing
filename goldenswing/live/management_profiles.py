"""Per-scenario management profiles + pure management verdicts
(P3, spec v3 S9/S10/F13 + A8 zone TP ladder).

F13 was contested in the mining and resolved by arithmetic:
breakeven at a +1R PRICE TOUCH deletes expectancy (the worked
example: ~77% of +0.35R gone when 30% of winners get BE-tagged
on shallow first pushes).  The desk default is therefore BE
EARNED — by a funding/structure event or by TP1 banking — with
the price touch kept only as a LATE BACKSTOP and as the whole
rule for the late/high-vol profile (whose entries are already
late; for them the touch IS the discipline, per S9's profile
table).  Profiles here are data; the runner enacts them.
"""
from __future__ import annotations

from dataclasses import dataclass


#: MarketSituation Pass 6 (SPEC §7.1): the declared reduce /
#: preserve rules every horizon plan carries.  The reduce rule
#: is the desk's standing S10 behaviour stated as the plan's
#: own term (the profile partial ladder is what a stall trim
#: banks through); the preserve rule is the standing
#: structure-intact discipline (the through-CHoCH test is
#: live/management.py's check_invalidation).
REDUCE_RULE = ("profile partial ladder fires on stall evidence "
               "as well as on R targets")
PRESERVE_RULE = ("runner preserved while structure intact on "
                 "management_tf (no through-CHoCH) and targets "
                 "remain")


@dataclass(frozen=True)
class ManagementProfile:
    name: str
    be_mode: str              # "earned" | "touch"
    be_backstop_r: float      # earned: price-touch backstop (R)
    progress_h1_bars: int     # H1 bars by which progress is due
    progress_floor_r: float   # best-R expected by then
    drift_margin: float       # score pts below entry floor = decay
    trail_method: str         # "structure" | "atr"
    partial_style: str        # "ladder" | "value" | "zone"
    # MarketSituation Pass 6 (SPEC §7.1): the four horizon
    # fields, all expressed in the position's MANAGEMENT clock
    # (the XD-4 conversion table below is the units authority).
    # validity_window_bars: max hold in management-TF bars —
    # the desk's standing staleness rule made a plan term (the
    # runner already exits a position after a SECOND full
    # progress window without progress; the validity window is
    # that horizon: two progress windows).  stall_definition
    # formalises the progress pair in management-clock bars
    # (by_bar trigger bars on the profile's clock; the H1 form
    # stays in progress_h1_bars / progress_floor_r and the two
    # agree by the XD-4 conversion).  exit_clock: the TF whose
    # closes evaluate structural exits for this profile.
    validity_window_bars: int = 0
    stall_definition: dict = None
    reduce_rule: str = REDUCE_RULE
    preserve_rule: str = PRESERVE_RULE
    exit_clock: str = ""

    def journal(self) -> dict:
        return {"profile": self.name, "be": self.be_mode,
                "backstop_r": self.be_backstop_r,
                "progress": [self.progress_h1_bars,
                             self.progress_floor_r],
                "trail": self.trail_method,
                "partial_style": self.partial_style,
                "validity_window_bars": self.validity_window_bars,
                "stall": (dict(self.stall_definition)
                          if self.stall_definition else None),
                "exit_clock": self.exit_clock}


#: Trigger bars per H1 bar, by clock.  Slow clocks are FRACTIONS:
#: one H4 bar spans 4 H1 bars (1/4 H4 bars per H1 bar) and one
#: D1 bar spans 24 (1/24).  The pre-XD-4 table mapped H4 and D1
#: to 1 — as if a daily bar were an hourly bar — so a D1
#: swing's progress window (e.g. 8 trigger bars) converted to
#: 8 H1 bars (8 hours) instead of 8 days.  XD-4 repair: the
#: conversion is wall-time faithful for every clock.
_BARS_PER_H1 = {"M1": 60, "M5": 12, "M15": 4, "H1": 1, "H4": 1 / 4,
                "D1": 1 / 24}

#: Minutes per bar, by clock (XD-4 elapsed conversion).
TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240,
              "D1": 1440}


def mgmt_clock_due(*, clock_tf: str, last_mgmt_bar: int | None,
                   current_mgmt_bar: int) -> bool:
    """XD-4: is a management-clock check due for this position?

    Due iff a new bar on the position's OWN clock has closed
    since its last management pass.  ``last_mgmt_bar is None``
    (pre-XD-4 contexts, adopted positions) is due — fail-open,
    today's behaviour.  The caller passes bar indices on the
    position's clock (see the runner's ``_mgmt_bar_now``); an
    unknown clock is the caller's H1 fallback, never a skip.
    """
    if last_mgmt_bar is None:
        return True
    return int(current_mgmt_bar) > int(last_mgmt_bar)


def elapsed_h1_bars(*, clock_tf: str, entry_mgmt_bar: int,
                    current_mgmt_bar: int) -> int:
    """XD-4: elapsed wall time in H1 bars, from management-clock
    bar indices.  Floor division: a partial H1 bar does not
    count yet (an M5 scalp 6 M5 bars in has 0 H1 bars elapsed;
    a D1 swing 1 D1 bar in has 24).  For H1 positions this
    reproduces the old ``cur_bar - entry_bar`` exactly.
    """
    bars = max(0, int(current_mgmt_bar) - int(entry_mgmt_bar))
    minutes = TF_MINUTES.get(str(clock_tf), 60)
    return (bars * minutes) // 60


def profile_for(*, style: str = "intraday",
                philosophy: str = "trend",
                late_confirm: bool = False,
                high_vol: bool = False,
                clock_tf: str = "H1") -> ManagementProfile:
    """The S9 profile table as data.  ``progress`` expectations
    are stated per style in TRIGGER bars in the mining; they are
    converted to H1 bars here (the runner's management clock)."""
    per_h1 = _BARS_PER_H1.get(clock_tf, 1)

    def h1(trigger_bars: int) -> int:
        return max(1, round(trigger_bars / per_h1))

    def build(name: str, *, be_mode: str, be_backstop_r: float,
              trigger_bars: int, floor_r: float,
              trail_method: str,
              partial_style: str) -> ManagementProfile:
        # Pass 6: the horizon fields derive from the branch's
        # OWN progress pair — the stall definition is that pair
        # stated in management-clock bars, and the validity
        # window is two full progress windows (the runner's
        # standing second-window staleness exit, named as the
        # plan's max hold).  No new numbers.
        return ManagementProfile(
            name, be_mode=be_mode, be_backstop_r=be_backstop_r,
            progress_h1_bars=h1(trigger_bars),
            progress_floor_r=floor_r,
            drift_margin=10.0, trail_method=trail_method,
            partial_style=partial_style,
            validity_window_bars=2 * int(trigger_bars),
            stall_definition={"by_bar": int(trigger_bars),
                              "below_r": float(floor_r)},
            exit_clock=str(clock_tf))

    if late_confirm or high_vol:
        return build("late_high_vol", be_mode="touch",
                     be_backstop_r=0.0, trigger_bars=4,
                     floor_r=0.3, trail_method="atr",
                     partial_style="ladder")
    if style == "scalp":
        return build("scalp", be_mode="earned", be_backstop_r=1.5,
                     trigger_bars=12, floor_r=0.3,
                     trail_method="atr", partial_style="ladder")
    if style == "v_reversal":
        return build("v_reversal", be_mode="earned",
                     be_backstop_r=1.25, trigger_bars=6,
                     floor_r=0.4, trail_method="structure",
                     partial_style="zone")
    if philosophy == "range":
        return build("range", be_mode="earned", be_backstop_r=1.5,
                     trigger_bars=6, floor_r=0.5,
                     trail_method="atr", partial_style="value")
    return build("trend", be_mode="earned", be_backstop_r=2.0,
                 trigger_bars=8, floor_r=0.5,
                 trail_method="structure", partial_style="ladder")


# ------------------------------------------------------------ verdicts

def progress_due(profile: ManagementProfile, *,
                 elapsed_h1_bars: int, best_r: float) -> bool:
    """S10 progress stop: the position has had its style's clock
    and has not made the expected progress — it is tying up the
    one position slot and the risk pool on a thesis the tape is
    not paying for."""
    return elapsed_h1_bars >= profile.progress_h1_bars \
        and best_r < profile.progress_floor_r


def drift_triggered(profile: ManagementProfile, *,
                    entry_floor: float, current_total: float,
                    through_invalidation: bool) -> bool:
    """S10 premise/drift exit: the thesis that was scored at
    entry has decayed below its own floor minus the drift
    margin, while price has NOT taken the invalidation (if it
    had, the stop owns the exit — drift is for the slow death,
    not the fast one)."""
    if through_invalidation:
        return False
    return current_total < entry_floor - profile.drift_margin


def tp_ladder(*, entry: float, stop: float, direction: int,
              zones_ahead: list) -> dict:
    """A8 zone-anchored TP ladder.

    ``zones_ahead`` = zone dicts (proximal / level / top / bottom)
    strictly ahead of entry in the trade direction, nearest
    first.  Desk rule (mining A8): bank AT intermediate magnets
    (just before the proximal edge), and the final target sits
    ~1R BEFORE the opposing zone — never assume the full
    traverse.  With no zones ahead the ladder is the plain
    R-rung ladder.  Returns the adjust_tps-shaped dict the
    runner already consumes (tp prices + banking fractions +
    a human log).
    """
    risk = abs(entry - stop)
    out = {"tp1": None, "tp2": None, "tp3": None,
           "tp1_pct": 0.0, "tp2_pct": 0.0, "tp3_pct": 0.0,
           "exit_all_at_tp2": False, "allow_runner": True,
           "trail_cap_price": None, "trail_lock_r": None,
           "log": []}
    if risk <= 0:
        out["log"].append("tp ladder: no risk distance — R rungs")
        return out
    rungs: list[float] = []
    labels: list[str] = []
    zones = []
    for z in zones_ahead or []:
        prox = z.get("proximal")
        if prox is None:
            prox = z.get("level")
        if prox is None:
            continue
        prox = float(prox)
        if (prox - entry) * direction <= 0:
            continue  # not actually ahead
        zones.append(prox)
    zones.sort(key=lambda p: (p - entry) * direction)
    for i, prox in enumerate(zones[:3]):
        dist_r = (prox - entry) * direction / risk
        if i == len(zones[:3]) - 1 and len(zones) >= 1 \
                and dist_r > 1.5:
            # Final opposing zone: sit ~1R before it.
            price = prox - direction * 1.0 * risk
            labels.append(f"opposing zone {prox:.1f} - 1R early")
        else:
            price = prox - direction * 0.25 * risk
            labels.append(f"magnet {prox:.1f} - 0.25R")
        if (price - entry) * direction >= 0.5 * risk:
            rungs.append(price)
    if not rungs:
        rungs = [entry + direction * risk * m
                 for m in (1.0, 2.0, 3.0)]
        labels = ["1R rung", "2R rung", "3R rung"]
    pcts = {3: (0.5, 0.25, 0.25), 2: (0.5, 0.5, 0.0),
            1: (1.0, 0.0, 0.0)}[len(rungs)]
    for i, price in enumerate(rungs[:3]):
        out[f"tp{i + 1}"] = float(price)
        out[f"tp{i + 1}_pct"] = pcts[i]
        out["log"].append(f"TP{i + 1} {price:.2f} ({labels[i]})")
    if len(rungs) == 1:
        out["exit_all_at_tp2"] = False
        out["allow_runner"] = False
    return out


# --------------------------------------- Pass 6: horizon plans
# MarketSituation Pass 6 (SPEC §7): the declared management
# PLAN.  A scenario declares its plan at proposal time, keyed
# to its horizon class through its resolved policy; the plan
# is part of the scenario version and freezes into the
# position's management context at fill.  Every number in a
# plan comes from an established rule set — the profile above,
# the standing TickConfig tiers, the sleeve's own target
# hierarchy — never from this module inventing one.

#: The thesis-exit arming a non-bracket plan declares: when
#: the scenario's parent S2 thesis is invalidated by the
#: keeper (its named conditions met), the position's thesis
#: is done and the manager exits on that evidence.
THESIS_EXIT_PARENT_INVALIDATION = "parent_thesis_invalidation"


def build_management_plan(*, policy_id: str, policy_version: int,
                          horizon: str, management_tf: str,
                          policy_params: dict,
                          profile: ManagementProfile | None,
                          invalidation_price: float | None,
                          invalidation_terms=(),
                          targets=(), partial_tiers=(),
                          tier_min_volume: float = 0.03,
                          sleeve_targets=()) -> dict:
    """Assemble one scenario's declared management plan.

    ``policy_params`` is the registry's parameter bundle for
    the resolved policy (profile style, profit-lock mode,
    partial source, thesis-exit arming); ``profile`` is the
    policy's canonical ManagementProfile on the management
    clock (None for the fixed bracket, which declares no
    profile — the placement bracket is its whole plan).
    Pure and JSON-safe: the plan rides the scenario, the
    lifecycle store and the journal unchanged."""
    target_list = [float(t) for t in (targets or ()) if t]
    base = {
        "policy_id": str(policy_id),
        "policy_version": int(policy_version),
        "horizon": str(horizon),
        "management_tf": str(management_tf),
        "invalidation": {
            "price": (float(invalidation_price)
                      if invalidation_price is not None
                      else None),
            "terms": [str(t) for t in (invalidation_terms or ())],
            "thesis_exit": None},
    }
    if str(policy_id) == "fixed_bracket_v1" or profile is None:
        # Step 13 / D7: the placement bracket ONLY — no time
        # stop, no trail, no profit-lock, no thesis exit.  The
        # plan states the bracket so the freeze carries the
        # whole truth about how this position is managed.
        base["profile"] = None
        base["time_stop"] = None
        base["partial_schedule"] = {
            "source": "placement_bracket", "tiers": [],
            "tier_min_volume": float(tier_min_volume),
            "style": None, "targets": target_list}
        base["trail"] = {"method": "none"}
        base["profit_lock"] = {"mode": "none"}
        return base
    style = policy_params.get("partial_style_override") \
        or profile.partial_style
    source = str(policy_params.get("partial_source")
                 or "tick_tiers")
    sched_targets = target_list
    if source == "sleeve_target_hierarchy":
        sleeve = [float(t) for t in (sleeve_targets or ()) if t]
        if sleeve:
            sched_targets = sleeve
    base["profile"] = profile.journal()
    base["time_stop"] = {
        "validity_window_bars": int(profile.validity_window_bars),
        "clock": str(management_tf),
        "progress": (dict(profile.stall_definition)
                     if profile.stall_definition else None),
        "basis": "two full progress windows without progress "
                 "= stale (the desk's standing S10 exit, named "
                 "as the plan's max hold)"}
    base["invalidation"]["thesis_exit"] = \
        policy_params.get("thesis_exit")
    base["partial_schedule"] = {
        "source": source,
        "tiers": [[float(r), float(f)]
                  for r, f in (partial_tiers or ())],
        "tier_min_volume": float(tier_min_volume),
        "style": style,
        "targets": sched_targets}
    base["trail"] = {"method": profile.trail_method}
    base["profit_lock"] = {
        "mode": str(policy_params.get("profit_lock_mode")
                    or "standard")}
    return base


def structural_verdicts(profile: ManagementProfile, *,
                        direction: int, elapsed_mgmt_bars: int,
                        best_r: float, tf_state: dict | None = None,
                        targets_open: bool = True) -> dict:
    """The structural management verdicts for one position on
    one management-TF close (SPEC §7.2): stall (the plan's
    reduce evidence), validity expiry (the plan's close
    review), structure break (exit review — a CHoCH/MSCH
    against the position on its OWN management TF, from the
    situation's TFState), and the preserve rule (structure
    intact on the management TF and targets remain).  Pure:
    the runner enacts and journals; nothing here touches a
    broker.  The stall test is the profile's progress rule
    stated in management-clock bars — it agrees with
    progress_due's H1 form by the XD-4 conversion."""
    stall_def = profile.stall_definition or {
        "by_bar": profile.progress_h1_bars,
        "below_r": profile.progress_floor_r}
    stall = int(elapsed_mgmt_bars) >= int(stall_def["by_bar"]) \
        and float(best_r) < float(stall_def["below_r"])
    validity = int(profile.validity_window_bars or 0) > 0 \
        and int(elapsed_mgmt_bars) \
        >= int(profile.validity_window_bars)
    event = (tf_state or {}).get("last_structure_event") or {}
    kind = str(event.get("kind") or "")
    ev_dir = event.get("direction")
    structure_break = kind in ("choch", "msch") \
        and ev_dir is not None \
        and int(ev_dir) == -int(direction)
    preserve = (not structure_break) and bool(targets_open)
    facts: list[str] = []
    if stall:
        facts.append(
            f"stall: {int(elapsed_mgmt_bars)} management bars "
            f">= by_bar {int(stall_def['by_bar'])} with best_r "
            f"{float(best_r):.2f} < {float(stall_def['below_r'])}R "
            "— reduce evidence (plan reduce rule)")
    if validity:
        facts.append(
            f"validity expired: {int(elapsed_mgmt_bars)} "
            f"management bars >= validity window "
            f"{int(profile.validity_window_bars)} — close review")
    if structure_break:
        facts.append(
            f"structure break: {kind} against the position on "
            f"{profile.exit_clock or 'its management TF'} "
            f"(event ts {event.get('ts')}) — exit review")
    if preserve:
        facts.append(
            "preserve: structure intact on the management TF "
            "and targets remain — runner preserved")
    return {"stall": bool(stall),
            "validity_expired": bool(validity),
            "structure_break": bool(structure_break),
            "preserve": bool(preserve),
            "facts": facts}


def management_census(records) -> dict:
    """Per-horizon census over management records (SPEC §10
    Pass 6 evidence): verdict counts and rates, validity
    windows in MINUTES (the minutes-vs-days separation the
    demo proof names), and exit reasons — grouped by the
    horizon stamped on each record.  A horizon policy whose
    census signature (validity window, stall rate, structure
    rate) is identical to standard_v1's is reported in
    ``indistinguishable_from_standard`` — such a policy does
    not promote on this evidence.  Pure over record dicts."""
    by_horizon: dict[str, dict] = {}
    windows: dict[str, list[float]] = {}

    def cell(horizon: str) -> dict:
        return by_horizon.setdefault(horizon, {
            "verdicts": 0, "stall": 0, "validity_expired": 0,
            "structure_break": 0, "thesis_exits": 0,
            "exit_reasons": {}, "validity_window_minutes": None})

    for rec in records or ():
        if not isinstance(rec, dict):
            continue
        rtype = rec.get("type")
        if rtype == "management_verdict":
            horizon = str(rec.get("horizon") or "unstamped")
            c = cell(horizon)
            c["verdicts"] += 1
            for key in ("stall", "validity_expired",
                        "structure_break"):
                if rec.get(key):
                    c[key] += 1
            tf = str(rec.get("management_tf") or "")
            bars = rec.get("validity_window_bars")
            if tf in TF_MINUTES and bars:
                windows.setdefault(horizon, []).append(
                    float(bars) * TF_MINUTES[tf])
        elif rtype == "signal" \
                and rec.get("decision") == "management":
            ctx = rec.get("context") or {}
            horizon = str(ctx.get("management_horizon")
                          or "unstamped")
            c = cell(horizon)
            reason = str(rec.get("reason") or "")
            c["exit_reasons"][reason] = \
                c["exit_reasons"].get(reason, 0) + 1
            if reason == "thesis_exit":
                c["thesis_exits"] += 1
    for horizon, values in windows.items():
        if values:
            ordered = sorted(values)
            by_horizon[horizon]["validity_window_minutes"] = \
                ordered[len(ordered) // 2]

    def signature(horizon: str):
        c = by_horizon.get(horizon)
        if not c or not c["verdicts"]:
            return None
        n = c["verdicts"]
        return (c["validity_window_minutes"],
                round(c["stall"] / n, 4),
                round(c["structure_break"] / n, 4))

    std = signature("intraday")
    indistinguishable: list[str] = []
    if std is not None:
        for horizon in ("scalp", "swing"):
            sig = signature(horizon)
            if sig is not None and sig == std:
                indistinguishable.append(horizon)
    return {"by_horizon": by_horizon,
            "indistinguishable_from_standard": indistinguishable}
