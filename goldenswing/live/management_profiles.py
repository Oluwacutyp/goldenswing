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

    def journal(self) -> dict:
        return {"profile": self.name, "be": self.be_mode,
                "backstop_r": self.be_backstop_r,
                "progress": [self.progress_h1_bars,
                             self.progress_floor_r],
                "trail": self.trail_method}


_BARS_PER_H1 = {"M1": 60, "M5": 12, "M15": 4, "H1": 1, "H4": 1,
                "D1": 1}


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

    if late_confirm or high_vol:
        return ManagementProfile(
            "late_high_vol", be_mode="touch", be_backstop_r=0.0,
            progress_h1_bars=h1(4), progress_floor_r=0.3,
            drift_margin=10.0, trail_method="atr",
            partial_style="ladder")
    if style == "scalp":
        return ManagementProfile(
            "scalp", be_mode="earned", be_backstop_r=1.5,
            progress_h1_bars=h1(12), progress_floor_r=0.3,
            drift_margin=10.0, trail_method="atr",
            partial_style="ladder")
    if style == "v_reversal":
        return ManagementProfile(
            "v_reversal", be_mode="earned", be_backstop_r=1.25,
            progress_h1_bars=h1(6), progress_floor_r=0.4,
            drift_margin=10.0, trail_method="structure",
            partial_style="zone")
    if philosophy == "range":
        return ManagementProfile(
            "range", be_mode="earned", be_backstop_r=1.5,
            progress_h1_bars=h1(6), progress_floor_r=0.5,
            drift_margin=10.0, trail_method="atr",
            partial_style="value")
    return ManagementProfile(
        "trend", be_mode="earned", be_backstop_r=2.0,
        progress_h1_bars=h1(8), progress_floor_r=0.5,
        drift_margin=10.0, trail_method="structure",
        partial_style="ladder")


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
