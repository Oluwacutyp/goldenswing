"""Bill Williams — the strategy as ONE rule set
(Pass 8, SPEC §6.6), plus its ablation variants and
the outcome simulator that pairs its signals.

The rule (Trading Chaos, as §6.6 states it):

- **Trend state** from the Alligator: a trade is only
  considered in the Alligator's direction, and never
  while it sleeps.
- **Entry** = a confirmed fractal break beyond the
  teeth in that direction, with AO/AC confirmation:
  the first close beyond the fractal level, inside
  the break window, adjudicated at that bar.  A setup
  whose price first closes back beyond the lips
  against the trade dies unbroken — no signal.
- **Invalidation** = opposite fractal / lips cross:
  the stop starts at the latest opposite fractal
  known at entry and trails to each new opposite
  fractal as it confirms (protective direction only);
  a close beyond the lips against the position also
  exits.  A position still open at the hold horizon
  exits at that close.

Ablation variants (the same base events, one organ
removed each — removing a gate can only ADD signals,
which the tests pin as set inclusion):

- ``full``: alligator direction + teeth + AO/AC.
- ``no_aoac``: drops the AO/AC confirmation.
- ``no_teeth``: drops the beyond-the-teeth condition.
- ``no_alligator``: drops the Alligator direction
  gate (breaks taken both ways).

The FULL rule's **abstain arm** is the candidate
population the AO/AC organ alone rejected: base
events passing the Alligator + teeth gates whose
break bar fails AO/AC.  Those are simulated with the
identical machinery — the evaluation's discrimination
question is precisely "does AO/AC earn its place",
and the other organs are graded by the ablation table
(each variant's own paired outcomes on the same
history).  Gator and BW-MFI are not gates: they ride
every signal as measured context, and the evaluation
reports their confirmation lift.

Pairing honesty: a signal with no opposite fractal
known at entry has no definable stop — it is counted
``no_stop`` and never paired (an unmeasurable trade
is not a degenerate one).  A non-positive initial
risk, or an outcome at |R| >= DEGENERATE_R, is
degenerate: counted, excluded from the arms.
"""
from __future__ import annotations

import numpy as np

from .indicators import PreparedWilliams

__all__ = [
    "BREAK_WINDOW_BARS",
    "DEGENERATE_R",
    "MAX_HOLD_BARS",
    "VARIANTS",
    "run_strategy",
    "scan_events",
    "simulate",
]

#: Bars after a fractal confirms in which its break
#: may fire.  Beyond the window the fractal is stale —
#: the market has had its chance to break it.
BREAK_WINDOW_BARS = 10

#: A simulated position still open after this many
#: bars exits at that bar's close (reason "horizon").
MAX_HOLD_BARS = 100

#: |R| at or beyond this is a degenerate outcome
#: (a stop so tight the R is arithmetic, not a trade).
DEGENERATE_R = 20.0

#: Ablation variants: which gates each keeps.
VARIANTS: dict[str, dict] = {
    "full": {"alligator": True, "teeth": True, "aoac": True},
    "no_aoac": {"alligator": True, "teeth": True, "aoac": False},
    "no_teeth": {"alligator": True, "teeth": False, "aoac": True},
    "no_alligator": {"alligator": False, "teeth": True,
                     "aoac": True},
}


def _ts(frame, k) -> str | None:
    try:
        return str(frame.index[k])
    except Exception:
        return None


def scan_events(prep: PreparedWilliams) -> list[dict]:
    """The base events: one per fractal, adjudicated
    causally.  An event records the fractal, its break
    bar (the first close beyond its level inside the
    window) or why there is none (``died`` — a close
    beyond the lips against the trade came first;
    ``expired`` — the window closed unbroken), and the
    gate reads AT the break bar.  Variant filtering is
    a pure function of these records."""
    events: list[dict] = []
    fr = prep.fractals
    for d, src in ((1, fr.up), (-1, fr.down)):
        for f in src:
            c = f["confirmed_at"]
            level = f["price"]
            break_bar = None
            died = False
            j = c
            last = min(c + BREAK_WINDOW_BARS, prep.n - 1)
            while j <= last:
                lines = prep.lines_at(j)
                if lines is not None and \
                        (prep.c[j] - lines["lips"]) * d < 0:
                    died = True
                    break
                if (prep.c[j] - level) * d > 0:
                    break_bar = j
                    break
                j += 1
            ev = {"direction": d, "fractal": dict(f),
                  "break_bar": break_bar,
                  "died": died,
                  "expired": break_bar is None and not died}
            if break_bar is not None:
                lines = prep.lines_at(break_bar)
                alli = prep.alligator_at(break_bar)
                ao = prep.ao_at(break_bar)
                ac = prep.ac_at(break_bar)
                ev["gates"] = {
                    "alligator_direction":
                        alli["direction"] == d,
                    "alligator_state": alli["state"],
                    "teeth": (lines is not None and
                              (level - lines["teeth"]) * d > 0),
                    "aoac": (ao["sign"] == d
                             and ac["sign"] == d),
                }
                ev["read"] = {
                    "alligator_state": alli["state"],
                    "gator_phase":
                        prep.gator_at(break_bar)["phase"],
                    "mfi_class":
                        prep.mfi_at(break_bar)["class"],
                    "ao_sign": ao["sign"],
                    "ac_sign": ac["sign"],
                }
            events.append(ev)
    events.sort(key=lambda e: (e["fractal"]["confirmed_at"],
                               e["direction"]))
    return events


def _passes(ev: dict, gates: dict) -> bool:
    g = ev["gates"]
    return not (
        (gates["alligator"] and not g["alligator_direction"])
        or (gates["teeth"] and not g["teeth"])
        or (gates["aoac"] and not g["aoac"]))


def simulate(prep: PreparedWilliams, entry_bar: int,
             direction: int) -> dict:
    """Pair one entry with its outcome under the
    strategy's invalidation.  Returns the outcome
    block: paired False (with the reason) when no stop
    was definable or the outcome is degenerate."""
    entry = float(prep.c[entry_bar])
    opp = prep.fractals.known_at(entry_bar, -direction)
    if opp is None:
        return {"paired": False, "no_stop": True,
                "degenerate": False, "r": None}
    stop = float(opp["price"])
    stop_initial = stop
    risk = (entry - stop) * direction
    if not np.isfinite(risk) or risk <= 0:
        return {"paired": False, "no_stop": False,
                "degenerate": True, "r": None,
                "stop_initial": stop_initial}
    last = min(entry_bar + MAX_HOLD_BARS, prep.n - 1)
    exit_bar = last
    exit_price = float(prep.c[last])
    reason = "horizon" if last == entry_bar + MAX_HOLD_BARS \
        else "end_of_data"
    j = entry_bar + 1
    while j <= last:
        new_opp = prep.fractals.known_at(j, -direction)
        if new_opp is not None:
            cand = float(new_opp["price"])
            if (cand - stop) * direction > 0:
                stop = cand
        # Stop touch: a long is stopped by the bar's LOW
        # reaching the stop, a short by the bar's HIGH.
        touch = prep.l[j] if direction > 0 else prep.h[j]
        if (touch - stop) * direction <= 0:
            exit_bar, exit_price, reason = j, stop, "stop"
            break
        lines = prep.lines_at(j)
        if lines is not None and \
                (prep.c[j] - lines["lips"]) * direction < 0:
            exit_bar, exit_price, reason = \
                j, float(prep.c[j]), "lips_cross"
            break
        j += 1
    r = direction * (exit_price - entry) / risk
    degenerate = abs(r) >= DEGENERATE_R
    return {"paired": not degenerate,
            "no_stop": False, "degenerate": degenerate,
            "r": (None if degenerate else round(float(r), 4)),
            "raw_r": round(float(r), 4),
            "stop_initial": stop_initial,
            "entry_price": entry,
            "exit_bar": exit_bar, "exit_price": exit_price,
            "exit_reason": reason}


def _signal(prep: PreparedWilliams, ev: dict, *,
            variant: str, arm: str) -> dict:
    j = ev["break_bar"]
    outcome = simulate(prep, j, ev["direction"])
    return {
        "variant": variant,
        "arm": arm,
        "direction": ev["direction"],
        "entry_bar": j,
        "entry_ts": _ts(prep.frame, j),
        "fractal": ev["fractal"],
        "gates": ev["gates"],
        "read": ev["read"],
        "outcome": outcome,
    }


def run_strategy(prep: PreparedWilliams) -> dict:
    """Every variant's signals over the prepared
    frame, plus the FULL rule's abstain arm.

    Returns ``{variant: [signals], ..., "abstains":
    [signals]}`` — abstains are the AO/AC-rejected
    candidates (variant "full", arm "abstain")."""
    events = scan_events(prep)
    out: dict[str, list] = {}
    for name, gates in VARIANTS.items():
        out[name] = [
            _signal(prep, ev, variant=name, arm="take")
            for ev in events
            if ev["break_bar"] is not None and _passes(ev, gates)]
    full = VARIANTS["full"]
    abstain_gates = {**full, "aoac": False}
    out["abstains"] = [
        _signal(prep, ev, variant="full", arm="abstain")
        for ev in events
        if ev["break_bar"] is not None
        and _passes(ev, abstain_gates)
        and not ev["gates"]["aoac"]]
    return out
