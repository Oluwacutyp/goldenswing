"""Funding-event detectors (spec §3.3(c) / §4.2, mining A12).

The dissolved STATE 3-5 chain's evaluators — displacement,
confirmation, sweep — REUSED as detectors.  Under the rework they
no longer permit or forbid anything: each detected event carries
a colour (the score points the old chain already assigned:
CHoCH +10 / MSB +15 / inversion +5 in live/confirmation.py,
very-strong displacement +10 in live/displacement.py) and the
event's invalidation level.  The runner's funding ladder consumes
events to release layers and roll stops; the score floor, not the
event, authorized the position.

Also here: the absorption detector (A12) — failed pushes into the
zone that close back inside with a rejection wick.  Distinct from
a sweep: it needs no liquidity grab beyond the band, only proof
that sellers/buyers pushed and failed, repeatedly.

And the runaway read: confirmation's ``missed`` decision (price
ran > 1 ATR from the FVG) is exported as a flag — on the arrival
path it arms the PullbackWatch (spec §5) instead of resetting the
setup.  Nothing here deletes or bypasses the chain modules; they
are called, not gated on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .confirmation import evaluate_confirmation
from .displacement import evaluate_displacement
from .sweep import classify_sweep

#: Colour table (spec §4.2).  Structure/displacement colours are
#: read off the verdicts' own score_adj (the chain's numbers);
#: these constants cover the events the chain never scored.
SWEEP_COLOUR = 5.0
WEAK_SWEEP_COLOUR = 2.0
FVG_COLOUR = 3.0
ABSORPTION_COLOUR = 4.0


@dataclass
class FundingEvent:
    kind: str               # structure_shift | displacement |
    # fvg_print | sweep | absorption
    colour: float           # score points this event is worth
    invalidation: float | None = None   # stop-roll target
    detail: str = ""
    key: str = ""           # stable dedupe key (kind:bar identity)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "colour": float(self.colour),
                "invalidation": (None if self.invalidation is None
                                 else float(self.invalidation)),
                "detail": self.detail, "key": self.key}


@dataclass
class FundingReport:
    events: list = field(default_factory=list)   # [FundingEvent]
    runaway: bool = False       # confirmation said "missed"
    confirm_decision: str = ""
    logs: list = field(default_factory=list)

    @property
    def colour(self) -> float:
        return sum(e.colour for e in self.events)

    def to_dict(self) -> dict:
        return {"events": [e.to_dict() for e in self.events],
                "colour_total": float(self.colour),
                "runaway": bool(self.runaway),
                "confirm_decision": self.confirm_decision}


def _absorption(frame, *, direction: int, atr: float,
                zone: dict) -> FundingEvent | None:
    """A12: >= 2 failed pushes into the zone over the recent bars,
    the latest closing back inside with a rejection wick."""
    if frame is None or len(frame) < 3 or atr <= 0:
        return None
    top, bottom = float(zone["top"]), float(zone["bottom"])
    bars = frame.tail(8)
    pushes = 0
    last_reject = False
    for _, row in bars.iterrows():
        h, l, c, o = (float(row["high"]), float(row["low"]),
                      float(row["close"]), float(row["open"]))
        rng = max(h - l, 1e-9)
        if direction > 0:
            into = l < top and c >= bottom       # pushed in, held
            wick = (c - l) / rng                  # lower rejection
        else:
            into = h > bottom and c <= top
            wick = (h - c) / rng                  # upper rejection
        if into:
            pushes += 1
            last_reject = wick >= 0.4
        else:
            last_reject = False
    if pushes >= 2 and last_reject:
        inv = bottom - 0.1 * atr if direction > 0 \
            else top + 0.1 * atr
        return FundingEvent(
            kind="absorption", colour=ABSORPTION_COLOUR,
            invalidation=inv,
            detail=f"{pushes} failed pushes into the zone, "
                   "latest rejected on closes",
            key=f"absorption:{bars.index[-1]}")
    return None


def detect_funding_events(frame, *, direction: int, atr: float,
                          zone: dict | None = None,
                          sweep: dict | None = None,
                          disp_min_atr: float = 1.5,
                          volume_spike_min: float = 1.5
                          ) -> FundingReport:
    """Run every detector over the entry frame once.  Pure: the
    caller owns dedupe (event keys) and the ladder.

    ``frame`` is the style's entry frame (the narrative chain's
    convention: M15 bars, ATR of the HTF clock — the caller passes
    the ATR it already trades on).  ``zone`` is the armed zone dict
    (zones.to_dict shape) when the decision is at a zone.
    """
    report = FundingReport()
    if frame is None or len(frame) < 3 or atr <= 0:
        report.logs.append("funding: no frame/atr — no events")
        return report
    # --- sweep (also feeds displacement's required input)
    sweep_dict = sweep
    if zone is not None:
        try:
            sv = classify_sweep(frame, zone, direction=direction,
                                atr=atr)
        except Exception as exc:
            sv = None
            report.logs.append(f"funding: sweep detector failed: {exc}")
        if sv is not None:
            if sv.kind == "valid_sweep":
                report.events.append(FundingEvent(
                    kind="sweep", colour=SWEEP_COLOUR,
                    invalidation=float(sv.wick),
                    detail=sv.log or "sweep confirmed at the zone",
                    key=f"sweep:{sv.bar_ts}"))
                sweep_dict = {"bar_pos": sv.bar_pos, "wick": sv.wick}
            elif sv.kind == "weak_sweep":
                report.events.append(FundingEvent(
                    kind="sweep", colour=WEAK_SWEEP_COLOUR,
                    invalidation=float(sv.wick),
                    detail=sv.log or "weak sweep at the zone",
                    key=f"sweep:{sv.bar_ts}"))
                sweep_dict = {"bar_pos": sv.bar_pos, "wick": sv.wick}
    # --- displacement (+ its FVG) and confirmation colour
    disp = None
    if sweep_dict is not None:
        try:
            disp = evaluate_displacement(
                frame, sweep=sweep_dict, direction=direction,
                atr=atr, disp_min_atr=disp_min_atr,
                volume_spike_min=volume_spike_min)
        except Exception as exc:
            report.logs.append(
                f"funding: displacement detector failed: {exc}")
    if disp is not None:
        if disp.valid:
            report.events.append(FundingEvent(
                kind="displacement",
                colour=float(disp.score_adj) or 5.0,
                invalidation=(float(disp.extreme)
                              if disp.extreme else None),
                detail=disp.log or f"displacement {disp.strength}",
                key=f"displacement:{disp.leg_end_pos}"))
        if disp.fvg:
            report.events.append(FundingEvent(
                kind="fvg_print", colour=FVG_COLOUR,
                invalidation=(float(disp.fvg["bottom"])
                              if direction > 0 else
                              float(disp.fvg["top"])),
                detail="FVG printed by the displacement leg",
                key=f"fvg:{disp.fvg.get('bar_pos')}"))
        try:
            conf = evaluate_confirmation(
                frame, displacement={"fvg": disp.fvg},
                direction=direction, atr=atr)
        except Exception as exc:
            conf = None
            report.logs.append(
                f"funding: confirmation detector failed: {exc}")
        if conf is not None:
            report.confirm_decision = conf.decision
            if conf.decision == "missed":
                report.runaway = True
            if conf.decision == "confirm" and conf.score_adj:
                kind = conf.kind or "structure"
                report.events.append(FundingEvent(
                    kind="structure_shift",
                    colour=float(conf.score_adj),
                    invalidation=(float(conf.choch_extreme)
                                  if conf.choch_extreme else None),
                    detail=conf.log or f"{kind} confirmed",
                    key=f"structure:{conf.break_pos}"))
    # --- absorption at the zone (A12; independent of any sweep)
    if zone is not None:
        try:
            ab = _absorption(frame, direction=direction, atr=atr,
                             zone=zone)
        except Exception as exc:
            ab = None
            report.logs.append(
                f"funding: absorption detector failed: {exc}")
        if ab is not None:
            report.events.append(ab)
    return report
