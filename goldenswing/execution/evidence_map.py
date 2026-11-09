"""Per-playbook evidence map (owner, 2026-10-26).

Truthful answer to: which recorded features actually affect a
decision, and which are recorded evidence only?  Read from the
code on 2026-10-26 (playbook live_signal bodies, the runner's
trigger path `_run_one_playbook`, and `_census_for_record`) —
not from feature lists or marketing names.

Two rules the whole map rests on:

1. CENSUS IS NEVER A DECISION INPUT.  The wired census
   families (chart patterns, premium/discount, breaker /
   mitigation blocks, relative strength, engulfing, named
   candle taxonomy) are recorded by `_census_for_record` on
   live evaluations only, rendered as "recorded, no decision
   weight", and read by no decision code.  A census family
   appearing on an assessment is groundwork, not integration.
   Legacy SMC is partially integrated in exactly the sense
   the marker says: breaker + mitigation blocks are recorded
   from live/patterns.py helpers, while the structure /
   order-block / FVG helpers there remain unconsumed legacy
   duplicates — live SMC evidence comes from the playbooks
   and live/zones.py.
2. A DETECTOR'S OWN INPUTS ARE ITS DECISION INPUTS.  Each
   roster detector reads only its own timeframe's closed OHLC
   (plus its own tracker state and, where noted, session
   windows or the causally sliced H4 frame).  No roster
   detector reads census families, DXY/silver, or another
   playbook's output.  What the shared trigger path then adds
   (asymmetry score, hard lines, floors, journey/zone context,
   D1 bias as weighed evidence — never a veto) is listed
   separately under SHARED_DECISION_INPUTS so it is never
   confused with a playbook's own detector.

Coverage is test-pinned: the keys here must equal the
playbook registry (test_assessment_journal_audit).  A new
playbook that is not added to this map fails the suite —
that is the anti-drift mechanism.  Rows flagged UNCERTAIN
state the open question instead of resolving it by guess.
"""
from __future__ import annotations

#: Census families: recorded evidence only, on every playbook.
CENSUS_EVIDENCE_ONLY = (
    "chart_patterns",
    "premium_discount",
    "breaker_blocks",
    "mitigation_blocks",
    "relative_strength",
    "engulfing",
    "named_candlestick_taxonomy",
)

#: What the shared h1_trigger path adds beyond the detector
#: (runner._run_one_playbook).  Decision-bearing, but shared —
#: never attributed to one playbook's detector.
SHARED_DECISION_INPUTS = (
    "asymmetry score (live/asymmetry.py) — the decision number",
    "hard lines only: declaration gates + manipulation filter",
    "per-style score floor (style/family, not a flat min_score)",
    "journey/zone context (zone behind the setup, magnet, "
    "journey geometry)",
    "structural D1 bias — weighed as evidence, NEVER a veto",
    "order-path invariants OUTSIDE any model: friction gate, "
    "margin, RiskPool, max drawdown / positions / losses",
)

#: Funding events earn a held position's breakeven (management),
#: never an entry.
SHARED_MANAGEMENT_ONLY = (
    "funding events (structure/displacement/sweep prints) — "
    "management only, never entry evidence",
)

#: Per-playbook map.  "detector" = what live_signal actually
#: reads; "authority" = which route may order; "notes" =
#: anything a reader must not over-read.
PLAYBOOK_EVIDENCE: dict[str, dict] = {
    "fvg_limit": {
        "route": "h1_roster",
        "detector": "own-TF closed OHLC + FVG tracker state; "
                    "first touch of a holding gap, limit at the "
                    "near edge",
        "authority": "h1_trigger streaming evaluation only; "
                     "scheduled scans observe, never route",
        "notes": "",
    },
    "fvg_midline": {
        "route": "h1_roster",
        "detector": "own-TF closed OHLC + FVG tracker state; "
                    "fresh CHoCH, resting limit at the midline "
                    "of a fresh untraded FVG",
        "authority": "h1_trigger streaming evaluation only; "
                     "scheduled scans observe, never route",
        "notes": "",
    },
    "inversion": {
        "route": "h1_roster",
        "detector": "own-TF closed OHLC + own FVG registry; "
                    "gap formed, bar CLOSES through the whole "
                    "gap (inversion), later bar retests without "
                    "a reclaim close — the retest bar's close IS "
                    "the entry; there is no second trigger "
                    "predicate after it",
        "authority": "h1_trigger streaming evaluation only; "
                     "scheduled scans observe, never route",
        "notes": "The 2026-10-26 assessment's inversion WATCH "
                 "came from a warmup_scan sweep: the detector "
                 "predicates (formed/inversion/retest indices) "
                 "are on the record, but the scan is not the "
                 "trigger path and records no trigger verdict.",
    },
    "judas_swing": {
        "route": "h1_roster",
        "detector": "own-TF closed OHLC + session windows "
                    "(DST-aware); Asia-edge sweep that fails, "
                    "then the real move at the London open",
        "authority": "h1_trigger streaming evaluation only; "
                     "scheduled scans observe, never route",
        "notes": "",
    },
    "killzone_open": {
        "route": "h1_roster",
        "detector": "own-TF closed OHLC + session blocks; "
                    "session-range break + retest at a session "
                    "open, dies at the session boundary or 6 "
                    "bars after breakout",
        "authority": "h1_trigger streaming evaluation only; "
                     "scheduled scans observe, never route",
        "notes": "",
    },
    "narrative": {
        "route": "h1_roster",
        "detector": "institutional chain over resampled frames "
                    "derived from the input frame: HTF zone -> "
                    "liquidity sweep -> displacement -> "
                    "CHoCH/BOS + FVG -> entry (FVG / OB / "
                    "breaker / mitigation block)",
        "authority": "h1_trigger streaming evaluation only; "
                     "scheduled scans observe, never route",
        "notes": "UNCERTAIN (flagged, not resolved): the chain's "
                 "frames are resampled from the input frame, "
                 "so they carry no future data, but their "
                 "granularity is derived, not native ticks — "
                 "a reader should not treat narrative's M15/M5 "
                 "reads as native M15/M5 bars.",
    },
    "order_block": {
        "route": "h1_roster",
        "detector": "own-TF closed OHLC; last opposing candle "
                    "before a displacement bar (body >= max(1.0 "
                    "x ATR14, threshold))",
        "authority": "h1_trigger streaming evaluation only; "
                     "scheduled scans observe, never route",
        "notes": "",
    },
    "retracement_trap": {
        "route": "h1_roster",
        "detector": "own-TF closed OHLC + swing fractals "
                    "(n=3); false breakdown/breakout trap at a "
                    "confirmed swing level",
        "authority": "h1_trigger streaming evaluation only; "
                     "scheduled scans observe, never route",
        "notes": "",
    },
    "sweep_reversal": {
        "route": "h1_roster",
        "detector": "bar states on the entry frame; sweep -> "
                    "displacement + CHoCH/BOS confirmation",
        "authority": "h1_trigger streaming evaluation only; "
                     "scheduled scans observe, never route",
        "notes": "UNCERTAIN (flagged, not resolved): the "
                 "detector's docstring still describes a "
                 "daily-bias veto (Phase-2 wording).  The "
                 "shared trigger path weighs D1 as evidence, "
                 "never a veto; whether the detector itself "
                 "still vetoes internally was not verified in "
                 "this pass.",
    },
    "trendline_retest": {
        "route": "h1_roster",
        "detector": "own-TF closed OHLC + trendline state; "
                    "break of a respected trendline, 2 bars "
                    "away, first return touch within 12 bars",
        "authority": "h1_trigger streaming evaluation only; "
                     "scheduled scans observe, never route",
        "notes": "",
    },
    "turtle_soup": {
        "route": "h1_roster",
        "detector": "own-TF closed OHLC; failed breakout of a "
                    "20-bar extreme",
        "authority": "h1_trigger streaming evaluation only; "
                     "scheduled scans observe, never route",
        "notes": "",
    },
    "v_reversal": {
        "route": "h1_roster",
        "detector": "own-TF closed OHLC; sharp V spike — fresh "
                    "20-bar extreme + capitulation reclaim",
        "authority": "h1_trigger streaming evaluation only; "
                     "scheduled scans observe, never route",
        "notes": "",
    },
    "range_sweep_reclaim": {
        "route": "range_h4_h1 (§G)",
        "detector": "H4 range state (causally sliced: only H4 "
                    "bars closed by the H1 bar feed the "
                    "detector) + H1 refinement; sweep of a "
                    "range edge reclaimed",
        "authority": "NONE in v1 — §G shadow: journaled for "
                     "grading, never orders",
        "notes": "D1 bias is recorded as context only, never "
                 "consulted as permission (§G: D1 never vetoes).",
    },
    "range_qm": {
        "route": "range_h4_h1 (§G)",
        "detector": "H4 range state (causally sliced, as "
                    "above) + H1 refinement; QM structure "
                    "inside the range",
        "authority": "NONE in v1 — §G shadow: journaled for "
                     "grading, never orders",
        "notes": "D1 bias is context only (§G).",
    },
    "range_mean_reversion": {
        "route": "range_h4_h1 (§G)",
        "detector": "H4 range state (causally sliced, as "
                    "above) + H1 refinement; fade of a range "
                    "edge",
        "authority": "NONE in v1 — §G shadow: journaled for "
                     "grading, never orders",
        "notes": "D1 bias is context only (§G).",
    },
    "range_compression_breakout": {
        "route": "range_h4_h1 (§G)",
        "detector": "H4 range state (causally sliced, as "
                    "above) + H1 refinement; compression then "
                    "breakout of the range",
        "authority": "NONE in v1 — §G shadow: journaled for "
                     "grading, never orders",
        "notes": "D1 bias is context only (§G).",
    },
    "vol_expansion_escape": {
        "route": "volatility_expansion (Pass 7a scheduled)",
        "detector": "the MarketSituation's TFStates + the "
                    "formation TF's closed bars: a Bollinger "
                    "squeeze run (>= 5 bars) with ATR "
                    "contraction (atr_regime method) defines "
                    "the coil on H4/H1/M15; corroboration = "
                    "another TF's contracting state in the "
                    "situation; displacement close beyond "
                    "the coil, entry at the ACCEPTANCE close "
                    "(a following bar whose whole range holds "
                    "beyond the coil)",
        "authority": "NONE — Pass 7a shadow: scheduled scans "
                     "journal evaluations + arm records and "
                     "propose book scenarios; never orders. "
                     "Promotion is the family's own "
                     "evaluation (execution/vol_promotion.py) "
                     "plus the owner's separate decision",
        "notes": "Distinct from range_compression_breakout: "
                 "that playbook trades a RangeDetector range "
                 "OBJECT on H4 in the D1-flat range regime, "
                 "entering on the breakout close; this family "
                 "reads no range object and no regime verdict "
                 "— the coil is the volatility state's own "
                 "geometry on the squeeze's clock, and the "
                 "entry is the acceptance close, never the "
                 "breakout print.",
    },
    "vol_failed_expansion": {
        "route": "volatility_expansion (Pass 7a scheduled)",
        "detector": "same coil + trigger as "
                    "vol_expansion_escape; the resolution is "
                    "a close back INSIDE the coil before any "
                    "acceptance (within 3 formation bars of "
                    "the trigger) — the escape rejected. "
                    "Fade to the coil mid, then the opposite "
                    "edge; stop beyond the escape's excursion "
                    "extreme",
        "authority": "NONE — Pass 7a shadow, as "
                     "vol_expansion_escape",
        "notes": "The family's second hypothesis shares the "
                 "escape's coil/trigger measurement by "
                 "construction (one state machine, two "
                 "resolutions) — it is not a separate "
                 "detector and cannot double-fire with the "
                 "escape on the same sequence.",
    },
    "trend_impulse_pullback": {
        "route": "trend_momentum (Pass 7b scheduled)",
        "detector": "two clocks, causally sliced: the "
                    "formation TF's (H4 / D1) closed bars + "
                    "the trigger TF's (H1 / H4) closed bars "
                    "+ the MarketSituation snapshot.  The "
                    "trend STATE is read from the situation "
                    "(formation TF's canonical bias + bedrock "
                    "structure phase + the formation->trigger "
                    "relationship kind); the sequence is "
                    "measured on the formation bars "
                    "(confirmed-swing impulse break with "
                    "momentum sustained at the extreme; "
                    "pullback retracing 0.15-0.85 of the leg "
                    "with no formation close beyond the "
                    "impulse origin; counter-momentum never "
                    "sustained; formation momentum never "
                    "absorbed at the turn); the entry is the "
                    "pullback's minor-structure turn on the "
                    "trigger clock",
        "authority": "NONE — Pass 7b shadow: scheduled scans "
                     "journal evaluations + arm records and "
                     "propose book scenarios; never orders. "
                     "Promotion is the family's own "
                     "evaluation (execution/trend_promotion.py) "
                     "plus the owner's separate decision",
        "notes": "Distinct from trendline_retest / fvg_*: "
                 "those are level-return patterns (price "
                 "returns to a drawn level / gap).  This "
                 "family reads no level — its trigger is "
                 "the momentum state transition (the "
                 "pullback's structure turn) and its "
                 "invalidation is the trend's structural "
                 "death point (a formation close beyond the "
                 "defended swing / the impulse origin), not "
                 "a level's far edge.  Phase is read from "
                 "structure (confirmed swings), never an "
                 "EMA.",
    },
    "trend_momentum_acceleration": {
        "route": "trend_momentum (Pass 7b scheduled)",
        "detector": "the formation TF's closed bars + the "
                    "MarketSituation snapshot: the trend "
                    "state must already exist (formation TF "
                    "trend-phased, canonical bias in the "
                    "trade's direction, formation->trigger "
                    "pair ALIGNED — live participation), "
                    "formation momentum sustained, velocity "
                    "(ATR-normalised close velocity) rising "
                    "and >= the 80th percentile of its own "
                    "trailing-100 distribution, and a fresh "
                    "acceptance — a close beyond the last "
                    "confirmed formation swing within the "
                    "last two formation bars, every close "
                    "since holding beyond it",
        "authority": "NONE — Pass 7b shadow, as "
                     "trend_impulse_pullback",
        "notes": "Continuation, not breakout: a break of a "
                 "swing in a non-trend state, or without "
                 "the velocity percentile, is not this "
                 "family's signal (it is the vol family's "
                 "ground when it escapes a coil, or "
                 "nobody's).  The hypothesis shares the "
                 "family's trend-state read with "
                 "trend_impulse_pullback by construction "
                 "— one state, two entries (pullback / "
                 "continuation).",
    },
    "qm_m5": {
        "route": "native M5",
        "detector": "M5 closed bars only (feeding it H1 bars "
                    "was the qm_m5 defect); QM head/neckline/ "
                    "level, break of structure, retracement, "
                    "resting limit at the POI; validity window "
                    "1 bar; source must be live_close to reach "
                    "the order path",
        "authority": "native M5 order path (live_close source, "
                     "inside validity window)",
        "notes": "",
    },
    "qm_d1": {
        "route": "native D1",
        "detector": "D1 closed bars; same QM structure as "
                    "qm_m5 (head, neckline, QM level, break of "
                    "structure, retracement, resting limit at "
                    "the POI); validity window 1 bar; source "
                    "must be live_close to reach the order path",
        "authority": "native D1 order path in the runner "
                     "(P3/A14); see note",
        "notes": "UNCERTAIN (flagged, not resolved): the "
                 "playbook docstring labels qm_d1 SHADOW while "
                 "the runner wires a live order path.  The "
                 "runner is treated as authority here; the "
                 "docstring needs reconciling, not silent "
                 "editing.",
    },
}


def evidence_map() -> dict:
    """The full map: shared inputs, census (evidence-only),
    and one row per registered playbook."""
    return {
        "shared_decision_inputs": list(SHARED_DECISION_INPUTS),
        "shared_management_only": list(SHARED_MANAGEMENT_ONLY),
        "census_evidence_only": list(CENSUS_EVIDENCE_ONLY),
        "playbooks": {name: dict(row)
                      for name, row in PLAYBOOK_EVIDENCE.items()},
    }
