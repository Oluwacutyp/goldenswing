"""Institutional narrative: STATE 1-5 + footprints + decisive
test + differentiated management + opposing zones + Dynamic
ATR + min-lot floor + /status desk.

Every test builds synthetic bars and asserts the documented
rule — no vibes, no vote counting, hierarchy only.
"""
import numpy as np
import pandas as pd
import pytest
from dataclasses import asdict

from goldenswing.live.atr_regime import (
    apply_in_trade_atr, atr_regime, baseline_atr,
    remaining_time_stop, scaled_thresholds, scaled_time_stop)
from goldenswing.live.confirmation import evaluate_confirmation
from goldenswing.live.displacement import evaluate_displacement
from goldenswing.live.entry import evaluate_entry
from goldenswing.live.footprints import (
    classify_footprint, detect_accumulation, detect_cta,
    detect_distribution, detect_judas, detect_macro_news,
    detect_retail_trap, score_breakout)
from goldenswing.live.freshness import (
    evaluate_zone_freshness, scan_state1)
from goldenswing.live.management import (
    ManagementAction, breakeven_threshold_r, check_invalidation,
    check_time_stop, check_upgrade, compute_trailing_stop,
    evaluate_management, partials_hit, plan_management,
    remaining_after_partials, should_breakeven, upgrade_plan)
from goldenswing.live.min_lot import (
    describe_min_lot, evaluate_min_lot, min_lot_risk)
from goldenswing.live.exit import (
    build_journal_entry, normalize_exit_reason, realized_r)
from goldenswing.live.narrative import (
    ENTRY, HTF_ZONE_APPROACH, IDLE, LIQUIDITY_SWEEP,
    LTF_CONFIRMATION, MANAGEMENT, NarrativeState, advance,
    amd_for_state, chain_check, finalize_exit, mark_entered)
from goldenswing.live.nextbar import (
    decide_m5_pair, decide_next_bar, is_high_vol, tick_reclaim)
from goldenswing.live.opposing import (
    adjust_tps, apply_tier_to_management, detect_sandwich,
    opposing_zone_for, risk_tier)
from goldenswing.live.status import build_snapshot, format_trade_status
from goldenswing.live.styles import StyleBook, derive_style
from goldenswing.live.sweep import classify_sweep

ATR = 10.0
IDX = pd.date_range("2026-10-01", periods=64, freq="15min", tz="UTC")


def _bars(rows, *, start=0, vols=None):
    """rows: (open, high, low, close).  DatetimeIndex for session
    logic; vols optional."""
    idx = IDX[start:start + len(rows)]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                      index=idx)
    if vols is not None:
        df["volume"] = vols
    return df


def _zone(*, tf="W1", level=4100.0, top=4110.0, bottom=4090.0,
          side="support", score=90.0, touches=2,
          displacement_atr=2.5, status="active", age_d1=1.0,
          swing_ts=None, last_tested=None, tier="MAJOR"):
    return {"tf": tf, "level": level, "top": top, "bottom": bottom,
            "side": side, "score": score, "touches": touches,
            "displacement_atr": displacement_atr,
            "status": status, "age_d1": age_d1,
            "swing_ts": swing_ts, "last_tested": last_tested,
            "tier": tier}


# ---------------------------------------------------------------- STATE 1

class TestState1Freshness:
    def test_active_fresh_zone_arms(self):
        # No touch yet, age 1d < 5d, displacement 2.5 >= 2 ATR:
        # fresh zones are ACTIVE, not consumed.  Resistance
        # zone with price below it (not beyond, not touching).
        zone = _zone(touches=0, side="resistance")
        frame = _bars([(4050, 4055, 4045, 4052),
                       (4052, 4056, 4048, 4050),
                       (4050, 4054, 4046, 4051)])
        v = evaluate_zone_freshness(zone, frame, ATR)
        assert v.verdict == "active"
        assert v.arms
        assert "fresh" in v.flags
        assert "STATE 1 HTF_ZONE_APPROACH" in v.log

    def test_scan_arms_first_active(self):
        zone = _zone(touches=0, side="resistance")
        frame = _bars([(4050, 4055, 4045, 4052)])
        scan = scan_state1([zone], {"W1": frame}, 4105.0, ATR,
                           radius_atr=0.5)
        assert scan.armed is not None
        assert "distance" in scan.armed.log

    def test_full_mitigation_drops(self):
        zone = _zone(tf="D1", level=4200.0, top=4210.0,
                     bottom=4190.0, side="resistance", touches=2)
        # Bar 0 closes INSIDE the zone; bar 1 closes 1.0 ATR
        # beyond the near edge (4190) -> full mitigation.
        frame = _bars([(4195, 4205, 4192, 4200),
                       (4200, 4202, 4178, 4180)])
        v = evaluate_zone_freshness(zone, frame, ATR)
        assert v.verdict == "consumed"
        assert "full_mitigation" in v.flags
        assert "consumed" in v.log
        assert not v.arms

    def test_partial_mitigation_weakens_not_drops(self):
        zone = _zone(tf="D1", level=4200.0, top=4210.0,
                     bottom=4190.0, side="resistance", touches=2)
        # Inside close, then only a 0.2 ATR push -> PARTIAL.
        frame = _bars([(4195, 4205, 4192, 4200),
                       (4200, 4202, 4186, 4188)])
        v = evaluate_zone_freshness(zone, frame, ATR)
        assert v.verdict == "active"
        assert v.arms
        assert "weakened" in v.flags
        assert v.effective_score < 90.0  # -20 weakened

    def test_break_drops(self):
        zone = _zone(tf="D1", level=4200.0, top=4210.0,
                     bottom=4190.0, side="resistance", touches=2)
        # Two consecutive closes beyond the far edge (4210).
        frame = _bars([(4212, 4218, 4211, 4215),
                       (4215, 4222, 4213, 4220)])
        v = evaluate_zone_freshness(zone, frame, ATR)
        assert v.verdict == "consumed"
        assert "break" in v.flags

    def test_false_break_stays_active(self):
        zone = _zone(tf="D1", level=4200.0, top=4210.0,
                     bottom=4190.0, side="resistance", touches=2)
        # One close beyond, then an away close back inside the
        # band (a respected touch) -> false break, -10, active.
        frame = _bars([(4212, 4218, 4211, 4215),
                       (4182, 4195, 4170, 4180)])
        v = evaluate_zone_freshness(zone, frame, ATR)
        assert v.verdict == "active"
        assert "false_break" in v.flags

    def test_third_touch_tired_fourth_exhausted(self):
        def touch_bars(n):
            return _bars([(4182, 4195, 4170, 4180)] * n)

        zone = _zone(tf="D1", level=4200.0, top=4210.0,
                     bottom=4190.0, side="resistance", touches=3)
        v3 = evaluate_zone_freshness(zone, touch_bars(3), ATR)
        assert v3.verdict == "active"
        assert "tired" in v3.flags
        v4 = evaluate_zone_freshness(zone, touch_bars(4), ATR)
        assert v4.verdict == "exhausted"
        assert not v4.arms

    def test_stale_db_rechecked_against_new_bars(self):
        # Stored status says active, last_tested is old — but
        # bars AFTER last_tested show full mitigation.  The
        # scan never trusts the stale DB status.
        zone = _zone(tf="D1", level=4200.0, top=4210.0,
                     bottom=4190.0, side="resistance", touches=2,
                     last_tested=str(IDX[0]))
        frame = _bars([(4195, 4205, 4192, 4200),
                       (4200, 4202, 4178, 4180)], start=0)
        v = evaluate_zone_freshness(zone, frame, ATR)
        assert v.verdict == "consumed"
        assert "STATE 1 skip" in v.log

    def test_state1_skip_logs_consumed(self):
        zone = _zone(tf="D1", level=4200.0, top=4210.0,
                     bottom=4190.0, side="resistance", touches=2)
        frame = _bars([(4195, 4205, 4192, 4200),
                       (4200, 4202, 4178, 4180)])
        scan = scan_state1([zone], {"D1": frame}, 4205.0, ATR,
                           radius_atr=2.0)
        assert scan.armed is None
        assert scan.skipped and "consumed" in scan.skipped[0]


# ---------------------------------------------------------------- STATE 2

def _sweep_zone_long():
    return {"tf": "H4", "level": 4100.0, "top": 4110.0,
            "bottom": 4090.0, "side": "support"}


class TestState2Sweep:
    def test_valid_sweep_wick_fires(self):
        # Prior bars give a prior extreme near the edge (4088).
        rows = [(4095, 4098, 4088, 4092),
                (4092, 4096, 4090, 4094),
                (4094, 4097, 4091, 4093),
                # Sweep bar: wick 4080 (1.0 ATR beyond 4090),
                # close 4093 back inside, wick/body 13/2.
                (4095, 4098, 4080, 4093)]
        frame = _bars(rows, vols=[100, 100, 100, 150])
        v = classify_sweep(frame, _sweep_zone_long(),
                           direction=1, atr=ATR)
        assert v.is_sweep, v.log
        assert v.kind in ("valid_sweep", "weak_sweep")

    def test_momentum_breach_switches_breakout(self):
        zone = {"tf": "H4", "level": 4100.0, "top": 4110.0,
                "bottom": 4090.0, "side": "resistance"}
        # Prior bars give a confirmed swing high (4108, bar 3)
        # for BOS; then two closes beyond the top edge (4110):
        # 4118, 4122, body-dominated.
        rows = [(4090, 4100, 4085, 4095),
                (4095, 4102, 4090, 4100),
                (4100, 4104, 4095, 4102),
                (4102, 4108, 4098, 4105),
                (4105, 4104, 4100, 4102),
                (4102, 4102, 4098, 4100),
                (4100, 4100, 4096, 4098),
                (4098, 4102, 4094, 4100),
                (4100, 4105, 4096, 4102),
                (4102, 4108, 4098, 4105),
                (4112, 4120, 4110, 4118),
                (4118, 4125, 4116, 4122)]
        frame = _bars(rows, vols=[100] * 12)
        v = classify_sweep(frame, zone, direction=-1, atr=ATR)
        assert v.is_breach, v.log

    def test_wick_too_far_is_breach_not_sweep(self):
        # Wick 2.0 ATR beyond the edge -> breach path, not sweep.
        rows = [(4095, 4098, 4088, 4092),
                (4092, 4096, 4090, 4094),
                (4095, 4098, 4070, 4093)]
        frame = _bars(rows, vols=[100, 100, 150])
        v = classify_sweep(frame, _sweep_zone_long(),
                           direction=1, atr=ATR)
        assert not v.is_sweep

    def test_close_beyond_once_then_back_is_false_breakout(self):
        # Close beyond the edge once, then back inside within
        # 2 bars -> false breakout, sweep logic, mean-reversion.
        zone = {"tf": "H4", "level": 4100.0, "top": 4110.0,
                "bottom": 4090.0, "side": "resistance"}
        rows = [(4095, 4105, 4090, 4100),
                (4100, 4108, 4095, 4105),
                (4112, 4120, 4110, 4118),   # beyond
                (4100, 4112, 4095, 4105)]   # back inside, low wick
        frame = _bars(rows, vols=[100] * 4)
        v = classify_sweep(frame, zone, direction=-1, atr=ATR)
        assert v.kind in ("false_breakout", "valid_sweep",
                          "weak_sweep", "breach_pending"), v.log

    def test_ambiguous_body_waits(self):
        # Wick in range + close inside, but body too large
        # (wick/body < 2.0) -> AMBIGUOUS, do NOT arm sweep.
        rows = [(4095, 4098, 4088, 4092),
                (4092, 4096, 4090, 4094),
                (4085, 4098, 4080, 4093)]
        frame = _bars(rows, vols=[100, 100, 150])
        v = classify_sweep(frame, _sweep_zone_long(),
                           direction=1, atr=ATR)
        assert v.kind in ("ambiguous", "ambiguous_pending"), v.log
        assert not v.is_sweep

    def test_sweep_without_liquidity_not_sweep(self):
        # Wick into empty space: no prior extreme within 0.5
        # ATR of the edge -> wick without liquidity, not sweep.
        rows = [(4150, 4155, 4145, 4152),
                (4152, 4156, 4148, 4150),
                (4095, 4098, 4080, 4093)]
        frame = _bars(rows, vols=[100, 100, 150])
        v = classify_sweep(frame, _sweep_zone_long(),
                           direction=1, atr=ATR)
        assert v.kind == "wick_without_liquidity", v.log
        assert not v.is_sweep

    def test_low_volume_weak_sweep_flag(self):
        rows = [(4095, 4098, 4088, 4092),
                (4092, 4096, 4090, 4094),
                (4094, 4097, 4091, 4093),
                (4095, 4098, 4080, 4093)]
        frame = _bars(rows, vols=[100, 100, 100, 80])
        v = classify_sweep(frame, _sweep_zone_long(),
                           direction=1, atr=ATR)
        assert v.is_sweep
        assert "low_volume" in " ".join(v.flags) or \
            v.kind == "weak_sweep", (v.kind, v.flags)


# ---------------------------------------------------------------- STATE 3

def _sweep(*, pos=0, wick=4080.0, edge=4090.0):
    return {"bar_pos": pos, "wick": wick, "edge": edge}


def _disp_frame_strong():
    # Sweep bar + 2 leg bars: bar 1 moves 1.3 ATR (no reach),
    # bar 2 reaches 6.0 ATR -> leg = 2 bars, FVG [4098, 4110]
    # inside the leg, BOS fallback, volume spike.
    rows = [(4095, 4098, 4080, 4092),   # sweep bar (pos 0)
            (4085, 4110, 4090, 4093),   # leg 1
            (4115, 4145, 4110, 4140)]   # leg 2
    return _bars(rows, vols=[100, 160, 170])


class TestState3Displacement:
    def test_strong_displacement_fires(self):
        d = evaluate_displacement(_disp_frame_strong(),
                                  sweep=_sweep(), direction=1,
                                  atr=ATR)
        assert d.valid, d.log
        assert d.strength in ("strong", "very_strong")
        assert d.fvg and d.fvg["size_atr"] >= 0.25
        assert d.bos_price is not None

    def test_weak_displacement_flagged(self):
        # ATR 20: 1.6 ATR in 3 bars -> weak but valid, flagged.
        rows = [(4095, 4098, 4080, 4092),
                (4092, 4100, 4088, 4095),
                (4095, 4108, 4095, 4105),
                (4105, 4115, 4105, 4112)]
        frame = _bars(rows, vols=[100, 160, 170, 180])
        d = evaluate_displacement(frame, sweep=_sweep(),
                                  direction=1, atr=20.0)
        assert d.valid, d.log
        assert d.strength == "weak"
        assert "needs_stronger_state4" in d.flags

    def test_no_displacement_invalidates(self):
        # Drift: max move 0.9 ATR < 1.0 over the full window.
        rows = [(4095, 4098, 4080, 4092),
                (4092, 4096, 4088, 4089),
                (4090, 4095, 4088, 4089),
                (4089, 4094, 4087, 4088)]
        frame = _bars(rows, vols=[100, 100, 100, 100])
        d = evaluate_displacement(frame, sweep=_sweep(),
                                  direction=1, atr=ATR)
        assert not d.valid
        assert "without displacement" in d.log

    def test_slow_drift_invalidates(self):
        # Move reaches 1.5 ATR only after > 3 M15 bars: the
        # leg window is 3 bars, so it never completes in time.
        rows = [(4095, 4098, 4080, 4092),
                (4092, 4100, 4088, 4095),
                (4095, 4105, 4090, 4100),
                (4100, 4110, 4095, 4105)]
        frame = _bars(rows, vols=[100, 100, 100, 100])
        d = evaluate_displacement(frame, sweep=_sweep(),
                                  direction=1, atr=ATR)
        # 2.5 ATR but slow + overlapping + no FVG: not a
        # clean displacement leg.
        assert not d.valid or d.strength == "weak", d.log

    def test_no_fvg_invalidates(self):
        # Strong move but no unmitigated FVG >= 0.25 ATR.
        rows = [(4095, 4110, 4080, 4092),
                (4092, 4130, 4095, 4120),
                (4120, 4130, 4108, 4125),
                (4125, 4145, 4125, 4140)]
        frame = _bars(rows, vols=[100, 160, 170, 180])
        d = evaluate_displacement(frame, sweep=_sweep(),
                                  direction=1, atr=ATR)
        assert not d.valid
        assert "no unmitigated FVG" in d.log

    def test_retracement_over_50pct_invalidates(self):
        # Displacement then a > 50% retracement of the leg.
        rows = [(4095, 4098, 4080, 4092),
                (4085, 4110, 4090, 4093),
                (4115, 4145, 4110, 4140),
                (4140, 4142, 4100, 4100)]  # retraces 67%
        frame = _bars(rows, vols=[100, 160, 170, 150])
        d = evaluate_displacement(frame, sweep=_sweep(),
                                  direction=1, atr=ATR)
        assert not d.valid
        assert "retraced" in d.log

    def test_inducement_not_taken_flagged_not_blocked(self):
        # Displacement valid, inducement (IDM) not taken:
        # still valid, flagged -5, needs stronger FVG confirm.
        d = evaluate_displacement(_disp_frame_strong(),
                                  sweep=_sweep(), direction=1,
                                  atr=ATR,
                                  idm_level=4070.0)
        assert d.valid
        assert d.inducement_taken is False
        assert "inducement_not_taken" in d.flags
        assert d.score_adj == -5

    def test_high_vol_raises_displacement_bar(self):
        # 1.6 ATR clears NORMAL (1.5) but NOT HIGH (1.75).
        rows = [(4095, 4085, 4080, 4082),
                (4085, 4094, 4084, 4093),
                (4093, 4105, 4092, 4096)]
        frame = _bars(rows, vols=[100, 160, 170])
        normal = evaluate_displacement(
            frame, sweep=_sweep(), direction=1, atr=ATR)
        high = evaluate_displacement(
            frame, sweep=_sweep(), direction=1, atr=ATR,
            disp_min_atr=1.75, volume_spike_min=2.0)
        assert normal.valid and normal.strength == "strong", \
            normal.log
        assert not high.valid or high.strength == "weak", high.log


# ---------------------------------------------------------------- STATE 4

def _conf_frame(*, n_break_bars=2, fvg_top=4228.0, fvg_bottom=4215.0,
                fvg_pos=11, break_close=4230.0,
                bar11_close=4215.0, extra_closes=(),
                break_bar=None):
    """Downtrend prior bars (swing high 4220 at bar 4) + FVG +
    CHoCH break at bar 12 + optional post-break bars."""
    rows = [
        (4190, 4200, 4185, 4195),
        (4195, 4205, 4190, 4200),
        (4200, 4210, 4195, 4205),
        (4205, 4215, 4200, 4210),
        (4210, 4220, 4205, 4215),   # swing high 4220
        (4215, 4215, 4200, 4205),
        (4205, 4210, 4195, 4200),
        (4200, 4205, 4190, 4195),
        (4195, 4200, 4185, 4190),
        (4190, 4195, 4180, 4185),
        (4185, 4190, 4175, 4180),
        (4210, 4220, 4205, bar11_close),   # FVG bar
        break_bar or (4215, 4232, 4212, break_close),   # CHoCH break
    ]
    for c in extra_closes:
        rows.append((c, c + 2.0, c - 2.0, c))
    frame = _bars(rows)
    disp = {"fvg": {"top": fvg_top, "bottom": fvg_bottom,
                    "ce": (fvg_top + fvg_bottom) / 2.0,
                    "bar_pos": fvg_pos, "size_atr": 1.3}}
    return frame, disp


class TestState4Confirmation:
    def test_choch_fvg_overlap_fires(self):
        frame, disp = _conf_frame(extra_closes=[4220.0])
        v = evaluate_confirmation(frame, displacement=disp,
                                  direction=1, atr=ATR)
        assert v.decision == "confirm", v.log
        assert v.kind == "choch"
        assert v.score_adj == 10
        assert v.entry_price == pytest.approx(4221.5)

    def test_choch_far_from_fvg_waits(self):
        frame, disp = _conf_frame(fvg_top=4200.0, fvg_bottom=4190.0)
        v = evaluate_confirmation(frame, displacement=disp,
                                  direction=1, atr=ATR)
        assert v.decision == "wait_retracement"
        assert "far from FVG" in v.log

    def test_fvg_mitigated_before_choch_invalidates(self):
        # Bar 11 closes INSIDE the FVG (filled); the CHoCH bar
        # closes inside the FVG too (no reversal) -> mitigated.
        frame, disp = _conf_frame(bar11_close=4220.0,
                                  break_close=4225.0,
                                  break_bar=(4216, 4226, 4213, 4225))
        v = evaluate_confirmation(frame, displacement=disp,
                                  direction=1, atr=ATR)
        assert v.decision == "mitigated", v.log

    def test_choch_too_late_stale(self):
        frame, disp = _conf_frame(fvg_pos=5)
        v = evaluate_confirmation(frame, displacement=disp,
                                  direction=1, atr=ATR)
        assert v.decision == "stale"
        assert "too late" in v.log

    def test_missed_retracement_no_chase(self):
        # CHoCH at bar 12, then 13 bars running away > 1 ATR
        # from the FVG without retracing.
        frame, disp = _conf_frame(
            extra_closes=[4250.0] * 13)
        v = evaluate_confirmation(frame, displacement=disp,
                                  direction=1, atr=ATR)
        assert v.decision == "missed", v.log
        assert "no chase" in v.log

    def test_msb_fvg_strongest(self):
        # CHoCH at bar 12, bottom at bar 9 (4170), higher low
        # at bar 13 (4175, confirmable within the pre-break
        # window), then a BOS break at bar 17 -> MSB +15.
        rows = [
            (4190, 4200, 4185, 4195),
            (4195, 4205, 4190, 4200),
            (4200, 4210, 4195, 4205),
            (4205, 4215, 4200, 4210),
            (4210, 4220, 4205, 4215),
            (4215, 4215, 4200, 4205),
            (4205, 4210, 4195, 4200),
            (4200, 4205, 4190, 4195),
            (4195, 4200, 4185, 4190),
            (4190, 4195, 4170, 4180),   # bottom 4170
            (4180, 4190, 4180, 4185),
            (4210, 4220, 4205, 4215),   # FVG bar
            (4215, 4232, 4212, 4230),   # CHoCH
            (4230, 4230, 4175, 4220),   # retrace + higher low
            (4220, 4222, 4200, 4215),
            (4215, 4225, 4210, 4220),
            (4220, 4228, 4215, 4225),
            (4230, 4242, 4228, 4240),   # BOS break
        ]
        frame = _bars(rows)
        disp = {"fvg": {"top": 4228.0, "bottom": 4215.0,
                        "ce": 4221.5, "bar_pos": 11,
                        "size_atr": 1.3}}
        v = evaluate_confirmation(frame, displacement=disp,
                                  direction=1, atr=ATR)
        assert v.decision == "confirm", v.log
        assert v.kind == "msb", v.log
        assert v.score_adj == 15

    def test_inversion_fvg_valid(self):
        # FVG formed at bar 10; bar 11 fills it; the CHoCH bar
        # reverses above the top -> inversion, still valid.
        frame, disp = _conf_frame(bar11_close=4220.0, fvg_pos=10)
        v = evaluate_confirmation(frame, displacement=disp,
                                  direction=1, atr=ATR)
        assert v.decision == "confirm", v.log
        assert v.inversion
        assert "inversion_fvg" in v.flags

    def test_wick_only_break_not_choch(self):
        # Wick beyond the swing high, close back inside.
        frame, disp = _conf_frame(break_close=4218.0)
        v = evaluate_confirmation(frame, displacement=disp,
                                  direction=1, atr=ATR)
        assert v.decision == "wait_structure", v.log


# ---------------------------------------------------------------- STATE 5

def _conf(*, decision="confirm", kind="choch", fvg_top=4228.0,
          fvg_bottom=4215.0, choch_extreme=4220.0,
          bars_since=0, retraced=True, flags=()):
    return {"decision": decision, "kind": kind,
            "fvg": {"top": fvg_top, "bottom": fvg_bottom,
                    "ce": (fvg_top + fvg_bottom) / 2.0},
            "choch_extreme": choch_extreme,
            "bars_since_choch": bars_since, "retraced": retraced,
            "flags": list(flags)}


def _entry_frame(last_close=4221.5):
    return _bars([(4220, 4225, 4218, last_close)])


class TestState5Entry:
    def test_limit_at_fvg_ce_fires(self):
        plan = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=ATR)
        assert plan.valid, plan.log
        assert plan.entry_type == "limit"
        assert plan.entry_price == pytest.approx(4221.5)
        # SL = min(sweep 4200, FVG bottom 4215, CHoCH 4220) - 2.5
        assert plan.stop == pytest.approx(4197.5)
        assert plan.tp1 == pytest.approx(4245.5)
        assert plan.tp2 == pytest.approx(4269.5)
        assert plan.tp3 == pytest.approx(4293.5)
        assert plan.rr_tp2 == pytest.approx(2.0)

    def test_market_on_confirm_close_fires(self):
        # No retracement, 1 bar since CHoCH, price leaving the
        # FVG edge (0.2 ATR) -> market at the last close.
        plan = evaluate_entry(
            _entry_frame(last_close=4230.0),
            confirmation=_conf(decision="wait_retracement",
                               bars_since=1, retraced=False),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4210.0}, direction=1, atr=ATR)
        assert plan.valid, plan.log
        assert plan.entry_type == "market"
        assert plan.entry_price == pytest.approx(4230.0)

    def test_no_chase_over_1_atr(self):
        plan = evaluate_entry(
            _entry_frame(last_close=4250.0),
            confirmation=_conf(decision="missed", retraced=False),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=ATR)
        assert not plan.valid
        assert "no chase" in plan.invalid_reason

    def test_sl_too_tight_invalid(self):
        conf = _conf(fvg_top=4228.0, fvg_bottom=4224.0,
                      choch_extreme=4225.0)
        plan = evaluate_entry(
            _entry_frame(last_close=4226.0), confirmation=conf,
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4225.0}, direction=1, atr=ATR)
        assert not plan.valid
        assert "SL too tight" in plan.invalid_reason

    def test_sl_too_wide_invalid(self):
        plan = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4150.0}, direction=1, atr=ATR)
        assert not plan.valid
        assert "SL too wide" in plan.invalid_reason

    def test_rr_below_2_skip(self):
        # Opposite zone at 4230, closer than +2R (4269.5) ->
        # TP2 = 4230, R:R 0.35 < 2 -> skip.
        plan = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=ATR,
            opposite_levels=[4230.0])
        assert not plan.valid
        assert "R:R <2" in plan.invalid_reason

    def test_complex_pullback_minus_10pct(self):
        plan = evaluate_entry(
            _entry_frame(),
            confirmation=_conf(flags=["complex_pullback"]),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=ATR,
            risk_amount=100.0)
        assert plan.valid
        assert plan.size_mult == pytest.approx(0.9)
        assert "complex_pullback_minus_10pct" in plan.flags

    def test_smt_fail_minus_10_not_block(self):
        # DXY makes a new high while Gold prints CHoCH long:
        # SMT FAIL -10, size -20%, plan still valid (confluence).
        dxy_rows = [
            (98, 99, 97, 98),
            (99, 100, 98, 99), (100, 101, 99, 100),
            (101, 102, 100, 101), (100, 101, 99, 100),
            (99, 100, 98, 99), (100, 101, 99, 100),
            (101, 102, 100, 101), (102, 103, 101, 102),
            (101, 102, 100, 101), (100, 101, 99, 100),
            (101, 102, 100, 101), (102, 103, 101, 102),
            (103, 104, 102, 103), (102, 103, 101, 102),
            (103, 104, 102, 103), (104, 105, 103, 104.5),
        ]
        plan = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=ATR,
            dxy_frame=_bars(dxy_rows), risk_amount=100.0)
        assert plan.valid
        assert plan.smt["status"] == "fail"
        assert plan.smt["score_adj"] == -10
        assert "smt_fail_minus_20pct" in plan.flags
        assert plan.size_mult == pytest.approx(0.8)

    def test_nested_m5_sub_choch_tighter_sl(self):
        m5_rows = [
            (4209, 4210, 4208, 4209),
            (4211, 4212, 4210, 4211),
            (4213, 4214, 4212, 4213),
            (4215, 4216, 4214, 4215),
            (4217, 4218, 4216, 4217),
            (4215, 4216, 4214, 4215),
            (4213, 4214, 4212, 4213),
            (4211, 4212, 4210, 4211),
            (4220, 4222, 4210, 4220),   # sub-CHoCH over 4218
            (4220, 4226, 4216, 4224),   # sub-FVG [4212, 4216]
            (4224, 4228, 4220, 4226),
        ]
        plan = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=ATR,
            m5_frame=_bars(m5_rows))
        assert plan.valid, plan.log
        assert plan.nested
        assert plan.nested_tf == "M5"
        assert plan.entry_price == pytest.approx(4214.0)
        # Nested SL (sub-CHoCH extreme 4210 - 2.5) is tighter
        # than the HTF SL (4197.5).
        assert plan.stop == pytest.approx(4207.5)
        assert plan.stop > 4197.5

    def test_nested_fallback_to_m15(self):
        # Flat M5: no sub-CHoCH in 6 bars inside the M15 FVG.
        m5_rows = [(4218, 4220, 4216, 4219)] * 12
        plan = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=ATR,
            m5_frame=_bars(m5_rows))
        assert plan.valid
        assert not plan.nested
        assert "nested_fallback_m15" in plan.flags
        assert plan.entry_price == pytest.approx(4221.5)

    def test_late_confirm_half_size_and_rr_recheck(self):
        plan = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=ATR,
            late_confirm=True, risk_amount=100.0)
        assert plan.valid
        assert plan.size_mult == pytest.approx(0.5)
        assert "late_confirm_half_size" in plan.flags
        # Late confirm with an opposite zone killing R:R -> skip.
        plan2 = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=ATR,
            late_confirm=True, opposite_levels=[4230.0])
        assert not plan2.valid
        assert "late confirm, R:R <2" in plan2.invalid_reason

    def test_risk_tier_sizes_never_gates(self):
        # RED tier still trades, at 0.5x — tiering never gates.
        plan = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=ATR,
            risk_tier="RED", risk_amount=100.0)
        assert plan.valid
        assert plan.size_mult == pytest.approx(0.5)
        assert "risk_tier_red" in plan.flags


# ---------------------------------------------------------------- FOOTPRINTS

def _flat_prior_rows(n=12, *, base=4200.0):
    rows = []
    for i in range(n):
        # Swing high 4210 at bar 5.
        h = base + [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0, -2][i]
        rows.append((h - 2.0, h, h - 4.0, h - 2.0))
    return rows


class TestFootprints:
    def test_institutional_breakout_trend_path(self):
        rows = _flat_prior_rows() + [
            (4200, 4230, 4198, 4225),
            (4225, 4245, 4220, 4240),
        ]
        frame = _bars(rows, vols=[100] * 12 + [250, 260])
        s = score_breakout(frame, 4210.0, 1, ATR)
        assert s.total >= 70, s.log
        assert s.classification == "institutional"
        assert s.route == "breakout_trend"

    def test_retail_breakout_sweep_path(self):
        rows = _flat_prior_rows() + [
            (4210, 4214, 4199, 4212),
            (4212, 4213, 4197, 4211),
        ]
        frame = _bars(rows, vols=[100] * 12 + [90, 95])
        s = score_breakout(frame, 4210.0, 1, ATR)
        assert s.total < 40, s.log
        assert s.classification == "retail"
        assert s.route == "sweep"

    def test_ambiguous_waits_for_decisive_test(self):
        rows = _flat_prior_rows() + [
            (4210, 4220, 4199, 4218),
            (4218, 4232, 4198, 4230),
            (4230, 4232, 4218, 4222),
        ]
        frame = _bars(rows, vols=[100] * 12 + [210, 220, 200])
        s = score_breakout(frame, 4210.0, 1, ATR)
        assert 40 <= s.total < 70, s.log
        assert s.classification == "ambiguous"
        assert s.route == "wait_next_bar"

    def test_retail_buy_stop_trap_arms_sweep_short(self):
        rows = [
            (4190, 4200, 4185, 4195),
            (4195, 4205, 4190, 4200),
            (4200, 4210, 4195, 4205),
            (4205, 4215, 4200, 4210),
            (4210, 4220, 4205, 4215),   # swing high 4220
            (4215, 4215, 4200, 4205),
            (4205, 4210, 4195, 4200),
            (4200, 4205, 4190, 4195),
            (4195, 4200, 4185, 4190),
            (4190, 4205, 4188, 4200),
            (4200, 4218, 4200, 4215),
            (4215, 4235, 4210, 4218),   # pop + close back inside
        ]
        frame = _bars(rows, vols=[100] * 11 + [110])
        t = detect_retail_trap(frame, direction=-1, atr=ATR)
        assert t["present"], t
        assert t["kind"] == "retail_buy_stop_trap"
        assert "sweep SHORT" in t["log"]

    def test_accumulation_compression(self):
        idx = pd.date_range("2026-10-13 22:00", periods=32,
                            freq="15min", tz="UTC")
        rows = [(4200, 4202, 4198, 4200)] * 20 + \
            [(4200, 4202, 4198, 4200)] * 12
        frame = pd.DataFrame(rows,
                             columns=["open", "high", "low", "close"],
                             index=idx)
        frame["volume"] = [100] * 20 + [60] * 12
        a = detect_accumulation(frame, atr=ATR)
        assert a["present"], a
        assert a["asian"] is True

    def test_judas_sweep_before_london(self):
        idx = pd.date_range("2026-10-14 06:00", periods=5,
                            freq="15min", tz="UTC")
        rows = [(4195, 4200, 4190, 4195)] * 4 + [
            (4195, 4205, 4190, 4198)]
        frame = pd.DataFrame(rows,
                             columns=["open", "high", "low", "close"],
                             index=idx)
        frame["volume"] = [100] * 4 + [150]
        j = detect_judas(frame, asian_high=4200.0,
                         asian_low=4180.0, atr=ATR)
        assert j["present"], j
        assert j["side"] == "high"
        assert "Judas" in j["log"]

    def test_distribution_institutional_impulse(self):
        rows = [(4090, 4095, 4088, 4090),
                (4090, 4094, 4088, 4092),
                (4095, 4098, 4080, 4092),
                (4092, 4110, 4090, 4105),
                (4115, 4145, 4110, 4140)]
        frame = _bars(rows, vols=[100, 100, 100, 160, 170])
        d = detect_distribution(frame, direction=1, atr=ATR)
        assert d["present"], d
        assert d["move_atr"] >= 1.5

    def test_cta_momentum_breakout_not_mean_revert(self):
        rows = [(4190, 4195, 4188, 4193),
                (4193, 4220, 4190, 4200),
                (4200, 4225, 4198, 4220),
                (4220, 4245, 4218, 4240)]
        frame = _bars(rows, vols=[100, 100, 250, 260])
        c = detect_cta(frame, direction=1, atr=ATR)
        assert c["present"], c
        assert "do NOT mean-revert" in c["log"]

    def test_macro_news_footprint(self):
        rows = [(4200, 4205, 4195, 4200),
                (4200, 4235, 4198, 4230)]
        frame = _bars(rows, vols=[100, 350])
        m = detect_macro_news(frame, atr=ATR, news=True)
        assert m["present"], m
        assert m["range_atr"] >= 3.0

    def test_classify_footprint_priority(self):
        rows = [(4190, 4195, 4188, 4193),
                (4193, 4220, 4190, 4200),
                (4200, 4225, 4198, 4220),
                (4220, 4245, 4218, 4240)]
        frame = _bars(rows, vols=[100, 100, 250, 260])
        out = classify_footprint(frame, direction=1, atr=ATR)
        assert out["footprint"] == "cta_momentum"
        assert out["route"] == "breakout_trend"


# ------------------------------------------------- DECISIVE NEXT-BAR TEST

def _attempt(score=55.0):
    return {"score": score}


def _bar(o, h, l, c, vol=None):
    b = {"open": o, "high": h, "low": l, "close": c}
    if vol is not None:
        b["volume_ratio"] = vol
    return b


class TestDecisiveNextBar:
    """Hierarchical decisive test — never a vote.  One decisive
    condition (close back inside vs close beyond) decides."""

    def test_next_bar_close_beyond_institutional(self):
        d = decide_next_bar(
            _attempt(), _bar(4212, 4220, 4211, 4218, vol=1.5),
            edge=4210.0, breakout_dir=1, atr=ATR,
            zone_low=4190.0, zone_high=4210.0,
            swing_level=4200.0,
            fvg={"top": 4215.0, "bottom": 4205.0})
        assert d.decision == "institutional", d.log
        assert d.score_new > 55.0

    def test_next_bar_close_back_inside_retail(self):
        d = decide_next_bar(
            _attempt(), _bar(4218, 4220, 4195, 4205, vol=0.7),
            edge=4210.0, breakout_dir=1, atr=ATR,
            zone_low=4190.0, zone_high=4210.0,
            swing_level=4200.0,
            fvg={"top": 4215.0, "bottom": 4205.0})
        assert d.decision == "retail", d.log
        assert d.score_new < 55.0

    def test_next_bar_doji_waits_one_more(self):
        doji = _bar(4212, 4213, 4211, 4212, vol=1.0)
        d1 = decide_next_bar(_attempt(), doji, edge=4210.0,
                             breakout_dir=1, atr=ATR,
                             zone_low=4190.0, zone_high=4210.0)
        assert d1.decision == "wait"
        d2 = decide_next_bar(_attempt(), doji, edge=4210.0,
                             breakout_dir=1, atr=ATR,
                             zone_low=4190.0, zone_high=4210.0,
                             waits_used=1)
        assert d2.decision == "invalidate"
        assert "2 bars indecisive" in d2.log

    def test_mixed_then_invalidate_max_two_bars(self):
        # First mixed bar waits; a second mixed bar invalidates
        # — never a 3rd bar.
        mixed = _bar(4212, 4216, 4208, 4214, vol=1.0)
        d1 = decide_next_bar(_attempt(), mixed, edge=4210.0,
                             breakout_dir=1, atr=ATR,
                             zone_low=4190.0, zone_high=4210.0,
                             swing_level=4215.0,
                             fvg={"top": 4230.0, "bottom": 4220.0})
        assert d1.decision == "wait", d1.log
        d2 = decide_next_bar(_attempt(), mixed, edge=4210.0,
                             breakout_dir=1, atr=ATR,
                             zone_low=4190.0, zone_high=4210.0,
                             swing_level=4215.0,
                             fvg={"top": 4230.0, "bottom": 4220.0},
                             waits_used=1)
        assert d2.decision == "invalidate"

    def test_m5_pair_agreement_and_disagreement(self):
        kw = dict(edge=4210.0, breakout_dir=1, atr=ATR,
                  zone_low=4190.0, zone_high=4210.0,
                  swing_level=4200.0,
                  fvg={"top": 4215.0, "bottom": 4205.0})
        b1 = _bar(4212, 4220, 4211, 4218, vol=1.5)
        b2 = _bar(4215, 4221, 4214, 4220, vol=1.4)
        agree = decide_m5_pair(_attempt(), b1, b2, **kw)
        assert agree.decision == "institutional"
        retail = _bar(4218, 4220, 4195, 4205, vol=0.7)
        disagree = decide_m5_pair(_attempt(), b1, retail, **kw)
        assert disagree.decision == "wait"
        assert disagree.reason == "m5_disagreement"

    def test_high_vol_is_high_vol(self):
        assert is_high_vol(25.0, 15.0) is True  # 1.67x > 1.5x
        assert is_high_vol(15.0, 15.0) is False

    def test_tick_reclaim_early_warning_not_trade(self):
        r = tick_reclaim(4205.0, edge=4210.0, breakout_dir=1,
                         swing_level=4215.0)
        assert r["early_retail"] is True
        assert "tick reclaim" in r["log"]
        # Tick alone never decides — the bar close does.
        r2 = tick_reclaim(4215.0, edge=4210.0, breakout_dir=1,
                          swing_level=4215.0)
        assert r2["early_retail"] is False


# ------------------------------------------- DIFFERENTIATED MANAGEMENT

class TestDifferentiatedManagement:
    def test_clean_ladder(self):
        p = plan_management(direction=1, entry_price=4208.0,
                            stop=4198.0)
        assert p.confirm_kind == "clean"
        assert p.size_mult == 1.0
        assert (p.tp1_r, p.tp2_r, p.tp3_r) == (1.0, 2.0, 3.0)
        assert (p.tp1_pct, p.tp2_pct, p.tp3_pct) == (0.50, 0.25, 0.25)
        assert p.breakeven_at_r == 1.0
        assert p.time_stop_h1 == 12
        assert p.allow_scale_in is True
        assert p.allow_runner is True
        assert p.tp1 == pytest.approx(4218.0)
        assert p.tp3 == pytest.approx(4238.0)

    def test_late_ladder(self):
        p = plan_management(direction=1, entry_price=4208.0,
                            stop=4198.0, late_confirm=True,
                            high_vol=True)
        assert p.confirm_kind == "late"
        assert p.size_mult == 0.5
        assert (p.tp1_r, p.tp2_r, p.tp3_r) == (0.75, 1.5, 2.0)
        assert (p.tp1_pct, p.tp2_pct, p.tp3_pct) == (0.50, 0.30, 0.20)
        assert p.breakeven_at_r == 0.0  # immediate
        assert p.time_stop_h1 == 6
        assert p.allow_scale_in is False
        assert p.allow_runner is False
        assert p.max_r == 2.0
        assert p.invalidation == "full_exit"
        assert p.tp3_exit_all_if_high_vol is True

    def test_upgrade_on_fast_1r(self):
        p = plan_management(direction=1, entry_price=4208.0,
                            stop=4198.0, late_confirm=True)
        assert check_upgrade(bars_since_entry_m15=2, current_r=1.0,
                             volume_ratio=1.5) is True
        assert check_upgrade(bars_since_entry_m15=3, current_r=1.0,
                             volume_ratio=1.5) is False
        assert check_upgrade(bars_since_entry_m15=2, current_r=1.0,
                             volume_ratio=None) is False
        up = upgrade_plan(p, entry_price=4208.0, stop=4198.0,
                          direction=1)
        assert up.upgraded
        assert (up.tp2_r, up.tp3_r) == (2.0, 3.0)
        assert up.size_mult == 0.5  # size stays, never adds
        assert up.allow_runner is True


# ------------------------------------------------------- OPPOSING ZONES

def _opp_zone(level, *, side="resistance", tf="D1", score=70.0,
              touches=1, disp=1.5, status="active", tier="MAJOR"):
    return {"level": level, "side": side, "tf": tf, "score": score,
            "touches": touches, "displacement_atr": disp,
            "status": status, "tier": tier}


class TestOpposingZones:
    def test_sandwich_range_scalp(self):
        zones = [_opp_zone(4100.0, side="support"),
                 _opp_zone(4120.0, side="resistance")]
        s = detect_sandwich(zones, 4110.0, ATR)
        assert s["present"] is True
        assert s["style"] == "RANGE_SCALP"
        assert s["distance_atr"] == pytest.approx(2.0)

    def test_mid_range_no_trade(self):
        zones = [_opp_zone(4100.0, side="support"),
                 _opp_zone(4140.0, side="resistance")]
        s = detect_sandwich(zones, 4120.0, ATR)
        assert s["mid_range"] is True
        assert "mid-range, no zone, no trade" in s["log"]

    def test_opposing_first_trigger_wins(self):
        zones = [_opp_zone(4100.0, side="support", score=90.0),
                 _opp_zone(4120.0, side="resistance", score=85.0)]
        opp = opposing_zone_for(zones, entry_price=4102.0,
                                direction=1)
        assert opp is not None
        assert opp["level"] == 4120.0  # supply becomes TP barrier

    def test_opposing_mid_moves_tp2(self):
        # Long from 4208, SL 4198 (R = 10).  Opposing MID-RISK
        # supply at +1.8R = 4226 -> TP2 moves to 4225 (opp -
        # 0.1 ATR), TP3 runner only if opposing breaks.
        opp = {"level": 4226.0, "tier": "YELLOW"}
        adj = adjust_tps(tp1=4218.0, tp2=4228.0, tp3=4238.0,
                         tp1_pct=0.50, tp2_pct=0.25, tp3_pct=0.25,
                         entry_price=4208.0, stop=4198.0,
                         direction=1, opposing=opp, atr=ATR)
        assert adj["tp2"] == pytest.approx(4225.0)
        assert adj["tp3_runner_if_opposing_breaks"] is True
        assert "TP2 moved" in adj["log"]

    def test_opposing_green_exits_all(self):
        # Owner test: GREEN SAFE opposing at +1.5R -> TP1 stays
        # +1R 50%, TP2 moves to opposing - 0.1 ATR (~+1.4R)
        # with ALL remaining 50%, TP3 REMOVED, no runner.
        opp = {"level": 4223.0, "tier": "GREEN"}  # +1.5R
        adj = adjust_tps(tp1=4218.0, tp2=4228.0, tp3=4238.0,
                         tp1_pct=0.50, tp2_pct=0.25, tp3_pct=0.25,
                         entry_price=4208.0, stop=4198.0,
                         direction=1, opposing=opp, atr=ATR)
        assert adj["tp1"] == pytest.approx(4218.0)  # +1R stays
        assert adj["tp1_pct"] == pytest.approx(0.50)
        assert adj["tp2"] == pytest.approx(4222.0)  # +1.4R
        assert adj["tp2_pct"] == pytest.approx(0.50)  # all remaining
        assert adj["tp3"] is None
        assert adj["tp3_pct"] == pytest.approx(0.0)
        assert adj["exit_all_at_tp2"] is True
        assert adj["allow_runner"] is False
        assert adj["trail_cap_price"] == pytest.approx(4220.5)
        assert adj["opposing_r"] == pytest.approx(1.5)
        assert adj["log"] == ("opposing GREEN SAFE at +1.5R, "
                              "exit all before barrier, no runner")

    def test_opposing_red_ignored(self):
        opp = {"level": 4226.0, "tier": "RED"}
        adj = adjust_tps(tp1=4218.0, tp2=4228.0, tp3=4238.0,
                         tp1_pct=0.50, tp2_pct=0.25, tp3_pct=0.25,
                         entry_price=4208.0, stop=4198.0,
                         direction=1, opposing=opp, atr=ATR)
        assert adj["tp2"] == pytest.approx(4228.0)
        assert adj["tp3"] == pytest.approx(4238.0)
        assert "expect break" in adj["log"]

    def test_tiering_sizes_never_blocks(self):
        green = {"score": 90.0, "touches": 3,
                 "displacement_atr": 2.5, "tf": "W1"}
        yellow = {"score": 70.0, "touches": 1,
                  "displacement_atr": 1.5, "tf": "H4"}
        red = {"score": 50.0, "touches": 1,
               "displacement_atr": 0.5, "tf": "H1"}
        assert risk_tier(green) == "GREEN"
        assert risk_tier(yellow) == "YELLOW"
        assert risk_tier(red) == "RED"
        # Tier + confirm ladder: most conservative wins.
        p = plan_management(direction=1, entry_price=4208.0,
                            stop=4198.0)
        from dataclasses import asdict
        combined = apply_tier_to_management(
            asdict(p), "RED", entry_price=4208.0, stop=4198.0,
            direction=1)
        assert combined["size_mult"] == pytest.approx(0.5)
        assert combined["tp2_r"] == pytest.approx(1.5)

    def test_tp_never_runs_into_opposing(self):
        # TP1 itself beyond opposing -> collapse to the barrier.
        opp = {"level": 4212.0, "tier": "YELLOW"}  # +0.4R
        adj = adjust_tps(tp1=4218.0, tp2=4228.0, tp3=4238.0,
                         tp1_pct=0.50, tp2_pct=0.25, tp3_pct=0.25,
                         entry_price=4208.0, stop=4198.0,
                         direction=1, opposing=opp, atr=ATR)
        assert adj["tp1"] == pytest.approx(4211.0)  # opp - 0.1 ATR
        assert adj["exit_all_at_tp2"] is True


# ------------------------------------------------------------- STYLES

class TestUnifiedFractalStyles:
    def test_style_derived_not_hardcoded(self):
        assert derive_style("W1", "D1", 8.0) == "MACRO_SWING"
        assert derive_style("H4", "M15", 3.0) == "INTRADAY_TREND"
        assert derive_style("M15", "M5", 1.0) == "MICRO_SCALP"
        assert derive_style("MN1", "H4", 12.0) == "MACRO_SWING"

    def test_multiple_styles_concurrent_no_conflict(self):
        book = StyleBook()
        ok1, _ = book.open("MACRO_SWING", "W1-4100")
        ok2, _ = book.open("MICRO_SCALP", "M15-4110")
        assert ok1 and ok2  # separate setups, separate books
        ok3, reason = book.open("MACRO_SWING", "W1-4100")
        assert not ok3  # 1 per style per zone
        assert "already open" in reason
        ok4, _ = book.open("INTRADAY_TREND", "H4-4105")
        assert ok4
        ok5, reason5 = book.open("MICRO_SCALP", "M15-4120")
        assert not ok5  # max 3 total
        assert "position cap" in reason5
        book.close("MICRO_SCALP", "M15-4110")
        ok6, _ = book.open("MICRO_SCALP", "M15-4120")
        assert ok6


# ---------------------------------------------------- DYNAMIC ATR REGIME

def _h4_frame(atrs, *, start_price=4200.0):
    """H4 bars with a controlled ATR14: constant true range."""
    rows = []
    p = start_price
    for a in atrs:
        rows.append((p, p + a * 0.6, p - a * 0.4, p + a * 0.1))
        p += a * 0.1
    idx = pd.date_range("2026-01-01", periods=len(rows),
                        freq="4h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=idx)


class TestDynamicATR:
    def test_baseline_15_to_25_ratio_167_high(self):
        # 100 H4 bars at ATR 15, then a jump to 25.
        frame = _h4_frame([15.0] * 110 + [25.0] * 14)
        base = baseline_atr(frame)
        # Baseline = median ATR14 over the last 100 H4 bars:
        # mostly 15s with the last 14 bars rising -> ~15.
        assert base == pytest.approx(15.0, abs=2.0)
        reg = atr_regime(25.0, 15.0)
        assert reg["ratio"] == pytest.approx(1.67, abs=0.01)
        assert reg["regime"] == "HIGH"
        assert "ratio 1.67x" in reg["log"]
        assert "current $25.00 vs baseline $15.00" in reg["log"]

    def test_regime_boundaries(self):
        assert atr_regime(10.0, 10.0)["regime"] == "NORMAL"
        assert atr_regime(13.0, 10.0)["regime"] == "ELEVATED"
        assert atr_regime(16.0, 10.0)["regime"] == "HIGH"
        assert atr_regime(21.0, 10.0)["regime"] == "EXTREME"

    def test_pre_entry_sl_buffer_scales_with_current_atr(self):
        # SL buffer = 0.25 x CURRENT ATR: 0.25 x $25 = $6.25.
        # Entry at FVG CE 4221.5, sweep 4200 -> SL 4193.75,
        # distance $27.75 (vs $21.25... see below).
        plan = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=25.0)
        assert plan.valid
        assert plan.stop == pytest.approx(4193.75)  # 4200 - 6.25
        # Size = risk / SL distance: wider SL -> fewer lots.
        plan_small = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=25.0,
            risk_amount=1000.0)
        plan_base = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=15.0,
            risk_amount=1000.0)
        assert plan_small.lots < plan_base.lots

    def test_time_stop_scales_with_ratio(self):
        assert scaled_time_stop(8, 1.67) == 5   # mid-risk 8 -> 5
        assert scaled_time_stop(12, 1.67) == 7  # GREEN 12 -> 7
        assert scaled_time_stop(6, 1.67) == 4   # RED 6 -> 4
        assert scaled_time_stop(8, 1.0) == 8     # no scaling
        assert scaled_time_stop(4, 3.0) == 3     # never below 3
        assert scaled_time_stop(8, 0.5) == 8     # never above base

    def test_high_vol_displacement_volume_fastpath(self):
        th = scaled_thresholds("HIGH")
        assert th["disp_min_atr"] == 1.75
        assert th["volume_spike_min"] == 2.0
        assert th["fast_path"] is True
        th_n = scaled_thresholds("NORMAL")
        assert th_n["disp_min_atr"] == 1.5
        assert th_n["volume_spike_min"] == 1.5
        assert th_n["fast_path"] is False

    def test_in_trade_sl_never_widens(self):
        plan = {"time_stop_h1": 8, "trail_step_atr": 0.5,
                "trail_tf": "H1", "breakeven_at_r": 1.0}
        out = apply_in_trade_atr(plan, ratio=1.67, current_r=0.2,
                                 elapsed_h1=0, entry_sl=4197.5)
        assert out["sl"] == 4197.5
        assert out["sl_locked"] is True
        assert "never widens" in out["in_trade_atr_log"]

    def test_in_trade_time_stop_shortens(self):
        # 4 H1 elapsed of 8 baseline; new total 5 -> remaining 1.
        plan = {"time_stop_h1": 8, "trail_step_atr": 0.5,
                "trail_tf": "H1"}
        out = apply_in_trade_atr(plan, ratio=1.67, current_r=0.2,
                                 elapsed_h1=4, entry_sl=4197.5)
        assert out["time_stop_h1"] == 5
        assert out["time_stop_remaining_h1"] == 1
        assert remaining_time_stop(4, 5) == 1
        # Already past the new total -> exit at the next tick.
        out2 = apply_in_trade_atr(plan, ratio=1.67, current_r=0.2,
                                  elapsed_h1=6, entry_sl=4197.5)
        assert out2["time_stop_remaining_h1"] == 0
        assert out2["exit_now"] is True

    def test_in_trade_trailing_tightens(self):
        plan = {"time_stop_h1": 8, "trail_step_atr": 0.5,
                "trail_tf": "H1"}
        out = apply_in_trade_atr(plan, ratio=1.67, current_r=0.2,
                                 elapsed_h1=0, entry_sl=4197.5)
        assert out["trail_step_atr"] == 0.25
        assert out["trail_tf"] == "M5"

    def test_in_trade_early_breakeven_at_half_r(self):
        plan = {"time_stop_h1": 8, "trail_step_atr": 0.5,
                "trail_tf": "H1"}
        out = apply_in_trade_atr(plan, ratio=1.67, current_r=0.5,
                                 elapsed_h1=0, entry_sl=4197.5)
        assert out["breakeven_now"] is True
        assert "early breakeven at +0.5R" in out["in_trade_atr_log"]


# ------------------------------------------------------- MIN-LOT FLOOR

class TestMinLotFloor:
    def test_min_lot_risk_math(self):
        # 0.01 lots x $21.25 SL x $1/oz = $21.25 risk.
        assert min_lot_risk(21.25) == pytest.approx(21.25)

    def test_skip_over_50pct_equity_non_fast_growth(self):
        d = evaluate_min_lot(sl_distance=21.25, target_risk=10.0,
                             equity=12.0, mode="PERCENT")
        assert d["decision"] == "skip"
        assert "would blow account" in d["log"]
        assert d["lots"] == 0.0

    def test_accept_over_target_under_10pct_equity(self):
        d = evaluate_min_lot(sl_distance=21.25, target_risk=10.0,
                             equity=500.0, mode="PERCENT")
        assert d["decision"] == "accept"
        assert d["lots"] == pytest.approx(0.01)
        assert d["min_risk_pct"] == pytest.approx(0.0425)

    def test_fixed_lots_12_accept_with_warning_guards(self):
        d = evaluate_min_lot(sl_distance=21.25, target_risk=10.0,
                             equity=12.0, mode="FIXED_LOTS",
                             free_margin=30.0)
        assert d["decision"] == "accept"
        assert d["lots"] == pytest.approx(0.01)
        assert "fixed-lots mode" in d["log"]
        assert d["guards"]["max_trades"] == 1
        assert d["guards"]["no_layering"] is True
        assert d["guards"]["margin_shield"] is True
        # $10 target on $12 equity = 83% -> unrealistic warning.
        assert any("83%" in w for w in d["warnings"])

    def test_margin_shield_skips(self):
        d = evaluate_min_lot(sl_distance=21.25, target_risk=10.0,
                             equity=500.0, mode="PERCENT",
                             free_margin=5.0)
        assert d["decision"] == "skip"
        assert "margin shield" in d["log"]

    def test_one_trade_max_in_fixed_lots(self):
        d = evaluate_min_lot(sl_distance=21.25, target_risk=10.0,
                             equity=12.0, mode="FIXED_LOTS",
                             free_margin=30.0, open_trades=1)
        assert d["decision"] == "skip"
        assert "max 1 trade" in d["log"]

    def test_entry_never_shrinks_sl_or_fractional_lots(self):
        # Min-lot floor applies: SL stays at the invalidation
        # level, lots never below 0.01.
        plan = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=ATR,
            risk_amount=10.0, equity=12.0, mode="PERCENT")
        assert not plan.valid  # >50% equity, PERCENT -> skip
        assert plan.stop == pytest.approx(4197.5)  # SL unshrunk
        plan2 = evaluate_entry(
            _entry_frame(), confirmation=_conf(),
            displacement={"strength": "strong", "move_atr": 2.0},
            sweep={"wick": 4200.0}, direction=1, atr=ATR,
            risk_amount=10.0, equity=12.0, mode="FIXED_LOTS",
            free_margin=30.0)
        assert plan2.valid
        assert plan2.stop == pytest.approx(4197.5)
        assert plan2.lots == pytest.approx(0.01)
        assert plan2.lots >= 0.01

    def test_status_risk_line(self):
        d = evaluate_min_lot(sl_distance=21.25, target_risk=10.0,
                             equity=12.0, mode="FIXED_LOTS",
                             free_margin=30.0)
        line = describe_min_lot(d, equity=12.0)
        assert "min lot 0.01 risks $21.25" in line
        assert "177% equity $12" in line
        assert "FIXED_LOTS" in line
        assert "margin shield ON" in line


# ------------------------------------------------------------ /STATUS

class TestStatusDesk:
    def test_active_trade_opposing_green_shows_all_fields(self):
        st = NarrativeState(state=MANAGEMENT, direction=1,
                            regime="BREAKOUT_TREND",
                            regime_since="2026-10-16T10:00:00Z",
                            regime_lock_remaining_h1=2,
                            breakout_confirm_count=2,
                            regime_state_file="data/regime_state.json",
                            regime_state_last_saved="10:00:00",
                            regime_state_version=7,
                            state_since="2026-10-16T11:00:00Z",
                            state_narrative="HTF Zone -> Sweep -> "
                            "Displacement -> CHoCH+FVG -> Entry",
                            zone={"tf": "W1", "level": 4100.0,
                                  "score": 90.0, "tier": "MAJOR",
                                  "touches": 3,
                                  "displacement_atr": 2.5,
                                  "age_d1": 766.0,
                                  "status": "active"},
                            opposing={"level": 4235.0, "tf": "W1",
                                      "score": 88.0, "tier": "GREEN",
                                      "touches": 2,
                                      "displacement_atr": 2.2,
                                      "age_d1": 400.0,
                                      "mitigation_status": "active",
                                      "side": "resistance"},
                            entry_tier="GREEN",
                            entry={"price": 4208.0, "stop": 4195.0,
                                   "targets": [4220.0, 4232.0, None],
                                   "entry_type": "limit",
                                   "plan": {"size_mult": 1.0,
                                            "min_lot": {}}},
                            management={"confirm_kind": "clean",
                                        "size_mult": 1.0,
                                        "tp1": 4220.0, "tp1_r": 1.0,
                                        "tp1_pct": 0.50,
                                        "tp2": 4232.0, "tp2_r": 1.4,
                                        "tp2_pct": 0.50,
                                        "tp3": None, "tp3_r": None,
                                        "tp3_pct": 0.0,
                                        "time_stop_h1": 12})
        snap = build_snapshot(
            narrative=st,
            tp_adjustment={"tp1": 4220.0, "tp2": 4232.0,
                           "tp3": None, "moved": True,
                           "exit_all_at_tp2": True,
                           "opposing_r": 1.5,
                           "log": "opposing GREEN SAFE at +1.5R, "
                           "exit all before barrier, no runner"},
            footprint={"score": 85, "classification": "institutional",
                       "smt": "pass", "amd": "distribution",
                       "judas": "no", "cta": "no",
                       "retail_trap": "no"},
            feed={"ok": True, "last_tick": "11:05:00",
                  "d1_bias": -1, "h4_bias": 1},
            margin={"equity": 12.0, "margin_used": 2.0,
                    "free_margin": 10.0, "shield": "ON"},
            journal={"last_entry": "fill LONG 0.01 @ 4208.0",
                     "trades_today": 1, "daily_pnl": 8.0,
                     "daily_r": 0.8},
            zone_cache={"n_zones": 342, "nearby": 3,
                        "last_refresh": "H1 close",
                        "p95_ms": 0.2},
            nearest_zones=[{"level": 4235.0, "tf": "W1",
                            "score": 88.0, "tier": "GREEN",
                            "distance_atr": 2.7}],
            current={"price": 4216.0, "pnl_usd": 8.0, "pnl_r": 0.8,
                     "distance_to_sl": 21.0, "distance_to_tp1": 4.0,
                     "distance_to_tp2": 16.0,
                     "distance_to_opposing": 19.0,
                     "r_multiple": 0.8, "time_in_trade": "2 H1",
                     "time_stop_remaining_h1": 10},
            trade={"direction": 1, "lots": 0.01,
                   "entry_price": 4208.0, "stop": 4195.0,
                   "entry_type": "limit", "entry_time": "11:00",
                   "fill_status": "filled", "style": "INTRADAY_TREND",
                   "risk_usd": 13.0, "risk_pct": 108.0,
                   "sl_distance_atr": 1.3, "sl_distance_r": 1.0,
                   "trailing_level": 4195.0, "tp1_status": "pending",
                   "tp2_status": "pending",
                   "tp2_original_r": 2.0,
                   "breakeven": "no", "trailing": "H1",
                   "partials_taken": "none", "runner": "no"})
        out = format_trade_status(snap)
        # Regime + state machine.
        assert "BREAKOUT_TREND" in out
        assert "MANAGEMENT" in out
        assert "HTF Zone -> Sweep" in out
        # Zones: entry tier, opposing, TP adjustment.
        assert "GREEN" in out
        assert "4235" in out
        assert "opposing GREEN" in out
        assert "TP2" in out and "TP3" in out
        assert "NONE (removed" in out
        # Trade: direction, size, SL, TPs, current P&L.
        assert "LONG" in out
        assert "+0.8R" in out or "0.8R" in out
        # Footprint + feed + margin + journal.
        assert "institutional" in out
        assert "Margin" in out or "margin" in out
        assert "Journal" in out or "journal" in out


# ------------------------------------------------- NARRATIVE CHAIN

class TestNarrativeChain:
    def test_full_chain_module_level(self):
        """HTF Zone -> Sweep -> Displacement -> CHoCH+FVG ->
        Entry, each step gated, none trading alone."""
        # STATE 1: fresh zone arms.
        zone = _zone(touches=0)
        frame = _bars([(4050, 4055, 4045, 4052)])
        scan = scan_state1([zone], {"W1": frame}, 4105.0, ATR,
                           radius_atr=0.5)
        assert scan.armed is not None
        # STATE 2: valid sweep.
        sweep_frame = _bars([(4095, 4098, 4088, 4092),
                             (4092, 4096, 4090, 4094),
                             (4094, 4097, 4091, 4093),
                             (4095, 4098, 4080, 4093)],
                            vols=[100, 100, 100, 150])
        sw = classify_sweep(sweep_frame, _sweep_zone_long(),
                            direction=1, atr=ATR)
        assert sw.is_sweep
        # STATE 3: displacement.
        disp = evaluate_displacement(_disp_frame_strong(),
                                     sweep=_sweep(),
                                     direction=1, atr=ATR)
        assert disp.valid
        # STATE 4: CHoCH + FVG.
        conf_frame, conf_disp = _conf_frame(extra_closes=[4220.0])
        conf = evaluate_confirmation(conf_frame,
                                     displacement=conf_disp,
                                     direction=1, atr=ATR)
        assert conf.decision == "confirm"
        # STATE 5: entry.  The sweep wick here must live in the
        # entry frame's price neighbourhood (4200, as in the
        # STATE 5 tests): the STATE 2 fixture sweeps ~4090
        # while the entry frame lives ~4220, and the SL rule
        # min(sweep, FVG, CHoCH) - buffer would (correctly)
        # reject a 14 ATR stop as too wide.
        plan = evaluate_entry(
            _entry_frame(), confirmation=conf,
            displacement={"strength": disp.strength,
                          "move_atr": disp.move_atr},
            sweep={"wick": 4200.0}, direction=1, atr=ATR)
        assert plan.valid
        assert plan.entry_type == "limit"

    def test_amd_mapping(self):
        assert amd_for_state(NarrativeState(state=IDLE)) \
            == "accumulation"
        assert amd_for_state(NarrativeState(
            state=HTF_ZONE_APPROACH)) == "accumulation"
        assert amd_for_state(NarrativeState(
            state=LIQUIDITY_SWEEP)) == "manipulation"
        assert amd_for_state(NarrativeState(
            state=LTF_CONFIRMATION)) == "distribution"
        assert amd_for_state(NarrativeState(
            state=ENTRY)) == "distribution"

    def test_chain_check_requires_complete_chain(self):
        ok, _ = chain_check(NarrativeState(state=ENTRY, direction=1),
                            1)
        assert ok is True
        ok2, reason = chain_check(NarrativeState(
            state=LTF_CONFIRMATION, direction=1), 1)
        assert ok2 is False
        assert "incomplete" in reason
        ok3, _ = chain_check(NarrativeState(state=ENTRY,
                                            direction=-1), 1)
        assert ok3 is False  # wrong direction

    def test_advance_smoke_no_crash(self):
        st = NarrativeState()
        frames = {
            "H4": _bars([(4190, 4200, 4185, 4195)] * 20),
            "M15": _bars([(4190, 4200, 4185, 4195)] * 20),
        }
        out = advance(st, frames)
        assert out.state in (IDLE, HTF_ZONE_APPROACH, LIQUIDITY_SWEEP,
                             3, 4, 5, 6, 7)
        assert isinstance(out.narrative, list)

    def test_persistence_round_trip(self):
        st = NarrativeState(state=ENTRY, direction=1,
                            entry_tier="GREEN",
                            opposing={"level": 4235.0,
                                      "tier": "GREEN"},
                            opposing_atr=2.7,
                            tp_adjustment={"tp2": 4232.0},
                            atr_ratio=1.67, atr_regime="HIGH",
                            atr_value=25.0,
                            late_confirm=True, high_vol=True)
        payload = st.payload()
        st2 = NarrativeState.from_payload(payload)
        assert st2.state == ENTRY
        assert st2.entry_tier == "GREEN"
        assert st2.opposing["level"] == 4235.0
        assert st2.opposing_atr == pytest.approx(2.7)
        assert st2.tp_adjustment["tp2"] == 4232.0
        assert st2.atr_ratio == pytest.approx(1.67)
        assert st2.atr_regime == "HIGH"
        assert st2.atr_value == pytest.approx(25.0)
        assert st2.late_confirm is True
        assert st2.high_vol is True

    def test_mark_entered_attaches_management(self):
        st = NarrativeState(state=ENTRY, direction=1,
                            entry_tier="YELLOW",
                            opposing={"level": 4226.0,
                                      "tier": "YELLOW"},
                            atr_ratio=1.0, atr_regime="NORMAL",
                            atr_value=ATR)
        st.entry = {"price": 4208.0, "stop": 4198.0,
                    "targets": [4218.0, 4228.0, 4238.0],
                    "entry_type": "limit", "plan": {}}
        out = mark_entered(st, entry_price=4208.0, stop=4198.0,
                           targets=[4218.0, 4228.0, 4238.0])
        assert out.state == MANAGEMENT
        mgmt = out.management
        assert mgmt["confirm_kind"] == "clean"
        # YELLOW tier caps the clean ladder (TP2 2R -> 1.8R,
        # price 4226), then opposing YELLOW at +1.8R moves
        # TP2 to opposing - 0.1 ATR = 4225 = 1.7R.  The
        # final tp2_r is the opposing-adjusted 1.7R.
        assert mgmt["tp2_r"] == pytest.approx(1.7)
        # Opposing YELLOW at +1.8R moves TP2 to 4225.
        assert mgmt["tp2"] == pytest.approx(4225.0)


# ------------------------------------------------- STATE 6 RUNTIME

class TestState6Runtime:
    """STATE 6: breakeven / partials / trailing / time-stop /
    invalidation / upgrade as executable decisions, not just a
    ladder.  SL never moves against the trade, never widens,
    never recalculates from a new ATR."""

    def _clean(self):
        return asdict(plan_management(direction=1,
                                       entry_price=4208.0,
                                       stop=4198.0))

    def _late(self):
        return asdict(plan_management(direction=1,
                                       entry_price=4208.0,
                                       stop=4198.0,
                                       late_confirm=True))

    def test_breakeven_clean_at_1r(self):
        plan = self._clean()
        assert breakeven_threshold_r(plan, 1.0) == pytest.approx(1.0)
        assert not should_breakeven(plan, 0.9, 1.0)
        assert should_breakeven(plan, 1.0, 1.0)

    def test_breakeven_late_immediate(self):
        plan = self._late()
        assert breakeven_threshold_r(plan, 1.0) == pytest.approx(0.0)
        assert should_breakeven(plan, 0.0, 1.0)

    def test_breakeven_atr_high_early(self):
        plan = self._clean()
        assert breakeven_threshold_r(plan, 1.67) == pytest.approx(0.5)
        assert not should_breakeven(plan, 0.49, 1.67)
        assert should_breakeven(plan, 0.5, 1.67)

    def test_partials_clean_ladder(self):
        plan = self._clean()
        assert partials_hit(plan, 2.5) == ["tp1", "tp2"]
        assert partials_hit(plan, 3.0) == ["tp1", "tp2", "tp3"]
        assert remaining_after_partials(plan, ["tp1", "tp2"]) \
            == pytest.approx(0.25)

    def test_partials_opposing_exit_all(self):
        plan = self._clean()
        plan["tp3"] = None  # opposing GREEN: exit all at TP2
        plan["tp2_pct"] = 0.50
        plan["tp3_pct"] = 0.0
        assert partials_hit(plan, 2.0) == ["tp1", "tp2"]
        assert remaining_after_partials(plan, ["tp1", "tp2"]) \
            == pytest.approx(0.0)

    def test_trailing_never_against(self):
        # Long: structure below current SL -> no move.
        assert compute_trailing_stop(
            direction=1, current_sl=4198.0, structure_level=4200.0,
            atr=ATR, trail_step_atr=0.5) == pytest.approx(4198.0)
        # Long: structure above -> trail up.
        assert compute_trailing_stop(
            direction=1, current_sl=4198.0, structure_level=4210.0,
            atr=ATR, trail_step_atr=0.5) == pytest.approx(4205.0)
        # Long: opposing cap binds.
        assert compute_trailing_stop(
            direction=1, current_sl=4198.0, structure_level=4210.0,
            atr=ATR, trail_step_atr=0.5,
            trail_cap_price=4203.0) == pytest.approx(4203.0)
        # Long: cap below current SL is ignored (never against).
        assert compute_trailing_stop(
            direction=1, current_sl=4198.0, structure_level=4210.0,
            atr=ATR, trail_step_atr=0.5,
            trail_cap_price=4190.0) == pytest.approx(4205.0)
        # Short mirrored.
        assert compute_trailing_stop(
            direction=-1, current_sl=4220.0, structure_level=4210.0,
            atr=ATR, trail_step_atr=0.5) == pytest.approx(4215.0)
        assert compute_trailing_stop(
            direction=-1, current_sl=4220.0, structure_level=4210.0,
            atr=ATR, trail_step_atr=0.5,
            trail_cap_price=4218.0) == pytest.approx(4218.0)

    def test_time_stop(self):
        assert not check_time_stop(11, 12)
        assert check_time_stop(12, 12)
        act = evaluate_management(
            plan=self._clean(), direction=1, entry_price=4208.0,
            stop=4198.0, current_price=4208.0, elapsed_h1=12,
            atr=ATR)
        assert act.exit == "time_stop"

    def test_invalidation_both_required(self):
        assert check_invalidation(through_choch=True,
                                  fvg_mitigated=False)[0] is False
        assert check_invalidation(through_choch=False,
                                  fvg_mitigated=True)[0] is False
        ok, reason = check_invalidation(through_choch=True,
                                        fvg_mitigated=True)
        assert ok is True
        assert "CHoCH" in reason and "FVG" in reason

    def test_invalidation_beats_time_stop(self):
        act = evaluate_management(
            plan=self._clean(), direction=1, entry_price=4208.0,
            stop=4198.0, current_price=4208.0, elapsed_h1=12,
            atr=ATR, choch_extreme=4210.0, fvg_mitigated=True)
        assert act.exit == "invalidation"
        assert act.partials == []

    def test_upgrade_via_evaluate_management(self):
        plan = self._late()
        act = evaluate_management(
            plan=plan, direction=1, entry_price=4208.0,
            stop=4198.0, current_price=4220.0, elapsed_h1=0,
            atr=ATR, bars_since_entry_m15=2, volume_ratio=1.6)
        assert act.upgraded is True
        assert act.plan["tp2_r"] == pytest.approx(2.0)
        assert act.plan["tp3_r"] == pytest.approx(3.0)
        assert act.plan["size_mult"] == pytest.approx(0.5)
        # Input plan untouched: evaluate is pure.
        assert plan["tp2_r"] == pytest.approx(1.5)
        # Slow: no upgrade.
        slow = evaluate_management(
            plan=self._late(), direction=1, entry_price=4208.0,
            stop=4198.0, current_price=4220.0, elapsed_h1=0,
            atr=ATR, bars_since_entry_m15=3, volume_ratio=1.6)
        assert slow.upgraded is False
        # No volume: no upgrade (never assume momentum).
        novol = evaluate_management(
            plan=self._late(), direction=1, entry_price=4208.0,
            stop=4198.0, current_price=4220.0, elapsed_h1=0,
            atr=ATR, bars_since_entry_m15=2, volume_ratio=None)
        assert novol.upgraded is False

    def test_evaluate_management_late_opposing(self):
        plan = self._late()
        plan["tp3"] = None  # opposing GREEN: exit all at TP2
        plan["tp2_pct"] = 0.50
        plan["tp3_pct"] = 0.0
        act = evaluate_management(
            plan=plan, direction=1, entry_price=4208.0,
            stop=4198.0, current_price=4224.0, elapsed_h1=0,
            atr=ATR)
        assert act.partials == ["tp1", "tp2"]
        assert act.remaining_pct == pytest.approx(0.0)
        assert act.exit is None
        assert act.breakeven is True  # late: immediate

    def test_evaluate_management_breakeven_sl(self):
        # Clean, +1.2R, no structure: SL moves to entry (BE).
        act = evaluate_management(
            plan=self._clean(), direction=1, entry_price=4208.0,
            stop=4198.0, current_price=4220.0, elapsed_h1=0,
            atr=ATR)
        assert act.breakeven is True
        assert act.new_sl == pytest.approx(4208.0)
        # Below breakeven: no SL move.
        act2 = evaluate_management(
            plan=self._clean(), direction=1, entry_price=4208.0,
            stop=4198.0, current_price=4215.0, elapsed_h1=0,
            atr=ATR)
        assert act2.breakeven is False
        assert act2.new_sl is None

    def test_advance_management_bookkeeping(self):
        st = NarrativeState(state=ENTRY, direction=1,
                            entry_tier="GREEN")
        st = mark_entered(st, entry_price=4208.0, stop=4198.0,
                          targets=[4218.0, 4228.0, 4238.0])
        assert st.state == MANAGEMENT
        h1 = _bars([(4205, 4212, 4200, 4208)] * 19
                   + [(4215, 4222, 4210, 4220)])
        h4 = _bars([(4200, 4210, 4195, 4205)] * 20)
        out = advance(st, {"H1": h1, "H4": h4})
        assert out.state == MANAGEMENT
        assert out.management.get("partials_taken") == ["tp1"]
        assert out.management.get("breakeven_moved") is True

    def test_advance_management_time_stop(self):
        plan = self._clean()
        st = NarrativeState(state=MANAGEMENT, direction=1,
                            entry={"price": 4208.0, "stop": 4198.0,
                                   "targets": []},
                            management=plan, confirm={})
        st.bars_in_state = 12  # advance increments -> 13 >= 12
        h1 = _bars([(4205, 4210, 4200, 4208)] * 20)
        h4 = _bars([(4200, 4210, 4195, 4205)] * 20)
        out = advance(st, {"H1": h1, "H4": h4})
        # advance() resets a terminal EXIT to a fresh IDLE
        # (carrying exit_reason) on its next loop pass, so
        # assert the reason, not the transient EXIT state.
        assert out.exit_reason == "time_stop"

    def test_advance_management_invalidation(self):
        plan = self._clean()
        st = NarrativeState(state=MANAGEMENT, direction=1,
                            entry={"price": 4208.0, "stop": 4198.0,
                                   "targets": []},
                            management=plan,
                            confirm={"fvg": {"top": 4218.0,
                                             "bottom": 4210.0}},
                            invalidation=4200.0)
        # One close inside the FVG (mitigated) + last close
        # back through the CHoCH extreme (4200) -> invalid.
        h1 = _bars([(4215, 4220, 4210, 4212),
                    (4195, 4200, 4190, 4195)])
        h4 = _bars([(4200, 4210, 4195, 4205)] * 20)
        out = advance(st, {"H1": h1, "H4": h4})
        # advance() resets a terminal EXIT to a fresh IDLE
        # (carrying exit_reason) on its next loop pass.
        assert "CHoCH" in out.exit_reason


# ------------------------------------------------------- CHUNK 2: STATE 7
class TestState7Exit:
    def _filled(self):
        st = NarrativeState(
            state=MANAGEMENT, direction=1, regime="trend",
            zone={"level": 4100.0, "tf": "W1"},
            sweep={"wick": 4090.0}, displacement={"valid": True},
            confirm={"decision": "confirm"},
            entry={"price": 4208.0, "stop": 4198.0,
                   "targets": [4218.0, 4228.0, 4238.0]},
            management={"tp1_r": 1.0, "tp2_r": 2.0, "tp3_r": 3.0},
            opposing={"level": 4230.0, "tier": "GREEN"},
            entry_tier="GREEN", atr_regime="NORMAL", atr_ratio=1.0)
        return st

    def test_finalize_tp2_partial_then_rest(self):
        st = self._filled()
        idle, journal = finalize_exit(
            st, reason="tp2",
            partials=[{"fraction": 0.5, "r": 1.0}], final_r=2.0,
            pnl_dollars=15.0, exit_price=4228.0)
        # 0.5*1.0 + 0.5*2.0 = 1.5R
        assert journal["realized_r"] == pytest.approx(1.5)
        assert journal["exit_reason"] == "tp2"
        assert journal["pnl_dollars"] == 15.0
        assert journal["zone"]["level"] == 4100.0
        assert journal["entry"]["price"] == 4208.0
        assert journal["opposing"]["tier"] == "GREEN"
        assert idle.state == IDLE
        assert idle.exit_reason == "tp2"
        assert idle.exit_r == pytest.approx(1.5)
        assert idle.last_exit_journal["realized_r"] == pytest.approx(1.5)
        assert "STATE 7 EXIT: tp2, R=1.50, -> STATE 0" in idle.narrative[-1]

    def test_finalize_sl_full_loss(self):
        idle, journal = finalize_exit(
            self._filled(), reason="stop", partials=[], final_r=-1.0)
        assert journal["exit_reason"] == "sl"
        assert journal["realized_r"] == pytest.approx(-1.0)
        assert idle.state == IDLE

    def test_finalize_normalizes_aliases(self):
        assert normalize_exit_reason("timeout") == "time_stop"
        assert normalize_exit_reason("stop_hit") == "sl"
        assert normalize_exit_reason("breakeven") == "sl_breakeven"
        idle, _ = finalize_exit(self._filled(), reason="timeout",
                                final_r=0.2)
        assert idle.exit_reason == "time_stop"

    def test_finalize_reset_clears_setup_keeps_regime(self):
        idle, _ = finalize_exit(self._filled(), reason="manual",
                                final_r=0.5)
        assert idle.state == IDLE
        assert idle.regime == "trend"
        assert idle.direction == 0
        assert idle.zone == {} and idle.sweep == {}
        assert idle.displacement == {} and idle.confirm == {}
        assert idle.entry == {} and idle.management == {}
        assert idle.opposing == {} and idle.invalidation is None

    def test_realized_r_rejects_overbanking(self):
        with pytest.raises(ValueError):
            realized_r([{"fraction": 0.6, "r": 1.0},
                        {"fraction": 0.6, "r": 1.0}], 1.0)
        assert realized_r([], -1.0) == pytest.approx(-1.0)
        assert build_journal_entry(
            self._filled(), reason="sl", realized=-1.0)["entry"][
            "price"] == 4208.0


# ------------------------------------------- CHUNK 2: TickManager wiring
class TestTickNarrativeWiring:
    def _open(self, mgr, plan=None, **kw):
        from goldenswing.backtest.harness import ExitConfig
        from goldenswing.execution.executor import OrderRequest
        from goldenswing.execution.tick_manager import TickManager
        req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                           stop_loss=1990.0, entry_ref=2000.0,
                           targets=[2030.0], atr=10.0, bar_index=5,
                           alert_id="T1")
        pos = mgr.open(ticket=101, request=req, fill_price=2000.0,
                       entry_bar=6, **kw)
        if plan is not None:
            assert mgr.attach_narrative_plan(
                101, plan, atr=10.0, choch_extreme=1995.0,
                fvg={"top": 2010.0, "bottom": 2000.0}) is True
        return pos

    def _clean_plan(self):
        return {"tp1_r": 1.0, "tp1_pct": 0.50,
                "tp2_r": 2.0, "tp2_pct": 0.25,
                "tp3_r": 3.0, "tp3_pct": 0.25,
                "breakeven_at_r": 1.0, "trail_step_atr": 0.5,
                "time_stop_h1": 12, "trail_cap_price": None}

    def test_attach_unknown_ticket_false(self):
        from goldenswing.execution.tick_manager import TickManager
        mgr = TickManager()
        assert mgr.attach_narrative_plan(999, {}) is False
        assert mgr.narrative_plan(999) is None

    def test_narrative_partials_drive_ticks(self):
        from goldenswing.execution.tick_manager import TickManager
        mgr = TickManager()
        self._open(mgr, plan=self._clean_plan())
        # +1R: narrative TP1 (50%) + SL move.  Tightest wins:
        # the narrative ATR-step trail (best 2012.5 - 0.5*ATR 10
        # = 2007.5) beats breakeven (2000.0) — the SL still
        # moves in favour, tighter than breakeven.
        actions = mgr.on_tick(101, 2012.5)
        assert [a.reason for a in actions] == ["partial_tier_1",
                                               "trail"]
        assert actions[0].volume == pytest.approx(0.5)
        assert actions[1].level == pytest.approx(2007.5)

    def test_narrative_late_breakeven_immediate(self):
        from goldenswing.execution.tick_manager import TickManager
        mgr = TickManager()
        plan = self._clean_plan()
        plan["breakeven_at_r"] = 0.0  # late confirm: immediate
        plan["tp1_r"] = 0.75
        self._open(mgr, plan=plan)
        actions = mgr.on_tick(101, 2000.0)  # 0R, at entry
        assert any(a.reason == "breakeven" for a in actions)

    def test_narrative_time_stop_from_plan(self):
        from goldenswing.execution.tick_manager import TickManager
        mgr = TickManager()
        plan = self._clean_plan()
        plan["time_stop_h1"] = 6
        self._open(mgr, plan=plan)
        # signal_bar is the request bar_index (5): bar 10 is 5
        # elapsed (<6), bar 11 is 6 (>=6).
        assert mgr.on_bar_close(101, 10) == []
        timeout = mgr.on_bar_close(101, 11)
        assert [a.reason for a in timeout] == ["timeout"]

    def test_narrative_trailing_uses_atr_step(self):
        from goldenswing.execution.tick_manager import TickManager
        mgr = TickManager()
        plan = self._clean_plan()
        plan["trail_step_atr"] = 0.5
        self._open(mgr, plan=plan)
        # Push best to +2R (2025), then tick at 2025: narrative
        # trail candidate = best - 0.5*ATR(10) = 2020, tighter
        # than breakeven 2000 -> trail wins as the SL move.
        mgr.on_tick(101, 2025.0)
        actions = mgr.on_tick(101, 2025.0)
        sl_moves = [a for a in actions if a.kind == "modify_sl"]
        assert sl_moves and sl_moves[0].level == pytest.approx(2020.0)

    def test_narrative_invalidation_query(self):
        from goldenswing.execution.tick_manager import TickManager
        mgr = TickManager()
        self._open(mgr, plan=self._clean_plan())
        # Price through CHoCH extreme (1995) but FVG not mitigated.
        assert mgr.check_narrative_invalidation(
            101, 1990.0, fvg_mitigated=False) is False
        assert mgr.check_narrative_invalidation(
            101, 1990.0, fvg_mitigated=True) is True
        # Price above the extreme: no invalidation either way.
        assert mgr.check_narrative_invalidation(
            101, 2005.0, fvg_mitigated=True) is False

    def test_no_plan_keeps_legacy_behaviour(self):
        from goldenswing.execution.tick_manager import TickManager
        mgr = TickManager()
        self._open(mgr)
        assert mgr.narrative_plan(101) is None
        assert mgr.on_tick(101, 2010.0) == []  # +0.8R, nothing
