"""Feature 1: tiered partials + min-lot profit-lock + Chandelier trailing.

Setup: long, entry 2000, invalidation 1990, ATR 10 -> buffered
stop0 = 1987.50, risk = 12.50.  Default tiers 50/25/25 at 1R/2R/3R.
"""
import pytest

from goldenswing.backtest.harness import ExitConfig
from goldenswing.execution.executor import OrderRequest
from goldenswing.execution.tick_manager import TickConfig, TickManager


def _request(direction=1, targets=(2030.0,), volume=1.0):
    return OrderRequest(symbol="XAUUSD", direction=direction,
                        volume=volume,
                        stop_loss=1990.0 if direction == 1 else 2010.0,
                        entry_ref=2000.0, targets=list(targets),
                        atr=10.0, bar_index=5, alert_id="T1")


def _open(mgr, direction=1, **kw):
    req = _request(direction, **kw)
    return mgr.open(ticket=101, request=req, fill_price=2000.0,
                    entry_bar=6)


# ------------------------------------------------------- tiered partials
def test_tiers_fire_in_order_at_r_levels():
    mgr = TickManager()
    _open(mgr)
    # +1.0R: tier 1 banks 50%.
    a = mgr.on_tick(101, 2012.5)
    tier1 = [x for x in a if x.kind == "partial"][0]
    assert tier1.reason == "partial_tier_1"
    assert tier1.volume == pytest.approx(0.5)
    mgr.apply(101, tier1)
    assert mgr.position(101).volume == pytest.approx(0.5)
    assert mgr.position(101).tiers_done == 1
    # +1.5R: tier 2 NOT yet (needs 2R).
    assert not [x for x in mgr.on_tick(101, 2018.75)
                if x.kind == "partial"]
    # +2.0R: tier 2 banks 25% of ORIG (0.25 lots).
    a = mgr.on_tick(101, 2025.0)
    tier2 = [x for x in a if x.kind == "partial"][0]
    assert tier2.reason == "partial_tier_2"
    assert tier2.volume == pytest.approx(0.25)
    mgr.apply(101, tier2)
    assert mgr.position(101).volume == pytest.approx(0.25)
    # +3.0R: tier 3 banks the last 25%.
    a = mgr.on_tick(101, 2037.5)
    tier3 = [x for x in a if x.kind == "partial"][0]
    assert tier3.reason == "partial_tier_3"
    assert tier3.volume == pytest.approx(0.25)
    mgr.apply(101, tier3)
    assert mgr.position(101).volume == pytest.approx(0.0)
    assert mgr.position(101).closed is True
    # No fourth tier.
    assert not [x for x in mgr.on_tick(101, 2100.0)
                if x.kind == "partial"]


def test_tiers_skip_small_positions_use_pool_partial():
    # Below the 0.03-lot tier gate: the legacy pool partial applies.
    mgr = TickManager()
    _open(mgr, volume=0.01)
    a = mgr.on_tick(101, 2030.0)  # pool at 2030
    partials = [x for x in a if x.kind == "partial"]
    assert len(partials) == 1
    assert partials[0].reason == "partial"
    assert partials[0].volume == pytest.approx(
        0.5 * 0.01)  # partial_frac of orig


def test_tiers_never_over_close_on_gap_tick():
    mgr = TickManager()
    _open(mgr)
    # One gap tick from 0R to +3.5R: tier 1 fires this tick; tiers 2
    # and 3 follow on the next ticks (best_r is sticky, so a
    # reversal cannot skip them) — never all at once, never over.
    a = mgr.on_tick(101, 2043.75)
    assert [x.reason for x in a if x.kind == "partial"] == [
        "partial_tier_1"]
    mgr.apply(101, a[0])
    a = mgr.on_tick(101, 2000.0)  # full reversal — tiers still due
    assert [x.reason for x in a if x.kind == "partial"] == [
        "partial_tier_2"]
    mgr.apply(101, a[0])
    a = mgr.on_tick(101, 2000.0)
    assert [x.reason for x in a if x.kind == "partial"] == [
        "partial_tier_3"]


def test_custom_tiers_and_empty_tiers():
    mgr = TickManager(tick=TickConfig(
        partial_tiers=((0.5, 1.0),)))
    _open(mgr)
    a = mgr.on_tick(101, 2006.25)  # +0.5R
    assert [x.reason for x in a if x.kind == "partial"] == [
        "partial_tier_1"]
    assert a[0].volume == pytest.approx(1.0)


# --------------------------------------------------------- profit-lock
def test_profit_lock_level_is_max_of_risk_and_atr_legs():
    # ATR 10: ATR leg 0.75*10 = 7.5 beats the 0.5R leg 6.25.
    mgr = TickManager()
    _open(mgr)
    a = mgr.on_tick(101, 2012.5)
    lock = [x for x in a if x.reason == "profit_lock"][0]
    assert lock.level == pytest.approx(2007.5)
    mgr.apply(101, lock)
    assert mgr.position(101).profit_lock_done is True
    # One-shot: never re-fires.
    assert not [x for x in mgr.on_tick(101, 2015.0)
                if x.reason == "profit_lock"]


def test_profit_lock_uses_risk_leg_when_atr_unknown():
    mgr = TickManager()
    mgr.adopt(ticket=303, symbol="XAUUSD", direction=1, volume=1.0,
              entry_price=2000.0, sl=1987.5)
    a = mgr.on_tick(303, 2012.5)
    lock = [x for x in a if x.reason == "profit_lock"][0]
    assert lock.level == pytest.approx(2006.25)  # 0.5 * 12.5, no ATR


def test_profit_lock_fires_on_progress_without_1r():
    # TP at 2030 (2.4R away).  At +0.9R price has covered 37.5% —
    # below the 50% progress bar, so no lock.  At +1.3R (54% of the
    # way to TP) the progress leg fires even though... 1.3R is
    # already past the 1.0R trigger; use a far TP instead: TP 2100
    # (8R away) — at +0.9R progress is 11%, no lock; at +4.5R
    # progress is 56% and the R trigger already fired.  Progress
    # matters when TP is CLOSE: TP at 2015 (1.2R) — at +0.7R the
    # price is 58% to TP, so the progress leg fires before 1R.
    mgr = TickManager(tick=TickConfig(partial_tiers=(),
                                      profit_lock_trigger_r=99.0))
    req = _request(targets=(2015.0,))
    mgr.open(ticket=101, request=req, fill_price=2000.0, entry_bar=6)
    # +0.7R, 58% to TP -> progress leg fires (R trigger disabled).
    a = mgr.on_tick(101, 2008.75)
    assert [x.reason for x in a] == ["profit_lock"]


def test_profit_lock_never_loosens_stop():
    mgr = TickManager()
    mgr.adopt(ticket=404, symbol="XAUUSD", direction=1, volume=1.0,
              entry_price=2000.0, sl=2010.0)  # SL already at +0.8R
    # Lock level 2006.25 is looser than the 2010 stop — no fire.
    assert not [x for x in mgr.on_tick(404, 2012.5)
                if x.reason == "profit_lock"]


# ---------------------------------------------------------- chandelier
def test_chandelier_trails_best_minus_3atr():
    # Chandelier only (no breakeven/profit-lock/R-trail) — isolate it.
    mgr = TickManager(
        exits=ExitConfig(trail_trigger_r=999.0),
        tick=TickConfig(
            partial_tiers=(), profit_lock_enabled=False,
            breakeven_trigger_r=999.0,
            chandelier_enabled=True, chandelier_atr_mult=3.0,
            chandelier_activation_r=1.5))
    _open(mgr, targets=())
    # Below activation: nothing.
    assert mgr.on_tick(101, 2012.5) == []
    # +2.0R: best 2025 -> chandelier 2025 - 30 = 1995, tighter than
    # stop 1987.5 -> fires.
    a = mgr.on_tick(101, 2025.0)
    assert [x.reason for x in a] == ["chandelier"]
    assert a[0].level == pytest.approx(1995.0)
    mgr.apply(101, a[0])
    # Price falls back: chandelier never loosens.
    assert mgr.on_tick(101, 2010.0) == []
    # New best 2050 -> chandelier 2020, ratchets up.
    a = mgr.on_tick(101, 2050.0)
    assert [x.reason for x in a] == ["chandelier"]
    assert a[0].level == pytest.approx(2020.0)


def test_chandelier_disabled_without_atr():
    mgr = TickManager(
        exits=ExitConfig(trail_trigger_r=999.0),
        tick=TickConfig(
            partial_tiers=(), profit_lock_enabled=False,
            breakeven_trigger_r=999.0, chandelier_enabled=True))
    mgr.adopt(ticket=303, symbol="XAUUSD", direction=1, volume=1.0,
              entry_price=2000.0, sl=1987.5)
    # ATR unknown (0.0) -> chandelier never fires.
    assert not [x for x in mgr.on_tick(303, 2100.0)
                if x.reason == "chandelier"]


def test_tightest_candidate_wins():
    # At +2.0R: profit-lock 2007.5, R-trail 2012.5, chandelier
    # 2025-30=1995.  Tightest is the R-trail.
    mgr = TickManager(tick=TickConfig(partial_tiers=()))
    _open(mgr)
    a = mgr.on_tick(101, 2025.0)
    sl = [x for x in a if x.kind == "modify_sl"]
    assert len(sl) == 1
    assert sl[0].reason == "trail"
    assert sl[0].level == pytest.approx(2012.5)


def test_exit_config_untouched_by_tick_features():
    # The bar stepper (backtest/paper) still uses the single pool
    # partial — tick features do not leak into ExitConfig.
    assert ExitConfig().partial_frac == 0.5
    assert not hasattr(ExitConfig(), "partial_tiers")


# ------------------------------------------------------------- panic
def _wide_stop_open(mgr):
    """Long, entry 2000, invalidation 1940, ATR 10 -> buffered stop
    1937.5, risk 62.5.  Panic needs adverse > 3*10*1.5 = 45, i.e.
    price < 1955 — inside the stop, so panic (not the stop) fires."""
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                       stop_loss=1940.0, entry_ref=2000.0,
                       targets=[2100.0], atr=10.0, bar_index=5,
                       alert_id="T1")
    return mgr.open(ticket=101, request=req, fill_price=2000.0,
                    entry_bar=6)


def test_panic_fires_past_threshold_inside_stop():
    mgr = TickManager()
    _wide_stop_open(mgr)
    # Adverse 44 (severity 1.47): not panic.
    assert mgr.on_tick(101, 1956.0) == []
    # Adverse 46 (severity 1.53): panic — full close.
    a = mgr.on_tick(101, 1954.0)
    assert len(a) == 1
    assert a[0].kind == "panic" and a[0].reason == "panic"
    assert a[0].volume == pytest.approx(1.0)
    mgr.apply(101, a[0])
    assert mgr.position(101).closed is True


def test_panic_never_beats_a_crossed_stop():
    # Narrow stop (12.5): a gap past BOTH the stop and the panic
    # threshold is a stop hit, not a panic close.
    mgr = TickManager()
    _open(mgr)
    a = mgr.on_tick(101, 1900.0)
    assert [x.kind for x in a] == ["stop_hit"]


def test_panic_disabled_or_no_atr_never_fires():
    mgr = TickManager(tick=TickConfig(panic_enabled=False))
    _wide_stop_open(mgr)
    # Below the stop -> stop_hit (panic disabled changes nothing).
    assert [x.kind for x in mgr.on_tick(101, 1900.0)] == ["stop_hit"]
    mgr2 = TickManager(tick=TickConfig(panic_enabled=False))
    _wide_stop_open(mgr2)
    # Inside the stop, past the panic threshold -> nothing (panic
    # disabled).
    assert mgr2.on_tick(101, 1954.0) == []
    # Adopted: ATR unknown -> no panic (stop still works).
    mgr3 = TickManager()
    mgr3.adopt(ticket=303, symbol="XAUUSD", direction=1, volume=1.0,
               entry_price=2000.0, sl=1900.0)
    assert not [x for x in mgr3.on_tick(303, 1954.0)
                if x.kind == "panic"]


def test_panic_precedes_partial_and_layer_stop():
    mgr = TickManager()
    _wide_stop_open(mgr)
    # At panic price the tier-1 partial would also trigger — panic
    # wins, whole position closes, no partial banked.
    a = mgr.on_tick(101, 1954.0)
    assert [x.kind for x in a] == ["panic"]


# ------------------------------------------- scalp cap + regime time exit
def _scalp_open(mgr, opened_at):
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2100.0], atr=10.0, bar_index=5,
                       alert_id="T1", is_scalp=True)
    return mgr.open(ticket=101, request=req, fill_price=2000.0,
                    entry_bar=6, now=opened_at)


def test_scalp_cap_fires_at_cap_not_before():
    mgr = TickManager()
    _scalp_open(mgr, opened_at=1000.0)
    # 14:59 into the 15-min cap: nothing.
    assert mgr.on_tick(101, 2005.0, now=1000.0 + 14 * 60 + 59) == []
    # Exactly 15:00: full close.
    a = mgr.on_tick(101, 2005.0, now=1000.0 + 15 * 60)
    assert len(a) == 1
    assert a[0].kind == "scalp_cap" and a[0].reason == "scalp_cap"
    assert a[0].volume == pytest.approx(1.0)
    mgr.apply(101, a[0])
    assert mgr.position(101).closed is True


def test_scalp_cap_ignored_for_non_scalp_and_adopted():
    mgr = TickManager()
    _open(mgr)  # is_scalp defaults False
    assert not [x for x in mgr.on_tick(101, 2005.0, now=1e9)
                if x.kind == "scalp_cap"]
    mgr2 = TickManager()
    mgr2.adopt(ticket=303, symbol="XAUUSD", direction=1, volume=1.0,
               entry_price=2000.0, sl=1987.5)
    assert not [x for x in mgr2.on_tick(303, 2005.0, now=1e9)
                if x.kind == "scalp_cap"]


def test_panic_outranks_scalp_cap():
    mgr = TickManager()
    # Wide stop so the stop does not fire first; scalp + past cap.
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                       stop_loss=1940.0, entry_ref=2000.0,
                       targets=[2100.0], atr=10.0, bar_index=5,
                       alert_id="T1", is_scalp=True)
    mgr.open(ticket=101, request=req, fill_price=2000.0, entry_bar=6,
             now=1000.0)
    # Wide adverse move past the panic threshold AND past the cap:
    # panic (emergency) wins over the time exit.
    a = mgr.on_tick(101, 1954.0, now=1000.0 + 60 * 60)
    assert [x.kind for x in a] == ["panic"]


def test_regime_time_exit_uses_map_for_structure():
    mgr = TickManager(tick=TickConfig(time_exit_bars={"range": 8}))
    req = _request()
    mgr.open(ticket=101, request=req, fill_price=2000.0, entry_bar=6)
    # signal_bar = 5 (request.bar_index).  Range horizon is 8.
    assert mgr.on_bar_close(101, 5 + 7, "range") == []
    a = mgr.on_bar_close(101, 5 + 8, "range")
    assert [x.kind for x in a] == ["timeout"]
    # Unknown structure falls back to the fixed 72-bar horizon.
    mgr2 = TickManager(tick=TickConfig(time_exit_bars={"range": 8}))
    mgr2.open(ticket=101, request=req, fill_price=2000.0, entry_bar=6)
    assert mgr2.on_bar_close(101, 5 + 8, "clean_trend") == []
    assert mgr2.on_bar_close(101, 5 + 72, "clean_trend")[0].kind \
        == "timeout"


def test_regime_time_exit_none_map_uses_fixed_horizon():
    mgr = TickManager()  # time_exit_bars None
    req = _request()
    mgr.open(ticket=101, request=req, fill_price=2000.0, entry_bar=6)
    assert mgr.on_bar_close(101, 5 + 8, "range") == []
    assert mgr.on_bar_close(101, 5 + 72, "range")[0].kind == "timeout"


def test_default_time_exit_bars_cover_all_regime_structures():
    """The runner default map covers exactly the structures
    classify_regime produces, with engine-derived values (NOT the
    old bot's unverified 8-20)."""
    from goldenswing.execution.tick_manager import DEFAULT_TIME_EXIT_BARS
    assert set(DEFAULT_TIME_EXIT_BARS) == {
        "clean_trend", "range", "compression", "breakout",
        "reversal", "choppy", "unclear"}
    assert DEFAULT_TIME_EXIT_BARS["clean_trend"] == 72  # swing horizon
    assert DEFAULT_TIME_EXIT_BARS["unclear"] == 72  # = fixed horizon
    assert DEFAULT_TIME_EXIT_BARS["choppy"] == 12  # shortest leash
    assert all(isinstance(v, int) and v > 0
               for v in DEFAULT_TIME_EXIT_BARS.values())


def test_runner_default_time_exit_bars_is_the_map(tmp_path):
    """MT5RunnerConfig defaults to the map (not None); env JSON still
    replaces it wholesale; bad env JSON fails closed to the map."""
    import os

    from goldenswing.execution.mt5_runner import (
        MT5RunnerConfig, _time_exit_bars_from_env)
    from goldenswing.execution.tick_manager import DEFAULT_TIME_EXIT_BARS

    assert MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl").time_exit_bars == \
        DEFAULT_TIME_EXIT_BARS
    os.environ.pop("GS_TIME_EXIT_BARS", None)
    assert _time_exit_bars_from_env() == DEFAULT_TIME_EXIT_BARS
    os.environ["GS_TIME_EXIT_BARS"] = '{"range": 10}'
    try:
        assert _time_exit_bars_from_env() == {"range": 10}
        os.environ["GS_TIME_EXIT_BARS"] = "not json"
        assert _time_exit_bars_from_env() == DEFAULT_TIME_EXIT_BARS
    finally:
        os.environ.pop("GS_TIME_EXIT_BARS", None)


def test_no_live_playbook_qualifies_as_scalp():
    """Scalp audit: every live playbook is H1-evaluated with
    structural targets and multi-hour horizons — none is a scalp, so
    the runner default stays empty.  qm_m5 is M5-frame but
    shadow-only (never opens a position)."""
    from goldenswing.execution.mt5_runner import MT5RunnerConfig
    from goldenswing.playbooks.fvg_limit import FVGLimitPlaybook
    from goldenswing.playbooks.fvg_midline import FVGMidlinePlaybook
    from goldenswing.playbooks.inversion import InversionPlaybook
    from goldenswing.playbooks.judas_swing import JudasSwingPlaybook
    from goldenswing.playbooks.killzone_open import KillzoneOpenPlaybook
    from goldenswing.playbooks.sweep_reversal import SweepReversalPlaybook
    from goldenswing.playbooks.trendline_retest import (
        TrendlineRetestPlaybook)

    assert MT5RunnerConfig().scalp_playbooks == ()
    for cls in (SweepReversalPlaybook, FVGMidlinePlaybook,
                TrendlineRetestPlaybook, FVGLimitPlaybook,
                InversionPlaybook, KillzoneOpenPlaybook,
                JudasSwingPlaybook):
        assert cls.live is True
