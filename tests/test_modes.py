"""Phase-5 modes: preset shapes, the shared-detector contract, entry
filters, and the scalp simulator's mechanics (target/stop/timeout,
stop-wins-ties, no-pool skip, cost-in-R reporting).
"""
import pytest
from conftest import make_df

from goldenswing.backtest.validation import COMBO_A
from goldenswing.golden.detector import GoldenSignal
from goldenswing.modes.config import FAST_SCALP, SCALP, SWING
from goldenswing.modes.runner import signal_passes_mode
from goldenswing.modes.simulate import simulate_scalp
from goldenswing.reader.state import MarketState


# ---------------------------------------------------------------- presets
def test_mode_presets():
    assert SWING.interval == "1h" and SWING.horizon == 72
    assert SWING.exit_style == "swing_v2" and not SWING.killzone_only
    assert SCALP.interval == "15m" and SCALP.horizon == 8
    assert SCALP.killzone_only and SCALP.exit_style == "nearest_pool"
    assert FAST_SCALP.interval == "5m" and FAST_SCALP.horizon == 6
    assert FAST_SCALP.allowed_kinds == frozenset({"continuation"})
    assert FAST_SCALP.min_displacement == 1.0 and FAST_SCALP.killzone_only
    # costs are deliberately NOT rescaled per mode
    assert SWING.cost_per_oz == SCALP.cost_per_oz \
        == FAST_SCALP.cost_per_oz == 0.45


def test_modes_share_one_detector():
    for mode in (SWING, SCALP, FAST_SCALP):
        cfg = mode.detector_config()
        assert cfg.feature_families == COMBO_A
        assert cfg.require_sweep_confirmation and cfg.daily_veto
        assert cfg.golden_threshold == 60.0
        assert cfg.route_table is None     # reproduction default


# ---------------------------------------------------------------- filters
def _state(killzone=True, bull_disp=1.3, bear_disp=0.0) -> MarketState:
    return MarketState(index=10, timestamp="t", close=100.0, atr=1.0,
                       trend="up", bias=1, killzone=killzone,
                       bull_disp_3=bull_disp, bear_disp_3=bear_disp)


def _sig(kind="continuation", direction=1) -> GoldenSignal:
    return GoldenSignal(index=10, timestamp="t", direction=direction,
                        score=70.0, kind=kind)


def test_killzone_filter():
    assert signal_passes_mode(SCALP, _state(killzone=True), _sig())
    assert not signal_passes_mode(SCALP, _state(killzone=False), _sig())
    assert signal_passes_mode(SWING, _state(killzone=False), _sig())


def test_fast_scalp_kind_and_displacement_filters():
    assert signal_passes_mode(FAST_SCALP, _state(), _sig("continuation"))
    assert not signal_passes_mode(FAST_SCALP, _state(), _sig("sweep_reversal"))
    weak = _state(bull_disp=0.5)
    assert not signal_passes_mode(FAST_SCALP, weak, _sig("continuation"))


def test_fast_scalp_short_uses_bear_displacement():
    st = _state(bull_disp=0.0, bear_disp=1.4)
    assert signal_passes_mode(FAST_SCALP, st, _sig("continuation", -1))
    st2 = _state(bull_disp=1.4, bear_disp=0.2)
    assert not signal_passes_mode(FAST_SCALP, st2, _sig("continuation", -1))


# -------------------------------------------------------------- simulator
def _scalp_sig(df_len=8, **kw) -> GoldenSignal:
    base = dict(index=2, timestamp="t", direction=1, score=70.0,
                kind="continuation", invalidation=99.0, atr=1.0,
                targets=[102.0])
    base.update(kw)
    return GoldenSignal(**base)


def test_scalp_target_hit():
    rows = [(100, 100.5, 99.5, 100)] * 3 \
        + [(100, 100.4, 99.6, 100.2), (100.2, 103.0, 100.0, 102.5)] \
        + [(102.5, 103.0, 102.0, 102.5)] * 3
    df = make_df(rows)
    o = simulate_scalp(df, _scalp_sig(), horizon=8)
    assert o is not None and o.exit_reason == "target"
    assert o.exit_index == 4
    # entry 100, stop 98.75 (0.25 buffer), risk 1.25, target 102
    assert o.target_r == pytest.approx(1.6)
    assert o.cost_r == pytest.approx(0.45 / 1.25)
    assert o.r_net == pytest.approx(1.6 - 0.36)


def test_scalp_stop_wins_same_bar_tie():
    rows = [(100, 100.5, 99.5, 100)] * 3 \
        + [(100, 100.4, 99.6, 100.2), (100.2, 103.0, 98.0, 99.0)] \
        + [(99.0, 99.5, 98.5, 99.0)] * 3
    df = make_df(rows)
    o = simulate_scalp(df, _scalp_sig(), horizon=8)
    assert o.exit_reason == "stop"
    assert o.r_net == pytest.approx(-1.0 - 0.36)


def test_scalp_no_pool_ahead_is_skipped():
    df = make_df([(100, 100.5, 99.5, 100)] * 8)
    assert simulate_scalp(df, _scalp_sig(targets=[])) is None
    assert simulate_scalp(df, _scalp_sig(targets=[95.0])) is None


def test_scalp_timeout_at_horizon():
    rows = [(100, 100.5, 99.5, 100)] * 3 \
        + [(100, 100.4, 99.6, 100.2)] * 5
    df = make_df(rows)
    o = simulate_scalp(df, _scalp_sig(), horizon=2)
    assert o.exit_reason == "timeout" and o.exit_index == 4
    assert o.hold_bars == 2


def test_scalp_short_side():
    rows = [(100, 100.5, 99.5, 100)] * 3 \
        + [(100, 100.4, 99.6, 99.8), (99.8, 100.0, 97.0, 97.5)] \
        + [(97.5, 98.0, 97.0, 97.5)] * 3
    df = make_df(rows)
    sig = _scalp_sig(direction=-1, invalidation=101.0, targets=[98.0])
    o = simulate_scalp(df, sig, horizon=8)
    assert o.exit_reason == "target"
    # entry 100, stop 101.25, risk 1.25, target 98 -> 1.6R gross
    assert o.r_net == pytest.approx(1.6 - 0.36)
