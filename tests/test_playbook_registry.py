"""Pin the executing-vs-shadow registry and standing rules.

- Executing (7): sweep_reversal, fvg_midline, trendline_retest,
  fvg_limit, inversion, killzone_open, judas_swing — live=True.
- Shadow: qm_m5, qm_d1 (Daily QM), turtle_soup, v_reversal,
  retracement_trap, order_block — live=False.  Daily QM promotion
  needs a separate owner decision; this test fails if any shadow
  playbook is silently flipped to executing.
- Hard veto stays OFF globally (HARD_VETO_ENABLED is False).
- AMD missed-manipulation fallback still exists.
- Session/timeframe mapping still exists (killzones, M15/M30 entry).
"""
from goldenswing.execution.mt5_config import (
    EXECUTE_PLAYBOOK_NAMES, SHADOW_PLAYBOOK_NAMES)
from goldenswing.live.hierarchy import HARD_VETO_ENABLED
from goldenswing.playbooks import (
    FVGLimitPlaybook, FVGMidlinePlaybook, InversionPlaybook,
    JudasSwingPlaybook, KillzoneOpenPlaybook, QMD1Playbook,
    QMM5Playbook, SweepReversalPlaybook, TrendlineRetestPlaybook)
from goldenswing.features.phases import _fallback_manipulation
from goldenswing.reader import session_timeframes


def test_executing_seven_are_live():
    assert tuple(EXECUTE_PLAYBOOK_NAMES) == (
        "sweep_reversal", "fvg_midline", "trendline_retest", "fvg_limit",
        "inversion", "killzone_open", "judas_swing")
    for cls in (SweepReversalPlaybook, FVGMidlinePlaybook,
                TrendlineRetestPlaybook, FVGLimitPlaybook,
                InversionPlaybook, KillzoneOpenPlaybook,
                JudasSwingPlaybook):
        assert cls.live is True, cls.name


def test_shadow_qm_and_daily_qm_stay_shadow():
    assert "qm_m5" in SHADOW_PLAYBOOK_NAMES
    assert "qm_d1" in SHADOW_PLAYBOOK_NAMES
    assert QMM5Playbook.live is False
    assert QMD1Playbook.live is False  # Daily QM: shadow until separate auth


def test_hard_veto_stays_off():
    assert HARD_VETO_ENABLED is False


def test_amd_fallback_exists():
    assert callable(_fallback_manipulation)


def test_session_timeframe_mapping_exists():
    assert callable(session_timeframes.mapping_for)
    assert callable(session_timeframes.compose)
