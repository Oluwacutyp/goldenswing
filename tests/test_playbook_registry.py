"""Pin the executing-vs-shadow registry and standing rules.

- Executing (14): the full roster — sweep_reversal, fvg_midline,
  trendline_retest, fvg_limit, inversion, killzone_open,
  judas_swing, turtle_soup, v_reversal, retracement_trap,
  order_block, qm_m5, qm_d1 (Daily QM), narrative — live=True.
- Shadow: NONE.  Owner directive 2026-10-22: the whole shadow
  roster was promoted to live for forward testing on the demo —
  backtests map where a setup historically worked, they are never
  a binary kill/adopt gate; forward live-demo performance is the
  verdict.  This test fails if any playbook silently drops back
  to shadow (live=False or a non-empty shadow list).
- Hard veto stays OFF globally (HARD_VETO_ENABLED is False).
- AMD missed-manipulation fallback still exists.
- Session/timeframe mapping still exists (killzones, M15/M30 entry).
"""
from goldenswing.execution.mt5_config import (
    EXECUTE_PLAYBOOK_NAMES, SHADOW_PLAYBOOK_NAMES)
from goldenswing.live.hierarchy import HARD_VETO_ENABLED
from goldenswing.playbooks import (
    FVGLimitPlaybook, FVGMidlinePlaybook, InversionPlaybook,
    JudasSwingPlaybook, KillzoneOpenPlaybook, NarrativePlaybook,
    OrderBlockPlaybook, QMD1Playbook, QMM5Playbook,
    RetracementTrapPlaybook, SweepReversalPlaybook,
    TrendlineRetestPlaybook, TurtleSoupPlaybook, VReversalPlaybook)
from goldenswing.features.phases import _fallback_manipulation
from goldenswing.reader import session_timeframes


def test_full_roster_executes_and_is_live():
    assert tuple(EXECUTE_PLAYBOOK_NAMES) == (
        "sweep_reversal", "fvg_midline", "trendline_retest", "fvg_limit",
        "inversion", "killzone_open", "judas_swing",
        "turtle_soup", "v_reversal", "retracement_trap", "order_block",
        "qm_m5", "qm_d1", "narrative")
    for cls in (SweepReversalPlaybook, FVGMidlinePlaybook,
                TrendlineRetestPlaybook, FVGLimitPlaybook,
                InversionPlaybook, KillzoneOpenPlaybook,
                JudasSwingPlaybook, TurtleSoupPlaybook, VReversalPlaybook,
                RetracementTrapPlaybook, OrderBlockPlaybook,
                QMM5Playbook, QMD1Playbook, NarrativePlaybook):
        assert cls.live is True, cls.name


def test_no_shadow_tier_remains():
    """The 2026-10-22 directive emptied the shadow tier: nothing is
    evaluated-but-never-ordered.  A playbook leaves execution via
    the per-playbook kill switch, never via this list."""
    assert tuple(SHADOW_PLAYBOOK_NAMES) == ()
    assert QMM5Playbook.live is True
    assert QMD1Playbook.live is True  # Daily QM: promoted by the directive


def test_hard_veto_stays_off():
    assert HARD_VETO_ENABLED is False


def test_amd_fallback_exists():
    assert callable(_fallback_manipulation)


def test_session_timeframe_mapping_exists():
    assert callable(session_timeframes.mapping_for)
    assert callable(session_timeframes.compose)
