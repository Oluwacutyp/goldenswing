"""Pin the executing-vs-shadow registry and standing rules.

- Executing (18): the full registry in canonical order — the 16
  H1-roster playbooks in signal-path order (sweep_reversal,
  fvg_midline, trendline_retest, fvg_limit, inversion,
  killzone_open, judas_swing, turtle_soup, v_reversal,
  retracement_trap, order_block, range_sweep_reclaim, range_qm,
  range_mean_reversion, range_compression_breakout, narrative),
  then the two native-clock playbooks (qm_m5, qm_d1) — live=True.
  FUNNEL §3 (2026-10-23): EXECUTE_PLAYBOOK_NAMES is DERIVED from
  the one playbook registry; the four range playbooks were
  missing from the hand-maintained list (14), which made them
  unkillable via Telegram and silently dropped their persisted
  kill entries — the registry derivation is the fix, and this
  test pins the full 18.
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
    RangeCompressionBreakoutPlaybook, RangeMeanReversionPlaybook,
    RangeQMPlaybook, RangeSweepReclaimPlaybook,
    RetracementTrapPlaybook, SweepReversalPlaybook,
    TrendlineRetestPlaybook, TurtleSoupPlaybook, VReversalPlaybook)
from goldenswing.features.phases import _fallback_manipulation
from goldenswing.reader import session_timeframes


def test_full_roster_executes_and_is_live():
    assert tuple(EXECUTE_PLAYBOOK_NAMES) == (
        "sweep_reversal", "fvg_midline", "trendline_retest", "fvg_limit",
        "inversion", "killzone_open", "judas_swing",
        "turtle_soup", "v_reversal", "retracement_trap", "order_block",
        "range_sweep_reclaim", "range_qm", "range_mean_reversion",
        "range_compression_breakout", "narrative", "qm_m5", "qm_d1")
    for cls in (SweepReversalPlaybook, FVGMidlinePlaybook,
                TrendlineRetestPlaybook, FVGLimitPlaybook,
                InversionPlaybook, KillzoneOpenPlaybook,
                JudasSwingPlaybook, TurtleSoupPlaybook, VReversalPlaybook,
                RetracementTrapPlaybook, OrderBlockPlaybook,
                RangeSweepReclaimPlaybook, RangeQMPlaybook,
                RangeMeanReversionPlaybook,
                RangeCompressionBreakoutPlaybook,
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
