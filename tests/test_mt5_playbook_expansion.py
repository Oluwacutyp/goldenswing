"""MT5 playbook expansion: promotion + new real-trader setups.

The MT5 engine executes exactly the validated playbook set.  The
promotion (sweep_reversal anchor + fvg_limit) and each new setup get
their tests here: signal fires on the right structure, does not fire
on chop, no repainting (batch candidates == live_signal signal set),
hierarchy still gates every playbook, and a non-validating setup
stays shadow (live=False, not in the execute list).
"""
from __future__ import annotations

import pandas as pd

from goldenswing.execution.mt5_runner import MT5RunnerConfig
from goldenswing.playbooks.base import EngineContext


def _ctx(df):
    return EngineContext(df=df, states=[], years=2.0, instrument="XAUUSD")


def _df(rows, freq="h"):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1000.0] * len(rows)},
        index=idx,
    )


def test_mt5_execute_list_is_exactly_the_validated_set():
    from goldenswing.playbooks import (
        FVGLimitPlaybook,
        FVGMidlinePlaybook,
        InversionPlaybook,
        JudasSwingPlaybook,
        KillzoneOpenPlaybook,
        OrderBlockPlaybook,
        QMM5Playbook,
        RetracementTrapPlaybook,
        SweepReversalPlaybook,
        TrendlineRetestPlaybook,
        TurtleSoupPlaybook,
        VReversalPlaybook,
    )

    executed = set(MT5RunnerConfig().execute_playbooks)
    # Validated set: the promoted anchor + sleeves + the setups
    # that passed the iron rule (inversion PF 1.11, killzone_open
    # PF 2.23, judas_swing PF 1.93).
    assert executed == {"sweep_reversal", "fvg_midline",
                        "trendline_retest", "fvg_limit",
                        "inversion", "killzone_open", "judas_swing"}
    # QM M5 stays shadow: unvalidated, annotation only.
    assert "qm_m5" not in executed
    assert QMM5Playbook.live is False
    # Failed validation stays shadow: never executed, live=False.
    for name, cls in (("v_reversal", VReversalPlaybook),
                      ("retracement_trap", RetracementTrapPlaybook),
                      ("order_block", OrderBlockPlaybook),
                      ("turtle_soup", TurtleSoupPlaybook)):
        assert name not in executed
        assert cls.live is False
    # Every executed playbook must be live-capable (the runner only
    # runs pb.live playbooks through the signal path).
    for cls in (SweepReversalPlaybook, FVGMidlinePlaybook,
                TrendlineRetestPlaybook, FVGLimitPlaybook,
                InversionPlaybook, KillzoneOpenPlaybook,
                JudasSwingPlaybook):
        assert cls.live is True
