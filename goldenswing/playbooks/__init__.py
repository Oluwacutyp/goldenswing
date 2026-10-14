"""GoldenSwing playbooks (Phase 9).

One AWARE system, multiple entry playbooks.  Each playbook owns its
setup detection, trigger, stop/target/exit policy and base quality
score — no playbook reaches into another's machinery.  The confluence
engine (goldenswing.confluence) collects candidates from all playbooks,
ranks them by confluence, and journals every decision like a trader's
log.

The iron rule: a playbook keeps the EXACT entry/exit machinery it was
validated with.  Mixing a trigger from one playbook with exits from
another is a new experiment, not a refactor.
"""
from .base import Candidate, EngineContext, Playbook
from .fvg_limit import FVGLimitPlaybook
from .fvg_midline import FVGMidlinePlaybook
from .qm_m5 import QMM5Playbook
from .sweep_reversal import SweepReversalPlaybook
from .trendline_retest import TrendlineRetestPlaybook

__all__ = [
    "Candidate",
    "EngineContext",
    "Playbook",
    "SweepReversalPlaybook",
    "FVGMidlinePlaybook",
    "FVGLimitPlaybook",
    "TrendlineRetestPlaybook",
    "QMM5Playbook",
]
