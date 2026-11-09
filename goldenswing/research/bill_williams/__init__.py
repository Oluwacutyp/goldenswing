"""Bill Williams research family (MarketSituation
Pass 8, SPEC §6.6) — a separately tested STRATEGY, not
six universal indicators.

The posture is the spec: this package is isolated.  It
lives outside ``playbooks/`` and outside the live
registry; nothing in the situation, the census, or any
existing playbook reads its outputs; it declares no
judgment route, proposes no scenario to the book, and
has NO path to order authority inside this program.
Its deliverables are its own journal stream and its own
promotion evaluation, run as a research line — a
``promotion_ready`` verdict here is a recommendation
to the owner, and any future promotion is a separate
owner decision taken on that evaluation, exactly like
a range route's.

The strategy, as one rule set (Trading Chaos): trend
state from the Alligator (sleeping / awakening /
eating / satiated), entry = a confirmed fractal break
beyond the teeth in the Alligator's direction with
AO/AC confirmation, invalidation = opposite fractal /
lips cross.  Gator and BW-MFI ride as measured organs:
the ablation harness shows which organs earn their
place, and an organ that adds nothing is removed, not
kept for completeness.

Every read is causal: a Williams fractal is known only
once its right-side bars have closed, the Alligator
lines are read at Williams' forward shifts (which, read
causally, means the smoothed values from ``shift`` bars
ago), and every state at bar ``j`` is a function of
bars ``<= j`` — judged by what was observable at the
time, never hindsight.

BW-MFI provenance: MT5's volume column on this feed is
tick volume, so every MFI reading carries
``volume_basis = "tick_volume"`` and the caveat with
it, on every record — never silent.
"""
from . import evaluation, harness, journal
from .indicators import PreparedWilliams, annotate
from .strategy import VARIANTS, run_strategy, scan_events

__all__ = [
    "VARIANTS",
    "PreparedWilliams",
    "annotate",
    "evaluation",
    "harness",
    "journal",
    "run_strategy",
    "scan_events",
]
