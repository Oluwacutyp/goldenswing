"""QMD1Playbook — Quasimodo reversal on D1 (SHADOW, see validation).

The SAME QM structure as ``playbooks/qm_m5.py`` (head, neckline, QM
level, break of structure, retracement, resting LIMIT AT POI), run on
daily bars through the ONE shared engine (:class:`QMEngine
<goldenswing.playbooks.qm_m5._QMEngine>` via
:class:`QMPlaybookBase <goldenswing.playbooks.qm_m5.QMPlaybookBase>`).
This file holds ONLY the D1 parameter row and the D1 playbook
metadata — no copied engine, no copied simulator, no copied gate.

Why D1 exists: on 2026-10-14 the owner spotted a Daily QM at
~4,133–4,140 that held as resistance and dropped to ~4,104 (~$36)
with ZERO engine signals and ZERO blocked.  Root cause was coverage,
not gates: the engine had no Daily QM playbook (``qm_m5`` is M5-only
+ shadow), so nothing got detected or blocked.  The fix is a Daily QM
playbook — this file — wired as SHADOW (evaluated + logged, never
ordered) until validation + a separate owner decision promote it.

D1 PARAMETERS — derivation (calendar-structural equivalents, NOT
blind rescalings of the M5 bar counts):

- The ATR-relative thresholds are scale-free, so they are UNCHANGED
  from M5: both legs >= 0.5 ATR, failed rally short of the head by
  >= 0.25 ATR, stop = head +/- 0.25 ATR, neckline target >= 1.0R,
  stop band 0.25–3.5 ATR, displacement break body >= 1.0x ATR,
  14-bar ATR, fractal swing_n = 3 (a D1 pivot confirms after 3
  daily bars — the same fractal definition as M5, with a 3-day
  confirmation lag instead of 15 minutes).
- The BAR-COUNT expiries ARE timeframe-specific, derived from what
  the M5 counts mean in market time:
  - M5 ``MAX_PATTERN_BARS = 120`` ~= 10h ~= ONE trading session:
    the formation must complete within one session, or slower
    formations mix sessions.  D1 equivalent of "one regime":
    a daily QM is a multi-week swing structure; the formation must
    complete within ONE QUARTER ~= 60 trading days.
    ``MAX_PATTERN_BARS = 60``.
  - M5 ``MAX_RETRACE_BARS = 48`` ~= 4h ~= HALF a session: the
    retracement must arrive promptly after the break.  D1
    equivalent: within ONE MONTH ~= 20 trading days.
    ``MAX_RETRACE_BARS = 20``.
  - M5 ``MAX_HOLD_BARS = 48`` ~= 4h (the fvg_limit precedent):
    the trade must resolve promptly after the touch.  D1
    equivalent: within ONE MONTH ~= 20 trading days.
    ``MAX_HOLD_BARS = 20``.

  These three counts are judgment calls from market structure, not
  measured optima — the validation script reports the measured
  numbers, and any re-tuning must come from those numbers, not from
  re-scaling the M5 counts again.

Entry clock: D1 (see ``reader/session_timeframes.py`` — ``qm_d1``
  keeps its own D1 clock the way ``qm_m5`` keeps M5).  A daily bar
  spans every session, so the playbook is session-agnostic by
  construction; the validation script breaks results down by
  direction, tags, drift alignment, and ATR regime instead of by
  session.

STATUS: LIVE on the demo (owner directive 2026-10-22).  Measured by
``scripts/validate_qm_d1.py`` — see its output and the numbers in
:attr:`QMD1Playbook.context_description`.  The measurement is the
historical MAP (positive R, far too few trades to mean anything);
under the owner's doctrine it never gated execution, and the
promotion decision the old text waited for is the 2026-10-22
directive itself.  Forward live-demo performance is the verdict.
"""
from __future__ import annotations

from ..contextmap import HuntingGround
from .qm_m5 import M5_PARAMS, QMParams, QMPlaybookBase

#: D1 parameter row: ATR-relative thresholds identical to M5
#: (scale-free); bar-count expiries are the D1 calendar-structural
#: equivalents derived in the module docstring (60 / 20 / 20).
D1_PARAMS = QMParams(
    swing_n=M5_PARAMS.swing_n,
    atr_n=M5_PARAMS.atr_n,
    min_leg_atr=M5_PARAMS.min_leg_atr,
    min_fail_atr=M5_PARAMS.min_fail_atr,
    stop_buffer_atr=M5_PARAMS.stop_buffer_atr,
    min_target_r=M5_PARAMS.min_target_r,
    max_pattern_bars=60,   # one quarter: formation must complete
    max_retrace_bars=20,   # one month: retracement must arrive
    max_hold_bars=20,      # one month: trade must resolve
    displacement_body_atr=M5_PARAMS.displacement_body_atr,
    min_stop_atr=M5_PARAMS.min_stop_atr,
    max_stop_atr=M5_PARAMS.max_stop_atr,
    base_score=M5_PARAMS.base_score,
    kind="qm_d1",
    name="qm_d1",
)


class QMD1Playbook(QMPlaybookBase):
    params = D1_PARAMS
    name = "qm_d1"
    description = (
        "Quasimodo reversal on D1: head -> neckline pullback -> failed "
        "rally (QM level) -> close through the neckline -> limit at the "
        "QM level, stop beyond the head, target the broken neckline"
    )
    live = True  # LIVE on the demo (owner directive 2026-10-22):
    # measured by scripts/validate_qm_d1.py (positive R, tiny sample —
    # the map, in context_description); the demo decides.
    live_threshold = 0.0  # every qualifying touch is graded
    context_description = (
        "Daily Quasimodo reversal, MEASURED by "
        "scripts/validate_qm_d1.py (OANDA XAUUSD H1 resampled to D1): "
        "PRIMARY 2y (2024-10 -> 2026-10): 4 trades, 50% win, +1.5R, "
        "PF 1.73 — under the old bar on sample size (< 30 trades). "
        "EXTENDED ~4.8y (2022-01 -> 2026-10): 8 trades, 50% win, "
        "+1.4R, PF 1.34 — also under the bar on sample. Positive R but far "
        "too few trades to mean anything: Daily QM on gold is RARE "
        "(~2 qualifying touches/year) under these strict params. "
        "Breakdowns at this n are noise, not findings — the only "
        "split that held across BOTH windows (extended contains "
        "primary, so this is one sample, not two): displacement_break "
        "touches went 0-for-4 (-4.0R) while non-displacement went "
        "4-for-4 (+5.4R), a WATCH hypothesis, not a gate. "
        "Diagnostic (NOT validation): loosening the 1R floor + "
        "windows fires 36 trades / +8.9R — a hypothesis for a future "
        "OOS experiment, never a post-hoc re-tune. LIVE on the demo "
        "since 2026-10-22 (owner directive) — the demo decides. "
        "Mechanics: confirmed fractal pivots (swing_n=3), both legs "
        ">= 0.5 ATR, failed rally short of the head by >= 0.25 ATR, "
        "close through the neckline within 60 bars of the head (one "
        "quarter), first retracement touch of the QM level within 20 "
        "bars of the break (one month), neckline target >= 1.0R, "
        "20-bar max hold. Session-agnostic by construction (a daily "
        "bar spans every session)."
    )
    #: Session-agnostic by construction — a daily bar spans sessions.
    session_scope = "any"
    #: No measured context map yet — fires everywhere, no map bonuses.
    hunting_ground = HuntingGround(excluded=set(), bonuses={})
