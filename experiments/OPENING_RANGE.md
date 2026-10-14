# Experiment: QQQ opening-range breakout + retest (user sleeve)

**Source:** the user's words from a trading comment section — mark the
high/low of the first 5m candle (09:30–09:35 ET), drop to 1m, wait for
a break of either level, wait *patiently* for the retest, enter on it.
FVG (3-candle rule) as optional confirmation. QQQ daily trend as the
bias read. TP 1:2 or PDH/PDL. Plus the second comment's range-width
filter: skip narrow opening ranges.

## Data (the honest part)

yfinance server caps are hard: **M5 → ~60 days, M1 → ~7 days**. Pulled
with overlapping end-anchored chunks (M5: 4×15d, M1: 4×2d, D1: 4×180d),
deduped by timestamp. Chunking maximized reliability but could not
extend past the caps — Yahoo rejected the oldest M5 chunk outright
("must be within the last 60 days"). Deeper intraday history will come
from the MT5 terminal's own data once deployed.

| Grid | Bars | Range (UTC) | ET regular-session days |
|---|---|---|---|
| M5 | 2,418 | 2026-08-27 → 2026-10-09 | **31** |
| M1 | 1,949 | 2026-10-05 → 2026-10-09 | **5** |
| D1 | 618 | 2024-04-24 → 2026-10-09 | 618 (bias + PDH/PDL source) |

The M1 window (5 sessions) is too thin to say anything — reported for
completeness, verdicts lean on M5.

## Mechanics

Per regular-session day (America/New_York): OR = 09:30–09:35 bar
high/low (first five 1m bars on the M1 grid). Break = first CLOSE
through either level. Resting limit at the broken level, 12-bar expiry
(M5) / 30-bar (M1); fill on wick touch, adverse open on gaps, filled-
then-stopped on break failure (close back through the level). Stop =
0.5×OR width beyond the level (risk is 0.5×OR by construction).
FVG variant: 3-candle gap in the breakout direction within 5 bars of
the retest. Bias variant: prior close vs 20-day SMA (completed daily
bars). Gate variant: skip when OR width < 0.5× median of prior 20
days' widths (min 10 days' history, else allow). Targets: fixed 1:2, or
PDH/PDL (skipped when < 1R room). Walk: stop wins ties, $0.02/share
round trip (stated assumption), time stop at the session close. One
trade per day max, sequential book. All causal — OR/break/retest from
that day's bars, bias/PDH/PDL/gate from completed history only.

## Funnel (M5)

31 sessions → 31 setups (every day broke the 5-minute range — the OR
is a tight level) → 24 plain fills / 21 FVG-confirmed fills →
7 / 10 cancelled on expiry. Range gate bound on **1** day; bias
filter blocked **18** (mostly counter-trend setups in this window).

## Books (M5, one position at a time)

| Variant | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe |
|---|---|---|---|---|---|---|---|
| plain + 1:2 | 24 | 20.8 | 1.17 | +0.061 | +1.32% | 3.25% | 0.79 |
| plain + PDH/PDL | 23 | 13.0 | 1.19 | +0.088 | +1.67% | 3.89% | 0.66 |
| **bias + 1:2** | 10 | 30.0 | 1.88 | **+0.278** | **+2.75%** | 2.05% | 2.05 |
| **bias + PDH/PDL** | 9 | 22.2 | 2.44 | **+0.662** | **+5.82%** | 3.05% | 2.13 |
| bias + FVG + 1:2 | 9 | 22.2 | 1.26 | +0.092 | +0.77% | 3.09% | 0.68 |
| bias + FVG + PDH/PDL | 8 | 12.5 | 0.75 | −0.132 | −1.12% | 4.06% | −0.78 |
| FVG + 1:2 (no bias) | 21 | 14.3 | 0.80 | −0.069 | −1.54% | 4.31% | −0.95 |
| FVG + PDH/PDL (no bias) | 20 | 5.0 | 0.33 | −0.317 | −6.22% | 8.05% | −4.36 |

(Sharpe is annualised over ~0.12 years — reported for completeness,
not comparable to the swing book.)

## Findings

1. **The daily-bias filter is the strongest effect.** Bias-aligned
   trades grade +0.278R / +0.662R vs +0.061R / +0.088R unfiltered —
   the comment's "QQQ for true bias direction" earns its keep. But
   n = 9–10 in one bullish window: promising, unproven.
2. **The FVG confirmation hurts — uniformly.** Every FVG-on row is
   worse than its FVG-off twin, and FVG+PDH/PDL without bias is the
   worst book on the page (−0.317R). A post-breakout 3-candle gap is
   near-ubiquitous after displacement, so the "confirmation" filters
   almost nothing real and drops good trades. The comment's FVG
   confirm does not earn its place; rejected.
3. **The range-width gate is untested, not disproven.** It bound on 1
   of 31 days — narrow ranges barely occurred in this window, so the
   gate ON/OFF rows are near-identical. The comment's claim needs a
   longer sample (more compression days); MT5 intraday history will
   supply it.
4. **PDH/PDL beats fixed 1:2 when bias-aligned** (+0.662R vs +0.278R)
   — critical levels as targets carry further than the mechanical
   ratio on the filtered set. On the unfiltered set they're
   equivalent.
5. **M1: inconclusive.** 5 sessions, 2–5 trades per variant. The
   comment's "drop to 1m" granularity cannot be evaluated on Yahoo's
   7-day cap; needs terminal data.

## Verdict

**Promising but unproven — paper/OOS candidate, not a book.** The
bias-aligned PDH/PDL variant (+0.662R, PF 2.44) is the one to carry
forward, with the FVG confirmation explicitly dropped and the range
gate parked until more compression days exist. Nothing from this
experiment is wired into the gold detector — it is a separate market,
separate session, separate playbook. Next data milestone: rerun on
MT5 terminal intraday history (months, not weeks) before any live
consideration.

Tests: `tests/test_experiments_opening_range.py` (18: OR extraction,
break/retest mechanics, FVG rule, gates, bias, PDH/PDL, trade walk,
session splitting, causality). Full suite 276/276 green.
