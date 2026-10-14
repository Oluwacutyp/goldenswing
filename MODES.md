# GoldenSwing — Modes + Phase Routing (Phase 5)

**Method:** one market read (MarketReader + FeatureEngine), one detector (Phase-2 flags + Combo A {relstr, volume, phases}), three execution modes differing only in timeframe, entry filters, exit style and horizon.  Costs are NOT rescaled per mode: $0.45/oz round trip everywhere, so the cost-as-R at scalp timeframes is measured, not assumed.  Every mode run is independent (own data, own reader pass, own detector).

**Reproduction anchors:** swing book @ ≥60 must equal VALIDATION.md combo A (+25.62% / PF 1.22 / 199 trades); the routing-OFF book is the same run by construction and must match it too.

## Mode: swing (1h)

**Data:** GC=F 1h, 11,426 bars, 2024-10-11 04:00:00+00:00 → 2026-10-09 20:00:00+00:00 (1.995 years). RS peers at the same interval: dollar available, silver available.

Filters: killzone_only=False, allowed_kinds=all, min_displacement=0.0, exit=swing_v2, horizon=72 bars, cost=$0.45/oz.

Signals: 467 scanned → 467 after mode filters → 467 simulated (0 skipped by the simulator).

| Book @ ≥60 | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R | Avg hold (bars) |
|---|---|---|---|---|---|---|---|---|
| real cost ($0.45/oz) | 199 | 45.2 | 1.22 | +25.62% | 11.04% | 0.91 | +0.124 | 19.5 |
| zero cost (signal quality only) | 199 | 45.7 | 1.27 | +31.28% | 10.72% | 1.07 | +0.146 | 19.5 |

Shadow (all simulated signals, standalone): n=467, win 42.4%, avg R +0.056, PF 1.10. Shadow @≥60: n=352, avg R +0.124.

## Phase→playbook routing (swing mode)

Matrix measured from the routing-OFF shadow book (467 simulated signals, GC=F H1, 2024-10-11 04:00:00+00:00 → 2026-10-09 20:00:00+00:00). Playbooks: **sweep_reversal** (detector kind), **breakout** (continuation riding a fresh BOS, age ≤ 5), **trend_continuation** (any other continuation). Muting rule: avg R < 0 AND n ≥ 30 → the playbook is capped at 59 in that phase.

| Phase | sweep_reversal n / win% / avg R | trend_continuation n / win% / avg R | breakout n / win% / avg R |
|---|---|---|---|
| accumulation | — (n=0) | 1 / 0.0 / -1.028 | 2 / 50.0 / +0.626 |
| distribution | 5 / 40.0 / -0.245 | 21 / 61.9 / +0.589 | 6 / 33.3 / +0.087 |
| consolidation | 2 / 0.0 / -1.013 | 5 / 40.0 / +0.046 | 1 / 100.0 / +0.893 |
| manipulation | 2 / 100.0 / +0.403 | 14 / 42.9 / +0.088 | 13 / 53.8 / +0.142 |
| expansion | 5 / 80.0 / +0.644 | 36 / 38.9 / +0.033 | 55 / 49.1 / +0.410 |
| trend | 34 / 47.1 / +0.045 | 144 / 33.3 / -0.158 **MUTED** | 114 / 45.6 / +0.091 |
| unknown | 1 / 0.0 / -1.024 | 5 / 20.0 / -0.517 | 1 / 0.0 / -1.039 |

Signals emitted: OFF 467 simulated, ON 467 simulated (routing caps scores; it does not change the emitted set).

| Book | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R | Avg hold |
|---|---|---|---|---|---|---|---|---|
| Routing OFF @ ≥60 | 199 | 45.2 | 1.22 | +25.62% | 11.04% | 0.91 | +0.124 | 19.5 |
| Routing ON @ ≥60 | 149 | 45.6 | 1.20 | +16.16% | 10.87% | 0.71 | +0.109 | 20.2 |
| Routing OFF @ ≥50 | 226 | 42.9 | 1.12 | +14.00% | 14.42% | 0.52 | +0.067 | 19.0 |
| Routing ON @ ≥50 | 226 | 42.9 | 1.12 | +14.00% | 14.42% | 0.52 | +0.067 | 19.0 |

Shadow @≥60 avg R: OFF +0.124 (n=352) → ON +0.180 (n=251) — the ON shadow grade mechanically excludes the muted cells from the ≥60 region, so it is not independent evidence; the book comparison is the test.

## Mode: scalp (15m)

**Data:** GC=F 15m, 4,488 bars, 2026-08-02 22:00:00+00:00 → 2026-10-09 20:45:00+00:00 (0.186 years). RS peers at the same interval: dollar available, silver available.

Filters: killzone_only=True, allowed_kinds=all, min_displacement=0.0, exit=nearest_pool, horizon=8 bars, cost=$0.45/oz.

Signals: 230 scanned → 78 after mode filters → 58 simulated (20 skipped by the simulator).

| Book @ ≥60 | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R | Avg hold (bars) |
|---|---|---|---|---|---|---|---|---|
| real cost ($0.45/oz) | 45 | 62.2 | 1.57 | +8.88% | 4.86% | 2.96 | +0.194 | 5.2 |
| zero cost (signal quality only) | 45 | 62.2 | 1.70 | +10.57% | 4.65% | 3.47 | +0.229 | 5.2 |

Shadow (all simulated signals, standalone): n=58, win 62.1%, avg R +0.196, PF 1.59. Shadow @≥60: n=51, avg R +0.182.

**Cost in R** (per simulated trade, cost / risk at the buffered stop): mean 0.034R, median 0.031R, p90 0.054R. For scale, the same $0.45 costs 0.022R on the swing mode's H1 stops (VALIDATION.md).
Book exit mix: stop 31.1%, target 35.6%, timeout 33.3%.

## Mode: fast_scalp (5m)

**Data:** GC=F 5m, 13,434 bars, 2026-08-02 22:10:00+00:00 → 2026-10-09 20:55:00+00:00 (0.186 years). RS peers at the same interval: dollar available, silver available.

Filters: killzone_only=True, allowed_kinds=['continuation'], min_displacement=1.0, exit=nearest_pool, horizon=6 bars, cost=$0.45/oz.

Signals: 700 scanned → 124 after mode filters → 98 simulated (26 skipped by the simulator).

| Book @ ≥60 | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R | Avg hold (bars) |
|---|---|---|---|---|---|---|---|---|
| real cost ($0.45/oz) | 83 | 37.3 | 0.68 | -10.35% | 13.82% | -3.19 | -0.128 | 4.7 |
| zero cost (signal quality only) | 83 | 43.4 | 0.80 | -6.28% | 10.85% | -1.85 | -0.075 | 4.7 |

Shadow (all simulated signals, standalone): n=98, win 38.8%, avg R -0.111, PF 0.72. Shadow @≥60: n=91, avg R -0.115.

**Cost in R** (per simulated trade, cost / risk at the buffered stop): mean 0.053R, median 0.050R, p90 0.083R. For scale, the same $0.45 costs 0.022R on the swing mode's H1 stops (VALIDATION.md).
Book exit mix: stop 27.7%, target 20.5%, timeout 51.8%.

## Verdicts

**Swing mode — the validated core, unchanged.** The Phase-5 swing
run reproduces VALIDATION.md combo A exactly (199 trades, PF 1.22,
+25.62%, MaxDD 11.04%), which also proves the mode framework and the
detector's routing hook changed nothing when routing is off.

**The matrix is real; the muting rule, as designed, does not help
the book.** Playbook performance genuinely depends on the phase —
that part of the owner's principle is measured fact, not opinion:
breakout is the best playbook in expansion (+0.410R, n=55) and
positive in trend (+0.091R, n=114); trend-continuation is the best
playbook in distribution (+0.589R, n=21) yet the single worst
populated cell in trend phase (−0.158R, n=144). But exactly ONE
cell met the muting rule (trend × trend-continuation, n=144,
−0.158R), and muting it made the traded book WORSE: +25.62% →
+16.16%, PF 1.22 → 1.20, Sharpe 0.91 → 0.71, on 50 fewer trades
(MaxDD a hair better, 11.04% → 10.87%). The diagnosis is path
dependence: a standalone shadow grade is not a book contribution.
In the sequential one-position book, the muted signals' realised
contribution was evidently not the −0.158R their standalone grade
promised — and removing them re-times the position lock, changing
which later signals get taken at all. (At ≥50 the books are
identical, as they must be: a 59 cap only binds at the ≥60 gate.)
**Routing verdict: the mechanism works exactly as specified and is
REFUTED as a book improvement on this window. It stays OFF.** The
matrix remains valuable as diagnostics — it correctly identifies
where standalone signal quality is poor — and the honest next test
is a walk-forward table (derived on one window, applied to the
next) plus book-aware cell evaluation, neither of which was run
here. Two structural limits are also visible in the matrix itself:
467 signals over 21 cells leaves most cells at n < 10 (only the
trend row is well populated — the phase machine spends most signal
bars in trend/expansion), so for now the matrix can only speak
confidently about the trend phase.

**Scalp (M15) — promising, unproven, and NOT killed by costs.**
45 book trades over ~68 days: +8.88% at real cost, PF 1.57, win
62.2%, avg R +0.194, MaxDD 4.86%. The expected cost story did not
materialise: because the stop is the buffered structural
invalidation (not a fixed small M15 stop), mean cost is 0.034R per
trade — the zero-cost book (+10.57%, PF 1.70) is only modestly
better. The honest limits: 45 trades is a small sample, the window
is 68 days inside the same bullish regime as everything else, and
the Sharpe (2.96) is annualised over 0.186 years and is NOT
comparable to the swing Sharpe. Verdict: the only sub-hourly mode
with a pulse — worth a longer M15 history (which yfinance will not
provide) before any trust is placed in it. 20 of 78 killzone signals
were skipped for having no mapped pool ahead; that skip rule is
part of the result, not a footnote.

**Fast scalp (M5) — structurally unviable as configured, and costs
are not the main cause.** The book loses at real cost (−10.35%,
PF 0.68, avg R −0.128) AND at zero cost (−6.28%, PF 0.80, avg R
−0.075): the gross signal is already negative before friction.
Costs take their measured share (mean 0.053R/trade — higher than
M15, as expected, but survivable for a real edge), they are not
what kills it. What kills it: at M5 the shared read produces
momentum-continuation signals whose nearest-pool target is reached
only 20.5% of the time inside the 6-bar hold — 51.8% of trades
drift to the time stop. The timeframe rescaling (AMD bases of 4h,
48-bar ranges of 4h, H1-calibrated swing structure on 5-minute
bars) produces a read that does not fit the clock. Verdict: DEAD
in this form — do not tune the cost line to resurrect it; a real
M5 mode would need its own parametrisation, which is a different
experiment from "same read, faster clock".

## Caveats

- The routing matrix is computed from the SAME window it routes on: the ON/OFF comparison tests the routing mechanism in-sample, it is not out-of-sample proof.  A walk-forward split (derive the table on window A, route on window B) is the honest next test and has NOT been run.
- Scalp samples are ~60 days (yfinance intraday cap) against the swing mode's 2 years, inside the same bullish gold regime.  Scalp verdicts are weaker claims and are labelled as such.
- Bar-count parameters of the shared reader/features rescale with the timeframe (48-bar AMD base = 48h on H1, 12h on M15, 4h on M5).  That is what 'same read, faster clock' means here; per-timeframe reparametrisation was NOT done and would be a different experiment.
- The MTF feature family's HTF mapping assumes hourly bars (it keys HTF states to bar start + 1h).  MTF is computed but NOT scored in any mode (not in Combo A); on sub-hourly frames its values can be up to one bar early and must not be enabled there before that mapping is fixed.
- GC=F 5m/15m volume is real futures volume; yfinance intraday bars can contain partial/odd-lot prints the live feed would differ on.  No slippage beyond the flat cost.
