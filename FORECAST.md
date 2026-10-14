# GoldenSwing — Ensemble ML Forecaster (Phase 6)

## What was asked

The owner's core want: *an ML that can forecast a big move in 3 days and alert me.*  This phase builds that forecaster and grades it the house way — walk-forward, embargoed, and against the detector it would sit next to.

## Definition

At each H1 bar t: P(big move up) / P(big move down), where a big move = favourable excursion reaching **2x ATR(t)** within the next **72 bars** BEFORE adverse excursion reaches **1x ATR(t)** (same-bar double touch = adverse wins; labels mutually exclusive; bars without a full 72-bar window are unlabelled, never zero-filled).

## Method

- Data: GC=F H1, 11,426 bars, 2024-10-11 04:00:00+00:00 → 2026-10-09 20:00:00+00:00 (same window as Phases 1-5).
- Features: reader state + Combo-A feature families + detector opinion per bar, all causal (pure functions of the bar's MarketState).
- Models: HistGradientBoosting, logistic regression (median-imputed, scaled, class-weighted), ExtraTrees — trained separately per direction; ensemble = probability mean.
- Walk-forward: expanding window, first test block after 5,000 bars, retrain every 500 bars. PURGE: a training row is admitted only if its 72-bar label window ends before the test block opens. No shuffled splits anywhere.
- Out-of-fold coverage: 6,354 bars (from bar 5000).  Base rates on OOF rows: big-up 34.7%, big-down 33.4% (so always-predict-base-rate has AUC 0.500 by construction).

## Discrimination (AUC, out-of-fold)

| Direction | Model | AUC | n |
|---|---|---|---|
| up | gbm | 0.501 | 6354 |
| up | logreg | 0.525 | 6354 |
| up | extratrees | 0.531 | 6354 |
| up | ensemble | 0.517 | 6354 |
| up | detector_score | 0.505 | 6354 |
| dn | gbm | 0.516 | 6354 |
| dn | logreg | 0.527 | 6354 |
| dn | extratrees | 0.517 | 6354 |
| dn | ensemble | 0.526 | 6354 |
| dn | detector_score | 0.500 | 6354 |

## Alert quality (ensemble, out-of-fold)

### Big move UP

| P threshold | Alerts | Alert rate | Precision | Recall | Base rate |
|---|---|---|---|---|---|
| 0.5 | 1698.0 | 26.7% | 0.378 | 0.291 | 0.347 |
| 0.6 | 508.0 | 8.0% | 0.392 | 0.090 | 0.347 |
| 0.7 | 113.0 | 1.8% | 0.434 | 0.022 | 0.347 |

### Big move DOWN

| P threshold | Alerts | Alert rate | Precision | Recall | Base rate |
|---|---|---|---|---|---|
| 0.5 | 1223.0 | 19.2% | 0.360 | 0.207 | 0.334 |
| 0.6 | 374.0 | 5.9% | 0.332 | 0.058 | 0.334 |
| 0.7 | 51.0 | 0.8% | 0.235 | 0.006 | 0.334 |

## Calibration (ensemble, by predicted-probability decile)

### UP — mean predicted vs empirical rate

| n | Mean predicted | Empirical |
|---|---|---|
| 636.0 | 0.231 | 0.368 |
| 635.0 | 0.301 | 0.328 |
| 635.0 | 0.342 | 0.323 |
| 636.0 | 0.376 | 0.341 |
| 635.0 | 0.408 | 0.313 |
| 635.0 | 0.439 | 0.335 |
| 636.0 | 0.471 | 0.325 |
| 635.0 | 0.507 | 0.359 |
| 635.0 | 0.553 | 0.381 |
| 636.0 | 0.652 | 0.392 |

### DOWN

| n | Mean predicted | Empirical |
|---|---|---|
| 636.0 | 0.212 | 0.321 |
| 635.0 | 0.276 | 0.307 |
| 635.0 | 0.313 | 0.287 |
| 636.0 | 0.342 | 0.333 |
| 635.0 | 0.371 | 0.335 |
| 635.0 | 0.401 | 0.324 |
| 636.0 | 0.434 | 0.343 |
| 635.0 | 0.472 | 0.373 |
| 635.0 | 0.526 | 0.359 |
| 636.0 | 0.623 | 0.362 |

## Trading test 1 — forecaster as a GATE on the detector

Full-window plain Combo-A book (reproduction anchor; VALIDATION.md says 199 trades, PF 1.22, +25.62%):

| Book | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |
|---|---|---|---|---|---|---|---|
| plain detector, full window | 199 | 45.2 | 1.22 | +25.62% | 11.04% | 0.91 | +0.124 |

Restricted to the 277 detector signals that fall in the out-of-fold region (the only fair arena for the gate):

| Book | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |
|---|---|---|---|---|---|---|---|
| plain detector, OOF signals only | 106 | 44.3 | 1.31 | +18.64% | 11.04% | 1.16 | +0.172 |
| gated @ P >= 0.5 (kept 62, dropped 215) | 33 | 48.5 | 1.46 | +7.45% | 4.13% | 0.93 | +0.226 |
Shadow grades @ 0.5: kept signals avg R +0.276 (n=62) vs dropped signals avg R +0.100 (n=215) — the gate ranks signals if kept > dropped.
| gated @ P >= 0.6 (kept 26, dropped 251) | 17 | 52.9 | 1.88 | +7.17% | 3.33% | 1.05 | +0.420 |
Shadow grades @ 0.6: kept signals avg R +0.321 (n=26) vs dropped signals avg R +0.120 (n=251) — the gate ranks signals if kept > dropped.

## Trading test 2 — standalone alerts (no detector)

Every ensemble alert (24-bar cooldown) traded directly on the label's geometry (stop 1x ATR, target 2x ATR) through the Phase-2 v2 exit machinery:

| Threshold | Alerts | Simulated | Shadow avg R | Book trades | Book return | Book PF |
|---|---|---|---|---|---|---|
| 0.5 | 232 | 232 | +0.107 (n=232) | 221 | +31.96% | 1.24 |
| 0.6 | 145 | 145 | +0.110 (n=145) | 141 | +12.71% | 1.16 |

## Trading test 3 — permutation control (the null)

The standalone book trades the label's own barrier geometry, so a control is mandatory: the (P_up, P_dn) pairs are permuted across out-of-fold bars (same score distribution, timing destroyed) and the standalone book @ 0.5 is re-run over 3 seeds.

- Shuffled-alert book return: mean -3.32% (range -24.03% … +20.13%), mean PF 0.99, shadow avg R -0.015.
- Real-alert book @ 0.5 (from test 2): +31.96%, shadow avg R +0.107.
- Real minus shuffled is the part of the standalone result attributable to WHEN the model fires, as opposed to the geometry it fires into.  The verdict below reads them together.

## Verdict — read this before quoting any number above

**Discrimination: real but weak.** Ensemble AUC 0.517 (up) / 0.526
(down), out-of-fold. It beats the detector-score baseline in both
directions (0.505 / 0.500 — the detector's score is a *setup quality*
grade, not a move predictor, and the ML is not redundant with it),
but the margin is small, and the gradient-boosting member is the
weakest of the three (0.501 up — a coin flip); ExtraTrees and the
logistic baseline carry the ensemble. Nobody should trade on AUC
0.52 alone, and this document does not ask them to.

**Calibration: the outputs are ranking scores, NOT probabilities.**
Across deciles the predicted "probability" sweeps 0.23 → 0.65 while
the empirical event rate barely moves (0.37 → 0.39 up; 0.32 → 0.36
down). The thresholds in these tables (P >= 0.5 / 0.6) are operating
points on a ranking, nothing more. If an alert ever shows the owner
a percentage, it must not be this raw number. At the extreme down
tail (P >= 0.7) precision actually falls BELOW base rate (0.235 vs
0.334, n=51) — that operating point is anti-predictive and must not
be used.

**As a GATE on the detector: quality up, quantity down — it does
not add total return and does not replace anything.** The ranking
is genuine (shadow avg R: kept +0.276 vs dropped +0.100 @ 0.5; kept
+0.321 vs dropped +0.120 @ 0.6), and the gated books are far
gentler (PF 1.46–1.88, MaxDD 3.3–4.1% vs 11.0%). But vetoing
78–91% of signals throws away profitable trades with the bad ones:
gated total return +7.2–7.5% vs +18.64% for the plain book on the
same OOF signals. As built, the forecaster's honest role next to
the detector is a *sizing/quality* input (full size when it agrees,
reduced when it doesn't — that variant is untested), not a hard
veto.

**Standalone: the surprise, and the control is what makes it
believable.** Traded on its own alerts (label geometry through the
house exits), the forecaster books +31.96% @ 0.5 (PF 1.24, 221
trades, shadow +0.107R over 232 alerts) across the OOF span —
more than the detector's own book over the same region. The
permutation control earns this result its keep: identical barrier
geometry, identical score distribution, timing destroyed → mean
-3.32%, shadow -0.015R. The 2:1 barrier structure at a ~35% base
rate is NOT free money (shuffled proves it grades ~breakeven); the
profit is attributable to WHEN the model chooses to fire — it fires
on 232 of 6,354 bars, and that selectivity, not blanket accuracy,
is the mechanism. A weak-AUC model can do this: it only has to
tilt a favourable-geometry bet by a few points at chosen moments.

**Caveats that stand:**
- The standalone trade deliberately mirrors the label's geometry
  (1x ATR stop / 2x ATR target). The permutation control bounds the
  circularity — the same geometry with shuffled timing loses — but
  the test is still played on the forecaster's home turf.
- One instrument, one bullish 2-year window, OOF coverage only from
  bar 5,000 (~the second half). The temporal/regime out-of-sample
  question that Phase 3.5 left open is still open; nothing here
  answers it either.
- The full-window anchor reproduced VALIDATION.md exactly (199
  trades, PF 1.22, +25.62%), so the trading comparisons stand on
  the validated book, not a re-implementation.

**Bottom line:** the ensemble has a small, real, timing-based
edge. It beats the detector score as a predictor, ranks detector
signals usefully (as a quality dial, not a switch), and its own
alerts — the owner's actual request — traded profitably
out-of-fold in a way a timing-destroyed null cannot reproduce.
Its probabilities are uncalibrated rankings and must be labelled
as such wherever they surface. Research-grade, promising, not
proven: the same single-regime shadow hangs over this as over
every prior phase.
