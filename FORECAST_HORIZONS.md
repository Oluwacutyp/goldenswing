# GoldenSwing — Multi-Horizon Forecaster (Phase 6.5)

## What changed

The owner's correction: *"Not limited to 3 days though."*  Phase 6 hard-coded the 72-bar (3-day) label.  This phase generalises the forecaster to every registered horizon and grades each one the house way — purged walk-forward, standalone trading test, permutation null — on the same data window as Phases 1-6 (GC=F H1, 11,426 bars, 2024-10-11 04:00:00+00:00 → 2026-10-09 20:00:00+00:00).

## Horizons and their geometry

A big move at bar t = favourable excursion reaching fav x ATR(t) BEFORE adverse excursion reaches adv x ATR(t) inside the window (same-bar double touch = adverse wins; labels mutually exclusive; incomplete windows unlabelled).  Barriers widen with the window (price wanders further given more time); the fav:adv ratio stays the house 2:1.  Standalone trades use the matching geometry and a time stop equal to the horizon.

| Horizon | Tag | Window | Fav | Adv | Alert cooldown |
|---|---|---|---|---|---|
| half-day | 12h | 12 bars | 1.50 ATR | 0.75 ATR | 24 bars |
| 1 day | 1d | 24 bars | 2.00 ATR | 1.00 ATR | 24 bars |
| 3 days | 3d | 72 bars | 2.00 ATR | 1.00 ATR | 24 bars |
| 1 week | 1w | 168 bars | 3.00 ATR | 1.50 ATR | 56 bars |
| 2 weeks | 2w | 360 bars | 4.00 ATR | 2.00 ATR | 120 bars |

## Method notes

- Walk-forward identical to Phase 6 (expanding window, first test block after 5,000 bars, retrain every 500), except the PURGE scales with the horizon: a training row is admitted only if its whole label window ends before the test block opens.
- EFFECTIVE SAMPLES: labels at neighbouring bars share outcome windows, so raw row counts lie — worst at long horizons.  n_eff = labelled rows / horizon (non-overlapping windows) is the honest denominator and is printed per horizon; verdicts shrink to fit it.
- Standalone test + permutation control per horizon, same as Phase 6: the control permutes (P_up, P_dn) pairs across bars — same distribution, timing destroyed — and re-books the alerts.  Real-minus-shuffled is the part attributable to WHEN the model fires.

## Comparison — all horizons, out-of-fold

| Horizon | Base up | Base dn | AUC up | AUC dn | OOF bars | n_eff (OOF) | Standalone @0.5: return (PF, trades) | Null: return | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 12h (12) | 0.314 | 0.317 | 0.522 | 0.511 | 6,414 | 534 | +7.77% (1.08, 213) | -5.89% | PARTIAL |
| 1d (24) | 0.341 | 0.327 | 0.523 | 0.520 | 6,402 | 266 | +34.35% (1.25, 229) | +11.98% | WEAK BUT REAL |
| 3d (72) | 0.347 | 0.334 | 0.517 | 0.526 | 6,354 | 88 | +31.96% (1.24, 221) | -3.32% | WEAK BUT REAL |
| 1w (168) | 0.351 | 0.332 | 0.497 | 0.509 | 6,258 | 37 | -3.62% (0.96, 104) | +6.35% | NOISE |
| 2w (360) | 0.367 | 0.325 | 0.511 | 0.493 | 6,066 | 16 | +5.72% (1.24, 49) | -6.02% | NOISE |

## Per-horizon detail

### half-day — 12 bars (12h), fav 1.50 / adv 0.75 ATR

Labelled bars: 11,414 (n_eff ~951); OOF bars: 6,414 (n_eff ~534); folds: 13; base rates up 0.314 / dn 0.317.

| Direction | Model | AUC | n |
|---|---|---|---|
| up | gbm | 0.506 | 6414 |
| up | logreg | 0.533 | 6414 |
| up | extratrees | 0.535 | 6414 |
| up | ensemble | 0.522 | 6414 |
| up | detector_score | 0.508 | 6414 |
| dn | gbm | 0.503 | 6414 |
| dn | logreg | 0.523 | 6414 |
| dn | extratrees | 0.508 | 6414 |
| dn | ensemble | 0.511 | 6414 |
| dn | detector_score | 0.500 | 6414 |

Alert quality (ensemble):

| Direction | P thr | Alerts | Precision | Recall | Base |
|---|---|---|---|---|---|
| up | 0.5 | 1164 | 0.359 | 0.208 | 0.314 |
| up | 0.6 | 286 | 0.434 | 0.062 | 0.314 |
| up | 0.7 | 50 | 0.560 | 0.014 | 0.314 |
| dn | 0.5 | 1044 | 0.346 | 0.177 | 0.317 |
| dn | 0.6 | 210 | 0.362 | 0.037 | 0.317 |
| dn | 0.7 | 16 | 0.062 | 0.000 | 0.317 |

Calibration (deciles): up — bottom-decile empirical 0.305, top-decile 0.374, mean |pred-emp| 0.109; down — bottom 0.298, top 0.360, MAE 0.095. As in Phase 6 these are ranking scores, not probabilities — read the decile SPREAD, not the level.

Standalone alerts (label geometry, horizon time stop):

| P thr | Alerts | Shadow avg R | Book return | Book PF |
|---|---|---|---|---|
| 0.5 | 214 | +0.043 (n=213) | +7.77% | 1.08 |
| 0.6 | 99 | +0.108 (n=99) | +10.36% | 1.20 |
Permutation null @ 0.5: mean book return -5.89% (range -16.93% … +15.10%), mean shadow avg R -0.021.

**Verdict: PARTIAL — only one direction carries anything, or the alerts do not survive the null; not a standalone input.**

### 1 day — 24 bars (1d), fav 2.00 / adv 1.00 ATR

Labelled bars: 11,402 (n_eff ~475); OOF bars: 6,402 (n_eff ~266); folds: 13; base rates up 0.341 / dn 0.327.

| Direction | Model | AUC | n |
|---|---|---|---|
| up | gbm | 0.507 | 6402 |
| up | logreg | 0.530 | 6402 |
| up | extratrees | 0.532 | 6402 |
| up | ensemble | 0.523 | 6402 |
| up | detector_score | 0.505 | 6402 |
| dn | gbm | 0.504 | 6402 |
| dn | logreg | 0.529 | 6402 |
| dn | extratrees | 0.517 | 6402 |
| dn | ensemble | 0.520 | 6402 |
| dn | detector_score | 0.499 | 6402 |

Alert quality (ensemble):

| Direction | P thr | Alerts | Precision | Recall | Base |
|---|---|---|---|---|---|
| up | 0.5 | 1535 | 0.379 | 0.267 | 0.341 |
| up | 0.6 | 462 | 0.403 | 0.085 | 0.341 |
| up | 0.7 | 114 | 0.491 | 0.026 | 0.341 |
| dn | 0.5 | 1161 | 0.355 | 0.197 | 0.327 |
| dn | 0.6 | 363 | 0.317 | 0.055 | 0.327 |
| dn | 0.7 | 55 | 0.200 | 0.005 | 0.327 |

Calibration (deciles): up — bottom-decile empirical 0.356, top-decile 0.393, mean |pred-emp| 0.107; down — bottom 0.304, top 0.339, MAE 0.098. As in Phase 6 these are ranking scores, not probabilities — read the decile SPREAD, not the level.

Standalone alerts (label geometry, horizon time stop):

| P thr | Alerts | Shadow avg R | Book return | Book PF |
|---|---|---|---|---|
| 0.5 | 229 | +0.138 (n=229) | +34.35% | 1.25 |
| 0.6 | 124 | +0.084 (n=123) | +9.25% | 1.13 |
Permutation null @ 0.5: mean book return +11.98% (range -4.91% … +22.86%), mean shadow avg R +0.052.

**Verdict: WEAK BUT REAL — discrimination is a few points over a coin flip, yet the alerts still beat the null; a ranking/sizing input, not an oracle.**

### 3 days — 72 bars (3d), fav 2.00 / adv 1.00 ATR

Labelled bars: 11,354 (n_eff ~157); OOF bars: 6,354 (n_eff ~88); folds: 13; base rates up 0.347 / dn 0.334.

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

Alert quality (ensemble):

| Direction | P thr | Alerts | Precision | Recall | Base |
|---|---|---|---|---|---|
| up | 0.5 | 1698 | 0.378 | 0.291 | 0.347 |
| up | 0.6 | 508 | 0.392 | 0.090 | 0.347 |
| up | 0.7 | 113 | 0.434 | 0.022 | 0.347 |
| dn | 0.5 | 1223 | 0.360 | 0.207 | 0.334 |
| dn | 0.6 | 374 | 0.332 | 0.058 | 0.334 |
| dn | 0.7 | 51 | 0.235 | 0.006 | 0.334 |

Calibration (deciles): up — bottom-decile empirical 0.368, top-decile 0.392, mean |pred-emp| 0.114; down — bottom 0.321, top 0.362, MAE 0.091. As in Phase 6 these are ranking scores, not probabilities — read the decile SPREAD, not the level.

Standalone alerts (label geometry, horizon time stop):

| P thr | Alerts | Shadow avg R | Book return | Book PF |
|---|---|---|---|---|
| 0.5 | 232 | +0.107 (n=232) | +31.96% | 1.24 |
| 0.6 | 145 | +0.110 (n=145) | +12.71% | 1.16 |
Permutation null @ 0.5: mean book return -3.32% (range -24.03% … +20.13%), mean shadow avg R -0.015.

**Verdict: WEAK BUT REAL — discrimination is a few points over a coin flip, yet the alerts still beat the null; a ranking/sizing input, not an oracle.**

### 1 week — 168 bars (1w), fav 3.00 / adv 1.50 ATR

Labelled bars: 11,258 (n_eff ~67); OOF bars: 6,258 (n_eff ~37); folds: 13; base rates up 0.351 / dn 0.332.

| Direction | Model | AUC | n |
|---|---|---|---|
| up | gbm | 0.480 | 6258 |
| up | logreg | 0.510 | 6258 |
| up | extratrees | 0.520 | 6258 |
| up | ensemble | 0.497 | 6258 |
| up | detector_score | 0.509 | 6258 |
| dn | gbm | 0.513 | 6258 |
| dn | logreg | 0.511 | 6258 |
| dn | extratrees | 0.504 | 6258 |
| dn | ensemble | 0.509 | 6258 |
| dn | detector_score | 0.499 | 6258 |

Alert quality (ensemble):

| Direction | P thr | Alerts | Precision | Recall | Base |
|---|---|---|---|---|---|
| up | 0.5 | 1983 | 0.348 | 0.314 | 0.351 |
| up | 0.6 | 771 | 0.379 | 0.133 | 0.351 |
| up | 0.7 | 207 | 0.420 | 0.040 | 0.351 |
| dn | 0.5 | 1299 | 0.336 | 0.210 | 0.332 |
| dn | 0.6 | 470 | 0.287 | 0.065 | 0.332 |
| dn | 0.7 | 98 | 0.265 | 0.013 | 0.332 |

Calibration (deciles): up — bottom-decile empirical 0.398, top-decile 0.382, mean |pred-emp| 0.132; down — bottom 0.310, top 0.319, MAE 0.107. As in Phase 6 these are ranking scores, not probabilities — read the decile SPREAD, not the level.

Standalone alerts (label geometry, horizon time stop):

| P thr | Alerts | Shadow avg R | Book return | Book PF |
|---|---|---|---|---|
| 0.5 | 106 | -0.016 (n=106) | -3.62% | 0.96 |
| 0.6 | 86 | +0.161 (n=86) | +11.99% | 1.25 |
Permutation null @ 0.5: mean book return +6.35% (range -2.27% … +20.35%), mean shadow avg R +0.058.

**Verdict: NOISE — ensemble discrimination is at coin-flip level in both directions; this horizon's forecast is a lottery ticket and must not be quoted.**

### 2 weeks — 360 bars (2w), fav 4.00 / adv 2.00 ATR

Labelled bars: 11,066 (n_eff ~30); OOF bars: 6,066 (n_eff ~16); folds: 13; base rates up 0.367 / dn 0.325.

| Direction | Model | AUC | n |
|---|---|---|---|
| up | gbm | 0.505 | 6066 |
| up | logreg | 0.506 | 6066 |
| up | extratrees | 0.511 | 6066 |
| up | ensemble | 0.511 | 6066 |
| up | detector_score | 0.507 | 6066 |
| dn | gbm | 0.499 | 6066 |
| dn | logreg | 0.502 | 6066 |
| dn | extratrees | 0.501 | 6066 |
| dn | ensemble | 0.493 | 6066 |
| dn | detector_score | 0.501 | 6066 |

Alert quality (ensemble):

| Direction | P thr | Alerts | Precision | Recall | Base |
|---|---|---|---|---|---|
| up | 0.5 | 2157 | 0.399 | 0.387 | 0.367 |
| up | 0.6 | 1076 | 0.421 | 0.204 | 0.367 |
| up | 0.7 | 344 | 0.477 | 0.074 | 0.367 |
| dn | 0.5 | 1133 | 0.305 | 0.175 | 0.325 |
| dn | 0.6 | 445 | 0.297 | 0.067 | 0.325 |
| dn | 0.7 | 92 | 0.196 | 0.009 | 0.325 |

Calibration (deciles): up — bottom-decile empirical 0.387, top-decile 0.450, mean |pred-emp| 0.153; down — bottom 0.269, top 0.298, MAE 0.119. As in Phase 6 these are ranking scores, not probabilities — read the decile SPREAD, not the level.

Standalone alerts (label geometry, horizon time stop):

| P thr | Alerts | Shadow avg R | Book return | Book PF |
|---|---|---|---|---|
| 0.5 | 50 | +0.176 (n=50) | +5.72% | 1.24 |
| 0.6 | 46 | +0.110 (n=46) | +5.78% | 1.23 |
Permutation null @ 0.5: mean book return -6.02% (range -10.44% … -2.70%), mean shadow avg R -0.088.

**Verdict: NOISE — ensemble discrimination is at coin-flip level in both directions; this horizon's forecast is a lottery ticket and must not be quoted. SAMPLE WARNING: only ~16 effective (non-overlapping) out-of-fold windows — too thin to grade; treat the line above as provisional at best.**

## Multi-horizon agreement — 1d + 3d + 1w firing together

Detector (Combo A) signals inside the intersection of the three horizons' OOF regions: 277.  Three books on the SAME arena outcomes; a gate keeps a signal only when its own direction qualifies.

| Book | Kept | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |
|---|---|---|---|---|---|---|---|---|
| plain detector (arena) | 277 | 106 | 44.3 | 1.31 | +18.64% | 11.04% | 1.17 | +0.172 |
| gated @ 72h P>=0.5 | 62 | 33 | 48.5 | 1.46 | +7.45% | 4.13% | 0.94 | +0.226 |
| gated @ agreement 24+72+168 | 34 | 21 | 52.4 | 1.73 | +7.39% | 3.00% | 1.08 | +0.350 |

Shadow grades — agreement gate: kept avg R +0.567 (n=34) vs dropped +0.079 (n=243); 72h gate: kept +0.276 (n=62) vs dropped +0.100 (n=215).

Agreement alerts traded standalone (72-bar clock): 188 alerts, book return +0.10%, PF 1.02, shadow avg R +0.009 (n=188).

**Agreement verdict: AGREEMENT IS A QUALITY FILTER ONLY: kept signals grade higher than the 72h gate's, but the book books less — agreement throws away profitable trades along with the bad ones.  A sizing input, not a switch.**

## Phase-6 reproduction check (72-bar row vs FORECAST.md)

| Quantity | FORECAST.md | This run | Match |
|---|---|---|---|
| base_rate_up | +0.347 | +0.347 | YES |
| base_rate_dn | +0.334 | +0.334 | YES |
| ens_auc_up | +0.517 | +0.517 | YES |
| ens_auc_dn | +0.526 | +0.526 | YES |
| standalone_05_return | +31.960 | +31.955 | YES |
| standalone_05_pf | +1.240 | +1.241 | YES |
| standalone_05_shadow_r | +0.107 | +0.107 | YES |
| standalone_06_return | +12.710 | +12.706 | YES |
| control_return_mean | -3.320 | -3.323 | YES |

## Bottom line

- **12h (12 bars):** PARTIAL — only one direction carries anything, or the alerts do not survive the null; not a standalone input.
- **1d (24 bars):** WEAK BUT REAL — discrimination is a few points over a coin flip, yet the alerts still beat the null; a ranking/sizing input, not an oracle.
- **3d (72 bars):** WEAK BUT REAL — discrimination is a few points over a coin flip, yet the alerts still beat the null; a ranking/sizing input, not an oracle.
- **1w (168 bars):** NOISE — ensemble discrimination is at coin-flip level in both directions; this horizon's forecast is a lottery ticket and must not be quoted.
- **2w (360 bars):** NOISE — ensemble discrimination is at coin-flip level in both directions; this horizon's forecast is a lottery ticket and must not be quoted. SAMPLE WARNING: only ~16 effective (non-overlapping) out-of-fold windows — too thin to grade; treat the line above as provisional at best.

Caveats that stand for every horizon: one instrument, one bullish 2-year window, OOF coverage only from bar 5,000; the temporal/regime out-of-sample question is still open. Long-horizon rows rest on a handful of effective windows and their numbers will move the most when new data arrives.  Outputs remain ranking scores, NOT calibrated probabilities, at every horizon.
