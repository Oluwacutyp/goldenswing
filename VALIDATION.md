# GoldenSwing — Validation (Phase 3.5)

Combo test + out-of-sample validation of the Phase-3 feature verdicts. Same detector, same Phase-2 flags (`require_sweep_confirmation`, `daily_veto`), same v2 exits, same $0.45/oz cost, 1% risk — NO parameter changes between in-sample and out-of-sample runs.

## Recommended config (decided in-sample, rule fixed pre-run)

**Winner: combo A {relstr, volume, phases}** — higher book-@≥60 return wins; if within 1.0pp, lower MaxDD wins.

### Test 1 — IN-SAMPLE combo test (development window)

**GC=F hourly (gold futures)** — 11,426 bars, 2024-10-11 04:00:00+00:00 → 2026-10-09 20:00:00+00:00 (1.99 years). RS peers: silver SI=F + dollar DX-Y.NYB, same caches as Phase 3.

| Config | Signals | Book trades | Win% | PF | Return | MaxDD | Sharpe | Book avg R | Shadow @≥60 avg R | Book@50 return |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (features off) | 467 | 199 | 43.7 | 1.16 | +18.17% | 12.63% | 0.68 | +0.093 | +0.122 (n=337) | +16.71% |
| combo A {relstr, volume, phases} | 467 | 199 | 45.2 | 1.22 | +25.62% | 11.04% | 0.91 | +0.124 | +0.124 (n=352) | +14.00% |
| combo B {A + bollinger} | 467 | 198 | 44.4 | 1.19 | +20.99% | 11.87% | 0.77 | +0.105 | +0.091 (n=354) | +16.95% |

Score-band discrimination (shadow book — every signal standalone; backing must rank: higher band, better avg R):

- **baseline (features off):** 35–50: -0.497R (n=22) · 50–60: -0.035R (n=108) · 60–75: +0.199R (n=197) · 75–101: +0.012R (n=140)
- **combo A {relstr, volume, phases}:** 35–50: -0.158R (n=20) · 50–60: -0.148R (n=95) · 60–75: +0.145R (n=189) · 75–101: +0.099R (n=163)
- **combo B {A + bollinger}:** 35–50: -0.211R (n=22) · 50–60: -0.015R (n=91) · 60–75: +0.064R (n=186) · 75–101: +0.121R (n=168)

### Test 2 — OUT-OF-SAMPLE: silver (instrument change)

**SI=F hourly (silver futures)** — 11,424 bars, 2024-10-11 04:00:00+00:00 → 2026-10-09 20:00:00+00:00 (1.99 years). RS peers: silver leg is the instrument itself (ratio 1.0, flat by construction); the dollar leg carries RS. Same peer dict, same config — no rewiring.

| Config | Signals | Book trades | Win% | PF | Return | MaxDD | Sharpe | Book avg R | Shadow @≥60 avg R | Book@50 return |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (features off) | 483 | 197 | 35.0 | 0.29 | -81.35% | 81.35% | -5.21 | -0.836 | -0.761 (n=381) | -84.22% |
| combo A {relstr, volume, phases} | 482 | 195 | 34.9 | 0.28 | -81.69% | 81.69% | -5.38 | -0.854 | -0.782 (n=386) | -84.06% |
| combo B {A + bollinger} | 482 | 193 | 35.2 | 0.28 | -81.20% | 81.20% | -5.30 | -0.850 | -0.772 (n=386) | -84.06% |

Score-band discrimination (shadow book — every signal standalone; backing must rank: higher band, better avg R):

- **baseline (features off):** 35–50: -0.476R (n=19) · 50–60: -0.873R (n=83) · 60–75: -0.677R (n=229) · 75–101: -0.887R (n=152)
- **combo A {relstr, volume, phases}:** 35–50: -0.705R (n=19) · 50–60: -0.705R (n=77) · 60–75: -0.806R (n=205) · 75–101: -0.755R (n=181)
- **combo B {A + bollinger}:** 35–50: -0.797R (n=20) · 50–60: -0.734R (n=76) · 60–75: -0.793R (n=190) · 75–101: -0.752R (n=196)

### Test 3 — OUT-OF-SAMPLE: BTC (asset-class change)

**BTC-USD hourly (Yahoo spot)** — 17,488 bars, 2024-10-11 04:00:00+00:00 → 2026-10-11 04:00:00+00:00 (2.00 years). RS peers: silver + dollar (both exogenous to BTC). 24/7 series; session/killzone tags are UTC-hour based as in-sample.

| Config | Signals | Book trades | Win% | PF | Return | MaxDD | Sharpe | Book avg R | Shadow @≥60 avg R | Book@50 return |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (features off) | 916 | 371 | 41.2 | 1.21 | +50.30% | 14.44% | 1.11 | +0.121 | +0.084 (n=700) | +38.96% |
| combo A {relstr, volume, phases} | 916 | 380 | 41.8 | 1.25 | +65.78% | 15.40% | 1.33 | +0.144 | +0.114 (n=747) | +28.95% |
| combo B {A + bollinger} | 916 | 386 | 42.0 | 1.26 | +67.43% | 15.79% | 1.35 | +0.145 | +0.116 (n=760) | +29.11% |

Score-band discrimination (shadow book — every signal standalone; backing must rank: higher band, better avg R):

- **baseline (features off):** 35–50: -0.164R (n=47) · 50–60: +0.189R (n=169) · 60–75: +0.132R (n=391) · 75–101: +0.022R (n=309)
- **combo A {relstr, volume, phases}:** 35–50: -0.290R (n=38) · 50–60: +0.066R (n=131) · 60–75: +0.211R (n=362) · 75–101: +0.022R (n=385)
- **combo B {A + bollinger}:** 35–50: -0.255R (n=36) · 50–60: +0.028R (n=120) · 60–75: +0.177R (n=376) · 75–101: +0.057R (n=384)

## The earlier-window test (Oct 2022 → Oct 2024 gold) — NOT RUN

Yahoo (the repo's only wired hourly source) refuses 1h bars older than 730 days: `1h data not available … The requested range must be within the last 730 days` (verified 2026-10-11 against GC=F for Nov 2022). Dukascopy's public datafeed and Stooq intraday both reset/empty-reply from this environment after repeated probes. No local cache of 2022–2024 hourly gold exists on this machine. Rather than dress an in-sample split up as out-of-sample, this test is recorded as BLOCKED on data; it is the first run to execute when a 2022–2024 hourly gold feed becomes available.

## Cost diagnostics (post-run analysis — NOT additional validation runs)

The harness cost is a flat $0.45 per unit round trip, calibrated for
gold. In R terms (cost / risk-per-unit at entry) that is wildly
different per instrument, and it decides both OOS stories:

| Instrument | Mean cost in R | Median | p90 |
|---|---|---|---|
| Gold GC=F (baseline signals) | 0.022R | 0.017R | 0.041R |
| Silver SI=F (baseline signals) | **0.900R** | 0.655R | 1.877R |

- **Silver at zero cost:** the same baseline signals grade
  shadow @≥60 **+0.154R, PF 1.28, 45.4% win (n=381)** — the same
  quality as gold's in-sample shadow grade (+0.122R). The detector's
  signals DO transfer to silver; the gold-priced flat cost (0.90R per
  trade against sub-dollar ATR stops) is what produces −81%. A real
  silver spread (~$0.03–0.05/oz ⇒ ~0.06–0.10R) sits between the two
  runs and implies a thin-positive shadow grade (~+0.05R) — but no
  silver-calibrated-cost book was run, and none is claimed here.
- **BTC at realistic cost:** combo A with a $100/BTC round trip
  (~0.11% of the $88,950 mean price) collapses from +65.78% to
  **−1.82% (PF 1.25 → 1.01)**; at $200 it is −42.07%. The BTC book's
  avg R (+0.144) is thinner than ordinary crypto friction. The
  baseline's avg R (+0.121) is thinner still, so it cannot survive a
  cost combo A cannot.

## Verdict — PARTIAL survival, narrowly scoped

1. **Recommended config: combo A `{relstr, volume, phases}`.**
   In-sample it beats the baseline on every headline metric
   (+25.62% vs +18.17%, PF 1.22 vs 1.16, MaxDD 11.04% vs 12.63%,
   Sharpe 0.91 vs 0.68), and the baseline reproduced Phase 2/3
   exactly, so the comparison is clean. **Bollinger stays OUT:**
   stacked on A it dilutes (return −4.6pp, shadow @≥60 +0.124 →
   +0.091) — its solo MaxDD benefit does not survive stacking.
   Caveat: combo A's gain is concentrated at the traded gate —
   A @≥50 returns +14.00%, WORSE than baseline @≥50 (+16.71%). The
   combo improves the top of the book, not the whole book.
2. **The signal edge transfers; the tradable edge (so far) does
   not — except on gold.** At zero/nominal cost the detector grades
   positively on silver (+0.154R shadow) and BTC (+0.114R shadow for
   combo A). Under each instrument's realistic friction, both OOS
   books die: silver by the harness's gold-priced cost (0.90R/trade —
   a cost-model mismatch, recoverable in principle, unproven as a
   book), BTC by genuine thinness (PF 1.01 at 0.11% round trip — no
   cost recalibration saves a +0.144R edge from real spreads/fees).
   Test 3's headline +65.78% must not be quoted without this
   paragraph: it is a near-zero-cost number.
3. **Score discrimination partially survives.** The low bands are
   negative and the 60–75 band is the best band on gold in-sample
   AND on BTC — that ranking is consistent. But the 75+ band
   underperforms 60–75 on both (+0.099 vs +0.145 gold; +0.022 vs
   +0.211 BTC): the top of the score scale does not rank, in-sample
   or out. On silver the with-cost bands are uniformly negative
   (cost swamps the grade); no discrimination claim is made there.
4. **The temporal out-of-sample test remains undone** (data-blocked,
   above). Everything here is one bullish 2024–2026 window for the
   in-sample claims, plus instrument transfer within that same
   window. Regime transfer — the 2022 chop/bear test — is still the
   open question, and nothing in this document answers it.

**Bottom line:** combo A is the right detector config and the
underlying signal quality is not a gold-only mirage — but as a
tradable system the edge currently survives realistic costs only on
the instrument and cost model it was built on. No spin: that is a
research result, not a deployable one.
