# GOD-CE Salvage — trap intelligence as confluence inputs (2026-10-11)

The GOD-CE engine was killed as tested (−180% full sample, all four
walk-forward folds negative — `experiments/GODCE_ENGINE.md`). Its
components were never measured standalone. This experiment extracts the
trap-intelligence machinery and measures each piece as a confluence
input to the surviving playbooks.

**Discipline:** components are confluence inputs, not new playbooks. A
component earns its place only if it improves the WHOLE sequential
book, not just its cell (the phase-routing lesson). Walk-forward on
anything promoted. ASOC stayed OFF.

## 1. What was extracted

`goldenswing/features/traps.py` — self-contained port of
`godce/stateful.py::zones_traps_features` §§1–5, zero imports from
`goldenswing.godce`:

- **SFP** (swing-failure pattern): wick-through + weak reclaim +
  1.15× volSMA confirmation, 12-bar memory, engineered-liquidity latch
  fed by the fake-breakout machine.
- **Fake-breakout detector**: close-break → fail-back-inside within
  6 bars (feeds the SFP latch; too sparse to filter on alone).
- **Beyond-FVG state machine**: sweep-through + reclaim, 0→1→2 states,
  12-bar expiry from reclaim.
- **iFVG**: failed-gap polarity flip + tests, 120-bar zone expiry.
- **Bait / lure**: inside-gap bait (parked inside an un-reclaimed gap),
  lure (suspicious breakout on below-average volume).

Regression: **100.000000% per-column agreement on all 18 columns vs
godce over 11,426 H1 bars.** `tests/test_traps.py`: 18/18 pass.

Direction semantics (from the pine): `trap_boost_long` :=
`sfp_bear_active | beyond_bull | ifvg_demand` (bearish trap context =
bullish continuation); `bait_long` = parked inside un-reclaimed bull
FVG (blocks longs); `lure_long` = suspicious upside breakout (blocks
longs). Mirror for shorts.

## 2. Measurement design

Harness: `scripts/experiment_trapsalvage.py` (subset maps + 7 filter
variants × 3 playbooks, each as a full sequential book rebuild with
each playbook's REAL booking rule — anchor live_threshold=60, sleeves
unfiltered). Canaries reproduce exactly (199/+28.92%, 48/+34.61%,
61/+15.40%). Trap row read at the entry bar; signal-bar convention
re-verified before wiring (same finding, slightly weaker).

Walk-forward: `scripts/experiment_trapwalkforward.py` — the Phase-10
grid (train 12mo, step 3mo), variant as candidate pre-filter, validated
booking rule per window.

## 3. Results

### 3a. Subset maps (baseline book, trap state at entry bar)

| playbook | boost (n / avgR) | oppose (n / avgR) | neutral (n / avgR) |
|---|---|---|---|
| sweep_reversal | 124 / **+0.257** | 30 / **−0.411** | 45 / +0.171 |
| fvg_midline | 28 / +0.668 | 11 / +0.326 | 9 / +0.890 |
| trendline_retest | 43 / **+0.574** | 16 / **−0.584** | 2 / −0.102 |

Bait/lure incidence: sweep bait n=12 avgR −0.101; fvg bait n=13 avgR
**+0.756** (the pine's bait block would have cut winners there);
trendline bait n=4 avgR +1.819, lure n=2 avgR +4.770 (tiny, positive —
blocking lure cuts winners). Fake-breakout: 0 trades in any book
(single-bar event, as expected).

### 3b. Filter variants (whole-book, re-built)

**sweep_reversal** (baseline 199 / +28.92% / PF 1.25):
boost_only → 137 / +37.47% / 1.47 · nobait → 188 / +38.26% / 1.34 ·
nooppose → 131 / +33.63% / 1.45 · nolure → 198 / **+23.65%** (hurts —
cut 2 winners avgR +1.468) · sfp_only/ifvg_only → PF improves but book
collapses to 27/32 trades (too thin to promote).

**fvg_midline** (baseline 48 / +34.61% / PF 2.47): EVERY variant hurts
or is neutral. boost_only → 28 / +20.16% · nobait → 35 / +22.21% ·
nooppose → 22 / +23.52%. The sleeve is strong in every trap context;
it needs nothing from the trap engine.

**trendline_retest** (baseline 61 / +15.40% / PF 1.43):
**boost_only → 43 / +27.12% / PF 2.21** (cut avgR −0.473, 26% win —
the filter cuts losers) · nooppose → 15 / +8.60% (too aggressive) ·
nobait → 57 / +7.47% (hurts) · nolure → 59 / +5.15% (hurts — cuts 2
huge winners) · beyond_only → flat · sfp_only/ifvg_only → PF up, book
too thin.

### 3c. Walk-forward (surviving variants only)

**trendline_retest boost_only** — survives OOS on both conventions:

| convention | IS (0–12mo) | OOS (12–24mo) |
|---|---|---|
| entry bar | +2.33% (16, PF 1.29) vs base +1.21% (22, 1.11) | **+24.22% (27, PF 2.87)** vs base +14.03% (39, 1.62) |
| signal bar | +1.10% (16, PF 1.14) vs base +1.21% (22, 1.11) — flat | **+20.72% (21, PF 3.09)** vs base +14.03% (39, 1.62) |

Every OOS window ≥ baseline at the entry-bar convention; the 18–21mo
window (baseline −2.49%) flips positive (+1.56%). The IS caveat at the
signal-bar convention is recorded: the edge there is OOS-concentrated.

**sweep_reversal rescue check** — no rescue. boost_only OOS +6.34% vs
+5.52% baseline (marginal); nobait OOS +11.22% (best, still modest);
nooppose +5.95%. The 18–21mo window stays negative for ALL variants —
the trap filter does not fix the anchor's regime problem. The anchor's
walk-forward failure stands; its rule stays frozen per the iron rule.

## 4. Verdicts (per component, per playbook)

| Component | sweep_reversal | fvg_midline | trendline_retest |
|---|---|---|---|
| trap boost (SFP+beyond+iFVG) | measured positive, **not promoted** (anchor rule frozen; OOS gain marginal, no rescue) | **killed as a filter** (cuts winners; sleeve strong everywhere) | **PROMOTED as +4 confluence bonus** (playbook-gated; survives OOS) |
| bait block | measured positive in-sample (+38.26%), not promoted (anchor frozen) | **killed** (bait trades +0.756R — the pine's block would cut winners) | **killed** (bait +1.819R, n=4) |
| lure block | **killed** (cut 2 winners, +1.468R avg) | n/a (no lure trades) | **killed** (cut 2 winners, +4.770R avg) |
| oppose-trap drop | measured positive, not promoted (oppose n=16 < n≥25 exclusion bar) | killed (hurts total return) | killed as exclusion (n=15 book, hurts total return) |
| fake-breakout (standalone) | n/a (0 book trades — feeds SFP latch only) | n/a | n/a |
| SFP / beyond-FVG / iFVG alone | PF up, book too thin (27/32 trades) — **not promotable as filters** | same | same |

Cost gate (from `experiments/COSTGATE.md`, workstream B): **adopted on
no playbook** — it cut winners on sweep_reversal (+0.397R avg) and was
a no-op on fvg_midline; trivially positive on trendline_retest.

## 5. What shipped

- `goldenswing/features/traps.py` — the extracted module (tested,
  regression-clean vs godce).
- `goldenswing/confluence.py` — `ConfluenceConfig.bonus_trap_boost =
  4.0`, `trap_bonuses()` (playbook-gated to `trendline_retest`,
  signal-bar convention), wired into `score()` and
  `score_live_signal()`; `ensure_trap_features()` lazy-computes and
  caches the frame on the context. `EngineContext.trap_features`
  field added.
- `CONFLUENCE.md` — measured bonus row with the full basis.
- `tests/test_confluence.py` — 4 new tests (fires, direction-gated,
  playbook-gated, silent without frame, folded into `score()`).
- Scripts: `scripts/experiment_trapsalvage.py`,
  `scripts/experiment_trapwalkforward.py`.

Bonus semantics (honest): confluence refines ranking, it does not
re-gate — book membership still follows each playbook's validated
rule. The +4 helps a trap-aligned trendline candidate win same-bar
conflicts; it does not claim the +27.12% filter book.

## 6. What's still weak

- The promoted bonus rests on **21 OOS trades** (signal-bar) — below
  the 50-trade promotion bar. It is the strongest measured context for
  the sleeve and it survived OOS, but it is thin. The 2022–2024 window
  or a second instrument is the real confirmation.
- The trendline sleeve's edge is OOS-concentrated (IS was +1.21%);
  the trap boost helps in both halves but the sleeve itself is still
  young.
- Single components (SFP alone, iFVG alone) concentrate PF but
  collapse the book — they are ranking inputs at best, unmeasured as
  such.
- The anchor is untouched: trap confluence improves its in-sample
  book but does not rescue its OOS regime problem. A future *variant*
  playbook (anchor + nobait) is a candidate, not a change.
- Lure and bait blocks, as the pine defined them, are dead as
  filters — both cut winners on the books that had them. The
  engineered-liquidity idea survives only inside the SFP latch.
