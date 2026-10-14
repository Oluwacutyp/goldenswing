# GoldenSwing — Phase 9: The Confluence Engine

**"A real trader in a bot."** One AWARE system, multiple entry
playbooks. The chart reader stays the single source of truth; the
playbooks are how a real trader acts on what the read says — different
setups, each with its own trigger, stop and exit, ranked by confluence,
one position at a time.

## Architecture

```
bars → MarketReader → FeatureEngine → MarketState (+FeatureSet)
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
          SweepReversalPlaybook     FVGMidlinePlaybook     TrendlineRetestPlaybook
          (LIVE — the anchor)       (SHADOW)               (SHADOW)
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                                   ConfluenceEngine
                              rank by confluence
                              one position at a time
                              risk = final gate
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                          live book      shadow books      journal
                          (P&L)          (paper only)      (JSONL)
```

### The playbook contract (`goldenswing/playbooks/`)

Each playbook owns: setup detection, trigger, stop/target/exit policy,
required context, base quality score. **No playbook reaches into
another's machinery.** The engine only ranks pre-simulated candidates —
it never re-simulates and never edits an outcome.

**The iron rule:** a playbook keeps the EXACT entry/exit it was
validated with. Mixing a trigger from one playbook with exits from
another is a new experiment, not a refactor.

| Playbook | Setup | Entry | Exit | Status | Session scope | Validated |
|---|---|---|---|---|---|---|
| `sweep_reversal` | Phase-2 detector scan | sweep → displacement + CHoCH/BOS | Phase-2 swing exits (buffered stop, partial @ pool, trail) | **LIVE** | `any` (session-agnostic core) | +28.92%, 199, PF 1.25 |
| `fvg_midline` | fresh CHoCH + untraded FVG | resting limit at FVG midline | **fixed 2R** (the OB exit diluted) | SHADOW | `any` | +34.61%, 48, PF 2.47 (hunting ground applied; experiment measured 97 / +25.88% unfiltered) |
| `trendline_retest` | break of respected line | return touch within 12 bars | Phase-2 swing exits | SHADOW | `any` | +15.40%, 61, PF 1.43 |

### Session scope (first-class playbook property)

`session_scope` is declared on the playbook, next to its other context
requirements — not a measurement dimension. Two kinds:

- **Session-agnostic** (`session_scope = "any"`): the works-always core.
  Fires in every session. All three current playbooks are `any` — their
  context maps show no decisively bad session.
- **Session-specific** (e.g. `session_scope = "london"`): only allowed to
  fire in its session — a London-open playbook, an Asian-range playbook.
  The engine never lets it fire outside that session, and its backtest
  is judged on its session only (walk-forward filters to the session).

A future playbook whose map shows it only works in one session should
be DECLARED session-scoped rather than left `any` with exclusions —
the declaration is the honest statement of where it was validated.

### Hunting grounds (measured context maps)

Backtests here do not produce binary verdicts — they produce MAPS of
where the strategy performed well (`goldenswing/contextmap.py`,
`scripts/measure_context_maps.py`, raw maps in
`goldenswing/playbooks/context_maps.json`). Each playbook's
`hunting_ground` is derived from its map by conservative rules
(n ≥ 25, decisively bad = avg R < -0.05 and negative in both halves;
proven = beats overall avg R by 0.15+):

- **Excluded cells never fire.** Exclusions are verified against the
  SEQUENTIAL book before adoption — removing trades re-times everything
  downstream, and the Phase-5 routing lesson stands: a cell that looks
  bad in isolation is only excluded if the book improves without it.
- **Proven cells earn confluence bonuses** — session/regime/HTF/phase
  bonuses come from these maps, never from hand-set priors. (AOC,
  divergence presence and the imbalance magnet stay as independent
  measured findings — they are features, not map dimensions.)

Measured 2026-10-11 (GC=F H1, Oct 2024–Oct 2026):

- `sweep_reversal`: no exclusions — fires everywhere. Proven:
  newyork session +0.397R, expanded vol +0.329R, Friday +0.323R (+2 each).
- `fvg_midline`: **trend=down at the fill bar excluded** (-0.119R, n=52,
  negative in both halves). Sequential book improves +25.88% → +34.61%,
  PF 1.41 → 2.47 — not the routing trap. Proven: trend=up +0.676R (+4),
  with-bias +0.482R (+2), trend_down phase +0.426R (+2).
- `trendline_retest`: no exclusions, no proven cells (61 trades too thin
  for verdicts) — fires everywhere, map is descriptive until OOS
  thickens it.

### Confluence scoring (`goldenswing/confluence.py`)

`confluence = base_score + context bonuses`, every bonus a measured
finding, every bonus named in the journal:

| Bonus | Size | Measured basis |
|---|---|---|
| AOC backing | +4 | entry inside a direction-aligned concentration (AOC.md; not re-added for the anchor — already in its score) |
| Divergence presence | +3 | any active RSI divergence marks *energy*; direction ignored (DIVERGENCE.md: aligned ≈ opposing) |
| Imbalance-magnet proximity | +2 | unfilled FVG midline within 0.5 ATR (IMBALANCE.md: weak but real) |
| Map-proven context | +2/+4 | the candidate's playbook firing in a proven cell of its MEASURED context map (hunting ground) |
| Trap-boost alignment | +4 | **trendline_retest only** (playbook-gated): direction-aligned GOD-CE trap boost (SFP / beyond-FVG reclaim / iFVG test) active at the signal bar (GODCE_SALVAGE.md: boost subset +0.574R, n=43, vs +0.248R overall; whole book +15.40%→+22.04%, PF 1.43→2.16; OOS +14.03%→+20.72%, 21 trades, PF 3.09) |

Session/regime/HTF/phase bonuses come from the maps, never from priors —
the old hand-set killzone/HTF bonuses were removed when the anchor's map
did not support them. Confluence ranks candidates when two fire on the
same bar (higher wins) and journals the reasoning. Book *membership*
still follows each playbook's validated rule (anchor ≥ 60; sleeves
unfiltered within their hunting grounds) — confluence refines, it does
not re-gate.

### Trader behaviors encoded

- **One position at a time** across all playbooks — a second signal
  while in a trade is skipped and logged as `position-lock` (never two
  same-direction positions from different playbooks at once).
- **Same-bar conflict**: the higher-confluence setup wins.
- **Risk is the final gate**: every live entry passes
  `RiskGuards.check_new_trade` (kill switch, drawdown halt, exposure
  caps, daily/weekly limits). No playbook bypasses it.
- **Every fill logs its full thesis**: playbook, trigger, context,
  confluence score and bonuses — the journal reads like a trader's log.

### Risk modes

- `enforce` (live trading): a denied trade is skipped and logged.
- `shadow` (backtest): every risk decision is computed and logged, but
  the historical sequence is never altered — the anchor canary stays
  exact. The shadow log shows exactly where risk *would* have
  intervened.

## Verification

`python scripts/run_confluence_book.py` — the canary gate:

- **Live book reproduces the anchor exactly**: 199 trades, PF 1.25,
  **+28.92%**, MaxDD 10.38%, Sharpe 1.00.
- **Shadow books reproduce the experiments**: FVG-midline 97 trades /
  +25.88% / PF 1.41; trendline-retest 61 trades / +15.40% / PF 1.43.
- 199 risk checks logged in shadow mode; 152 position-lock skips;
  1,203-line journal at `data/confluence_journal.jsonl`.

## What remains weak (honest)

1. **Risk defaults would halt the anchor almost immediately in live
   mode.** The shadow log shows the 3-consecutive-loss halt latching at
   bar 210, which would have blocked 193/199 entries. A 45%-win-rate
   book hits 3 straight losses routinely — the limit is calibrated for
   a different system. Recommend recalibrating (e.g. 5–6, or an
   R-multiple drawdown trip instead of a loss-count trip) on paper
   before live. NOT changed here: the risk layer's defaults are
   user-approved; this is a flagged finding, not a silent fix.
2. **Sleeve base scores are placeholders (60.0)**, not measured
   scores — the sleeves were validated without scoring. They only rank
   sleeve-vs-sleeve today. OOS promotion must calibrate them.
3. **Single window, single instrument.** Everything — anchor and
   sleeves — is in-sample on 2y gold H1. The promotion checklist below
   is the gate before any sleeve touches live P&L.
4. **Map bonuses are in-sample.** Session/regime/HTF/phase bonuses now
   come from measured context maps rather than priors — but the maps are
   still one 2y in-sample window. The hunting-ground exclusions
   (fvg_midline's trend=down) need OOS confirmation before they are
   trusted the way the anchor's gates are.

## Promotion checklist (shadow → live)

A sleeve is promoted only when ALL of these hold — no exceptions, no
"looks good":

1. **Out-of-sample window**: same code, zero refits, on data the
   sleeve never saw (2022–2024 gold H1 once MT5 data lands, or a
   second instrument). Positive return, PF ≥ 1.2.
2. **Both halves positive** on the OOS window (no one-regime artifact).
3. **Sample**: ≥ 50 OOS trades.
4. **Diversifies**: daily-R correlation with the anchor book < 0.5
   (the trendline sleeve measured −0.01 in-sample — the bar it must
   clear again OOS).
5. **Base score calibrated**: the sleeve's 60.0 placeholder replaced
   by a score that discriminates on OOS (top band beats bottom band),
   so cross-playbook ranking means something.
6. **Risk coexistence**: a full confluence run with the sleeve live
   keeps the anchor canary exact and the combined book within the
   risk layer's limits without the halt latching in the first month.

### Walk-forward status (2026-10-11, Phase 10)

Expanding 12mo train / 3mo test (`scripts/walkforward.py`):

| Playbook | OOS trades | Return | PF | Sharpe | Verdict |
|----------|-----------|--------|----|--------|---------|
| sweep_reversal (anchor) | 94 | -0.28% | 1.01 | 0.04 | **FAILS** — edge concentrated in months 0-12 |
| fvg_midline (shadow) | 27 | +18.82% | 2.54 | 2.30 | **SURVIVES** (thin; hg caveat) |
| trendline_retest (shadow) | 39 | +14.03% | 1.62 | 1.23 | **SURVIVES** |

**The anchor does not meet criterion #1.** It is live on in-sample
strength only. Options: (a) resolve the months 0-12 vs 12-24 regime
difference, (b) demote to shadow, (c) promote a sleeve that clears the
checklist. See SYSTEM.md §5 and §10.

The sleeves clear the walk-forward smoke test but still need: (a) the
full OOS window (2022–2024 or second instrument) per criterion #1,
(b) ≥ 50 OOS trades per criterion #3 (fvg has 27, trendline has 39),
(c) `live_signal` implementations (only sweep_reversal has one).

Promotion is a config flip (`live = True` on the playbook) plus a
commit — the engine already handles multi-playbook live books
(tie-breaks, one-position rule, per-playbook thresholds).
