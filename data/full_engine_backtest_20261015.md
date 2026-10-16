# Full-engine backtest — 2026-10-15 (local, NOT pushed, NOT deployed)

One integrated run of the WHOLE MT5 engine as one system (real
`MT5Runner`, all playbooks, hierarchy, gates, sizing, tick management,
exits) over 2026-04-01 → 2026-10-01.  Driver:
`scripts/backtest_full_engine.py`.

## Command

```
cd ~/workspace/goldenswing && .venv/bin/python -u scripts/backtest_full_engine.py --out /tmp/bt_full
```

Runtime: 863.8s wall (142,901 ticks). Exit 0.

## Result: the engine produces ZERO trades

| Metric | Value |
|---|---|
| Ticks / H1 bars processed | 142,901 / 2,978 |
| Triggers | 0 |
| Signals (any playbook) | 0 |
| Gated / hierarchy-blocked | 0 / 0 |
| Executed / shadow | 0 / 0 |
| Pendings placed / filled / cancelled | 0 / 0 / 0 |
| Trades closed / open at end | 0 / 0 |
| Total R / win rate / PF / max DD | 0.00R / 0% / 0.0 / 0.00R |
| Partials / profit-locks / panic / timeout closes | 0 / 0 / 0 / 0 |
| Golden Swing trades | 0 (untestable on this window, see below) |

## Why: the trigger pre-filter is geometrically unsatisfiable

Diagnostic (`/tmp/bt_diag.py`, replicates the runner's `live_setup`
refresh cadence on the same frames):

- `live_setup` armed: **573 / 3,137 H1 bars (18.3%)**.
- Not armed: H4 disagrees 1,444 (46%), D1 bias 0: 1,120 (36%).
  No zone / swing / ATR failures — legs 1–2 are the whole arming story.
- **Trigger possible in the armed setup's window: 0 / 573.**

The live trigger pre-filter (`_evaluate_live_tick`) requires, for a
long, `mid > ref_high` (last H1 swing high) AND
`zone.bottom <= mid <= zone.top` (H4 POI support band) at the same
tick; `check_live_trigger` repeats the in-zone requirement.  The
support zone sits at an H4 swing low; `ref_high` is an H1 swing high.
In all 573 armed setups the two intervals never overlapped, so no
tick could ever pass the pre-filter.  The closed-bar range check is
an upper bound (ticks live inside [low, high]), so 0 overlap means
0 possible triggers — not a sampling artifact, not a sim artifact.
Trigger firing itself is proven by unit tests
(`test_mt5_tick_clock.py` asserts `"trigger fired BOS_bull"`).

This is the hierarchy audit's leg-5 finding ("displacement breakout
and H4-zone requirements nearly mutually exclusive") confirmed on
the live path with exact numbers — and it is a DESIGN flaw in the
trigger conjunction, not a crash / lookahead / impossible-fill bug.
It was NOT fixed here: redesigning the trigger (e.g. breakout OR
zone-touch, or a visited-zone-then-breakout sequence) is an owner
decision, not a threshold re-tune.  Do not "fix" it by loosening a
number.

## Comparison vs the audit baseline

Audit (`mtf_hierarchy.py`, verdict per H1 bar, no playbooks/orders):
0 / 3,078 passed.  Integrated run (full live path): 0 triggers in
2,978 bars.  Different measurements, same conclusion: the
hierarchy/trigger chain as built produces nothing.  The integrated
run adds the mechanism: the chain dies at the trigger pre-filter,
not at the gate (0 gated, 0 hierarchy-blocked — nothing ever
reached the gate because no trigger ever fired).

## Backtest handling of new features (all stated, none silent)

- Confirmation: DISABLED = the configured timeout default
  (`fire` = auto-fire) with the 300s delay collapsed to zero.
- News pause: DISABLED — no historical calendar source exists;
  fetcher wired to raise if touched (it was not).
- Ladder: OFF (engine default). Scalp cap: OFF (no live scalp
  playbook per the audit). Golden Swing: UNTESTABLE — W1 needs 200
  weekly bars, dataset holds 106 (MN1: 25); W1/MN1 bias is NEUTRAL
  throughout, alignment never reaches Golden Swing.
- Reconcile every tick (live: every 20) so exits book at the first
  tick beyond the level instead of up to 25 sim-minutes late.
- Fills: market at tick mid (zero slippage model); pendings at level
  on cross. Spread constant 0.20. Account $10,000, 1% risk/trade,
  leverage 500. Ticks: 4 per M5 bar (O/H/L/C, fixed OHLC order).
- Wall-clock logic (cooldowns, dedup TTL, guard day/week windows)
  ran on sim time; `datetime.now` journal labels stay real time.

## What remains

- Push + deploy: NOT done (local commits only).
- Owner decision: trigger redesign (see above).
- The full feature set (partials, panic, time exits, score decay,
  DXY, reasoning, health, urgency/ladder) is implemented and
  unit-tested but has ZERO live-path coverage from this run —
  nothing ever reached it.  It must be exercised (by the trigger
  redesign, or by a targeted harness) before it can be called
  verified in the integrated system.

---

# Re-run: redesigned trigger (2026-10-15, same window)

Same driver, same window (2026-04-01 → 2026-10-01), after the
trigger redesign (containment-first + sequential state machine +
armed TTL + Close Rule + fill transition + one-trigger-per-H1-bar
fired-window guard).  Hierarchy gates downstream UNCHANGED.

## Command

```
cd ~/workspace/goldenswing && TMPDIR=$HOME/workspace/.pytest-tmp .venv/bin/python scripts/backtest_full_engine.py --out ~/workspace/.backtest-full-engine-20261015-fixed
```

Runtime: 1070.3s wall (142,901 ticks). Exit 0.

## Result: the deadlock is resolved — the engine now fires and trades (rarely)

| Metric | Value (was, pre-redesign) |
|---|---|
| Ticks / H1 bars processed | 142,901 / 2,978 (same) |
| Triggers | **19** (was 0) |
| Signal records | 8 (was 0) |
| Signals accepted / gated / hierarchy-blocked | 1 / 5 / 5 (was 0 / 0 / 0) |
| Pendings placed / filled / cancelled | 1 / 1 / 0 (was 0 / 0 / 0) |
| Executed | 1 (was 0) |
| Trades closed / open at end | 1 / 0 (was 0 / 0) |
| Total R / win rate / PF / max DD | **-0.77R / 0.0% / 0.0 / 0.77R** (was 0.00R / 0% / 0.0 / 0.00R) |
| Per-playbook | inversion: 1 trade, -0.77R, $-61.02 |
| Close reason | stop — "SL hit (trail)" |
| Management | profit_locks 1, partials 0, breakevens 0, panic 0, timeout 0 |
| Trigger invalidations | 29 (left_zone 26, stale 2, invalidation 1) |
| Golden Swing trades | 0 (untestable, same W1/MN1 sample limit) |
| qm_d1 shadow | 4 (unchanged) |
| Broker end balance | $9,938.98 (from $10,000) |

Trigger shapes (19): failed_retest 6, fvg_reaction 5,
volume_spike 2, bos 2, tweezer_top 1, pin_bar 1,
sweep_reclaim 1, displacement 1.  Entry clocks: M15 5, M30 14.
Max fires per H1 bar: 1 (the fired-window guard holds — no storm).

Signal records (8): fvg_limit:gated 2, inversion:accepted 1,
inversion:gated 3, sweep_reversal:below_floor 2.
Blocked: direction_mismatch 5, min_score 2, total 7.
Gated reasons (all direction_mismatch): "hierarchy allows -1,
signal is +1" (3), "hierarchy allows +1, signal is -1" (2).

## The one trade

inversion SHORT: pending LIMIT @ 4353.34 (0.12 lots),
SL 4359.92 → filled → profit-lock moved the SL to 4358.42 →
stopped out at 4358.42 for -0.77R.  The profit-lock machinery
got its first live-path coverage and reduced the loss from
-1.00R to -0.77R.  Sample size 1 — no validation.  The result
is "the engine now trades (rarely)", not "the engine is
profitable".

## Why only 1 trade from 19 triggers (measured, not tuned)

- 11 of 19 fired triggers produced NO playbook emission at all:
  the playbooks evaluate the H1 frame independently and mostly
  do not see their setup on the fired verdict.
- Of the 8 emissions, 5 were gated on direction_mismatch: the
  playbook emitted the OPPOSITE direction to the trigger's
  (e.g. trigger +1, inversion emits -1; trigger -1, fvg_limit
  emits +1).  The hierarchy direction gate (unchanged, per the
  task) refused them.
- 2 more were below the score floor.
- 1 accepted → 1 trade.

The trigger redesign alone does not produce a trading engine:
the downstream playbooks are not trigger-aware (they re-derive
direction from the frame instead of taking the fired trigger's
direction).  Whether to make playbooks trigger-aware, or to
investigate why they oppose the trigger, is an OWNER decision —
not done here (gates unchanged per the task, no thresholds
re-tuned).

## The trigger storm (found and fixed mid-run)

The first re-run attempt was killed at ~26 min: the same H1 bar
fired `tweezer_top` on EVERY tick (dozens of fires per bar),
because after a fire the next tick re-armed and the immediate
confirm shot re-fired on the same already-closed composing
bars.  Fix: one trigger per H1-bar window
(`MT5Runner._fired_h1_bar`) — the H1 bar IS the trigger's
window (the pending expires at the H1 close) and the live setup
is fixed within it, so a second fire there is necessarily a
duplicate.  Regression test:
`test_runner_second_touch_same_h1_bar_does_not_refire`.
The fixed run verified max 1 fire per H1 bar.

## Backtest handling (same as the baseline run)

Confirmation disabled = configured timeout default (`fire` =
auto-fire) with the 300s delay collapsed; news pause disabled
(no historical calendar; fetcher wired to raise if touched — it
wasn't); ladder OFF, scalp cap OFF; reconcile every tick;
spread constant 0.20; market fills at tick mid, pendings at
level; $10,000, 1% risk, leverage 500; wall-clock logic ran on
sim time.

## What remains (after the redesign)

- Push + deploy: NOT done (local commits only).
- Owner decision: playbook/trigger direction gap (see above) —
  playbooks re-derive direction instead of taking the trigger's.
- Owner decision: the 11 silent fires (playbooks emit nothing
  on most fired verdicts).
- Golden Swing still untestable on this window (W1 needs 200
  bars, dataset holds 106; MN1: 25).
- Partial fills still unit-tested only (sim broker never
  partially fills).
- Score-decay-on-linger: follow-up, not built (TTL expiry alone).
- Fill + close between reconciles = lost trade: pre-existing
  hole, not fixed (reconcile interval is small, window is tiny).
- Remainder fill after base closed opens fresh: open design
  question (whether a close should cancel a resting remainder).
- Invalidation block zone-change lift: code-reviewed, not
  directly tested.
