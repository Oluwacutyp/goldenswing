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
