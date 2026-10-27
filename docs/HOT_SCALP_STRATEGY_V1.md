# GoldenSwing Hot Scalp — Research Sleeve v1

**State: implemented in source, shadow-only. Not deployed and not authorized to place orders.**

This is the first connected strategy slice. The separate Hot Scalp candidate engine remains isolated, while the same evidence ledger is now attached in shadow/audit mode to emitted signals from the existing MT5 playbooks through the common signal logger. This gives the wider engine a comparable evidence record without changing any current decision or execution authority. It is not yet a wholesale rewrite of all 18 strategy algorithms.

## What it looks for

The sleeve builds three entry families from closed market structure, then waits for live tick behavior at the resulting level. A historical candle can establish a still-valid setup; that candle never becomes a retroactive order.

1. **Pullback continuation.** M15 market structure is directional; a same-direction M5 structure break forms near an unfilled fair-value gap. The candidate is armed around the gap's midpoint. A live touch and reclaim is required before it can emit a candidate.
2. **Sweep and reclaim.** A sell-side or buy-side liquidity pool is swept and price closes back inside. A later same-direction structure event must confirm the reversal side. The sweep itself is not an entry.
3. **Breakout retest.** A directional M5 structure event establishes the broken level. Price must return to the level and reclaim it live before a candidate can be emitted.

Setup formation and entry trigger are separate events. Setup identity is stable across rescans and restarts; expiry and invalidation terminate the state. Candidate creation is based on the live tick path, not on seeing a future candle in replay.

## Evidence and confluence

The evidence ledger records a key, label, source, observation timestamp, chart interval, category, correlation group, stance (`supports`, `contradicts`, `neutral`), strength, availability and detail. It aggregates:

- broader M15 swing structure;
- M5 swing structure and BOS/CHoCH/MSB event;
- nearby unfilled imbalance and unmitigated order block;
- measured liquidity pool plus a verified wick-through / close-back-inside sweep;
- fresh named candle patterns, correlated with any sweep on the same bar;
- premium/discount location;
- live tick reaction (for Hot Scalp candidate creation);
- M5 volume expansion, with the candle's direction checked;
- M1 EMA(9)/EMA(21) momentum as a small, non-gating indicator input;
- session/killzone state;
- live spread relative to M1 ATR and executable reward/risk geometry.

The initial category budgets are research weights, not learned weights. Scores use a neutral 50 baseline. Positive evidence raises the score; contradictory evidence lowers it. Missing or neutral evidence contributes zero. Items sharing the same correlation group cannot add independent votes; an equal-strength conflict within one event cancels. There is deliberately **no Hot Scalp minimum-score gate**: the score explains/ranks the evidence and does not fabricate eligibility or suppress a valid trigger on its own.

**Existing MT5 playbook integration:** every non-management signal sent through `goldenswing/execution/signal_log.py` invokes the runner's shared evidence provider and stores `shared_evidence` beside the original decision. It includes measured score, explicit evidence, source timestamps, age in seconds, session, category budgets and unavailable reasons. This is observational only (`execution_gate=false`): it does not replace the playbook's score, rewrite `decision`, alter orders, or override risk. If M5/M15 context is missing or the evidence provider errors, the signal continues through the existing path and records the evidence as unavailable rather than making up a supportive reading.

The score is **not a probability** and must not be described as one. The weights require chronological, cost-aware replay calibration before any future model-promotion decision.

## Persistence and duplicate safety

`goldenswing/live/setup_store.py` uses SQLite transactions (`BEGIN IMMEDIATE`), a unique setup identity, durable lifecycle rows, event idempotency and a fencing token on trigger claims. Setup rows store formation, arm, touch, trigger, invalidation and expiry times. A `claimed` row is not silently re-claimed after an ambiguous interruption. The provided reconciliation API requires an authoritative positive or negative broker lookup before an execution claim can be released.

The current Hot Scalp engine has no order-dispatch path, so it cannot place a broker order. Its output is research/journal data only. Storage is intended to be a shared file on the same EC2 host; SQLite on a local disk is not a distributed lock for multiple hosts or unreliable network filesystems.

## Stacking policy

`goldenswing/live/stacking.py` provides the reason-coded risk eligibility check for both requested stacking modes:

- **Pyramid:** requires a fresh, independent same-direction trigger after the parent position has reached at least +0.50R favourable progress.
- **Multiple entries under one setup:** requires a separate unused trigger and refuses additions while the parent position is losing.

Both modes cap at three active layers, require a verified broker-normalized stop and charge the proposed layer against the same parent risk budget. Unknown risk fails closed. The candidate schema carries the parent ID and the stacking policy.

**Important implementation boundary:** this v1 does not yet receive authoritative open-position P&L, broker-normalized per-layer exposure and current parent budget inside the Hot Scalp runner. Therefore `evaluate_layer_add()` is implemented/tested as a policy component, but the Hot Scalp sleeve does not yet submit layer orders or claim the layer planner is fully wired to the broker position manager. That is a separate integration step, and no execution authority is enabled by this component.

## Session and market state

The sleeve calls the repository's DST-aware session mapping and includes session evidence even outside a killzone. The pattern session label's formerly shadowed 10:00 New York silver-bullet window is now reachable. M1 data is fetched as an *opt-in auxiliary stream* and held outside `TimeframeStack`, so it does not become a new hierarchy/confluence leg by accident.

This is not yet an independent broker trading-calendar implementation. A DST-aware session label is not proof that the broker is tradable, the symbol is open, the feed is fresh, or an exchange holiday is absent. Those states must be integrated before execution authority can even be considered.

## Historical tick replay

`scripts/replay_hot_scalp.py` reads three CSVs of OHLC bars (timestamps must be candle-open times in UTC) and a tick CSV with timestamp, bid and ask. At each tick it exposes only candles that have fully closed by that tick time. Outputs are appended to a JSONL event stream and a caller-selected SQLite state file. It never connects to MT5 and never sends orders.

For a clean run, supply a fresh SQLite path. Reusing a state file intentionally resumes/reuses the setup/event ledger. The replay harness is a research tool; it does not manufacture trades from candle highs/lows when tick data is unavailable.

## Tests added/changed

- Evidence scoring: correlation cap, duplicate-key suppression, missing evidence neutrality, conflicting event cancellation and fixed budget validation.
- SQLite lifecycle: repeated setup IDs, concurrent trigger claims, ambiguous claim recovery and event-idempotency.
- Hot Scalp: strategy family setup discovery, stable parent/child IDs, tick touch/reclaim candidate, repeated-tick duplicate prevention, and no-lookahead closed-bar replay.
- Pattern timestamps: date-index regression for order blocks, FVGs and sweeps.
- Session label: DST-aware New York wall clock and reachable silver-bullet window.
- Feed: M1 auxiliary stream is opt-in and separate from hierarchy legs.
- Existing MT5 signal integration: shared evidence callback, management-record exclusion, provider-failure isolation, auditable timestamps and explicit non-gating contract.
- Shared confluence's additional sensor links: liquidity pools/sweeps, order blocks, recent named candle patterns; each uses causal frames and reasoned source/time/stance.

The original working source passed 122 targeted tests across the prior new and existing suites; shared signal integration adds a separate focused test file, which is reported in the build notes. The full suite was not clean in this container: one unrelated test hard-codes `/home/hatch/workspace/goldenswing`, and another existing runner test requires `yfinance`, which is not installed here. Do not infer a full-suite pass from the targeted result.

## Before any execution integration

1. Run the exact source on historical MT5 ticks and record candidate, invalidation, expiry, spread, net reward/risk and per-parent outcome data.
2. Wire broker-normalized stop risk, shared parent budgets and open-position state into the stacking policy; test concurrent processes and crash recovery.
3. Integrate the single session source with broker market status, symbol trading hours, holidays, tick freshness and feed/calendar disagreement handling.
4. The non-gating evidence snapshot is now connected to the common MT5 signal logger. Next, compare those snapshots against outcomes in chronological replay; only then consider per-playbook strategy changes, with regression checks and preserved hard risk limits.
5. Run chronological walk-forward and locked out-of-sample evaluation. Keep the Hot Scalp order route disabled until invariants and replay tests are accepted.

This source bundle is based on the latest complete source archive available in this workspace. It has no Git metadata and has not been verified byte-for-byte against the live EC2 runtime commit. It is a patch bundle, not a remote deployment.
