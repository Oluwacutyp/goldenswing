# Bias contract — structure gates, macro sizes, DXY nudges, flat is HOLD

Three separate concerns. Never confused, never conflated.

## 1. `evaluate_bias()` — STRUCTURE direction, the ONLY bias that gates

`goldenswing/live/multitf.py::evaluate_bias`, hierarchy leg 1 (`d1_bias`).

- **+1** iff the last two confirmed swings show higher high AND higher low.
- **−1** iff lower high AND lower low.
- **0 (flat)** iff anything else: mixed (lower highs + higher lows = contracting
  range; higher highs + lower lows = expanding range), or fewer than 2
  confirmed swings per side.

A swing at index `i` needs `swing_n` bars AFTER it (default 3) before
`find_swings` recognises it. A D1 bias flip is therefore only recognised
≥ 3 daily closes after the extreme printed. Price action alone cannot flip it.

**`d1_flat` is a legitimate HOLD** during choppy / transitioning regimes,
caused by that confirmation lag. It is NOT a feed error and NOT a bug.

HOLD tracking: `goldenswing/live/hierarchy.py::D1BiasState` /
`bias_hold_state` — current bias, pending (unconfirmed) swing highs/lows,
`bars_remaining` per pending swing (1..`swing_n`), previous confirmed pair,
which legs are already satisfied vs missing, and `earliest_flip` (earliest
date the bias could flip, both legs of one direction confirming).
`live_setup()` returning `None` with reason `d1_flat` is accompanied by
`live_setup_hold_state()`; `evaluate_hierarchy()` carries `hold_state` on
flat verdicts. `/status` Live Setup renders the HOLD line.

Logging: `log_hold_state` → INFO, `HOLD …`. Never ERROR/WARNING, never
`FEED_ERROR`.

## 2. `compute_macrotrend()` — SIZING ONLY, never gates

`goldenswing/live/macrotrend.py::compute_macrotrend`.

- Per-TF bias = last close vs EMA200 (EWMA, span=200). Fewer than 200 bars → 0.
- Golden Swing = D1, W1, MN1 EMA biases all equal and nonzero → 1.5x, size 1.5x.
- D1 == W1 (nonzero) → `D1_W1`, 1.25x, size 1.0x.
- Else → `D1_Only`, 1.0x.
- D1/W1 conflict (both nonzero, disagreeing) → size 0.5x, still trades.
- `structure disagrees` is a label only; it changes nothing about gating.

Alignment never touches hierarchy legs. It only scales lots on a trade the
hierarchy already allowed, plus a Golden Swing confirmation tap.

## 3. DXY — CONFLUENCE NUDGE ONLY, never gates

`goldenswing/live/dxy.py`.

- Correlation < −0.8 → +0.5 confluence; > −0.2 → −0.5; else 0.
- Unavailable → 0. A missing DXY feed degrades to 0; it is not a blocker.

## Feed errors — the separate channel

`log_feed_error` → ERROR, `FEED_ERROR symbol=… reason=…`.

Call only when the feed itself failed (`feed_status` ok=False: terminal
disconnected, symbol unselected, fetch failed, stale bars past the
staleness bound), never for a flat bias on a healthy feed.

Distinguishing test: flat bias + healthy feed → log contains `HOLD`, never
`FEED_ERROR`. Stale feed + nonzero bias → log contains `FEED_ERROR`, never
`HOLD`.
