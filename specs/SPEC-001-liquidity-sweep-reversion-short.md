# SPEC-001: Liquidity Sweep Reversion — Short

## Conventions

- `T` = the most recently **closed** bar. Never the forming bar.
- `Close[i]`, `High[i]`, `Low[i]`, `Open[i]` = OHLC of bar `T-i`.
  `Close[0]` = close of last closed bar. `Close[1]` = one bar before that.
- `ATR[i]` = 14-period ATR evaluated at bar `T-i`, using only bars `<= T-i`.
- All swing/level calculations use `offset=1` minimum (never include `T` itself
  in the lookback window that defines the level being tested at `T`).
- Every gate returns binary TRUE/FALSE. No scores, no "strong," no "clean."
- Gates evaluate in order. If any gate is FALSE, evaluation stops.

---

## GATE 1 — Macro Permission (D1)

```
INPUT:  daily_bias  (from D1: +1 / -1 / 0)
OUTPUT: TRUE / FALSE

RULE:
  IF daily_bias == -1 → TRUE
  ELSE → FALSE
```

Rationale: D1 assigns direction. No D1 permission, no evaluation.

---

## GATE 2 — H1 Trend Alignment

```
INPUT:  ema_stack_H1  (+1 / -1 / 0)
OUTPUT: TRUE / FALSE

RULE:
  IF ema_stack_H1 == -1 → TRUE
  ELSE → FALSE
```

---

## GATE 3 — Liquidity Pool Defined

```
INPUT:  High[1..N], N = 20
OUTPUT: swing_high (float) / FALSE

RULE:
  swing_high = MAX(High[1], High[2], ..., High[20])
  swing_idx  = index of first occurrence in [1..20]
  IF swing_idx >= 3 → TRUE, swing_high = value
  ELSE → FALSE   (swing too recent, not a real pool)
```

Rationale: the pool must be at least 3 closed bars old. A high printed
1–2 bars ago is not a liquidity pool, it is current price action.

---

## GATE 4 — Sweep Occurred (at bar T)

```
INPUT:  High[0], Close[0], swing_high, ATR[0]
OUTPUT: TRUE / FALSE

RULE:
  penetration = High[0] - swing_high
  swept       = penetration > 0
  closed_back = Close[0] < swing_high
  not_excess  = penetration <= 0.5 * ATR[0]

  IF swept AND closed_back AND not_excess → TRUE
  ELSE → FALSE
```

Rationale:
- `swept`: the wick took out the high. Liquidity was grabbed.
- `closed_back`: the close rejected. No acceptance.
- `not_excess`: penetration beyond 0.5 ATR is not a sweep, it is expansion.
  (Tunable, but binary.)

---

## GATE 5 — Rejection Confirmed (M15, at bar T)

```
INPUT:  M15 bars aligned to H1 bar T (the 4 M15 bars forming H1 T)
OUTPUT: TRUE / FALSE

RULE:
  Let m15 = the 4 M15 closes within H1 bar T: c1, c2, c3, c4 (chronological).
  micro_low = MIN(Low of m15 bars 1..3)

  IF Close of m15 bar 4 < micro_low → TRUE
  ELSE → FALSE
```

Rationale: the sweep is not complete until lower-timeframe structure breaks
in the reversal direction. Wick alone is not enough.

---

## GATE 6 — Entry, Stop, Invalidation (all defined at close of T)

```
ENTRY:
  entry = Close[0]   (market on H1 close of bar T)

STOP:
  stop = High[0] + 0.25 * ATR[0]

  Rationale: the sweep extreme is the line. 0.25 ATR buffer avoids
  stop-hunting noise. Binary: stop is a number, not a zone.

INVALIDATION (evaluated on every subsequent closed bar):
  IF Close[i] > High[0] for any i >= 0 after entry → EXIT IMMEDIATELY
  (Do not wait for stop. The thesis was "no acceptance above the sweep."
   A close above the sweep extreme falsifies it.)

TARGETS:
  target_1 = swing_low_20   (MIN(Low[1..20]), opposite liquidity pool)
  target_2 = 1.0 * (entry - stop) projected from entry  (1R, partial option)
```

---

## POSITION SIZING

```
INPUT:  equity, risk_pct = 1.0
RULE:
  risk_usd      = equity * risk_pct / 100
  stop_distance = entry - stop          (positive for short)
  lots          = risk_usd / (stop_distance * point_value_per_lot)
  lots          = FLOOR(lots * 100) / 100   (round down to 0.01)
```

No discretion. The contract outputs lots.

---

## COMPLETE EVALUATION ORDER

```
1. GATE 1 (D1 permission)      → FALSE = stop, no trade
2. GATE 2 (H1 stack)           → FALSE = stop, no trade
3. GATE 3 (pool defined)       → FALSE = stop, no trade
4. GATE 4 (sweep at T)         → FALSE = stop, no trade
5. GATE 5 (M15 confirmation)   → FALSE = stop, no trade
6. Compute entry/stop/size     → emit signal
7. Invalidation runs every bar after entry
```

---

## What this spec eliminates

| Ambiguity (natural language)        | Resolution (this contract)              |
|-------------------------------------|-----------------------------------------|
| "wick beyond the high"              | GATE 4: `High[0] > swing_high`, binary  |
| "close back inside"                 | GATE 4: `Close[0] < swing_high`, binary |
| "recent swing high"                 | GATE 3: MAX(High[1..20]), idx ≥ 3      |
| "don't use live prices"             | All inputs are `T-i`, never forming bar |
| "strong rejection"                  | GATE 5: M15 micro-structure break       |
| "stop above the high"               | `High[0] + 0.25 * ATR[0]`, exact       |
| "exit if it fails"                  | Invalidation: any close > High[0]       |

---

*Status: SPEC. Not yet implemented. Implementation must follow this contract
exactly — any deviation is a bug, not a judgment call.*
