# SPEC-002: Liquidity Sweep Reversion — Long

## Conventions

Same as SPEC-001. `T` = last closed bar. All inputs `T-i`. Binary gates.

---

## GATE 1 — Macro Permission (D1)

```
IF daily_bias == +1 → TRUE
ELSE → FALSE
```

## GATE 2 — H1 Trend Alignment

```
IF ema_stack_H1 == +1 → TRUE
ELSE → FALSE
```

## GATE 3 — Liquidity Pool Defined

```
swing_low = MIN(Low[1], Low[2], ..., Low[20])
swing_idx = index of first occurrence in [1..20]
IF swing_idx >= 3 → TRUE, swing_low = value
ELSE → FALSE
```

## GATE 4 — Sweep Occurred (at bar T)

```
penetration = swing_low - Low[0]
swept       = penetration > 0
closed_back = Close[0] > swing_low
not_excess  = penetration <= 0.5 * ATR[0]

IF swept AND closed_back AND not_excess → TRUE
ELSE → FALSE
```

## GATE 5 — Rejection Confirmed (M15, at bar T)

```
m15 = 4 M15 bars forming H1 bar T: c1..c4 chronological
micro_high = MAX(High of m15 bars 1..3)

IF Close of m15 bar 4 > micro_high → TRUE
ELSE → FALSE
```

## GATE 6 — Entry, Stop, Invalidation

```
ENTRY:  entry = Close[0]

STOP:   stop = Low[0] - 0.25 * ATR[0]

INVALIDATION:
  IF Close[i] < Low[0] for any i >= 0 after entry → EXIT IMMEDIATELY

TARGETS:
  target_1 = swing_high_20  (MAX(High[1..20]))
  target_2 = 1R projection
```

## POSITION SIZING

Same as SPEC-001.

---

*Status: SPEC. Mirror of SPEC-001.*
