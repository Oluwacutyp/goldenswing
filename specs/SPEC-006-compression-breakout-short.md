# SPEC-006: Compression Breakout — Short

## Conventions

Same as SPEC-001. Mirror of SPEC-005.

---

## GATE 1 — Macro Permission (D1)

```
IF daily_bias == -1 → TRUE
ELSE → FALSE
```

## GATE 2 — Compression Defined

Same as SPEC-005 (direction-agnostic).

```
range_high = MAX(High[1..30])
range_low  = MIN(Low[1..30])
contracting = ATR[0] < 0.7 * ATR[20]
tight = (range_high - range_low) < 3.0 * ATR[0]
upper_touches >= 2, lower_touches >= 2

IF all TRUE → TRUE, output range_high, range_low
ELSE → FALSE
```

## GATE 3 — Breakout (at bar T)

```
displacement = Open[0] - Close[0]   (positive for bearish)
broke_down   = Close[0] < range_low
strong_body  = displacement > 0.8 * ATR[0]
not_wick     = (Close[0] - Low[0]) < 0.3 * ATR[0]

IF broke_down AND strong_body AND not_wick → TRUE
ELSE → FALSE
```

## GATE 4 — Volume Expansion (at bar T)

```
IF V[0] > 1.5 * MEAN(V[1..30]) → TRUE
ELSE → FALSE
```

## GATE 5 — Entry, Stop, Invalidation

```
ENTRY:  entry = Close[0]

STOP:   stop = range_low + 0.5 * (range_high - range_low)

INVALIDATION:
  IF Close[i] > range_low for any i >= 0 after entry → EXIT IMMEDIATELY

TARGETS:
  measured = range_high - range_low
  target_1 = range_low - 1.0 * measured
  target_2 = range_low - 1.5 * measured
```

## POSITION SIZING

```
risk_pct = 1.0
Same formula as SPEC-001.
```

---

*Status: SPEC. Mirror of SPEC-005.*
