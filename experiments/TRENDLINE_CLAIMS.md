# GoldenSwing — Trendline claims #29–#31 (Element Of Trading Vol V, Doctor Pips)

**Data:** GC=F hourly, 11,426 bars, 2024-10-11 → 2026-10-09 (1.99y). $0.45/oz
cost, Phase-2 exits (buffered structural stop, 50% partial at first pool
≥ 1R, trail after +1.5R, 72-bar horizon). Same reporting everywhere:
trades, win%, PF, avg R, return, MaxDD, Sharpe, window halves.

**Anchor (verified, must reproduce):** Combo A + aoc @ ≥60 — 199 trades,
PF 1.25, +28.92%, MaxDD 10.38%, Sharpe 1.00. Reproduced exactly before any
claim was measured (the canary).

All knobs pre-committed, not fitted: 12-bar forward windows, 2-bar rearm,
touch tolerance 0.20 ATR (production value), wedge contraction 0.75 over
30 bars with ≥ 3 combined touches.

---

## Claim #29 — anchor convention (shadow-to-shadow vs body-to-body)

**The book's claim:** shadow-anchored trendlines give lower-risk entries
(less floating/drawdown) than body-to-body construction.

**Method:** two streaming builders over the SAME confirmed swing sequence —
production `_Builder` (shadow extremes, the current system) and
`_BodyBuilder` (body extremes: max(open,close) at swing highs,
min(open,close) at swing lows). Validation, dedupe, lifecycle and
tolerances inherited unchanged, so any difference is the anchor convention
alone. Two roles: (a) touch quality, (b) entry context on the 467 anchor
signals.

### #29a — touch quality

| Convention | Lines | Touches | Bounce rate | Mean adverse | Median adverse | Failed-touch |
|---|---|---|---|---|---|---|
| Shadow | 2070 | 1716 | 0.622 | +0.880 ATR | +0.738 ATR | 0.702 |
| Body | 1034 | 613 | 0.540 | +0.858 ATR | +0.678 ATR | 0.747 |

Adverse excursion = worst move against the line over the 12 bars after a
touch, in ATR. Failed touch = close through the line within 12 bars.

**Read:** the book's "less floating" claim is not supported — adverse
excursion is identical (+0.88 vs +0.86 ATR). Shadow touches are marginally
cleaner (higher bounce, lower fail rate) but the gap is small. The big
structural difference is SELECTIVITY: body anchoring produces half the
lines (1034 vs 2070) — the between-anchor validation kills far more
candidates when anchors sit inside the range. It is a selectivity
difference, not a quality difference.

### #29b — entry context (anchor-signal shadow R by touch proximity)

| Bucket | n | Avg R | Win% | PF |
|---|---|---|---|---|
| Shadow touch | 135 | +0.073 | 42.2 | 1.12 |
| Shadow no-touch | 332 | +0.049 | 42.5 | 1.09 |
| Body touch | 67 | +0.213 | 46.3 | 1.41 |
| Body no-touch | 400 | +0.030 | 41.8 | 1.05 |

**Read:** shadow touches barely discriminate (+0.073 vs +0.049). Body
touches mark better context (+0.213, PF 1.41) — the selectivity shows up
again: fewer, harder-earned lines, and their touches mean more. But n=67
is ~1 standard error; this is a weak candidate, not a finding.

**Verdict #29: the book's claim KILLED** (no measurable risk reduction from
the anchor convention). Body-touch entry context logged as a weak
candidate (#42) — needs OOS before it is scored or gated anywhere.

---

## Claim #30 — wedges as continuation

**The book's claim:** rising/falling wedges signal continuation in the
prior-trend direction. H&S and double top/bottom were killed (#12); wedges
were never measured. Low prior, stated upfront.

**Method:** causal wedge detection — converging support + resistance, both
respected (≥ 1 touch each), price between them, boundary distance
contracting (≤ 0.75× over 30 bars), ≥ 3 combined touches. Prior trend =
sign of the 50-bar ROC. First breakout within 20 bars classified in-trend
or counter-trend. Drift measured over 24 bars. Control: the same drift
metric on every bar (unconditional prior-trend following).

### Drift

| Group | n | Mean drift | Win% |
|---|---|---|---|
| In-trend wedge breakouts | 14 | **−0.250 ATR** | 28.6 |
| Counter-trend breakouts | 3 | +1.642 ATR | 66.7 |
| Base rate (all bars) | 11351 | +0.252 ATR | 50.8 |

### Gated book (in-trend breakouts, Phase-2 exits)

| Book | Trades | Win% | PF | Return | MaxDD | Avg R |
|---|---|---|---|---|---|---|
| Full | 4 | 0.0 | 0.00 | −3.98% | 3.98% | −1.009 |
| H1 | 1 | 0.0 | 0.00 | −1.01% | 1.01% | −1.013 |
| H2 | 3 | 0.0 | 0.00 | −2.99% | 2.99% | −1.008 |

(Only 4 of 14 breakouts passed the non-score gates; the drift test is the
verdict, the book is confirmation.)

**Read:** the wedge underperforms naive trend-following by half an ATR.
In-trend breakouts drift the WRONG way (−0.250 vs +0.252 base). Two extra
notes: (1) zero textbook rising/falling wedges under a strict taxonomy —
all 21 converging pairs were symmetric ("mixed"); (2) the 3 counter-trend
breakouts drifted +1.642 in the breakout direction — n=3 noise, but it
rhymes with the graveyard: when these patterns resolve, the breakout
direction is the trade, not the prior trend.

**Verdict #30: KILLED, decisively.** Wedges join H&S and double tops in the
pattern graveyard. The low prior was correct.

---

## Claim #31 — trendline breakout-retest entry

**The book's claim:** after a trendline break, enter on the retest touch of
the flipped (RBS/SBR) trendline. The JeaFx zone-retest was killed
structurally (#9) — that kill is NOT inherited; a trendline-retest is a
different mechanic.

**Method:** break events on respected lines (touches ≥ 1, i.e. ACTIVE)
arm the setup in the breakout direction. Price must first spend 2 bars away
from the broken line (mirrors the builder's rearm — a genuine retest,
not the bar after the break); the first return touch (wick reaches the
line, close stays on the breakout side) within 12 bars is the entry, one
trade per break. Structural stop beyond the retest extreme, Phase-2 exits,
detector non-score gates (daily veto hard, room-to-pool, stop band).

### Book (61 traded)

| Book | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |
|---|---|---|---|---|---|---|---|
| Shadow (all) | 67 | 46.3 | 1.42 | — | — | — | +0.240 |
| Traded | 61 | 45.9 | 1.43 | **+15.40%** | 9.32% | 0.83 | +0.248 |
| H1 | 22 | 45.5 | 1.11 | +1.21% | 6.92% | 0.22 | +0.063 |
| H2 | 39 | 46.2 | 1.62 | +14.03% | 9.32% | 1.23 | +0.353 |

Exit mix: 31 trail / 36 stop / 0 timeout — the Phase-2 exits do the work;
half the book rides the trail.

**Diversification:** daily-R correlation with the anchor book = **−0.01** —
an uncorrelated P&L stream despite 37/67 trades overlapping anchor trades
in time. It fishes a different part of the market.

**Verdict #31: CANDIDATE (paper/OOS).** +15.40%, PF 1.43, positive in both
halves — but n=61, one window, and H1 is soft (+1.21%, PF 1.11). Not
adopted into the book. Paper/OOS next, same shelf as the FVG-midline
sleeve. If it dies there too, trendline-retests are buried for good.

---

## Honest limits

- One instrument (gold H1), one 2-year bullish window, in-sample design.
- Claim #30's wedge taxonomy is strict; a looser shape matcher would find
  more wedges but the drift verdict (−0.25 vs +0.25) has enough margin that
  shape-tweaking will not rescue it.
- Claim #31's rearm rule (2 bars away) is the honest reading of "retest";
  a no-rearm variant would degenerate into a delayed breakout entry.
- The #29b body-touch cell (n=67) is suggestive, not proven.
