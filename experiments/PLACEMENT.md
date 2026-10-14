# Placement Review — the same objects in practitioner roles (+ inverse check)

**Doctrine.** The owner's correction: our kills tested each item in ONE role
(usually: standalone entry trigger, all contexts). "Everything has a place —
we just put them where they don't belong." This review re-examines every
candle/pattern kill through the placement lens, with three questions each:

1. **Killed in what role?** — the exact scope of the kill, so we stop
   over-generalizing it.
2. **Where might it belong?** — practitioner logic first: confirmation
   inside a setup, invalidation information, regime input, session/regime
   scoping.
3. **Measured** — the 2–3 most promising placements backtested; every
   directional probe also tested INVERTED (same trigger, opposite
   direction, mirrored stop).

**Inverse doctrine.** A consistently-wrong bull signal is a bear signal —
but "real" means it survives both window halves AND parameter variants,
not a sign flip on the in-sample number. Asymmetry is the null hypothesis
(bearish pins at supply worked while bull pins didn't; bear_hid backs
shorts while bear_reg never positive). Inverse claims face the same
honesty bar as normal claims.

**Conventions** (all probes): 2y GC=F H1 (11,426 bars, Oct 2024–Oct 2026),
$0.45/oz cost, causal (no bar beyond the signal bar consulted), entry at
next-bar open, structural stop + 0.25 ATR, fixed 2R target, 72-bar horizon.
`fail_bars` swept {8, 12, 16}: a placement is only real if stable across
all three. One signal per (bar, direction) — a trader takes one position
(this dedup matters: see T3).

---

## The four probes

| Probe | Practitioner placement tested | Inverse tested |
|---|---|---|
| T1 failed-pattern | Turtle Soup: pattern breaks down, reclaims through neckline → trade the failure (trapped shorts fuel it) | Fade the failure: trade with the pattern's bias at the reclaim |
| T2 pin-violation | Stop-hunt: pin extreme closed through → rejection failed → trade the violation | Trade with the pin's rejection at the violation bar |
| T3 zone-breaker | ICT breaker: JeaFx zone touched, then closed through against the thesis → trade the continuation | Fade the breaker: trade back with the original zone thesis |
| T4 doji-expansion | ICT "indecision marks zones": 2+ doji compression → trade the range expansion | Fade the expansion |

## Measured results (deduped, fail_bars=12; sweep confirmed stable)

| Probe | n | Win% | Avg R | PF | H1 | H2 | Verdict |
|---|---|---|---|---|---|---|---|
| T1 bias | 212 | 34.9 | −0.002 | 1.00 | −0.025 | +0.022 | KILLED as placement |
| T1 inv | 212 | 33.5 | −0.064 | 0.91 | −0.108 | −0.021 | dead |
| T2 bias | 1369 | 36.5 | **+0.042** | 1.06 | +0.053 | +0.031 | WEAK CANDIDATE (context) |
| T2 inv | 651 | 17.7 | −1.323 | 0.19 | −1.370 | −1.276 | catastrophic |
| T3 bias | 7 | 57.1 | +0.687 | 2.57 | −0.043 | +1.235 | UNPROVEN (n=7) |
| T3 inv | 1 | 0.0 | −1.618 | 0.00 | — | −1.618 | dead |
| T4 bias | 92 | 39.1 | +0.080 | 1.13 | −0.023 | +0.183 | KILLED (H1 fails) |
| T4 inv | 44 | 38.6 | −0.257 | 0.74 | +0.476 | −0.990 | dead |
| Wedge bdir drift | 17 | 35.3 | +0.084 ATR | — | — | — | KILLED (see below) |

Stability across fail_bars 8/12/16: T1 bias −0.022/−0.002/+0.021 (flat
everywhere); T2 bias +0.034/+0.042/+0.038 with both halves positive at
every setting; T4 bias +0.054/+0.080/+0.094 but H1 ≤ 0 at every setting;
T3 identical 7 signals at every setting (the setups are what they are).

### The T3 dedup lesson (read this)

The first T3 run printed **55 signals, +0.609R, PF 2.32** — exciting and
wrong. Seven setups failing at the same bar are not seven trades; they
are one market event counted seven times. After one-signal-per-(bar,
direction) dedup: **7 unique events**. The breaker placement is not
disproven — it is **unproven**. This is why the dedup rule exists, and
why the first number is reported here: as a warning about how good a
double-counted backtest can look.

### The T2 inverse lesson (read this too)

Fading the pin violation loses **−1.32R per trade, PF 0.19**, in both
halves, at every parameter setting. The violation direction is weakly
right (+0.04R); the fade is catastrophically wrong. This is the
asymmetry doctrine measured: the edge, such as it is, runs exactly one
way, and the opposite is not a milder version — it is a different,
much worse trade. (Mechanically: the inverse short's stop sits inside
the violation bar's range, so many inverses die immediately.)

---

## Per-kill three-question treatment

### #9 — JeaFx zone retest

1. **Killed in role:** *entry trigger* — resting limit at the zone
   (midline or edge) after sweep→zone→BoS confirmation. Structural:
   0/63 pool-target wins (edge), 75% same-bar stopouts (mid), and the
   setup-level control (enter at confirmation, no retest wait) graded
   −0.105R. The sequence itself carries no edge on gold H1.
2. **Candidate placements:**
   - (a) The retest *mechanic* as entry — already answered by #31:
     it works on trendlines (+15.40%), not zones. Different mechanic,
     different verdict. Correctly not inherited.
   - (b) The retest *failure* as invalidation → **T3 tested**: the ICT
     breaker. Result: UNPROVEN (7 unique events after dedup; was
     55 double-counted). Not a candidate, not a kill — an open thin
     file. Would need OOS events to judge.
   - (c) First-touch-only retests — already the tested condition
     (fresh zones). No new test warranted.
3. **Verdict:** kill stands as entry trigger. Breaker placement open
   but unproven (n=7). Ledger #9 scope-noted.

### #12 — H&S / double top / double bottom

1. **Killed in role:** *scored entry family* (`patterns_ctx`) inside the
   Combo A book, even with context gating (major level + displacement):
   +19.04% vs +25.62% anchor — dilutes.
2. **Candidate placements:**
   - (a) **Level-strength input** (Bulkowski/practitioner): a double top
     is two rejections of a level — the pattern marks a *strong level*,
     not a trade. Placement: feed pattern-confirmed levels into the
     level/AOC machinery as strength votes. System placement, not a
     backtest — noted for the reader, not measured here.
   - (b) **Measured-move targets** (textbook): pattern height projects a
     target. Placement: *exit/target* logic, not entry. The pattern's
     target field already exists on PatternInstance; noted as
     exit-machinery input, not measured here.
   - (c) **Failed-pattern continuation** (Turtle Soup) → **T1 tested**:
     flat both ways (bias −0.002R, inverse −0.064R, n=212, stable
     across params). The failure carries no signal on gold H1.
3. **Verdict:** kill stands as entry family AND as failure-trade. The
   patterns' honest places are (a) level-strength context and (b)
   measured-move targets — both as inputs to other machinery, never as
   the trigger. Ledger #12 scope-noted.

### #21 — pins (killed as standalone trigger, −0.083R over 3,129)

1. **Killed in role:** *standalone entry trigger* (next-bar-open, fixed
   2R) across all contexts. Best cell (at_zone|against, +0.058R, n=104)
   doesn't earn a place.
2. **Candidate placements:**
   - (a) **Wick as level definer** (ICT rejection block: "the long-wick
     candle IS the level") — chart-context placement. Already encoded:
     pools are wick-defined. Noted, not re-tested.
   - (b) **Pin-wick violation as continuation** (stop-hunt) → **T2
     tested**: +0.042R, PF 1.06, n=1,369, both halves positive at all
     three parameter settings. WEAK CANDIDATE as context (same shelf
     as the imbalance magnet: real but gentle, ~1 SE). The inverse
     (−1.32R, PF 0.19) confirms the direction is load-bearing.
   - (c) **Bearish pin at supply with full ICT conventions** — already
     adopted (#10, +$1,755/31). The placement that works was never the
     standalone trigger; it was the pin *inside a setup*.
3. **Verdict:** kill stands as standalone trigger. Violation-continuation
   is a weak context candidate (#44). Ledger #21 scope-noted.

### #21 — dojis (killed as standalone trigger, −0.095R)

1. **Killed in role:** *standalone entry trigger*. Bulkowski's coin-flip
   confirmed.
2. **Candidate placements:**
   - (a) **Compression marker → expansion trade** (ICT "indecision marks
     zones") → **T4 tested**: +0.080R overall but H1 ≤ 0 at every
     parameter setting (−0.023/−0.008/−0.090). Fails the both-halves
     bar. KILLED as placement.
   - (b) **Zone/compression marker** — chart-context placement (where
     to look for the coil), not a trigger. Noted.
3. **Verdict:** kill stands, including as expansion trigger. Ledger #21
   scope-noted.

### #21 — harami (mined as coin flip, never separately measured)

1. **Killed in role:** not measured standalone; mined at ~53% reversal
   (Bulkowski) — a coin flip before costs.
2. **Candidate placements:** same shelf as the doji — indecision context
   inside compression, never the trigger. No new test warranted: the
   doji-expansion probe (T4) is the closest measurable placement and it
   failed. Noted as context-only.
3. **Verdict:** stays context-only. No pardon without a distinct
   measurable placement.

### #27 — engulfing as sweep-reversal trigger

1. **Killed in role:** *replacement* for displacement+CHoCH confirmation:
   −21.94% book, both halves negative. Not earlier (3.93 vs 3.71 bars),
   just 3.3× looser.
2. **Candidate placements:**
   - (a) **Gated standalone sleeve** (reversal at pool, counter-trend)
     — already the adopted placement (+0.107R, n=549). This kill was
     always about the trigger role, never the candle.
   - (b) **Inverse (fade engulfings)** — mapped, not re-tested: the
     candle map's with-trend engulf cells (−0.037R overall, −0.009 at
     pool) ARE the fade, measured across 821 signals. Flat/negative.
3. **Verdict:** kill stands as trigger; sleeve placement stands; inverse
   already measured dead. Ledger #27 scope-noted. This is the template
   the whole review follows.

### #30 — wedges (killed as continuation entry, −0.250 ATR drift)

1. **Killed in role:** *continuation entry* in the prior-trend direction:
   in-trend breakout drift −0.250 ATR vs +0.252 unconditional base.
2. **Candidate placements:**
   - (a) **Trade the breakout direction** (the wedge as compression,
     resolution is the information) → **tested**: +0.084 ATR over 17
     breakouts — but the 14 in-trend breakouts still drift −0.250 and
     the +0.084 is carried entirely by 3 counter-trend events (+1.642).
     n=17 cannot carry it; the in-trend component (the placeable part)
     is unchanged. KILLED as placement.
   - (b) The counter-trend footnote (+1.642, n=3) stays a footnote:
     noted, unproven, untestable at this sample.
3. **Verdict:** kill stands, including breakout-direction placement. The
   wedge joins H&S in the graveyard with no open placement. Ledger #30
   scope-noted.

---

## Inverse summary (all probes, fail_bars=12)

| Probe | Bias | Inverse | Asymmetry read |
|---|---|---|---|
| T1 failed pattern | −0.002R (flat) | −0.064R (flat-neg) | No signal either way |
| T2 pin violation | +0.042R (weak) | −1.323R (catastrophic) | Edge runs one way only |
| T3 zone breaker | +0.687R (n=7, unproven) | −1.618R (n=1) | Too thin both ways |
| T4 doji expansion | +0.080R (H1 fails) | −0.257R (neg) | No |
| Engulfing (mapped) | +0.107R at pool | −0.037R w/trend | Known lopsided |

No inverse carries stable signal. The lopsidedness the owner named is
confirmed as the norm: where anything works, it works in one direction,
and the opposite is worse than flat — it is a different, losing trade.

## What changes in the system

- **T2 → context input** (weak): pin-wick violation as a small
  continuation-context vote, same shelf as the imbalance magnet.
  Never a trigger, never scored directionally beyond the violation
  direction.
- **Patterns → level-strength + measured-move inputs**: PatternInstance
  already carries `target` (measured move) and level identity; the
  placement review's system recommendation is to consume patterns as
  context (a twice-rejected level is a strong level; a pattern's height
  is a target candidate), never as entries.
- **T3 → open thin file**: 7 unique breaker events cannot judge the
  placement. If OOS data produces more events, re-test; until then it
  is neither candidate nor kill.
- **Ledger**: every kill below carries its tested scope; new claims
  #44–#48 record the placement probes.

## Honest limits

- One instrument (gold H1), one 2-year window, in-sample throughout.
- T3's 7 events and the wedge's 17 breakouts are too thin for verdicts
  beyond "not proven".
- T2's +0.042R is ~1 standard error: weak context, not a sleeve.
- The dedup rule (one signal per bar+direction) is load-bearing for T3;
  without it the review would have printed a false candidate.
- Practitioner placements (a)/(b) for #12 and the doji zone-marker role
  are system-design notes, not measured claims.
