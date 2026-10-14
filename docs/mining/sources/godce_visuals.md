# Mining note: GOD Conviction Engine visuals → our alerts & /status

Date: 2026-10-11. Source: the original ~4000-line GOD CE Pine (visual layer
reconstructed from the owner's brief; the slim build at
`~/workspace/your_files/GOD-CE-SLIM.pine` has every table stripped, so the
visuals below describe the ORIGINAL, not the slim file).

Reference state of our alerts: `goldenswing/alerts/model.py` (`build_alert_text`
— header, entry/stop/targets, risk distance, then `narrator.py`'s event-ordered
story, then `Backing:` factor list, then optional forecast note) sent as a
Telegram photo caption (1024-char hard cap, photo falls back to text) plus a
'signal' chart style. There is no /status command yet.

Philosophy: TradingView sells screen real estate; we sell **phone screen real
estate at 2 AM**. Every STEAL below is judged by: does it survive a 1024-char
caption on a phone, and does it answer "what happened / what do I do / what's
the risk" faster than what we already have.

---

## 1. Execution HUD (28 rows × 2 cols, bottom-right)

State banner (SCANNING / LONG ACTIVE / SHORT ACTIVE / BUY-SELL SIGNAL / LAST
exit), mode+TF, regime+vol-regime, session+caution flags, macro (DXY bias), MTF
bias, market structure, Wyckoff phase, MACD/RSI/VWAP/sentiment/footprint/candle
pattern rows, liquidity+PD rows (sweep state, premium/discount, EQ pools), zone
status (nearest OB/FVG/RBS/SBR), confluence buy% vs sell% with threshold,
ASOC win-rate/threshold, position sizing (units + $ risk), live R:R + TP1 state,
Hydra trail phase, bias agreement (aligned/counter/mixed), setup grade
(A+ GOD / A APEX / B PRIME / C WEAK), catalyst "why" string, Asian range,
vol band position, trap intelligence (traps sprung / gap reclaimed / bait /
lure), broker economics (breakeven move, cost gate, day R, trades-vs-cap,
circuit-breaker banner).

**Verdict: mostly DECORATION — but mine it for two STEALs.**

- DECORATION: the HUD itself. 28 rows on a chart is the opposite of phone-first
  legibility; ~20 of its rows restate what the radar, analysis table, and
  narrative label already said. This is the exhibit for the anti-lessons
  section. Never a 28-row caption.
- **STEAL — the state banner as a caption header line.** One line at the top
  that says what mode the system is in: `🔎 SCANNING` / `🟢 LONG ACTIVE`
  / `🟥 SHORT ACTIVE` / `⚡ BUY SIGNAL` / `🏁 LAST exit`. Our caption header
  currently jumps straight to direction+score. A state line gives the owner
  instant orientation before reading the trade math. Cheap, one line.
- **STEAL — setup grade as a one-token suffix on the header.** Our `★ GOLDEN
  SWING` marker is binary. The HUD's A+/A/B/C grade is a better dial:
  `LONG XAUUSD — score 82 · A+` reads instantly and composes with the existing
  score. Implementation: grade from score+confluence agreement tiers; no new
  machinery, just a label band.

**DECORATION with one-line reasons** (the rest of the HUD): MACD/RSI/VWAP/
sentiment/footprint/candle rows — already our `Backing:` list, no second
home needed; vol band position — our ATR-context strings cover it; broker
economics / day-R / trades-vs-cap — belongs in /status (below), not a signal
caption; live R:R + Hydra phase — position-tracking surface, not signal
surface (we'd only steal this if/when we ship live position updates).

---

## 2. Confluence radar (14 rows, top-right)

Per-factor rows (Macro, Trend Stack, HMA baseline, Elastic EMA, Multi-TF bias,
Liquidity/EQ, SMC zone/breaker, ADX strength, Wyckoff, Score-vs-threshold
EXECUTE/WAIT, Ichimoku daily, H&S, Trap intelligence) each with a status text
and a green/red pass-fail cell.

**Verdict: DECORATION as a table; STEAL as a principle.**

- DECORATION: a 14-row per-factor matrix is a desktop artifact. Pasting it
  into a caption burns ~400 of our 1024 characters to restate the `Backing:`
  list in a grid. It answers "what did each sensor say" — but our caption
  already answers "what matters for this signal" via the narrator + Backing.
- **STEAL — the disagreement row.** The single most valuable radar cell is the
  one that says "biases AGREE / COUNTER / MIXED". Our caption never surfaces
  counter-evidence as a first-class citizen; the narrator has a one-caveat
  slot but no explicit agreement state. Proposal: one line after the narration,
  e.g. `🧭 Bias: aligned (12/14 factors)` or `⚠️ Mixed (9/14) — HTF counter`.
  That is the radar's job compressed to one line, and it's the honest line a
  real trader wants before sizing.

---

## 3. Analysis table (26 rows, top-left)

MTF alignment block (intraday/short/medium-term strings + alignment strength),
MACD/RSI/VWAP/Wyckoff rows, LEVELS block (3 resistances + 3 supports with
★–★★★ cluster-weight ratings, daily-pivot ⭕ marker, LAST price above/below
pivot), auto-generated Preference ("Upside prevails above X, targets Y & Z"),
Alternative (invalidation line), Comment (bias + confluence + driver + cloud +
risk flags + threshold), Ichimoku daily/TF, H&S daily/TF, Auto-Read.

**Verdict: DECORATION — this whole table IS the /status command spec.**

Don't put it in captions. But it is the blueprint for a future `/status`
command: MTF block, S&R with cluster weights, Preference/Alternative pair.
Two concrete STEALs:

- **STEAL — the Preference / Alternative pair for /status.** The pine's best
  sentence template is: *"Upside prevails above {level}, targets {Y} & {Z}"*
  with a matching *"Alternative: below {X}, look for {Y}"*. Our captions have
  entry/stop/targets but never state the trade as a plain-language plan with
  an explicit invalidation branch. Proposal for `/status` output section:
  `📈 Preference:` one line, `🔀 Alternative:` one line. Also worth appending
  a shortened Preference line to the caption itself (one line, ~60 chars).
- **STEAL — cluster-weighted S&R for /status.** The ★–★★★ cluster-weight
  rating on S&R levels is a compact honesty device: it says how much we
  trust the level. Our LEVELS block equivalent in /status should be 3R/3S
  with weight marks (e.g. `R2 2,712 ★★★`), plus the ⭕ daily-pivot marker
  and LAST-vs-pivot position. Cheap to render from features we already
  compute; nothing like it exists in our alert surface today.

**DECORATION (one-line reasons):** MTF alignment prose — our narrator covers
trend context; MACD/RSI/VWAP rows — Backing list covers them; Ichimoku/H&S
rows — detector-factor territory, not a status surface; Auto-Read — overlaps
the narrator's job.

---

## 4. Narrative voice (single chart label)

One label: mode + win-rate + symbol + conviction; in-trade adds direction,
live R, BE state, soft-invalidation warning, counter-bias flag, catalyst.
Otherwise a context line picked by a fixed priority stack: circuit breaker >
caution > spike shield > V-reversal > pattern > weekend > asia break > H&S
break > traps > gap reclaim > bait warning > buy/sell ready > compression >
volatile > premium/discount.

**Verdict: STEAL — the priority stack is the single best idea in the whole
visual layer.**

- **STEAL — priority-ordered single context line.** Our narrator writes a
  good 60-word story, but it has no notion of *what matters most right now*.
  The pine's label resolves competing context into ONE line by priority
  (safety flags first: circuit breaker > caution > spike shield; then
  opportunity: V-reversal > pattern > session breaks; then state: traps >
  compression > vol). Proposal: add a `_headline(signal)` to `narrator.py`
  that returns one ≤80-char line chosen by a fixed priority list over the
  parsed facts — e.g. `🛡️ Spike shield — thin weekend liquidity` or
  `⚡ V-reversal forming off the sweep low` — placed as the caption's first
  line under the header. It composes with the existing story instead of
  replacing it, and it gives the "glance at the phone" tier that the HUD
  tried and failed to give.
- **STEAL — the in-trade label fields for future position alerts.** When we
  ship live position updates (not yet), the in-trade set — direction, live R,
  BE state, soft-invalidation warning, counter-bias flag, catalyst — is the
  exact payload. File it under "when positions.py grows teeth".
- DECORATION: win-rate in the label. A claimed win-rate without the sample
  window is the kind of number the pine gets away with on a chart and we
  must not put in a caption. Our ASOC threshold work is the honest version;
  until it's measured, no win-rate strings.

---

## 5. Chart layer

Confluence heatmap background (tiered green/red opacity by score-vs-threshold),
entry/SL/TP boxes + entry line, signal labels (LONG/SHORT with units, entry,
SL, TPs, why-string, grade, score, mode; TP1/2/3 hit labels; stops; conviction
invalidation), trap labels (🪤 traps, FAKE BREAK, GAP RAID, BEYOND GAP reclaim,
iFVG flips), sweep/structure labels.

**Verdict: STEAL selectively — this is the chart, and the chart is already
our strongest surface.**

- **STEAL — keep the heatmap; it maps to our 'signal' chart style.** We
  already ship a phone-first signal chart. The tiered-opacity heatmap by
  score-vs-threshold is worth adopting as the background of that chart
  (it reads at a glance where the engine had conviction, without reading
  anything). Implementation note: map our score/threshold to alpha bands.
- **STEAL — the signal label's compact spec block.** LONG/SHORT + units +
  entry/SL/TPs + why + grade + score + mode in ONE label is the chart
  equivalent of our caption header — check our chart annotation against
  this list; anything missing (units, grade, why) should be added if it
  fits legibly at phone resolution.
- **STEAL — trap labels.** 🪤 trap markers, FAKE BREAK, GAP RAID, iFVG flips
  as chart annotations are high-signal low-clutter: they mark *events the
  engine reacted to*, which is exactly what a phone chart needs. Our
  sweep/structure labels already overlap here; add the trap set.
- DECORATION: entry/SL/TP boxes + lines — standard, we already draw these;
  quantum gravity / dark-pool / liquidity-heat lines — the slim build kept
  their *scoring* and dropped their *drawings* for a reason; don't re-add
  chart spaghetti.

---

## Anti-lessons: what made the pine's visuals cluttered

1. **Three tables saying the same thing three ways.** HUD (28 rows), radar
   (14 rows), analysis table (26 rows) overlap on regime, MTF, confluence,
   and levels. On a chart you can glance past redundancy; in a caption every
   repeated fact costs characters against the 1024 cap. Rule: one fact, one
   home — caption header (state), headline (priority context), narrator
   (story), Backing (evidence), /status (reference).
2. **Desktop density != information.** 28 rows bottom-right is legible on a
   27" monitor and invisible on a phone. We already learned phone-first
   legibility (the 'signal' chart style); the HUD is the counterexample to
   keep on the wall.
3. **Scores without denominators.** Pass/fail cells and win-rate strings with
   no sample window look authoritative and mean nothing. Our ASOC threshold
   work is the honest version — never ship the pine's bare win-rate label.
4. **Static priority.** The pine's context stack is a fixed ordering; fine.
   What we must not copy is *showing all of it at once*. The stack exists to
   pick ONE line, not to justify showing fourteen.

## Steal list (ranked)

1. Priority-stack headline line → new `_headline()` in `narrator.py`,
   first line under the caption header. (Biggest win: glanceable context.)
2. State banner line → caption header (`🔎 SCANNING` / `🟢 LONG ACTIVE` /
   `⚡ BUY SIGNAL` / …).
3. Setup grade token (A+ / A / B / C) on the header, next to the score.
4. Bias-agreement line (`🧭 aligned 12/14` / `⚠️ mixed — HTF counter`) after
   the narration — the radar's one honest row.
5. Preference / Alternative pair → /status command (and a one-line
   Preference in the caption).
6. Cluster-weighted S&R (★–★★★, ⭕ pivot, LAST-vs-pivot) → /status LEVELS
   block.
7. Heatmap-by-threshold background + trap labels → 'signal' chart style.
8. In-trade label payload (live R, BE state, soft-invalidation, counter-bias,
   catalyst) → future position alerts, when `positions.py` grows teeth.

## Explicitly not stolen

- The 28-row HUD, the 14-row radar, the 26-row analysis table as caption
  content — all decoration; their best rows are extracted above as single
  lines.
- Win-rate label strings without measured denominators.
- Score-only pass/fail grids that duplicate the Backing list.
- Chart spaghetti the slim build deliberately dropped (gravity lines,
  dark-pool lines, heat boxes) — scoring kept, drawings stay dead.
