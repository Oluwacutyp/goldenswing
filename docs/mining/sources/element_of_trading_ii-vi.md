# Element Of Trading (Doctor Pips, 2021) — Volumes II–VI

User-provided PDFs, read 2026-10-11. Beginner retail-forex education series.
Files: `docs/mining/inbox/_E__Element_Of_Trading_{II,III,IV,V,VI}_*.pdf`.

## Volume II — Forex Broker (8 pages)
Broker infrastructure: market maker vs STP vs ECN, account types (micro/standard),
MT4/MT5 platforms, account-opening requirements. No trading mechanics.
Relevance: broker choice matters for the future MT5 deployment (ECN/STP, no
strategy restrictions), but nothing for the strategy ledger.

## Volume III — Candlestick Patterns (11 pages)
OHLC basics, body = buying/selling pressure, small body = indecision.
Five textbook patterns, definitions only, no entry/stop/target rules:
- Bearish/bullish engulfing: reversal at the end of a trend
- Hammer: bullish reversal at end of downtrend (long lower shadow)
- Shooting star: bearish reversal at end of uptrend (long upper shadow)
- Morning/evening star: 3-candle reversal patterns
All covered by (and weaker than) our candle context map (#21).

## Volume IV — Market Structure Basics (16 pages)
- Wyckoff market cycle: accumulation (sideways, smart money collects) →
  mark-up (breaks resistance, rises) → distribution (releases holdings, momentum
  fades) → decline. Volume confirms the move.
- HH/HL/LL/LH trend definitions; buy low / sell high.
- SNR as zones; strength = retest frequency + timeframe (H1+ strong, M30-down
  weak); fresh zones preferred; "always look left for the nearest SNR".
- SNR flip: resistance-becomes-support (RBS), support-becomes-resistance (SBR).
- **Breakout = body candle CLOSE beyond prior high/low. Wick-only is not a
  breakout — called the "shadow killer"** (a fake-out that traps traders).
  Breakouts need further confirmation to be valid.
Context for: our BOS/CHoCH confirmation pillar (#1), gated-zone freshness,
level flips, timeframe weighting. Nothing new to test except nothing.

## Volume V — Trendlines + Chart Patterns (12 pages)
- Trendlines drawn high-to-high (resistance) / low-to-low (support).
- **Construction choice: shadow-to-shadow (low risk, less floating) vs
  body-to-body (high risk, more floating).** → claim #29 (untested)
- Major trendline (big TF: D1 for intraday, MN for swing) vs minor trendline
  (entry TF: H1/H4 for intraday). Confluence: enter where trendline meets SNR.
- Trendline SNR entries: buy trendline support, sell trendline resistance,
  sell trendline SBR, buy trendline RBS; a trendline break = cut the trade.
- **Trendline breakouts come in 3 forms: clean breakout, breakout-and-retest,
  fakey (fake breakout).** RBS/SBR trendline retest: after breakout, enter on
  the retest touch of the flipped trendline. → claim #31 (untested)
- Chart patterns: rising/falling wedges = continuation; head & shoulders /
  inverse = reversal, best combined with SNR. H&S covered by #12 (killed);
  **wedges untested** → claim #30.

## Volume VI — Money & Risk Management (4 pages)
Drawdown math, 2%-vs-10% risk per trade, don't trade against the trend,
emotional discipline. No mechanics; our risk layer (never-round-up sizing,
daily/weekly limits, drawdown halt) is strictly more sophisticated.

## Honest assessment
These are beginner primers, not practitioner gold. The series' real value for
the mine is confirmation, not discovery: body-close breakouts, SNR flips,
fresh-zone preference, and trend-direction discipline all rhyme with what the
backtests already proved. Three claims survive as genuinely untested (#29–#31);
the rest is `existing` or `context-only`. The user's PDF pipeline is worth more
than these volumes — the deep practitioner material is still to come.
