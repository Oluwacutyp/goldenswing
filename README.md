# GoldenSwing

Not a pile of strategies. One **aware system** that reads the chart and the
market itself — structure, liquidity, regime, session — and identifies a
**Golden Swing**: a *big* move (expected expansion ≥ ~2× ATR over the next
~72 H1 bars / 3 days) that is *backed* by verifiable confluence. A move
without backing is not a Golden Swing, no matter how big it later gets.
The system's edge is supposed to live in that discrimination — and the
backtest grades exactly that claim (see `BACKTEST.md`).

## Architecture

```
OHLCV bars
   │
   ▼
MarketReader  (goldenswing/reader/)      — the awareness layer
   │  one causal pass → MarketState per bar:
   │  • Structure   fractal swings, BOS / CHoCH, trend state
   │  • Liquidity   pools of resting stops (equal highs/lows, swing
   │                extremes), sweep events (wick through + close back)
   │  • ICT levels  fair value gaps (3-candle), order blocks (last
   │                opposite candle before displacement), premium /
   │                discount position in the dealing range
   │  • Regime      ATR percentile (compressed / normal / expanded),
   │                ATR-normalised momentum, displacement strength
   │  • Sessions    Asia / London / overlap / New York, ICT killzones,
   │                running session range
   ▼
GoldenSwingDetector  (goldenswing/golden/)
   │  scores each bar 0–100 across five backing families:
   │  structure alignment · liquidity event · displacement ·
   │  room-to-run · regime support · session timing
   │  hard gates: minimum room to the next liquidity pool, sane
   │  structural invalidation distance, a real directional thesis
   │  output: direction, score, human-readable backing factors,
   │  entry zone, invalidation level, targets (liquidity pools ahead)
   ▼
Backtest harness  (goldenswing/backtest/)
      next-bar entry, invalidation stop, target = nearest opposing
      pool (2R floor), stop wins ties, 72-bar time stop, $0.45/oz cost,
      1% risk — plus a shadow book that grades every score band.
```

**Causality is a hard contract.** Every event is timestamped at the bar it
became *knowable* (a fractal swing at bar `i` confirms at `i+n`; an FVG at
its third candle's close). `tests/test_no_lookahead.py` proves truncating
the series cannot change earlier states.

## Quick start

```bash
pip install -e .            # or just use the source tree
python scripts/backtest.py  # fetches GC=F H1 (2y, cached in data/), writes BACKTEST.md
pytest                      # unit + causality tests
```

Data note: GC=F is COMEX front-month gold futures — a proxy for spot
XAUUSD. Spreads, rolls and the daily settlement gap differ from a spot
feed; results should be read with that in mind.

## Roadmap (later phases — not in this repo yet)

- **Phase 2:** ensemble ML forecaster — 3-day big-move probability model
  (gradient boosting + DRL position/exit agent) layered on the reader's
  state features, retraining on the system's own trade history.
- **Phase 3:** execution — MT5 Expert Advisor bridge (brain on Linux/AWS,
  EA inside a terminal on demo or real Exness), paper mode by default.
- **Phase 4:** Telegram alerts — every signal with the setup *drawn* on a
  chart image (levels shaded, entry/SL/TP zones) and inline
  Execute / Skip / Paper buttons.

## Phase 1 status

MarketReader + GoldenSwing detector + backtest harness are implemented
and tested. Results — including the score-discrimination test of the core
concept — are in `BACKTEST.md`. The honest summary lives there, not here.
