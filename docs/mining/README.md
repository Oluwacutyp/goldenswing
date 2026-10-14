# Knowledge Mine

The research backbone of the Cutyp trading bot. Every source we mine — PDFs,
books, papers, videos, comment sections, forums — lands here as a claim ledger.
Nothing enters the system on authority; everything earns its place by backtest.

## How it works

1. **Mine**: a source goes in `sources/` as its own file (or a link + extraction notes
   when the source can't be stored).
2. **Extract**: every testable claim gets a row in `CLAIMS.md` — the claim, the
   required context (how/where/when), the source, and its status.
3. **Test**: claims move `untested → testing → adopted | killed | context-only`.
4. **Build**: adopted claims become features, sleeves, or gates in the system.

## Claim statuses

- `untested` — extracted, not yet backtested
- `testing` — experiment running
- `adopted` — backtested, earned a place (book improved or graded positive)
- `killed` — backtested, rejected (with the reason: no edge, structural flaw, costs)
- `context-only` — true as description, not as a trade (stays on charts, out of the score)
- `candidate` — graded positive but needs paper trading / out-of-sample first

## The rule

A claim without a context is not a claim. "Hammer candles work" is not testable.
"Hammer candles at fresh demand, with-trend, in killzone" is. Every claim must
state its how/where/when before it gets tested.
