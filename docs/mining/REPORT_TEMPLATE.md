# Experiment Report Template

Every experiment report follows this structure. Copy it, fill it in,
keep the honest-limits section — it is not optional.

```markdown
# Experiment: <name>

**Source:** <where the claim came from>
**Claim:** <the precise mechanical claim being tested>

**Data:** <instrument, timeframe, bar count, date range>.
<costs, risk, horizon, exit conventions>.

**Method:** <exactly what was built and how the variants differ>.
<what was pre-committed vs fitted — knobs are pre-committed, not fitted>.

## Results

<table(s)>: trades, win%, PF, avg R, return, MaxDD, Sharpe, halves.

## Context map

The goal is a MAP of where the strategy performed well, not just a
headline verdict.  Report the signal set through these splits
(`scripts/measure_context_maps.py` does this mechanically):

- session (asia / london / overlap / newyork / off)
- killzone vs off-killzone
- volatility regime (compressed / normal / expanded)
- trend (up / down / range)
- HTF bias (with / against / no bias)
- AMD phase
- day of week

Per cell: n, win%, avg R, total R, both halves.  Name the cells where
the R concentrates AND the cells that are decisively bad (n ≥ 25,
avg R < -0.05, negative in both halves).  Thin cells (n < 25) are
shown, never gated.

## Verdict

<adopted | candidate | killed as tested | killed as tested — context variant open>

- If killed: state the TESTED SCOPE (market, timeframe, sessions,
  exit system).  A kill is "killed as tested", never universal.
- If the source named a specific context that was NOT tested
  (a session window, a timeframe, a regime), mark it
  "killed as tested — context variant open" and name the variant.
- If candidate: what would promote it (OOS, paper, sample size).

## Honest limits

- <window, instrument, in-sample vs OOS>
- <sample sizes where thin>
- <what was NOT tested>
```

## The ledger

Log the claim in `docs/mining/CLAIMS.md` with status:
`untested | testing | adopted | killed as tested |
killed as tested — context variant open | context-only | candidate`.
The Evidence column carries the headline numbers AND the tested scope.
