# Cost-gate filter experiment (GOD-CE salvage, Workstream B)

**Data:** GC=F H1 (gold futures), 11,426 bars, 2024-10-11 -> 2026-10-09 (1.99y) — the same window every Phase-9 book is measured on.

**The gate** (pine-faithful port of the GOD-CE broker-cost gate, `goldenswing/confluence.py::cost_gate_ok`, from GOD-CE-SLIM.pine §entries):

- `cost_px = entry * 4 / 10000` (round-trip cost in price terms)
- leg 1: `risk = |entry - invalidation| >= 2 * cost_px`
- leg 2: `tp1_r - cost_px / risk >= 0.6`, where `tp1_r` is the nearest liquidity-pool target ahead in the trade direction; fallback when no target lies ahead: fvg_midline -> 2.0 (its validated fixed-2R exit), otherwise 1.5 (the pine's configured tp1RR default)
- entry = open of the entry bar (the harness fill price); the gate never raises — missing data fails closed with a reason

**Doctrine:** the gate is a filter, not a score tweak. `cost_bps=4` and `min_net_tp1_r=0.6` are the PINE DEFAULTS — nothing was tuned against these results.  This report is what the defaults do.

**Method:** baseline = the playbook's existing book (engine context gates applied, score >= live_threshold, greedy chronological via `harness.build_book` — verified against the known targets first).  Cost-gated = gate failures removed BEFORE booking, book re-built greedily (the phase-routing lesson: removing trades re-times everything downstream).

## Per-playbook results

### sweep_reversal

Eligible candidates: 351 (kept 322, cut 29).

| Book | Trades | Total R | PF | Win% | Avg R | MaxDD% | Return% |
|---|---|---|---|---|---|---|---|
| baseline | 199 | +27.3 | 1.25 | 45.7 | +0.137 | 10.38 | +28.92 |
| cost-gated | 200 | +7.4 | 1.06 | 42.0 | +0.037 | 13.41 | +5.81 |

**Gate failure legs** (among the 29 cut): `net-tp1-r-too-low` ×28, `risk-too-small` ×1

**Cut-trade profile** (standalone r_net of the removed trades — did the gate cut losers or winners?):

- n=29, win 58.6%, avg +0.397R, total +11.5R, PF 1.93
- buckets: <=-1R ×12, -1R..0R ×0, 0R..+1R ×4, >+1R ×13
- exit reasons: stop ×12, timeout ×3, trail ×14

**Verdict: HURTS** — return +28.92% -> +5.81% (-23.11pt), PF 1.25 -> 1.06 (-0.19), MaxDD 10.38% -> 13.41%.

### fvg_midline

Eligible candidates: 194 (kept 147, cut 47).

| Book | Trades | Total R | PF | Win% | Avg R | MaxDD% | Return% |
|---|---|---|---|---|---|---|---|
| baseline | 48 | +30.3 | 2.47 | 58.3 | +0.631 | 5.10 | +34.61 |
| cost-gated | 37 | +18.0 | 2.02 | 54.1 | +0.485 | 4.10 | +19.16 |

**Gate failure legs** (among the 47 cut): `net-tp1-r-too-low` ×47

**Cut-trade profile** (standalone r_net of the removed trades — did the gate cut losers or winners?):

- n=47, win 74.5%, avg +1.175R, total +55.2R, PF 5.36
- buckets: <=-1R ×12, -1R..0R ×0, 0R..+1R ×0, >+1R ×35
- exit reasons: stop ×12, target ×35

**Verdict: HURTS** — return +34.61% -> +19.16% (-15.45pt), PF 2.47 -> 2.02 (-0.44), MaxDD 5.10% -> 4.10%. (But see mechanism notes: this is an entry-definition artifact — at the sleeve's true limit entry the gate cuts 0/194 and the book is unchanged.)

### trendline_retest

Eligible candidates: 67 (kept 66, cut 1).

| Book | Trades | Total R | PF | Win% | Avg R | MaxDD% | Return% |
|---|---|---|---|---|---|---|---|
| baseline | 61 | +15.1 | 1.43 | 45.9 | +0.248 | 9.32 | +15.40 |
| cost-gated | 60 | +16.2 | 1.48 | 46.7 | +0.271 | 9.32 | +16.69 |

**Gate failure legs** (among the 1 cut): `risk-too-small` ×1

**Cut-trade profile** (standalone r_net of the removed trades — did the gate cut losers or winners?):

- n=1, win 0.0%, avg -1.104R, total -1.1R, PF 0.00
- buckets: <=-1R ×1, -1R..0R ×0, 0R..+1R ×0, >+1R ×0
- exit reasons: stop ×1

**Verdict: IMPROVES** — return +15.40% -> +16.69% (+1.29pt), PF 1.43 -> 1.48 (+0.05), MaxDD 9.32% -> 9.32%.

## Cross-playbook summary

| Playbook | Δ trades | Δ total R | Δ PF | Δ win% | Δ MaxDD% | Δ return% | Verdict |
|---|---|---|---|---|---|---|---|
| sweep_reversal | +1 | -19.8 | -0.19 | -3.7 | +3.03 | -23.11 | HURTS |
| fvg_midline | -11 | -12.4 | -0.44 | -4.3 | -0.99 | -15.45 | HURTS |
| trendline_retest | -1 | +1.1 | 0.05 | +0.8 | +0.00 | +1.29 | IMPROVES |

## Mechanism notes (why the gate behaves this way)

**fvg_midline — the HURTS verdict is an entry-definition artifact, not
a real filter effect.** The sleeve fills at a RESTING LIMIT
(`signal.entry_ref`, e.g. 2668.2); the experiment's mandated entry
(open of the fill bar, e.g. 2673.1) can sit ~1R away from it. The
gate therefore evaluated a phantom entry — a trade "already half
done" with `tp1_r ≈ 0.46` — and rejected 47 fills that went on to
bank +55.2R standalone (35 hit the 2R target). Diagnostic, same
defaults, gate evaluated at the TRUE limit entry (`signal.entry_ref`):
**cuts 0 of 194, book unchanged (48 / +34.61% / PF 2.47).** At its
real fill the gate is a no-op on this sleeve — nothing to adopt,
nothing genuinely hurt.

**sweep_reversal — the HURTS verdict is real (entry definition is
correct here: the anchor really fills at open[t+1]).** Leg 2 prices
the trade off the NEAREST liquidity pool (`tp1_r ≈ 0.5–0.6` on the
cut set), but the anchor's Phase-2 exits (50% banked at the first
pool, remainder TRAILED after +1.5R) routinely bank R well beyond
the nearest pool — 14 of the 29 cut trades exited via `trail`,
several above +1.4R. The pine's gate was built for a fixed ladder
(TP1 at 1.5R, TP2 at 2R); its leg-2 assumption does not transfer to
trailing exits. The cut set was +0.397R avg / PF 1.93 — the gate cut
winners.

**trendline_retest — 1 cut, a genuine `risk-too-small` loser.**
Marginal improvement on n=1 is not an adoption case; it is a
curiosity until more data.

## What didn't survive

- The GOD-CE engine as a whole stays killed as tested
  (experiments/GODCE_ENGINE.md) — this salvage measured ONLY its
  broker-cost gate, standalone, on real playbooks.
- The gate is NOT adopted on any playbook: sweep_reversal (hurts on
  the correct entry — fixed-ladder assumption vs trailing exits),
  fvg_midline (no-op at the true limit entry; the measured HURTS is
  an entry-definition artifact), trendline_retest (n=1, not evidence).
- Any playbook where the whole book does not improve keeps its
  existing firing rules: the gate is not adopted there.

## What's still weak

- One window, one instrument, one strongly bullish stretch for gold — nothing here is validated out of sample.
- The gate uses the pine's flat 4 bps round-trip; real costs vary by session, spread regime and broker.
- Cut-trade profiles are standalone simulations — descriptive of what the gate removed, not a counterfactual book.
- Entry-price definition: the experiment prices every candidate at open[entry_index] (the harness fill convention). That matches the detector-based playbooks' real fills, but NOT the fvg_midline sleeve's resting-limit fill — see the mechanism notes. Any future filter work on limit-fill sleeves must price at `signal.entry_ref`.
