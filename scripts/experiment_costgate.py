#!/usr/bin/env python3
"""Cost-gate filter experiment (Workstream B, part 2) — writes
experiments/COSTGATE.md.

Measures the GOD-CE broker-cost gate STANDALONE on the three Phase-9
playbooks, through the reusable salvage harness
(``scripts/experiment_salvage.py``):

- Baseline: the playbook's EXISTING book — engine context gates
  applied (session scope, setup tripwire, hunting ground), score >=
  the playbook's live_threshold, greedy chronological via
  ``harness.build_book``.  Verified against the known targets
  (sweep_reversal canary 199/+28.92%/PF 1.25, fvg_midline 48/+34.61%,
  trendline_retest 61/+15.40%) before anything is measured.
- Cost-gated: candidates failing ``confluence.cost_gate_ok`` removed
  BEFORE booking; the book is re-built greedily (the phase-routing
  lesson: removing trades re-times everything downstream — never
  subtract cut trades from the baseline book).

DOCTRINE: the gate is a filter, not a score tweak.  ``cost_bps=4.0``
and ``min_net_tp1_r=0.6`` are the PINE DEFAULTS — no tuning against
these results.  This script reports what the defaults do.

Importable: ``from experiment_costgate import run_costgate_experiment,
main`` (scripts dir on sys.path).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from experiment_salvage import (costgate_keep_fn, engine_gated_candidates,
                                load_ctx_and_candidates,
                                run_filter_experiment)
from goldenswing.confluence import ConfluenceConfig, cost_gate_ok
from goldenswing.playbooks.base import Candidate, EngineContext
from run_confluence_book import CANARY, SHADOW_TARGETS

REPORT_PATH = _REPO_ROOT / "experiments" / "COSTGATE.md"


# ------------------------------------------------------------ gate run
def gate_with_reasons(ctx: EngineContext, candidates: list[Candidate],
                      cfg: ConfluenceConfig | None = None
                      ) -> list[tuple[Candidate, bool, str]]:
    """Apply the cost gate to every candidate, keeping the reason.

    Entry price is the harness fill price: open of the entry bar
    (``ctx.df["open"].iloc[candidate.entry_index]``).  Never raises —
    ``cost_gate_ok`` fails closed, and an out-of-range entry bar is a
    fail-closed cut with its own reason.
    """
    cfg = cfg or ConfluenceConfig()
    opens = ctx.df["open"].to_numpy(dtype=float)
    out: list[tuple[Candidate, bool, str]] = []
    for cand in candidates:
        i = cand.entry_index
        if i < 0 or i >= len(opens):
            out.append((cand, False, "no-entry-bar: entry_index out of range"))
            continue
        ok, reason = cost_gate_ok(cand.outcome.signal, float(opens[i]),
                                  cfg, playbook=cand.playbook)
        out.append((cand, ok, reason))
    return out


def run_costgate_experiment(playbook_name: str, ctx: EngineContext,
                            candidates: list[Candidate],
                            cfg: ConfluenceConfig | None = None) -> dict:
    """Cost-gate experiment for one playbook, through the salvage
    harness.

    Uses the SHARED ``costgate_keep_fn`` (same function object as the
    harness demo — the two paths agree by construction) and returns
    the harness result dict plus:

    - ``gate_reasons``: failure-leg counts among the cut candidates
      (``risk-too-small`` / ``net-tp1-r-too-low`` / fail-closed reasons),
    - ``cfg_used``: the (untuned) gate parameters.
    """
    cfg = cfg or ConfluenceConfig()
    gated = gate_with_reasons(ctx, candidates, cfg)
    verdict = {id(c): ok for c, ok, _ in gated}
    keep = costgate_keep_fn(ctx, cfg)

    # Belt and braces: the shared keep_fn must agree with the recorded
    # per-candidate verdicts (same gate, same entry price).
    for cand, ok, _reason in gated:
        assert keep(cand, None) == ok, \
            f"keep_fn/gate_with_reasons disagree on {cand.playbook} " \
            f"bar {cand.signal_index}"

    res = run_filter_experiment(playbook_name, ctx, candidates, keep)
    legs: dict[str, int] = {}
    for _cand, ok, reason in gated:
        if not ok:
            leg = reason.split(":")[0]
            legs[leg] = legs.get(leg, 0) + 1
    res["gate_reasons"] = legs
    res["cfg_used"] = {"cost_bps": cfg.cost_bps,
                       "min_net_tp1_r": cfg.min_net_tp1_r}
    return res


# ------------------------------------------------------------ report
def _fmt(x: float, nd: int = 2) -> str:
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    return f"{x:.{nd}f}"


def _total_r(m: dict) -> float:
    return m["avg_r"] * m["trades"]


def _row(label: str, m: dict) -> str:
    return (f"| {label} | {m['trades']} | {_total_r(m):+.1f} | "
            f"{_fmt(m['pf'])} | {m['win_rate']:.1f} | "
            f"{m['avg_r']:+.3f} | {m['max_dd_pct']:.2f} | "
            f"{m['return_pct']:+.2f} |")


def verdict(b: dict, f: dict) -> tuple[str, str]:
    """Whole-book verdict.  Rule (documented, not tuned): the book
    IMPROVES if equity return is up >= 1pt with PF not materially
    worse; HURTS if return is down >= 1pt; otherwise NEUTRAL.  PF and
    MaxDD are reported alongside — the rule is a summary, not the
    evidence."""
    d_ret = f["return_pct"] - b["return_pct"]
    d_pf = (f["pf"] if f["pf"] != float("inf") else 99.0) - \
           (b["pf"] if b["pf"] != float("inf") else 99.0)
    if d_ret >= 1.0 and d_pf >= -0.05:
        s = "IMPROVES"
    elif d_ret <= -1.0:
        s = "HURTS"
    else:
        s = "NEUTRAL"
    why = (f"return {b['return_pct']:+.2f}% -> {f['return_pct']:+.2f}% "
           f"({d_ret:+.2f}pt), PF {_fmt(b['pf'])} -> {_fmt(f['pf'])} "
           f"({d_pf:+.2f}), MaxDD {b['max_dd_pct']:.2f}% -> "
           f"{f['max_dd_pct']:.2f}%")
    return s, why


def build_report(results: dict[str, dict], cfg: ConfluenceConfig) -> str:
    L = [
        "# Cost-gate filter experiment (GOD-CE salvage, Workstream B)",
        "",
        "**Data:** GC=F H1 (gold futures), 11,426 bars, 2024-10-11 -> "
        "2026-10-09 (1.99y) — the same window every Phase-9 book is "
        "measured on.",
        "",
        "**The gate** (pine-faithful port of the GOD-CE broker-cost gate, "
        "`goldenswing/confluence.py::cost_gate_ok`, from GOD-CE-SLIM.pine "
        "§entries):",
        "",
        f"- `cost_px = entry * {cfg.cost_bps:g} / 10000` "
        "(round-trip cost in price terms)",
        "- leg 1: `risk = |entry - invalidation| >= 2 * cost_px`",
        "- leg 2: `tp1_r - cost_px / risk >= "
        f"{cfg.min_net_tp1_r:g}`, where `tp1_r` is the nearest "
        "liquidity-pool target ahead in the trade direction; fallback "
        "when no target lies ahead: fvg_midline -> 2.0 (its validated "
        "fixed-2R exit), otherwise 1.5 (the pine's configured tp1RR "
        "default)",
        "- entry = open of the entry bar (the harness fill price); the "
        "gate never raises — missing data fails closed with a reason",
        "",
        "**Doctrine:** the gate is a filter, not a score tweak. "
        f"`cost_bps={cfg.cost_bps:g}` and "
        f"`min_net_tp1_r={cfg.min_net_tp1_r:g}` are the PINE DEFAULTS — "
        "nothing was tuned against these results.  This report is what "
        "the defaults do.",
        "",
        "**Method:** baseline = the playbook's existing book (engine "
        "context gates applied, score >= live_threshold, greedy "
        "chronological via `harness.build_book` — verified against the "
        "known targets first).  Cost-gated = gate failures removed "
        "BEFORE booking, book re-built greedily (the phase-routing "
        "lesson: removing trades re-times everything downstream).",
        "",
        "## Per-playbook results",
        "",
    ]
    for name in ("sweep_reversal", "fvg_midline", "trendline_retest"):
        r = results[name]
        b, f = r["baseline"], r["filtered"]
        cp = r["cut_profile"]
        v, why = verdict(b, f)
        L += [
            f"### {name}",
            "",
            f"Eligible candidates: {r['n_candidates']} "
            f"(kept {r['n_kept']}, cut {r['n_cut']}).",
            "",
            "| Book | Trades | Total R | PF | Win% | Avg R | MaxDD% | Return% |",
            "|---|---|---|---|---|---|---|---|",
            _row("baseline", b),
            _row("cost-gated", f),
            "",
            "**Gate failure legs** (among the "
            f"{r['n_cut']} cut): " +
            (", ".join(f"`{leg}` ×{n}"
                       for leg, n in sorted(r["gate_reasons"].items()))
             if r["gate_reasons"] else "none — gate cut nothing"),
            "",
            "**Cut-trade profile** (standalone r_net of the removed "
            "trades — did the gate cut losers or winners?):",
            "",
            f"- n={cp['n']}, win {cp['win_rate']:.1f}%, avg "
            f"{cp['avg_r']:+.3f}R, total {cp['total_r']:+.1f}R, "
            f"PF {_fmt(cp['pf'])}",
            f"- buckets: " + ", ".join(
                f"{k} ×{v}" for k, v in cp["buckets"].items()),
            f"- exit reasons: " + ", ".join(
                f"{k} ×{v}" for k, v in
                sorted(cp["exit_reasons"].items())) if cp["exit_reasons"]
            else "- exit reasons: —",
            "",
            f"**Verdict: {v}** — {why}.",
            "",
        ]
    L += [
        "## Cross-playbook summary",
        "",
        "| Playbook | Δ trades | Δ total R | Δ PF | Δ win% | Δ MaxDD% | "
        "Δ return% | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ("sweep_reversal", "fvg_midline", "trendline_retest"):
        r = results[name]
        b, f = r["baseline"], r["filtered"]
        v, _ = verdict(b, f)
        L.append(
            f"| {name} | {f['trades'] - b['trades']:+d} | "
            f"{_total_r(f) - _total_r(b):+.1f} | "
            f"{_fmt(f['pf'] - b['pf']) if f['pf'] != float('inf') and b['pf'] != float('inf') else 'n/a'} | "
            f"{f['win_rate'] - b['win_rate']:+.1f} | "
            f"{f['max_dd_pct'] - b['max_dd_pct']:+.2f} | "
            f"{f['return_pct'] - b['return_pct']:+.2f} | {v} |")
    L += [
        "",
        "## What didn't survive",
        "",
        "- The GOD-CE engine as a whole stays killed as tested "
        "(experiments/GODCE_ENGINE.md) — this salvage measured ONLY its "
        "broker-cost gate, standalone, on real playbooks.",
        "- Any playbook where the whole book does not improve keeps its "
        "existing firing rules: the gate is not adopted there.",
        "",
        "## What's still weak",
        "",
        "- One window, one instrument, one strongly bullish stretch for "
        "gold — nothing here is validated out of sample.",
        "- The gate uses the pine's flat 4 bps round-trip; real costs "
        "vary by session, spread regime and broker.",
        "- Cut-trade profiles are standalone simulations — descriptive "
        "of what the gate removed, not a counterfactual book.",
        "",
    ]
    return "\n".join(L)


# ------------------------------------------------------------ main
def _check_baseline(name: str, m: dict, trades: int, ret: float,
                    pf: float | None = None) -> None:
    assert m["trades"] == trades, \
        f"BASELINE DRIFT {name}: trades {m['trades']} != {trades}"
    assert abs(m["return_pct"] - ret) <= 0.01, \
        f"BASELINE DRIFT {name}: return {m['return_pct']:.2f} != {ret}"
    if pf is not None:
        assert abs(m["pf"] - pf) <= 0.01, \
            f"BASELINE DRIFT {name}: PF {m['pf']:.2f} != {pf}"


def main(argv: list[str] | None = None) -> dict:
    cfg = ConfluenceConfig()  # pine defaults — untuned, by doctrine
    ctx, playbooks, _raw = load_ctx_and_candidates()
    gated = engine_gated_candidates(ctx, playbooks)

    results: dict[str, dict] = {}
    for name, pb in playbooks.items():
        eligible = [c for c in gated[name]
                    if c.outcome.signal.score >= pb.live_threshold]
        res = run_costgate_experiment(name, ctx, eligible, cfg)
        results[name] = res
        b = res["baseline"]
        print(f"== {name}: eligible {res['n_candidates']}, "
              f"kept {res['n_kept']}, cut {res['n_cut']} "
              f"(legs: {res['gate_reasons']}) ==")
        print(f"  baseline: {_row('base', b)}")
        print(f"  gated:    {_row('gate', res['filtered'])}")
        v, why = verdict(b, res["filtered"])
        print(f"  verdict: {v} — {why}")

    # Baselines must reproduce the known books before any claim is made.
    b = results["sweep_reversal"]["baseline"]
    _check_baseline("sweep_reversal", b, CANARY["trades"],
                    CANARY["return_pct"], CANARY["pf"])
    for name in ("fvg_midline", "trendline_retest"):
        tgt = SHADOW_TARGETS[name]
        _check_baseline(name, results[name]["baseline"], tgt["trades"],
                        tgt["return_pct"])
    print("baselines reproduce the known books (canary + shadow targets)")

    REPORT_PATH.write_text(build_report(results, cfg))
    print(f"\n[written to {REPORT_PATH}]")
    return results


if __name__ == "__main__":
    main()
