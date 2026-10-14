"""GOD-CE salvage: measure trap-intelligence components standalone per playbook.

For each playbook (sweep_reversal, fvg_midline, trendline_retest):
  1. Subset MAP: baseline book trades split by trap state at the entry bar
     (boost / oppose / neutral) — where each component fires, how those
     trades did. Fake-breakout incidence reported (single-bar event, too
     sparse for a filter; it feeds the SFP latch).
  2. Book-level filter variants (the phase-routing lesson: a cell can look
     good while the book gets worse — only whole-book improvement counts):
       - boost_only   : keep candidates with direction-aligned trap boost
       - sfp_only     : keep candidates with direction-aligned SFP active
       - beyond_only  : keep candidates with direction-aligned beyond-FVG reclaim
       - ifvg_only    : keep candidates with direction-aligned iFVG test
       - nobait       : drop candidates with direction bait flag
       - nooppose     : drop candidates with opposed trap boost active
       - nolure       : drop candidates with direction lure flag
  3. Books use each playbook's REAL booking rule: the anchor's
     live_threshold=60 on signal.score (the 199-trade canary); sleeves
     unfiltered (threshold 0). Trap row read at the candidate's ENTRY bar
     (harness convention); trap memory is 12 bars.

A component earns promotion only if its variant improves the WHOLE
sequential book. No thresholds tuned — trap params are pine/godce defaults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiment_salvage import (
    engine_gated_candidates,
    load_ctx_and_candidates,
    run_filter_experiment,
)
from goldenswing.backtest.harness import book_metrics, build_book
from goldenswing.features.traps import compute_trap_features

# Playbook booking thresholds (mirror the confluence books).
THRESHOLDS = {"sweep_reversal": 60.0, "fvg_midline": 0.0, "trendline_retest": 0.0}
# Canary: baseline books must reproduce the validated numbers.
CANARY = {
    "sweep_reversal": {"trades": 199, "return_pct": 28.92},
    "fvg_midline": {"trades": 48, "return_pct": 34.61},
    "trendline_retest": {"trades": 61, "return_pct": 15.40},
}


def _dir_flags(row: pd.Series, direction: int) -> dict:
    long = direction > 0
    return {
        "boost": bool(row["trap_boost_long"] if long else row["trap_boost_short"]),
        "oppose": bool(row["trap_boost_short"] if long else row["trap_boost_long"]),
        "bait": bool(row["bait_long"] if long else row["bait_short"]),
        "lure": bool(row["lure_long"] if long else row["lure_short"]),
        "sfp": bool(row["sfp_bear_active"] if long else row["sfp_bull_active"]),
        "beyond": bool(row["beyond_bull"] if long else row["beyond_bear"]),
        "ifvg": bool(row["ifvg_demand"] if long else row["ifvg_supply"]),
        "fake": bool(row["fake_brk_dn"] if long else row["fake_brk_up"]),
    }


def playbook_book(candidates, threshold: float, years: float):
    return build_book([c.outcome for c in candidates], threshold)


def subset_map(candidates, threshold: float, trap_df) -> dict:
    """Baseline book trades grouped by trap state at entry bar."""
    book = playbook_book(candidates, threshold, 0.0)
    groups: dict[str, list[float]] = {"boost": [], "oppose": [], "neutral": []}
    bait_rs, lure_rs = [], []
    fake_n = 0
    for o in book:
        if o.entry_index >= len(trap_df):
            continue
        f = _dir_flags(trap_df.iloc[o.entry_index], o.signal.direction)
        if f["fake"]:
            fake_n += 1
        if f["boost"]:
            groups["boost"].append(o.r_net)
        elif f["oppose"]:
            groups["oppose"].append(o.r_net)
        else:
            groups["neutral"].append(o.r_net)
        if f["bait"]:
            bait_rs.append(o.r_net)
        if f["lure"]:
            lure_rs.append(o.r_net)

    def summ(rs):
        rs = np.array(rs, dtype=float)
        return {"n": int(len(rs)),
                "avg_r": float(rs.mean()) if len(rs) else 0.0,
                "win_rate": float(np.mean(rs > 0) * 100) if len(rs) else 0.0,
                "total_r": float(rs.sum()) if len(rs) else 0.0}

    out = {k: summ(v) for k, v in groups.items()}
    out["bait"] = summ(bait_rs)
    out["lure"] = summ(lure_rs)
    out["fake_brk_trades"] = fake_n
    return out


VARIANTS = {
    "boost_only": lambda c, f: f["boost"],
    "sfp_only": lambda c, f: f["sfp"],
    "beyond_only": lambda c, f: f["beyond"],
    "ifvg_only": lambda c, f: f["ifvg"],
    "nobait": lambda c, f: not f["bait"],
    "nooppose": lambda c, f: not f["oppose"],
    "nolure": lambda c, f: not f["lure"],
}


def run_all(trap_df=None) -> dict:
    ctx, playbooks, raw = load_ctx_and_candidates()
    gated = engine_gated_candidates(ctx, playbooks)
    if trap_df is None:
        print("computing trap features ...", flush=True)
        trap_df = compute_trap_features(ctx.df)
    assert len(trap_df) == len(ctx.df)

    results: dict = {}
    for name in ("sweep_reversal", "fvg_midline", "trendline_retest"):
        cands = gated[name]
        thr = THRESHOLDS[name]
        base_book = playbook_book(cands, thr, ctx.years)
        base = book_metrics(base_book, ctx.years)
        # canary
        can = CANARY[name]
        assert base["trades"] == can["trades"], \
            f"CANARY {name}: {base['trades']} != {can['trades']}"
        assert abs(base["return_pct"] - can["return_pct"]) < 0.5, \
            f"CANARY {name}: {base['return_pct']} != {can['return_pct']}"
        res = {"n_candidates": len(cands), "baseline": base,
               "subset_map": subset_map(cands, thr, trap_df), "variants": {}}
        for v, pred in VARIANTS.items():
            def keep(cand, trap_row, _p=pred):
                if trap_row is None:
                    return True
                return bool(_p(cand, _dir_flags(trap_row, cand.direction)))

            # trap row at ENTRY bar (harness convention)
            kept = [c for c in cands
                    if keep(c, trap_df.iloc[c.entry_index]
                            if c.entry_index < len(trap_df) else None)]
            filt_book = playbook_book(kept, thr, ctx.years)
            f_ = book_metrics(filt_book, ctx.years)
            cut_rs = np.array([c.outcome.r_net for c in cands if c not in kept])
            res["variants"][v] = {
                "n_kept": len(kept), "n_cut": len(cands) - len(kept),
                "base_trades": base["trades"], "filt_trades": f_["trades"],
                "base_ret": base["return_pct"], "filt_ret": f_["return_pct"],
                "base_pf": base["pf"], "filt_pf": f_["pf"],
                "base_avg_r": base["avg_r"], "filt_avg_r": f_["avg_r"],
                "cut_avg_r": float(cut_rs.mean()) if len(cut_rs) else 0.0,
                "cut_win": float(np.mean(cut_rs > 0) * 100) if len(cut_rs) else 0.0,
            }
        results[name] = res
    return results


def main() -> dict:
    results = run_all()
    for name, res in results.items():
        b = res["baseline"]
        print(f"\n{'=' * 72}\n{name}  (candidates: {res['n_candidates']}, "
              f"threshold: {THRESHOLDS[name]})")
        print(f"baseline: n={b['trades']} ret={b['return_pct']:.2f}% "
              f"PF={b['pf']:.2f} win={b['win_rate']:.1f}% avgR={b['avg_r']:+.3f}")
        print("-- subset map (baseline book, trap state at entry bar) --")
        for k, s in res["subset_map"].items():
            if k == "fake_brk_trades":
                print(f"  fake-breakout trades in book: {s}")
                continue
            print(f"  {k:8s} n={s['n']:4d} avgR={s['avg_r']:+.3f} "
                  f"win={s['win_rate']:.1f}% totalR={s['total_r']:+.1f}")
        print("-- filter variants (whole-book, re-built) --")
        for v, d in res["variants"].items():
            print(f"  {v:12s} kept={d['n_kept']:4d} cut={d['n_cut']:4d} | "
                  f"n {d['base_trades']}->{d['filt_trades']} | "
                  f"ret {d['base_ret']:+.2f}%->{d['filt_ret']:+.2f}% | "
                  f"PF {d['base_pf']:.2f}->{d['filt_pf']:.2f} | "
                  f"cut avgR={d['cut_avg_r']:+.3f} win={d['cut_win']:.0f}%")
    return results


if __name__ == "__main__":
    main()
