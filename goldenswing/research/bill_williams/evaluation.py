"""Bill Williams — the family's OWN promotion
evaluation (Pass 8, SPEC §6.6), run as a research
line over its own journal stream only.

The bar is §6's common bar, the same shape the live
families' evaluations use (one evaluator voice, one
rule): paired outcomes, degenerate-R excluded,
medians and hit rates, a floor of MIN_PAIRED per arm,
a negative take median fails, inverted discrimination
can never read ready.  Two honest differences from
the live §G shape, both stated on every snapshot:

- the pairing is by the strategy's own invalidation
  (the simulated exit — opposite fractal / lips cross
  / hold horizon), not a fixed +4h horizon;
- the abstain arm is the population the AO/AC organ
  alone rejected (candidates passing the Alligator +
  teeth gates whose break bar failed AO/AC, simulated
  identically) — so the headline discrimination grades
  AO/AC specifically, and the remaining organs are
  graded by the ablation table below it.

The ablation table reports every variant's paired
outcomes on the same history (full / no_aoac /
no_teeth / no_alligator), plus confirmation lift:
the full rule's takes split by the Gator phase and
the BW-MFI class at entry.  An organ that adds
nothing shows it here — that showing is the family's
ablation deliverable.

It promotes NOTHING and can promote nothing: this
package declares no route and has no order path.
A ``promotion_ready`` verdict is a recommendation to
the owner; any future promotion of this research
line is a separate owner decision taken on this
evaluation, outside this program.
"""
from __future__ import annotations

import statistics

__all__ = [
    "FAMILY_LABEL",
    "MIN_PAIRED",
    "RECORD_TYPE",
    "VERDICT_INSUFFICIENT",
    "VERDICT_NOT_READY",
    "VERDICT_READY",
    "evaluate_records",
    "render_lines",
]

RECORD_TYPE = "bw_evaluation"
FAMILY_LABEL = "bill williams research"

VERDICT_INSUFFICIENT = "insufficient_data"
VERDICT_NOT_READY = "not_ready"
VERDICT_READY = "promotion_ready"

#: The common bar's floor: paired outcomes per arm
#: before any verdict but insufficient_data may read.
MIN_PAIRED = 20

#: A confirmation-lift cell thinner than this is
#: reported as thin, never as a lift.
MIN_LIFT_CELL = 5

_WINDOW_BASIS = (
    "bill williams research journal (bw_signal records) "
    "only.  Outcomes are paired by the strategy's own "
    "invalidation simulation (opposite fractal / lips "
    "cross / hold horizon), not a fixed horizon; "
    "degenerate-R pairs are excluded and counted; "
    "signals with no definable stop are unpaired and "
    "counted as no_stop.  The abstain arm is the "
    "AO/AC-rejected candidate population (Alligator + "
    "teeth gates passed, AO/AC failed at the break "
    "bar), simulated identically.")


def _arm_stats(rs: list[float]) -> dict:
    n = len(rs)
    hits = sum(1 for r in rs if r > 0)
    return {"paired": n, "hits": int(hits),
            "hit_rate": (round(hits / n, 4) if n else None),
            "median_r": (round(float(statistics.median(rs)), 4)
                         if n else None)}


def _fold(signals: list[dict]) -> dict:
    """Fold bw_signal records into per-variant stats,
    the full rule's two arms, and the lift cells."""
    variants: dict[str, dict] = {}
    arms = {"take": [], "abstain": []}
    counts = {"signals": 0, "no_stop": 0, "degenerate": 0}
    lift = {"gator_eating": [], "gator_other": [],
            "mfi_green": [], "mfi_fake": [], "mfi_other": []}
    for rec in signals:
        oc = rec.get("outcome") or {}
        variant = rec.get("variant") or "full"
        slot = variants.setdefault(
            variant, {"r": [], "no_stop": 0, "degenerate": 0,
                      "signals": 0})
        slot["signals"] += 1
        counts["signals"] += 1
        if oc.get("no_stop"):
            slot["no_stop"] += 1
            counts["no_stop"] += 1
            continue
        if oc.get("degenerate") or not oc.get("paired"):
            slot["degenerate"] += 1
            counts["degenerate"] += 1
            continue
        r = oc.get("r")
        if r is None:
            slot["degenerate"] += 1
            counts["degenerate"] += 1
            continue
        slot["r"].append(float(r))
        if variant == "full":
            arm = rec.get("arm") or "take"
            if arm in arms:
                arms[arm].append(float(r))
            if arm == "take":
                read = rec.get("read") or {}
                key = "gator_eating" \
                    if read.get("gator_phase") == "eating" \
                    else "gator_other"
                lift[key].append(float(r))
                mfi = read.get("mfi_class")
                key = {"green": "mfi_green",
                       "fake": "mfi_fake"}.get(mfi, "mfi_other")
                lift[key].append(float(r))
    return {"variants": variants, "arms": arms,
            "counts": counts, "lift": lift}


def _verdict(take: dict, abstain: dict) -> tuple[str, str]:
    if take["paired"] == 0 and abstain["paired"] == 0:
        return VERDICT_INSUFFICIENT, (
            "no paired outcomes yet — the research line "
            "has journaled no simulatable signals since "
            "its window opened")
    floor = MIN_PAIRED
    if take["paired"] < floor or abstain["paired"] < floor:
        return VERDICT_INSUFFICIENT, (
            f"paired outcomes below the readable floor of "
            f"{floor} per arm: takes {take['paired']}/{floor}, "
            f"abstains {abstain['paired']}/{floor} "
            f"(degenerate-R and no-stop pairs excluded, "
            f"never counted)")
    take_med = float(take["median_r"])
    abstain_med = float(abstain["median_r"])
    if take_med < 0:
        return VERDICT_NOT_READY, (
            f"measured and failing: take median "
            f"{take_med:+.2f}R is negative over "
            f"{take['paired']} paired outcomes — the full "
            f"rule's typical trade loses")
    inverted = abstain_med > take_med or \
        float(abstain["hit_rate"]) > float(take["hit_rate"])
    if inverted:
        return VERDICT_NOT_READY, (
            f"measured and failing: INVERTED discrimination "
            f"— the AO/AC-rejected abstains (median "
            f"{abstain_med:+.2f}R, hit "
            f"{abstain['hit_rate']:.0%}) outperform the "
            f"takes (median {take_med:+.2f}R, hit "
            f"{take['hit_rate']:.0%}); the AO/AC organ is "
            f"subtracting, not confirming")
    return VERDICT_READY, (
        f"takes (median {take_med:+.2f}R, hit "
        f"{take['hit_rate']:.0%}, {take['paired']} paired) "
        f"beat the AO/AC-rejected abstains (median "
        f"{abstain_med:+.2f}R, hit "
        f"{abstain['hit_rate']:.0%}, "
        f"{abstain['paired']} paired) over the floor — "
        f"PROMOTION READY as a recommendation only: this "
        f"research line has no route and no order path, "
        f"and any promotion is a separate owner decision "
        f"taken on this evaluation")


def evaluate_records(records: list[dict]) -> dict:
    """One evaluation snapshot over a record list (the
    family's journal stream, or any slice of it)."""
    signals = [r for r in records
               if r.get("type") == "bw_signal"]
    folded = _fold(signals)
    take = _arm_stats(folded["arms"]["take"])
    abstain = _arm_stats(folded["arms"]["abstain"])
    verdict, reason = _verdict(take, abstain)
    variants = {}
    for name, slot in sorted(folded["variants"].items()):
        st = _arm_stats(slot["r"])
        st.update({"signals": slot["signals"],
                   "no_stop": slot["no_stop"],
                   "degenerate": slot["degenerate"]})
        variants[name] = st
    lift = {}
    for key, rs in folded["lift"].items():
        st = _arm_stats(rs)
        st["thin"] = st["paired"] < MIN_LIFT_CELL
        lift[key] = st
    return {
        "family": FAMILY_LABEL,
        "verdict": verdict,
        "verdict_reason": reason,
        "take": take,
        "abstain": abstain,
        "variants": variants,
        "confirmation_lift": lift,
        "counts": folded["counts"],
        "floors": {"min_paired_per_arm": MIN_PAIRED},
        "basis": _WINDOW_BASIS,
        "binds": False,
    }


def _r(v) -> str:
    return "—" if v is None else f"{float(v):+.2f}R"


def _pct(v) -> str:
    return "—" if v is None else f"{float(v):.0%}"


_VERDICT_PLAIN = {
    VERDICT_READY: "PROMOTION READY (recommendation only)",
    VERDICT_NOT_READY: "not ready",
    VERDICT_INSUFFICIENT: "not enough data yet",
}


def render_lines(snapshot: dict) -> list[str]:
    """The written evaluation, plain words first: the
    verdict and the numbers behind it, the ablation
    table, the confirmation lifts, and the provenance
    caveats (one evaluator voice, the family's own
    journal)."""
    snap = snapshot or {}
    if not snap:
        return ["bill williams research (Pass 8): no "
                "evaluation has run yet — verdicts land "
                "here once the research journal is graded"]
    lines = [
        f"bill williams research (Pass 8): "
        f"{_VERDICT_PLAIN.get(snap.get('verdict'), snap.get('verdict'))}"
        f" — {snap.get('verdict_reason') or ''}",
        f"  takes median {_r(snap.get('take', {}).get('median_r'))}"
        f" (hit {_pct(snap.get('take', {}).get('hit_rate'))}, "
        f"{snap.get('take', {}).get('paired', 0)} paired) vs "
        f"AO/AC-rejected abstains median "
        f"{_r(snap.get('abstain', {}).get('median_r'))}"
        f" (hit {_pct(snap.get('abstain', {}).get('hit_rate'))}, "
        f"{snap.get('abstain', {}).get('paired', 0)} paired)"
        f" · floor {MIN_PAIRED} per arm"
        f" · {snap.get('counts', {}).get('no_stop', 0)} no-stop"
        f" · {snap.get('counts', {}).get('degenerate', 0)}"
        f" degenerate excluded",
        "  ablation (same history, one organ removed):",
    ]
    for name, st in (snap.get("variants") or {}).items():
        lines.append(
            f"    {name}: median {_r(st.get('median_r'))}"
            f" (hit {_pct(st.get('hit_rate'))}, "
            f"{st.get('paired', 0)} paired of "
            f"{st.get('signals', 0)} signals)")
    lift = snap.get("confirmation_lift") or {}
    def _lift_line(label, key):
        st = lift.get(key) or {}
        tail = " (thin — not a lift)" if st.get("thin") else ""
        return (f"    {label}: median {_r(st.get('median_r'))}"
                f" over {st.get('paired', 0)} paired{tail}")
    lines.append("  confirmation lift (full rule's takes):")
    lines.append(_lift_line("gator eating at entry", "gator_eating"))
    lines.append(_lift_line("gator other at entry", "gator_other"))
    lines.append(_lift_line("BW-MFI green at entry", "mfi_green"))
    lines.append(_lift_line("BW-MFI fake at entry", "mfi_fake"))
    lines.append(
        "  BW-MFI is weighted on tick volume (MT5's volume "
        "column is a trade count), never traded volume — "
        "the caveat rides every MFI record in the journal")
    lines.append(
        "  this line is research only: no route, no order "
        "path, nothing on the live path reads it")
    return lines
