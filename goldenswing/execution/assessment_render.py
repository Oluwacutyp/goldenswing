"""Render ONE funnel evaluation record as a MARKET ASSESSMENT.

The visible market-reading milestone (owner, 2026-10-23):
"print signals" means a readable, structured assessment the
owner can look at and challenge — not a bare BUY/SELL and not a
score.  Phase 2 made every evaluation record carry an
``assessment`` block (funnel.assessment_block: verdict in the
owner's vocabulary, plain-language reason, structure read, HTF
context, itemized evidence with detector + source-bar
timestamps, correlation groups, competing scenarios, capability
markers).  This module is the PRESENTATION layer for that
block: a pure function from record to plain text.

Honesty contract (test-pinned):

- the renderer prints ONLY what the record carries.  A section
  the route did not produce renders as an explicit
  "not produced" / "none recorded" line — never invented prose,
  never a silently omitted section;
- a record with NO assessment block renders an explicit
  not-produced statement, not a fabricated read;
- SHADOW records (§G range evaluation routes, observed roster
  scans) are labelled SHADOW in the header and can never render
  as a signal — shadow routes hold no order authority;
- capability markers (absent / not-integrated census facts) are
  rendered as explicit lines, exactly as the record carries
  them;
- supporting and contradictory evidence render in SEPARATE
  groups; evidence items carry their correlation group, so
  correlated observations never read as stacked independent
  confirmations.
"""
from __future__ import annotations

#: Verdict words the renderer will ever print (the owner's
#: vocabulary, funnel.ASSESSMENT_VERDICTS).  An unknown verdict
#: on a record renders verbatim but is never upgraded.
_VERDICT_EMOJI = {
    "signal": "🟢",
    "watch": "👀",
    "no-setup": "➖",
    "rejected": "⛔",
    "expired": "⌛",
}


def is_shadow_record(record: dict) -> bool:
    """True when the record came from a route with NO order
    authority (§G range shadow evaluations / observed scans).
    Read from the record's own facts: the shadow flag on the
    evidence block, a shadow decision result, or the
    path.shadow_route reason code."""
    if not isinstance(record, dict):
        return False
    ev = record.get("evidence") or {}
    if isinstance(ev, dict) and ev.get("shadow") is True:
        return True
    dec = record.get("decision") or {}
    if isinstance(dec, dict) and dec.get("result") == "shadow":
        return True
    codes = {record.get("reason_code"),
             dec.get("reason_code") if isinstance(dec, dict)
             else None}
    return "path.shadow_route" in codes


def latest_evaluations(records) -> dict:
    """Latest playbook_eval record per playbook, in journal
    order (later records win).  Trigger-lifecycle records are
    not playbook assessments and are skipped."""
    latest: dict[str, dict] = {}
    for r in records or []:
        if r.get("type") != "evaluation":
            continue
        if r.get("kind") not in (None, "playbook_eval"):
            continue
        name = r.get("playbook")
        if name:
            latest[str(name)] = r
    return latest


def _fmt_num(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        f = float(v)
        if f == int(f) and abs(f) < 1e15:
            return f"{f:,.2f}" if abs(f) >= 100 else f"{f:g}"
        return f"{f:,.4f}".rstrip("0").rstrip(".")
    return str(v)


def _fmt_price(v) -> str:
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return str(v)


def _kv_lines(block: dict, skip=("status", "note")) -> list[str]:
    """Itemized key/value lines for a context block (structure
    read / HTF context).  Nested values render compactly; the
    block's own facts only."""
    lines = []
    for key, value in block.items():
        if key in skip or value is None:
            continue
        if isinstance(value, dict):
            inner = ", ".join(
                f"{k}={_fmt_num(v)}" for k, v in value.items()
                if v is not None)
            lines.append(f"  {key}: {inner or '(empty)'}")
        elif isinstance(value, (list, tuple)):
            inner = ", ".join(_fmt_num(v) for v in value)
            lines.append(f"  {key}: {inner}")
        else:
            lines.append(f"  {key}: {_fmt_num(value)}")
    return lines


def _context_section(title: str, block, absent_label: str) -> list[str]:
    """One context section (structure read / HTF context): the
    block's facts, or ONE explicit not-produced line when the
    route produced none — a marked absence is a fact, a missing
    section would be a silent gap."""
    if not isinstance(block, dict) or not block:
        return [f"{title}: not produced — {absent_label}"]
    if block.get("status") == "absent":
        note = block.get("note") or absent_label
        return [f"{title}: not produced — {note}"]
    lines = _kv_lines(block)
    if not lines:
        return [f"{title}: not produced — {absent_label}"]
    return [f"{title}:"] + lines


def _evidence_section(title: str, items: list[dict]) -> list[str]:
    """Itemized evidence for one stance: each item carries its
    detector and source-bar timestamp; items sharing a
    correlation group are tagged with it (one event, not
    stacked confirmations)."""
    if not items:
        return [f"{title}:", "  ➖ none recorded"]
    lines = [f"{title}:"]
    for it in items:
        group = it.get("evidence_group")
        tag = f"[{group}] " if group else ""
        det = it.get("detector") or "detector not recorded"
        ts = it.get("source_bar_ts") or "bar ts not recorded"
        lines.append(f"  • {tag}{it.get('item')} "
                     f"({det}, bar {ts})")
    return lines


def render_assessment(record: dict, *,
                      symbol: str | None = None) -> str:
    """Render one evaluation record's market assessment as
    plain text (Telegram-shaped, no markup beyond the house
    emoji vocabulary).  Pure: no I/O, no state, no invention —
    every line traces to a field on the record."""
    rec = record or {}
    playbook = rec.get("playbook") or "unknown playbook"
    tf = rec.get("intended_tf") or rec.get("actual_input_tf") \
        or "TF not recorded"
    clock = rec.get("evaluation_clock") or "clock not recorded"
    instrument = symbol or rec.get("instrument") \
        or "instrument not recorded"
    shadow = is_shadow_record(rec)
    lines: list[str] = []
    lines.append(f"🧭 MARKET ASSESSMENT — {instrument}")
    route = (rec.get("evidence") or {}).get("route") \
        if isinstance(rec.get("evidence"), dict) else None
    head = f"{playbook} · {tf} · {clock}"
    if route:
        head += f" · route {route}"
    lines.append(head)
    prov = {}
    ev = rec.get("evidence") or {}
    if isinstance(ev, dict) and isinstance(ev.get("provenance"),
                                          dict):
        prov = ev["provenance"]
    cand = rec.get("candidate") or {}
    if isinstance(cand, dict) and isinstance(
            cand.get("provenance"), dict):
        prov = prov or cand["provenance"]
    evaluated = rec.get("ts") or prov.get("evaluation_ts") \
        or "not recorded"
    bar_ts = rec.get("last_bar_ts") or "bar ts not recorded"
    lines.append(f"bar {bar_ts} · evaluated {evaluated}")
    if shadow:
        lines.append("⚠️ SHADOW EVALUATION — journaled for "
                     "grading; this route holds NO order "
                     "authority and this is not a signal")
    lines.append("")

    assessment = rec.get("assessment")
    dec = rec.get("decision") or {}
    if not isinstance(assessment, dict):
        # Honesty floor: a record with no assessment block (a
        # pre-Phase-2 journal line, or the funnel's assessment
        # step failed) renders the absence itself, plus the
        # decision facts the record DOES carry — never a
        # fabricated read.
        lines.append("⚠️ Market assessment: NOT PRODUCED — this "
                     "record carries no assessment block")
        if dec:
            lines.append(
                f"Decision on record: {dec.get('result', '?')} "
                f"({dec.get('reason_code') or 'no code'}) — "
                f"{dec.get('reason') or 'no reason recorded'}")
        elif rec.get("outcome"):
            lines.append(
                f"Outcome on record: {rec.get('outcome')} — "
                f"{rec.get('reason') or 'no reason recorded'}")
        return "\n".join(lines)

    # --- verdict + plain-language reason ----------------------
    verdict = str(assessment.get("verdict") or "no-setup")
    if shadow and verdict == "signal":
        # Belt and braces: a shadow record's derived verdict is
        # "watch" by construction; if a future record shape
        # ever carried signal here, it still must not render
        # as one.
        verdict = "watch"
    emoji = _VERDICT_EMOJI.get(verdict, "•")
    lines.append(f"{emoji} VERDICT: {verdict.upper()}")
    reason_text = assessment.get("reason_text") \
        or rec.get("reason_text") or dec.get("reason") \
        or rec.get("reason") or "no reason recorded"
    lines.append(f"Reason: {reason_text}")
    if rec.get("reason_code") or dec.get("reason_code"):
        lines.append(
            f"Reason code: "
            f"{dec.get('reason_code') or rec.get('reason_code')}")

    # --- candidate geometry ------------------------------------
    if isinstance(cand, dict) and cand.get("direction") is not None:
        side = "LONG" if int(cand["direction"]) > 0 else "SHORT"
        parts = [f"Candidate: {side}"]
        if cand.get("entry") is not None:
            parts.append(f"entry {_fmt_price(cand['entry'])}")
        if cand.get("stop") is not None:
            parts.append(
                f"invalidation (stop) {_fmt_price(cand['stop'])}")
        targets = cand.get("targets") or []
        if targets:
            parts.append("targets " + ", ".join(
                _fmt_price(t) for t in targets))
        lines.append(" · ".join(parts))
        score_bits = []
        if cand.get("asymmetry_score") is not None:
            score_bits.append(
                f"asymmetry score "
                f"{_fmt_num(cand['asymmetry_score'])}")
        if cand.get("sleeve_base_score") is not None:
            score_bits.append(
                f"sleeve base "
                f"{_fmt_num(cand['sleeve_base_score'])} "
                f"(ranking placeholder, never a floor)")
        if not score_bits and cand.get("score") is not None:
            score_bits.append(f"score {_fmt_num(cand['score'])}")
        if score_bits:
            lines.append("Scores: " + " · ".join(score_bits))

    # --- the two verdict layers --------------------------------
    av = dec.get("analytical_verdict") or {}
    ex = dec.get("execution") or {}
    if av or ex:
        layer = []
        if av:
            layer.append(
                f"analytical verdict: {av.get('verdict', '?')}"
                + (f" ({av['reason_code']})"
                   if av.get("reason_code") else ""))
        if ex:
            layer.append(
                f"execution: "
                f"{'eligible' if ex.get('eligible') else 'refused'}"
                f" — {ex.get('outcome', 'not_attempted')}"
                + (f" ({ex['reason_code']})"
                   if ex.get("reason_code") else ""))
        lines.append(" · ".join(layer))
    elif dec:
        lines.append(
            f"Decision: {dec.get('result', '?')} · "
            f"order result: {dec.get('order_result', 'not_attempted')}")

    # --- structure read + HTF context ---------------------------
    lines.append("")
    lines += _context_section(
        "Market structure", assessment.get("structure_read"),
        "this route produces no structure read")
    lines += _context_section(
        "HTF context", assessment.get("htf_context"),
        "this route records no HTF context snapshot")

    # --- observed levels the record carries (QM geometry) -------
    observed = ev.get("observed") if isinstance(ev, dict) else None
    if isinstance(observed, dict):
        level_bits = []
        for key, label in (("qm_level", "QM level"),
                           ("head", "head"),
                           ("neckline", "neckline")):
            if observed.get(key) is not None:
                level_bits.append(
                    f"{label} {_fmt_price(observed[key])}")
        if level_bits:
            lines.append("Levels: " + " · ".join(level_bits))

    # --- evidence: supporting vs contradictory, grouped ---------
    lines.append("")
    items = assessment.get("evidence_items") or []
    supporting = [i for i in items
                  if i.get("stance") == "supporting"]
    contradictory = [i for i in items
                     if i.get("stance") == "contradictory"]
    if not items:
        # Fall back to the flat lists (older record shapes):
        # still the record's own evidence, detector/ts marked
        # as not recorded rather than invented.
        for text in assessment.get("supporting_evidence") or []:
            supporting.append({"item": text, "stance": "supporting",
                               "detector": None,
                               "source_bar_ts": None,
                               "evidence_group": None})
        for text in assessment.get("contradictory_evidence") or []:
            contradictory.append(
                {"item": text, "stance": "contradictory",
                 "detector": None, "source_bar_ts": None,
                 "evidence_group": None})
    groups = assessment.get("independent_evidence_groups")
    if groups:
        lines.append("Independent evidence groups (correlated "
                     "items inside one group are ONE event, not "
                     "stacked confirmations): "
                     + ", ".join(str(g) for g in groups))
    lines += _evidence_section("Supporting evidence", supporting)
    lines += _evidence_section("Contradictory evidence",
                               contradictory)

    # --- competing scenarios -------------------------------------
    lines.append("")
    scenarios = assessment.get("competing_scenarios")
    if isinstance(scenarios, list) and scenarios:
        lines.append("Competing scenarios:")
        for s in scenarios:
            if isinstance(s, dict):
                lines.append(f"  • {s.get('scenario', '?')} — "
                             f"{s.get('status', '?')}")
            else:
                lines.append(f"  • {s}")
    elif isinstance(scenarios, dict):
        note = scenarios.get("note") or \
            "this route produces no competing scenarios"
        lines.append(f"Competing scenarios: not produced — {note}")
    else:
        lines.append("Competing scenarios: not produced — "
                     "this route produces no competing scenarios")

    # --- selected action / rejected alternatives ------------------
    if dec.get("selected_action"):
        lines.append(f"Selected action: {dec['selected_action']}")
    for alt in dec.get("rejected_alternatives") or []:
        lines.append(f"Rejected alternative: {alt}")
    if dec.get("floor_authority"):
        lines.append(f"Floor authority: {dec['floor_authority']}")

    # --- capability markers: explicit, never omitted --------------
    lines.append("")
    markers = assessment.get("capability_markers") or {}
    if markers:
        lines.append("Capability markers (census facts):")
        for name, text in markers.items():
            lines.append(f"  • {name}: {text}")
    else:
        lines.append("Capability markers: none recorded on this "
                     "record")
    return "\n".join(lines)
