"""Gap RESPONSE (MarketSituation Pass 9 follow-through).

A coverage gap names a market configuration the whole book can
see and no family describes.  This module answers it: for an open
UNWATCHED gap it proposes a candidate detector — the missing
behaviour, stated as data — and evaluates that detector on
chronological data.  The outcome (detector + evaluation) rides
beside the gap in the coverage report record.

What a detector is: a hypothesis, never an order.  It fires once
per gap episode, at first sight of the configuration, and states
a direction, a reference price, a target and an invalidation —
all read from the situation's own fields (swings, ATR).  Nothing
here moves a parameter, spawns a family, or reaches a decision
path: the response is evidence, and promotion stays a separate,
governed step (§G discipline stands — the response feeds no
decision automatically).

What a gap gets:

- ``unwatched`` + a detector template exists → proposal AND
  chronological evaluation (``status: "evaluated"``).
- ``unwatched`` + no template for the class yet → the response
  says so, naming the class (``status: "untemplated"``) — the
  missing work is stated, not silently skipped.
- ``deliberate_data`` / ``deliberate_news`` → NO detector
  (``status: "standing_aside"``).  The system stands aside on
  purpose there; proposing into a halted feed or a news
  blackout would violate the fail-closed design.

Evaluation (chronological, no lookahead).  The input is the
journal's situation stream — the same records that fired the
gap — walked in timestamp order.  The trigger is the first
record of an episode (first sight of the configuration after
an absence); the outcome is measured ONLY from confirmed
swings whose bar timestamp falls after the trigger timestamp
(the swing's bar time is when the price was actually there,
wherever in the stream the swing was reported).  A trigger
whose forward window runs past the end of the stream is
EXCLUDED (counted as ``excluded_incomplete``, never as a
miss); a trigger with no reference swings or no ATR is
``excluded_no_reference``; a correction detector with no
higher-timeframe bias as of the trigger is
``excluded_no_htf_bias``.  Outcome thresholds are fixed by
the template (target 1.0 ATR beyond the trigger-time swing
with the direction, invalidation 0.5 ATR beyond the
trigger-time swing against it); the first threshold reached
wins.  The evaluation reports counts and mean excursions in
ATR units — never a claim without the numbers beside it.
"""
from __future__ import annotations

from . import range_promotion as _eval

#: Forward window for a detector evaluation, in hours.  6h is 24
#: M15 bars — one session's worth of follow-through for the
#: fastest gap class this module answers.
DEFAULT_WINDOW_HOURS = 6.0

#: Target: 1.0 ATR beyond the trigger-time swing with the
#: direction.  Invalidation: 0.5 ATR beyond the trigger-time
#: swing against it.  Fixed by the template, never tuned to
#: the data under evaluation.
TARGET_ATR = 1.0
INVALIDATION_ATR = 0.5

STATUS_EVALUATED = "evaluated"
STATUS_PROPOSED = "proposed"
STATUS_UNTEMPLATED = "untemplated"
STATUS_STANDING_ASIDE = "standing_aside"


# ------------------------------------------------------------ proposals

def propose_detector(configuration: dict,
                     trigger_state: dict | None = None
                     ) -> dict | None:
    """The candidate detector for one gap configuration — or
    None when no template describes the class yet.

    ``trigger_state`` is the trigger record's own per-TF state
    (swings + ATR): when present the detector states its
    reference, target and invalidation as prices; when absent
    it states the rule in ATR units and marks prices pending.
    A correction detector carries no fixed direction: the
    direction is the HTF bias as of each trigger, read at
    evaluation time (``direction_source: "htf_bias"``)."""
    config = configuration or {}
    kind = config.get("kind")
    if kind == "tf" and config.get("phase") == "clean_trend":
        bias = config.get("bias")
        if bias not in (-1, 1):
            return None
        tf = config.get("tf")
        detector = {
            "detector_key": (
                f"trend_continuation:{tf}:bias={int(bias):+d}"),
            "kind": "trend_continuation",
            "gap_config_key": config.get("key"),
            "tf": tf,
            "direction": int(bias),
            "rule": (
                f"on first sight of {tf} clean_trend with "
                f"bias {int(bias):+d}, hypothesise "
                f"continuation {'down' if bias == -1 else 'up'}: "
                f"target 1.0 ATR beyond the trigger-time "
                f"confirmed swing with the direction, "
                f"invalidation 0.5 ATR beyond the "
                f"trigger-time confirmed swing against it"),
            "target_atr": TARGET_ATR,
            "invalidation_atr": INVALIDATION_ATR,
        }
        _fill_reference(detector, trigger_state, int(bias))
        return detector
    if kind == "relationship" \
            and config.get("rel_kind") == "correcting":
        htf = config.get("htf")
        ltf = config.get("ltf")
        # The direction is the HTF bias AS OF EACH TRIGGER —
        # it is read per episode at evaluation time (the HTF
        # bias can flip between episodes), so the proposal
        # carries no fixed direction, only its source.
        detector = {
            "detector_key": (
                f"correction_completion:{htf}->{ltf}"),
            "kind": "correction_completion",
            "gap_config_key": config.get("key"),
            "tf": ltf,
            "htf": htf,
            "direction": None,
            "direction_source": "htf_bias",
            "rule": (
                f"on first sight of {htf}->{ltf} correcting, "
                f"hypothesise the {ltf} correction completes "
                f"and price resumes the {htf} bias as of the "
                f"trigger: target 1.0 ATR beyond the "
                f"trigger-time confirmed {ltf} swing with "
                f"that direction, invalidation 0.5 ATR "
                f"beyond the trigger-time confirmed {ltf} "
                f"swing against it"),
            "target_atr": TARGET_ATR,
            "invalidation_atr": INVALIDATION_ATR,
        }
        _fill_reference(detector, trigger_state, None)
        return detector
    return None


def _fill_reference(detector: dict, trigger_state: dict | None,
                    direction: int | None) -> None:
    """Price the detector from the trigger record's own state:
    reference swings + ATR.  Absent state, or no fixed
    direction (a correction detector's direction is read per
    episode): prices pending, the rule in ATR units still
    stands."""
    state = trigger_state or {}
    swings = state.get("swings") or {}
    atr = (state.get("volatility") or {}).get("atr")
    detector["atr"] = atr
    detector["reference_favourable"] = None
    detector["reference_adverse"] = None
    detector["target_price"] = None
    detector["invalidation_price"] = None
    if direction is None:
        return
    if direction == -1:
        fav = (swings.get("confirmed_low") or {}).get("price")
        adv = (swings.get("confirmed_high") or {}).get("price")
    else:
        fav = (swings.get("confirmed_high") or {}).get("price")
        adv = (swings.get("confirmed_low") or {}).get("price")
    detector["reference_favourable"] = fav
    detector["reference_adverse"] = adv
    if fav is None or adv is None or atr is None:
        return
    if direction == -1:
        detector["target_price"] = fav - TARGET_ATR * atr
        detector["invalidation_price"] = (
            adv + INVALIDATION_ATR * atr)
    else:
        detector["target_price"] = fav + TARGET_ATR * atr
        detector["invalidation_price"] = (
            adv - INVALIDATION_ATR * atr)


# ------------------------------------------------------------ evaluation

def _as_of_index(situations: list[dict]) -> dict[str, list]:
    """Per-TF (ts, state) lists in timestamp order — every
    lookup reads the latest state at or BEFORE the trigger
    timestamp, never after it."""
    index: dict[str, list] = {}
    for row in situations or []:
        state = row.get("state")
        if not isinstance(state, dict):
            continue
        tf = state.get("tf") or row.get("tf")
        ts = _eval._ts(row.get("ts"))
        if not tf or ts is None:
            continue
        index.setdefault(str(tf), []).append((ts, state))
    for tf in index:
        index[tf].sort(key=lambda pair: pair[0])
    return index


def _as_of(index: dict[str, list], tf: str, ts):
    """Latest state for ``tf`` at or before ``ts`` — None when
    the stream has not measured the timeframe yet."""
    rows = index.get(str(tf)) or []
    best = None
    for row_ts, state in rows:
        if row_ts <= ts:
            best = state
        else:
            break
    return best


def _swings_after(index: dict[str, list], tf: str, start,
                  end) -> list[tuple]:
    """Confirmed swings for ``tf`` with bar timestamps in
    (start, end]: (ts, price, side) with side 'high'/'low'.
    Only swings strictly after the trigger count — a swing
    at or before the trigger is the reference, never the
    outcome.  The swing counts at its own BAR timestamp (when
    the price was actually there), wherever in the stream it
    was reported: a swing confirmed after the window still
    records a price event inside it, and the record's own
    timestamp says nothing about when the price moved."""
    out = []
    seen: set[tuple] = set()
    for _row_ts, state in index.get(str(tf)) or []:
        swings = state.get("swings") or {}
        for side, key in (("high", "confirmed_high"),
                          ("low", "confirmed_low")):
            swing = swings.get(key) or {}
            price = swing.get("price")
            swing_ts = _eval._ts(swing.get("ts"))
            if price is None or swing_ts is None:
                continue
            if start < swing_ts <= end:
                marker = (swing_ts, price, side)
                if marker in seen:
                    continue
                seen.add(marker)
                out.append(marker)
    out.sort(key=lambda item: item[0])
    return out


def _trigger_episodes(detector: dict, situations: list[dict],
                      index: dict[str, list]) -> list[dict]:
    """First-sight trigger records for the detector, in
    timestamp order.  Consecutive records carrying the
    configuration are ONE episode; a new episode starts only
    after a record where the configuration was absent."""
    kind = detector["kind"]
    episodes: list[dict] = []
    carrying = False
    ordered = sorted(
        (row for row in situations or []
         if _eval._ts(row.get("ts")) is not None),
        key=lambda row: _eval._ts(row.get("ts")))
    for row in ordered:
        ts = _eval._ts(row.get("ts"))
        present = False
        trigger_state = None
        htf_state = None
        observed = False
        if kind == "trend_continuation":
            state = row.get("state") or {}
            # Only a record for THIS timeframe observes the
            # configuration: records for other TFs say nothing
            # about it and must not break the episode.
            if state.get("tf") != detector["tf"]:
                continue
            observed = True
            present = (
                state.get("bias") == detector["direction"]
                and state.get("structure_phase")
                == "clean_trend")
            trigger_state = state if present else None
        elif kind == "correction_completion":
            rels = row.get("relationships")
            # Only a record carrying a relationships list
            # observes the pair: a record without one says
            # nothing about it and must not break the episode.
            if rels is None:
                continue
            observed = True
            present = any(
                (rel.get("htf"), rel.get("ltf"),
                 rel.get("kind")) == (
                    detector["htf"], detector["tf"],
                    "correcting")
                for rel in rels)
            if present:
                htf_state = _as_of(index, detector["htf"], ts)
                trigger_state = _as_of(
                    index, detector["tf"], ts)
        if not observed:
            continue
        if present and not carrying:
            episodes.append({
                "trigger_ts": ts,
                "trigger_state": trigger_state,
                "htf_state": htf_state,
            })
        carrying = present
    return episodes


def evaluate_detector(detector: dict, situations: list[dict],
                      *, window_hours: float = DEFAULT_WINDOW_HOURS
                      ) -> dict:
    """Chronological evaluation of one detector over a
    situation stream.  Returns counts, hit rate and mean
    excursions in ATR units, plus one row per episode so the
    numbers can be checked by hand."""
    from datetime import timedelta

    index = _as_of_index(situations)
    episodes = _trigger_episodes(detector, situations, index)
    # The stream's end is the EVALUATION timeframe's last
    # record — a trigger is incomplete when ITS timeframe's
    # forward window runs past that, even if other timeframes
    # kept reporting longer.
    tf_rows = index.get(str(detector["tf"])) or []
    stream_end = tf_rows[-1][0] if tf_rows else None
    detail: list[dict] = []
    counts = {"episodes": len(episodes), "evaluated": 0,
              "excluded_incomplete": 0,
              "excluded_no_reference": 0,
              "excluded_no_htf_bias": 0,
              "hits": 0, "invalidations": 0, "neither": 0}
    fav_sum = 0.0
    adv_sum = 0.0
    for ep in episodes:
        ts = ep["trigger_ts"]
        row = {"trigger_ts": str(ts)}
        window_end = ts + timedelta(hours=window_hours)
        if stream_end is not None and window_end > stream_end:
            counts["excluded_incomplete"] += 1
            row["outcome"] = "excluded_incomplete"
            detail.append(row)
            continue
        if detector["kind"] == "correction_completion":
            # Direction is the HTF bias as of THIS trigger —
            # the HTF bias can flip between episodes, so it
            # is read per episode, never fixed at proposal.
            htf_state = ep.get("htf_state") or {}
            direction = htf_state.get("bias")
            if direction not in (-1, 1):
                counts["excluded_no_htf_bias"] += 1
                row["outcome"] = "excluded_no_htf_bias"
                detail.append(row)
                continue
        else:
            direction = detector["direction"]
        row["direction"] = direction
        probe = dict(detector)
        _fill_reference(probe, ep.get("trigger_state"),
                        direction)
        atr = probe.get("atr")
        fav_ref = probe.get("reference_favourable")
        adv_ref = probe.get("reference_adverse")
        if atr is None or fav_ref is None or adv_ref is None:
            counts["excluded_no_reference"] += 1
            row["outcome"] = "excluded_no_reference"
            detail.append(row)
            continue
        target = probe["target_price"]
        invalidation = probe["invalidation_price"]
        swings = _swings_after(index, detector["tf"], ts,
                               window_end)
        target_first = None
        invalidation_first = None
        best_fav = None
        best_adv = None
        for swing_ts, price, side in swings:
            if direction == -1:
                if side == "low":
                    exc = (fav_ref - price) / atr
                    best_fav = exc if best_fav is None \
                        else max(best_fav, exc)
                    if price <= target and target_first is None:
                        target_first = swing_ts
                else:
                    exc = (price - adv_ref) / atr
                    best_adv = exc if best_adv is None \
                        else max(best_adv, exc)
                    if price >= invalidation \
                            and invalidation_first is None:
                        invalidation_first = swing_ts
            else:
                if side == "high":
                    exc = (price - fav_ref) / atr
                    best_fav = exc if best_fav is None \
                        else max(best_fav, exc)
                    if price >= target and target_first is None:
                        target_first = swing_ts
                else:
                    exc = (adv_ref - price) / atr
                    best_adv = exc if best_adv is None \
                        else max(best_adv, exc)
                    if price <= invalidation \
                            and invalidation_first is None:
                        invalidation_first = swing_ts
        row["favourable_atr"] = (
            round(best_fav, 3) if best_fav is not None else None)
        row["adverse_atr"] = (
            round(best_adv, 3) if best_adv is not None else None)
        row["target_first_ts"] = (
            str(target_first) if target_first else None)
        row["invalidation_first_ts"] = (
            str(invalidation_first)
            if invalidation_first else None)
        if target_first is not None and (
                invalidation_first is None
                or target_first < invalidation_first):
            outcome = "hit"
            counts["hits"] += 1
        elif invalidation_first is not None and (
                target_first is None
                or invalidation_first < target_first):
            outcome = "invalidation"
            counts["invalidations"] += 1
        else:
            outcome = "neither"
            counts["neither"] += 1
        counts["evaluated"] += 1
        fav_sum += best_fav or 0.0
        adv_sum += best_adv or 0.0
        row["outcome"] = outcome
        detail.append(row)
    evaluated = counts["evaluated"]
    return {
        "detector_key": detector["detector_key"],
        "kind": detector["kind"],
        "direction": detector["direction"],
        "direction_source": detector.get(
            "direction_source", "fixed"),
        "window_hours": window_hours,
        "target_atr": TARGET_ATR,
        "invalidation_atr": INVALIDATION_ATR,
        **counts,
        "hit_rate": (round(counts["hits"] / evaluated, 4)
                     if evaluated else None),
        "mean_favourable_atr": (
            round(fav_sum / evaluated, 3) if evaluated else None),
        "mean_adverse_atr": (
            round(adv_sum / evaluated, 3) if evaluated else None),
        "evaluated_through": (
            str(stream_end) if stream_end is not None else None),
        "episodes_detail": detail,
    }


# ------------------------------------------------------------ the response

def respond_to_gap(gap: dict, situations: list[dict] | None,
                   *, window_hours: float = DEFAULT_WINDOW_HOURS
                   ) -> dict:
    """One open gap's response: proposal + evaluation when the
    class has a template and the stream is supplied; the honest
    lesser answer otherwise.  ``gap`` is an open-gap row from
    the coverage snapshot (config_key, configuration,
    classification, first_seen_ts)."""
    gap = gap or {}
    config = gap.get("configuration") or {}
    classification = gap.get("classification")
    base = {"config_key": gap.get("config_key"),
            "classification": classification,
            "first_seen_ts": gap.get("first_seen_ts")}
    if classification != "unwatched":
        return {**base, "status": STATUS_STANDING_ASIDE,
                "detector": None, "evaluation": None,
                "reason": (
                    f"gap is {classification} — the system "
                    "stands aside on purpose there, so no "
                    "detector is proposed into it")}
    detector = propose_detector(config)
    if detector is None:
        return {**base, "status": STATUS_UNTEMPLATED,
                "detector": None, "evaluation": None,
                "reason": (
                    f"no detector template describes a "
                    f"{config.get('kind')} gap of this class "
                    f"yet ({config.get('phase') or config.get('rel_kind')}) "
                    "— the missing work is stated here, the "
                    "gap is not silently skipped")}
    if situations is None:
        return {**base, "status": STATUS_PROPOSED,
                "detector": detector, "evaluation": None,
                "reason": (
                    "detector proposed; no situation stream "
                    "was supplied, so the chronological "
                    "evaluation is pending")}
    evaluation = evaluate_detector(
        detector, situations, window_hours=window_hours)
    return {**base, "status": STATUS_EVALUATED,
            "detector": detector, "evaluation": evaluation,
            "reason": None}


def respond_to_open_gaps(open_gaps: list[dict],
                         situations: list[dict] | None,
                         *, window_hours: float
                         = DEFAULT_WINDOW_HOURS) -> dict:
    """Every open gap's response, keyed by config_key."""
    return {str(gap.get("config_key") or ""): respond_to_gap(
                gap, situations, window_hours=window_hours)
            for gap in open_gaps or []
            if gap.get("config_key")}


def new_state() -> dict:
    """The response cache (rides inside the coverage state):
    one response per gap episode, keyed
    ``config_key|first_seen_ts|classification`` — the
    classification is part of the key because a gap's response
    depends on it (an unwatched gap that turns deliberate
    stops getting a detector)."""
    return {"responses": {}}


def cycle_responses(state: dict, *, open_gaps: list[dict],
                    situations: list[dict] | None,
                    window_hours: float = DEFAULT_WINDOW_HOURS
                    ) -> dict:
    """The cycle's responses for its open gaps.  A gap episode
    already answered is kept — unless the situation stream has
    grown past the stored ``evaluated_through`` (new complete
    episodes may exist), in which case the fixed evaluation
    rule runs again over the longer stream.  Deterministic in
    the stream: the same stream gives the same response."""
    for key, value in new_state().items():
        state.setdefault(key, value)
    stored: dict = state["responses"]
    stream_max = None
    if situations:
        stamps = [_eval._ts(row.get("ts"))
                  for row in situations]
        stamps = [ts for ts in stamps if ts is not None]
        stream_max = max(stamps) if stamps else None
    out: dict = {}
    for gap in open_gaps or []:
        key = str(gap.get("config_key") or "")
        if not key:
            continue
        # The response depends on the CURRENT classification
        # (a gap can turn deliberate as the situation changes),
        # so the classification is part of the episode key.
        episode_key = (
            f"{key}|{gap.get('first_seen_ts')}"
            f"|{gap.get('classification')}")
        existing = stored.get(episode_key)
        if existing is not None and situations is None:
            out[key] = existing
            continue
        if existing is not None and existing.get(
                "status") == STATUS_EVALUATED:
            through = _eval._ts(
                (existing.get("evaluation") or {}).get(
                    "evaluated_through"))
            if through is not None and stream_max is not None \
                    and stream_max <= through:
                out[key] = existing
                continue
        response = respond_to_gap(
            gap, situations, window_hours=window_hours)
        stored[episode_key] = response
        out[key] = response
    return out
