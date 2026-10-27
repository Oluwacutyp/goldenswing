from goldenswing.live.evidence import EvidenceItem, score_evidence

WEIGHTS = {"structure": 50.0, "reaction": 50.0}


def item(key, *, cat="structure", group=None, stance="supports", strength=1.0,
         available=True):
    return EvidenceItem(
        key=key, label=key, source="test", observed_at="2026-10-27T00:00:00+00:00",
        timeframe="M5", category=cat, correlation_group=group or key,
        stance=stance, strength=strength, available=available,
    )


def test_correlation_group_counts_only_strongest_event_once():
    score = score_evidence([
        item("bos", group="same-impulse", strength=1.0),
        item("fvg", group="same-impulse", strength=0.8),
        item("reclaim", cat="reaction", group="separate-reaction", strength=0.5),
    ], WEIGHTS)
    assert score.score == 87.5
    assert len(score.counted) == 2
    assert {x["key"] for x in score.counted} == {"bos", "reclaim"}


def test_missing_and_neutral_evidence_never_become_support():
    score = score_evidence([
        item("unknown-feed", available=False),
        item("neutral-pattern", stance="neutral", cat="reaction"),
    ], WEIGHTS)
    assert score.score == 50.0
    assert score.support_points == 0.0
    assert score.contradiction_points == 0.0
    assert len(score.neutral_or_missing) == 2


def test_equal_strength_conflicting_evidence_from_same_event_cancels():
    score = score_evidence([
        item("bull", group="single-event", stance="supports"),
        item("bear", group="single-event", stance="contradicts"),
    ], WEIGHTS)
    assert score.score == 50.0
    assert not score.counted
    assert score.neutral_or_missing[0]["why_neutral"].startswith("equal-strength")


def test_duplicate_evidence_key_is_ignored_deterministically():
    score = score_evidence([
        item("bos", group="event-1"),
        item("bos", group="event-2", stance="contradicts"),
    ], WEIGHTS)
    assert score.score == 75.0
    assert score.duplicates == ["bos"]


def test_weights_must_be_exact_budget():
    try:
        score_evidence([item("x")], {"structure": 99.0})
    except ValueError as exc:
        assert "sum to 100" in str(exc)
    else:
        raise AssertionError("invalid category budget accepted")
