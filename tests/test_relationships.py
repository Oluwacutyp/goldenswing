"""Cross-timeframe relationship engine (SPEC §4.2(a);
AMENDMENTS-1 §A1) — Pass 1 pins.

- every kind in the §4.2(a) table is produced by a constructed
  state pair, and a named near-miss does NOT produce it;
- the A1 disagreement fixture: a D1-bullish / H4-bearish /
  H1-bullish / M15-bearish stack yields the layered readings
  (larger trend · intermediate correction · continuation
  attempt · local retracement) and NO flattened single-direction
  verdict anywhere in the relationship records;
- the hierarchy reading maps the verdict's measured facts
  (never its gate outcome) and rides as one row among rows.
"""
import numpy as np
import pandas as pd

from goldenswing.live import relationships as R
from goldenswing.live.hierarchy import TimeframeStack


# ------------------------------------------------------------------
# crafted states (the classifier is a pure function of states)
# ------------------------------------------------------------------
def st(bias, phase="clean_trend", *, word=None, event=None,
       hi=None, lo=None):
    if word is None:
        word = {1: "up", -1: "down", 0: "flat", None: None}[bias]
    return {
        "bias": bias, "structure_phase": phase,
        "direction_word": word, "last_structure_event": event,
        "swings": {
            "confirmed_high": ({"price": hi, "ts": None}
                               if hi is not None else None),
            "confirmed_low": ({"price": lo, "ts": None}
                              if lo is not None else None),
            "pending_high": None, "pending_low": None},
    }


def kind_of(htf_state, ltf_state, *, price=None, pair=("D1", "H1")):
    rec = R.classify_pair(pair[0], pair[1], htf_state, ltf_state,
                          price=price)
    return rec["kind"]


# ------------------------------------------------------------------
# each kind, with a near-miss
# ------------------------------------------------------------------
def test_aligned_and_near_miss():
    assert kind_of(st(1), st(1)) == "aligned"
    assert kind_of(st(-1), st(-1)) == "aligned"
    # Near-miss: the faster side's phase turns transitional —
    # the pair is no longer a clean aligned read.
    assert kind_of(st(1), st(1, "reversal")) == "transitioning"


def test_correcting_and_near_miss():
    h = st(1, hi=2100.0, lo=2000.0)
    assert kind_of(h, st(-1), price=2050.0) == "correcting"
    # A ranging faster TF under a slower trend is also a
    # correction (SPEC §4.2(a), letter of the definition).
    assert kind_of(h, st(0, "range"), price=2050.0) == "correcting"
    # Near-miss: price through the defended swing — the
    # counter-move is no longer inside H's structure.
    assert kind_of(h, st(-1), price=1990.0) != "correcting"


def test_conflicting_and_near_miss():
    h = st(1, hi=2100.0, lo=2000.0)
    # Both trend-phased, opposite biases, defended swing broken.
    assert kind_of(h, st(-1), price=1990.0) == "conflicting"
    # The slower TF's own against-bias CHoCH is the second
    # break signal — no price needed.
    h_broken = st(1, hi=2100.0, lo=2000.0, event={
        "kind": "CHoCH", "direction": -1, "against_bias": True})
    assert kind_of(h_broken, st(-1)) == "conflicting"
    # Near-miss: same opposing biases, defended swing intact.
    assert kind_of(h, st(-1), price=2050.0) == "correcting"


def test_reversal_attempt_and_near_miss():
    h = st(1, hi=2100.0, lo=2000.0)
    choch = {"kind": "CHoCH", "direction": -1,
             "against_bias": False}
    # The faster TF still measures bullish, but its last bar
    # printed a CHoCH against the slower bias: an attempt,
    # before the slower thesis is invalidated.
    assert kind_of(h, st(1, event=choch), price=2050.0) == \
        "reversal_attempt"
    msch = {"kind": "MSCH", "direction": -1,
            "against_bias": False}
    assert kind_of(h, st(1, event=msch), price=2050.0) == \
        "reversal_attempt"
    # Near-miss: a BOS against is continuation-grade displacement
    # math, not the msch/choch attempt the table names.
    bos = {"kind": "BOS", "direction": -1, "against_bias": True}
    assert kind_of(h, st(1, event=bos), price=2050.0) != \
        "reversal_attempt"


def test_nested_and_near_miss():
    h = st(0, "range", hi=2100.0, lo=1900.0)
    inner = st(0, "range", hi=2050.0, lo=1950.0)
    assert kind_of(h, inner) == "nested"
    # Near-miss: the faster range escapes the slower range.
    escaped = st(0, "range", hi=2150.0, lo=1950.0)
    assert kind_of(h, escaped) == "independent"


def test_transitioning_and_near_miss():
    # Bedrock disagreement on the slower side (phase unclear,
    # direction flat against a measured up-bias).
    h = st(1, "unclear", word="flat")
    assert kind_of(h, st(-1)) == "transitioning"
    # Near-miss: same biases with a clean slower phase is a
    # correction, not a transition.
    h_clean = st(1, hi=2100.0, lo=2000.0)
    assert kind_of(h_clean, st(-1), price=2050.0) == "correcting"


def test_independent_and_insufficient():
    h = st(0, "range", hi=2100.0, lo=1900.0)
    wide = st(1, hi=2200.0, lo=1800.0)
    assert kind_of(h, wide) == "independent"
    rec = R.classify_pair("D1", "H1", st(None), st(1))
    assert rec["kind"] is None
    assert rec["reason"] == "insufficient_state"


def test_pairs_are_the_spec_pairs_in_order():
    rels = R.compute_relationships({})
    assert [(r["htf"], r["ltf"]) for r in rels] == list(R.PAIRS)
    assert all(r["kind"] is None for r in rels)
    assert all(r["source"] == "engine" for r in rels)


# ------------------------------------------------------------------
# A1: disagreement without suppression
# ------------------------------------------------------------------
def test_a1_disagreement_is_layered_not_flattened():
    """D1 bullish / H4 bearish / H1 bullish / M15 bearish must
    read as larger trend + intermediate correction +
    continuation attempt + local retracement — simultaneously,
    with no single flattened directional verdict."""
    per_tf = {
        "D1": st(1, hi=2150.0, lo=1950.0),
        "H4": st(-1, hi=2100.0, lo=1980.0),
        "H1": st(1, hi=2120.0, lo=2000.0),
        "M30": st(0, "range"),
        "M15": st(-1, hi=2090.0, lo=2010.0),
        "M5": st(-1, "range", hi=2080.0, lo=2020.0),
    }
    rels = R.compute_relationships(per_tf, price=2060.0)
    by_pair = {(r["htf"], r["ltf"]): r["kind"] for r in rels}
    assert by_pair[("D1", "H1")] == "aligned"        # larger trend
    assert by_pair[("D1", "H4")] == "correcting"     # intermediate
    assert by_pair[("H4", "H1")] == "correcting"     # continuation
    # attempt rising against the H4 leg — a different pair with
    # its own reading, not a vote against the D1 trend.
    assert by_pair[("H1", "M15")] == "correcting"    # local
    # retracement
    assert by_pair[("H4", "M15")] == "aligned"
    # The disagreement survives: more than one kind is live, and
    # no record flattens the stack into one direction.
    assert len({r["kind"] for r in rels}) >= 2
    for rec in rels:
        assert "verdict" not in rec
        assert "direction" not in rec
        assert rec["evidence"]["bias_h"] is not None


# ------------------------------------------------------------------
# hierarchy reading over real frames
# ------------------------------------------------------------------
TF_FREQ = {"D1": "D", "H4": "4h", "H1": "h", "M30": "30min",
           "M15": "15min", "M5": "5min"}
_TF_N = {"D1": 150, "H4": 300, "H1": 500, "M30": 600,
         "M15": 800, "M5": 1200}


def _frame(end, n, freq, drift=0.8, base=2000.0, amp=6.0,
           period=12):
    idx = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    closes = np.array([
        base + drift * i
        + amp * (1.0 - abs((i % period) / (period / 2.0) - 1.0))
        for i in range(n)], dtype=float)
    opens = closes - 0.3
    return pd.DataFrame(
        {"open": opens, "high": closes + 0.7,
         "low": closes - 0.7, "close": closes, "volume": 100.0},
        index=idx)


def _aligned_stack(drifts):
    end = pd.Timestamp("2026-03-02", tz="UTC")
    return TimeframeStack.from_frames({
        tf: _frame(end, _TF_N[tf], TF_FREQ[tf],
                   drift=drifts.get(tf, 0.8))
        for tf in TF_FREQ})


def test_hierarchy_reading_aligned_and_correcting():
    stack = _aligned_stack({})
    reading = R.hierarchy_reading(stack, asof=stack.cursors["H1"])
    assert reading["source"] == "hierarchy"
    assert reading["kind"] == "aligned"
    assert reading["evidence"]["bias_d1"] == 1
    stack_down = _aligned_stack({"H4": -0.8})
    reading = R.hierarchy_reading(
        stack_down, asof=stack_down.cursors["H1"])
    assert reading["kind"] == "correcting"
    assert reading["evidence"]["bias_h4"] == -1


def test_hierarchy_reading_never_raises_and_stays_single():
    assert R.hierarchy_reading(None) is None
    pairs = R.compute_relationships({})
    one = R.with_hierarchy(pairs, {"htf": "D1", "ltf": "H1",
                                   "source": "hierarchy",
                                   "kind": "aligned"})
    two = R.with_hierarchy(one, {"htf": "D1", "ltf": "H1",
                                 "source": "hierarchy",
                                 "kind": "correcting"})
    rows = [r for r in two if r["source"] == "hierarchy"]
    assert len(rows) == 1
    assert rows[0]["kind"] == "correcting"
