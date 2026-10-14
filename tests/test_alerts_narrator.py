"""Signal narrator: real numbers in, no invented numbers out, never crashes."""
import re

import pandas as pd

from goldenswing.alerts import build_alert
from goldenswing.alerts.model import build_alert_text
from goldenswing.alerts.narrator import narrate
from goldenswing.golden.detector import GoldenSignal


def _ctx():
    return {
        "direction": 1,
        "score": 76,
        "targets": [2678.3, 2684.2],
        "factors": [
            "trend-aligned continuation (up)",
            "fresh CHOCH 0 bars ago",
            "momentum aligned (+0.7 ATR)",
            "liquidity sweep of sell-side pool @ 2669.3, 1 bars ago",
            "displacement 1.2x ATR",
            "room to next liquidity pool: 1.4x ATR",
            "london killzone",
        ],
    }


# ------------------------------------------------- real numbers, event order
def test_narration_carries_the_actual_numbers():
    out = narrate(_ctx())
    for needle in ("2,669.3", "1.2x ATR", "2,678.3", "London killzone",
                   "one bar ago", "+0.7 ATR"):
        assert needle in out, needle


def test_narration_is_event_ordered_and_phone_sized():
    out = narrate(_ctx())
    # sweep story first, trade math last
    assert out.index("Swept") < out.index("Displacement") < out.index("T1 at")
    words = out.split()
    assert 15 <= len(words) <= 90, len(words)
    sentences = [s for s in re.split(r"\.\s+", out.strip()) if s]
    assert 2 <= len(sentences) <= 6, sentences  # 2-4 sentences, not a dump


def test_short_mirrors_correctly():
    out = narrate({
        "direction": -1, "targets": [2640.0],
        "factors": [
            "BOS confirms the sweep reversal",
            "liquidity sweep of buy-side pool @ 2655.4, 3 bars ago",
            "strong displacement 1.8x ATR",
            "short from premium",
            "room to next liquidity pool: 2.1x ATR",
        ],
    })
    assert "buy-side pool at 2,655.4" in out
    assert "sellers in control" in out
    assert "buyers" not in out


# ------------------------------------------------- no invention, no crashes
def test_no_numbers_invented_when_none_present():
    out = narrate({"direction": 1,
                   "factors": ["london killzone",
                               "trend-aligned continuation (up)"]})
    assert "London killzone" in out
    # no price-like numbers anywhere in the narration
    assert not re.search(r"\d{3,}", out)


def test_missing_fields_degrade_gracefully():
    for ctx in ({}, {"factors": []}, None,
                {"factors": ["unparseable factor xyz"]},
                {"direction": 1}):  # no factors key at all
        out = narrate(ctx)
        assert isinstance(out, str) and out, ctx


def test_broken_factor_values_do_not_crash():
    out = narrate({"factors": [
        "liquidity sweep of sell-side pool @ abc, x bars ago",
        "momentum aligned (NaN ATR)",
    ]})
    assert isinstance(out, str)


def test_honest_caveat_for_penalties():
    out = narrate({"direction": 1,
                   "factors": ["buying in premium — penalised",
                               "london killzone"]})
    assert "Weak spot: buying in premium" in out


# ------------------------------------------------- caption wiring
def test_caption_leads_with_narration_above_backing():
    sig = GoldenSignal(
        index=500, timestamp=pd.Timestamp("2025-03-01 13:00", tz="UTC"),
        direction=1, score=78.0, kind="sweep_reversal",
        factors=["liquidity sweep of sell-side pool @ 2,900.0, 2 bars ago",
                 "CHOCH confirms the sweep reversal",
                 "displacement 1.2x ATR",
                 "room to next liquidity pool: 1.4x ATR"],
        entry_zone=(2909.0, 2913.0), entry_ref=2912.0,
        invalidation=2901.0, targets=[2925.0, 2941.0], atr=10.0)
    a = build_alert(sig, instrument="XAUUSD", mode="swing")
    assert a.text.index("Swept the sell-side pool at 2,900.0") \
        < a.text.index("Backing:")
    assert "buyers in control" in a.text
    # the evidence list survives untouched
    assert "• CHOCH confirms the sweep reversal" in a.text


# ------------------------------------------------- snapshot on a real signal
def test_snapshot_real_signal_narration():
    out = narrate(_ctx())
    expected = (
        "Swept the sell-side pool at 2,669.3 one bar ago, taking the sell "
        "stops. Fresh CHOCH just printed — buyers in control. Displacement "
        "ran 1.2x ATR, momentum aligned (+0.7 ATR). In the London killzone. "
        "1.4x ATR of room to T1 at 2,678.3."
    )
    assert out == expected
