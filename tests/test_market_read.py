"""Market Read: trend + key zones + scenarios, rendered from engine
state only (owner request 2026-10-26: the analysis message WITH the
engine status in one message).

Pins the honesty contract: every level comes from a zone band or
the D1 hold state; missing inputs render explicit ``not produced``
lines, never invented levels.  Also pins the runner method (inputs
gathered from the live zone cache + hold state) and the /status
wiring (Market Read section appears when the controller provides
it, and is skipped for old controllers).
"""

from types import SimpleNamespace

from goldenswing.alerts.commands import CommandContext
from goldenswing.execution.market_read import render_market_read
from goldenswing.execution.mt5_commands import mt5_router
from goldenswing.execution.mt5_runner import MT5Runner

OWNER = "12345"


def _zone(side, level, top, bottom, **kw):
    z = {"side": side, "level": level, "top": top, "bottom": bottom,
         "score": 90.0, "tier": "MAJOR", "touches": 2}
    z.update(kw)
    return z


def _hold(**kw):
    h = {"confirmed_highs": [4120.0, 4100.0],
         "confirmed_lows": [3990.0, 3982.0],
         "pending_low": {"price": 3975.4, "bars_confirmed": 1},
         "pending_high": None,
         "flip_needed": "need confirmed lower high for SHORT",
         "swing_n": 3}
    h.update(kw)
    return h


# ------------------------------------------------------------ trend

def test_trend_bearish_lower_highs_lower_lows():
    out = render_market_read(price=4015.0, hold=_hold(), zones=[])
    assert out.splitlines()[0].startswith(
        "trend: bearish — lower highs and lower lows")


def test_trend_bullish_and_mixed():
    bull = render_market_read(
        price=4015.0,
        hold=_hold(confirmed_highs=[4100.0, 4120.0],
                   confirmed_lows=[3982.0, 3990.0]), zones=[])
    assert "bullish — higher highs and higher lows" in bull
    mixed = render_market_read(
        price=4015.0,
        hold=_hold(confirmed_highs=[4120.0, 4100.0],
                   confirmed_lows=[3982.0, 3990.0]), zones=[])
    assert "mixed — lower highs, higher lows" in mixed


def test_trend_insufficient_and_pending_and_flip():
    out = render_market_read(
        price=4015.0,
        hold=_hold(confirmed_highs=[4120.0], confirmed_lows=[]),
        zones=[])
    assert "insufficient confirmed swings (1 highs, 0 lows)" in out
    full = render_market_read(price=4015.0, hold=_hold(), zones=[])
    assert "pending swing low 3975.4 (1/3 bars confirmed)" in full
    assert "need confirmed lower high for SHORT" in full


def test_trend_not_produced_without_hold():
    out = render_market_read(price=4015.0, hold=None, zones=[])
    assert out.splitlines()[0] == \
        "trend: not produced (no D1 structure state)"


# ------------------------------------------------------------ zones

def test_zones_split_caps_and_wrong_side_skipped():
    zones = [
        _zone("resistance", 4117.4, 4124.6, 4117.4),
        _zone("resistance", 4078.3, 4086.8, 4078.3),
        _zone("resistance", 4034.5, 4041.1, 4034.5),
        _zone("resistance", 4045.0, 4050.0, 4045.0),  # 4th above: capped
        _zone("support", 3990.0, 3995.0, 3990.0),
        _zone("support", 3982.0, 3985.0, 3982.0),
        _zone("support", 3970.0, 3975.0, 3970.0),  # 3rd below: capped
        _zone("resistance", 3900.0, 3905.0, 3900.0),  # wrong side: skipped
        _zone("support", 4200.0, 4205.0, 4200.0),  # wrong side: skipped
    ]
    out = render_market_read(price=4015.0, hold=_hold(), zones=zones)
    assert "🔵 4034.5-4041.1: overhead supply" in out
    assert "🔵 4045.0-4050.0: overhead supply" in out
    assert "🔵 4078.3-4086.8: overhead supply" in out
    assert "4117.4-4124.6" not in out  # furthest above: capped, not hidden
    assert "🔵 3990.0-3995.0: demand" in out
    assert "🔵 3982.0-3985.0: demand" in out
    assert "3970.0-3975.0" not in out
    assert "3900.0-3905.0" not in out
    assert "4200.0-4205.0" not in out


def test_price_inside_zone():
    zones = [_zone("resistance", 4015.0, 4020.0, 4010.0)]
    out = render_market_read(price=4015.0, hold=_hold(), zones=zones)
    assert "🔵 4010.0-4020.0: price inside" in out
    assert "neutral: price inside 4010.0-4020.0" in out


def test_empty_book_and_no_price():
    out = render_market_read(price=4015.0, hold=_hold(), zones=[])
    assert "no active zones in book" in out
    assert "bullish: not produced (no overhead supply in book)" in out
    assert "bearish: not produced (no demand below in book)" in out
    assert "neutral: not produced (one-sided book)" in out
    out = render_market_read(price=None, hold=_hold(), zones=[])
    assert "key zones: not produced (no last price)" in out
    assert "scenarios: not produced (no last price)" in out
    assert out.splitlines()[0].startswith("trend:")  # trend needs no price


# --------------------------------------------------------- scenarios

def test_scenarios_derive_triggers_from_bands():
    zones = [
        _zone("resistance", 4015.2, 4018.0, 4014.0),
        _zone("resistance", 4034.5, 4041.1, 4034.5),
        _zone("support", 3990.0, 3995.0, 3990.0),
        _zone("support", 3982.0, 3985.0, 3982.0),
    ]
    out = render_market_read(price=4013.0, hold=_hold(), zones=zones)
    assert ("bullish: a sustained break and hold above 4018.0 could "
            "extend toward 4034.5-4041.1") in out
    assert ("bearish: rejection at 4014.0-4018.0 followed by a break "
            "under 3990.0 would favor a retest of 3982.0-3985.0") in out
    assert ("neutral: consolidation between 3990.0-3995.0 and "
            "4014.0-4018.0 would indicate a pause") in out


def test_scenarios_one_sided_book():
    zones = [_zone("resistance", 4034.5, 4041.1, 4034.5)]
    out = render_market_read(price=4015.0, hold=_hold(), zones=zones)
    assert "extend toward no further supply in book" in out
    assert "bearish: not produced (no demand below in book)" in out
    assert "neutral: not produced (one-sided book)" in out


# ------------------------------------------------------ runner + wiring

def _runner_with_state():
    r = object.__new__(MT5Runner)
    r._last_tick = {"mid": 4015.0}
    r._live_hold_state = SimpleNamespace(to_dict=lambda: _hold())
    r._structural_zones = [
        SimpleNamespace(tf="H1", side="resistance", level=4034.5,
                        top=4041.1, bottom=4034.5, score=90.0,
                        tier="MAJOR", status="active", touches=2),
        SimpleNamespace(tf="H1", side="support", level=3990.0,
                        top=3995.0, bottom=3990.0, score=85.0,
                        tier="MAJOR", status="active", touches=1),
        SimpleNamespace(tf="H1", side="resistance", level=4100.0,
                        top=4105.0, bottom=4100.0, score=80.0,
                        tier="MAJOR", status="broken", touches=3),
    ]
    return r


def test_runner_method_uses_zone_cache_and_hold_state():
    out = _runner_with_state().market_read_line()
    assert "bearish — lower highs and lower lows" in out
    assert "🔵 H1 4034.5-4041.1: overhead supply" in out
    assert "🔵 H1 3990.0-3995.0: demand" in out
    assert "4100.0-4105.0" not in out  # broken zone is not a key zone


def _update(text, chat_id=OWNER, update_id=1):
    return {"update_id": update_id,
            "message": {"message_id": 1,
                        "chat": {"id": chat_id, "type": "private"},
                        "text": text}}


def test_status_wiring_shows_market_read(tmp_path):
    ctx = CommandContext(journal=None, journal_path=None,
                         reports_dir=tmp_path / "reports")
    ctx.mt5_control = SimpleNamespace(
        market_read_line=lambda: "trend: bearish — test read",
        get_live_config=lambda: None,
        blocked_stats=lambda: {"legs": {}, "last": None})
    router = mt5_router(OWNER)
    res, _ = router.dispatch(_update("/status"), ctx)
    assert "Market Read" in res.text
    assert "trend: bearish — test read" in res.text


def test_status_wiring_skips_old_controllers(tmp_path):
    ctx = CommandContext(journal=None, journal_path=None,
                         reports_dir=tmp_path / "reports")
    ctx.mt5_control = SimpleNamespace(
        get_live_config=lambda: None,
        blocked_stats=lambda: {"legs": {}, "last": None})
    router = mt5_router(OWNER)
    res, _ = router.dispatch(_update("/status"), ctx)
    assert "Market Read" not in res.text


def test_marketread_standalone_command(tmp_path):
    ctx = CommandContext(journal=None, journal_path=None,
                         reports_dir=tmp_path / "reports")
    ctx.mt5_control = SimpleNamespace(
        market_read_line=lambda: "trend: bearish — test read")
    router = mt5_router(OWNER)
    res, _ = router.dispatch(_update("/marketread"), ctx)
    assert "Market Read" in res.text
    assert "trend: bearish — test read" in res.text
    # The standalone read carries no status sections.
    assert "MT5 ENGINE STATUS" not in res.text


def test_marketread_without_runner_is_graceful(tmp_path):
    ctx = CommandContext(journal=None, journal_path=None,
                         reports_dir=tmp_path / "reports")
    ctx.mt5_control = SimpleNamespace()
    router = mt5_router(OWNER)
    res, _ = router.dispatch(_update("/marketread"), ctx)
    assert "needs the live MT5 engine" in res.text
