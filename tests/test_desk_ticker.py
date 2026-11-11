"""Desk market ticker strip pins (design #12 remainder).

The desk build shipped the card without its optional market
ticker strip (XAUUSD/DXY/session): the shared stats dict
carried no price/DXY seam, and inventing market values would
have broken the card's honesty contract.  The seam now exists —
the runner's ``desk_ticker`` (its own last tick mid + the
health monitor's freshness class, the DXY tracker's last
closed H1 close, the session off the tick's clock) — merged
into the desk stats by the ONE /desk assembly, drawn by
charts/desk.py as a one-line strip, and carried in the /desk
caption in the same words.

Honesty pins: a stale price is NAMED stale with its age (never
a bare price dressed as live); a closed market is named; no
tick yet renders a dash + reason, never a number; a dead DXY
feed renders its status, never a value; without the seam the
card and caption are exactly the ones that shipped before.
"""
from __future__ import annotations

import io
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image

from goldenswing.alerts.markup import render_messages
from goldenswing.charts.desk import render_desk_card, ticker_line
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.mt5_commands import (
    MT5_COMMANDS, CommandContext, _desk_caption, _desk_day_stats)
from goldenswing.execution.mt5_runner import MT5Runner
from goldenswing.live.dxy import DXYState
from goldenswing.live.health import HealthMonitor


# ---------------------------------------------------------------- fixtures

def _full_stats(**over):
    stats = {"day": "2026-10-13", "pnl": 8.75, "r_net": 0.5,
             "trades": 2, "wins": 1, "losses": 1, "win_rate": 50.0,
             "open_positions": 1, "equity": 67.24, "balance": 67.24,
             "currency": "USD", "drawdown_pct": 0.212,
             "drawdown_limit": 0.15, "halted": True,
             "halt_reason": "max drawdown hit"}
    stats.update(over)
    return stats


def _live_ticker(**over):
    ticker = {"symbol": "XAUUSD", "price": 2645.30,
              "tick_state": "live", "tick_age_s": 3.0,
              "tick_time": "2026-10-13T13:00:00+00:00",
              "dxy": 106.42, "dxy_time": "2026-10-13T12:00:00+00:00",
              "dxy_available": True,
              "dxy_status": "measured (50 bars)",
              "session": "overlap"}
    ticker.update(over)
    return ticker


def _band_ink(png: bytes, y0f: float, y1f: float) -> int:
    """Dark-pixel count inside a horizontal band of the card."""
    im = Image.open(io.BytesIO(png)).convert("L")
    w, h = im.size
    px = im.load()
    n = 0
    for y in range(int(h * (1 - y1f)), int(h * (1 - y0f))):
        for x in range(0, w, 3):
            if px[x, y] < 200:
                n += 1
    return n


# ------------------------------------------------------------ ticker words

class TickerLineTests(unittest.TestCase):
    def test_live_line_carries_price_dxy_session(self):
        line = ticker_line(_live_ticker())
        self.assertEqual(
            line,
            "XAUUSD 2,645.30 · DXY 106.42 · London/NY overlap")

    def test_stale_price_is_named_with_its_age_never_bare(self):
        line = ticker_line(_live_ticker(tick_state="stale",
                                        tick_age_s=754.0))
        self.assertIn("XAUUSD 2,645.30 (stale — last tick 13m ago)",
                      line)
        # The bare live form must not appear anywhere in it.
        self.assertNotIn("XAUUSD 2,645.30 ·", line)

    def test_closed_market_price_is_named_closed(self):
        line = ticker_line(_live_ticker(
            tick_state="closed_expected", tick_age_s=9000.0))
        self.assertIn("(market closed — last tick 2.5h ago)", line)

    def test_no_tick_yet_is_a_dash_and_reason_never_a_number(self):
        line = ticker_line(_live_ticker(
            price=None, tick_state="no_ticks", tick_age_s=None))
        self.assertIn("XAUUSD — no tick yet", line)
        self.assertNotIn("0.00", line)

    def test_no_tick_on_a_closed_market_says_closed(self):
        line = ticker_line(_live_ticker(
            price=None, tick_state="closed_expected",
            tick_age_s=None))
        self.assertIn("XAUUSD — market closed — no tick yet", line)

    def test_dxy_unavailable_renders_its_status_never_a_value(self):
        line = ticker_line(_live_ticker(
            dxy=None, dxy_available=False,
            dxy_status="no DXY data yet"))
        self.assertIn("DXY — no DXY data yet", line)

    def test_dxy_fetch_failure_reason_is_shortened(self):
        line = ticker_line(_live_ticker(
            dxy=None, dxy_available=False,
            dxy_status="fetch failed: terminal not connected (long)"))
        self.assertIn("DXY — feed fetch failed", line)
        self.assertNotIn("terminal not connected", line)


# ---------------------------------------------------------------- renderer

class DeskCardStripTests(unittest.TestCase):
    def test_strip_draws_only_when_the_ticker_is_present(self):
        plain = render_desk_card(_full_stats())
        with_ticker = render_desk_card(
            _full_stats(ticker=_live_ticker()))
        self.assertNotEqual(plain, with_ticker)
        ink_plain = _band_ink(plain, 0.245, 0.315)
        ink_ticker = _band_ink(with_ticker, 0.245, 0.315)
        self.assertGreater(ink_ticker, ink_plain * 4)
        # Shape + determinism hold with the strip on.
        self.assertEqual(
            Image.open(io.BytesIO(with_ticker)).size, (1200, 627))
        self.assertEqual(
            render_desk_card(_full_stats(ticker=_live_ticker())),
            with_ticker)

    def test_unavailable_ticker_still_draws_the_strip_named(self):
        # A dead feed must SAY so on the card — the strip is
        # present (the seam supplied the dict) with every market
        # value unavailable, and the rest of the card renders.
        dead = _live_ticker(price=None, tick_state="no_ticks",
                            tick_age_s=None, dxy=None,
                            dxy_available=False,
                            dxy_status="no DXY data yet")
        png = render_desk_card(_full_stats(ticker=dead))
        self.assertEqual(Image.open(io.BytesIO(png)).size,
                         (1200, 627))
        self.assertGreater(_band_ink(png, 0.245, 0.315),
                           _band_ink(render_desk_card(_full_stats()),
                                     0.245, 0.315) * 4)

    def test_ticker_and_target_strips_coexist(self):
        target = {"monthly_target": 500.0, "month_pnl": 125.0,
                  "month_pct": 25.0, "month_label": "October"}
        both = render_desk_card(_full_stats(ticker=_live_ticker(),
                                            **target))
        ticker_only = render_desk_card(
            _full_stats(ticker=_live_ticker()))
        target_only = render_desk_card(_full_stats(**target))
        self.assertNotEqual(both, ticker_only)
        self.assertNotEqual(both, target_only)
        # The target strip moved up one slot: its band carries
        # ink in the both-render where the target-only render's
        # same band is its (lower) strip — both bands inked.
        self.assertGreater(_band_ink(both, 0.33, 0.395), 100)
        self.assertGreater(_band_ink(both, 0.245, 0.31), 100)


# -------------------------------------------------------------- runner seam

def _bare_runner(*, tick=None, health=None, dxy=None):
    runner = MT5Runner.__new__(MT5Runner)
    runner.cfg = SimpleNamespace(symbol="XAUUSD")
    runner._last_tick = tick
    runner._health = health if health is not None else HealthMonitor()
    runner._dxy = dxy
    runner._refresh_health_market = lambda: None
    return runner


def _dxy_tracker(*, available=True, closes=(106.10, 106.42),
                 status="measured (50 bars)"):
    frame = None
    if closes is not None:
        idx = pd.DatetimeIndex(
            ["2026-10-13T11:00:00+00:00", "2026-10-13T12:00:00+00:00"])
        frame = pd.DataFrame({"close": list(closes)}, index=idx)
    state = DXYState(corr=-0.42, bias=1, available=available,
                     status=status, n_bars=50)
    return SimpleNamespace(state=state, frame=frame)


class RunnerTickerTests(unittest.TestCase):
    def test_live_tick_fields_come_from_the_runners_own_tick(self):
        now = datetime.now(timezone.utc)
        health = HealthMonitor()
        health.note_market_open(True)
        health.note_tick(now - timedelta(seconds=3))
        tick = {"mid": 2645.30, "time": pd.Timestamp(
            "2026-10-13T13:00:00+00:00")}
        out = _bare_runner(tick=tick, health=health,
                           dxy=_dxy_tracker()).desk_ticker()
        self.assertEqual(out["symbol"], "XAUUSD")
        self.assertEqual(out["price"], 2645.30)
        self.assertEqual(out["tick_state"], "live")
        self.assertLess(out["tick_age_s"], 10)
        # DXY: the tracker's last CLOSED H1 close, not a fetch.
        self.assertEqual(out["dxy"], 106.42)
        self.assertEqual(out["dxy_time"], "2026-10-13T12:00:00+00:00")
        self.assertTrue(out["dxy_available"])
        # Session off the tick's own clock: 13:00 UTC = overlap.
        self.assertEqual(out["session"], "overlap")

    def test_no_tick_yet_price_is_none_with_a_state_word(self):
        health = HealthMonitor()
        health.note_market_open(True)
        out = _bare_runner(tick=None, health=health,
                           dxy=_dxy_tracker(
                               available=False, closes=None,
                               status="no DXY data yet")).desk_ticker()
        self.assertIsNone(out["price"])
        # Calendar-dependent class (no_ticks on an open weekday,
        # closed_expected across the weekly close) — either way
        # it is a named state, and the words name it.
        self.assertIn(out["tick_state"],
                      ("no_ticks", "closed_expected"))
        self.assertIsNone(out["dxy"])
        self.assertEqual(out["dxy_status"], "no DXY data yet")
        line = ticker_line(out)
        self.assertIn("XAUUSD — ", line)
        self.assertNotIn("0.00", line)

    def test_unavailable_dxy_keeps_its_frame_unread(self):
        # Tracker marked unavailable (a failed refresh): the
        # retained frame is stale-marked by state.available and
        # must NOT be dressed as a current DXY value.
        tracker = _dxy_tracker(available=False,
                               status="fetch failed: boom")
        out = _bare_runner(tick={"mid": 2600.0, "time": pd.Timestamp(
            "2026-10-13T13:00:00+00:00")},
            dxy=tracker).desk_ticker()
        self.assertIsNone(out["dxy"])
        self.assertEqual(out["dxy_status"], "fetch failed: boom")


# ------------------------------------------------------------- desk assembly

class _TickerCtrl:
    """Fake control: vitals/config absent, ticker seam present."""

    def __init__(self, ticker=None, *, raises=False):
        self._ticker = ticker
        self._raises = raises

    def desk_ticker(self):
        if self._raises:
            raise RuntimeError("seam exploded")
        return self._ticker


class _LegacyCtrl:
    """Pre-ticker control: no desk_ticker seam at all."""


def _desk_ctx(ctrl, tmp):
    journal = TradeJournal(Path(tmp) / "journal.jsonl")
    return CommandContext(journal=journal, mt5_control=ctrl,
                          live_positions=[])


class DeskAssemblyTests(unittest.TestCase):
    def test_stats_gain_the_ticker_from_the_runner_seam(self):
        tmp = tempfile.mkdtemp()
        stats, _today = _desk_day_stats(
            _desk_ctx(_TickerCtrl(_live_ticker()), tmp))
        self.assertEqual(stats["ticker"]["price"], 2645.30)

    def test_legacy_runner_stats_carry_no_ticker(self):
        tmp = tempfile.mkdtemp()
        stats, _today = _desk_day_stats(
            _desk_ctx(_LegacyCtrl(), tmp))
        self.assertNotIn("ticker", stats)

    def test_a_raising_seam_never_takes_the_desk_down(self):
        tmp = tempfile.mkdtemp()
        stats, _today = _desk_day_stats(
            _desk_ctx(_TickerCtrl(raises=True), tmp))
        self.assertIsNotNone(stats)
        self.assertNotIn("ticker", stats)

    def test_an_unavailable_ticker_dict_passes_through_whole(self):
        tmp = tempfile.mkdtemp()
        dead = _live_ticker(price=None, tick_state="no_ticks",
                            tick_age_s=None, dxy=None,
                            dxy_available=False,
                            dxy_status="no DXY data yet")
        stats, _today = _desk_day_stats(
            _desk_ctx(_TickerCtrl(dead), tmp))
        self.assertIsNone(stats["ticker"]["price"])
        self.assertEqual(stats["ticker"]["tick_state"], "no_ticks")


# ------------------------------------------------------------ /desk surface

class DeskCommandTests(unittest.TestCase):
    def _run_desk(self, ctrl):
        tmp = tempfile.mkdtemp()
        journal = TradeJournal(Path(tmp) / "journal.jsonl")
        ctx = CommandContext(journal=journal, mt5_control=ctrl,
                             live_positions=[])
        return MT5_COMMANDS["/desk"][1](ctx, [])

    def test_desk_with_a_live_feed_carries_the_ticker(self):
        res = self._run_desk(_TickerCtrl(_live_ticker()))
        self.assertIsNotNone(res.photo)
        self.assertEqual(Image.open(io.BytesIO(res.photo)).size,
                         (1200, 627))
        self.assertIn("📈 XAUUSD 2,645.30", res.caption)
        self.assertIn("DXY 106.42", res.caption)
        self.assertIn("London/NY overlap", res.caption)
        self.assertNotIn("\n", res.caption)

    def test_desk_with_no_feed_still_renders_and_names_it(self):
        dead = _live_ticker(price=None, tick_state="no_ticks",
                            tick_age_s=None, dxy=None,
                            dxy_available=False,
                            dxy_status="no DXY data yet")
        res = self._run_desk(_TickerCtrl(dead))
        self.assertIsNotNone(res.photo)
        self.assertIn("XAUUSD — no tick yet", res.caption)
        self.assertIn("DXY — no DXY data yet", res.caption)
        # The ticker segment itself carries no number standing
        # in for an unread market value.  (The caption's own
        # "+0.00" day P&L is the shipped contract's REAL zero —
        # no closes yet — not a market value.)
        ticker_seg = res.caption.split("📈", 1)[1]
        self.assertNotIn("0.00", ticker_seg)

    def test_desk_with_a_legacy_runner_is_the_shipped_surface(self):
        res = self._run_desk(_LegacyCtrl())
        self.assertIsNotNone(res.photo)
        self.assertNotIn("📈", res.caption)
        self.assertNotIn("XAUUSD", res.caption)

    def test_caption_without_a_ticker_has_no_market_segment(self):
        caption = _desk_caption(_full_stats())
        self.assertNotIn("XAUUSD", caption)
        self.assertNotIn("📈", caption)

    def test_caption_survives_the_markup_boundary_whole(self):
        # The /desk caption through the Telegram markup renderer:
        # every part within the limit, no ellipsis introduced,
        # the ticker words present whole in the joined parts.
        caption = _desk_caption(_full_stats(
            ticker=_live_ticker(),
            monthly_target=500.0, month_pnl=125.0, month_pct=25.0,
            month_label="October"))
        parts = render_messages("🖥️ " + caption)
        self.assertTrue(parts)
        for part in parts:
            self.assertLessEqual(len(part), 4096)
            self.assertNotIn("…", part)
        joined = " ".join(parts)
        self.assertIn("XAUUSD", joined)
        self.assertIn("2,645.30", joined)


if __name__ == "__main__":
    unittest.main()
