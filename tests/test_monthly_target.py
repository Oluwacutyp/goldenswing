"""WS2 P7 — the owner-set monthly target line (desk + recap).

The target is DISPLAY-ONLY: the owner's ``monthly_target``
live-config key, set via the existing ``/set`` surface
(``/set monthly_target <amount>`` / ``/set monthly_target off``),
rendered as a month-to-date progress strip on the desk card and
the daily recap image, and as one line on the daily recap text —
only when set.  Unset (the default) every render is
byte-identical to what shipped before targets existed: the line
is ABSENT, never zero, never a placeholder.

Pinned here:

* config: defaults unset; ``_clean`` keeps a positive finite
  number, drops anything else (text / zero / negative / bool /
  NaN / inf) to unset; ``parse_set`` amount + off/none/clear,
  errors that never echo the supplied value; the store
  round-trip; apply/snapshot; ``dropped_monthly_target`` names
  the corrupt-on-disk case for the startup journal.
* reporting: ``monthly_target_progress`` — UTC calendar month
  through the recap day, from the same ``recap_trades`` machinery
  the recap renders; signed unclamped percentage; overshoot and
  a losing month read truthfully.
* text recap: unset == exactly today's output; set appends the
  one line; a day with no trades still shows the month's line.
* renders: absent key == explicit None == byte-identical;
  deterministic; set != unset with the bar's track/fill ink in
  the strip band; overshoot does not error.
* /desk: a control exposing the target merges the strip into
  the card + caption; a legacy control has no key and the card
  renders exactly as before.
* runner: ``set_live_config`` applies + persists + a fresh
  store reloads it; clearing persists None.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from goldenswing.execution.learning import TradeJournal  # noqa: E402
from goldenswing.execution.mt5_config import (  # noqa: E402
    LiveConfigStore, describe_change, dropped_monthly_target,
    parse_set)
from goldenswing.execution.reporting import (  # noqa: E402
    build_daily_recap, monthly_target_progress)

# Pinned palette token (charts/tiles.py BAR_IDLE) for the ink
# checks — the target bar's fill colour.
_BAR = (0x9A / 255, 0xA3 / 255, 0xB2 / 255)


def write_journal(path: Path, closes) -> TradeJournal:
    """Seed closes as (close-day, pnl_money) pairs through the
    journal's own append path — the record shape
    ``reporting.iter_paper_trades`` joins on (type / alert_id /
    ts), the same shape the desk suite seeds."""
    journal = TradeJournal(path)
    for idx, (day, pnl) in enumerate(closes):
        opened = datetime(day.year, day.month, day.day, 9, 0,
                          tzinfo=timezone.utc).isoformat()
        closed = datetime(day.year, day.month, day.day, 10, 0,
                          tzinfo=timezone.utc).isoformat()
        journal.append({
            "type": "open", "alert_id": f"tj{idx}",
            "instrument": "XAUUSD", "playbook": "qm_m5_native",
            "direction": 1, "entry": 4000.0, "stop": 3995.0,
            "volume": 0.01, "risk_money": 17.5, "ts": opened})
        journal.append({
            "type": "close", "alert_id": f"tj{idx}",
            "ticket": idx + 1, "exit": 4010.0, "reason": "target",
            "pnl_money": pnl, "r_net": pnl / 17.5, "hold_bars": 3,
            "ts": closed})
    return journal


def desk_stats(**over) -> dict:
    stats = {
        "day": "2026-10-13", "pnl": 8.75, "r_net": 0.52,
        "trades": 2, "wins": 1, "losses": 1, "win_rate": 50.0,
        "open_positions": 1, "equity": 67.24, "balance": 67.24,
        "currency": "USD", "drawdown_pct": 0.212,
        "drawdown_limit": 0.15, "halted": True,
        "halt_reason": "max drawdown hit"}
    stats.update(over)
    return stats


def target_prog(**over) -> dict:
    prog = {"monthly_target": 500.0, "month_pnl": 312.4,
            "month_trades": 9, "month_pct": 62.48,
            "month_label": "Oct 2026"}
    prog.update(over)
    return prog


def _ink_near_edge(png: bytes, box) -> int:
    """Count pixels close to the target bar's BAR_IDLE fill
    colour inside a figure-fraction box (x0, x1, y0, y1)."""
    from PIL import Image
    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size
    px = img.load()
    x0, x1, y0, y1 = box
    hits = 0
    for yy in range(int(h * (1 - y1)), int(h * (1 - y0))):
        for xx in range(int(w * x0), int(w * x1)):
            r, g, b = px[xx, yy]
            if (abs(r / 255 - _BAR[0]) < 0.03
                    and abs(g / 255 - _BAR[1]) < 0.03
                    and abs(b / 255 - _BAR[2]) < 0.03):
                hits += 1
    return hits


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.path = Path(self._tmp) / "mt5_config.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _store_with(self, payload) -> LiveConfigStore:
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        return LiveConfigStore(self.path)

    def test_default_is_unset(self):
        store = LiveConfigStore(self.path)  # no file at all
        self.assertIsNone(store.load()["monthly_target"])

    def test_valid_target_loads_as_float(self):
        store = self._store_with({"monthly_target": 500})
        self.assertEqual(store.load()["monthly_target"], 500.0)
        self.assertIsInstance(store.load()["monthly_target"], float)

    def test_invalid_values_read_unset_never_crash(self):
        for bad in ("abc", 0, -50, True, [500], {"x": 1}):
            store = self._store_with({"monthly_target": bad})
            self.assertIsNone(store.load()["monthly_target"],
                              msg=f"{bad!r} must read unset")
            self.assertTrue(dropped_monthly_target(
                {"monthly_target": bad}), msg=f"{bad!r} named")

    def test_nan_and_inf_in_raw_json_read_unset(self):
        # Python's json parser accepts NaN / Infinity literals;
        # a hand-edited file can carry them.
        self.path.write_text('{"monthly_target": NaN}',
                             encoding="utf-8")
        self.assertIsNone(LiveConfigStore(self.path)
                          .load()["monthly_target"])
        self.assertTrue(dropped_monthly_target(
            {"monthly_target": float("nan")}))
        self.assertTrue(dropped_monthly_target(
            {"monthly_target": float("inf")}))

    def test_absent_and_null_are_unset_but_not_drops(self):
        self.assertFalse(dropped_monthly_target({}))
        self.assertFalse(dropped_monthly_target(
            {"monthly_target": None}))
        self.assertFalse(dropped_monthly_target(
            {"monthly_target": 250.0}))
        self.assertFalse(dropped_monthly_target("not a dict"))

    def test_parse_set_amount_and_clear(self):
        change, err = parse_set("monthly_target", "500")
        self.assertIsNone(err)
        self.assertEqual(change,
                         {"field": "monthly_target", "value": 500.0})
        change, err = parse_set("monthly_target", "1,250.5".replace(
            ",", ""))
        self.assertIsNone(err)
        self.assertEqual(change["value"], 1250.5)
        for word in ("off", "none", "clear", "OFF"):
            change, err = parse_set("monthly_target", word)
            self.assertIsNone(err)
            self.assertEqual(change, {"field": "monthly_target",
                                      "value": None})

    def test_parse_set_rejects_without_echoing_value(self):
        # Rejection covers zero too; the no-echo assertion uses
        # distinctive values (a bare "0" is a substring of any
        # example amount, so it cannot prove echoing either way).
        change, err = parse_set("monthly_target", "0")
        self.assertIsNone(change)
        self.assertIsNotNone(err)
        for raw in ("-50", "abc", "1e999"):
            change, err = parse_set("monthly_target", raw)
            self.assertIsNone(change)
            self.assertIsNotNone(err)
            self.assertNotIn(raw, err)

    def test_describe_change(self):
        self.assertEqual(
            describe_change({"field": "monthly_target",
                             "value": 500.0}),
            "monthly_target = 500.00")
        self.assertEqual(
            describe_change({"field": "monthly_target",
                             "value": None}),
            "monthly_target = off (target line hidden)")

    def test_save_load_round_trip_and_clear(self):
        store = LiveConfigStore(self.path)
        vals = store.load()
        vals["monthly_target"] = 750.0
        store.save(vals)
        self.assertEqual(LiveConfigStore(self.path)
                         .load()["monthly_target"], 750.0)
        vals["monthly_target"] = None
        store.save(vals)
        self.assertIsNone(LiveConfigStore(self.path)
                          .load()["monthly_target"])

    def test_apply_and_snapshot(self):
        from goldenswing.execution.mt5_runner import MT5RunnerConfig
        cfg = MT5RunnerConfig()
        store = self._store_with({"monthly_target": 300})
        store.apply_to(cfg)
        self.assertEqual(cfg.monthly_target, 300.0)
        cfg.monthly_target = None
        snap = store.snapshot_from(cfg)
        self.assertIsNone(snap["monthly_target"])
        cfg.monthly_target = 125.5
        snap = store.snapshot_from(cfg)
        self.assertEqual(snap["monthly_target"], 125.5)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.path = Path(self._tmp) / "journal.jsonl"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_month_window_boundaries(self):
        journal = write_journal(self.path, [
            (date(2026, 9, 30), 175.0),  # previous month: excluded
            (date(2026, 10, 1), 35.0),   # month start: included
            (date(2026, 10, 13), 8.75),  # the recap day: included
            (date(2026, 10, 14), 70.0),  # after the day: excluded
            (date(2026, 11, 1), 140.0),  # next month: excluded
        ])
        prog = monthly_target_progress(
            journal, date(2026, 10, 13), target=437.5)
        # 35.0 + 8.75 = 43.75 of 437.5 = exactly 10%.
        self.assertEqual(prog["month_pnl"], 43.75)
        self.assertEqual(prog["month_trades"], 2)
        self.assertEqual(prog["month_pct"], 10.0)
        self.assertEqual(prog["monthly_target"], 437.5)
        self.assertEqual(prog["month_label"], "Oct 2026")

    def test_overshoot_is_unclamped(self):
        journal = write_journal(self.path, [
            (date(2026, 10, 5), 140.0)])
        prog = monthly_target_progress(
            journal, date(2026, 10, 13), target=100.0)
        self.assertEqual(prog["month_pnl"], 140.0)
        self.assertEqual(prog["month_pct"], 140.0)

    def test_negative_month_reads_negative(self):
        journal = write_journal(self.path, [
            (date(2026, 10, 5), -35.0)])
        prog = monthly_target_progress(
            journal, date(2026, 10, 13), target=500.0)
        self.assertEqual(prog["month_pnl"], -35.0)
        self.assertEqual(prog["month_pct"], -7.0)

    def test_empty_month_is_zero_not_missing(self):
        journal = write_journal(self.path, [])
        prog = monthly_target_progress(
            journal, date(2026, 10, 13), target=500.0)
        self.assertEqual(prog["month_pnl"], 0.0)
        self.assertEqual(prog["month_trades"], 0)
        self.assertEqual(prog["month_pct"], 0.0)


class TextRecapTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.path = Path(self._tmp) / "journal.jsonl"
        self.journal = write_journal(self.path, [
            (date(2026, 10, 12), 35.0),
            (date(2026, 10, 13), 8.75)])

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_unset_text_is_exactly_todays_output(self):
        plain = build_daily_recap(self.journal, date(2026, 10, 13))
        explicit = build_daily_recap(
            self.journal, date(2026, 10, 13), monthly_target=None)
        self.assertEqual(plain, explicit)
        self.assertNotIn("Month target", plain)

    def test_set_text_appends_the_one_line(self):
        text = build_daily_recap(
            self.journal, date(2026, 10, 13), monthly_target=437.5)
        # Month: 35.0 + 8.75 = 43.75 of 437.50 = +10%.
        self.assertIn(
            "Month target (Oct 2026): +43.75 of 437.50 — +10% · "
            "2 closed trades this month", text)

    def test_quiet_month_day_still_shows_line(self):
        text = build_daily_recap(
            self.journal, date(2026, 10, 20), monthly_target=437.5)
        self.assertIn("Month target (Oct 2026): +43.75 of 437.50",
                      text)


class RenderTests(unittest.TestCase):
    def test_desk_unset_absent_equals_explicit_none(self):
        from goldenswing.charts.desk import render_desk_card
        absent = render_desk_card(desk_stats())
        explicit = render_desk_card(
            desk_stats(monthly_target=None, month_pnl=None,
                       month_pct=None, month_label=None))
        self.assertEqual(absent, explicit)
        self.assertEqual(absent, render_desk_card(desk_stats()))

    def test_recap_unset_absent_equals_explicit_none(self):
        from goldenswing.charts.recap import render_daily_recap
        absent = render_daily_recap(desk_stats())
        explicit = render_daily_recap(
            desk_stats(monthly_target=None, month_pnl=None,
                       month_pct=None, month_label=None))
        self.assertEqual(absent, explicit)
        self.assertEqual(absent, render_daily_recap(desk_stats()))

    def test_desk_set_draws_the_bar(self):
        from goldenswing.charts.desk import render_desk_card
        base = render_desk_card(desk_stats())
        with_t = render_desk_card(desk_stats(**target_prog()))
        self.assertNotEqual(base, with_t)
        self.assertEqual(with_t,
                         render_desk_card(desk_stats(**target_prog())))
        # The bar band: figure x 0.315-0.60, y inside the strip
        # (clear of the unset card's tile outline at y 0.27).
        band = (0.315, 0.60, 0.272, 0.29)
        self.assertLess(_ink_near_edge(base, band), 200)
        self.assertGreater(_ink_near_edge(with_t, band), 800)

    def test_recap_set_draws_the_bar(self):
        from goldenswing.charts.recap import render_daily_recap
        base = render_daily_recap(desk_stats())
        with_t = render_daily_recap(desk_stats(**target_prog()))
        self.assertNotEqual(base, with_t)
        band = (0.315, 0.60, 0.302, 0.318)
        self.assertLess(_ink_near_edge(base, band), 200)
        self.assertGreater(_ink_near_edge(with_t, band), 800)

    def test_overshoot_and_negative_render(self):
        from goldenswing.charts.desk import render_desk_card
        from goldenswing.charts.recap import render_daily_recap
        over = desk_stats(**target_prog(
            month_pnl=685.0, month_pct=137.0))
        neg = desk_stats(**target_prog(
            month_pnl=-40.0, month_pct=-8.0))
        self.assertNotEqual(render_desk_card(over),
                            render_desk_card(desk_stats()))
        self.assertNotEqual(render_daily_recap(over),
                            render_daily_recap(desk_stats()))
        self.assertNotEqual(render_desk_card(neg),
                            render_desk_card(over))

    def test_set_but_month_unavailable_shows_honest_strip(self):
        from goldenswing.charts.desk import render_desk_card
        stats = desk_stats(monthly_target=500.0, month_pnl=None,
                           month_pct=None, month_label="Oct 2026")
        self.assertNotEqual(render_desk_card(stats),
                            render_desk_card(desk_stats()))


class _DeskCtrl:
    """The desk suite's control shape (account_vitals + guards),
    optionally carrying the live-config read."""

    def __init__(self, *, target="absent"):
        from types import SimpleNamespace
        self.guards = SimpleNamespace(
            state={"day": date.today().isoformat()})
        if target != "absent":
            self.get_live_config = lambda: {
                "monthly_target": target}

    def account_vitals(self):
        return None


class DeskHandlerTests(unittest.TestCase):
    def _run_desk(self, ctrl):
        from goldenswing.execution.mt5_commands import (
            MT5_COMMANDS, CommandContext)
        tmp = tempfile.mkdtemp()
        today = date.today()
        first = today.replace(day=1)
        # Two closes inside the CURRENT month summing +35.00 —
        # the desk day is always the real today, so the seed
        # rides the live calendar like the desk suite's does.
        journal = write_journal(Path(tmp) / "journal.jsonl", [
            (first, 17.5), (today, 17.5)])
        ctx = CommandContext(journal=journal, mt5_control=ctrl,
                             live_positions=[])
        return MT5_COMMANDS["/desk"][1](ctx, [])

    def test_desk_with_target_shows_strip_and_caption(self):
        res = self._run_desk(_DeskCtrl(target=70.0))
        self.assertIsNotNone(res.photo)
        self.assertIn("🎯", res.caption)
        self.assertIn("+35.00 of 70.00", res.caption)
        self.assertIn("(+50%)", res.caption)
        self.assertNotIn("\n", res.caption)

    def test_desk_with_legacy_control_renders_as_before(self):
        res = self._run_desk(_DeskCtrl())  # no get_live_config
        self.assertIsNotNone(res.photo)
        self.assertNotIn("🎯", res.caption)


class RunnerWiringTests(unittest.TestCase):
    def test_set_live_config_applies_and_persists(self):
        from goldenswing.execution.mt5_runner import (
            MT5Runner, MT5RunnerConfig)
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "mt5_config.json"
        runner = MT5Runner.__new__(MT5Runner)
        runner.cfg = MT5RunnerConfig()
        runner._config_store = LiveConfigStore(path)
        runner.journal = TradeJournal(Path(tmp) / "journal.jsonl")
        runner.set_live_config(
            {"field": "monthly_target", "value": 500.0})
        self.assertEqual(runner.cfg.monthly_target, 500.0)
        # Persisted: a fresh store over the same file reloads it.
        self.assertEqual(LiveConfigStore(path)
                         .load()["monthly_target"], 500.0)
        runner.set_live_config(
            {"field": "monthly_target", "value": None})
        self.assertIsNone(runner.cfg.monthly_target)
        self.assertIsNone(LiveConfigStore(path)
                          .load()["monthly_target"])

    def test_config_text_shows_target_line(self):
        from goldenswing.execution.mt5_commands import (
            CommandContext, _config_text)
        ctrl = mock.Mock()
        ctrl.get_live_config = lambda: {
            "max_layers": 3, "manipulation_filter": True,
            "pending_expiry_bars": 4, "monthly_target": 500.0,
            "paused": False, "risk_pct": None,
            "disabled_playbooks": []}
        ctx = CommandContext(journal=None, mt5_control=ctrl)
        text = _config_text(ctx)
        self.assertIn("`monthly_target` = `500.00`", text)
        ctrl.get_live_config = lambda: {
            "max_layers": 3, "manipulation_filter": True,
            "pending_expiry_bars": 4, "monthly_target": None,
            "paused": False, "risk_pct": None,
            "disabled_playbooks": []}
        text = _config_text(ctx)
        self.assertIn("off (no target line)", text)


if __name__ == "__main__":
    unittest.main()
