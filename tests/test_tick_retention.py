"""Bounded tick retention + disarmed-ADR-gauge rendering.

Tick warehouse diagnosis (bb140fa): the disk store was UNBOUNDED —
TickLogger appends one CSV per UTC day forever and nothing ever
pruned it.  Memory was already bounded (_last_tick single,
_spread_samples capped at 600, TickManager per-position only), and
live management never reads the CSVs (only harvest/export do) —
so the bound lives on the disk warehouse, with the open-position
guard: files from a live position's open day onward are never
evicted while it is open.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

from goldenswing.execution import mt5_runner as mr
from goldenswing.feed.tick_log import TickLogger, tick_csv_day
from goldenswing.live import day_context as dc


def _mk_files(directory, days):
    for day in days:
        (directory / f"mt5_ticks_{day}.csv").write_text(
            "time,bid,ask,tick_volume,real_volume,spread\n")


def _days_2026_oct():
    return [f"202610{day:02d}" for day in range(1, 11)]


def test_prune_enforces_retain_days(tmp_path):
    logger = TickLogger(tmp_path, retain_days=3)
    _mk_files(tmp_path, _days_2026_oct())
    deleted = logger.prune(today="20261010")
    assert deleted == [f"mt5_ticks_202610{day:02d}.csv"
                       for day in range(1, 8)]
    remaining = sorted(p.name for p in tmp_path.glob("mt5_ticks_*.csv"))
    assert remaining == ["mt5_ticks_20261008.csv",
                         "mt5_ticks_20261009.csv",
                         "mt5_ticks_20261010.csv"]


def test_prune_keep_since_extends_retention_backward(tmp_path):
    # A position opened 20261002 keeps its open day onward even
    # though retention alone (3 days) would evict 02..07.
    logger = TickLogger(tmp_path, retain_days=3)
    _mk_files(tmp_path, _days_2026_oct())
    deleted = logger.prune(keep_since_day="20261002", today="20261010")
    assert deleted == ["mt5_ticks_20261001.csv"]
    assert (tmp_path / "mt5_ticks_20261002.csv").exists()


def test_prune_never_deletes_today(tmp_path):
    logger = TickLogger(tmp_path, retain_days=0)
    _mk_files(tmp_path, _days_2026_oct())
    deleted = logger.prune(today="20261010")
    assert "mt5_ticks_20261010.csv" not in deleted
    assert (tmp_path / "mt5_ticks_20261010.csv").exists()
    assert len(deleted) == 9


def test_prune_skips_on_bad_keep_since(tmp_path):
    logger = TickLogger(tmp_path, retain_days=3)
    _mk_files(tmp_path, _days_2026_oct())
    assert logger.prune(keep_since_day="not-a-day",
                        today="20261010") == []
    assert len(list(tmp_path.glob("mt5_ticks_*.csv"))) == 10


def test_prune_ignores_non_tick_files(tmp_path):
    logger = TickLogger(tmp_path, retain_days=0)
    _mk_files(tmp_path, _days_2026_oct())
    (tmp_path / "mt5_ticks_notes.txt").write_text("keep me")
    (tmp_path / "mt5_ticks_2026.csv").write_text("keep me")
    (tmp_path / "other.csv").write_text("keep me")
    logger.prune(today="20261010")
    assert (tmp_path / "mt5_ticks_notes.txt").exists()
    assert (tmp_path / "mt5_ticks_2026.csv").exists()
    assert (tmp_path / "other.csv").exists()
    assert tick_csv_day("mt5_ticks_2026.csv") is None


class _FakeTickManager:
    def __init__(self, opened_by_ticket):
        self._opened = opened_by_ticket

    def tickets(self):
        return list(self._opened)

    def position(self, ticket):
        return SimpleNamespace(opened_at=self._opened[ticket])


class _FakeJournal:
    def __init__(self):
        self.notes = []

    def log_note(self, msg):
        self.notes.append(msg)


def _runner_fake(tmp_path, opened_by_ticket, retain_days=30):
    logger = TickLogger(tmp_path, retain_days=retain_days)
    return SimpleNamespace(
        _tick_logger=logger,
        tick_manager=_FakeTickManager(opened_by_ticket),
        journal=_FakeJournal())


def _today():
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _rel_day(delta):
    from datetime import timedelta
    return (datetime.now(timezone.utc).date()
            + timedelta(days=delta)).strftime("%Y%m%d")


def test_runner_prune_skips_adopted_position(tmp_path):
    # Adopted positions carry opened_at=None: unknown span, so
    # pruning skips entirely (fail-closed) rather than guessing.
    fake = _runner_fake(tmp_path, {12345: None})
    _mk_files(tmp_path, [_rel_day(d) for d in range(-40, 1)])
    deleted = mr.MT5Runner._prune_tick_csvs(fake)
    assert deleted == []
    assert len(list(tmp_path.glob("mt5_ticks_*.csv"))) == 41
    assert any("SKIPPED" in n for n in fake.journal.notes)


def test_runner_prune_keeps_open_position_days(tmp_path):
    opened = datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()
    fake = _runner_fake(tmp_path, {12345: opened}, retain_days=30)
    old = "20260801"
    _mk_files(tmp_path, [old, "20260901", _rel_day(-40), _rel_day(0)])
    deleted = mr.MT5Runner._prune_tick_csvs(fake)
    assert f"mt5_ticks_{old}.csv" in deleted
    assert (tmp_path / "mt5_ticks_20260901.csv").exists()
    assert any("tick-CSV prune deleted" in n
               for n in fake.journal.notes)


def _day_line_fake(**dc_kw):
    kw = dict(day_key="2026-10-10", day_type=dc.DAY_TREND,
              ib_high=2010.0, ib_low=2002.0, ib_complete=True)
    kw.update(dc_kw)
    return SimpleNamespace(
        _day_ctx=dc.DayContext(**kw),
        _posture=None,
        _last_tick=None,
        _spread_ratio=1.0,
        _ny_checkpoint="",
        _smt_read=None)


def test_day_line_disarmed_render(tmp_path):
    # The P3 stub case: 0.5-point ADR against a real day reads
    # thousands of percent consumed — the status line must never
    # print that pct once the gauge is disarmed.
    line = mr.MT5Runner.day_line(_day_line_fake(
        adr=0.5, adr_days=0, adr_consumed_pct=5549.0,
        adr_state="exhausted"))
    assert "ADR — (disarmed)" in line
    assert "5549" not in line


def test_day_line_valid_gauge_still_renders_pct(tmp_path):
    line = mr.MT5Runner.day_line(_day_line_fake(
        adr=12.0, adr_days=14, adr_consumed_pct=62.0,
        adr_state="working"))
    assert "ADR 62% (working)" in line
    assert "disarmed" not in line


def test_decision_context_keeps_numeric_adr_pct(tmp_path):
    # Only the human-facing day_line string changes: the journal
    # context keeps its numeric adr_pct shape untouched.
    fake = SimpleNamespace(
        _bedrock=None, _posture=None,
        _day_ctx=dc.DayContext(
            day_key="2026-10-10", day_type=dc.DAY_TREND,
            adr=0.5, adr_days=0, adr_consumed_pct=5549.0,
            adr_state="exhausted"),
        _ny_checkpoint="", _smt_read=None, _last_clock=None)
    ctx = mr.MT5Runner._decision_context(fake)
    assert ctx["adr_pct"] == 5549.0
    assert ctx["adr_state"] == "exhausted"
