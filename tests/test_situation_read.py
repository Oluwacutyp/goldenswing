"""GS /status §9 — the MarketSituation surface rows.

The rows are the /status promises of the MarketSituation
programme (SPEC §9 pass plan + the Pass 1 ship remainder,
"GS /status market-read surface (§9) not built"):

1. per-TF state lines — Pass 1 demo proof.  Built here.
2. relationship lines — Pass 1 demo proof.  Built here.
3. the §3.3 situation blocks (regime / location / activity /
   cross_asset / data_quality) rendered from the snapshot.
   Built here.
4. forecasts block (F1 outlook + F2 book + F3 battery + E3
   trust) — already rendered by forecast_line; pinned in the
   forecast suites, not re-pinned here.
5. effective route-authority table — shipped Pass 5
   (judgment_authority_line), pinned in its own suites.
6. whole-book evaluation blocks — shipped Pass 9
   (book_evaluation / coverage_report / book_grading lines),
   pinned in their own suites.

Pins here: the pure render over REAL builder output (no
hand-made snapshot dicts for the main path), the honesty
edges (absent stays absent, in words; never 0.00, never a
bare None), the /status wiring (guarded section beside
Market Read), and the runner seam's three states (no
snapshot / disabled / snapshot present).
"""
import pandas as pd
import pytest

from goldenswing.alerts.commands import CommandContext
from goldenswing.execution.mt5_commands import _mt5_status
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.situation_read import render_situation_read
from goldenswing.live.situation import SituationKeeper, build_situation
from tests.test_situation import TF_FREQ, flat_frame, full_stack, stack_of

# Runner-shaped §3.3 blocks (the shapes _situation_blocks
# supplies on the live path).
BLOCKS = {
    "regime": {"day": {"day_type": "trend day", "adr_pct": 62.0,
                       "adr_state": "normal",
                       "opening_type": "london drive",
                       "prior_day": {"vah": 1.0, "val": 2.0,
                                     "poc": 3.0}}},
    "location": {"price": 2136.5, "magnet_room_r": None,
                 "value": {"poc": 2130.0, "vah": 2140.0,
                           "val": 2120.0, "developing": True},
                 "zones_above": [], "zones_below": [],
                 "pools_above": [{"level": 2145.0}],
                 "pools_below": [],
                 "premium_discount": {"zone": "premium",
                                      "equilibrium": 2100.0,
                                      "range_high": 2160.0,
                                      "range_low": 2040.0},
                 "day_levels": {"ib_high": 2142.0, "ib_low": 2128.0,
                                "overnight_high": 2144.0,
                                "overnight_low": 2125.0},
                 "opening_range": {"high": 2142.0, "low": 2128.0,
                                   "mid": 2135.0, "width_atr": 1.4}},
    "activity": {"session": "london", "killzone": "london_open",
                 "spread": 0.3, "spread_ratio": 0.9,
                 "rvol": {"H1": 1.2}, "vwap_session": 2131.0,
                 "news_blackout": False},
    "cross_asset": {"silver_rs": None, "dxy_rs": None,
                    "dxy": {"available": True, "correlation": -0.62,
                            "bias": -1, "bonus": 0.1,
                            "status": "ok"},
                    "smt_event": None},
    "data_quality": {"health": "healthy", "reason": "",
                     "h4_gap_bars": 0, "d1_stale": False},
}


def _full_situation():
    return build_situation(full_stack(), blocks=BLOCKS, price=2136.5)


# ------------------------------------------------------------------
# the pure render
# ------------------------------------------------------------------
def test_none_situation_renders_not_produced():
    assert render_situation_read(None) == (
        "market situation: not produced yet "
        "(no snapshot — the read builds on the "
        "first timeframe close)")


def test_per_tf_lines_from_the_real_builder():
    lines = render_situation_read(_full_situation()).splitlines()
    tf_lines = [ln for ln in lines
                if ln.startswith(tuple(f"  {tf}:"
                                       for tf in
                                       ("D1", "H4", "H1", "M30",
                                        "M15", "M5")))]
    assert len(tf_lines) == 6
    assert tf_lines[0].startswith(
        "  D1: bull · clean_trend/up · atr ")
    assert "swings H " in tf_lines[0]


def test_relationship_lines_from_the_real_builder():
    text = render_situation_read(_full_situation())
    for pair in ("D1→H4", "D1→H1", "H4→H1", "H4→M15",
                 "H1→M15", "H1→M5", "M15→M5"):
        assert f"  {pair}: aligned — same measured bias, " \
               f"both trend-phased" in text


def test_block_lines_exact():
    lines = render_situation_read(_full_situation()).splitlines()
    assert ("  regime: bedrock H1: "
            "up/clean_trend/normal/manipulation/offhours · "
            "day: trend day · ADR 62% (normal) · "
            "open london drive") in lines
    assert ("  location: price 2136.5 · value POC 2130.0 · "
            "VA 2120.0–2140.0 (developing) · premium side · "
            "eq 2100.0") in lines
    assert ("    IB 2128.0–2142.0 · overnight 2125.0–2144.0 · "
            "opening range 2128.0–2142.0 (1.4 ATR wide) · "
            "pools 1 above / 0 below") in lines
    assert ("  activity: london · killzone london_open · "
            "spread 0.30 ×0.90 · rvol H1 1.2 · VWAP 2131.0 · "
            "news clear") in lines
    # Relative strength has no live feed: absent in words —
    # the situation's own honesty, never a number.
    assert ("  cross-asset: DXY corr -0.62 (bonus 0.10) · "
            "SMT none · silver RS not produced (no live feed) · "
            "DXY RS not produced (no live feed)") in lines
    assert "  data quality: health healthy · missing: none" in lines


def test_no_fabricated_numbers_anywhere():
    text = render_situation_read(_full_situation())
    assert "0.00" not in text
    assert "None" not in text


def test_bare_situation_names_its_missing_blocks():
    lines = render_situation_read(
        build_situation(full_stack(), price=2136.5)).splitlines()
    assert "  location: not produced" in lines
    assert "  activity: not produced" in lines
    assert "  cross-asset: not produced" in lines
    quality = [ln for ln in lines
               if ln.startswith("  data quality:")]
    assert len(quality) == 1
    assert "missing: location, activity, cross_asset" in quality[0]


def test_collecting_tf_and_unmeasured_pairs():
    frames = {tf: flat_frame(120, freq=TF_FREQ[tf])
              for tf in ("D1", "H4", "H1", "M30", "M15")}
    frames["M5"] = flat_frame(5, freq=TF_FREQ["M5"])
    sit = build_situation(stack_of(**frames), price=2000.0)
    lines = render_situation_read(sit).splitlines()
    assert ("  M5: collecting (5 bars — state not measured yet)"
            in lines)
    assert ("  H1→M5: unmeasured (insufficient state)" in lines)
    assert ("  M15→M5: unmeasured (insufficient state)" in lines)


def test_hierarchy_row_and_sentence_reason_rows():
    sit = _full_situation()
    sit["relationships"] = list(sit["relationships"]) + [
        {"htf": "D1", "ltf": "H1", "source": "hierarchy",
         "kind": "correcting",
         "reason": "hierarchy leg 2: H4 pullback against D1"},
        {"htf": "D1", "ltf": "H1", "source": "hierarchy",
         "kind": None,
         "reason": "hierarchy regime 'range' — flat D1 bias "
                   "carries no directional reading"},
    ]
    lines = render_situation_read(sit).splitlines()
    assert ("  hierarchy D1→H1: correcting — hierarchy leg 2: "
            "H4 pullback against D1") in lines
    assert ("  hierarchy D1→H1: unmeasured — hierarchy regime "
            "'range' — flat D1 bias carries no directional "
            "reading") in lines


# ------------------------------------------------------------------
# the /status wiring
# ------------------------------------------------------------------
class _FakeControl:
    def __init__(self, *, with_situation: bool):
        self._with_situation = with_situation

    def get_live_config(self):
        return None

    def blocked_stats(self):
        return {"legs": {}, "signal_counts": {}}

    def market_read_line(self):
        return "READ-LINE"

    def situation_line(self):
        return "SIT-LINE"


class _BareControl:
    """An old runner: no situation_line at all."""

    def get_live_config(self):
        return None

    def blocked_stats(self):
        return {"legs": {}, "signal_counts": {}}


def _status_text(ctrl, tmp_path):
    ctx = CommandContext(journal=None, journal_path=None,
                         reports_dir=tmp_path / "reports")
    ctx.mt5_control = ctrl
    return _mt5_status(ctx, []).text


def test_status_carries_the_situation_section(tmp_path):
    text = _status_text(_FakeControl(with_situation=True), tmp_path)
    assert "Market Situation" in text
    assert "SIT-LINE" in text
    # Beside the standing read it completes, not instead of it.
    assert text.index("Market Read") < text.index("Market Situation")


def test_status_without_the_seam_omits_the_section(tmp_path):
    text = _status_text(_BareControl(), tmp_path)
    assert "Market Situation" not in text


# ------------------------------------------------------------------
# the runner seam
# ------------------------------------------------------------------
def _runner(tmp_path, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        poll_interval_seconds=0, **cfg_kw)
    return MT5Runner(cfg)


def test_runner_line_without_a_snapshot(tmp_path):
    runner = _runner(tmp_path)
    assert runner.situation_line() == (
        "market situation: not produced yet "
        "(no snapshot — the read builds on the "
        "first timeframe close)")


def test_runner_line_when_the_program_is_disabled(tmp_path):
    runner = _runner(tmp_path, situation_enabled=False)
    assert runner.situation_line().startswith(
        "market situation: disabled in config")


def test_runner_line_renders_the_keeper_snapshot(tmp_path):
    keeper = SituationKeeper(tmp_path / "journal")
    keeper.on_tf_close(
        "H1", full_stack(),
        asof=pd.Timestamp("2026-01-06", tz="UTC"))
    runner = _runner(tmp_path)
    runner._situation_keeper = keeper
    line = runner.situation_line()
    assert "  D1: bull" in line
    assert "  D1→H1: aligned" in line
    assert "as of 2026-01-06" in line
