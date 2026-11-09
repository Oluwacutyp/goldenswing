"""Pass 10 (SPEC §8) — ChartClocks resolver + StudySpec registry.

Pins: the §8.1 resolution table row by row; the fallback chain
(scenario clocks → registry native → standing default, STAMPED);
the study registry (every playbook-declared study resolves, an
undeclared name fails loudly, each study stamped with its own
clock); the defect class (role A on the management clock, role B
off its hardcoded H1, role C studies on their spec clocks);
PNG/HTML study parity; M1 in the study; chart provenance.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.charts import engine, tv_page  # noqa: E402
from goldenswing.charts.signal_model import (  # noqa: E402
    ChartClocks, ContextStudy, EntryStudy, LifecycleStudy,
    chart_clocks_for, resolve_study_specs, study_kind, study_spec)
from goldenswing.charts.signal_view import (  # noqa: E402
    TF_ORDER as VIEW_TF_ORDER, build_signal_view_html)
from goldenswing.charts.terminal_server import (  # noqa: E402
    TF_INTERVALS, TF_ORDER as TERM_TF_ORDER)
from goldenswing.execution.playbook_registry import (  # noqa: E402
    PLAYBOOK_REGISTRY, spec_for)

import matplotlib.image as mpimg  # noqa: E402

from test_mt5_mtf_charts import _runner, _sig, _verdict  # noqa: E402


def _frame(n, freq, t0, base=4100.0, drift=0.3, seed=11):
    idx = pd.date_range(t0, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(0, 1.1, n)) \
        + np.sin(np.arange(n) * 0.7) * 1.2
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 0.2
    vol = 100.0 + (np.arange(n) % 7) * 13.0
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + 0.5,
         "low": np.minimum(open_, close) - 0.5, "close": close,
         "volume": vol, "real_volume": 0, "spread": 30}, index=idx)


def _frames():
    return {
        "D1": _frame(90, "D", "2026-08-03", base=3980.0),
        "W1": _frame(40, "W-MON", "2026-02-02", base=3900.0),
        "H4": _frame(220, "4h", "2026-10-09", base=4050.0),
        "H1": _frame(220, "h", "2026-10-26", base=4050.0),
        "M30": _frame(220, "30min", "2026-11-02", base=4080.0),
        "M15": _frame(220, "15min", "2026-11-04", base=4090.0),
        "M5": _frame(220, "5min", "2026-11-05", base=4100.0),
        "M1": _frame(300, "min", "2026-11-05 06:00", base=4105.0),
    }


def _lc():
    return LifecycleStudy(
        direction=1, entry_ref=4102.0, structural_stop=4098.0,
        targets=(4108.0, 4112.0), state="planned")


def _cfg(page_html: str) -> dict:
    start = page_html.index('<script id="chart-cfg" '
                            'type="application/json">')
    start = page_html.index(">", start) + 1
    end = page_html.index("</script>", start)
    return json.loads(page_html[start:end])


# ------------------------------------------------------------ resolver
class TestResolverTable:
    def test_roster_row(self):
        cc = chart_clocks_for(playbook="h1_trigger",
                              route="h1_roster", entry_clock="M30")
        assert (cc.management_tf, cc.context_tf, cc.entry_tf) \
            == ("H1", "H4", "M30")
        assert cc.source == "route_default"

    def test_range_row(self):
        cc = chart_clocks_for(playbook="range_x",
                              route="range_h4_h1")
        assert (cc.management_tf, cc.context_tf, cc.entry_tf) \
            == ("H1", "D1", "H1")

    def test_qm_m5_row(self):
        cc = chart_clocks_for(playbook="qm_m5")
        assert (cc.management_tf, cc.context_tf, cc.entry_tf) \
            == ("M5", "H1", "M5")

    def test_qm_d1_a_is_d1_and_b_is_w1_when_available(self):
        cc = chart_clocks_for(playbook="qm_d1",
                              available={"D1", "W1", "H1"})
        assert cc.management_tf == "D1"
        assert cc.context_tf == "W1"
        assert cc.context_hold is False

    def test_qm_d1_b_holds_on_d1_without_w1(self):
        cc = chart_clocks_for(playbook="qm_d1",
                              available={"D1", "H1"})
        assert cc.context_tf == "D1"
        assert cc.context_hold is True

    def test_hot_scalp_row_offers_m1(self):
        cc = chart_clocks_for(playbook="hot_scalp")
        assert (cc.management_tf, cc.context_tf, cc.entry_tf) \
            == ("M15", "H1", "M1")

    def test_scenario_clocks_win(self):
        cc = chart_clocks_for(
            playbook="trend_x", route="trend_momentum",
            native_tf="H4",
            clocks={"formation_tf": "H4", "trigger_tf": "M15",
                    "entry_tf": "M15", "management_tf": "H4"})
        assert (cc.management_tf, cc.context_tf, cc.entry_tf) \
            == ("H4", "H4", "M15")
        assert cc.source == "scenario_clocks"

    def test_partial_scenario_clocks_fill_from_route(self):
        cc = chart_clocks_for(playbook="h1_trigger",
                              route="h1_roster", entry_clock="M15",
                              clocks={"entry_tf": "M30"})
        assert cc.entry_tf == "M30"
        assert cc.management_tf == "H1"
        assert cc.context_tf == "H4"
        assert cc.source == "scenario_clocks"

    def test_family_without_clocks_uses_registry_native(self):
        cc = chart_clocks_for(playbook="auction_x",
                              route="auction_value", native_tf="H1")
        assert cc.management_tf == "H1"
        assert cc.source == "registry_native"

    def test_unknown_falls_back_stamped(self):
        cc = chart_clocks_for(playbook="mystery_playbook")
        assert (cc.management_tf, cc.context_tf) == ("H1", "H4")
        assert cc.source == "standing_default"


# ------------------------------------------------------------ registry
class TestStudyRegistry:
    def test_every_registry_declaration_resolves(self):
        cc = chart_clocks_for(playbook="x", route="h1_roster")
        declared = set()
        for spec in PLAYBOOK_REGISTRY:
            declared.update(spec.chart_studies or ())
        assert declared  # the registry does declare studies
        for name in declared:
            specs = resolve_study_specs([name], cc)
            assert specs[0].kind == name
            assert specs[0].clock_tf  # stamped with a clock

    def test_undeclared_study_fails_loudly(self):
        with pytest.raises(ValueError):
            study_spec("not_a_study")
        with pytest.raises(ValueError):
            study_kind("rsi_deluxe")

    def test_study_clocks_come_from_the_spec(self):
        cc = ChartClocks(management_tf="H1", context_tf="H4",
                         entry_tf="M15", formation_tf="H4")
        specs = {s.kind: s for s in resolve_study_specs(
            ["volume_profile", "ema_pair", "ao"], cc)}
        # volume profile belongs to the formation clock
        assert specs["volume_profile"].clock_tf == "H4"
        # entry studies belong to the entry clock
        assert specs["ema_pair"].clock_tf == "M15"
        assert specs["ao"].clock_tf == "M15"
        assert specs["ao"].pane == "sub"

    def test_entry_study_sync_both_directions(self):
        legacy = EntryStudy(overlays=("ema_pair", "qm"))
        assert {s.kind for s in legacy.studies} \
            == {"ema_pair", "qm"}
        assert len(legacy.overlay_families()) <= 4
        modern = EntryStudy(studies=(study_spec("ao"),))
        assert modern.overlays == ()  # ao is not a legacy overlay
        mixed = EntryStudy(studies=(study_spec("ema_pair"),))
        assert mixed.overlays == ("ema_pair",)

    def test_build_entry_study_composes_declared_and_evidence(self):
        class _Sig:
            entry_zone = None
            factors = ["EMA 9/21 cross"]
        cc = chart_clocks_for(playbook="vol_x",
                              route="volatility_expansion",
                              native_tf="H4")
        es = engine.build_entry_study(
            sig=_Sig(), verdict={}, declared_studies=("bollinger",),
            clocks=cc)
        kinds = {s.kind: s for s in es.studies}
        assert kinds["bollinger"].clock_tf == "H4"  # formation
        assert kinds["ema_pair"].clock_tf == "H4"   # entry = native
        with pytest.raises(ValueError):
            engine.build_entry_study(
                sig=_Sig(), verdict={},
                declared_studies=("bogus_study",), clocks=cc)


# ------------------------------------------------------------ defect pins
class TestDefectPins:
    def test_role_pages_follow_resolved_clocks(self):
        frames = _frames()
        cc = chart_clocks_for(playbook="qm_d1",
                              available=set(frames))
        pages = tv_page.build_role_pages(
            frames=frames, lifecycle=_lc(), playbook="qm_d1",
            entry_clock="M15", clocks=cc)
        a, b, c = (_cfg(p) for p in pages)
        assert "D1" in a["header"]["role"]          # A = management
        assert b["header"]["role"].startswith("B · W1")
        assert "D1" in c["header"]["role"]          # C = entry

    def test_roster_context_is_h4_not_hardcoded_h1(self):
        frames = _frames()
        cc = chart_clocks_for(playbook="h1_trigger",
                              route="h1_roster", entry_clock="M15")
        pages = tv_page.build_role_pages(
            frames=frames, lifecycle=_lc(), playbook="h1_trigger",
            entry_clock="M15", clocks=cc)
        b = _cfg(pages[1])
        assert b["header"]["role"].startswith("B · H4")
        # B's bars ARE the H4 frame's bars
        assert len(b["bars"]) == 120
        assert b["bars"][-1]["close"] == pytest.approx(
            float(frames["H4"]["close"].iloc[-1]))

    def test_missing_context_frame_is_unavailable_not_substituted(
            self):
        frames = _frames()
        del frames["H4"]
        cc = chart_clocks_for(playbook="h1_trigger",
                              route="h1_roster", entry_clock="M15")
        pages = tv_page.build_role_pages(
            frames=frames, lifecycle=_lc(), playbook="h1_trigger",
            entry_clock="M15", clocks=cc)
        assert "H4 frame unavailable" in pages[1]

    def test_legacy_calls_keep_legacy_frames(self):
        # No clocks passed: the pre-Pass-10 derivation holds, so
        # direct renderer callers are byte-stable.
        frames = _frames()
        pages = tv_page.build_role_pages(
            frames=frames, lifecycle=_lc(), playbook="hot_scalp",
            entry_clock="M5")
        assert _cfg(pages[1])["header"]["role"].startswith(
            "B · H1")

    def test_mpl_roles_render_three_pngs_on_resolved_clocks(self):
        frames = _frames()
        cc = chart_clocks_for(playbook="qm_d1",
                              available=set(frames))
        pngs = engine._render_mt5_roles_mpl(
            frames=frames, lifecycle=_lc(), playbook="qm_d1",
            entry_clock="M15", clocks=cc)
        assert len(pngs) == 3
        for png in pngs:
            img = mpimg.imread(io.BytesIO(png), format="png")
            assert img.shape[1] == 1600


# ------------------------------------------------------------ study parity
class TestStudyParity:
    def _es(self, frames):
        cc = chart_clocks_for(playbook="vol_x",
                              route="volatility_expansion",
                              native_tf="H4", available=set(frames))
        return engine.build_entry_study(
            sig=type("S", (), {"entry_zone": None,
                               "factors": []})(),
            verdict={}, declared_studies=("bollinger", "ao"),
            clocks=cc), cc

    def test_html_entry_cfg_carries_series_and_subpane(self):
        frames = _frames()
        es, cc = self._es(frames)
        pages = tv_page.build_role_pages(
            frames=frames, lifecycle=_lc(), entry_study=es,
            playbook="vol_x", entry_clock="M15", clocks=cc)
        c = _cfg(pages[2])
        assert len(c["lineSeries"]) == 3       # bollinger mid/±2σ
        assert len(c["subSeries"]) == 1        # AO histogram pane
        assert any("Bollinger" in line
                   for line in c["ledger"])
        assert any("Awesome Oscillator" in line
                   for line in c["ledger"])

    def test_mpl_renders_same_study_set(self):
        frames = _frames()
        es, cc = self._es(frames)
        pngs = engine.render_mt5_roles(
            frames=frames, lifecycle=_lc(), entry_study=es,
            playbook="vol_x", entry_clock="M15", clocks=cc)
        assert len(pngs) == 3
        prov = engine.last_chart_provenance()
        assert prov["clocks"]["source"] == "registry_native"
        drawn = {(s["kind"], s["clock_tf"])
                 for s in prov["studies"]}
        assert drawn == {("bollinger", "H4"), ("ao", "H4")}

    def test_volume_profile_levels_on_formation_frame(self):
        frames = _frames()
        # Formation on M15: the developing-value organ needs a
        # full day of bars on its own clock (MIN_BARS law) — the
        # M15 frame's last day is dense, the H1 frame's is not.
        cc = chart_clocks_for(playbook="auction_x",
                              route="auction_value", native_tf="M15",
                              available=set(frames))
        es = EntryStudy(studies=resolve_study_specs(
            ["volume_profile"], cc))
        pages = tv_page.build_role_pages(
            frames=frames, lifecycle=_lc(), entry_study=es,
            playbook="auction_x", entry_clock="M15", clocks=cc)
        c = _cfg(pages[2])
        titles = [ln["title"] for ln in c["lines"]]
        assert any(t.startswith("POC") for t in titles)
        assert any("Volume profile" in line or "VP POC" in line
                   for line in c["ledger"])

    def test_study_with_missing_frame_is_named_not_drawn(self):
        frames = _frames()
        cc = ChartClocks(management_tf="H1", context_tf="H4",
                         entry_tf="M15", formation_tf="D1")
        es = EntryStudy(studies=resolve_study_specs(
            ["volume_profile"], cc))
        del frames["D1"]
        pages = tv_page.build_role_pages(
            frames=frames, lifecycle=_lc(), entry_study=es,
            playbook="q", entry_clock="M15", clocks=cc)
        c = _cfg(pages[2])
        assert any("frame unavailable" in line
                   for line in c["ledger"])


# ------------------------------------------------------------ provenance
class TestProvenance:
    def test_freshness_lands_in_role_headers(self):
        frames = _frames()
        cc = chart_clocks_for(playbook="h1_trigger",
                              route="h1_roster", entry_clock="M15")
        fresh = {"H1": {"last_bar": "11-09 07:00",
                        "quality": "healthy"},
                 "H4": {"last_bar": "11-09 04:00", "quality": ""},
                 "M15": {"last_bar": "11-09 07:45", "quality": ""}}
        pages = tv_page.build_role_pages(
            frames=frames, lifecycle=_lc(), playbook="h1_trigger",
            entry_clock="M15", clocks=cc, freshness=fresh)
        assert "last bar 11-09 07:00" in pages[0]
        assert "data healthy" in pages[0]
        assert _cfg(pages[1])["header"]["fresh"] \
            == "last bar 11-09 04:00"

    def test_png_stamps_carry_clocks(self):
        from goldenswing.charts import screenshot
        frames = _frames()
        cc = chart_clocks_for(playbook="qm_m5")
        pngs = engine.render_mt5_roles(
            frames=frames, lifecycle=_lc(), playbook="qm_m5",
            entry_clock="M15", clocks=cc)
        text = screenshot.read_png_text(pngs[0])
        assert text.get("Chart-Clocks", "").startswith(
            "A=M5 B=H1 C=M5")

    def test_signal_renderer_records_provenance(self):
        frames = _frames()
        cc = chart_clocks_for(playbook="qm_d1",
                              available=set(frames))
        pngs = engine.render_mt5_signal(
            frames["H1"], zone=None, entry=4102.0, stop=4098.0,
            targets=[4108.0], direction=1,
            verdict={"entry_clock": "D1"}, playbook="qm_d1",
            clocks=cc, frames=frames)
        assert len(pngs) == 3
        prov = engine.last_chart_provenance()
        assert prov["clocks"]["management_tf"] == "D1"
        assert prov["style"].startswith("signal-")


# ------------------------------------------------------------ M1 study
class TestM1InTheStudy:
    def test_tf_orders_carry_m1(self):
        assert "M1" in VIEW_TF_ORDER
        assert "M1" in TERM_TF_ORDER
        assert TF_INTERVALS["M1"] == 60

    def test_signal_view_offers_m1(self):
        frames = _frames()
        html = build_signal_view_html(
            frames={"H1": frames["H1"], "M1": frames["M1"]},
            entry=4102.0, stop=4098.0, targets=[4108.0],
            direction=1, playbook="hot_scalp",
            decision_score=None, broker_stop=None,
            verdict={"entry_clock": "M1"}, zone=None,
            nearby_zones=[], fib_levels=[], fib_range=None,
            ote_band=None, fib_info={}, alert_id="m1-pin")
        assert '"M1"' in html

    def test_hot_scalp_pages_offer_m1_entry(self):
        frames = _frames()
        cc = chart_clocks_for(playbook="hot_scalp")
        pages = tv_page.build_role_pages(
            frames=frames, lifecycle=_lc(), playbook="hot_scalp",
            entry_clock="M5", clocks=cc)
        c = _cfg(pages[2])
        assert "M1" in c["header"]["role"]
        assert len(c["bars"]) == 37  # the M1 frame's own window


# ------------------------------------------------------------ runner level
class TestRunnerWiring:
    def test_build_chart_resolves_roster_clocks(self, tmp_path,
                                                monkeypatch):
        runner = _runner(tmp_path)
        seen = {}

        def recorder(*args, **kwargs):
            seen.update(kwargs)
            from goldenswing.charts.engine import \
                render_mt5_roles as real
            return real(*args, **kwargs)

        monkeypatch.setattr(
            "goldenswing.execution.mt5_runner.render_mt5_roles",
            recorder)
        pngs, paths = runner._build_chart(
            _sig(), _verdict(), alert_id="MT5-p10-1",
            playbook="fvg_midline")
        assert pngs is not None and len(pngs) == 3
        clocks = seen["clocks"]
        # The roster's role B is H4 now — resolved, not hardcoded.
        assert (clocks.management_tf, clocks.context_tf,
                clocks.entry_tf) == ("H1", "H4", "M15")
        prov = engine.last_chart_provenance()
        assert prov["clocks"]["context_tf"] == "H4"

    def test_runner_scenario_clocks_win(self, tmp_path):
        runner = _runner(tmp_path)
        name = next(s.name for s in PLAYBOOK_REGISTRY
                    if s.route == "trend_momentum")
        sig = _sig(live_context={"scenario_clocks": {
            "formation_tf": "H4", "refinement_tf": None,
            "trigger_tf": "M15", "entry_tf": "M15",
            "management_tf": "H4"}})
        frames = runner._chart_frames()
        cc = runner._chart_clocks_for(sig, _verdict(), name,
                                      "M15", frames)
        assert cc.source == "scenario_clocks"
        assert (cc.management_tf, cc.context_tf, cc.entry_tf) \
            == ("H4", "H4", "M15")

    def test_runner_frames_merge_m1_aux(self, tmp_path):
        runner = _runner(tmp_path)
        runner._hot_scalp_aux_frames = {"M1": _frames()["M1"]}
        frames = runner._chart_frames()
        assert "M1" in frames
        cc = runner._chart_clocks_for(_sig(), _verdict(),
                                      "hot_scalp", "M5", frames)
        assert cc.entry_tf == "M1"
        # The stack itself is untouched by the charting merge.
        assert "M1" not in (runner._stack.frames or {})
