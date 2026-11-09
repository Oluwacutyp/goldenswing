"""MarketSituation Pass 8 — the Bill Williams RESEARCH
family (SPEC §6.6).  Pins, in the GS conventions
(fixtures judged by what was observable AT THE TIME):

- the measurement core (research/bill_williams/
  indicators.py): SMMA from the reader's EMA machinery
  read at Williams' shifts; 5-bar fractals known only
  from their confirmation bar (center + 2) — a read at
  bar j never sees a fractal whose right side closes
  after j, and every read is identical on truncated
  frames; Alligator state words on constructed series
  (chop never eats; an accelerating trend eats); AO/AC
  against hand-computed simple-average values; Gator
  distances/phase; BW-MFI classes with the tick-volume
  basis named on every read, absent when volume is;
- the strategy (strategy.py): entry = confirmed
  fractal break beyond the teeth in the Alligator's
  direction with AO/AC confirmation; a setup that
  closes back beyond the lips first DIES unbroken;
  invalidation = opposite fractal / lips cross, paired
  by simulation; ablation variants are supersets of
  the full rule (removing a gate only adds signals);
  the abstain arm is exactly the AO/AC-rejected
  candidates;
- the evaluation (evaluation.py): §6's common bar
  over the family's OWN journal — floor per arm,
  negative take median fails, inverted discrimination
  fails, ready reads as a recommendation only and
  binds nothing; ablation table + confirmation lift;
- the ISOLATION contract (§6.6's posture): the
  package's names appear in no registry, route table,
  judgment table, or evidence map, and no module
  outside goldenswing/research/ references it — there
  is no authority path to pin because there is none.

Fixture geometry: scripted closes with DIRECTIONAL
wicks (the wick extends in the bar's own direction),
so a swing peak is decisive against the next bar,
which opens at its close — stated once, as in the
Pass 7 family suites.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import goldenswing.execution.mt5_runner as mr
from goldenswing.execution import route_promotion
from goldenswing.execution.evidence_map import PLAYBOOK_EVIDENCE
from goldenswing.execution.playbook_registry import FLOOR_AUTHORITY, PLAYBOOK_REGISTRY
from goldenswing.reader import indicators as reader_ind
from goldenswing.research.bill_williams import evaluation, harness, journal
from goldenswing.research.bill_williams.indicators import (
    VOLUME_BASIS_ABSENT,
    VOLUME_BASIS_TICK,
    PreparedWilliams,
    annotate,
    smma,
)
from goldenswing.research.bill_williams.strategy import (
    run_strategy,
    scan_events,
    simulate,
)

# ------------------------------------------------------------ fixtures

def _frame(closes, vols=None):
    bars = []
    prev = closes[0]
    for c in closes:
        o = prev
        m = c - o
        if m >= 0:
            h, l = c + 0.30 * m + 0.05, o - 0.05
        else:
            h, l = o + 0.05, c + 0.30 * m - 0.05
        bars.append((o, h, l, c))
        prev = c
    idx = pd.date_range("2026-01-05 00:00", periods=len(bars),
                        freq="1h")
    df = pd.DataFrame(bars, columns=["open", "high", "low",
                                     "close"], index=idx)
    if vols is not None:
        df["volume"] = vols
    return df


def _explicit_frame(rows):
    idx = pd.date_range("2026-01-05 00:00", periods=len(rows),
                        freq="1h")
    return pd.DataFrame(
        rows, columns=["open", "high", "low", "close"],
        index=idx)


def _trend_series():
    """Chop base (both fractal directions form) into a
    rising staircase with real pullbacks and a deeper
    dip every fourth cycle (mid-trend down fractals)."""
    base = [2000.0 + 2.5 * np.sin(2 * np.pi * i / 8.0)
            for i in range(40)]
    stair = [base[-1]]
    for i in range(132):
        step = [3.0, 3.0, -4.5][i % 3]
        if i % 12 == 11:
            step = -9.0
        stair.append(stair[-1] + step)
    return base + stair[1:]


def _trend_frame():
    series = _trend_series()
    vols = [100 + (i % 7) * 13 for i in range(len(series))]
    return _frame(series, vols)


def _reversal_series():
    """A declining staircase, then a violent reversal
    into a rising one: during the decline AO/AC lag the
    turns, so the AO/AC-rejected abstain arm is
    populated on both sides."""
    dec = [2200.0]
    for i in range(60):
        dec.append(dec[-1] + [-3.0, -3.0, 4.5][i % 3])
    rev = [dec[-1] + 10.0, dec[-1] + 19.0, dec[-1] + 27.0,
           dec[-1] + 34.0]
    tail = [rev[-1]]
    for i in range(36):
        tail.append(tail[-1] + [3.0, 3.0, -4.5][i % 3])
    return dec + rev + tail[1:]


# ------------------------------------------------------------ SMMA / lines

class TestSmmaAndLines:
    def test_smma_is_the_reader_ema_machinery(self):
        x = np.array([3.0, 5.0, 4.0, 8.0, 6.0, 9.0, 7.0])
        np.testing.assert_allclose(
            smma(x, 5), reader_ind.ema(x, 5))

    def test_smma_of_constant_is_constant(self):
        out = smma(np.full(30, 42.5), 13)
        assert np.allclose(out, 42.5)

    def test_lines_are_read_at_williams_shifts(self):
        closes = [2000.0 + i for i in range(80)]
        df = _frame(closes)
        prep = PreparedWilliams(df)
        median = (df["high"] + df["low"]).to_numpy() / 2.0
        j = 60
        lines = prep.lines_at(j)
        assert lines["jaw"] == pytest.approx(
            float(smma(median, 13)[j - 8]))
        assert lines["teeth"] == pytest.approx(
            float(smma(median, 8)[j - 5]))
        assert lines["lips"] == pytest.approx(
            float(smma(median, 5)[j - 3]))

    def test_lines_unformed_before_shifts_reach(self):
        prep = PreparedWilliams(_frame([2000.0] * 12))
        assert prep.lines_at(5) is None
        assert prep.alligator_at(5)["state"] == "unformed"


# ------------------------------------------------------------ fractals

class TestFractals:
    def _peak_frame(self, tie=False):
        rows = [(100.0, 100.4, 99.6, 100.0)] * 40
        rows = [list(r) for r in rows]
        rows[18] = [100.0, 102.0, 99.6, 101.0]
        rows[19] = [101.0, 104.0, 100.4, 103.0]
        rows[20] = [103.0, 110.0, 102.4, 106.0]
        rows[21] = [103.0, 110.0 if tie else 104.0, 100.4, 103.0]
        rows[22] = [101.0, 102.0, 99.6, 100.0]
        return _explicit_frame([tuple(r) for r in rows])

    def test_fractal_known_only_from_confirmation_bar(self):
        prep = PreparedWilliams(self._peak_frame())
        assert prep.fractals.up == [
            {"center": 20, "price": 110.0, "confirmed_at": 22}]
        # One bar before confirmation the fractal does
        # not exist for any read — no hindsight.
        assert prep.fractals.known_at(21, 1) is None
        known = prep.fractals.known_at(22, 1)
        assert known["center"] == 20
        assert known["price"] == 110.0

    def test_tie_is_never_a_fractal(self):
        prep = PreparedWilliams(self._peak_frame(tie=True))
        assert prep.fractals.up == []

    def test_reads_identical_on_truncated_frames(self):
        df = _trend_frame()
        full = PreparedWilliams(df)
        trunc = PreparedWilliams(df.iloc[:100])
        for j in range(100):
            assert annotate(full, j, timeframe="H1") == \
                annotate(trunc, j, timeframe="H1")


# ------------------------------------------------------------ alligator / gator

class TestAlligatorStates:
    def test_chop_never_eats(self):
        closes = [2000.0 + 1.5 * np.sin(2 * np.pi * i / 8.0)
                  for i in range(120)]
        prep = PreparedWilliams(_frame(closes))
        states = [prep.alligator_at(j)["state"]
                  for j in range(40, 120)]
        assert "eating" not in states
        assert states.count("sleeping") > len(states) / 2

    def test_accelerating_trend_eats_in_direction(self):
        closes = [2000.0]
        for i in range(70):
            closes.append(closes[-1] + 0.4 + 0.05 * i)
        prep = PreparedWilliams(_frame(closes))
        reads = [prep.alligator_at(j) for j in range(30, 71)]
        assert all(r["direction"] == 1 for r in reads)
        assert any(r["state"] == "eating" for r in reads)
        assert all(r["lips"] > r["teeth"] > r["jaw"]
                   for r in reads)

    def test_declining_mirror_orders_down(self):
        closes = [2200.0]
        for i in range(70):
            closes.append(closes[-1] - 0.4 - 0.05 * i)
        prep = PreparedWilliams(_frame(closes))
        r = prep.alligator_at(70)
        assert r["direction"] == -1
        assert r["jaw"] > r["teeth"] > r["lips"]

    def test_gator_eating_on_acceleration(self):
        closes = [2000.0]
        for i in range(70):
            closes.append(closes[-1] + 0.4 + 0.05 * i)
        prep = PreparedWilliams(_frame(closes))
        phases = [prep.gator_at(j)["phase"]
                  for j in range(30, 71)]
        assert "eating" in phases
        g = prep.gator_at(70)
        assert g["upper"] > 0 and g["lower"] > 0


# ------------------------------------------------------------ AO / AC

def _median_frame(medians):
    rows = [(m, m + 0.5, m - 0.5, m) for m in medians]
    return _explicit_frame(rows)


class TestOscillators:
    def test_ao_hand_computed_on_a_ramp(self):
        # median == bar index: AO = SMA5 - SMA34 of the
        # median = (j-2) - (j-16.5) = 14.5 once full.
        prep = PreparedWilliams(
            _median_frame([float(i) for i in range(60)]))
        ao = prep.ao_at(40)
        assert ao["value"] == pytest.approx(14.5)
        assert ao["sign"] == 1

    def test_ao_negative_on_a_falling_ramp(self):
        prep = PreparedWilliams(
            _median_frame([100.0 - i for i in range(60)]))
        assert prep.ao_at(40)["sign"] == -1

    def test_ac_sign_follows_acceleration(self):
        accel = [i + 0.02 * i * i for i in range(60)]
        prep = PreparedWilliams(_median_frame(accel))
        assert prep.ac_at(59)["sign"] == 1
        decel = [2.0 * i - 0.01 * i * i for i in range(60)]
        prep = PreparedWilliams(_median_frame(decel))
        assert prep.ao_at(59)["sign"] == 1
        assert prep.ac_at(59)["sign"] == -1


# ------------------------------------------------------------ BW-MFI

class TestMfi:
    def _mfi_frame(self):
        rows = [
            (100.0, 102.0, 98.0, 100.0),   # range 4
            (100.0, 103.0, 97.0, 100.0),   # range 6
            (100.0, 104.0, 96.0, 100.0),   # range 8
            (100.0, 102.0, 98.0, 100.0),   # range 4
            (100.0, 101.0, 99.0, 100.0),   # range 2
            (100.0, 103.0, 97.0, 100.0),   # range 6
        ]
        vols = [100.0, 200.0, 100.0, 200.0, 100.0, 50.0]
        df = _explicit_frame(rows)
        df["volume"] = vols
        return df

    def test_value_is_range_over_volume(self):
        prep = PreparedWilliams(self._mfi_frame())
        assert prep.mfi_at(1)["value"] == pytest.approx(6 / 200)

    def test_williams_classes(self):
        prep = PreparedWilliams(self._mfi_frame())
        # bar1: MFI down (4/100 -> 6/200), vol up -> squat
        assert prep.mfi_at(1)["class"] == "squat"
        # bar2: MFI up (0.03 -> 0.08), vol down -> fake
        assert prep.mfi_at(2)["class"] == "fake"
        # bar3: MFI down (0.08 -> 0.02), vol up -> squat
        assert prep.mfi_at(3)["class"] == "squat"
        # bar4: MFI down (0.02 -> 0.02)? 4/200=0.02 -> 2/100
        # = 0.02: flat on the MFI leg.
        assert prep.mfi_at(4)["class"] == "flat"
        # bar5: MFI up (0.02 -> 0.12), vol down -> fake
        assert prep.mfi_at(5)["class"] == "fake"

    def test_green_and_fade_classes(self):
        rows = [(100.0, 101.0, 99.0, 100.0),
                (100.0, 103.0, 97.0, 100.0),
                (100.0, 101.0, 99.0, 100.0)]
        df = _explicit_frame(rows)
        df["volume"] = [100.0, 200.0, 400.0]
        prep = PreparedWilliams(df)
        # bar1: MFI 0.02 -> 0.03 (up), vol up -> green
        assert prep.mfi_at(1)["class"] == "green"
        # bar2: MFI 0.03 -> 0.005 (down), vol up -> squat
        assert prep.mfi_at(2)["class"] == "squat"
        df["volume"] = [100.0, 200.0, 100.0]
        prep = PreparedWilliams(df)
        # bar2: MFI 0.03 -> 0.02 (down), vol down -> fade
        assert prep.mfi_at(2)["class"] == "fade"

    def test_basis_named_tick_volume_with_caveat(self):
        prep = PreparedWilliams(self._mfi_frame())
        read = prep.mfi_at(2)
        assert read["volume_basis"] == VOLUME_BASIS_TICK
        assert "tick volume" in read["volume_caveat"]

    def test_absent_without_volume(self):
        df = _explicit_frame([(100.0, 102.0, 98.0, 100.0)] * 6)
        prep = PreparedWilliams(df)
        read = prep.mfi_at(3)
        assert read["value"] is None and read["class"] is None
        assert read["volume_basis"] == VOLUME_BASIS_ABSENT
        assert prep.volume_basis == VOLUME_BASIS_ABSENT


# ------------------------------------------------------------ strategy

class TestStrategy:
    def test_full_rule_fires_and_pairs_on_the_trend(self):
        prep = PreparedWilliams(_trend_frame())
        sigs = run_strategy(prep)["full"]
        assert len(sigs) >= 10
        paired = [s for s in sigs if s["outcome"]["paired"]]
        assert len(paired) >= 10
        for s in paired:
            # every gate the full rule names actually held
            assert all(s["gates"].values())
            d = s["direction"]
            assert (s["outcome"]["entry_price"]
                    - s["fractal"]["price"]) * d > 0
            assert (s["outcome"]["entry_price"]
                    - s["outcome"]["stop_initial"]) * d > 0
            assert s["outcome"]["exit_reason"] in (
                "stop", "lips_cross", "horizon", "end_of_data")

    def test_ablation_variants_are_supersets(self):
        prep = PreparedWilliams(_trend_frame())
        out = run_strategy(prep)
        key = lambda s: (s["fractal"]["center"], s["direction"])
        full = {key(s) for s in out["full"]}
        for variant in ("no_aoac", "no_teeth", "no_alligator"):
            assert full <= {key(s) for s in out[variant]}

    def test_abstain_arm_is_exactly_the_aoac_rejects(self):
        prep = PreparedWilliams(_frame(_reversal_series()))
        out = run_strategy(prep)
        assert len(out["abstains"]) >= 1
        for s in out["abstains"]:
            assert s["gates"]["alligator_direction"]
            assert s["gates"]["teeth"]
            assert not s["gates"]["aoac"]
            assert s["arm"] == "abstain"

    def test_setup_dies_unbroken_beyond_the_lips(self):
        dec = [2200.0]
        for _ in range(30):
            dec.append(dec[-1] - 2.0)
        bounce = [dec[-1] + 5.0, dec[-1] + 9.0,
                  dec[-1] + 6.0, dec[-1] + 2.0]
        tail = [bounce[-1]]
        for _ in range(30):
            tail.append(tail[-1] - 2.0)
        prep = PreparedWilliams(_frame(dec + bounce + tail[1:]))
        ups = [e for e in scan_events(prep)
               if e["direction"] > 0]
        assert len(ups) == 1
        assert ups[0]["died"] and ups[0]["break_bar"] is None
        out = run_strategy(prep)
        assert all(s["direction"] < 0 or
                   s["fractal"]["center"] != ups[0]["fractal"]["center"]
                   for s in out["full"])

    def test_no_stop_when_no_opposite_fractal(self):
        # A pure staircase: up fractals only — a long has
        # no opposite fractal, hence no definable stop.
        stair = [2000.0]
        for i in range(60):
            stair.append(stair[-1] + [3.0, 3.0, -4.5][i % 3])
        prep = PreparedWilliams(_frame(stair))
        ev = next(e for e in scan_events(prep)
                  if e["break_bar"] is not None)
        outcome = simulate(prep, ev["break_bar"], ev["direction"])
        if ev["direction"] > 0 and not prep.fractals.down:
            assert outcome["no_stop"] and not outcome["paired"]


# ------------------------------------------------------------ evaluation

def _sig_record(r, arm="take", variant="full", **kw):
    rec = {"type": "bw_signal", "variant": variant, "arm": arm,
           "outcome": {"paired": True, "no_stop": False,
                       "degenerate": False, "r": r},
           "read": {"gator_phase": kw.get("gator", "eating"),
                    "mfi_class": kw.get("mfi", "green")}}
    return rec


def _arm_records(rs, arm):
    return [_sig_record(r, arm=arm) for r in rs]


class TestEvaluation:
    def test_no_records_is_insufficient(self):
        snap = evaluation.evaluate_records([])
        assert snap["verdict"] == evaluation.VERDICT_INSUFFICIENT
        assert snap["binds"] is False

    def test_below_floor_is_insufficient(self):
        recs = _arm_records([0.5] * 10, "take") + \
            _arm_records([0.1] * 10, "abstain")
        snap = evaluation.evaluate_records(recs)
        assert snap["verdict"] == evaluation.VERDICT_INSUFFICIENT

    def test_negative_take_median_fails(self):
        recs = _arm_records([-0.5] * 25, "take") + \
            _arm_records([-1.0] * 25, "abstain")
        snap = evaluation.evaluate_records(recs)
        assert snap["verdict"] == evaluation.VERDICT_NOT_READY
        assert "negative" in snap["verdict_reason"]

    def test_inverted_discrimination_fails(self):
        recs = _arm_records([0.5] * 25, "take") + \
            _arm_records([0.9] * 25, "abstain")
        snap = evaluation.evaluate_records(recs)
        assert snap["verdict"] == evaluation.VERDICT_NOT_READY
        assert "INVERTED" in snap["verdict_reason"]

    def test_ready_is_a_recommendation_only(self):
        recs = _arm_records([0.6] * 15 + [0.4] * 10, "take") + \
            _arm_records([0.1] * 25, "abstain")
        snap = evaluation.evaluate_records(recs)
        assert snap["verdict"] == evaluation.VERDICT_READY
        assert "separate owner decision" in snap["verdict_reason"]
        assert snap["binds"] is False

    def test_degenerate_and_no_stop_are_counted_not_paired(self):
        recs = _arm_records([0.5] * 25, "take") + \
            _arm_records([0.1] * 25, "abstain")
        recs.append({"type": "bw_signal", "variant": "full",
                     "arm": "take",
                     "outcome": {"paired": False, "no_stop": True,
                                 "degenerate": False, "r": None},
                     "read": {}})
        recs.append({"type": "bw_signal", "variant": "full",
                     "arm": "take",
                     "outcome": {"paired": False, "no_stop": False,
                                 "degenerate": True, "r": None},
                     "read": {}})
        snap = evaluation.evaluate_records(recs)
        assert snap["counts"]["no_stop"] == 1
        assert snap["counts"]["degenerate"] == 1
        assert snap["take"]["paired"] == 25

    def test_ablation_table_and_lift(self):
        recs = _arm_records([0.5] * 25, "take")
        recs += [_sig_record(0.2, variant="no_aoac")
                 for _ in range(30)]
        recs += [_sig_record(-0.3, gator="sleeping", mfi="fake")
                 for _ in range(3)]
        snap = evaluation.evaluate_records(recs)
        assert snap["variants"]["no_aoac"]["paired"] == 30
        lift = snap["confirmation_lift"]
        assert lift["gator_eating"]["paired"] == 25
        assert lift["mfi_fake"]["paired"] == 3
        assert lift["mfi_fake"]["thin"] is True

    def test_render_names_the_caveats(self):
        recs = _arm_records([0.5] * 25, "take") + \
            _arm_records([0.1] * 25, "abstain")
        text = "\n".join(evaluation.render_lines(
            evaluation.evaluate_records(recs)))
        assert "tick volume" in text
        assert "no route, no order path" in text
        assert "ablation" in text


# ------------------------------------------------------------ journal + harness

class TestJournalAndHarness:
    def test_journal_round_trip_counts_damage(self, tmp_path):
        path = tmp_path / "bw.jsonl"
        recs = [{"type": "bw_read", "bar": 1},
                {"type": "bw_signal", "bar": 2}]
        assert journal.append_records(path, recs) == 2
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        back, damaged = journal.read_records(path)
        assert back == recs and damaged == 1

    def test_missing_journal_is_an_empty_stream(self, tmp_path):
        assert journal.read_records(tmp_path / "none.jsonl") == ([], 0)

    def test_harness_drives_the_full_stream(self):
        out = harness.drive_history(
            _trend_frame(), timeframe="H1",
            situation_id="sit-42")
        types = [r["type"] for r in out["records"]]
        assert "bw_read" in types and "bw_signal" in types
        assert types[-1] == "bw_evaluation"
        sigs = [r for r in out["records"]
                if r["type"] == "bw_signal"]
        assert all(r["volume_basis"] == VOLUME_BASIS_TICK
                   for r in sigs)
        assert all(r["situation_id"] == "sit-42" for r in sigs)
        reads = [r for r in out["records"] if r["type"] == "bw_read"]
        assert all(r["timeframe"] == "H1" for r in reads)
        assert out["evaluation"]["binds"] is False
        assert out["evaluation"]["verdict"] in (
            evaluation.VERDICT_INSUFFICIENT,
            evaluation.VERDICT_NOT_READY,
            evaluation.VERDICT_READY)


# ------------------------------------------------------------ isolation

class TestIsolation:
    """§6.6's posture, pinned: no authority path exists
    because no path exists at all."""

    BW_WORDS = ("bill_williams", "williams", "alligator",
                "gator", "bw_mfi", "bw-mfi")

    def _registry_names(self):
        names = set()
        reg = PLAYBOOK_REGISTRY
        if isinstance(reg, dict):
            names |= set(reg.keys())
            specs = list(reg.values())
        else:
            specs = list(reg)
        for spec in specs:
            name = getattr(spec, "name", None)
            if name:
                names.add(name)
        return names

    def test_absent_from_every_live_table(self):
        tables = {
            "registry": self._registry_names(),
            "order routes": set(
                route_promotion.ORDER_ROUTE_BY_PLAYBOOK.keys()),
            "judgment routes": set(mr.JUDGMENT_ROUTE_STATES.keys()),
            "evidence map": set(PLAYBOOK_EVIDENCE.keys()),
            "floor authority": set(FLOOR_AUTHORITY.keys()),
        }
        for label, names in tables.items():
            for name in names:
                low = str(name).lower()
                assert not any(w in low for w in self.BW_WORDS), \
                    f"{label} carries a Bill Williams name: {name}"

    def test_no_live_module_references_the_package(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1] \
            / "goldenswing"
        offenders = []
        for path in root.rglob("*.py"):
            if "research" in path.parts or \
                    "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8",
                                  errors="ignore")
            if "bill_williams" in text or \
                    "goldenswing.research" in text:
                offenders.append(str(path))
        assert offenders == []

    def test_package_references_no_live_authority(self):
        import pathlib
        pkg = pathlib.Path(__file__).resolve().parents[1] \
            / "goldenswing" / "research" / "bill_williams"
        banned = ("playbook_registry", "mt5_runner",
                  "scenario_book", "route_promotion",
                  "ORDER_ROUTE", "JUDGMENT_ROUTE")
        for path in pkg.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for word in banned:
                assert word not in text, \
                    f"{path.name} references {word}"
