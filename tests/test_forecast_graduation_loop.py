"""Completion-04 — the predictive loop, closed and observable.

Pins: retrain accrual as DATA agreeing with the needs_*
verdicts on both lifecycles; the threshold actually triggering
a retrain through the real F2 cycle and the completed retrain
triggering an F3 battery run through the real calibration
cycle; the battery's per-horizon verdict journaled as a
forecast_validation record (incl. the insufficient-data
verdict) and rendered on the forecast status; graduation
flipping the active method ONLY through the probability gate,
held in shadow by a FIRM battery outside_band with the reason
named on the record; the degenerate-stop guard excluding the
+226R record shape from S4 grading AND from the S3 cycle's
calibration feed, with the exclusion journaled/counted; the E3
trust state surfaced as data + status lines with the weigh
gate's behavior unchanged.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from goldenswing.execution import forecast_evidence as fe
from goldenswing.execution.judgment_audit import degenerate_stop
from goldenswing.execution.judgment_validation import (
    build_pairs, validate as validate_judgment)
from goldenswing.forecast.mtf import (
    MIN_NEW_BARS_RETRAIN, bundle_accrual, needs_training,
    train_from_frames)
from goldenswing.forecast.outlook import (
    OUTLOOK_METHOD, TRAINED_METHOD, build_outlook)
from goldenswing.forecast.trained import (
    GRAD_MIN_N, ProbabilityBook, TrainedOutlook, needs_retrain,
    retrain_accrual, train_trained)
from goldenswing.forecast.validation import (
    battery_block_for, battery_summary, validate)
from tests.test_forecast_mt5 import _frames as _mtf_frames
from tests.test_forecast_trained import _frames
from tests.test_mt5_runner import _runner

H = 24  # the horizon most fixtures speak about

#: The per-horizon method stamp the live F1 baseline answers
#: with (the outlook BLOCK method OUTLOOK_METHOD is a different
#: stamp — the battery's _method_of grades the horizon's own).
F1_METHOD = "unconditional_h1"


# ── shared fixture builders ─────────────────────────────────────

@pytest.fixture(scope="module")
def frames():
    # 1400 bars: comfortably above the training row floor, a
    # third of the full-size fixture's cost — full-size
    # training is the trained suite's own pin, not this one's.
    return _frames(1400)


@pytest.fixture(scope="module")
def trained_path(tmp_path_factory, frames):
    path = tmp_path_factory.mktemp("c04") / "trained.joblib"
    train_trained(frames, path)
    return path


@pytest.fixture(scope="module")
def bundle_frames():
    # The ensemble bundle trains on the full multi-timeframe
    # stack (its own fixtures' frames) — never on H1 alone.
    return _mtf_frames()


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory, bundle_frames):
    path = tmp_path_factory.mktemp("c04b") / "bundle.joblib"
    train_from_frames(bundle_frames, path)
    return path


def _fill_book(book: ProbabilityBook, horizon: int = H) -> None:
    """The known-good graduation fill (n=80, low ECE)."""
    probs = {"up": 0.7, "dn": 0.2, "neutral": 0.1}
    realized = (["up"] * 7 + ["dn"] * 2 + ["neutral"]) * 8
    for r in realized:
        assert book.add(horizon, "trained", probs, r)


def _forecast_pair(bar: pd.Timestamp, method: str, *,
                   ret: float, direction: str = "UP",
                   close: float = 4100.0,
                   horizon: int = H) -> list[dict]:
    """One journaled forecast + reconciliation in the exact
    shape the F3 battery consumes.  The point price IS the
    realized endpoint (skill 0) — a synthetic journal for the
    loop's plumbing, not a claim about forecasting."""
    endpoint = close * (1.0 + ret / 100.0)
    ho = {"method": method, "direction": direction,
          "point_price": endpoint,
          "interval": {"low": close * 0.99, "high": close * 1.01,
                       "coverage": 0.8}}
    bar_s = str(bar)
    return [
        {"type": "forecast", "bar_time": bar_s,
         "generated_at": bar_s, "cadence": "H1", "close": close,
         "horizons": {str(horizon): {"up": 60.0, "dn": 20.0}},
         "outlook": {"horizons": {str(horizon): ho}}},
        {"type": "forecast_reconciliation", "bar_time": bar_s,
         "horizons": {str(horizon): {
             "ret_pct": ret, "run_up": abs(ret) + 0.4,
             "run_dn": 0.3, "y_up": ret > 0, "y_dn": ret < 0}}},
    ]


def _journal_records(n_bars: int, method: str, *,
                       start: str = "2025-01-01",
                       wrong: bool = False) -> list[dict]:
    """n bars of one method at horizon H: ~86% interval
    coverage, honest direction unless ``wrong`` (direction
    systematically against the realized move)."""
    records: list[dict] = []
    bars = pd.date_range(start, periods=n_bars, freq="h")
    for i, bar in enumerate(bars):
        ret = 1.5 if i >= int(n_bars * 0.86) else 0.5
        direction = "DOWN" if wrong else "UP"
        records.extend(_forecast_pair(bar, method, ret=ret,
                                      direction=direction))
    return records


# ── 1a. accrual as data, verdicts agreeing ──────────────────────

def test_f2_accrual_agrees_with_needs_retrain(trained_path,
                                              frames):
    import datetime as dt

    svc = TrainedOutlook(trained_path)
    art, _ = svc.artifact()
    assert art is not None
    idx = frames["H1"].index
    now = dt.datetime.now(dt.timezone.utc)

    # No artifact: due, cause named, verdict agrees.
    acc = retrain_accrual(None, idx, now)
    assert acc["due"] and acc["cause"] == "no_artifact"
    assert needs_retrain(None, idx, now)[0] is True

    # Fresh artifact against its own frame: not due, counted.
    acc = retrain_accrual(art, idx, now)
    assert acc["due"] is False and acc["new_bars"] == 0
    assert acc["threshold"] == 500 and acc["age_days"] < 1
    assert needs_retrain(art, idx, now) == (False, acc["reason"])

    # 600 fresh bars past the cutoff: due on bars, counted.
    ext = pd.date_range(idx[-1] + pd.Timedelta(hours=1),
                        periods=600, freq="h")
    idx_ext = idx.append(ext)
    acc = retrain_accrual(art, idx_ext, now)
    assert acc["due"] and acc["cause"] == "new_bars"
    assert acc["new_bars"] == 600
    assert needs_retrain(art, idx_ext, now) == (True, acc["reason"])

    # An old artifact (few new bars): due on age, counted.
    old = dict(art)
    trained_at = pd.Timestamp(art["trained_at"])
    old["trained_at"] = str(
        trained_at - pd.Timedelta(days=30))
    acc = retrain_accrual(old, idx, now)
    assert acc["due"] and acc["cause"] == "age"
    assert acc["age_days"] >= 29
    assert needs_retrain(old, idx, now)[0] is True


def test_ensemble_accrual_agrees_with_needs_training(
        bundle_path, bundle_frames, tmp_path):
    h1 = bundle_frames["H1"]
    acc = bundle_accrual(bundle_path, h1)
    assert acc["due"] is False and acc["new_bars"] == 0
    assert acc["threshold"] == MIN_NEW_BARS_RETRAIN
    assert needs_training(bundle_path, h1) == (False, acc["reason"])

    ext = pd.DataFrame(
        index=pd.date_range(h1.index[-1] + pd.Timedelta(hours=1),
                            periods=600, freq="h"),
        columns=h1.columns, dtype=float)
    for col in ("open", "high", "low", "close"):
        ext[col] = h1["close"].iloc[-1]
    h1_ext = pd.concat([h1, ext])
    acc = bundle_accrual(bundle_path, h1_ext)
    assert acc["due"] and acc["cause"] == "new_bars"
    assert acc["new_bars"] == 600
    assert needs_training(bundle_path, h1_ext)[0] is True

    missing = tmp_path / "nope.joblib"
    acc = bundle_accrual(missing, h1)
    assert acc["due"] and acc["cause"] == "no_bundle"
    assert needs_training(missing, h1)[0] is True


# ── 1b. battery summary: the winner rule + the block ────────────

def _cell(state: str, firm: bool, n: int = 80) -> dict:
    return {"horizon": H, "method": "fake",
            "verdict": {"state": state, "firm": firm,
                        "segment": "holdout" if firm else "pooled",
                        "n": n, "fails": [] if state == "within_band"
                        else ["direction"]}}


def _result(cells: dict, h2h: dict) -> dict:
    return {"overall": "report_ready", "journal_records": 999,
            "malformed_lines": 0,
            "coverage": {"scored_rows": 160},
            "holdout_boundary": "2025-02-01 00:00:00",
            "cells": cells, "head_to_head": h2h}


def test_battery_summary_winner_rule():
    from goldenswing.forecast.validation import (
        METHOD_F1, METHOD_F2)
    both = _result(
        {f"{H}|{METHOD_F1}": _cell("within_band", True),
         f"{H}|{METHOD_F2}": _cell("within_band", True)},
        {str(H): {"leader": METHOD_F2, "contested": True}})
    s = battery_summary(both, generated_at="t0",
                        trigger="f2_retrain")
    assert s["horizons"][str(H)]["winner"] == "trained"
    assert s["horizons"][str(H)]["contested"] is True
    assert s["scored_rows"] == 160 and s["trigger"] == "f2_retrain"

    f1_only = _result(
        {f"{H}|{METHOD_F1}": _cell("within_band", False),
         f"{H}|{METHOD_F2}": _cell("insufficient_data", False,
                                   n=3)},
        {str(H): {"leader": None, "contested": False}})
    s = battery_summary(f1_only)
    assert s["horizons"][str(H)]["winner"] == "baseline"

    neither = _result(
        {f"{H}|{METHOD_F1}": _cell("insufficient_data", False,
                                   n=3),
         f"{H}|{METHOD_F2}": _cell("outside_band", False)},
        {str(H): {"leader": None, "contested": False}})
    s = battery_summary(neither)
    entry = s["horizons"][str(H)]
    # The winner reads insufficient_data — but the per-method
    # states keep "unmeasured" and "measured and failed" apart.
    assert entry["winner"] == "insufficient_data"
    assert entry["f1"]["state"] == "insufficient_data"
    assert entry["f2"]["state"] == "outside_band"


def test_battery_block_only_on_firm_outside_band():
    from goldenswing.forecast.validation import (
        METHOD_F1, METHOD_F2)
    firm = battery_summary(_result(
        {f"{H}|{METHOD_F2}": _cell("outside_band", True)},
        {}), generated_at="t1")
    block = battery_block_for(firm, H)
    assert block is not None and block["firm"] is True
    assert block["generated_at"] == "t1"
    assert battery_block_for(firm, 12) is None

    provisional = battery_summary(_result(
        {f"{H}|{METHOD_F2}": _cell("outside_band", False)},
        {}))
    assert battery_block_for(provisional, H) is None

    within = battery_summary(_result(
        {f"{H}|{METHOD_F2}": _cell("within_band", True)},
        {}))
    assert battery_block_for(within, H) is None
    assert battery_block_for(None, H) is None
    assert battery_block_for({}, H) is None


def test_battery_on_real_records_names_winners():
    # F1 measured and passing (provisional — pooled segment),
    # F2 never answered: the winner is the baseline, honestly.
    records = _journal_records(70, F1_METHOD)
    s = battery_summary(validate(records), generated_at="t",
                        trigger="test")
    entry = s["horizons"][str(H)]
    assert entry["f1"]["state"] == "within_band"
    assert entry["f2"] is None
    assert entry["winner"] == "baseline"
    assert s["overall"] == "report_ready"

    # Thin journal: everything reads insufficient_data.
    s = battery_summary(validate(_journal_records(3,
                                                  F1_METHOD)))
    assert s["horizons"][str(H)]["winner"] == "insufficient_data"
    assert s["overall"] == "insufficient_data"


def test_battery_firm_outside_band_for_trained():
    # 210 answering bars for the trained method, direction
    # systematically wrong: the holdout (63 bars ≥ VERDICT_MIN_N)
    # is verdict-grade → a FIRM outside_band, and the block
    # engages.
    records = _journal_records(210, TRAINED_METHOD, wrong=True)
    s = battery_summary(validate(records), generated_at="t")
    entry = s["horizons"][str(H)]
    assert entry["f2"]["state"] == "outside_band"
    assert entry["f2"]["firm"] is True
    assert entry["winner"] == "insufficient_data"
    assert battery_block_for(s, H) is not None


# ── 1c. the loop through the real cycles ────────────────────────

def _seeded_runner(tmp_path, trained_path, frames):
    runner = _runner(tmp_path)
    runner._forecast_trained = TrainedOutlook(trained_path)
    runner._stack = SimpleNamespace(frames=frames)
    return runner


def test_battery_journaled_through_calibration_cycle(
        tmp_path, trained_path, frames):
    runner = _seeded_runner(tmp_path / "a", trained_path, frames)
    for rec in _journal_records(70, F1_METHOD):
        runner.journal.append(rec)
    runner._battery_pending = "f2_retrain"
    runner._forecast_calibration_cycle()
    validations = [r for r in runner.journal.records()
                   if r.get("type") == "forecast_validation"]
    assert len(validations) == 1
    rec = validations[0]
    assert rec["trigger"] == "f2_retrain"
    assert rec["horizons"][str(H)]["winner"] == "baseline"
    assert rec["horizons"][str(H)]["f1"]["state"] == "within_band"
    # Persisted where status/manifest/graduation read it.
    st = runner._forecast_cal_state()
    assert st["battery"]["horizons"][str(H)]["winner"] == "baseline"
    # The status surface names the verdict + the accrual —
    # on the NOT READY path too (this runner has no ensemble
    # bundle; the loop's state does not live in the bundle).
    text = runner.forecast_line()
    assert "NOT READY" in text
    assert "battery (F3" in text and "winner baseline" in text
    assert "retrain accrual" in text


def test_threshold_retrain_then_battery_end_to_end(
        tmp_path):
    # Train an artifact through bar 900 of 1500: exactly 600
    # new H1 bars accrue — the retrain threshold.
    frames_full = _frames(1500)
    cut = frames_full["H1"].index[899]
    part = {"H1": frames_full["H1"].iloc[:900]}
    path = tmp_path / "trained.joblib"
    summary = train_trained(part, path, through=cut)
    assert summary["train_rows"] > 0
    runner = _runner(tmp_path / "r")
    runner._forecast_trained = TrainedOutlook(path)
    st = runner._forecast_cal_state()
    runner._forecast_f2_cycle(frames_full, frames_full["H1"], st)
    assert runner._f2_training is True  # the threshold FIRED
    # A real train on a shared box takes minutes, not seconds —
    # the bound is generous so the test measures the LOOP, not
    # the machine's mood.
    deadline = time.time() + 1800
    while runner._f2_train_result is None and time.time() < deadline:
        time.sleep(0.5)
    assert runner._f2_train_result is not None
    ok, _note = runner._f2_train_result
    assert ok
    # The completed train journals + arms the battery trigger.
    runner._forecast_f2_cycle(frames_full, frames_full["H1"], st)
    assert runner._battery_pending == "f2_retrain"
    trainings = [r for r in runner.journal.records()
                 if r.get("type") == "forecast_trained_training"]
    assert trainings and trainings[-1]["ok"] is True
    # The calibration cycle runs the battery over the journal.
    runner._stack = SimpleNamespace(frames=frames_full)
    runner._forecast_calibration_cycle()
    validations = [r for r in runner.journal.records()
                   if r.get("type") == "forecast_validation"]
    assert len(validations) == 1
    assert validations[0]["trigger"] == "f2_retrain"
    # An empty forecast journal reads insufficient_data, never
    # a dressed-up verdict.
    assert validations[0]["overall"] == "insufficient_data"


# ── 1d. graduation: flips only on a pass; battery can hold ──────

def _firm_block_summary() -> dict:
    from goldenswing.forecast.validation import (
        METHOD_F1, METHOD_F2)
    return battery_summary(_result(
        {f"{H}|{METHOD_F1}": _cell("within_band", True),
         f"{H}|{METHOD_F2}": _cell("outside_band", True)},
        {str(H): {"leader": METHOD_F1, "contested": True}}),
        generated_at="t-block", trigger="f2_retrain")


def test_graduation_holds_under_firm_battery_block(tmp_path):
    runner = _runner(tmp_path)
    book = ProbabilityBook()
    _fill_book(book)
    ok, _info, block = runner._forecast_graduation(
        book, None, H)
    assert ok and block is None
    ok, info, block = runner._forecast_graduation(
        book, _firm_block_summary(), H)
    assert not ok and block is not None
    assert info["battery_block"]["state"] == "outside_band"
    # A provisional failure never holds.
    from goldenswing.forecast.validation import METHOD_F2
    prov = battery_summary(_result(
        {f"{H}|{METHOD_F2}": _cell("outside_band", False)}, {}))
    ok, _info, block = runner._forecast_graduation(book, prov, H)
    assert ok and block is None


def test_trained_param_and_outlook_carry_the_evidence(
        tmp_path, trained_path, frames):
    runner = _runner(tmp_path)
    runner._forecast_trained = TrainedOutlook(trained_path)
    book = ProbabilityBook()
    _fill_book(book)
    h1 = frames["H1"]
    close = float(h1["close"].iloc[-1])

    idx = pd.date_range("2026-01-01", periods=240, freq="h")
    f1_close = 2000.0 + np.cumsum(
        np.sin(np.arange(240) / 9.0) * 3)
    f1_h1 = pd.DataFrame(
        {"open": f1_close, "high": f1_close + 4.0,
         "low": f1_close - 4.0, "close": f1_close}, index=idx)
    ranks = {str(H): {"up": 85.0, "dn": 20.0}}
    model_probs = {H: {"up": [0.7, 0.9], "dn": [0.2, 0.4]}}

    st = {"prob_book": book.data, "battery": None}
    param = runner._forecast_trained_param(frames, close, 5.0, st)
    assert param is not None
    entry = param["horizons"][str(H)]
    assert entry["graduated"] is True
    assert entry["graduation"]["n"] >= GRAD_MIN_N

    outlook = build_outlook(
        ranks=ranks, close=2000.0, atr=8.0, samples=[], h1=f1_h1,
        model_probs=model_probs, trained=param)
    ho = outlook["horizons"][str(H)]
    assert ho["method"] == TRAINED_METHOD
    assert ho["trained"]["graduated"] is True
    assert ho["trained"]["graduation"]["n"] >= GRAD_MIN_N

    # Under a firm battery block: shadow again, and the record
    # NAMES the block + the book evidence it overrode.
    st = {"prob_book": book.data, "battery": _firm_block_summary()}
    param = runner._forecast_trained_param(frames, close, 5.0, st)
    entry = param["horizons"][str(H)]
    assert entry["graduated"] is False
    assert entry["battery"]["state"] == "outside_band"
    outlook = build_outlook(
        ranks=ranks, close=2000.0, atr=8.0, samples=[], h1=f1_h1,
        model_probs=model_probs, trained=param)
    ho = outlook["horizons"][str(H)]
    assert ho["method"] != TRAINED_METHOD  # F1 answered, not F2
    assert ho["trained"]["graduated"] is False
    assert ho["trained"]["battery"]["state"] == "outside_band"
    assert ho["trained"]["graduation"]["battery_block"]["firm"] \
        is True


# ── 2. the degenerate-R guard ───────────────────────────────────

def test_degenerate_stop_predicate():
    # The evidence-pack shape: ~$0.135 stop on a ~$4,100 entry.
    assert degenerate_stop(4100.0, 4099.865) == "price_frac"
    assert degenerate_stop(4100.0, 4099.865, atr=6.0) == "price_frac"
    # Ordinary structural geometry passes both floors.
    assert degenerate_stop(4100.0, 4080.0, atr=6.0) is None
    assert degenerate_stop(4100.0, 4095.0) is None
    # ATR floor alone: distance clears the price floor (2.05)
    # but is under 2% of a 200-ATR regime.
    assert degenerate_stop(4100.0, 4097.0, atr=200.0) == "atr_frac"
    # Unusable inputs never exclude.
    assert degenerate_stop(4100.0, 4100.0) == "price_frac"


def _shadow_row(ts: str, decision: str = "take") -> dict:
    return {"type": "judgment_shadow", "ts": ts,
            "playbook": "inversion",
            "judgment": {"decision": decision,
                         "expected_r": 1.0, "confidence": 0.6,
                         "reason_category": "edge"},
            "gate_result": "ordered", "gate_order_result": None}


def _recon_row(ts: str, *, entry: float, move: float,
               realized_r: float) -> dict:
    return {"type": "judgment_reconciliation", "judged_ts": ts,
            "playbook": "inversion", "horizon_h": 4,
            "entry": entry, "exit_price": entry + move,
            "move": move, "move_atr": 2.0, "hit": move > 0,
            "realized_r": realized_r}


def test_s4_grading_excludes_degenerate_pair():
    records = []
    # The +226R poison record: stop ≈ $0.135 implied.
    records.append(_shadow_row("2025-11-04T13:00:00"))
    records.append(_recon_row("2025-11-04T13:00:00", entry=4100.0,
                              move=30.525, realized_r=226.11))
    # Three clean pairs.
    for i, r in enumerate((1.0, -0.5, 0.5)):
        ts = f"2025-11-04T1{4 + i}:00:00"
        records.append(_shadow_row(ts))
        records.append(_recon_row(ts, entry=4100.0, move=r * 20.0,
                                  realized_r=r))
    pairs = build_pairs(records, horizon_h=4)
    poison = [p for p in pairs if p["ts"].startswith("2025-11-04T13")]
    assert poison[0]["r_excluded"] == "degenerate_stop"
    assert poison[0]["realized_r"] is None
    assert abs(poison[0]["implied_stop_distance"] - 0.135) < 0.01
    clean = [p for p in pairs if not p["ts"].startswith(
        "2025-11-04T13")]
    assert all(p["r_excluded"] is None for p in clean)

    result = validate_judgment(records, horizon_h=4)
    assert result["excluded_degenerate_r"] == 1
    seg = result["evaluation"]
    total_excluded = seg["excluded_degenerate_r"] \
        + result["holdout"]["excluded_degenerate_r"]
    assert total_excluded == 1
    # Means are computed on the clean set only: the take mean
    # is the clean pairs' mean, nowhere near the poisoned one.
    assert seg["judgment"]["take_mean_r"] is not None
    assert seg["judgment"]["take_mean_r"] < 5.0


def test_runner_audit_cycle_excludes_degenerate(tmp_path):
    runner = _runner(tmp_path)
    idx = pd.date_range("2025-11-04 00:00", periods=40, freq="h")
    closes = np.full(40, 4100.0)
    closes[10:] = np.linspace(4100.0, 4130.0, 30)
    h1 = pd.DataFrame(
        {"open": closes, "high": closes + 2.0,
         "low": closes - 2.0, "close": closes}, index=idx)
    runner._stack = SimpleNamespace(frames={"H1": h1})
    judged = idx[10]
    due = {"1": str(judged + pd.Timedelta(hours=1)),
           "4": str(judged + pd.Timedelta(hours=4))}
    st = runner._audit_state()
    st["pending"] = [
        {"ts": str(judged), "playbook": "inversion",
         "direction": 1, "entry": 4100.0, "stop": 4099.865,
         "decision": "take", "expected_r": 1.2, "confidence": 0.6,
         "due": dict(due), "done": []},
        {"ts": str(judged), "playbook": "inversion",
         "direction": 1, "entry": 4100.0, "stop": 4080.0,
         "decision": "take", "expected_r": 1.2, "confidence": 0.6,
         "due": dict(due), "done": []},
    ]
    runner._audit_save()
    runner._judgment_audit_cycle()
    recs = [r for r in runner.journal.records()
            if r.get("type") == "judgment_reconciliation"]
    assert len(recs) == 4  # 2 items × 2 horizons
    degen = [r for r in recs if r.get("r_excluded")]
    clean = [r for r in recs if not r.get("r_excluded")]
    assert len(degen) == 2 and len(clean) == 2
    for r in degen:
        assert r["r_excluded"] == "degenerate_stop"
        assert r["realized_r"] is None
        assert abs(r["stop_distance"] - 0.135) < 1e-6
        assert r["hit"] is True  # direction evidence survives
    for r in clean:
        assert r["realized_r"] is not None
    st = runner._audit_state()
    assert st["tallies"]["r_degenerate_excluded"] == 2
    # Calibration was fed ONLY the clean R values (the view is
    # per confidence band; both clean adds share the 0.6 band).
    from goldenswing.execution.judgment_audit import (
        calibration_view)
    view = calibration_view(st["calibration"])
    assert sum(b["n"] for b in view.values()) == 2
    assert all(b["mean_realized_r"] < 5.0 for b in view.values())


# ── 4. E3 trust: surfaced, behavior unchanged ───────────────────

def _book_view(pool: list[tuple[str, int, float]], base: float,
               ) -> dict:
    bands = [{"band": b, "n": n, "rate": rate,
              "lift": (rate / base if base else None)}
             for b, n, rate in pool]
    return {"horizons": {str(H): {"up": {
        "n": sum(n for _b, n, _r in pool), "base_rate": base,
        "bands": bands}}}}


def test_trust_state_surfaces_the_evidence():
    view = _book_view([("80-90", 12, 0.6), ("90+", 10, 0.55)],
                      base=0.4)
    st = fe.trust_state(view, H, "up")
    assert st["state"] == "trusted"
    assert st["n_pool"] == 22 and st["min_n"] == 20
    assert abs(st["lift"] - 1.4432) < 0.01
    assert [b["band"] for b in st["bands"]] == ["80-90", "90+"]
    assert fe.trust_for(view, H, "up") == "trusted"

    thin = _book_view([("80-90", 2, 0.9)], base=0.4)
    st = fe.trust_state(thin, H, "up")
    assert st["state"] == "unproven" and st["n_pool"] == 2
    lines = fe.render_trust_lines(thin)
    assert any("unproven" in ln and "2/20" in ln for ln in lines)
    assert fe.render_trust_lines({}) == []
    assert fe.render_trust_lines(None) == []

    anti = _book_view([("80-90", 15, 0.3), ("90+", 10, 0.3)],
                      base=0.5)
    assert fe.trust_state(anti, H, "up")["state"] == \
        "anti_calibrated"
    assert fe.trust_for(anti, H, "up") == "anti_calibrated"


class _Ranks:
    bar_time = "2025-11-04 13:00:00"

    def rank(self, horizon, word):
        return 90.0 if word == "up" else 10.0


def test_subthreshold_feed_still_weighs_nothing():
    thin = _book_view([("80-90", 2, 0.9)], base=0.4)
    block = fe.forecast_block(ranks=_Ranks(), direction=1,
                              horizon=H, fresh=True,
                              book_view=thin)
    assert block["stance"] == "supporting"
    assert block["trust"] == "unproven"
    assert block["weighed"] is False
    assert fe.merge_into_evidence({"supporting": []}, block) == \
        {"supporting": []}
    # The same stance with an earned book DOES weigh — the gate
    # moved nothing; only the evidence crossed it.
    earned = _book_view([("80-90", 12, 0.6), ("90+", 10, 0.55)],
                        base=0.4)
    block = fe.forecast_block(ranks=_Ranks(), direction=1,
                              horizon=H, fresh=True,
                              book_view=earned)
    assert block["trust"] == "trusted" and block["weighed"] is True


# ── manifest: the loop's standing state ─────────────────────────

def test_manifest_carries_forecast_block(tmp_path, trained_path,
                                         frames):
    runner = _seeded_runner(tmp_path / "m", trained_path, frames)
    runner._journal_manifest()
    manifests = [r for r in runner.journal.records()
                 if r.get("type") == "manifest"]
    assert manifests
    block = manifests[-1]["forecast"]
    assert block["ensemble"]["threshold"] == MIN_NEW_BARS_RETRAIN
    assert block["trained"]["threshold"] == 500
    assert block["horizons"][str(H)]["standing"] in (
        "graduated", "shadow", "shadow_battery_hold",
        "no artifact")
    assert "battery" in block and "trust" in block
