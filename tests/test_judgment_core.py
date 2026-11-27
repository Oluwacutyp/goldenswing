"""Judgment (owner orders 2026-10-23 + 2026-11-27 — the
formula core is RETIRED, and no AI judges a trade).

One verdict, code only, pinned here:
- the CODE verdict (goldenswing/execution/judgment.py):
  the scenarios compete on the market's own measured
  evidence — displacement / acceptance / follow-through
  read off the entry-clock bars, HTF bias, the Asymmetry
  score's edge over its floor.  The proof: the same setup
  under two different market-structure inputs gets two
  different verdicts.  The 26 Nov acceptance case (a
  pullback SELL against an upward displacement reads BUY,
  not SELL) is pinned — the evidence reads buy AND the
  verdict follows it.
- the BRAIN is explanation drafting only
  (goldenswing/execution/judgment_brain.py): it drafts a
  plain-words explanation of a verdict already decided,
  never the decision.  A configured brain calling the other
  way never overrides the code verdict — the live order
  path never consults it.
- the runner wiring: a candidate-bearing live funnel record
  gains a judgment record, warmup/non-candidate records do
  not, the flag turns it off, and the evaluation record
  itself is byte-identical with the judgment on or off (the
  verdict can never steer the path it observes).
"""
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from goldenswing.execution.judgment import (
    evaluate_judgment,
    measure_decision_structure,
    verdict_from_structure,
)
from goldenswing.execution.judgment_brain import (
    BrainJudgmentProvider,
    draft_judgment_explanation,
)
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig


# ------------------------------------------------------------ fixtures
def _candidate(**kw):
    base = {"direction": 1, "entry": 3900.0, "stop": 3890.0,
            "targets": [3920.0], "score": 72.0,
            "asymmetry_score": 72.0, "sleeve_base_score": 60.0,
            "score_components": None, "factors": [], "index": 5}
    base.update(kw)
    return base


def _evidence(*, supporting=(), contradictory=(), groups=None,
              observed=None, inferred=None):
    return {
        "supporting": list(supporting),
        "contradictory": list(contradictory),
        "evidence_groups": groups or {},
        "independent_evidence_groups": sorted(groups or {}),
        "observed": observed or {
            "entry": 3900.0, "stop": 3890.0, "stop_dist": 10.0,
            "bias_d1": 1, "bias_h4": 1,
            "session_bucket": "overlap"},
        "inferred": inferred or {
            "score_total": 72.0, "score_flipped": False,
            "regime": "clean_trend", "posture": "TREND_PULLBACK",
            "journey_state": "with_journey", "room_r": 2.0},
    }


def _decision(**kw):
    base = {"result": "accepted", "reason_code": None,
            "order_result": "filled",
            "floor_result": {"floor": 55.0, "total": 72.0,
                             "path": "INTRADAY_CONTINUATION"}}
    base.update(kw)
    return base


_TREND_CTX = {"regime": "up/clean_trend/normal/accumulation/"
                        "overlap",
              "posture": "TREND_PULLBACK", "day_type": "trend",
              "adr_pct": 40.0, "adr_state": "open", "session": "overlap",
              "smt": None}


def _frame(closes, *, rng=2.0):
    """A synthetic entry-clock frame: 20 bars, each with the
    given close path and a fixed range (so the 14-period ATR
    is ~``rng``).  The last three closes are the displacement
    leg the measurement reads."""
    rows = []
    for i, close in enumerate(closes):
        rows.append({"open": close - 0.2, "high": close + rng / 2,
                     "low": close - rng / 2, "close": close,
                     "volume": 100})
    return pd.DataFrame(rows)


def _judge(candidate=None, evidence=None, context=None,
           decision=None, frames=None, entry_tf=None):
    return evaluate_judgment(
        candidate=candidate or _candidate(),
        decision=decision or _decision(),
        evidence=evidence or _evidence(),
        context=context or _TREND_CTX,
        frames=frames, entry_tf=entry_tf)


# ------------------------------------------------------------ code rung: the proof
def test_verdict_changes_with_structure():
    """The proof the owner ordered: the SAME setup under two
    different market-structure inputs gets two different
    verdicts — the verdict is the structure's reads, never a
    hand-set constant."""
    # Structure A: displacement up, closes accepted above the
    # level, follow-through held, D1+H4 with, score over floor.
    up = _frame([3900.0] * 17 + [3901.0, 3903.0, 3905.0])
    take = _judge(frames={"H1": up}, entry_tf="H1")
    assert take["decision"] == "take"
    assert take["direction"] == 1
    assert take["expected_r"] == pytest.approx(2.0)
    assert take["rung"] == "code"
    # Structure B: the same candidate, the structure fighting
    # it — displacement down, closes accepted below the level,
    # follow-through held down, D1+H4 against, score under floor.
    down = _frame([3900.0] * 17 + [3899.0, 3897.0, 3895.0])
    ev = _evidence(
        observed={"entry": 3900.0, "stop": 3890.0,
                  "stop_dist": 10.0, "bias_d1": -1,
                  "bias_h4": -1, "session_bucket": "overlap"},
        inferred={"score_total": 40.0, "score_flipped": True,
                  "regime": "clean_trend", "posture": None,
                  "journey_state": None, "room_r": 2.0})
    dec = _decision(floor_result={"floor": 55.0, "total": 40.0,
                                  "path": "INTRADAY_CONTINUATION"})
    abstain = _judge(evidence=ev, decision=dec,
                     frames={"H1": down}, entry_tf="H1")
    assert abstain["decision"] == "abstain"
    assert abstain["direction"] == -1
    assert "fights the measured direction" in abstain["reason"]


def test_26_nov_pullback_sell_reads_buy():
    """Acceptance case (owner's screenshot, Exness demo
    81800936, XAUUSDm H1, 26 Nov evening): Hot Scalp fired a
    SELL around 3571 on the pullback; gold then ripped to
    ~3666.  The chart showed it AT THE MOMENT of execution:
    the pullback held above the 3557-3564 lows and price
    displaced upward.  Fed that structure, the reworked
    judgment must say BUY, not SELL — and the H1-close
    trigger must not fire a sell on that pullback."""
    # The structure at the moment: the pullback held above
    # 3557-3564, then the closes displaced upward through
    # 3571 (3573, 3576 — both accepted above the level).
    frame = _frame([3560.0] * 17 + [3564.0, 3573.0, 3576.0],
                   rng=3.0)
    sell = _candidate(direction=-1, entry=3571.0, stop=3578.0,
                      targets=[3557.0], asymmetry_score=72.0)
    ev = _evidence(
        observed={"entry": 3571.0, "stop": 3578.0,
                  "stop_dist": 7.0, "bias_d1": 0, "bias_h4": 1,
                  "session_bucket": "overlap"},
        inferred={"score_total": 72.0, "score_flipped": False,
                  "regime": "clean_trend", "posture": None,
                  "journey_state": None, "room_r": 2.0})
    out = _judge(candidate=sell, evidence=ev,
                 frames={"H1": frame}, entry_tf="H1")
    assert out["direction"] == 1  # BUY, not SELL
    assert out["decision"] == "abstain"
    assert "fights the measured direction" in out["reason"]
    # The same structure with a LONG candidate takes.
    buy = _candidate(direction=1, entry=3571.0, stop=3564.0,
                     targets=[3585.0], asymmetry_score=72.0)
    out_buy = _judge(candidate=buy, evidence=ev,
                     frames={"H1": frame}, entry_tf="H1")
    assert out_buy["decision"] == "take"
    assert out_buy["direction"] == 1


def test_invalid_geometry_abstains():
    out = _judge(candidate=_candidate(stop=3900.0))
    assert out["decision"] == "abstain"
    assert out["direction"] == 0
    assert out["expected_r"] is None
    assert "invalid geometry" in out["reason"]


def test_no_reaction_read_abstains():
    """The structure reads the candidate's side on bias and
    score, but no reaction read (displacement / acceptance /
    follow-through) confirms it — the move has not shown."""
    flat = _frame([3900.0] * 20)
    out = _judge(frames={"H1": flat}, entry_tf="H1")
    assert out["decision"] == "abstain"
    assert "no reaction read confirms" in out["reason"]


def test_thin_evidence_abstains():
    # Nothing measured at all: no bias, no score, no floor,
    # no regime, no bars — no reads, no direction.
    out = _judge(
        candidate=_candidate(asymmetry_score=None),
        decision=_decision(floor_result={"floor": None,
                                         "total": None,
                                         "path": None}),
        evidence=_evidence(
            observed={"entry": 3900.0, "stop": 3890.0,
                      "stop_dist": 10.0, "bias_d1": 0,
                      "bias_h4": 0, "session_bucket": "overlap"},
            inferred={
                "score_total": None, "score_flipped": False,
                "regime": None, "posture": None,
                "journey_state": None, "room_r": None}),
        context={})
    assert out["decision"] == "abstain"
    assert out["confidence"] == 0.0
    assert "no measured direction" in out["reason"]


def test_deterministic_same_structure_same_verdict():
    kw = dict(candidate=_candidate(), decision=_decision(),
              evidence=_evidence(), context=_TREND_CTX)
    assert evaluate_judgment(**kw) == evaluate_judgment(**kw)


def test_confidence_is_the_agreement_share():
    """Confidence is the share of measured reads agreeing
    with the measured direction — a count, never a hand-set
    formula."""
    up = _frame([3900.0] * 17 + [3901.0, 3903.0, 3905.0])
    structure = measure_decision_structure(
        candidate=_candidate(), evidence=_evidence(),
        context=_TREND_CTX, frames={"H1": up}, entry_tf="H1",
        decision=_decision())
    verdict = verdict_from_structure(structure, _candidate())
    present = len(structure.reads)
    agreeing = len(structure.reads_for(verdict["direction"]))
    assert present > 0
    assert verdict["confidence"] == pytest.approx(
        agreeing / present)


def test_structure_is_plain_data_for_the_record():
    """The measured structure serialises to plain data — the
    record's evidence, verbatim."""
    up = _frame([3900.0] * 17 + [3901.0, 3903.0, 3905.0])
    structure = measure_decision_structure(
        candidate=_candidate(), evidence=_evidence(),
        context=_TREND_CTX, frames={"H1": up}, entry_tf="H1",
        decision=_decision())
    data = structure.to_dict()
    json.dumps(data)  # must be plain data, not objects
    assert data["displacement"] == "strong"
    assert data["displacement_direction"] == 1
    assert data["acceptance"] == "accepted"
    assert data["follow_through"] == "held"


# ------------------------------------------------------------ explanation drafting
def _fake_urlopen(payload, status=200):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    def _open(request, timeout=None):
        return _Resp()

    return _open


def _provider():
    return BrainJudgmentProvider(
        url="http://localhost:9/v1", api_key="test-key",
        model="test-model")


def test_brain_provider_from_env(monkeypatch):
    monkeypatch.setenv("GS_JUDGMENT_BRAIN_API_KEY", "k")
    monkeypatch.setenv("GS_JUDGMENT_BRAIN_MODEL", "m")
    provider = BrainJudgmentProvider.from_env()
    assert provider is not None
    # provider=groq (the default) + no URL -> Groq's
    # OpenAI-compatible endpoint.
    assert provider.url == "https://api.groq.com/openai/v1"
    monkeypatch.delenv("GS_JUDGMENT_BRAIN_API_KEY")
    assert BrainJudgmentProvider.from_env() is None


def test_brain_drafts_an_explanation_of_the_decided_verdict(
        monkeypatch):
    """The brain explains; it never decides.  Given the
    structure AND the verdict the code already reached, it
    returns a plain-words explanation — a string, never a
    verdict — and the decided verdict is untouched."""
    payload = {"choices": [{"message": {"content": json.dumps({
        "explanation": "displacement up, closes accepted "
                       "above the level, bias with — the "
                       "pullback held and the long takes it"})}}]}
    monkeypatch.setattr(
        "goldenswing.execution.judgment_brain.urllib.request.urlopen",
        _fake_urlopen(payload))
    structure = measure_decision_structure(
        candidate=_candidate(), evidence=_evidence(),
        context=_TREND_CTX, decision=_decision()).to_dict()
    # A decided TAKE verdict: the up structure (strong long
    # displacement, accepted, held) behind it.
    up = _frame([3900.0] * 17 + [3901.0, 3903.0, 3905.0])
    verdict = _judge(frames={"H1": up}, entry_tf="H1")
    assert verdict["decision"] == "take"
    out = draft_judgment_explanation(
        structure, verdict, _candidate(), _provider())
    assert isinstance(out, str)
    assert "pullback held" in out
    # The decided verdict is untouched by the drafting.
    assert verdict["decision"] == "take"
    assert verdict["direction"] == 1


def test_brain_drafting_incoherent_reply_is_no_explanation(
        monkeypatch):
    # A verdict-shaped reply (direction/decision, no
    # explanation) is not an explanation — the brain never
    # decides, so a reply that tries to is discarded.  The
    # caller keeps the code verdict's own reason.
    payload = {"choices": [{"message": {"content": json.dumps({
        "direction": -1, "decision": "take",
        "reason": "sell it anyway"})}}]}
    monkeypatch.setattr(
        "goldenswing.execution.judgment_brain.urllib.request.urlopen",
        _fake_urlopen(payload))
    structure = measure_decision_structure(
        candidate=_candidate(), evidence=_evidence(),
        context=_TREND_CTX, decision=_decision()).to_dict()
    assert draft_judgment_explanation(
        structure, _judge(), _candidate(), _provider()) is None


def test_brain_drafting_unreachable_is_no_explanation(monkeypatch):
    import urllib.error

    def _raise(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(
        "goldenswing.execution.judgment_brain.urllib.request.urlopen",
        _raise)
    structure = measure_decision_structure(
        candidate=_candidate(), evidence=_evidence(),
        context=_TREND_CTX, decision=_decision()).to_dict()
    assert draft_judgment_explanation(
        structure, _judge(), _candidate(), _provider()) is None


def test_brain_drafting_unconfigured_is_no_explanation():
    structure = measure_decision_structure(
        candidate=_candidate(), evidence=_evidence(),
        context=_TREND_CTX, decision=_decision()).to_dict()
    assert draft_judgment_explanation(
        structure, _judge(), _candidate(), None) is None


# ------------------------------------------------------------ wiring
def _runner(tmp_path, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        news_pause_enabled=False, **cfg_kw)
    return MT5Runner(cfg)


def _journal_records(tmp_path):
    path = tmp_path / "mt5_journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text().splitlines() if line.strip()]


def _emit_candidate(runner):
    return runner._emit_eval(
        playbook="qm_m5", evaluation_clock="h1_trigger",
        intended_tf="M5", outcome="candidate_emitted",
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence())


def test_judgment_record_journaled_beside_evaluation(tmp_path):
    runner = _runner(tmp_path)
    _emit_candidate(runner)
    records = _journal_records(tmp_path)
    kinds = [r["type"] for r in records]
    assert "evaluation" in kinds
    judged = [r for r in records if r["type"] == "judgment"]
    assert len(judged) == 1
    rec = judged[0]
    assert rec["playbook"] == "qm_m5"
    assert rec["gate_result"] == "accepted"
    assert rec["gate_order_result"] == "filled"
    assert rec["judgment"]["mode"] == "active"
    assert rec["judgment"]["rung"] == "code"
    assert rec["judgment"]["decision"] in ("take", "abstain")


def test_no_candidate_no_judgment(tmp_path):
    runner = _runner(tmp_path)
    runner._emit_eval(playbook="qm_m5",
                      evaluation_clock="h1_trigger",
                      outcome="no_setup")
    assert [r for r in _journal_records(tmp_path)
            if r["type"] == "judgment"] == []


def test_warmup_records_get_no_judgment(tmp_path):
    runner = _runner(tmp_path)
    runner._emit_eval(
        playbook="qm_m5", evaluation_clock="h1_trigger",
        intended_tf="M5", outcome="candidate_emitted",
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(), source="warmup_scan")
    assert [r for r in _journal_records(tmp_path)
            if r["type"] == "judgment"] == []


def test_judgment_flag_off_and_evaluation_record_identical(tmp_path):
    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()
    _emit_candidate(_runner(on_dir))
    _emit_candidate(_runner(off_dir, judgment_record_enabled=False))
    on_records = _journal_records(on_dir)
    off_records = _journal_records(off_dir)
    assert [r for r in off_records
            if r["type"] == "judgment"] == []
    # Time-bearing fields legitimately differ between two runs
    # (assessment_text embeds the evaluation instant in prose);
    # everything else must be identical — the verdict cannot
    # steer the record it observes.
    strip = ("ts", "run_id", "commit", "config_fingerprint",
             "assessment_text")
    on_eval = [r for r in on_records if r["type"] == "evaluation"]
    off_eval = [r for r in off_records if r["type"] == "evaluation"]
    assert len(on_eval) == len(off_eval) == 1
    for key in strip:
        on_eval[0].pop(key, None)
        off_eval[0].pop(key, None)
    assert on_eval[0] == off_eval[0]


# ------------------------------------------------------------ the code governs
def _authority_runner(tmp_path, frame):
    """A runner with the entry-clock stack the seam reads."""
    from goldenswing.live.hierarchy import TimeframeStack

    runner = _runner(tmp_path)
    runner._stack = TimeframeStack.from_frames({"H1": frame})
    return runner


def _authority(runner, *, candidate=None, evidence=None,
               decision=None, entry_tf="H1"):
    return runner._judgment_authority(
        candidate=candidate or _candidate(),
        decision=decision or _decision(),
        evidence=evidence or _evidence(),
        intended_tf=entry_tf,
        eval_now=datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc),
        route="h1_trigger")


def _never_called_urlopen(request, timeout=None):
    """The verdict path must never consult the brain — if the
    seam (or anything on it) calls out, this fires."""
    raise AssertionError(
        "the verdict path consulted the brain")


def test_code_governs_over_brain_abstain(monkeypatch, tmp_path):
    """The pin (owner, 2026-11-27 — no AI judges a trade):
    the code TAKES (strong long displacement, HTF with,
    score over floor) and a configured brain would say
    abstain.  The verdict is the CODE's take — the order
    path proceeds — and the brain is never even consulted
    (the stub fires if it is)."""
    monkeypatch.setenv("GS_JUDGMENT_BRAIN_API_KEY", "k")
    monkeypatch.setenv("GS_JUDGMENT_BRAIN_MODEL", "m")
    monkeypatch.setattr(
        "goldenswing.execution.judgment_brain.urllib.request.urlopen",
        _never_called_urlopen)
    frame = _frame([3895.0] * 17 + [3898.0, 3902.0, 3905.0])
    bundle, refusal = _authority(_authority_runner(tmp_path, frame))
    assert bundle is not None
    judgment = bundle["judgment"]
    assert judgment["rung"] == "code"
    assert judgment["decision"] == "take"
    assert "brain_note" not in judgment
    assert refusal is None


def test_code_governs_over_brain_take(monkeypatch, tmp_path):
    """The mirror pin: the code ABSTAINS (the structure
    reads short against a long candidate) and a configured
    brain would say take.  The verdict is the CODE's
    abstain — the order path refuses — and the brain is
    never even consulted."""
    monkeypatch.setenv("GS_JUDGMENT_BRAIN_API_KEY", "k")
    monkeypatch.setenv("GS_JUDGMENT_BRAIN_MODEL", "m")
    monkeypatch.setattr(
        "goldenswing.execution.judgment_brain.urllib.request.urlopen",
        _never_called_urlopen)
    frame = _frame([3905.0] * 17 + [3902.0, 3898.0, 3895.0])
    evidence = _evidence(
        observed={"entry": 3900.0, "stop": 3890.0, "stop_dist": 10.0,
                  "bias_d1": -1, "bias_h4": -1,
                  "session_bucket": "overlap"},
        inferred={"score_total": 40.0, "score_flipped": False,
                  "regime": "clean_trend", "posture": "TREND_PULLBACK",
                  "journey_state": "with_journey", "room_r": 2.0})
    bundle, refusal = _authority(
        _authority_runner(tmp_path, frame), evidence=evidence)
    assert bundle is not None
    judgment = bundle["judgment"]
    assert judgment["rung"] == "code"
    assert judgment["decision"] == "abstain"
    assert "brain_note" not in judgment
    assert refusal is not None
    assert refusal.startswith("judgment core abstained")


def test_no_brain_configured_code_decides_with_no_brain_note(
        monkeypatch, tmp_path):
    """No brain configured (no key, no model): the code
    decides, and the record carries no brain_note at all —
    there is no brain part to be missing.  The record is
    the verdict + its evidence."""
    for name in ("GS_JUDGMENT_BRAIN_API_KEY",
                 "GS_JUDGMENT_BRAIN_MODEL",
                 "GS_JUDGMENT_BRAIN_URL",
                 "GS_JUDGMENT_BRAIN_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    frame = _frame([3895.0] * 17 + [3898.0, 3902.0, 3905.0])
    bundle, refusal = _authority(_authority_runner(tmp_path, frame))
    assert bundle is not None
    judgment = bundle["judgment"]
    assert judgment["rung"] == "code"
    assert judgment["decision"] == "take"
    assert refusal is None
    assert "brain_note" not in judgment


def test_chart_rides_the_structure_record():
    """The chart rides the record's evidence: the recent
    entry-clock candles ride the chart read, most recent
    last, with the clock named.  No frame = no chart, never
    invented."""
    frame = _frame([3900.0] * 17 + [3901.0, 3903.0, 3905.0])
    structure = measure_decision_structure(
        candidate=_candidate(), evidence=_evidence(),
        context=_TREND_CTX, frames={"H1": frame}, entry_tf="H1",
        decision=_decision())
    data = structure.to_dict()
    json.dumps(data)  # plain data, as before
    chart = data["chart"]
    assert chart["tf"] == "H1"
    assert len(chart["bars"]) == 12
    assert chart["bars"][-1]["c"] == 3905.0
    assert chart["bars"][0]["c"] == 3900.0
    assert set(chart["bars"][-1]) == {"t", "o", "h", "l", "c"}
    no_frame = measure_decision_structure(
        candidate=_candidate(), evidence=_evidence(),
        context=_TREND_CTX, decision=_decision()).to_dict()
    assert no_frame["chart"] == {"tf": None, "bars": []}


def test_drafting_request_carries_the_chart_and_the_decided_verdict(
        monkeypatch):
    """The request the drafting brain actually sends carries
    the chart AND the verdict already decided — the candles
    and the decision reach the model; the reply is an
    explanation, never a verdict."""
    seen = {}

    class _CapResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {
                "content": json.dumps({
                    "explanation": "the pullback held and the "
                                   "long takes it"})}}]}).encode(
                        "utf-8")

    def _capture(request, timeout=None):
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _CapResp()

    monkeypatch.setattr(
        "goldenswing.execution.judgment_brain.urllib.request.urlopen",
        _capture)
    frame = _frame([3900.0] * 17 + [3901.0, 3903.0, 3905.0])
    structure = measure_decision_structure(
        candidate=_candidate(), evidence=_evidence(),
        context=_TREND_CTX, frames={"H1": frame}, entry_tf="H1",
        decision=_decision()).to_dict()
    verdict = _judge(frames={"H1": frame}, entry_tf="H1")
    out = draft_judgment_explanation(
        structure, verdict, _candidate(), _provider())
    assert isinstance(out, str)
    assert "pullback held" in out
    sent = json.loads(seen["body"]["messages"][1]["content"])
    assert sent["structure"]["chart"]["tf"] == "H1"
    assert len(sent["structure"]["chart"]["bars"]) == 12
    assert sent["structure"]["chart"]["bars"][-1]["c"] == 3905.0
    assert sent["verdict"]["decision"] == "take"
    assert sent["verdict"]["direction"] == 1
