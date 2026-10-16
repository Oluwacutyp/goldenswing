"""CHUNK 3: runner <-> narrative wiring.

MT5Runner must call ``mark_entered`` (via ``_narrative_fill_plan``)
on narrative fills so TickManager runs the narrative management
plan, and ``finalize_exit`` (STATE 7) on narrative closes.
Non-narrative fills/closes are untouched; any wiring failure is
fail-open (position still tracked on TickConfig defaults).
"""
from types import SimpleNamespace

from goldenswing.execution.mt5_runner import MT5Runner
from goldenswing.live.narrative import (
    ENTRY, MANAGEMENT, NarrativeState, mark_entered)


def _bare_runner():
    r = object.__new__(MT5Runner)
    r._stack = None
    r.journal = SimpleNamespace(log_note=lambda *a, **k: None)
    r._narrative_states = {}
    return r


def _entry_state():
    st = NarrativeState()
    st.state = ENTRY
    st.direction = 1
    st.late_confirm = False
    st.high_vol = False
    st.entry_tier = ""
    st.opposing = None
    st.atr_value = 0.0
    st.atr_ratio = 0.0
    return mark_entered(
        st, entry_price=100.0, stop=90.0, targets=[110.0, 120.0, 130.0])


def test_fill_plan_ignores_non_narrative():
    r = _bare_runner()
    sig = SimpleNamespace(kind="fvg_limit")
    state, kwargs = r._narrative_fill_plan(sig, 100.0)
    assert state is None and kwargs == {}


def test_fill_plan_no_stack_is_fail_open():
    r = _bare_runner()
    sig = SimpleNamespace(kind="narrative")
    state, kwargs = r._narrative_fill_plan(sig, 100.0)
    assert state is None and kwargs == {}


def test_fill_plan_bad_frames_is_fail_open():
    r = _bare_runner()
    notes = []
    r.journal = SimpleNamespace(log_note=lambda m: notes.append(m))
    r._stack = SimpleNamespace(frames={"H1": None})
    sig = SimpleNamespace(kind="narrative", invalidation=90.0,
                          targets=[110.0])
    state, kwargs = r._narrative_fill_plan(sig, 100.0)
    assert state is None and kwargs == {}
    assert notes  # the failure was journalled, not swallowed


def _close_runner():
    r = _bare_runner()
    notes = []
    r.journal = SimpleNamespace(
        log_note=lambda m: notes.append(m),
        log_close=lambda *a, **k: None)
    r._det_journal = lambda *a, **k: None
    r._send_alert = lambda *a, **k: None
    r._playbook_from_alert_id = lambda *a: "narrative"
    r._classify_close = lambda *a: "TP hit"
    r.guards = SimpleNamespace(record_outcome=lambda *a: None)
    r.entry_gate = SimpleNamespace(record_close=lambda *a: None)
    r.feed = SimpleNamespace(
        account_state=lambda: {"balance": 1000.0})
    r.executor = SimpleNamespace(
        symbol_spec=lambda *a: SimpleNamespace(
            money_for_move=lambda *a: 0.0))
    removed = []
    pos = SimpleNamespace(
        ticket=7, direction=1, entry_price=100.0, risk_price=10.0,
        alert_id="a1", partial_done=False, volume=0.01)
    r.tick_manager = SimpleNamespace(
        position=lambda ticket: pos,
        remove=lambda ticket: removed.append(ticket))
    r._notes = notes
    r._removed = removed
    return r


def test_record_close_finalizes_narrative_state():
    r = _close_runner()
    st = _entry_state()
    assert st.state == MANAGEMENT
    r._narrative_states[7] = st
    r._record_close(7, 120.0, "TP hit", {"reconciled_closed": 0})
    assert 7 not in r._narrative_states
    assert r._removed == [7]
    assert any("STATE 7 EXIT" in n for n in r._notes)


def test_record_close_without_narrative_state_is_unchanged():
    r = _close_runner()
    r._record_close(7, 120.0, "TP hit", {"reconciled_closed": 0})
    assert not any("STATE 7 EXIT" in n for n in r._notes)
    assert r._removed == [7]
