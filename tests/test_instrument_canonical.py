"""Broker-symbol canonicalization (live bug 2026-10-21).

The MT5 runner adopts the broker's resolved gold symbol for orders
(Exness: ``XAUUSDm``).  That spelling leaked into
``EngineContext.instrument``, so every playbook's required-context
tripwire ("validated universe is XAUUSD H1") rejected live MT5
candidates — the EC2 /status showed ``retracement_trap`` blocked with
``got 'XAUUSDm'`` while price sat inside an armed POI.

The tripwire validates the ASSET, not the broker's spelling.
``EngineContext`` canonicalizes any ``XAUUSD*`` variant to
``XAUUSD`` at construction; a genuinely wrong instrument still trips.
"""
import pandas as pd

from goldenswing.playbooks.base import EngineContext, canonical_instrument
from goldenswing.playbooks.retracement_trap import RetracementTrapPlaybook


def _ctx(instrument):
    return EngineContext(df=pd.DataFrame(), states=[], years=0.0,
                         instrument=instrument)


def test_canonical_instrument_mapping():
    assert canonical_instrument("XAUUSD") == "XAUUSD"
    assert canonical_instrument("XAUUSDm") == "XAUUSD"
    assert canonical_instrument("XAUUSD.r") == "XAUUSD"
    assert canonical_instrument("XAUUSD.pro") == "XAUUSD"
    assert canonical_instrument("GC=F") == "GC=F"
    assert canonical_instrument("EURUSD") == "EURUSD"


def test_engine_context_canonicalizes_broker_symbol():
    assert _ctx("XAUUSDm").instrument == "XAUUSD"
    assert _ctx("XAUUSD").instrument == "XAUUSD"
    assert _ctx("GC=F").instrument == "GC=F"


def test_retracement_trap_tripwire_accepts_exness_symbol():
    pb = RetracementTrapPlaybook()
    facts = {"swing_idx": 10, "break_idx": 12, "reclaim_idx": 14,
             "direction": 1, "level": 4100.0}
    ok, reason = pb.check_live_signal(object(), _ctx("XAUUSDm"),
                                      context=facts)
    assert ok, reason


def test_tripwire_still_rejects_wrong_instrument():
    pb = RetracementTrapPlaybook()
    facts = {"swing_idx": 10, "break_idx": 12, "reclaim_idx": 14,
             "direction": 1, "level": 4100.0}
    ok, reason = pb.check_live_signal(object(), _ctx("EURUSD"),
                                      context=facts)
    assert not ok
    assert "validated universe" in reason
