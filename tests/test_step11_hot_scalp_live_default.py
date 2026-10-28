"""Step 11 (2026-10-27): Hot Scalp live ON by default, zero config.

Owner order, twice emphatic: "I meant live" (21:33 MDT),
then "I don't have the time to be configuring" (21:34 MDT).
The step-6 design made live a deliberate host-side act
(config/env flag, default OFF).  The owner has now given
that approval AND refused the configuration step — so the
approval moves into the default: pulling the code enables
live Hot Scalp authority with nothing to set on the host.

Mechanics (verified before editing): ``__post_init__``
applies the env override ONLY when the variable is set
(``if env_live is not None``), so flipping the config
default is sufficient — no env handling change needed.
Opt-out still works: GS_HOT_SCALP_LIVE=0 (or the config
flag False) disables it.  /set still cannot touch it.

v1 live limits are unchanged (step 6): flat-only,
market-shape-only, shared order path, claim-ownership,
paused guard, broker-agreement check, demo account only.
"""

import os

import pytest

from goldenswing.execution.mt5_runner import MT5RunnerConfig


@pytest.fixture(autouse=True)
def _no_live_env(monkeypatch):
    monkeypatch.delenv("GS_HOT_SCALP_LIVE", raising=False)
    monkeypatch.delenv("GS_HOT_SCALP_SHADOW", raising=False)


def test_live_on_by_default_with_no_configuration(tmp_path):
    cfg = MT5RunnerConfig(journal_path=tmp_path / "j.jsonl",
                          risk_state_path=tmp_path / "r.json",
                          kill_switch_path=tmp_path / "KILL")
    assert cfg.hot_scalp_live_enabled is True


def test_env_can_still_turn_live_off(monkeypatch, tmp_path):
    monkeypatch.setenv("GS_HOT_SCALP_LIVE", "0")
    cfg = MT5RunnerConfig(journal_path=tmp_path / "j.jsonl",
                          risk_state_path=tmp_path / "r.json",
                          kill_switch_path=tmp_path / "KILL")
    assert cfg.hot_scalp_live_enabled is False


def test_env_true_keeps_live_on(monkeypatch, tmp_path):
    monkeypatch.setenv("GS_HOT_SCALP_LIVE", "true")
    cfg = MT5RunnerConfig(journal_path=tmp_path / "j.jsonl",
                          risk_state_path=tmp_path / "r.json",
                          kill_switch_path=tmp_path / "KILL")
    assert cfg.hot_scalp_live_enabled is True


def test_shadow_still_on_by_default(tmp_path):
    cfg = MT5RunnerConfig(journal_path=tmp_path / "j.jsonl",
                          risk_state_path=tmp_path / "r.json",
                          kill_switch_path=tmp_path / "KILL")
    assert cfg.hot_scalp_shadow_enabled is True
