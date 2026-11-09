"""WS3-P7 — the desk draft bundle (GoldenSwing side).

Pins:
1. build_desk_bundle packs the /desk card (PNG + caption) as a
   nomorals-bundle in the Devon contract shape: manifest format
   nomorals-bundle, kind "gs-desk-card", written LAST, every
   other member recorded with matching size + sha256.
2. Devon's REAL reader verifies the bundle and Devon's landing
   inspection recognizes it (cross-repo subprocess proof when a
   Devon checkout with the P7 landing is on the box; skipped
   otherwise — the contract cannot be faked locally).
3. Daily identity: one filename per day; a same-day rebuild
   replaces it (supersede by rebuild), never a second identity.
4. /deskdraft delivers the bundle document; /desk itself is
   unchanged (same photo + caption from the same stats assembly)
   apart from the new affordance button, and renders/writes
   nothing extra when the affordance is unused.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

import pytest

from goldenswing.alerts.bundle import BUNDLE_MANIFEST_NAME
from goldenswing.alerts.commands import CommandContext
from goldenswing.alerts.desk_bundle import (
    DESK_BUNDLE_KIND,
    build_desk_bundle,
    desk_bundle_name,
)
from goldenswing.charts.desk import render_desk_card
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.mt5_commands import (
    MT5_COMMANDS,
    _desk_caption,
    _desk_day_stats,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"card" * 16
CAPTION = "🖥️ Desk 2026-11-09: +26.25 USD · 100% win rate · 1 closed"
STATS = {"day": "2026-11-09", "pnl": 26.25, "currency": "USD",
         "trades": 1, "win_rate": 100.0, "equity": 67.24,
         "halted": False}


def _assert_contract(zf: zipfile.ZipFile):
    names = zf.namelist()
    assert names[-1] == BUNDLE_MANIFEST_NAME
    manifest = json.loads(zf.read(BUNDLE_MANIFEST_NAME))
    assert manifest["format"] == "nomorals-bundle"
    assert manifest["version"] == 1
    assert manifest["created_by"] == "goldenswing"
    assert manifest["kind"] == DESK_BUNDLE_KIND == "gs-desk-card"
    assert isinstance(manifest["created_at"], float)
    recorded = {m["name"]: m for m in manifest["members"]}
    assert set(recorded) == set(names) - {BUNDLE_MANIFEST_NAME}
    for name, m in recorded.items():
        blob = zf.read(name)
        assert m["size"] == len(blob), name
        assert m["sha256"] == hashlib.sha256(blob).hexdigest(), name
        assert m["is_dir"] is False, name
    return manifest


def test_desk_bundle_contract_shape(tmp_path):
    zip_path, family = build_desk_bundle(
        out_dir=tmp_path, day="2026-11-09", png=PNG,
        caption=CAPTION, stats=STATS)
    assert zip_path.name == "goldenswing-desk-2026-11-09.zip"
    assert family["day"] == "2026-11-09"
    assert family["caption"] == CAPTION
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist()[:3] == ["desk.png", "caption.txt",
                                     "manifest.json"]
        assert zf.read("desk.png") == PNG          # the card as-is
        assert zf.read("caption.txt").decode("utf-8") == CAPTION
        fam = json.loads(zf.read("manifest.json"))
        assert fam["day"] == "2026-11-09"
        assert fam["stats"]["pnl"] == 26.25
        _assert_contract(zf)


def test_desk_bundle_daily_identity_supersede(tmp_path):
    first, _ = build_desk_bundle(
        out_dir=tmp_path, day="2026-11-09", png=PNG,
        caption=CAPTION, stats=STATS)
    second, _ = build_desk_bundle(
        out_dir=tmp_path, day="2026-11-09", png=PNG + b"v2",
        caption=CAPTION + " · later", stats=STATS)
    assert first == second == tmp_path / desk_bundle_name("2026-11-09")
    assert list(tmp_path.glob("goldenswing-desk-*.zip")) == [first]
    with zipfile.ZipFile(second) as zf:     # the rebuild replaced it
        assert zf.read("desk.png") == PNG + b"v2"
        _assert_contract(zf)
    assert desk_bundle_name("2026-11-10") != desk_bundle_name("2026-11-09")


# ── cross-repo proof through Devon's real reader + landing ─────────

def _devon_p7_checkout() -> Path | None:
    candidates = []
    env = os.environ.get("DEVON_REPO_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [Path.home() / "workspace" / "ws3p7-tree",
                   Path.home() / "workspace" / "provfix",
                   Path.home() / "workspace" / "nomorals-2.0"]
    for cand in candidates:
        landing = cand / "nomorals" / "agents" / "desk_drafts.py"
        if landing.is_file() and \
                "GS_DESK_BUNDLE_KIND" in landing.read_text(
                    encoding="utf-8", errors="ignore"):
            return cand
    return None


def test_devon_reader_verifies_and_landing_recognizes(tmp_path):
    devon = _devon_p7_checkout()
    if devon is None:
        pytest.skip("no Devon checkout with the P7 desk landing"
                    " on this box")
    zip_path, _ = build_desk_bundle(
        out_dir=tmp_path, day="2026-11-09", png=PNG,
        caption=CAPTION, stats=STATS)
    script = (
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from nomorals.archives import open_bundle, verify_bundle\n"
        "from nomorals.agents.desk_drafts import (\n"
        "    GS_DESK_BUNDLE_KIND, inspect_desk_bundle)\n"
        "info = open_bundle(sys.argv[2])\n"
        "verdict = verify_bundle(sys.argv[2])\n"
        "seen = inspect_desk_bundle(open(sys.argv[2], 'rb').read())\n"
        "print(json.dumps({'kind': (info['manifest'] or {})\n"
        "                          .get('kind'),\n"
        "                  'ok': verdict['ok'],\n"
        "                  'problems': verdict['problems'],\n"
        "                  'landing_kind': GS_DESK_BUNDLE_KIND,\n"
        "                  'desk': seen.get('desk'),\n"
        "                  'seen_ok': seen.get('ok'),\n"
        "                  'day': seen.get('day'),\n"
        "                  'caption': seen.get('caption')}))\n")
    proc = subprocess.run(
        [sys.executable, "-c", script, str(devon), str(zip_path)],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["kind"] == "gs-desk-card" == out["landing_kind"]
    assert out["ok"] is True and out["problems"] == []
    assert out["desk"] is True and out["seen_ok"] is True
    assert out["day"] == "2026-11-09"
    assert out["caption"] == CAPTION


# ── the commands ─────────────────────────────────────────────────────

def _today_journal(tmp_path):
    today = date.today().isoformat()
    journal = TradeJournal(tmp_path / "today.jsonl")
    journal.append({"type": "open", "alert_id": "t1",
                    "instrument": "XAUUSD", "playbook": "fvg_limit",
                    "direction": 1, "entry": 2005.0, "stop": 1990.0,
                    "volume": 0.01, "risk_money": 17.5,
                    "ts": today + "T10:00:00+00:00"})
    journal.append({"type": "close", "alert_id": "t1", "ticket": 1,
                    "exit": 2031.0, "reason": "target",
                    "pnl_money": 26.25, "r_net": 1.5, "hold_bars": 3,
                    "ts": today + "T10:05:00+00:00"})
    return journal, today


def test_deskdraft_command_delivers_the_bundle(tmp_path):
    journal, today = _today_journal(tmp_path)
    reports = tmp_path / "reports"
    ctx = CommandContext(journal=journal, mt5_control=None,
                         live_positions=[], reports_dir=reports)
    res = MT5_COMMANDS["/deskdraft"][1](ctx, [])
    assert res.document is not None
    assert Path(res.document).name == desk_bundle_name(today)
    assert "desk draft bundle" in (res.caption or "")
    with zipfile.ZipFile(res.document) as zf:
        _assert_contract(zf)
        fam = json.loads(zf.read("manifest.json"))
        assert fam["day"] == today
        # the bundle's card is the card /desk renders for the day
        stats, _ = _desk_day_stats(ctx)
        assert zf.read("desk.png") == render_desk_card(stats)
        assert zf.read("caption.txt").decode("utf-8") == \
            _desk_caption(stats)


def test_desk_unchanged_apart_from_the_affordance(tmp_path):
    journal, today = _today_journal(tmp_path)
    reports = tmp_path / "reports"
    ctx = CommandContext(journal=journal, mt5_control=None,
                         live_positions=[], reports_dir=reports)
    res = MT5_COMMANDS["/desk"][1](ctx, [])
    stats, _ = _desk_day_stats(ctx)
    assert res.photo == render_desk_card(stats)      # same card
    assert res.caption == _desk_caption(stats)       # same caption
    assert res.document is None
    # the affordance: a /deskdraft button now rides the keyboard
    buttons = [b for row in res.reply_markup["inline_keyboard"]
               for b in row]
    assert any(b["callback_data"] == "mt5:cmd:/deskdraft"
               for b in buttons)
    # unused, the affordance writes nothing anywhere
    assert not reports.exists() or not list(reports.iterdir())


def test_deskdraft_honest_refusals(tmp_path):
    ctx = CommandContext(journal=None, mt5_control=None,
                         live_positions=[],
                         reports_dir=tmp_path / "reports")
    res = MT5_COMMANDS["/deskdraft"][1](ctx, [])
    assert res.text == "no journal attached."
    journal, _today = _today_journal(tmp_path)
    ctx = CommandContext(journal=journal, mt5_control=None,
                         live_positions=[],
                         reports_dir=tmp_path / "reports")
    res = MT5_COMMANDS["/deskdraft"][1](ctx, ["extra"])
    assert res.text == "usage: /deskdraft"
