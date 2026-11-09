"""WS1-P7 — the GoldenSwing side of the shared bundle contract.

Three pins:

1. Every harvest ZIP (full AND slim fallback, both registries —
   they share build_zip_bundle) carries ``bundle.manifest.json``
   in the Devon contract shape, written LAST, listing EVERY other
   member with matching size + sha256, so Devon's Archivist
   verifies the whole bundle with nothing unmanifested.

2. A GS-produced bundle opens and verifies through DEVON's real
   reader (nomorals.archives open_bundle/verify_bundle) when a
   Devon checkout is available on the box; skipped otherwise —
   cross-repo proof cannot be faked with a local re-implementation.

3. backtest_export.chunk_text keeps its exact pre-fold behavior
   while delegating to alerts.markup.split_text (one splitting
   authority), and /statebundle exports journal + state as one
   contract bundle ONLY after integrity verification — damaged
   journals (bad JSONL lines, GSJ1 CRC/chain damage, a torn GSJ1
   tail) and unparseable state files refuse the export with the
   damage named, and the export never mutates the journal.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from goldenswing.alerts import commands as _commands
from goldenswing.alerts.bundle import BUNDLE_MANIFEST_NAME
from goldenswing.alerts.commands import (COMMANDS, CommandContext,
                                         StateBundleError,
                                         build_state_bundle,
                                         build_zip_bundle)
from goldenswing.alerts.markup import split_text
from goldenswing.engine import chunk_text
from goldenswing.engine.journal import (DeterministicJournal,
                                        FencedAuthority)
from goldenswing.execution.learning import TradeJournal


# ------------------------------------------------------------ helpers

def _assert_contract_manifest(zf: zipfile.ZipFile, *, kind: str):
    """The bundle manifest in ``zf`` satisfies the Devon contract
    against the archive's own contents.  Returns the manifest."""
    names = zf.namelist()
    assert names[-1] == BUNDLE_MANIFEST_NAME, (
        "contract manifest must be written LAST")
    manifest = json.loads(zf.read(BUNDLE_MANIFEST_NAME))
    assert manifest["format"] == "nomorals-bundle"
    assert manifest["version"] == 1
    assert manifest["created_by"] == "goldenswing"
    assert manifest["kind"] == kind
    assert isinstance(manifest["created_at"], float)
    recorded = {m["name"]: m for m in manifest["members"]}
    # EVERY other member is recorded — nothing unmanifested.
    assert set(recorded) == set(names) - {BUNDLE_MANIFEST_NAME}
    for name, m in recorded.items():
        blob = zf.read(name)
        assert m["size"] == len(blob), name
        assert m["sha256"] == hashlib.sha256(blob).hexdigest(), name
        assert m["is_dir"] is False, name
    return manifest


def _results_md(manifest: dict) -> str:
    return "# results\n" + "\n".join(
        f"- {f['arcname']}" for f in manifest["files"])


# ------------------------------------------------------------ (A) harvest

def test_harvest_bundle_carries_contract_manifest(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "trade_journal.jsonl").write_text('{"type": "note"}\n')
    (data_dir / "risk_state.json").write_text('{"halted": false}')
    zip_path, _manifest = build_zip_bundle(
        out_dir=tmp_path / "out", zip_prefix="goldenswing-harvest",
        files=[data_dir / "trade_journal.jsonl",
               data_dir / "risk_state.json"],
        data_dir=data_dir, manifest_base={"scope": "all"},
        results_md_fn=_results_md)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert {"trade_journal.jsonl", "risk_state.json", "RESULTS.md",
                "manifest.json", BUNDLE_MANIFEST_NAME} <= names
        manifest = _assert_contract_manifest(zf, kind="harvest")
        by_name = {m["name"]: m for m in manifest["members"]}
        # The family manifest is itself a verified member.
        assert "manifest.json" in by_name
        assert "RESULTS.md" in by_name


def test_harvest_slim_fallback_carries_contract_manifest(
        tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    journal = data_dir / "trade_journal.jsonl"
    journal.write_text('{"type": "note"}\n')
    big = data_dir / "big_state.json"
    big.write_bytes(os.urandom(30_000))          # incompressible
    # Cap below the full bundle (~30KB) but above the slim one.
    monkeypatch.setattr(_commands, "HARVEST_MAX_ZIP_BYTES", 10_000)
    zip_path, manifest = build_zip_bundle(
        out_dir=tmp_path / "out", zip_prefix="goldenswing-harvest",
        files=[journal, big], data_dir=data_dir,
        manifest_base={"scope": "all"}, results_md_fn=_results_md,
        slim_files=[journal])
    assert manifest["fallback_slim"] is True
    assert zip_path.name.endswith("-slim.zip")
    with zipfile.ZipFile(zip_path) as zf:
        assert "big_state.json" not in zf.namelist()
        _assert_contract_manifest(zf, kind="harvest")


# ------------------------------------------------- (A) cross-repo proof

def _devon_checkout() -> Path | None:
    """A Devon checkout whose archives.py carries the bundle seam
    (content-checked, so a stale tree never poses as the reader)."""
    candidates = []
    env = os.environ.get("DEVON_REPO_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [Path.home() / "workspace" / "ws1p7-verify"
                   / "devon-main",
                   Path.home() / "workspace" / "provfix",
                   Path.home() / "workspace" / "nomorals-2.0"]
    for cand in candidates:
        archives = cand / "nomorals" / "archives.py"
        if archives.is_file() and \
                "def verify_bundle" in archives.read_text(
                    encoding="utf-8", errors="ignore"):
            return cand
    return None


def test_devon_archivist_opens_and_verifies_gs_harvest(tmp_path):
    devon = _devon_checkout()
    if devon is None:
        pytest.skip("no Devon checkout with the bundle seam on this box")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "trade_journal.jsonl").write_text('{"type": "note"}\n')
    zip_path, _ = build_zip_bundle(
        out_dir=tmp_path / "out", zip_prefix="goldenswing-harvest",
        files=[data_dir / "trade_journal.jsonl"], data_dir=data_dir,
        manifest_base={"scope": "journal"}, results_md_fn=_results_md)
    script = (
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from nomorals.archives import open_bundle, verify_bundle\n"
        "info = open_bundle(sys.argv[2])\n"
        "verdict = verify_bundle(sys.argv[2])\n"
        "print(json.dumps({'legacy': info['legacy'],\n"
        "                  'kind': (info['manifest'] or {}).get('kind'),\n"
        "                  'members': [m['name'] for m in info['members']],\n"
        "                  'family_scope': (info['family_manifest'] or {})\n"
        "                                  .get('scope'),\n"
        "                  'ok': verdict['ok'],\n"
        "                  'checked': verdict['checked'],\n"
        "                  'problems': verdict['problems']}))\n")
    proc = subprocess.run(
        [sys.executable, "-c", script, str(devon), str(zip_path)],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["legacy"] is False
    assert out["kind"] == "harvest"
    assert out["family_scope"] == "journal"   # family manifest rides along
    assert out["ok"] is True and out["problems"] == []
    assert out["checked"] == len(out["members"]) >= 3


# ------------------------------------------------------------ (B) chunk fold

def _legacy_chunk_text(text: str, limit: int = 4096) -> list[str]:
    """The pre-fold chunk_text algorithm, verbatim — the fold must
    be behavior-identical to it, not merely similar."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    def _flush():
        nonlocal current, size
        if current:
            chunks.append("".join(current))
            current, size = [], 0

    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            room = limit - size
            if room <= 0:
                _flush()
                room = limit
            take = line[:room]
            current.append(take)
            size += len(take)
            line = line[room:]
            if size >= limit:
                _flush()
        if line:
            if size + len(line) > limit:
                _flush()
            current.append(line)
            size += len(line)
    _flush()
    return chunks


_CHUNK_CASES = [
    "",
    "short",
    "x" * 4096,
    "x" * 9000,
    "line one\nline two\n" * 500,
    "no newline at end\n" * 300 + "tail",
    "a" * 100 + "\n" + "b" * 5000 + "\n" + "c" * 100,
    "\n".join(str(i) * (i % 97 + 1) for i in range(400)),
]


@pytest.mark.parametrize("limit", [4096, 100, 17])
def test_chunk_text_fold_is_behavior_identical(limit):
    for text in _CHUNK_CASES:
        expected = _legacy_chunk_text(text, limit)
        assert chunk_text(text, limit) == expected
        assert split_text(text, limit) == expected
        assert "".join(expected) == text          # lossless
        assert all(len(c) <= limit for c in expected)


# ------------------------------------------------------------ (C) state bundle

def _authority() -> FencedAuthority:
    return FencedAuthority(epoch=1, term=1, leader_id="test-node",
                           token_version=1, secret=b"ws1p7-test-secret")


def _det_journal(det_dir: Path, frames: int = 2) -> Path:
    det_dir.mkdir(parents=True, exist_ok=True)
    auth = _authority()
    journal = DeterministicJournal(str(det_dir / "deterministic.gsj"),
                                   auth)
    for i in range(frames):
        journal.append(epoch=1, term=1, leader_id="test-node",
                       token_version=1, token=auth.token(),
                       data={"kind": "note", "i": i})
    return det_dir / "deterministic.gsj"


def _state_ctx(tmp_path: Path, *, journal_name: str = "trade_journal.jsonl",
               det_frames: int | None = 2):
    """A data dir in the live shape + its CommandContext."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    journal_path = data_dir / journal_name
    journal = TradeJournal(journal_path)
    journal.log_note("boot")
    journal.append({"type": "open", "alert_id": "A1"})
    journal.append({"type": "close", "alert_id": "A1", "r_net": 1.0})
    (data_dir / "risk_state.json").write_text('{"halted": false}')
    (data_dir / "paper_state").mkdir(exist_ok=True)
    (data_dir / "paper_state" / "paper.json").write_text('{"balance": 100.0}')
    (data_dir / "paper_state" / "meta.json").write_text('{"last": "bar"}')
    if det_frames is not None:
        det_dir = data_dir / "deterministic"
        _det_journal(det_dir, det_frames)
        snaps = det_dir / "deterministic_snapshots"
        snaps.mkdir(exist_ok=True)
        (snaps / "manifest.json").write_text('{"snapshots": []}')
    ctx = CommandContext(journal=journal, journal_path=journal_path,
                         reports_dir=tmp_path / "reports")
    return ctx, data_dir


def test_state_bundle_happy_path(tmp_path):
    ctx, _data_dir = _state_ctx(tmp_path)
    zip_path, manifest = build_state_bundle(ctx)
    assert manifest["journal_records"] == 3
    assert manifest["deterministic_frames"] == 2
    assert manifest["deterministic_head_chain"]
    assert manifest["zip_bytes"] == zip_path.stat().st_size
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert {"trade_journal.jsonl", "risk_state.json",
                "paper_state/paper.json", "paper_state/meta.json",
                "deterministic/deterministic.gsj",
                "deterministic/deterministic_snapshots/manifest.json",
                "STATE.md", "manifest.json",
                BUNDLE_MANIFEST_NAME} <= names
        _assert_contract_manifest(zf, kind="journal-state")
        state_md = zf.read("STATE.md").decode()
        assert "integrity verified" in state_md
        assert "3 records" in state_md
        family = json.loads(zf.read("manifest.json"))
        assert family["journal_records"] == 3
        assert family["deterministic_frames"] == 2


def test_state_bundle_without_deterministic_journal(tmp_path):
    ctx, _ = _state_ctx(tmp_path, det_frames=None)
    zip_path, manifest = build_state_bundle(ctx)
    assert manifest["deterministic_frames"] is None
    with zipfile.ZipFile(zip_path) as zf:
        assert not any(n.endswith(".gsj") for n in zf.namelist())
        _assert_contract_manifest(zf, kind="journal-state")
        assert "not present" in zf.read("STATE.md").decode()


def test_state_bundle_refuses_damaged_trade_journal(tmp_path):
    ctx, data_dir = _state_ctx(tmp_path)
    with open(data_dir / "trade_journal.jsonl", "a") as fh:
        fh.write("this is not json\n")
        fh.write('{"type": "note"}\n')
    with pytest.raises(StateBundleError) as ei:
        build_state_bundle(ctx)
    assert "1 unreadable line" in str(ei.value)
    assert "refused" in str(ei.value)


def test_state_bundle_refuses_corrupt_deterministic_journal(tmp_path):
    ctx, data_dir = _state_ctx(tmp_path)
    gsj = data_dir / "deterministic" / "deterministic.gsj"
    blob = bytearray(gsj.read_bytes())
    blob[len(blob) // 2] ^= 0xFF          # payload/CRC damage
    gsj.write_bytes(bytes(blob))
    with pytest.raises(StateBundleError) as ei:
        build_state_bundle(ctx)
    assert "deterministic journal corrupt" in str(ei.value)


def test_state_bundle_refuses_torn_tail_and_never_truncates(tmp_path):
    ctx, data_dir = _state_ctx(tmp_path)
    gsj = data_dir / "deterministic" / "deterministic.gsj"
    before = gsj.stat().st_size
    with open(gsj, "ab") as fh:
        fh.write(b"GSJ1\x00\x00")          # partial frame: torn tail
    with pytest.raises(StateBundleError) as ei:
        build_state_bundle(ctx)
    assert "torn tail" in str(ei.value)
    # The export is read-only: the torn bytes are still there (the
    # journal's own loader would have truncated them on open).
    assert gsj.stat().st_size == before + 6


def test_state_bundle_refuses_unparseable_state_file(tmp_path):
    ctx, data_dir = _state_ctx(tmp_path)
    (data_dir / "risk_state.json").write_text("{not json")
    with pytest.raises(StateBundleError) as ei:
        build_state_bundle(ctx)
    assert "risk_state.json" in str(ei.value)


def test_state_bundle_command_surfaces():
    assert "/statebundle" in COMMANDS
    from goldenswing.execution.mt5_commands import MT5_COMMANDS
    assert "/statebundle" in MT5_COMMANDS


def test_statebundle_handler_refuses_without_journal(tmp_path):
    ctx = CommandContext(journal=None,
                         journal_path=tmp_path / "nope.jsonl",
                         reports_dir=tmp_path / "reports")
    _desc, handler = COMMANDS["/statebundle"]
    result = handler(ctx, [])
    assert result.document is None
    assert "refused" in result.text


def test_statebundle_mt5_handler_builds_bundle(tmp_path):
    ctx, _ = _state_ctx(tmp_path, journal_name="mt5_journal.jsonl")
    (ctx.journal_path.parent / "mt5_risk_state.json").write_text(
        '{"halted": false}')
    from goldenswing.execution.mt5_commands import MT5_COMMANDS
    _desc, handler = MT5_COMMANDS["/statebundle"]
    result = handler(ctx, [])
    assert result.document is not None
    zpath = Path(result.document)
    assert zpath.name.startswith("goldenswing-mt5-state-")
    with zipfile.ZipFile(zpath) as zf:
        _assert_contract_manifest(zf, kind="journal-state")
        assert "mt5_risk_state.json" in zf.namelist()
