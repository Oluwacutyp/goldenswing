"""Unicode regression test: /assess all on Windows (2026-10-25).

The owner hit this live: ``/assess failed: UnicodeEncodeError:
'charmap' codec can't encode character '\\U0001f9ed'`` — the
assessment text starts with 🧭 and ``Path.write_text`` without an
explicit encoding defaults to the Windows locale codec (cp1252),
which cannot encode emoji. The fix is ``encoding="utf-8"`` at the
write sites. This test proves the handler passes that argument
(by recording the call — the platform codec itself is not
mockable, since CPython resolves it below the Python attribute)
and that the written bytes round-trip as UTF-8 with 🧭 intact.
"""

import pathlib
from unittest import mock

from goldenswing.alerts.commands import CommandContext
from goldenswing.execution.mt5_commands import _mt5_assess


class _Journal:
    def __init__(self, records):
        self._records = records

    def records(self):
        return list(self._records)


def _record(text):
    return {"type": "evaluation", "kind": "playbook_eval",
            "playbook": "sweep_reversal", "assessment_text": text}


def test_assess_all_writes_utf8_explicitly(tmp_path):
    long_text = "🧭 MARKET ASSESSMENT — XAUUSD\n" + "x" * 3600
    ctx = CommandContext(
        journal=_Journal([_record(long_text)]),
        reports_dir=tmp_path)
    calls = []
    orig = pathlib.Path.write_text

    def recording(self, data, encoding=None, errors=None, newline=None):
        calls.append({"encoding": encoding})
        return orig(self, data, encoding=encoding, errors=errors,
                    newline=newline)

    with mock.patch.object(pathlib.Path, "write_text", recording):
        result = _mt5_assess(ctx, ["all"])
    assert result.document is not None
    assert calls and calls[-1]["encoding"] == "utf-8"
    raw = pathlib.Path(result.document).read_bytes()
    assert raw.decode("utf-8").startswith("🧭")
