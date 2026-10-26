"""Telegram sender — Bot API sendPhoto/sendMessage + inline buttons, over HTTPS.

Configuration (environment):

- ``GS_TELEGRAM_BOT_TOKEN`` — bot token from BotFather
- ``GS_TELEGRAM_CHAT_ID``   — destination chat id
- ``GS_ALERT_OUTBOX``       — outbox dir for offline mode (optional)

With both credentials present the sender is LIVE; otherwise every send
goes to the outbox and reports so.  The button taps come back as
``callback_query`` updates — see :mod:`goldenswing.alerts.callbacks`
for the handler interface the execution phase implements.

Resilience contract (Phase 11, mined from the v18 bot's notifier):

- every live send retries with exponential backoff (429 honours the
  API's ``retry_after``);
- retryable: 429 / 5xx / network errors / timeouts.  Fatal (no retry):
  400 / 401 / 403 / 404 — retrying those is just noise;
- a photo that will not send falls back to the caption as a text
  message, so the thesis still arrives when the chart doesn't;
- :class:`ErrorDeduper` stops a stuck condition from flooding the
  phone: same error text (first 80 chars) is sent at most once per
  600s window.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import outbox
from .callbacks import AlertAction, make_callback_data
from .model import Alert, SendResult

_API_PHOTO = "https://api.telegram.org/bot{token}/sendPhoto"
_API_MESSAGE = "https://api.telegram.org/bot{token}/sendMessage"
_API_DOCUMENT = "https://api.telegram.org/bot{token}/sendDocument"
_CAPTION_LIMIT = 1024          # Telegram photo-caption hard limit
_MESSAGE_LIMIT = 4096          # Telegram text-message hard limit
_DOCUMENT_LIMIT_MB = 50        # Telegram bot sendDocument hard limit

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_FATAL_STATUS = {400, 401, 403, 404}

#: Skill transport: when no GS_TELEGRAM_BOT_TOKEN is configured, sends go
#: through the telegram skill's tg-call CLI, which authenticates via the
#: vault-backed custom.telegram connector (surrogate exchange with authd).
#: The surrogate must be inserted RAW into the URL path -- see
#: ~/workspace/skills/telegram/SKILL.md.  Falls back to outbox only when
#: there is no chat id at all.
_TG_CALL = os.path.expanduser("~/workspace/skills/telegram/bin/tg-call")


def _skill_available() -> bool:
    return os.path.isfile(_TG_CALL) and os.access(_TG_CALL, os.X_OK)


def inline_keyboard(alert: Alert) -> dict:
    """The Execute / Paper / Skip inline keyboard for one alert."""
    return {"inline_keyboard": [[
        {"text": "✅ Execute",
         "callback_data": make_callback_data(AlertAction.EXECUTE,
                                             alert.alert_id)},
        {"text": "📄 Paper",
         "callback_data": make_callback_data(AlertAction.PAPER,
                                             alert.alert_id)},
        {"text": "⏭ Skip",
         "callback_data": make_callback_data(AlertAction.SKIP,
                                             alert.alert_id)},
    ]]}


class ErrorDeduper:
    """Flood guard for error/status chatter.

    ``should_send`` returns True the first time a message key is seen,
    then False until ``window_seconds`` have passed.  Keys are the
    message's first ``key_len`` chars — a stuck condition repeats the
    same text, a new condition gets through.
    """

    def __init__(self, window_seconds: float = 600.0, key_len: int = 80):
        self.window_seconds = float(window_seconds)
        self.key_len = int(key_len)
        self._last: dict[str, float] = {}

    def should_send(self, message: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        key = (message or "")[:self.key_len]
        last = self._last.get(key)
        if last is not None and now - last < self.window_seconds:
            return False
        self._last[key] = now
        return True


class TelegramSender:
    def __init__(self, bot_token: str | None = None,
                 chat_id: str | None = None,
                 outbox_dir: str | Path | None = None,
                 session=None, timeout: float = 30.0,
                 max_retries: int = 3, backoff_base: float = 1.0,
                 sleeper=None):
        self.bot_token = bot_token or os.environ.get(
            "GS_TELEGRAM_BOT_TOKEN") or None
        self.chat_id = chat_id or os.environ.get(
            "GS_TELEGRAM_CHAT_ID") or None
        self.outbox_dir = outbox.resolve_outbox(outbox_dir)
        self._session = session        # injectable for tests
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = float(backoff_base)
        self._sleeper = sleeper or time.sleep

    @classmethod
    def from_env(cls, **kw) -> "TelegramSender":
        return cls(**kw)

    @property
    def mode(self) -> str:
        if self.bot_token and self.chat_id:
            return "live"
        if self.chat_id and _skill_available():
            return "skill"
        return "outbox"

    # ------------------------------------------------------------- core
    def _attempt(self, url: str, data: dict, files: dict | None = None
                 ) -> tuple[bool, dict | str]:
        """One HTTP attempt.  Returns (ok, body) or (False, error)."""
        import requests

        session = self._session or requests
        try:
            if files is not None:
                resp = session.post(url, data=data, files=files,
                                    timeout=self.timeout)
            else:
                resp = session.post(url, json=data, timeout=self.timeout)
            ctype = resp.headers.get("content-type", "") if hasattr(
                resp, "headers") else ""
            body = resp.json() if ctype.startswith("application/json") \
                else {}
            if resp.status_code == 200 and body.get("ok"):
                return True, body
            if resp.status_code in _FATAL_STATUS:
                return False, (f"fatal Telegram API {resp.status_code}: "
                               f"{body.get('description', '')}"[:200])
            if resp.status_code == 429:
                wait = 1.0
                try:
                    wait = float(body.get("parameters", {})
                                 .get("retry_after", 1))
                except (TypeError, ValueError):
                    pass
                return False, f"rate-limited, retry_after={wait:.0f}s"
            return False, (f"Telegram API {resp.status_code}: "
                           f"{body.get('description', '')}"[:200])
        except Exception as exc:  # network down, timeout, DNS — retryable
            return False, f"{type(exc).__name__}: {exc}"

    def _skill_attempt(self, url: str, data: dict,
                       files: dict | None = None
                       ) -> tuple[bool, dict | str]:
        """One send via the telegram skill's tg-call CLI.

        Same (ok, body) contract as :meth:`_attempt`.  File bytes are
        staged to temp files; the CLI authenticates via the vault
        surrogate, so no token ever touches this process.
        """
        import subprocess
        import tempfile

        method = url.rstrip("/").rsplit("/", 1)[-1] or "sendMessage"
        tmp_paths: list[str] = []
        try:
            cmd = [_TG_CALL, "--method", method,
                   "--data", json.dumps(data)]
            for field, (fname, content, _ctype) in (files or {}).items():
                tmp = tempfile.NamedTemporaryFile(
                    prefix="tgskill_", suffix=f"_{fname}", delete=False)
                tmp.write(content)
                tmp.close()
                tmp_paths.append(tmp.name)
                cmd += ["--file", f"{field}:{tmp.name}"]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.timeout + 30)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        try:
            body = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            return False, (f"tg-call bad output: "
                           f"{(proc.stdout or proc.stderr)[:150]}")
        if body.get("ok"):
            return True, body
        desc = str(body.get("description") or body.get("error", ""))[:200]
        code = body.get("error_code", 0)
        if code in _FATAL_STATUS:
            return False, f"fatal Telegram API {code}: {desc}"
        if code == 429:
            return False, "rate-limited, retry_after=1s"
        return False, f"Telegram API {code or '?'}: {desc}"

    def _skill_retry(self, url: str, data: dict,
                     files: dict | None = None) -> tuple[bool, str]:
        """Retry wrapper around :meth:`_skill_attempt`, same contract."""
        last_error = "no attempts made"
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            ok, info = self._skill_attempt(url, data, files)
            if ok:
                mid = info.get("result", {}).get("message_id") \
                    if isinstance(info, dict) else None
                return True, f"ok, message_id={mid}"
            last_error = str(info)
            retryable = not last_error.startswith("fatal")
            if not retryable or attempt == attempts - 1:
                break
            wait = self.backoff_base * (2 ** attempt)
            if "retry_after=" in last_error:
                try:
                    wait = max(wait, float(
                        last_error.split("retry_after=")[1].split("s")[0]))
                except (ValueError, IndexError):
                    pass
            self._sleeper(wait)
        return False, f"gave up after {attempts} attempt(s): {last_error}"

    def _post_with_retry(self, url: str, data: dict,
                         files: dict | None = None) -> tuple[bool, str]:
        """POST with backoff.  Returns (delivered, detail)."""
        if self.mode == "skill":
            return self._skill_retry(url, data, files)
        last_error = "no attempts made"
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            ok, info = self._attempt(url, data, files)
            if ok:
                mid = info.get("result", {}).get("message_id") \
                    if isinstance(info, dict) else None
                return True, f"ok, message_id={mid}"
            last_error = str(info)
            retryable = not last_error.startswith("fatal")
            if not retryable or attempt == attempts - 1:
                break
            wait = self.backoff_base * (2 ** attempt)
            if "retry_after=" in last_error:
                try:
                    wait = max(wait, float(
                        last_error.split("retry_after=")[1].split("s")[0]))
                except (ValueError, IndexError):
                    pass
            self._sleeper(wait)
        return False, f"gave up after {attempts} attempt(s): {last_error}"

    # ------------------------------------------------------------- API
    def send_text(self, text: str, *, silent: bool = False,
                  reply_markup: dict | None = None) -> SendResult:
        """A plain text message (position updates, summaries, errors).

        ``silent`` maps to Telegram's ``disable_notification`` — the
        phone doesn't buzz for routine chatter, only for things that
        need a human.
        """
        if self.mode == "outbox":
            return SendResult(
                "telegram", False, "outbox",
                "no GS_TELEGRAM_BOT_TOKEN/GS_TELEGRAM_CHAT_ID — "
                "text not queued (outbox is for chart alerts)")
        if len(text) > _MESSAGE_LIMIT:
            text = text[:_MESSAGE_LIMIT - 1] + "…"
        payload: dict = {"chat_id": self.chat_id, "text": text,
                         "disable_notification": silent}
        if reply_markup is not None:
            payload["reply_markup"] = json.dumps(reply_markup)
        ok, detail = self._post_with_retry(
            _API_MESSAGE.format(token=self.bot_token), payload)
        return SendResult("telegram", ok, "live", detail)

    def send_document(self, path: str | Path, *,
                      caption: str | None = None,
                      reply_markup: dict | None = None) -> SendResult:
        """Send a file as a Telegram document (reports, the journal DB).

        Same retry/backoff contract as the other sends.  Bot API limit
        is 50MB per document — larger files are refused up front with
        an honest message rather than a doomed upload.
        """
        from pathlib import Path as _Path
        p = _Path(path)
        if self.mode == "outbox":
            return SendResult(
                "telegram", False, "outbox",
                "no GS_TELEGRAM_BOT_TOKEN/GS_TELEGRAM_CHAT_ID — "
                f"document {p.name} not sent")
        if not p.exists():
            return SendResult("telegram", False, "live",
                              f"document not found: {p}")
        size_mb = p.stat().st_size / 1e6
        if size_mb > _DOCUMENT_LIMIT_MB:
            return SendResult(
                "telegram", False, "live",
                f"{p.name} is {size_mb:.1f}MB — over Telegram's "
                f"{_DOCUMENT_LIMIT_MB}MB bot document limit")
        data: dict = {"chat_id": self.chat_id,
                      "disable_notification": True}
        if caption:
            data["caption"] = caption[:_CAPTION_LIMIT - 1] + "…" \
                if len(caption) > _CAPTION_LIMIT else caption
        if reply_markup is not None:
            data["reply_markup"] = json.dumps(reply_markup)
        with p.open("rb") as fh:
            content = fh.read()
        ok, detail = self._post_with_retry(
            _API_DOCUMENT.format(token=self.bot_token), data,
            files={"document": (p.name, content,
                                "application/octet-stream")})
        return SendResult("telegram", ok, "live",
                          f"sendDocument {p.name}: {detail}")

    def send_photo(self, png: bytes, caption: str | None = None, *,
                   filename: str = "chart.png") -> SendResult:
        """Send raw PNG bytes as a Telegram photo (MT5 signal/fill charts).

        Same retry/backoff contract as the other sends.  A photo that
        will not send falls back to the caption as a text message, so
        the thesis still arrives when the chart doesn't.
        """
        if self.mode == "outbox":
            return SendResult(
                "telegram", False, "outbox",
                "no GS_TELEGRAM_BOT_TOKEN/GS_TELEGRAM_CHAT_ID — "
                "photo not queued (outbox is for chart alerts)")
        cap = caption or ""
        if len(cap) > _CAPTION_LIMIT:
            cap = cap[:_CAPTION_LIMIT - 1] + "…"
        ok, detail = self._post_with_retry(
            _API_PHOTO.format(token=self.bot_token),
            data={"chat_id": self.chat_id, "caption": cap},
            files={"photo": (filename, png, "image/png")})
        if ok:
            return SendResult("telegram", True, "live",
                              f"sendPhoto {detail}")
        text_res = self.send_text(cap)
        return SendResult(
            "telegram", text_res.delivered, "live",
            f"photo failed ({detail}); text fallback "
            f"{'sent' if text_res.delivered else 'failed'}: "
            f"{text_res.detail}")

    def send(self, alert: Alert) -> SendResult:
        if self.mode == "outbox":
            paths = outbox.write_alert(self.outbox_dir, alert, "telegram")
            return SendResult(
                "telegram", False, "outbox",
                "no GS_TELEGRAM_BOT_TOKEN/GS_TELEGRAM_CHAT_ID — wrote "
                + ", ".join(p.name for p in paths))
        caption = alert.text
        if len(caption) > _CAPTION_LIMIT:
            caption = caption[:_CAPTION_LIMIT - 1] + "…"
        ok, detail = self._post_with_retry(
            _API_PHOTO.format(token=self.bot_token),
            data={"chat_id": self.chat_id, "caption": caption,
                  "reply_markup": json.dumps(inline_keyboard(alert))},
            files={"photo": (f"{alert.alert_id}.png", alert.chart_png,
                             "image/png")})
        if ok:
            return SendResult("telegram", True, "live",
                              f"sendPhoto {detail}")
        # Photo refused — the thesis still matters more than the chart.
        text_res = self.send_text(caption)
        return SendResult(
            "telegram", text_res.delivered, "live",
            f"photo failed ({detail}); text fallback "
            f"{'sent' if text_res.delivered else 'failed'}: "
            f"{text_res.detail}")
