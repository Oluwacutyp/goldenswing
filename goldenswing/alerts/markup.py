"""Telegram HTML markup + lossless splitting for long bot messages.

Why this module exists (owner directive 2026-11-09): the bot's long
messages — /status with its forecast blocks above all — were being
CUT OFF.  The sender sliced any text over Telegram's 4096-char limit
to ``text[:4095] + "…"``, silently swallowing the tail (the owner's
screenshot ends mid-data at "72h up: unproven — top-band 1/2...").
Two fixes, both applied HERE, at the Telegram boundary, so every
producer (command replies, runner-pushed recaps, summaries) gets
the same standard and the plain-text sources stay untouched for
the other surfaces:

1. **Markup** — the plain structured text the codebase already
   writes (emoji section headers, backtick-monospace values,
   indented detail blocks) renders as Telegram HTML: section
   headers bold, backtick spans become ``<code>``, and the
   deep-detail blocks — the probability book's per-horizon lines,
   the forecast-trust per-band lines, the battery lines, the
   retrain-accrual lines — are wrapped in ``<blockquote
   expandable>`` so the message opens compact with the detail one
   tap away.  All dynamic text is HTML-escaped first; a literal
   ``<key>`` placeholder in usage text renders as ``<key>``, never
   as a tag.

2. **Lossless splitting** — expandable content still counts toward
   the 4096-char limit, so markup alone cannot fix overflow.  When
   a rendered message exceeds the limit it splits into multiple
   messages at SECTION boundaries (never mid-section when the
   section fits a message of its own, never mid-line, never an
   ellipsis).  A single section longer than the limit splits at
   line boundaries, with an open ``<blockquote>`` closed and
   reopened across the seam so every part is complete, balanced
   HTML on its own.

Facts are never altered: rendering changes presentation only, and
the split-integrity contract is that the parts, stripped of tags,
contain every source line exactly once.
"""
from __future__ import annotations

import html
import re
from typing import NamedTuple

MESSAGE_LIMIT = 4096           # Telegram text-message hard limit
_CAPTION_LIMIT = 1024          # Telegram photo/document caption limit
_QUOTE_OPEN = "<blockquote expandable>"
_QUOTE_CLOSE = "</blockquote>"

# Markers that open a verdict/marker row, never a section: lines
# starting with these are content, not headers.
_MARKER_HEADS = {"🟢", "🔴", "🟡", "➖", "✅", "❌", "⚠️", "⏸", "⏸️",
                 "▶", "▶️", "⏭", "⏭️", "•", "-", "—"}

# Sub-header openers whose MORE-INDENTED children are deep detail:
# the opener line itself stays visible; the children collapse into
# an expandable quote.
_DETAIL_HEADERS = (
    "probability book (F2",
    "forecast trust (E3",
    "battery (F3",
)

# Detail openers that are data lines themselves: the line AND its
# more-indented followers collapse together.
_DETAIL_LINES = (
    "retrain accrual:",
)

_CODE_RE = re.compile(r"`([^`\n]+)`")


def escape_html(text: str) -> str:
    """Escape the three characters Telegram HTML cares about."""
    return html.escape(text, quote=False)


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_header(line: str) -> bool:
    """A section header in the house plain-text style.

    Shape: not indented, opens with a symbol/emoji token (never a
    status marker), then a short title — either Title/UPPER case
    (``📡 Feed``, ``📊 MT5 ENGINE STATUS``) or ending in a colon
    (``📊 last 12 MT5 trades:``, ``🚫 why no trades — blocked
    legs:``).  Content lines fail one of the arms: marker-led rows
    start with a marker, verdict lines start lowercase, trade rows
    start with ``-`` or a digit.
    """
    if not line or line[0] == " ":
        return False
    head, sep, rest = line.partition(" ")
    if not sep or not rest.strip():
        return False
    if head in _MARKER_HEADS:
        return False
    if any(ch.isalnum() for ch in head):
        return False                    # a word/digit-led line
    if "`" in rest or len(rest.split()) > 9:
        return False
    if rest.rstrip().endswith(":"):
        return True
    first_alpha = next((ch for ch in rest if ch.isalpha()), "")
    return first_alpha.isupper()


def _detail_child_flags(lines: list[str]) -> list[bool]:
    """Per line: is it inside an expandable detail block?

    A detail block opens at one of the known detail headers/lines
    and covers the following lines that are MORE indented than the
    opener (a blank line or a dedent closes it).  ``_DETAIL_HEADERS``
    openers stay visible themselves; ``_DETAIL_LINES`` openers are
    detail too.
    """
    flags = [False] * len(lines)
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        is_header_detail = any(stripped.startswith(p)
                               for p in _DETAIL_HEADERS)
        is_line_detail = any(stripped.startswith(p)
                             for p in _DETAIL_LINES)
        if not (is_header_detail or is_line_detail):
            i += 1
            continue
        base = _indent_of(lines[i])
        if is_line_detail:
            flags[i] = True
        j = i + 1
        while j < len(lines):
            if not lines[j].strip():
                break
            if _indent_of(lines[j]) <= base:
                break
            flags[j] = True
            j += 1
        i = j
    return flags


def _render_line(line: str, *, header: bool) -> str:
    out = _CODE_RE.sub(r"<code>\1</code>", escape_html(line))
    if header:
        out = f"<b>{out}</b>"
    return out


def _wrap_long_line(line: str, limit: int) -> list[str]:
    """Split one over-long PLAIN line at spaces (hard-splitting a
    monster word) so a rendered line can never exceed the limit on
    its own.  Presentation-only: the words survive whole."""
    if len(line) <= limit:
        return [line]
    out: list[str] = []
    rest = line
    while len(rest) > limit:
        cut = rest.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
            out.append(rest[:cut])
            rest = rest[cut:]
        else:
            out.append(rest[:cut])
            rest = rest[cut + 1:]
    if rest:
        out.append(rest)
    return out


class _Line(NamedTuple):
    html: str          # rendered line, quote tags NOT included
    in_run: bool       # belongs to an expandable quote run
    run_first: bool    # first line of its quote run
    run_last: bool     # last line of its quote run
    is_header: bool


def render_messages(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Render plain structured text to Telegram HTML messages.

    Returns one HTML string per Telegram message, each ``<= limit``
    chars, tags balanced per message, every source line present
    exactly once across the parts (modulo the whitespace of wrapped
    monster lines).  Splitting prefers section boundaries: sections
    pack whole whenever they fit a message; only a section longer
    than the limit in itself splits at line boundaries, and an open
    blockquote is closed/reopened across such a seam.
    """
    raw_lines: list[str] = []
    for line in (text or "").split("\n"):
        raw_lines.extend(_wrap_long_line(line, limit - 64))
    detail = _detail_child_flags(raw_lines)
    headers = [_is_header(line) for line in raw_lines]

    lines: list[_Line] = []
    for idx, raw in enumerate(raw_lines):
        in_run = detail[idx]
        run_first = in_run and (idx == 0 or not detail[idx - 1])
        run_last = in_run and (idx == len(raw_lines) - 1
                               or not detail[idx + 1])
        lines.append(_Line(_render_line(raw, header=headers[idx]),
                           in_run, run_first, run_last, headers[idx]))

    # Sections: a header line starts a new section.
    sections: list[list[_Line]] = []
    current: list[_Line] = []
    for ln in lines:
        if ln.is_header and current:
            sections.append(current)
            current = []
        current.append(ln)
    if current:
        sections.append(current)

    def packed_len(items: list[_Line]) -> int:
        """Length of items packed into a FRESH part (quote runs get
        their open/close tags once per run they span here)."""
        total = 0
        for k, ln in enumerate(items):
            total += len(ln.html) + (1 if k else 0)
            if ln.run_first:
                total += len(_QUOTE_OPEN)
            if ln.run_last:
                total += len(_QUOTE_CLOSE)
        return total

    parts: list[str] = []
    cur: list[str] = []
    cur_len = 0
    quote_open = False        # a blockquote is open in the current part

    def flush() -> None:
        nonlocal cur, cur_len, quote_open
        if cur:
            body = "\n".join(cur)
            if quote_open:
                body += _QUOTE_CLOSE
            if body.strip():
                parts.append(body)
        cur, cur_len, quote_open = [], 0, False

    def add(ln: _Line) -> None:
        nonlocal cur_len, quote_open
        rendered = ln.html
        if ln.in_run and not quote_open:
            rendered = _QUOTE_OPEN + rendered
        closes = ln.run_last
        need = len(rendered) + (1 if cur else 0)
        if ln.in_run and not closes:
            need += len(_QUOTE_CLOSE)     # reserve the seam close
        if closes:
            need += len(_QUOTE_CLOSE)     # the run's own close tag
        if cur and cur_len + need > limit:
            flush()
            if ln.in_run:
                rendered = _QUOTE_OPEN + ln.html
        if closes:
            rendered += _QUOTE_CLOSE
        cur.append(rendered)
        cur_len += len(rendered) + (1 if len(cur) > 1 else 0)
        quote_open = ln.in_run and not closes

    for sec in sections:
        if packed_len(sec) <= limit:
            need = packed_len(sec) + (1 if cur else 0)
            if quote_open:
                need += len(_QUOTE_CLOSE)
            if cur and cur_len + need > limit:
                flush()
            for ln in sec:
                add(ln)
        else:
            # Oversized section: line packing.  A header that would
            # end up as a part's last line moves with its content.
            for idx, ln in enumerate(sec):
                if idx == 0 and ln.is_header and cur and len(sec) > 1:
                    room = limit - (cur_len + len(ln.html) + 1)
                    if room < packed_len(sec[1:2]) + 1:
                        flush()
                add(ln)
    flush()
    return parts or [""]


def split_text(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Split PLAIN text into ``<= limit`` chunks on line boundaries.

    The plain-text sibling of :func:`render_messages`: surfaces
    that must deliver text verbatim (no HTML rendering — the
    backtest export summary above all) split here, so this module
    is the ONE splitting authority for Telegram-bound output.
    Lossless: ``"".join(split_text(text, limit)) == text``.  A
    single line longer than ``limit`` is hard-split into
    limit-sized pieces — otherwise one long line would produce one
    over-limit chunk.
    """
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


def render_html(text: str) -> str:
    """Render to ONE HTML message; raises ValueError when the
    rendered text cannot fit one message (callers that need the
    multi-part guarantee use :func:`render_messages`)."""
    parts = render_messages(text)
    if len(parts) != 1:
        raise ValueError(
            f"message renders to {len(parts)} parts, not one")
    return parts[0]


def caption_parts(text: str) -> tuple[str, str | None]:
    """Split a photo/document caption at the caption limit.

    Returns ``(caption_html, overflow_plain)``: the caption rendered
    as HTML when it fits Telegram's caption limit, else ``""`` plus
    the ORIGINAL plain text, to be delivered as follow-up text
    messages — captions are never sliced anymore.
    """
    if not text:
        return "", None
    rendered = render_messages(text, limit=_CAPTION_LIMIT)
    if len(rendered) == 1 and len(rendered[0]) <= _CAPTION_LIMIT:
        return rendered[0], None
    return "", text
