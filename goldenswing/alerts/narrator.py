"""Signal narrator — the alert text tells the story, not just the factors.

The caption keeps its ``Backing:`` bullet list (the evidence).  The
narrator is the lead paragraph above it: 2-4 sentences of plain trader
language, in event order, saying what actually happened to produce
this signal.

Rules:
- every claim comes from the parsed factors / alert fields — no
  invented numbers, no adjectives the score doesn't support;
- unknown or missing fields are SKIPPED, never hallucinated;
- ~60 words, readable on a phone.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Factor-string parsing.
#
# The detector and playbooks emit factors as free-text strings (see
# golden/detector.py).  The narrator parses them with regexes rather than
# reaching into detector internals, so it works for every playbook whose
# signal carries a ``factors`` list.  Anything unparseable is ignored.
# ---------------------------------------------------------------------------

_SWEEP = re.compile(
    r"liquidity sweep of (sell-side|buy-side) pool @ ([\d.,]+), (\d+) bars? ago")
_CONFIRM_SWEEP = re.compile(r"(CHOCH|BOS) confirms the sweep reversal")
_FLIPPED = re.compile(r"trend already flipped with the sweep")
_AWAITING = re.compile(r"sweep reversal against structure \(awaiting CHoCH\)")
_CONT = re.compile(r"trend-aligned continuation \((up|down)\)")
_FRESH = re.compile(r"fresh (CHOCH|BOS) (\d+) bars? ago")
_MOM = re.compile(r"momentum aligned \(([+-][\d.]+) ATR\)")
_DISP = re.compile(r"(strong )?displacement ([\d.]+)x ATR")
_LEVEL = re.compile(r"price reacting at an (order block|fair value gap)")
_ROOM = re.compile(r"room to next liquidity pool: ([\d.]+)x ATR")
_COMPRESSED = re.compile(r"volatility compressed \(ATR p(\d+)\)")
_EXPANDED = re.compile(r"volatility expanded \(ATR p(\d+)\)")
_EXTENDED = re.compile(r"price is extended at the range extreme")
_PENALTY = re.compile(r"(buying in premium|selling in discount) — penalised")
_ZONE = re.compile(r"(long from discount|short from premium)")
_KZ = re.compile(r"(\w+) killzone")
_OVERLAP = re.compile(r"London/NY overlap")
_SQUEEZE = re.compile(r"Bollinger squeeze \(bandwidth p(\d+)\)")
_WALK = re.compile(r"walking the band with the move")
_WALK_OPP = re.compile(r"price walking the OPPOSITE band")
_REJ = re.compile(r"band-rejection wick supports the direction")
_REJ_BAD = re.compile(r"band rejected the move's direction")
_OTE = re.compile(r"entry inside the OTE zone")
_FIB = re.compile(r"price at Fib ([\d.]+)")
_FIB_CAP = re.compile(r"Fib ([\d.]+) caps the move just ahead")
_CLOUD_ABOVE = re.compile(r"above the Ichimoku cloud")
_CLOUD_BELOW = re.compile(r"below the Ichimoku cloud")
_CLOUD_IN = re.compile(r"inside the cloud — no-man's land")
_TK = re.compile(r"fresh TK cross in the direction")
_TWIST = re.compile(r"cloud twist in the direction")


_SESSION_NAMES = {
    "london": "London", "newyork": "New York", "asia": "Asia",
    "overlap": "London/NY overlap",
}


def _cap(s: str) -> str:
    """Uppercase the first letter only — str.capitalize() would mangle
    ATR, TK, OTE and other acronyms mid-sentence."""
    return s[:1].upper() + s[1:] if s else s


def _bars_ago(n: int) -> str:
    if n <= 0:
        return "just now"
    if n == 1:
        return "one bar ago"
    return f"{n} bars ago"


def _fmt_px(x: float) -> str:
    return f"{x:,.1f}"


class _Facts:
    """Structured facts pulled out of the factor strings."""

    def __init__(self) -> None:
        self.sweep_side: str | None = None      # "sell-side" | "buy-side"
        self.sweep_level: float | None = None
        self.sweep_age: int | None = None
        self.confirm: str | None = None         # "CHOCH" | "BOS" (sweep confirm)
        self.flipped = False
        self.awaiting = False
        self.trend: str | None = None           # "up" | "down" (continuation)
        self.fresh_kind: str | None = None      # "CHOCH" | "BOS"
        self.fresh_age: int | None = None
        self.momentum: float | None = None
        self.disp: float | None = None
        self.disp_strong = False
        self.level: str | None = None           # "order block" | "fair value gap"
        self.room_atr: float | None = None
        self.compressed = False
        self.expanded = False
        self.extended = False
        self.penalty: str | None = None
        self.zone: str | None = None
        self.killzone: str | None = None
        self.overlap = False
        self.squeezed = False
        self.walk = False
        self.walk_opp = False
        self.rejection = False
        self.rejection_bad = False
        self.ote = False
        self.fib: float | None = None
        self.fib_cap: float | None = None
        self.cloud: str | None = None           # "above" | "below" | "inside"
        self.tk = False
        self.twist = False


def _parse(factors: list[str]) -> _Facts:
    f = _Facts()
    for s in factors or []:
        m = _SWEEP.search(s)
        if m:
            f.sweep_side = m.group(1)
            f.sweep_level = float(m.group(2).replace(",", ""))
            f.sweep_age = int(m.group(3))
            continue
        m = _CONFIRM_SWEEP.search(s)
        if m:
            f.confirm = m.group(1)
            continue
        if _FLIPPED.search(s):
            f.flipped = True
            continue
        if _AWAITING.search(s):
            f.awaiting = True
            continue
        m = _CONT.search(s)
        if m:
            f.trend = m.group(1)
            continue
        m = _FRESH.search(s)
        if m:
            f.fresh_kind, f.fresh_age = m.group(1), int(m.group(2))
            continue
        m = _MOM.search(s)
        if m:
            f.momentum = float(m.group(1))
            continue
        m = _DISP.search(s)
        if m:
            f.disp_strong = bool(m.group(1))
            f.disp = float(m.group(2))
            continue
        m = _LEVEL.search(s)
        if m:
            f.level = m.group(1)
            continue
        m = _ROOM.search(s)
        if m:
            f.room_atr = float(m.group(1))
            continue
        if _COMPRESSED.search(s):
            f.compressed = True
            continue
        if _EXPANDED.search(s):
            f.expanded = True
            continue
        if _EXTENDED.search(s):
            f.extended = True
            continue
        m = _PENALTY.search(s)
        if m:
            f.penalty = m.group(1)
            continue
        m = _ZONE.search(s)
        if m:
            f.zone = m.group(1)
            continue
        m = _KZ.search(s)
        if m:
            f.killzone = m.group(1)
            continue
        if _OVERLAP.search(s):
            f.overlap = True
            continue
        if _SQUEEZE.search(s):
            f.squeezed = True
            continue
        if _WALK_OPP.search(s):
            f.walk_opp = True
            continue
        if _WALK.search(s):
            f.walk = True
            continue
        if _REJ_BAD.search(s):
            f.rejection_bad = True
            continue
        if _REJ.search(s):
            f.rejection = True
            continue
        if _OTE.search(s):
            f.ote = True
            continue
        m = _FIB_CAP.search(s)
        if m:
            f.fib_cap = float(m.group(1))
            continue
        m = _FIB.search(s)
        if m:
            f.fib = float(m.group(1))
            continue
        if _CLOUD_IN.search(s):
            f.cloud = "inside"
            continue
        if _CLOUD_ABOVE.search(s):
            f.cloud = "above"
            continue
        if _CLOUD_BELOW.search(s):
            f.cloud = "below"
            continue
        if _TK.search(s):
            f.tk = True
            continue
        if _TWIST.search(s):
            f.twist = True
            continue
    return f


def _get(ctx, *names, default=None):
    """Duck-typed field access: Alert, GoldenSignal, or plain dict."""
    if isinstance(ctx, dict):
        for n in names:
            if n in ctx:
                return ctx[n]
        return default
    for n in names:
        v = getattr(ctx, n, None)
        if v is not None:
            return v
    return default


def narrate(signal_ctx) -> str:
    """Event-ordered narration of what happened to produce this signal.

    Accepts an Alert, a GoldenSignal, or a dict with ``factors`` /
    ``direction`` / ``targets`` / ``score`` keys.  Never raises on
    missing fields — it narrates what it can and skips the rest.
    """
    try:
        factors = _get(signal_ctx, "factors", default=[]) or []
        direction = _get(signal_ctx, "direction", default=1)
        targets = _get(signal_ctx, "targets", default=[]) or []
        score = _get(signal_ctx, "score", default=None)
        f = _parse(list(factors))
        long = direction >= 0
        in_control = "buyers" if long else "sellers"
        sentences: list[str] = []

        # 1 — the trigger story: sweep, then what structure did.
        if f.sweep_side is not None:
            stops = "sell stops" if f.sweep_side == "sell-side" else "buy stops"
            when = _bars_ago(f.sweep_age or 0)
            lvl = _fmt_px(f.sweep_level) if f.sweep_level is not None else "the pool"
            sentences.append(
                f"Swept the {f.sweep_side} pool at {lvl} {when}, "
                f"taking the {stops}.")
        if f.confirm:
            sentences.append(
                f"{f.confirm} confirms it — {in_control} in control.")
        elif f.fresh_kind:
            when = "just printed" if (f.fresh_age or 0) <= 0 \
                else _bars_ago(f.fresh_age or 0)
            sentences.append(
                f"Fresh {f.fresh_kind} {when} — {in_control} in control.")
        elif f.flipped:
            sentences.append("Trend already flipped with the sweep.")
        elif f.trend:
            sentences.append(f"Trend-aligned continuation — trend's {f.trend}.")
        elif f.awaiting:
            sentences.append(
                "Still waiting on the CHoCH — structure unconfirmed.")

        # 2 — the confirmation: displacement, momentum, level.
        s2: list[str] = []
        if f.disp is not None:
            adj = "Strong displacement" if f.disp_strong else "Displacement"
            s2.append(f"{adj.lower()} ran {f.disp:.1f}x ATR")
        if f.momentum is not None:
            s2.append(f"momentum aligned ({f.momentum:+.1f} ATR)")
        if f.level:
            s2.append(f"reacting at an {f.level}")
        if f.walk:
            s2.append("price walking the band with the move")
        if f.rejection:
            s2.append("band-rejection wick backs the direction")
        if f.ote:
            s2.append("entry sits inside the OTE zone")
        if f.tk:
            s2.append("fresh TK cross in the direction")
        if s2:
            sentences.append(_cap(", ".join(s2)) + ".")

        # 3 — the context: session, zone, regime.
        s3: list[str] = []
        if f.killzone:
            kz = _SESSION_NAMES.get(f.killzone.lower(), f.killzone)
            s3.append(f"in the {kz} killzone")
        elif f.overlap:
            s3.append("London/NY overlap — peak liquidity")
        if f.zone:
            s3.append(f.zone)
        if f.compressed:
            s3.append("volatility's coiled")
        elif f.expanded:
            s3.append("volatility's expanded")
        if f.squeezed:
            s3.append("Bollinger squeeze underneath")
        if f.cloud == "above" and long:
            s3.append("above the cloud")
        elif f.cloud == "below" and not long:
            s3.append("below the cloud")
        if f.twist:
            s3.append("cloud twist in the direction")
        if s3:
            sentences.append(_cap(", ".join(s3)) + ".")

        # 4 — the trade math: room and the first target.
        if f.room_atr is not None and targets:
            t1 = _fmt_px(targets[0])
            sentences.append(
                f"{f.room_atr:.1f}x ATR of room to T1 at {t1}.")

        # honest caveats, kept short — one at most
        caveat = None
        if f.penalty:
            caveat = f"Weak spot: {f.penalty}"
        elif f.extended:
            caveat = "Weak spot: price is extended at the range extreme"
        elif f.walk_opp:
            caveat = "Weak spot: price walking the opposite band"
        elif f.rejection_bad:
            caveat = "Weak spot: the band rejected the move's direction"
        elif f.cloud == "inside":
            caveat = "Weak spot: inside the cloud — no-man's land"
        if caveat:
            sentences.append(caveat + ".")

        text = " ".join(sentences).strip()
        if text:
            return text
    except Exception:
        pass
    # graceful floor: never crash the caption, never invent a story
    if score is not None:
        try:
            return f"Signal fired with score {float(score):.0f} — see the backing below."
        except (TypeError, ValueError):
            pass
    return "Signal fired — see the backing below."
