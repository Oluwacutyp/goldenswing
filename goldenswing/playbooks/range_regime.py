"""Range-regime playbooks (D1 flat + defined range).

Traders don't wait out a range — they read every candle in it.
Three of these playbooks trade the RANGE regime only
(``regime_scope = "range"``); the confluence gate enforces regime
separation, so a range playbook can never fire in a TREND verdict
and a trend playbook can never fire in a RANGE verdict.  The
fourth — compression breakout — is the boundary playbook
(``regime_scope = "both"``): it trades the breakout moment itself,
and the signal path only runs it after the boundary machine
confirmed the breakout (2 H1 closes beyond + M15 BOS), when the
verdict is BREAKOUT_TREND.

Shared anatomy (all four):

- The range comes from the incremental ``RangeDetector`` fed with
  the terminal's native H4 frame (threaded in by the runner via
  ``set_h4_frame``) + the H1 frame. No H4 -> no range -> no
  signal. There is NO H4-resample fallback: resampled H4
  fabricates TF structure.
- Every signal points at the range structure: edge-sweep / QM /
  rejection entries stop beyond the edge extreme (+ buffer) and
  target the range mid (or the opposite edge for breakouts, the
  measured move).  Stops and targets are prices, not wishes.
- Scores are the playbook's own quality score (60-78).  The
  runner's ``min_score`` floor (60), confluence, manipulation
  filter, cost gate, margin/shield still apply downstream — nothing
  here loosens a gate.
- ``live_signal`` is causal: only bars ``<= t`` inform a signal at
  ``t``.  ``candidates`` scans the same detectors bar by bar with a
  range recomputed every 24 bars (causal cache) and pre-simulates
  with the standard harness.

Patterns (mutually exclusive on the same bar by construction):

- ``range_sweep_reclaim``: sweep bar wicks BEYOND the edge level and
  closes back INSIDE; confirm bar closes further inside.  Trade
  back to mid.
- ``range_qm``: QM head beyond the edge, pullback inside, failed
  retest (lower high / higher low), close through the neckline.
  Trade to mid.
- ``range_mean_reversion``: rejection wick INSIDE the edge band
  (touched the band, never exceeded the level) + close back
  inside.  Trade to mid.  (A wick beyond the level is a sweep,
  not a mean-reversion touch — the two never double-fire.)
- ``range_compression_breakout``: range compressed + entry bar
  CLOSES beyond the edge with displacement.  Trade outward, target
  the measured move (edge +/- width), stop just inside the edge.
  Breakouts leave the zone, so the live tick clock arms them on
  the breakout close (see ``MT5Runner._evaluate_live_tick``) —
  the touch-arm path cannot carry them, by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.harness import Outcome, simulate_signal_v2
from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..live.range import RangeDef, RangeDetector, detect_range
from .base import Candidate, EngineContext, Playbook

STOP_BUFFER_ATR = 0.25
MIN_SWEEP_ATR = 0.15    # sweep must clear the edge by this much (noise)
RANGE_RECOMPUTE_BARS = 24  # candidates(): range cache refresh cadence


def _arrays(df):
    return (
        df["open"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        df["close"].to_numpy(dtype=float),
    )


def _wick_ratio(o: float, h: float, l: float, c: float,
                direction: int) -> float:
    """Rejection-wick size / body size (inf when body is ~0)."""
    body = abs(c - o)
    wick = (h - max(o, c)) if direction < 0 else (min(o, c) - l)
    if body < 1e-9:
        return float("inf") if wick > 0 else 0.0
    return wick / body


def _to_signal(*, index, timestamp, direction, score, kind, factors,
               entry_ref, invalidation, targets, atr,
               context: dict) -> GoldenSignal:
    sig = GoldenSignal(
        index=index, timestamp=timestamp, direction=direction,
        score=float(score), kind=kind, factors=list(factors),
        entry_zone=(float(entry_ref), float(entry_ref)),
        entry_ref=float(entry_ref), invalidation=float(invalidation),
        targets=[float(t) for t in targets], atr=float(atr))
    sig.live_context = dict(context)
    return sig


class _RangeBase(Playbook):
    """Shared range detection + scan plumbing for range playbooks."""

    regime_scope = "range"
    session_scope = "any"
    hunting_ground = HuntingGround(excluded=set(), bonuses={})

    def __init__(self) -> None:
        self._processed = -1  # last bar run through live_signal
        # Native H4 frame, threaded in by the runner
        # (``MT5Runner`` sets it from ``TimeframeStack`` before
        # calling ``live_signal``). No H4 -> no range -> no signal:
        # there is NO H4-resample fallback anywhere on this path.
        self._h4 = None
        self._detector = RangeDetector()

    def set_h4_frame(self, h4) -> None:
        """Thread the terminal's native H4 frame into the playbook.

        Cheap (stores a reference); the incremental detector does
        the work in ``_range_upto``. Called by the runner on every
        signal evaluation — the stack frame grows as bars close.
        """
        self._h4 = h4

    # -- detection hook: subclasses implement _detect_at(j) -> signal
    def _detect_at(self, o, h, l, c, j: int, range_def: RangeDef,
                   ts) -> GoldenSignal | None:
        raise NotImplementedError

    def _range_upto(self, df, j: int) -> RangeDef | None:
        """Range as of H1 bar ``j``, via the incremental detector.

        H4 is causally sliced: only H4 bars CLOSED by H1 bar ``j``
        (open-stamped: index + 240min <= H1 close) feed the
        detector — no lookahead. The detector only processes
        newly appended bars, so a full-history fast-forward is
        O(n) total, not O(n^2) like the old per-bar full-prefix
        ``detect_range`` calls.
        """
        if self._h4 is None:
            return None
        h1_close = df.index[j] + pd.Timedelta(minutes=60)
        h4_closed = self._h4[self._h4.index
                              <= h1_close - pd.Timedelta(minutes=240)]
        self._detector.update(h4_closed, df.iloc[:j + 1])
        return self._detector.detect()

    def _scan(self, df, start: int, end: int):
        """Yield signals for bars ``start..end`` (inclusive).

        One causal code path for ``live_signal`` and
        ``candidates``: the range is recomputed from bars ``<= j``
        only, cached and refreshed every ``RANGE_RECOMPUTE_BARS``
        bars (edges move slowly; the cache is never fed a bar
        past ``j``).  Yields ``(j, signal, range_def)``.
        """
        o, h, l, c = _arrays(df)
        cached: RangeDef | None = None
        cached_at = -1
        for j in range(start, end + 1):
            # The LAST bar always gets a fresh range: compression
            # is fast-moving state, and a 24-bar-stale cache could
            # miss a compression that formed in the last few bars
            # (exactly when the breakout playbook needs it).
            if cached is None or j - cached_at >= RANGE_RECOMPUTE_BARS \
                    or j == end:
                cached = self._range_upto(df, j)
                cached_at = j
            if cached is None:
                continue
            sig = self._detect_at(o, h, l, c, j, cached, df.index[j])
            if sig is not None:
                yield j, sig, cached

    def live_signal(self, states: list, t: int, df=None) -> list:
        if df is None or t < 0 or t >= len(df):
            return []
        out = [sig for _j, sig, _r in
               self._scan(df, self._processed + 1, t)]
        self._processed = t
        return out

    def candidates(self, ctx: EngineContext) -> list[Candidate]:
        df = ctx.df
        if df is None or len(df) < 40:
            return []
        out: list[Candidate] = []
        for j, sig, cached in self._scan(df, 0, len(df) - 1):
            oc = simulate_signal_v2(df, sig)
            if oc is None:
                continue
            side = "long" if sig.direction > 0 else "short"
            out.append(Candidate(
                playbook=self.name, outcome=oc,
                base_score=float(sig.score),
                thesis=(f"{self.name} {side} @ {sig.entry_ref:.1f} "
                        f"(bar {j}): {'; '.join(sig.factors)}"),
                context={"range": cached.to_dict(),
                         **dict(sig.live_context or {})}))
        return out

    def export_state(self) -> dict:
        return {"processed": self._processed}

    def import_state(self, state: dict) -> None:
        self._processed = int((state or {}).get("processed", -1))


class RangeSweepReclaimPlaybook(_RangeBase):
    name = "range_sweep_reclaim"
    # Wick alone never fires: the boundary machine must have
    # CONFIRMED the sweep on this edge (wick + close inside + M15
    # BOS reclaim + FVG), enforced by the confluence gate against
    # the verdict's sweep_confirmed_edges.
    requires_sweep_confirmation = True
    description = (
        "Range edge sweep (wick beyond the edge, close back inside) "
        "+ reclaim confirm (close further inside). Trade back to mid; "
        "stop beyond the sweep extreme."
    )
    live = True
    live_threshold = 60.0
    context_description = (
        "RANGE regime only (D1 flat + defined range). Sweep bar must "
        "wick beyond the edge level by >= 0.15 ATR and close back "
        "inside; confirm bar closes further inside. Stop beyond the "
        "sweep extreme + 0.25 ATR; target the range mid."
    )

    def _detect_at(self, o, h, l, c, j, range_def, ts):
        if j < 2:
            return None
        atr = range_def.atr
        s = j - 1  # sweep bar
        # Short: sweep above the high edge.
        if h[s] > range_def.high + MIN_SWEEP_ATR * atr \
                and c[s] < range_def.high \
                and c[j] < c[s] and c[j] < range_def.high:
            depth = h[s] - range_def.high
            score = 60.0 + min(10.0, 20.0 * depth / atr)
            if abs(c[j] - o[j]) > 0.5 * atr:
                score += 5.0
            return _to_signal(
                index=j, timestamp=ts, direction=-1,
                score=min(score, 78.0), kind=self.name,
                factors=["range_sweep_reclaim", "sweep_high",
                         "reclaim"],
                entry_ref=c[j],
                invalidation=h[s] + STOP_BUFFER_ATR * atr,
                targets=[range_def.mid], atr=atr,
                context={"range": range_def.to_dict(),
                         "sweep_bar": s, "sweep_extreme": float(h[s]),
                         "edge": range_def.high})
        # Long: sweep below the low edge.
        if l[s] < range_def.low - MIN_SWEEP_ATR * atr \
                and c[s] > range_def.low \
                and c[j] > c[s] and c[j] > range_def.low:
            depth = range_def.low - l[s]
            score = 60.0 + min(10.0, 20.0 * depth / atr)
            if abs(c[j] - o[j]) > 0.5 * atr:
                score += 5.0
            return _to_signal(
                index=j, timestamp=ts, direction=+1,
                score=min(score, 78.0), kind=self.name,
                factors=["range_sweep_reclaim", "sweep_low",
                         "reclaim"],
                entry_ref=c[j],
                invalidation=l[s] - STOP_BUFFER_ATR * atr,
                targets=[range_def.mid], atr=atr,
                context={"range": range_def.to_dict(),
                         "sweep_bar": s, "sweep_extreme": float(l[s]),
                         "edge": range_def.low})
        return None


class RangeQMPlaybook(_RangeBase):
    name = "range_qm"
    description = (
        "Quasimodo at a range edge: head beyond the edge, pullback "
        "inside, failed retest, close through the neckline. Trade "
        "to mid; stop beyond the head."
    )
    live = True
    live_threshold = 60.0
    context_description = (
        "RANGE regime only (D1 flat + defined range). QM head must "
        "print beyond the edge level; the failed retest must not "
        "exceed the head; the trigger is a close through the "
        "neckline. Stop beyond the head + 0.25 ATR; target mid."
    )

    def _detect_at(self, o, h, l, c, j, range_def, ts):
        atr = range_def.atr
        # Search the last 20 bars for head -> pullback -> failed
        # retest, with the neckline break closing at j.
        lo = max(0, j - 20)
        # Short QM at the high edge.
        for p0 in range(lo, j - 2):
            head = h[p0]
            if head <= range_def.high:
                continue  # head must exceed the edge
            if p0 + 2 >= j:
                continue  # need room for pullback + retest bars
            # Pullback low after the head.
            p1 = int(np.argmin(l[p0 + 1:j])) + p0 + 1
            if p1 + 1 >= j:
                continue  # no room for a retest bar
            neckline = l[p1]
            if neckline >= range_def.high:
                continue  # pullback must come back inside
            # Failed retest: highest high after the pullback.
            p2 = int(np.argmax(h[p1 + 1:j])) + p1 + 1
            if h[p2] >= head:
                continue  # retest exceeded the head: not QM
            if h[p2] < range_def.high - 0.5 * atr:
                continue  # never tested the edge zone
            if c[j] < neckline:
                score = 62.0 + min(
                    10.0, 10.0 * (head - range_def.high) / atr)
                return _to_signal(
                    index=j, timestamp=ts, direction=-1,
                    score=min(score, 76.0), kind=self.name,
                    factors=["range_qm", "qm_high", "neckline_break"],
                    entry_ref=c[j],
                    invalidation=head + STOP_BUFFER_ATR * atr,
                    targets=[range_def.mid], atr=atr,
                    context={"range": range_def.to_dict(),
                             "head": float(head),
                             "neckline": float(neckline),
                             "edge": range_def.high})
        # Long QM at the low edge (mirror).
        for p0 in range(lo, j - 2):
            head = l[p0]
            if head >= range_def.low:
                continue
            if p0 + 2 >= j:
                continue
            p1 = int(np.argmax(h[p0 + 1:j])) + p0 + 1
            if p1 + 1 >= j:
                continue
            neckline = h[p1]
            if neckline <= range_def.low:
                continue
            p2 = int(np.argmin(l[p1 + 1:j])) + p1 + 1
            if l[p2] <= head:
                continue
            if l[p2] > range_def.low + 0.5 * atr:
                continue
            if c[j] > neckline:
                score = 62.0 + min(
                    10.0, 10.0 * (range_def.low - head) / atr)
                return _to_signal(
                    index=j, timestamp=ts, direction=+1,
                    score=min(score, 76.0), kind=self.name,
                    factors=["range_qm", "qm_low", "neckline_break"],
                    entry_ref=c[j],
                    invalidation=head - STOP_BUFFER_ATR * atr,
                    targets=[range_def.mid], atr=atr,
                    context={"range": range_def.to_dict(),
                             "head": float(head),
                             "neckline": float(neckline),
                             "edge": range_def.low})
        return None


class RangeMeanReversionPlaybook(_RangeBase):
    name = "range_mean_reversion"
    description = (
        "Rejection wick inside the edge band (touched the band, "
        "never exceeded the level) + close back inside. Trade to "
        "mid; stop beyond the wick extreme."
    )
    live = True
    live_threshold = 60.0
    context_description = (
        "RANGE regime only (D1 flat + defined range). The rejection "
        "bar's extreme must stay inside the edge band (level not "
        "exceeded — beyond the level is a sweep, handled by "
        "range_sweep_reclaim) with wick >= 2x body. Stop beyond "
        "the wick extreme + 0.25 ATR; target the mid."
    )

    def _detect_at(self, o, h, l, c, j, range_def, ts):
        atr = range_def.atr
        # Short: rejection at the high edge band, bar j.
        if range_def.high - 0.5 * atr <= h[j] <= range_def.high \
                and c[j] < range_def.high \
                and _wick_ratio(o[j], h[j], l[j], c[j], -1) >= 2.0:
            wick = h[j] - max(o[j], c[j])
            score = 60.0 + min(8.0, 8.0 * wick / atr)
            return _to_signal(
                index=j, timestamp=ts, direction=-1,
                score=min(score, 74.0), kind=self.name,
                factors=["range_mean_reversion", "rejection_high"],
                entry_ref=c[j],
                invalidation=h[j] + STOP_BUFFER_ATR * atr,
                targets=[range_def.mid], atr=atr,
                context={"range": range_def.to_dict(),
                         "edge": range_def.high})
        # Long: rejection at the low edge band, bar j.
        if range_def.low <= l[j] <= range_def.low + 0.5 * atr \
                and c[j] > range_def.low \
                and _wick_ratio(o[j], h[j], l[j], c[j], +1) >= 2.0:
            wick = min(o[j], c[j]) - l[j]
            score = 60.0 + min(8.0, 8.0 * wick / atr)
            return _to_signal(
                index=j, timestamp=ts, direction=+1,
                score=min(score, 74.0), kind=self.name,
                factors=["range_mean_reversion", "rejection_low"],
                entry_ref=c[j],
                invalidation=l[j] - STOP_BUFFER_ATR * atr,
                targets=[range_def.mid], atr=atr,
                context={"range": range_def.to_dict(),
                         "edge": range_def.low})
        return None


class RangeCompressionBreakoutPlaybook(_RangeBase):
    name = "range_compression_breakout"
    # BOUNDARY playbook: it trades the breakout moment itself, so
    # its scope is "both" — it passes the regime gate in a RANGE
    # verdict AND in a BREAKOUT_TREND verdict.  This is NOT a
    # mixing loophole: the signal path only ever runs for it after
    # the boundary machine confirmed the breakout (2 H1 closes
    # beyond + M15 BOS), at which point the verdict is
    # BREAKOUT_TREND.  The other three range playbooks stay
    # range-scoped and never fire in a trend verdict.
    regime_scope = "both"
    description = (
        "Compressed range + close beyond the edge with displacement. "
        "Trade outward; target the measured move (edge +/- width); "
        "stop just inside the edge."
    )
    live = True
    live_threshold = 60.0
    context_description = (
        "RANGE regime only (D1 flat + defined range, compressed). "
        "The breakout bar must close beyond the edge by >= 0.25 ATR "
        "with the prior bar inside, and print a fresh 10-bar "
        "extreme (displacement). Stop just inside the edge; target "
        "edge +/- width (measured move)."
    )

    def _detect_at(self, o, h, l, c, j, range_def, ts):
        if not range_def.compressed:
            return None
        if j < 11:
            return None
        atr = range_def.atr
        # Long breakout above the high edge.
        if c[j] > range_def.high + STOP_BUFFER_ATR * atr \
                and c[j - 1] <= range_def.high \
                and c[j] > float(np.max(h[max(0, j - 10):j])):
            return _to_signal(
                index=j, timestamp=ts, direction=+1,
                score=64.0, kind=self.name,
                factors=["range_compression_breakout",
                         "breakout_high", "displacement"],
                entry_ref=c[j],
                invalidation=range_def.high - STOP_BUFFER_ATR * atr,
                targets=[range_def.high + range_def.width],
                atr=atr,
                context={"range": range_def.to_dict(),
                         "edge": range_def.high})
        # Short breakout below the low edge.
        if c[j] < range_def.low - STOP_BUFFER_ATR * atr \
                and c[j - 1] >= range_def.low \
                and c[j] < float(np.min(l[max(0, j - 10):j])):
            return _to_signal(
                index=j, timestamp=ts, direction=-1,
                score=64.0, kind=self.name,
                factors=["range_compression_breakout",
                         "breakout_low", "displacement"],
                entry_ref=c[j],
                invalidation=range_def.low + STOP_BUFFER_ATR * atr,
                targets=[range_def.low - range_def.width],
                atr=atr,
                context={"range": range_def.to_dict(),
                         "edge": range_def.low})
        return None


RANGE_PLAYBOOKS = (
    RangeSweepReclaimPlaybook,
    RangeQMPlaybook,
    RangeMeanReversionPlaybook,
    RangeCompressionBreakoutPlaybook,
)
