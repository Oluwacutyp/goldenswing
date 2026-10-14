"""TrendlineRetestPlaybook (Phase 9; PAPER-LIVE).

Claim #31 (Element Of Trading Vol V, user-mined PDFs): break of a
respected trendline -> price spends 2 bars away -> first return touch
within 12 bars -> enter in the breakout direction, structural stop
beyond the retest extreme, Phase-2 exits.  +15.40%, PF 1.43, 61 trades,
-0.01 daily-R correlation with the anchor book (experiments/
TRENDLINE_CLAIMS.md).

Iron rule: the retest scanner and Phase-2 exits are kept exactly as
tested.  The JeaFx zone-retest kill was deliberately NOT inherited —
zones and trendlines are different mechanics.  Paper-live in the paper
loop (live_signal parity-proven); MT5-live only after OOS promotes it
(see CONFLUENCE.md).
"""
from __future__ import annotations

from ..backtest.harness import simulate_signal_v2
from ..contextmap import HuntingGround
from ..experiments.trendline_claims import (
    RETEST_REARM_BARS,
    RETEST_WINDOW,
    TL_TOL_ATR,
    _make_signal,
    _passes_gates,
    build_lines,
    scan_retests,
)
from ..features.config import FeatureConfig
from ..features.trendlines import _Builder
from ..golden.detector import DetectorConfig, GoldenSwingDetector
from .base import Candidate, EngineContext, Playbook

#: Base score is a documented placeholder, NOT a measured score — the
#: sleeve was validated without scoring.  OOS promotion must calibrate
#: it (see CONFLUENCE.md); until then it only ranks sleeve-vs-sleeve.
SLEEVE_BASE_SCORE = 60.0

#: The non-score gates the experiment applied (via the gates-only
#: detector): hard daily-bias veto, room to the first opposing pool,
#: stop-distance band.  Re-verified here as the tripwire — same
#: predicate the scanner used, so no legitimate candidate can fail.
MIN_STOP_ATR, MAX_STOP_ATR = 0.25, 3.5

VALIDATED_INSTRUMENTS = ("GC=F", "XAUUSD")


class TrendlineRetestPlaybook(Playbook):
    name = "trendline_retest"
    description = (
        "Trendline breakout-retest: break -> 2 bars away -> return touch "
        "within 12 bars, structural stop, Phase-2 exits"
    )
    live = True   # paper-live: real live_signal, parity-proven
    live_threshold = 0.0  # every retest is graded
    context_description = (
        "Break of a respected trendline (≥1 touch) → price spends 2 bars "
        "away → first return touch within 12 bars (wick reaches the line, "
        "close holds the breakout side) → structural stop, Phase-2 exits "
        "(experiments/TRENDLINE_CLAIMS.md). Detector non-score gates apply "
        "(hard daily veto, room-to-pool, stop band). Validated on XAUUSD "
        "H1; H1 soft (+1.2%) vs H2 strong (+14.0%) — regime sensitivity "
        "noted, not a gate."
    )
    #: Session-agnostic: 61 trades are too thin for session verdicts and no
    #: cell was decisively bad.
    session_scope = "any"
    #: Hunting ground from the measured context map (61 trades): no
    #: exclusions, no proven cells — fires everywhere, no map bonuses.
    #: The map is descriptive until OOS thickens it.
    hunting_ground = HuntingGround(excluded=set(), bonuses={})

    def __init__(self) -> None:
        # The gates-only detector: identical non-score gates to the
        # experiment (daily veto hard, room-to-pool, stop-distance band).
        self._gates = GoldenSwingDetector(DetectorConfig(
            require_sweep_confirmation=True, daily_veto=True))
        self._cfg = FeatureConfig()
        # Live incremental state: the streaming line builder plus the
        # break->retest pending machine from scan_retests.
        self._builder: _Builder | None = None
        self._built_bars = 0     # bars the builder's arrays cover
        self._processed = -1     # last bar run through the machine
        self._pending: list[dict] = []

    # ------------------------------------------------------- live signal
    def _sync_builder(self, df) -> None:
        n = len(df)
        if self._builder is None:
            self._builder = _Builder(df, self._cfg)
            self._built_bars = n
        elif n > self._built_bars:
            self._builder.extend(df)
            self._built_bars = n

    def _process_bar(self, states, t: int, collect: bool) -> list:
        """One bar of the scan_retests state machine.

        Mirrors the batch loop body exactly (same ageing, same break
        ingestion, same arm/retest logic, same gates).  One deliberate
        divergence: the batch ``t + 1 < n`` guard is dropped — it only
        exists because the outcome simulator needs a forward bar, while
        a live loop trades the latest closed bar.  On interior bars the
        behaviour is identical (the parity test proves it).
        """
        out = []
        s = states[t]
        atr = s.atr
        # age out stale setups — same predicate as scan_retests
        self._pending = [b for b in self._pending
                         if t - b["break_bar"] <= RETEST_WINDOW
                         and not b["traded"]]
        ls = self._builder.update(t, s)
        for ev in ls.events:
            if ev.kind != "break" or ev.touches < 1:
                continue
            tl = None
            for cand in ls.active:
                if cand.direction == ev.direction and abs(
                        cand.anchor1_price
                        + cand.slope * (t - cand.anchor1_index)
                        - ev.line_price) < 1e-6:
                    tl = cand
                    break
            if tl is None:
                continue
            self._pending.append({
                "slope": tl.slope, "a1i": tl.anchor1_index,
                "a1p": tl.anchor1_price,
                "d_trade": -tl.direction,   # breakout direction
                "break_bar": t, "traded": False, "away": 0,
                "armed": False,
            })
        if atr is None or atr <= 0:
            return out
        tol = TL_TOL_ATR * atr
        for b in self._pending:
            if b["traded"]:
                continue
            lp = b["a1p"] + b["slope"] * (t - b["a1i"])
            d = b["d_trade"]
            if d == -1:  # broken support -> resistance; short the retest
                near = self._builder.high[t] >= lp - tol
                away = self._builder.high[t] < lp - tol
                retest = near and s.close < lp - tol
                extreme = self._builder.high[t]
            else:        # broken resistance -> support; long the retest
                near = self._builder.low[t] <= lp + tol
                away = self._builder.low[t] > lp + tol
                retest = near and s.close > lp + tol
                extreme = self._builder.low[t]
            if not b["armed"]:
                if away:
                    b["away"] += 1
                    if b["away"] >= RETEST_REARM_BARS:
                        b["armed"] = True
                elif near:
                    b["away"] = 0  # hovering: re-arm clock resets
                continue
            if retest:
                if collect and _passes_gates(self._gates, s, d, extreme):
                    out.append(_make_signal(s, d, extreme,
                                            "trendline_retest"))
                b["traded"] = True  # one trade per break, gated or not
        return out

    def live_signal(self, states: list, t: int, df=None) -> list:
        """Incremental breakout-retest check at closed bar ``t``.

        The streaming line builder and the pending-break machine are
        advanced through any missed bars (signals for past bars are
        discarded — the loop only trades the latest close); the parity
        test drives every bar and reproduces the batch signals.
        """
        if df is None or t < 0 or t >= len(states):
            return []
        self._sync_builder(df)
        if t >= self._built_bars:
            return []
        # Never rewind the cursor (see fvg_limit: rebase fresh loop
        # re-calls the tail; rewinding re-emits duplicates).
        if t <= self._processed:
            return []
        out: list = []
        for t2 in range(self._processed + 1, t + 1):
            out.extend(self._process_bar(states, t2, collect=(t2 == t)))
        self._processed = t
        return out

    def export_state(self) -> dict:
        return {"processed": self._processed, "pending": self._pending}

    def import_state(self, state: dict) -> None:
        # The line builder itself is rebuilt by replaying history (the
        # paper loop fast-forwards on start); only the pending-break
        # machine is restored here.
        self._processed = int(state.get("processed", -1))
        self._pending = [dict(b) for b in state.get("pending", [])]

    def candidates(self, ctx: EngineContext) -> list[Candidate]:
        line_states = build_lines(ctx.df, ctx.states, self._cfg, "shadow")
        signals = scan_retests(ctx.df, ctx.states, line_states, self._gates)
        out: list[Candidate] = []
        for sig in signals:
            oc = simulate_signal_v2(ctx.df, sig)
            if oc is None:
                continue
            side = "long" if sig.direction == 1 else "short"
            thesis = (
                f"trendline_retest {side} (bar {sig.index}): breakout-retest "
                f"of a respected trendline, structural stop "
                f"{sig.invalidation:.1f} "
                f"({abs(sig.entry_ref - sig.invalidation) / sig.atr:.2f} "
                f"ATR), Phase-2 exits (partial at first pool, trail "
                f"after +1.5R)"
            )
            out.append(Candidate(
                playbook=self.name, outcome=oc,
                base_score=SLEEVE_BASE_SCORE, thesis=thesis,
                context={"kind": sig.kind, "atr": sig.atr}))
        return out

    def check_live_signal(self, signal, ctx: EngineContext,
                          context: dict | None = None) -> tuple[bool, str]:
        # Re-verification of the validated context: the scanner's
        # non-score gates (hard daily veto, stop-distance band).  The
        # predicates are identical to _passes_gates, so no legitimate
        # candidate can fail — a failure is a scanner regression.
        if ctx.instrument not in VALIDATED_INSTRUMENTS:
            return False, (
                f"validated universe is XAUUSD H1, got {ctx.instrument!r}")
        t = signal.index
        if t >= len(ctx.states):
            return False, "signal bar beyond available chart state"
        s = ctx.states[t]
        if s.daily_bias != 0 and signal.direction == -s.daily_bias:
            return False, (
                f"against completed daily bias ({s.daily_bias:+d}) — the "
                f"hard veto was bypassed")
        atr = signal.atr
        if atr and atr > 0:
            stop_atr = abs(signal.entry_ref - signal.invalidation) / atr
            if not (MIN_STOP_ATR <= stop_atr <= MAX_STOP_ATR):
                return False, (
                    f"stop {stop_atr:.2f} ATR outside the validated "
                    f"{MIN_STOP_ATR}–{MAX_STOP_ATR} band")
        return True, ""
