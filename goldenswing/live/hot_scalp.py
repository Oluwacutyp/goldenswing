"""Hot Scalping research sleeve: persistent structure, live tick reaction.

This first implementation is intentionally SHADOW-ONLY. It builds three
playbook families from closed market structure, persists their setups, watches
live ticks for a reaction at the level, scores independent evidence, and emits
an auditable candidate. It has no broker/order dependency by design.

Families:
  * pullback_continuation: aligned higher context + M5 structure break + fresh
    unfilled imbalance, entered only after a live tap/reclaim of its midpoint.
  * sweep_reclaim: a confirmed liquidity sweep plus structure shift, entered
    only after a later retest/reclaim of the swept pool.
  * breakout_retest: a confirmed structure break retested from the broken side.

Patterns are source facts, not votes by themselves. A structure break and the
imbalance created by the same impulse share one correlation group. Missing
inputs remain neutral. Score weights are research weights, not calibrated
probabilities, and never authorize an order.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any
import math
import uuid

import pandas as pd

from .evidence import (BASE_CATEGORY_WEIGHTS, EvidenceItem, score_evidence)
from .multitf import atr14, find_swings
from .patterns import (detect_sweep, liquidity_pools, premium_discount,
                       session_label, structure_event, trend_bias,
                       unfilled_fvgs, volume_confirm)
from .setup_store import SetupStore, SetupStoreError
from .stacking import StackingConfig, child_entry_id


CATEGORY_WEIGHTS = dict(BASE_CATEGORY_WEIGHTS)


@dataclass(frozen=True)
class HotScalpConfig:
    enabled: bool = True
    max_setup_age_minutes: float = 45.0
    max_active_setups: int = 40
    min_m5_bars: int = 35
    min_m15_bars: int = 35
    min_m1_bars: int = 20
    stop_buffer_atr: float = 0.15
    trigger_band_atr: float = 0.08
    max_spread_atr_m1: float = 0.20
    min_net_reward_risk: float = 1.20
    fresh_m1_seconds: float = 120.0
    fresh_m5_seconds: float = 600.0
    fresh_m15_seconds: float = 1200.0

    def __post_init__(self):
        if self.max_setup_age_minutes <= 0:
            raise ValueError("max_setup_age_minutes must be positive")
        if self.max_active_setups < 1:
            raise ValueError("max_active_setups must be positive")
        if self.stop_buffer_atr < 0 or self.trigger_band_atr <= 0:
            raise ValueError("ATR buffers are invalid")
        if self.max_spread_atr_m1 <= 0 or self.min_net_reward_risk <= 0:
            raise ValueError("cost and reward-risk limits must be positive")


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _close_ts(frame: pd.DataFrame, tf: str) -> pd.Timestamp | None:
    if frame is None or not len(frame):
        return None
    minutes = {"M1": 1, "M5": 5, "M15": 15}.get(tf)
    if minutes is None:
        return None
    return _utc(frame.index[-1]) + pd.Timedelta(minutes=minutes)


def _fresh(frame: pd.DataFrame, tf: str, now: pd.Timestamp,
           max_age_seconds: float) -> tuple[bool, float | None]:
    closed = _close_ts(frame, tf)
    if closed is None:
        return False, None
    age = (now - closed).total_seconds()
    return age >= -2.0 and age <= max_age_seconds, float(max(0.0, age))


def _stable_setup_id(symbol: str, strategy: str, direction: int,
                     formation_ts: Any, level: float,
                     stop: float) -> str:
    # Do NOT include scan time: two processes and repeated scans must produce
    # the same identity for the same market structure.
    material = "|".join((str(symbol).upper(), strategy, str(int(direction)),
                         _utc(formation_ts).isoformat(),
                         f"{float(level):.5f}", f"{float(stop):.5f}"))
    return "hs-" + sha256(material.encode("utf-8")).hexdigest()[:28]


def _stable_parent_id(symbol: str, strategy: str, direction: int,
                       structure_ts: Any) -> str:
    """Parent thesis identity shared by distinct entry zones from one structure."""
    material = "|".join((str(symbol).upper(), strategy, str(int(direction)),
                         _utc(structure_ts).isoformat()))
    return "hp-" + sha256(material.encode("utf-8")).hexdigest()[:28]


def _setup_payload(*, symbol: str, strategy: str, direction: int,
                   formation_ts: Any, now: pd.Timestamp, level: float,
                   zone_low: float, zone_high: float, stop: float,
                   structure_ts: Any, context_direction: int,
                   atr_m5: float, description: str,
                   cfg: HotScalpConfig) -> dict:
    formed = _utc(formation_ts)
    setup_id = _stable_setup_id(symbol, strategy, direction, formed,
                                level, stop)
    return {
        "setup_id": setup_id,
        "parent_setup_id": _stable_parent_id(symbol, strategy, direction, structure_ts),
        "symbol": str(symbol),
        "strategy": str(strategy),
        "direction": int(direction),
        "formation_ts": formed.isoformat(),
        "armed_ts": now.isoformat(),
        "expires_at": (formed + pd.Timedelta(
            minutes=cfg.max_setup_age_minutes)).isoformat(),
        "level": float(level),
        "zone_low": float(min(zone_low, zone_high)),
        "zone_high": float(max(zone_low, zone_high)),
        "stop": float(stop),
        "structure_ts": _utc(structure_ts).isoformat(),
        "context_direction": int(context_direction),
        "atr_m5": float(atr_m5),
        "description": str(description),
    }


def _recent_structure_events(m5: pd.DataFrame, m15_bias: int,
                             lookback: int = 18) -> list[dict]:
    """Evaluate each recent closed bar as-of that bar; no future candles leak."""
    if m5 is None or len(m5) < 35:
        return []
    start = max(2 * 3 + 5, len(m5) - lookback)
    out = []
    for pos in range(start, len(m5)):
        prefix = m5.iloc[:pos + 1]
        try:
            event = structure_event(prefix, trend=m15_bias)
        except (ValueError, TypeError, IndexError, KeyError):
            continue
        if event.get("present") and event.get("direction") in (-1, 1):
            out.append({**event, "event_ts": _utc(m5.index[pos]),
                        "event_pos": pos})
    # The detector may describe the same last-bar event on more than one
    # historical prefix only if its implementation permits it; keep one/bar.
    unique = {item["event_ts"].isoformat(): item for item in out}
    return sorted(unique.values(), key=lambda x: x["event_ts"])


def build_hot_scalp_setups(symbol: str, frames: dict[str, pd.DataFrame],
                           now: Any,
                           cfg: HotScalpConfig | None = None) -> list[dict]:
    """Create persistent setup descriptions from M1/M5/M15 closed history.

    This function may run during startup warmup: it can arm structure that is
    still valid now, but it never emits an order or replays a past trigger.
    Actual candidate creation only occurs on a new live tick in the engine.
    """
    cfg = cfg or HotScalpConfig()
    if not cfg.enabled:
        return []
    now_ts = _utc(now)
    m5 = frames.get("M5")
    m15 = frames.get("M15")
    m1 = frames.get("M1")
    if (m1 is None or len(m1) < cfg.min_m1_bars
            or m5 is None or len(m5) < cfg.min_m5_bars
            or m15 is None or len(m15) < cfg.min_m15_bars):
        return []
    fresh_m1, _ = _fresh(m1, "M1", now_ts, cfg.fresh_m1_seconds)
    fresh_m5, _ = _fresh(m5, "M5", now_ts, cfg.fresh_m5_seconds)
    fresh_m15, _ = _fresh(m15, "M15", now_ts, cfg.fresh_m15_seconds)
    if not (fresh_m1 and fresh_m5 and fresh_m15):
        return []
    atr = float(atr14(m5) or 0.0)
    if not math.isfinite(atr) or atr <= 0:
        return []
    context = trend_bias(m15)
    context_dir = int(context.get("bias", 0) or 0)
    events = _recent_structure_events(m5, context_dir)
    setups: list[dict] = []
    max_age = pd.Timedelta(minutes=cfg.max_setup_age_minutes)

    # Strategy A — continuation into a fresh unfilled M5 imbalance, with a
    # recent same-direction structure event. Structure + FVG are later grouped
    # under the same impulse so they cannot create two votes for one leg.
    if context_dir in (-1, 1):
        try:
            fvgs = unfilled_fvgs(m5, direction=context_dir, lookback=100)
        except (ValueError, TypeError, IndexError, KeyError):
            fvgs = []
        for fvg in fvgs[:8]:
            fvg_ts = _utc(fvg.get("ts", m5.index[-1]))
            if now_ts < fvg_ts or now_ts - fvg_ts > max_age:
                continue
            candidates = [e for e in events
                          if e["direction"] == context_dir
                          and abs((fvg_ts - e["event_ts"]).total_seconds())
                          <= 6 * 5 * 60]
            if not candidates:
                continue
            ev = min(candidates, key=lambda e: abs(
                (fvg_ts - e["event_ts"]).total_seconds()))
            bottom, top = float(fvg["bottom"]), float(fvg["top"])
            level = float(fvg.get("ce", (top + bottom) / 2.0))
            stop = bottom - cfg.stop_buffer_atr * atr if context_dir > 0 \
                else top + cfg.stop_buffer_atr * atr
            setups.append(_setup_payload(
                symbol=symbol, strategy="pullback_continuation",
                direction=context_dir, formation_ts=fvg_ts, now=now_ts,
                level=level, zone_low=bottom, zone_high=top, stop=stop,
                structure_ts=ev["event_ts"], context_direction=context_dir,
                atr_m5=atr,
                description=(f"M15 {context['detail']}; M5 {ev['kind']} "
                             f"and unfilled FVG {bottom:.5f}-{top:.5f}"),
                cfg=cfg))

    # Strategy B — sweep and reclaim, only after a subsequent structure event
    # confirms the reversal side. A sweep alone is not an entry trigger.
    try:
        pools = liquidity_pools(m5, lookback=min(240, len(m5)))
    except (ValueError, TypeError, IndexError, KeyError):
        pools = []
    for pool in pools[:12]:
        direction = 1 if pool.get("side") == "sell_side" else -1
        level = float(pool.get("level", 0.0) or 0.0)
        if level <= 0:
            continue
        try:
            sweep = detect_sweep(m5, level, direction=direction, atr=atr)
        except (ValueError, TypeError, IndexError, KeyError):
            continue
        if not sweep.get("present"):
            continue
        pos = int(sweep.get("bar_index", -1))
        if pos < 0 or pos >= len(m5):
            continue
        sweep_ts = _utc(m5.index[pos])
        # A wick from an old session cannot stay armed indefinitely.
        if now_ts < sweep_ts or now_ts - sweep_ts > max_age:
            continue
        confirms = [e for e in events if e["direction"] == direction
                    and e["event_pos"] >= pos
                    and e["event_pos"] - pos <= 6]
        if not confirms:
            continue
        ev = confirms[-1]
        wick = float(sweep.get("invalidation") or sweep.get("wick") or level)
        stop = wick - cfg.stop_buffer_atr * atr if direction > 0 \
            else wick + cfg.stop_buffer_atr * atr
        band = cfg.trigger_band_atr * atr
        setups.append(_setup_payload(
            symbol=symbol, strategy="sweep_reclaim", direction=direction,
            formation_ts=sweep_ts, now=now_ts, level=level,
            zone_low=level - band, zone_high=level + band, stop=stop,
            structure_ts=ev["event_ts"], context_direction=context_dir,
            atr_m5=atr,
            description=(f"M5 confirmed {('sell-side' if direction > 0 else 'buy-side')} "
                         f"sweep at {level:.5f}; {ev['kind']} confirms reclaim"),
            cfg=cfg))

    # Strategy C — broken-level retest, independent of the FVG family. The
    # higher context aligns the setup; tick reaction, not a later candle close,
    # supplies the actual trigger.
    if context_dir in (-1, 1):
        for ev in events[-8:]:
            if ev["direction"] != context_dir:
                continue
            formed = ev["event_ts"] + pd.Timedelta(minutes=5)
            if now_ts < formed or now_ts - formed > max_age:
                continue
            level = float(ev["ref_high"] if context_dir > 0
                          else ev["ref_low"])
            band = cfg.trigger_band_atr * atr
            stop = float(ev["invalidation"]) - cfg.stop_buffer_atr * atr \
                if context_dir > 0 else \
                float(ev["invalidation"]) + cfg.stop_buffer_atr * atr
            setups.append(_setup_payload(
                symbol=symbol, strategy="breakout_retest",
                direction=context_dir, formation_ts=formed, now=now_ts,
                level=level, zone_low=level - band, zone_high=level + band,
                stop=stop, structure_ts=ev["event_ts"],
                context_direction=context_dir, atr_m5=atr,
                description=(f"M5 {ev['kind']} of {level:.5f}; "
                             "waiting for a live retest and reclaim"), cfg=cfg))

    # Same strategy + same structure identity can be found via multiple
    # detectors. Keep only the first deterministic representation.
    unique: dict[str, dict] = {}
    for setup in sorted(setups, key=lambda x: (x["formation_ts"], x["setup_id"])):
        unique.setdefault(setup["setup_id"], setup)
    return list(unique.values())[-cfg.max_active_setups:]


def _target_from_structure(m5: pd.DataFrame, direction: int,
                           entry: float) -> float | None:
    """Nearest confirmed opposing swing beyond the actual tick entry."""
    try:
        highs, lows = find_swings(m5["high"].to_numpy(dtype=float),
                                  m5["low"].to_numpy(dtype=float), 3)
        levels = [float(p) for _, p in (highs if direction > 0 else lows)
                  if (float(p) > entry if direction > 0 else float(p) < entry)]
        if levels:
            return min(levels) if direction > 0 else max(levels)
    except (ValueError, TypeError, IndexError, KeyError):
        pass
    recent = m5.tail(60)
    if direction > 0:
        candidates = recent.loc[recent["high"] > entry, "high"]
        return float(candidates.min()) if len(candidates) else None
    candidates = recent.loc[recent["low"] < entry, "low"]
    return float(candidates.max()) if len(candidates) else None


class HotScalpResearchEngine:
    """Persistent, multi-process-safe *shadow* candidate evaluator."""

    def __init__(self, store: SetupStore, cfg: HotScalpConfig | None = None,
                 *, instance_id: str | None = None):
        self.store = store
        self.cfg = cfg or HotScalpConfig()
        self.instance_id = instance_id or uuid.uuid4().hex
        self._last_setup_scan_key: tuple[str | None, str | None] | None = None
        self._previous_mid: float | None = None
        self._forming_minute: pd.Timestamp | None = None
        self._forming_ohlc: dict[str, float] = {}

    def _update_live_m1(self, now: pd.Timestamp, mid: float) -> None:
        minute = now.floor("min")
        if self._forming_minute != minute:
            self._forming_minute = minute
            self._forming_ohlc = {"open": mid, "high": mid,
                                  "low": mid, "close": mid}
        else:
            self._forming_ohlc["high"] = max(self._forming_ohlc["high"], mid)
            self._forming_ohlc["low"] = min(self._forming_ohlc["low"], mid)
            self._forming_ohlc["close"] = mid

    def on_tick(self, *, symbol: str, tick: dict, frames: dict[str, pd.DataFrame],
                now: Any = None) -> list[dict]:
        """Process one live tick; returns only lifecycle events/candidates."""
        if not self.cfg.enabled:
            return []
        events: list[dict] = []
        try:
            now_ts = _utc(now if now is not None else tick.get("time"))
            mid = float(tick.get("mid"))
            if not math.isfinite(mid) or mid <= 0:
                return []
            self._update_live_m1(now_ts, mid)
            m1, m5, m15 = (frames.get("M1"), frames.get("M5"),
                           frames.get("M15"))
            missing_tfs = [tf for tf, frame in (("M1", m1), ("M5", m5), ("M15", m15))
                           if frame is None or not len(frame)]
            if missing_tfs:
                self._previous_mid = mid
                return [{"event": "scan_degraded",
                         "reason_code": "data.missing_timeframe",
                         "reason": "Hot Scalp scan withheld because required bars are missing",
                         "missing_timeframes": missing_tfs,
                         "execution_authority": "none"}]
            fresh1, age1 = _fresh(m1, "M1", now_ts, self.cfg.fresh_m1_seconds)
            fresh5, age5 = _fresh(m5, "M5", now_ts, self.cfg.fresh_m5_seconds)
            fresh15, age15 = _fresh(m15, "M15", now_ts, self.cfg.fresh_m15_seconds)
            if not (fresh1 and fresh5 and fresh15):
                self._previous_mid = mid
                # Do not age/invalidate setups using stale market data.
                return [{"event": "scan_degraded", "reason_code": "data.stale",
                         "reason": "Hot Scalp scan withheld: M1/M5/M15 data stale",
                         "freshness_seconds": {"M1": age1, "M5": age5, "M15": age15}}]

            scan_key = (str(_utc(m5.index[-1])), str(_utc(m15.index[-1])))
            if self._last_setup_scan_key != scan_key:
                for setup in build_hot_scalp_setups(symbol, frames, now_ts, self.cfg):
                    result = self.store.arm(setup)
                    if result["inserted"]:
                        event_payload = {"event": "setup_armed", "setup": setup,
                                         "status": result["status"]}
                        inserted = self.store.write_event_once(
                            event_id=f"armed:{setup['setup_id']}",
                            setup_id=setup["setup_id"], event_type="setup_armed",
                            now=now_ts.isoformat(), payload=event_payload)
                        if inserted:
                            events.append(event_payload)
                self._last_setup_scan_key = scan_key

            # Current tick may arrive after a restart; setup identity/status are
            # recovered from SQLite, but the first tick in a process is never
            # mistaken for a cross/reclaim because previous_mid is unknown.
            prev_mid = self._previous_mid
            self._previous_mid = mid
            atr1 = float(atr14(m1) or 0.0) if len(m1) >= 15 else 0.0
            if not math.isfinite(atr1) or atr1 <= 0:
                atr1 = float(atr14(m5) or 0.0) * 0.2
            if not math.isfinite(atr1) or atr1 <= 0:
                return [{"event": "scan_degraded",
                         "reason_code": "volatility.unavailable",
                         "reason": "Hot Scalp trigger withheld because ATR cannot be measured",
                         "execution_authority": "none"}]
            forming = dict(self._forming_ohlc)
            for row in self.store.active():
                setup = row["payload"]
                sid = row["setup_id"]
                status = row["status"]
                expires = _utc(setup["expires_at"])
                if now_ts >= expires:
                    if self.store.terminate(sid, "expired", "setup TTL elapsed", now_ts):
                        events.append({"event": "setup_expired", "setup_id": sid,
                                       "reason_code": "setup.expired"})
                    continue
                # Invalidation is close-confirmed on the fast one-minute chart;
                # a transient tick wick alone does not destroy the thesis.
                last_close = float(m1["close"].iloc[-1]) if len(m1) else mid
                direction = int(setup["direction"])
                stop = float(setup["stop"])
                if (direction > 0 and last_close <= stop) or \
                        (direction < 0 and last_close >= stop):
                    if self.store.terminate(sid, "invalidated",
                                            "closed M1 candle breached structural invalidation",
                                            now_ts):
                        events.append({"event": "setup_invalidated",
                                       "setup_id": sid,
                                       "reason_code": "setup.invalidated_close"})
                    continue
                lo, hi = float(setup["zone_low"]), float(setup["zone_high"])
                if status == "armed" and lo <= mid <= hi:
                    if self.store.mark_touched(sid, now_ts):
                        events.append({"event": "setup_touched", "setup_id": sid,
                                       "reason_code": "setup.level_touched"})
                    continue
                # Cross/reclaim is an intrabar event: it cannot fire from a
                # historical bar or on the first tick after process restart.
                if status != "touched" or prev_mid is None or atr1 <= 0:
                    continue
                level = float(setup["level"])
                crossed = ((direction > 0 and prev_mid <= level < mid) or
                           (direction < 0 and prev_mid >= level > mid))
                if not crossed:
                    continue
                bar_range = max(float(forming.get("high", mid)) -
                                float(forming.get("low", mid)), 1e-12)
                if direction > 0:
                    reaction_strength = (mid - float(forming.get("low", mid))) / bar_range
                else:
                    reaction_strength = (float(forming.get("high", mid)) - mid) / bar_range
                if reaction_strength < 0.55:
                    continue

                bid, ask = tick.get("bid"), tick.get("ask")
                if bid is None or ask is None:
                    events.append({"event": "candidate_blocked",
                                   "setup_id": sid,
                                   "reason_code": "costs.quote_missing",
                                   "reason": "live bid/ask unavailable; no cost estimate"})
                    continue
                bid, ask = float(bid), float(ask)
                spread = ask - bid
                if not math.isfinite(spread) or spread < 0:
                    continue
                if spread > self.cfg.max_spread_atr_m1 * atr1:
                    if self.store.terminate(sid, "blocked",
                                            "spread exceeds Hot Scalp cost envelope",
                                            now_ts):
                        events.append({"event": "candidate_blocked",
                                       "setup_id": sid,
                                       "reason_code": "costs.spread_too_wide",
                                       "spread": spread, "atr_m1": atr1})
                    continue

                target = _target_from_structure(m5, direction, mid)
                if target is None:
                    if self.store.terminate(sid, "blocked",
                                            "no opposing structural target beyond entry",
                                            now_ts):
                        events.append({"event": "candidate_blocked",
                                       "setup_id": sid,
                                       "reason_code": "geometry.no_target"})
                    continue
                risk = (mid - stop) if direction > 0 else (stop - mid)
                reward = (target - mid) if direction > 0 else (mid - target)
                net_rr = ((reward - spread) / (risk + spread)
                          if risk > 0 and reward > spread else 0.0)
                if risk <= 0 or reward <= 0 or net_rr < self.cfg.min_net_reward_risk:
                    if self.store.terminate(sid, "blocked",
                                            "net reward/risk below Hot Scalp geometry requirement",
                                            now_ts):
                        events.append({"event": "candidate_blocked",
                                       "setup_id": sid,
                                       "reason_code": "geometry.net_rr_below_min",
                                       "risk": risk, "reward": reward,
                                       "net_rr": net_rr})
                    continue

                evidence = self._evidence(setup, frames, now_ts, mid,
                                         spread, atr1, reaction_strength)
                score = score_evidence(evidence, CATEGORY_WEIGHTS)
                claim = self.store.claim_trigger(sid, self.instance_id,
                                                  now_ts.isoformat())
                if claim is None:
                    continue  # another process already claimed this setup
                candidate = {
                    "event": "candidate_shadow",
                    "setup_id": sid,
                    "parent_setup_id": str(setup.get("parent_setup_id") or sid),
                    "entry_id": child_entry_id(
                        str(setup.get("parent_setup_id") or sid),
                        "same_setup", sid),
                    "symbol": str(symbol),
                    "session_context": session_label(now_ts),
                    "strategy": str(setup["strategy"]),
                    "direction": direction,
                    "entry_reference": mid,
                    "structural_stop": stop,
                    "target_reference": float(target),
                    "risk_distance": risk,
                    "reward_distance": reward,
                    "spread": spread,
                    "net_reward_risk": net_rr,
                    "score": score.score,
                    "score_is_calibrated_probability": False,
                    "evidence_score": score.to_dict(),
                    "evidence": [i.to_dict() for i in evidence],
                    "formation_ts": setup["formation_ts"],
                    "armed_ts": setup["armed_ts"],
                    "trigger_ts": now_ts.isoformat(),
                    "invalidation_ts": None,
                    "expiry_ts": setup["expires_at"],
                    "trigger_source": "live_tick_reclaim",
                    "stacking_policy": {
                        "supported_modes": ["pyramid", "same_setup"],
                        "max_layers": StackingConfig().max_layers,
                        "shared_parent_risk_budget": True,
                        "pyramid_min_favourable_r": StackingConfig().pyramid_min_favourable_r,
                        "same_setup_min_parent_r": StackingConfig().same_setup_min_parent_r,
                        "no_averaging_down": True,
                        "requires_fresh_independent_trigger_per_layer": True,
                        "requires_broker_normalized_stop_risk": True,
                        "execution_authority": "research_policy_only",
                    },
                    "execution_authority": "shadow_only",
                    "claim": claim,
                }
                if self.store.write_event_once(
                        event_id=f"candidate:{sid}", setup_id=sid,
                        event_type="candidate_shadow", now=now_ts.isoformat(),
                        payload=candidate):
                    events.append(candidate)
                self.store.finish_shadow_trigger(
                    setup_id=sid, claimant=self.instance_id,
                    fencing_token=int(claim["fencing_token"]), now=now_ts.isoformat())
        except SetupStoreError as exc:
            # Fail closed: a persistence/ownership failure cannot produce a
            # candidate that looks eligible for execution.
            return [{"event": "scan_degraded", "reason_code": "setup_store.unavailable",
                     "reason": str(exc), "execution_authority": "none"}]
        except Exception as exc:  # the tick loop must survive this research sleeve
            return [{"event": "scan_degraded", "reason_code": "hot_scalp.scan_error",
                     "reason": f"{type(exc).__name__}: {exc}",
                     "execution_authority": "none"}]
        return events

    def _evidence(self, setup: dict, frames: dict[str, pd.DataFrame],
                  now: pd.Timestamp, mid: float, spread: float,
                  atr_m1: float, reaction_strength: float) -> list[EvidenceItem]:
        direction = int(setup["direction"])
        m1, m5, m15 = frames["M1"], frames["M5"], frames["M15"]
        evidence: list[EvidenceItem] = []
        ctx = trend_bias(m15)
        cdir = int(ctx.get("bias", 0) or 0)
        evidence.append(EvidenceItem(
            key=f"context:{_utc(m15.index[-1]).isoformat()}:{direction}",
            label="broader price structure", source="patterns.trend_bias",
            observed_at=_close_ts(m15, "M15").isoformat(), timeframe="M15",
            category="higher_context", correlation_group=f"m15-trend:{_utc(m15.index[-1]).isoformat()}",
            stance=("supports" if cdir == direction else
                    "contradicts" if cdir == -direction else "neutral"),
            strength=1.0 if cdir else 0.0,
            detail=str(ctx.get("detail", "bias unavailable"))))
        ev_ts = str(setup.get("structure_ts") or setup["formation_ts"])
        evidence.append(EvidenceItem(
            key=f"structure:{setup['setup_id']}", label="confirmed M5 structure event",
            source="patterns.structure_event", observed_at=ev_ts,
            timeframe="M5", category="structure",
            correlation_group=f"impulse:{ev_ts}", stance="supports",
            strength=1.0, detail=setup["description"]))
        if setup["strategy"] == "pullback_continuation":
            evidence.append(EvidenceItem(
                key=f"imbalance:{setup['setup_id']}", label="unfilled price imbalance",
                source="patterns.unfilled_fvgs", observed_at=setup["formation_ts"],
                timeframe="M5", category="imbalance",
                correlation_group=f"impulse:{ev_ts}", stance="supports",
                strength=0.8, detail="FVG remains unfilled at setup scan"))
        pd_loc = premium_discount(m15)
        position = str(pd_loc.get("zone", pd_loc.get("position", pd_loc.get("location", "")))).lower()
        # The exact schema evolved across source snapshots; only score it when
        # the detector explicitly says premium/discount. Unknown is neutral.
        loc_stance = "neutral"
        if position in ("discount", "below_eq", "below equilibrium"):
            loc_stance = "supports" if direction > 0 else "contradicts"
        elif position in ("premium", "above_eq", "above equilibrium"):
            loc_stance = "supports" if direction < 0 else "contradicts"
        evidence.append(EvidenceItem(
            key=f"location:{_utc(m15.index[-1]).isoformat()}:{direction}",
            label="premium/discount location", source="patterns.premium_discount",
            observed_at=_close_ts(m15, "M15").isoformat(), timeframe="M15",
            category="location", correlation_group=f"range-location:{_utc(m15.index[-1]).isoformat()}",
            stance=loc_stance, strength=0.7 if loc_stance != "neutral" else 0.0,
            detail=str(pd_loc.get("detail", "location schema not recognized"))))
        if len(m5) >= 21:
            vc = volume_confirm(m5)
            if vc.get("present"):
                last_bar_direction = (1 if float(m5["close"].iloc[-1]) > float(m5["open"].iloc[-1])
                                      else -1 if float(m5["close"].iloc[-1]) < float(m5["open"].iloc[-1])
                                      else 0)
                volume_stance = ("supports" if last_bar_direction == direction else
                                  "contradicts" if last_bar_direction == -direction else
                                  "neutral")
                evidence.append(EvidenceItem(
                    key=f"volume:{_utc(m5.index[-1]).isoformat()}",
                    label="direction-aware tick-volume expansion", source="patterns.volume_confirm + M5 candle direction",
                    observed_at=_close_ts(m5, "M5").isoformat(), timeframe="M5",
                    category="activity", correlation_group=f"volume:{_utc(m5.index[-1]).isoformat()}",
                    stance=volume_stance,
                    strength=min(1.0, float(vc.get("ratio", 1.5)) / 2.0) if volume_stance != "neutral" else 0.0,
                    detail=str(vc.get("detail", "volume surge")) + f"; candle direction={last_bar_direction}"))

        # A small, non-gating momentum indicator vote: EMA(9)/EMA(21) alignment
        # on M1. This is evidence, not a standalone signal or hard veto.
        if len(m1) >= 22:
            closes = m1["close"].astype(float)
            fast = closes.ewm(span=9, adjust=False).mean()
            slow = closes.ewm(span=21, adjust=False).mean()
            ema_gap = float(fast.iloc[-1] - slow.iloc[-1])
            normalized_gap = abs(ema_gap) / max(float(atr_m1), 1e-12)
            if normalized_gap < 0.05:
                momentum_stance = "neutral"
                momentum_strength = 0.0
            else:
                momentum_stance = "supports" if direction * ema_gap > 0 else "contradicts"
                momentum_strength = min(1.0, normalized_gap / 0.25)
            evidence.append(EvidenceItem(
                key=f"ema9x21:{_utc(m1.index[-1]).isoformat()}:{direction}",
                label="M1 EMA(9)/EMA(21) momentum alignment",
                source="pandas.ewm EMA(9), EMA(21)",
                observed_at=_close_ts(m1, "M1").isoformat(), timeframe="M1",
                category="momentum", correlation_group=f"m1-ema-momentum:{_utc(m1.index[-1]).isoformat()}",
                stance=momentum_stance, strength=momentum_strength,
                detail=f"EMA9-EMA21={ema_gap:.5f}; normalized by ATR={normalized_gap:.3f}"))
        session = session_label(now)
        in_killzone = bool(session.get("in_killzone"))
        evidence.append(EvidenceItem(
            key=f"session:{now.date()}:{session.get('session')}:{session.get('killzone')}",
            label="session / killzone state", source="reader.session_timeframes + patterns.session_label",
            observed_at=now.isoformat(), timeframe="clock", category="timing",
            correlation_group=f"session-window:{now.date()}:{session.get('session')}:{session.get('killzone')}",
            stance="supports" if in_killzone else "neutral",
            strength=0.7 if in_killzone else 0.0,
            detail=str(session.get("detail", "session state unavailable"))))
        evidence.append(EvidenceItem(
            key=f"reaction:{setup['setup_id']}:{now.isoformat()}",
            label="live tick crossed the setup level after a touch",
            source="HotScalpResearchEngine.on_tick", observed_at=now.isoformat(),
            timeframe="tick/M1", category="reaction",
            correlation_group=f"live-reaction:{setup['setup_id']}",
            stance="supports", strength=max(0.0, min(1.0, reaction_strength)),
            detail="intrabar rejection/reclaim; no completed-bar wait"))
        spread_ratio = spread / atr_m1 if atr_m1 > 0 else 1.0
        evidence.append(EvidenceItem(
            key=f"cost:{setup['setup_id']}:{now.isoformat()}",
            label="spread relative to fast-chart volatility", source="tick.bid/ask + atr14",
            observed_at=now.isoformat(), timeframe="tick/M1",
            category="execution_cost", correlation_group=f"spread:{now.floor('min')}",
            stance="supports" if spread_ratio <= self.cfg.max_spread_atr_m1 else "contradicts",
            strength=max(0.0, min(1.0, 1.0 - spread_ratio / self.cfg.max_spread_atr_m1)),
            detail=f"spread/ATR={spread_ratio:.3f}"))
        return evidence
