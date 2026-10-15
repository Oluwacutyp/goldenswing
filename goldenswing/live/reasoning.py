"""Dynamic reasoning + study packages for accepted signals.

The owner's bot's ``generate_dynamic_reasoning``: a confidence
label + technical factors + macro factors, all built from the live
signal's own numbers — never canned text.  This module is that
structure over this engine's verdict:

- confidence label from the confluence score (the old bot's bands);
- technical factors: playbook, POI zone, trigger, D1/H4 bias, AMD
  phase, volume ratio, alignment, session, urgency, the playbook's
  own factor list;
- macro factors: DXY correlation (Ghost Sentinel bands, same as the
  confluence bonus — strong inverse = gold tailwind, toxic
  positive = headwind);
- risk: entry / stop / targets / R, computed from the signal.

Every line is derived from ``sig`` + ``verdict``.  Nothing is
invented: a missing field drops its line.  Both builders never
raise — the fallback is a minimal one-line reasoning from the
signal alone (an accepted signal must never lose its alert because
the narrator crashed).

The study package is the key levels + data behind the signal
(zone, entry, stop, targets, R, biases, AMD, volume, DXY, urgency,
alignment, session, score, factors) as a JSON-safe dict — it rides
on the signal record; the Telegram alert carries the reasoning
text + the existing chart PNG (``_build_chart``), which is the
visual half of the study.
"""
from __future__ import annotations


def confidence_label(score: float) -> str:
    """The old bot's confidence bands."""
    if score >= 90:
        return "Very high"
    if score >= 80:
        return "High"
    if score >= 70:
        return "Moderate"
    if score >= 60:
        return "Marginal"
    return "Low"


def _bias_txt(value) -> str:
    if value is None:
        return "n/a"
    try:
        v = int(value)
    except (TypeError, ValueError):
        return str(value)
    return {1: "bullish", -1: "bearish", 0: "flat"}.get(v, str(v))


def _risk_reward(sig) -> float | None:
    try:
        entry = float(sig.entry_ref)
        stop = float(sig.invalidation)
        targets = list(sig.targets or [])
        if not targets or entry == stop:
            return None
        risk = abs(entry - stop)
        return abs(float(targets[0]) - entry) / risk
    except (TypeError, ValueError, AttributeError, ZeroDivisionError):
        return None


def _dxy_line(corr) -> str | None:
    if corr is None:
        return None
    try:
        c = float(corr)
    except (TypeError, ValueError):
        return None
    if c < -0.8:
        return (f"DXY correlation {c:.2f} (strong inverse — "
                f"gold tailwind)")
    if c > -0.2:
        return (f"DXY correlation {c:.2f} (toxic positive — "
                f"DXY headwind)")
    return f"DXY correlation {c:.2f} (neutral)"


def build_study(pb_name: str, sig, verdict: dict | None,
                *, alignment_label: str | None = None) -> dict:
    """The key levels / data behind a signal, JSON-safe."""
    verdict = verdict or {}
    zone = verdict.get("zone") or {}
    volume = verdict.get("volume") or {}
    session = verdict.get("session") or {}
    rr = _risk_reward(sig)
    try:
        factors = list(getattr(sig, "factors", []) or [])
    except TypeError:
        factors = []
    return {
        "playbook": pb_name,
        "direction": int(getattr(sig, "direction", 0) or 0),
        "score": float(getattr(sig, "score", 0.0) or 0.0),
        "confidence": confidence_label(
            float(getattr(sig, "score", 0.0) or 0.0)),
        "entry": float(getattr(sig, "entry_ref", 0.0) or 0.0),
        "stop": float(getattr(sig, "invalidation", 0.0) or 0.0),
        "targets": [float(t) for t in (getattr(sig, "targets", [])
                                       or [])],
        "risk_reward": rr,
        "zone": {"bottom": float(zone.get("bottom", 0.0) or 0.0),
                 "top": float(zone.get("top", 0.0) or 0.0)}
        if zone else None,
        "trigger": verdict.get("trigger"),
        "bias_d1": verdict.get("bias_d1"),
        "bias_h4": verdict.get("bias_h4"),
        "amd_phase": verdict.get("amd_phase"),
        "amd_confidence": verdict.get("amd_confidence"),
        "volume_ratio": volume.get("volume_ratio"),
        "dxy_corr": verdict.get("dxy_corr"),
        "urgency": verdict.get("urgency"),
        "alignment": alignment_label,
        "session": session.get("session")
        if isinstance(session, dict) else session,
        "factors": factors,
    }


def build_reasoning(pb_name: str, sig, verdict: dict | None,
                    *, alignment_label: str | None = None) -> str:
    """Short dynamic reasoning text for an accepted signal.

    Confidence label + technical factors + macro factors + risk,
    every line derived from the signal + verdict.  Never raises.
    """
    try:
        verdict = verdict or {}
        direction = int(getattr(sig, "direction", 0) or 0)
        side = "LONG" if direction > 0 else "SHORT"
        score = float(getattr(sig, "score", 0.0) or 0.0)
        entry = float(getattr(sig, "entry_ref", 0.0) or 0.0)
        stop = float(getattr(sig, "invalidation", 0.0) or 0.0)
        targets = [float(t) for t in (getattr(sig, "targets", [])
                                       or [])]
        rr = _risk_reward(sig)
        lines = [f"Why {side} ({confidence_label(score)} conviction, "
                 f"score {score:.0f}):"]
        lines.append(f"• {pb_name} at the POI — entry {entry:,.1f}, "
                     f"stop {stop:,.1f}"
                     + (f", target {targets[0]:,.1f}"
                        if targets else "")
                     + (f" ({rr:.2f}R)" if rr is not None else ""))
        zone = verdict.get("zone") or {}
        if zone:
            lines.append(f"• Zone {float(zone.get('bottom', 0.0)):,.1f}"
                         f"–{float(zone.get('top', 0.0)):,.1f}"
                         + (f" · trigger: {verdict['trigger']}"
                            if verdict.get("trigger") else ""))
        lines.append(f"• Bias: D1 {_bias_txt(verdict.get('bias_d1'))}"
                     f" / H4 {_bias_txt(verdict.get('bias_h4'))}")
        if verdict.get("amd_phase"):
            conf = verdict.get("amd_confidence")
            lines.append("• AMD phase: "
                         f"{verdict['amd_phase']}"
                         + (f" (confidence {float(conf):.2f})"
                            if conf is not None else ""))
        volume = verdict.get("volume") or {}
        if volume.get("volume_ratio") is not None:
            lines.append("• Volume ratio: "
                         f"{float(volume['volume_ratio']):.2f}x "
                         f"20-bar mean")
        if alignment_label:
            lines.append(f"• Alignment: {alignment_label}")
        session = verdict.get("session") or {}
        sess_name = session.get("session") \
            if isinstance(session, dict) else session
        if sess_name:
            lines.append(f"• Session: {sess_name}")
        if verdict.get("urgency") is not None:
            lines.append(f"• Urgency: "
                         f"{float(verdict['urgency']):.0f}/100")
        dxy = _dxy_line(verdict.get("dxy_corr"))
        if dxy:
            lines.append(f"• {dxy}")
        factors = list(getattr(sig, "factors", []) or [])
        for f in factors:
            lines.append(f"• {f}")
        return "\n".join(lines)
    except Exception:
        # Fallback: the signal alone, never an empty reasoning.
        try:
            return (f"{pb_name} "
                    f"{'LONG' if int(getattr(sig, 'direction', 0)) > 0 else 'SHORT'} "
                    f"@ {float(getattr(sig, 'entry_ref', 0.0)):,.1f}, "
                    f"stop {float(getattr(sig, 'invalidation', 0.0)):,.1f} "
                    f"(reasoning fallback — verdict unavailable)")
        except Exception:
            return f"{pb_name} signal (reasoning unavailable)"
