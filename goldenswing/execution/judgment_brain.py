"""The judgment's brain rung (owner order 2026-11-27).

The code rung (execution/judgment.py) is the deterministic
floor.  This is what makes the judgment a real trader, not
a formula: the brain reads the SAME measured structure
evidence the code rung reads and returns the verdict, the
way a trader reads a chart and calls it.

This module is the smallest real version of the brain
plumbing: the GS repo has NO LLM machinery at all (no
needs_brain / ToolSpec / braindispatch / provider env —
verified by grep 2026-11-27; that machinery lives in the
Devon repo, not here), so the provider is built here, on
an OpenAI-compatible chat endpoint, configured by env:

    GS_JUDGMENT_BRAIN_PROVIDER   groq | openai (default groq)
    GS_JUDGMENT_BRAIN_URL        chat-completions base URL
    GS_JUDGMENT_BRAIN_API_KEY    the key
    GS_JUDGMENT_BRAIN_MODEL      the model name

With provider=groq and no URL, the URL defaults to Groq's
OpenAI-compatible endpoint (a documented fact, not a
guess).  No provider configured -> the brain rung is
unavailable and the code rung is the verdict (the runner
says so plainly — the no-model rule: the deterministic
part plus an honest note, never a faked brain call).

The brain is handed the measured structure (the chart
read, plain numbers — the body) AND the recent entry-clock
candles around the level (structure.chart — the chart
itself, most recent last, with the clock named), plus the
candidate's geometry, and returns a JSON verdict.  The
reply is VALIDATED before it is trusted: direction in
{1, -1, 0}, decision in {take, abstain}, an optional
confidence in 0..1, and a take only when the direction is
the candidate's own direction.  An incoherent,
unparseable, or unreachable reply is not a verdict — it
is None, and the code rung decides.  Never a stub, never
a fake brain call.

What the box needs for the brain to be up (nothing is
assumed — an unconfigured brain is unavailable, and the
code rung decides every judgment with a plain brain_note
on the record):

    GS_JUDGMENT_BRAIN_API_KEY   the key (a Groq key, by default)
    GS_JUDGMENT_BRAIN_MODEL     the model name (e.g. a Groq model)

GS_JUDGMENT_BRAIN_PROVIDER defaults to groq (its URL
defaults to Groq's OpenAI-compatible endpoint); set
GS_JUDGMENT_BRAIN_URL only for another OpenAI-compatible
endpoint.  Without the key AND the model, no brain verdict
is ever attempted — the code rung is the verdict, and the
record says so.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

# The marker: this judgment entry point needs a brain.
# (The Devon repo's ToolSpec.needs_brain, in GS's shape.)
needs_brain = True

DEFAULT_TIMEOUT_S = 15.0
_GROQ_URL = "https://api.groq.com/openai/v1"


@dataclass
class BrainJudgmentProvider:
    """One brain endpoint.  Built from env; nothing is
    assumed — an unconfigured provider is unavailable."""
    url: str
    api_key: str
    model: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def from_env(cls, environ=None) -> BrainJudgmentProvider | None:
        env = os.environ if environ is None else environ
        provider = (env.get("GS_JUDGMENT_BRAIN_PROVIDER")
                    or "groq").strip().lower()
        url = (env.get("GS_JUDGMENT_BRAIN_URL") or "").strip()
        if not url and provider == "groq":
            url = _GROQ_URL
        api_key = (env.get("GS_JUDGMENT_BRAIN_API_KEY")
                   or "").strip()
        model = (env.get("GS_JUDGMENT_BRAIN_MODEL") or "").strip()
        if not url or not api_key or not model:
            return None
        timeout = env.get("GS_JUDGMENT_BRAIN_TIMEOUT_S")
        try:
            timeout_s = float(timeout) if timeout else DEFAULT_TIMEOUT_S
        except ValueError:
            timeout_s = DEFAULT_TIMEOUT_S
        return cls(url=url, api_key=api_key, model=model,
                   timeout_s=timeout_s)

    def available(self) -> bool:
        return bool(self.url and self.api_key and self.model)


_PROMPT = (
    "You are a trader reading a chart. Below is the market at "
    "decision time: the recent entry-clock candles around the "
    "level (structure.chart — tf names the clock, bars are "
    "most recent last, each with time, open, high, low, "
    "close), the measured structure reads — displacement, "
    "acceptance-by-closes, follow-through, phase, HTF biases, "
    "the Asymmetry score — and one candidate trade (direction, "
    "entry, stop, first structural target). Read the CANDLES "
    "first, like a trader looking at the chart: where price "
    "came from, whether the pullback held, whether the last "
    "closes confirm or reject the level — then check the "
    "measured reads against what the chart shows, and call "
    "it: which side does the market favour, and does the "
    "candidate take it? Reply with JSON only, exactly this "
    "shape: {\"direction\": 1 | -1 | 0, \"decision\": \"take\" | "
    "\"abstain\", \"reason\": \"one plain sentence naming the "
    "reads that decided it\", \"confidence\": 0..1 (optional, "
    "your own confidence in the call)}. direction is the side "
    "the market favours (1 = LONG, -1 = SHORT, 0 = no side "
    "measured). decision is take only when the candidate "
    "agrees with the measured direction and the reaction "
    "confirms it; otherwise abstain."
)


def _brain_verdict(structure: dict, candidate: dict,
                   provider: BrainJudgmentProvider) -> dict | None:
    """The brain's raw verdict, validated.  None = no verdict
    (unavailable, unreachable, unparseable, or incoherent) —
    the caller falls back to the code rung."""
    body = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": _PROMPT},
            {"role": "user", "content": json.dumps({
                "structure": structure,
                "candidate": {
                    "direction": candidate.get("direction"),
                    "entry": candidate.get("entry"),
                    "stop": candidate.get("stop"),
                    "targets": candidate.get("targets"),
                    "asymmetry_score": candidate.get(
                        "asymmetry_score"),
                }}, default=str)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        provider.url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {provider.api_key}"},
        method="POST")
    try:
        with urllib.request.urlopen(
                request, timeout=provider.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, OSError, ValueError, KeyError):
        return None
    try:
        content = payload["choices"][0]["message"]["content"]
        verdict = json.loads(content)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    direction = verdict.get("direction")
    decision = verdict.get("decision")
    reason = verdict.get("reason")
    if direction not in (1, -1, 0):
        return None
    if decision not in ("take", "abstain"):
        return None
    if not isinstance(reason, str) or not reason.strip():
        return None
    candidate_direction = candidate.get("direction")
    if decision == "take" and (
            direction != candidate_direction or direction == 0):
        # A take in the opposite direction is incoherent —
        # not a verdict.
        return None
    out = {"mode": "live", "rung": "brain",
           "decision": decision, "direction": direction,
           "reason": reason.strip(),
           "reason_category": verdict.get("reason_category")}
    # The brain's own confidence, when it stated one —
    # validated 0..1.  Absent = absent (the caller never
    # fills it with the code rung's agreement share).
    confidence = verdict.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return None
        if not 0.0 <= confidence <= 1.0:
            return None
        out["confidence"] = confidence
    return out


def judge_with_brain(structure: dict, candidate: dict,
                     provider: BrainJudgmentProvider | None
                     ) -> dict | None:
    """The brain rung: the brain reads the measured structure
    and returns the verdict.  None = no brain verdict — the
    caller uses the code rung and says so plainly."""
    if provider is None or not provider.available():
        return None
    return _brain_verdict(structure, candidate, provider)
