"""The judgment's explanation drafting (owner orders
2026-10-23 + 2026-11-27).

No AI judges a trade.  The verdict is CODE, always
(execution/judgment.py) — the live order path never
consults this module, and it returns no verdict.  What
this module does, and all it does: when a brain is
configured, it drafts a plain-words explanation of a
verdict ALREADY DECIDED, from the same measured structure
evidence the code verdict read.  The drafted explanation
never changes the verdict; when no brain is up, the
verdict's own reason stands (the code's plain words).

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
guess).  No provider configured -> no drafting (None),
never a faked explanation.

The brain is handed the measured structure (the chart
read, plain numbers — the body) AND the recent entry-clock
candles around the level (structure.chart — the chart
itself, most recent last, with the clock named), plus the
candidate's geometry AND the verdict the code already
reached.  The reply is VALIDATED before it is trusted: a
non-empty plain-words explanation string.  An incoherent,
unparseable, or unreachable reply is not an explanation —
it is None, and the code's reason stands.  Never a stub,
never a fake brain call.

What the box needs for drafting to be up (nothing is
assumed — an unconfigured brain drafts nothing):

    GS_JUDGMENT_BRAIN_API_KEY   the key (a Groq key, by default)
    GS_JUDGMENT_BRAIN_MODEL     the model name (e.g. a Groq model)

GS_JUDGMENT_BRAIN_PROVIDER defaults to groq (its URL
defaults to Groq's OpenAI-compatible endpoint); set
GS_JUDGMENT_BRAIN_URL only for another OpenAI-compatible
endpoint.  These env vars configure the drafting provider
only — they gate no trade decision.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

# The marker: this drafting tool needs a brain (a model
# to draft with).  It never decides a trade.
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
    "You are a trader explaining a decision that is already "
    "made. Below is the market at decision time: the recent "
    "entry-clock candles around the level (structure.chart — "
    "tf names the clock, bars are most recent last, each with "
    "time, open, high, low, close), the measured structure "
    "reads — displacement, acceptance-by-closes, "
    "follow-through, phase, HTF biases, the Asymmetry score — "
    "one candidate trade (direction, entry, stop, first "
    "structural target), and the verdict the system already "
    "reached (decision, direction, reason). Do NOT decide — "
    "the decision is made and you cannot change it. Explain "
    "it in plain words, one or two sentences, naming the "
    "reads that decided it, the way a trader would tell "
    "another trader why this trade was taken or passed. "
    "Reply with JSON only, exactly this shape: "
    "{\"explanation\": \"plain words\"}."
)


def _draft_explanation(structure: dict, verdict: dict,
                       candidate: dict,
                       provider: BrainJudgmentProvider
                       ) -> str | None:
    """The brain's drafted explanation of the decided
    verdict, validated.  None = no explanation (unavailable,
    unreachable, unparseable, or incoherent) — the caller
    keeps the code verdict's own reason."""
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
                },
                "verdict": {
                    "decision": verdict.get("decision"),
                    "direction": verdict.get("direction"),
                    "reason": verdict.get("reason"),
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
        reply = json.loads(content)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    explanation = reply.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        return None
    return explanation.strip()


def draft_judgment_explanation(structure: dict, verdict: dict,
                               candidate: dict,
                               provider: BrainJudgmentProvider | None
                               ) -> str | None:
    """Draft a plain-words explanation of the ALREADY-DECIDED
    verdict.  The brain explains; it never decides.  None =
    no explanation drafted — the caller keeps the code
    verdict's own reason."""
    if provider is None or not provider.available():
        return None
    return _draft_explanation(structure, verdict, candidate,
                              provider)
