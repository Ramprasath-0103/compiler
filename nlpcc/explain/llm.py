"""Step 5b: the model explains what the rules could not.

This layer sits BEHIND nlpcc/explain/rules.py and never in front of it. The
ordering is the whole design:

  * A rule-matched finding never reaches the model, so the common cases are
    instant, free, identical between runs, and reviewable by a human.
  * Only findings no rule covered are sent, and whatever comes back is
    labelled ``explained_by: "llm"`` so the report can mark it as generated
    rather than authored.
  * Every answer is cached on disk by diagnostic code, so the second run of
    the same program costs nothing and works with the network unplugged.
  * It runs on whatever model is available, local first -- LM Studio or
    Ollama need no API key (see nlpcc/models.py).
  * Nothing here can fail the pipeline. No model, no network, a rate limit, a
    malformed reply -- each leaves the diagnostic exactly as it arrived, and
    the CLI goes on reporting it without an explanation.

Each answer the model gives is a candidate for promotion: an explanation that
turns out to be right belongs in rules.py as a permanent rule, which is how
the deterministic layer grows.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .. import models
from ..diagnostics import Diagnostic

CACHE_DIR = Path(".nlpcc-cache") / "explanations"

SYSTEM_PROMPT = """You explain C compiler diagnostics to someone in their first \
year of programming.

Talk the way a friend who knows C would, sitting next to them:
- Start with what happened in their code, naming the variable, function or line.
- Then why C cares, in everyday words. No jargon unless you define it right there.
- Never restate the compiler's message; explain the cause it is pointing at.
- Two or three short sentences for the explanation. Do not pad it.
- The fix is one concrete sentence; show the corrected line when you can.
- If the diagnostic is likely a knock-on effect of an earlier error, say so."""


@dataclass
class ModelExplanation:
    title: str
    explanation: str
    fix: str

    def to_dict(self) -> dict:
        return {"title": self.title, "explanation": self.explanation, "fix": self.fix}


@dataclass
class Outcome:
    """What happened, so the CLI can say why nothing came back."""

    explained: int = 0
    from_cache: int = 0
    attempted: int = 0
    unavailable_reason: str = ""
    provider: str = ""

    @property
    def available(self) -> bool:
        return not self.unavailable_reason


def _cache_key(diag: Diagnostic) -> str:
    # The code plus the shape of the message, with program-specific names
    # stripped, so `undeclared identifier 'totl'` and `'xyz'` share an entry.
    shape = re.sub(r"'[^']*'", "'X'", diag.message)
    raw = f"{diag.code}|{shape}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _read_cache(key: str, cache_dir: Path) -> ModelExplanation | None:
    path = cache_dir / f"{key}.json"
    try:
        data = json.loads(path.read_text())
        return ModelExplanation(data["title"], data["explanation"], data["fix"])
    except (OSError, ValueError, KeyError):
        return None


def _write_cache(key: str, explanation: ModelExplanation, cache_dir: Path) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{key}.json").write_text(json.dumps(explanation.to_dict(), indent=2))
    except OSError:
        pass            # a cache that cannot be written is not an error


def _ask(provider, diag: Diagnostic, source_line: str) -> ModelExplanation | None:
    from pydantic import BaseModel

    class Schema(BaseModel):
        title: str
        explanation: str
        fix: str

    prompt = (
        f"A C compiler reported this diagnostic.\n\n"
        f"Severity: {diag.severity.value}\n"
        f"Phase: {diag.phase.value}\n"
        f"Message: {diag.message}\n"
        f"Diagnostic code: {diag.code or '(none)'}\n"
    )
    if source_line:
        prompt += f"\nThe line it points at:\n\n    {source_line.strip()}\n"
    related = diag.meta.get("related") or []
    if related:
        prompt += "\nThe compiler also noted:\n"
        for item in related:
            prompt += f"  line {item['line']}: {item['message']}\n"
    prompt += ("\nWrite a title of four or five words, the explanation, and the fix.")

    parsed = models.complete(SYSTEM_PROMPT, prompt, Schema, provider=provider,
                             max_tokens=2048)
    if parsed is None:
        return None
    return ModelExplanation(parsed.title.strip(), parsed.explanation.strip(),
                            parsed.fix.strip())


def explain_unmatched(diagnostics: Iterable[Diagnostic], source: Path | None = None,
                      cache_dir: Path | None = None,
                      allow_network: bool | None = None) -> Outcome:
    """Fill in explanations for diagnostics no rule matched. Modifies in place."""
    cache_dir = cache_dir or CACHE_DIR
    pending = [d for d in diagnostics if not d.explanation]
    outcome = Outcome(attempted=len(pending))
    if not pending:
        return outcome

    lines: list[str] = []
    if source is not None:
        try:
            lines = source.read_text(errors="replace").splitlines()
        except OSError:
            lines = []

    # Serve from cache first, so an offline run still improves on nothing.
    still_pending: list[Diagnostic] = []
    for diag in pending:
        cached = _read_cache(_cache_key(diag), cache_dir)
        if cached is None:
            still_pending.append(diag)
            continue
        _apply(diag, cached, source="llm-cache")
        outcome.explained += 1
        outcome.from_cache += 1

    if not still_pending:
        return outcome

    if allow_network is None:
        allow_network = os.environ.get("NLPCC_OFFLINE", "") == ""
    if not allow_network:
        outcome.unavailable_reason = "offline mode (NLPCC_OFFLINE is set)"
        return outcome

    provider = models.available()
    if provider is None:
        outcome.unavailable_reason = models.describe()
        return outcome
    outcome.provider = str(provider)

    for diag in still_pending:
        line = ""
        if diag.position and 0 < diag.position.line <= len(lines):
            line = lines[diag.position.line - 1]
        answer = _ask(provider, diag, line)
        if answer is None:
            continue
        _apply(diag, answer, source="llm")
        _write_cache(_cache_key(diag), answer, cache_dir)
        outcome.explained += 1

    if outcome.explained == outcome.from_cache:
        outcome.unavailable_reason = "the model returned no usable explanation"
    return outcome


def _apply(diag: Diagnostic, answer: ModelExplanation, source: str) -> None:
    diag.explanation = answer.explanation
    diag.fix = answer.fix
    diag.meta["title"] = answer.title
    diag.meta["explained_by"] = source
    diag.meta["generated"] = True        # the report marks these as model-written
