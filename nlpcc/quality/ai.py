"""AI code analysis -- the diagram's analysis stage.

Reads the program and reports problems across the same five dimensions the
rule engine uses. It runs alongside the rules, not instead of them: the rules
catch what they know and always work offline, and this catches what no fixed
threshold can describe.

Runs on whatever model is available -- Gemini when GEMINI_API_KEY is set,
otherwise a local one through LM Studio or Ollama (see nlpcc/models.py).
Optional either way: no
model, or a reply that does not parse, leaves the analysis exactly as the
rules produced it. Each finding records which provider wrote it.
"""

from __future__ import annotations

from pathlib import Path

from .. import models
from ..diagnostics import Diagnostic, Origin, Phase, Position, Severity
from .rules import MAINTAINABILITY, PERFORMANCE, PRACTICES, READABILITY, SECURITY

DIMENSIONS = (SECURITY, READABILITY, MAINTAINABILITY, PERFORMANCE, PRACTICES)
MAX_FINDINGS = 8

# One review per program text and model, for the life of the process. A model
# reviewing the same code twice does not report the same findings twice, so
# without this the report said 91 and the improve view, analyzing the very
# same program a moment later, started from a different number.
_REVIEWS: dict[tuple[str, str, str], list] = {}
MAX_CACHED_REVIEWS = 64

SYSTEM_PROMPT = f"""You review C code across exactly five dimensions: \
{', '.join(DIMENSIONS)}.

Report only real problems in the code you are given. Say nothing about style
preferences, and nothing a compiler already reports.

For each finding give: the dimension, the line number, a short title, an
explanation a first-year student understands, a one-sentence fix, and a
penalty from 5 to 40 for how much it should cost that dimension's score.

Report at most {MAX_FINDINGS} findings. If the code is fine, report none."""


def analyze(source: Path) -> list[Diagnostic]:
    """AI findings for one file. Empty when the model is unavailable."""
    provider = models.available()
    if provider is None:
        return []

    from pydantic import BaseModel

    class Finding(BaseModel):
        dimension: str
        line: int
        title: str
        explanation: str
        fix: str
        penalty: int

    class Review(BaseModel):
        findings: list[Finding]

    try:
        code = source.read_text(errors="replace")
    except OSError:
        return []

    key = (provider.kind, provider.model, code)
    findings = _REVIEWS.get(key)
    if findings is None:
        numbered = "\n".join(f"{i:4d}  {line}"
                             for i, line in enumerate(code.splitlines(), 1))
        parsed = models.complete(SYSTEM_PROMPT,
                                 f"Review this C program:\n\n{numbered}",
                                 Review, provider=provider)
        if parsed is None:
            return []           # not cached: the next call may get a reply
        findings = parsed.findings
        if len(_REVIEWS) >= MAX_CACHED_REVIEWS:
            _REVIEWS.pop(next(iter(_REVIEWS)))
        _REVIEWS[key] = findings

    lines = code.splitlines()
    out: list[Diagnostic] = []
    for finding in findings[:MAX_FINDINGS]:
        if finding.dimension not in DIMENSIONS:
            continue
        line = max(0, min(finding.line, len(lines)))
        out.append(Diagnostic(
            phase=Phase.QUALITY,
            severity=Severity.WARNING,
            message=finding.title.strip(),
            file=str(source),
            position=Position(line) if line else None,
            code=f"AI_{finding.dimension.upper()}",
            origin=Origin.AI,
            snippet=lines[line - 1].strip() if line else "",
            explanation=finding.explanation.strip(),
            fix=finding.fix.strip(),
            meta={"title": finding.title.strip(),
                  "explained_by": "ai",
                  "generated": True,
                  "provider": provider.kind,
                  "model": provider.model,
                  "dimension": finding.dimension,
                  "penalty": max(5, min(40, finding.penalty))},
        ))
    return out
