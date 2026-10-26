"""Carry out what the intent detector decided.

One router for both the CLI and the HTTP API, so a sentence typed into the
web page and the same sentence typed into the terminal do the same thing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .. import answer as answerer
from .. import generate as generator
from .. import improve, models, pipeline
from ..diagnostics import Severity
from ..optimize import OBJECTIVES, recommend
from ..report import build as build_report
from .intent import (ANALYZE, EDIT, EXPLAIN, GENERATE, OPTIMIZE, QUESTION, RUN,
                     SCORE, Intent, detect)

NO_CODE = "Paste your code in the editor first, then ask again."


def _objectives(intent: Intent) -> tuple[str, ...]:
    return intent.objectives or OBJECTIVES


def run(text: str, code: str = "", source: Path | None = None,
        use_model: bool = False) -> dict:
    """Detect what ``text`` asks for, do it, and return a JSON-ready result."""
    if source is not None and not code.strip():
        code = source.read_text(errors="replace")
    has_code = bool(code.strip())
    intent = detect(text, has_code=has_code)

    # Nothing matched. A model can make sense of a sentence the keyword rules
    # cannot, so let it answer rather than printing a report nobody asked for.
    if intent.confidence == 0.0 and models.available() is not None:
        intent = Intent(action=QUESTION, confidence=0.0, scores=intent.scores,
                        because=["no specific action matched"])

    result: dict = {"intent": intent.to_dict()}

    if intent.action == GENERATE:
        generated = generator.generate(intent.description or text)
        result["generated"] = generated.to_dict()
        return result

    if intent.action == QUESTION:
        answered = answerer.answer(text, code)
        result["answer"] = answered.to_dict()
        if not answered.error or not has_code:
            return result
        # No model: the analysis is still worth having.
        result["answer"]["error"] += " Here is the analysis of your code instead."
        intent = Intent(action=ANALYZE, confidence=intent.confidence)

    if not has_code:
        result["error"] = NO_CODE
        return result

    if intent.action == EDIT:
        result["edit"] = improve.edit(code, text).to_dict()
        return result

    with tempfile.TemporaryDirectory(prefix="nlpcc-ask-") as tmp:
        path = source
        if path is None:
            path = Path(tmp) / "submitted.c"
            path.write_text(code)

        analysis = pipeline.analyze(
            path,
            run=(intent.action == RUN),
            quality=intent.action in (SCORE, OPTIMIZE, ANALYZE),
            ai=use_model and intent.action in (SCORE, ANALYZE),
        )

        if use_model and intent.action == EXPLAIN:
            from ..explain import llm
            llm.explain_unmatched(analysis.diagnostics, source=path)

        # "Explain" with nothing wrong means "explain the code".
        if intent.action == EXPLAIN and models.available() is not None and not any(
                d.severity in (Severity.ERROR, Severity.FATAL, Severity.WARNING)
                for d in analysis.diagnostics):
            result["answer"] = answerer.answer(text, code).to_dict()
            # Say what was done, not what was first guessed: the badge read
            # "explain the errors" above an explanation of error-free code.
            result["intent"] = Intent(action=QUESTION, confidence=intent.confidence,
                                      because=["nothing wrong to explain"]).to_dict()
            return result

        objectives = _objectives(intent)
        recommendations = []
        if intent.action in (OPTIMIZE, ANALYZE):
            findings = [d for d in analysis.diagnostics if d.origin.value == "rules"]
            recommendations = recommend(path, objectives,
                                        quality_findings=findings,
                                        score=analysis.score)

        report = build_report(analysis, recommendations, objectives)
        payload = report.to_dict()
        payload["source"] = path.name
        for diag in payload["diagnostics"]:
            diag["file"] = path.name
        result["report"] = payload
        return result
