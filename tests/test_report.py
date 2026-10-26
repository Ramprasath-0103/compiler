
from __future__ import annotations

import json

from conftest import CORPUS
from nlpcc import pipeline
from nlpcc.optimize import recommend
from nlpcc.report import build, to_markdown


def _report(name: str, objectives=("readability", "maintainability")):
    source = CORPUS / name
    analysis = pipeline.analyze(source, quality=True)
    return build(analysis, recommend(source, objectives), objectives)


def test_json_is_valid_and_complete() -> None:
    data = json.loads(_report("quality/deep_nesting.c").to_json())
    assert data["compiles"] is True
    assert data["score"]["overall"] > 0
    assert data["summary"]["findings"] == len(data["diagnostics"])
    assert data["metrics"]["functions"]
    assert data["recommendations"]


def test_score_breakdown_is_in_the_json() -> None:

    data = json.loads(_report("quality/deep_nesting.c").to_json())
    readability = data["score"]["dimensions"]["readability"]
    assert readability["weight"] == 0.20
    assert sum(d["penalty"] for d in readability["deductions"]) == 100 - readability["score"]


def test_markdown_covers_every_section() -> None:
    text = to_markdown(_report("quality/deep_nesting.c"))
    for heading in ("# Code analysis", "## Problems found", "## Quality score",
                    "## Measurements", "## Recommendations"):
        assert heading in text, heading
    assert "What happened." in text
    assert "Fix." in text


def test_a_capped_score_says_why_in_the_report() -> None:
    text = to_markdown(_report("security/unsafe_strcpy.c"))
    assert "capped" in text.lower()


def test_clean_program_reports_no_problems() -> None:
    report = _report("clean.c")
    assert report.compiles
    assert report.diagnostics == []
    assert report.score.overall == 100
    assert "None." in to_markdown(report)


def test_a_security_error_is_not_a_compile_error() -> None:

    report = _report("security/unsafe_strcpy.c")
    assert report.compiles is True
    assert report.errors                     
    assert report.compile_errors == []       
    assert "does not compile" not in to_markdown(report).lower()
