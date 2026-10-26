
from __future__ import annotations

from conftest import CORPUS
from nlpcc import pipeline
from nlpcc.optimize import OBJECTIVES
from nlpcc.server import Options, _options, _parse, analyze_source


def _code(name: str) -> str:
    return (CORPUS / name).read_text()


def test_options_describe_what_the_analyzer_supports() -> None:

    data = _options()
    assert data["objectives"] == list(OBJECTIVES)
    assert {d["name"] for d in data["dimensions"]}
    assert sum(d["weight"] for d in data["dimensions"]) == 1.0
    assert data["rules"]


def test_web_and_cli_agree_on_the_score() -> None:
    source = CORPUS / "quality" / "deep_nesting.c"
    web = analyze_source(Options(code=source.read_text(),
                                 objectives=("readability",), run=False, ai=False))
    cli = pipeline.analyze(source, quality=True)
    assert web["score"]["overall"] == cli.score.overall


def test_temp_path_never_leaks_to_the_client() -> None:
    result = analyze_source(Options(code=_code("quality/deep_nesting.c"),
                                    objectives=OBJECTIVES, run=False, ai=False))
    assert result["source"] == "submitted.c"
    assert all(d["file"] == "submitted.c" for d in result["diagnostics"])


def test_a_program_that_does_not_compile_has_no_score() -> None:

    result = analyze_source(Options(code="int main(void) { int x = 1 return 0; }",
                                    objectives=OBJECTIVES, run=False, ai=False))
    assert result["compiles"] is False
    assert result["score"] is None
    assert result["diagnostics"]


def test_clean_code_scores_full_marks_over_the_api() -> None:
    result = analyze_source(Options(code=_code("clean.c"), objectives=OBJECTIVES,
                                    run=False, ai=False))
    assert result["score"]["overall"] == 100
    assert result["diagnostics"] == []


def test_bad_requests_are_rejected() -> None:
    assert _parse({}) == "no code submitted"
    assert _parse({"code": "   "}) == "no code submitted"
    assert "unknown objective" in _parse({"code": "int main(void){}",
                                          "objectives": ["faster"]})
    assert _parse({"code": "int main(void){}", "objectives": "speed"}) == \
        "objectives must be a list"


def test_defaults_to_every_objective() -> None:
    parsed = _parse({"code": "int main(void){return 0;}"})
    assert isinstance(parsed, Options)
    assert parsed.objectives == OBJECTIVES
    assert parsed.run is False and parsed.ai is False


def test_report_says_when_runtime_checks_were_off() -> None:

    import json

    from nlpcc.server import Options, analyze_source

    result = analyze_source(Options(code=_code("clean.c"), objectives=OBJECTIVES,
                                    run=True, ai=False))
    assert result["runtime_checks"] is True
    assert json.loads(json.dumps(result))["runtime_checks"] is True
