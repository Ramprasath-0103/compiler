
from __future__ import annotations

import json

from conftest import CORPUS
from nlpcc import generate, pipeline
from nlpcc.diagnostics import Diagnostic, Phase, Position, Severity
from nlpcc.explain import llm


def _unexplained() -> Diagnostic:
    return Diagnostic(phase=Phase.SYNTAX, severity=Severity.ERROR,
                      message="something the rules have never seen",
                      file="x.c", position=Position(3, 1), code="clang:novel_thing")


def test_offline_mode_leaves_diagnostics_untouched() -> None:
    diag = _unexplained()
    outcome = llm.explain_unmatched([diag], allow_network=False)
    assert not outcome.available
    assert outcome.attempted == 1
    assert diag.explanation == ""      


def test_rule_explained_findings_are_never_sent_to_the_model() -> None:
    analysis = pipeline.analyze(CORPUS / "syntax" / "missing_semicolon.c")
    outcome = llm.explain_unmatched(analysis.diagnostics, allow_network=False)
    assert outcome.attempted == 0


def test_cache_round_trip(tmp_path) -> None:
    diag = _unexplained()
    answer = llm.ModelExplanation("A novel problem", "The long explanation.", "Do this.")
    llm._write_cache(llm._cache_key(diag), answer, tmp_path)

    outcome = llm.explain_unmatched([diag], cache_dir=tmp_path, allow_network=False)
    assert outcome.explained == 1 and outcome.from_cache == 1
    assert diag.explanation == "The long explanation."
    assert diag.meta["explained_by"] == "llm-cache"
    assert diag.meta["generated"] is True       


def test_cache_key_ignores_program_specific_names() -> None:
    a = Diagnostic(phase=Phase.SEMANTIC, severity=Severity.ERROR,
                   code="clang:undeclared", message="use of undeclared identifier 'totl'")
    b = Diagnostic(phase=Phase.SEMANTIC, severity=Severity.ERROR,
                   code="clang:undeclared", message="use of undeclared identifier 'xyz'")
    assert llm._cache_key(a) == llm._cache_key(b)


def test_model_output_is_labelled_as_generated() -> None:
    diag = _unexplained()
    llm._apply(diag, llm.ModelExplanation("T", "E", "F"), source="llm")
    assert diag.meta["explained_by"] == "llm"
    assert diag.meta["generated"] is True


def test_markdown_fences_are_stripped() -> None:
    assert generate._strip_fence("```c\nint main(void){return 0;}\n```") == \
        "int main(void){return 0;}"
    assert generate._strip_fence("int main(void){return 0;}") == \
        "int main(void){return 0;}"


def test_generation_without_a_model_fails_honestly() -> None:

    import nlpcc.generate as module

    result = module.generate("a program that prints hello")

    assert result.accepted is False
    assert result.code == ""
    assert result.failure_reason
    assert json.loads(json.dumps(result.to_dict()))["accepted"] is False


def test_rejected_program_is_not_presented_as_working() -> None:
    from nlpcc.diagnostics import Origin

    attempt = generate.Attempt(
        code="int main(void){",
        diagnostics=[Diagnostic(phase=Phase.SYNTAX, severity=Severity.ERROR,
                                message="expected '}'", code="clang:expected_rbrace",
                                origin=Origin.CLANG)])
    assert attempt.errors


def test_generated_code_is_judged_by_the_compiler_not_our_subset() -> None:

    from nlpcc.diagnostics import Origin

    attempt = generate.Attempt(
        code="int main(void){ size_t n = 0; return (int)n; }",
        diagnostics=[Diagnostic(phase=Phase.SYNTAX, severity=Severity.ERROR,
                                message="expected ';'", code="E_MISSING_SEMICOLON",
                                origin=Origin.FRONTEND)])
    assert attempt.errors == []


def test_code_is_recovered_from_whatever_wraps_it() -> None:

    expected = "#include <stdio.h>\nint main(void) { return 0; }"

    assert generate._clean(expected) == expected
    assert generate._clean(f"```c\n{expected}\n```") == expected
    assert generate._clean(f"Here you go:\n\n{expected}") == expected
    assert generate._clean(f"<think>planning</think>\n{expected}") == expected
    assert generate._clean(f"Sure!\n```\n{expected}\n```\nHope that helps") \
        .startswith("#include")


def test_failures_say_which_failure_it_was() -> None:
    from nlpcc import models

    reasons: list[str] = []
    models.complete_text("sys", "prompt", reasons=reasons)
    assert reasons == ["no model available"]


def test_the_same_program_gets_the_same_ai_review(monkeypatch, tmp_path) -> None:

    from nlpcc import models
    from nlpcc.quality import ai

    monkeypatch.setattr(ai, "_REVIEWS", {})
    monkeypatch.setattr(models, "available",
                        lambda prefer=None: models.Provider("openai-compatible", "fake"))
    calls = []

    def complete(system, prompt, schema, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return None          
        finding = {"dimension": "security", "line": 1, "title": "t",
                   "explanation": "e", "fix": "f", "penalty": 9}
        return schema.model_validate({"findings": [finding]})

    monkeypatch.setattr(models, "complete", complete)
    code = "int main(void) { return 0; }\n"
    first, second, third = (tmp_path / n for n in ("a.c", "b.c", "c.c"))
    for path in (first, second, third):
        path.write_text(code)

    assert ai.analyze(first) == []
    assert len(ai.analyze(second)) == 1
    assert [d.file for d in ai.analyze(third)] == [str(third)]
    assert len(calls) == 2
