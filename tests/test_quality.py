
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import CORPUS
from nlpcc.diagnostics import Phase, Severity
from nlpcc.quality import astjson, metrics, rules, score


def test_includes_are_excluded() -> None:
  
    funcs = astjson.functions(CORPUS / "clean.c")
    assert sorted(f.name for f in funcs) == ["add", "main"]


def test_locations_are_resolved_per_statement() -> None:

    source = CORPUS / "security" / "unchecked_malloc.c"
    body = astjson.body_of(astjson.functions(source)[0])
    lines = {n.kind: n.line for n in body.walk()}
    assert lines["VarDecl"] == 4
    assert lines["ArraySubscriptExpr"] == 5
    assert lines["ReturnStmt"] == 6


def test_callee_name_is_the_function_not_an_argument() -> None:
    source = CORPUS / "security" / "unchecked_malloc.c"
    fn = metrics.measure(source).functions[0]
    assert fn.calls == ["malloc"]       


def test_fortified_builtins_resolve_to_the_written_name() -> None:

    assert metrics._normalize_call("__builtin___strcpy_chk") == "strcpy"
    assert metrics._normalize_call("__builtin___sprintf_chk") == "sprintf"
    assert metrics._normalize_call("printf") == "printf"



def test_metrics_match_hand_counts() -> None:
    fns = {f.name: f for f in metrics.measure(CORPUS / "quality" / "deep_nesting.c").functions}
    classify = fns["classify"]
    assert classify.parameters == 4
    assert classify.max_nesting == 5         
    assert classify.complexity == 6          
    assert (classify.line, classify.end_line) == (3, 16)


def test_comment_density_ignores_trailing_comments() -> None:

    total, code, comment, blank = metrics._count_lines(
        "int x = 1; // set x\n// a real comment\n\n/* block */\nint y = 2;\n")

    assert (code, comment, blank) == (2, 2, 1)
    assert total == 5



def test_correct_code_produces_no_findings() -> None:

    assert rules.analyze(CORPUS / "clean.c") == []


@pytest.mark.parametrize("case,expected_code", [
    ("quality/deep_nesting.c", "R_DEEP_NESTING"),
    ("quality/poor_naming.c", "R_OPAQUE_NAMES"),
    ("security/unchecked_malloc.c", "S_UNCHECKED_ALLOCATION"),
    ("security/unchecked_malloc.c", "S_LEAKED_ALLOCATION"),
    ("security/unsafe_strcpy.c", "S_UNSAFE_FUNCTION"),
])
def test_rule_fires_on_its_case(case: str, expected_code: str) -> None:

    found = rules.analyze(CORPUS / case)
    assert expected_code in {d.code for d in found}, [d.code for d in found]


def test_findings_arrive_already_explained() -> None:

    for diag in rules.analyze(CORPUS / "security" / "unchecked_malloc.c"):
        assert diag.phase is Phase.QUALITY
        assert diag.explanation and diag.fix
        assert diag.meta["title"]
        assert diag.meta["dimension"] in score.DIMENSIONS


def test_two_parameter_helper_is_not_flagged_for_naming() -> None:

    codes = {d.code for d in rules.analyze(CORPUS / "clean.c")}
    assert "R_OPAQUE_NAMES" not in codes


def test_short_file_is_not_asked_for_comments() -> None:
    codes = {d.code for d in rules.analyze(CORPUS / "clean.c")}
    assert "R_NO_COMMENTS" not in codes


def test_every_rule_has_a_weightable_dimension() -> None:
    for code, rule in rules.RULES.items():
        assert rule.dimension in score.WEIGHTS, code
        assert rule.penalty > 0, code
        assert rule.explanation and rule.fix, code


def test_clean_code_scores_full_marks() -> None:
    assert score.compute(rules.analyze(CORPUS / "clean.c")).overall == 100


def test_weights_sum_to_one() -> None:
    assert sum(score.WEIGHTS.values()) == pytest.approx(1.0)


def test_critical_security_finding_caps_the_overall_score() -> None:

    result = score.compute(rules.analyze(CORPUS / "security" / "unsafe_strcpy.c"))
    assert result.capped_by_security
    assert result.overall == score.CRITICAL_SECURITY_CAP

    assert result.dimensions["readability"] == 100


def test_one_bad_dimension_is_not_averaged_away() -> None:
    result = score.compute(rules.analyze(CORPUS / "quality" / "deep_nesting.c"))
    assert result.limited_by == "readability"
    assert result.overall <= min(result.dimensions.values()) + score.WEAKEST_LINK_MARGIN
    assert result.grade != "excellent"


def test_every_deduction_is_traceable() -> None:
    result = score.compute(rules.analyze(CORPUS / "quality" / "deep_nesting.c"))
    readability = result.for_dimension("readability")
    assert sum(c.penalty for c in readability) == 100 - result.dimensions["readability"]
    for contribution in readability:
        assert contribution.code in rules.RULES
        assert contribution.detail


def test_compiler_errors_cost_the_practices_dimension() -> None:

    from nlpcc import pipeline
    analysis = pipeline.analyze(CORPUS / "semantic" / "undeclared_variable.c")
    result = score.compute(analysis.diagnostics)
    assert result.dimensions["practices"] < 100


def test_scores_stay_in_range() -> None:
    for case in sorted(CORPUS.glob("*/*.c")):
        from nlpcc import pipeline
        analysis = pipeline.analyze(case, quality=True)
        if analysis.score is None:
            continue
        assert 0 <= analysis.score.overall <= 100, case
        for name, value in analysis.score.dimensions.items():
            assert 0 <= value <= 100, (case, name)


def test_ai_findings_are_scored_by_their_own_dimension() -> None:

    from nlpcc.diagnostics import Diagnostic, Origin, Phase, Position, Severity

    finding = Diagnostic(
        phase=Phase.QUALITY, severity=Severity.WARNING,
        message="a problem no threshold describes", code="AI_SECURITY",
        position=Position(4), origin=Origin.AI,
        explanation="...", fix="...",
        meta={"dimension": "security", "penalty": 25, "explained_by": "ai"})

    result = score.compute([finding])
    assert result.dimensions["security"] == 75
    assert result.for_dimension("security")[0].code == "AI_SECURITY"


def test_ai_analysis_is_silent_without_a_model() -> None:

    from nlpcc.quality import ai

    assert ai.analyze(CORPUS / "clean.c") == []


def test_ai_stage_adds_to_the_rules_without_changing_them(monkeypatch) -> None:

    from nlpcc import pipeline
    from nlpcc.diagnostics import Diagnostic, Origin, Phase, Position, Severity
    from nlpcc.quality import ai

    invented = Diagnostic(
        phase=Phase.QUALITY, severity=Severity.WARNING,
        message="something only a model would notice", code="AI_SECURITY",
        position=Position(3), origin=Origin.AI, explanation="...", fix="...",
        meta={"dimension": "security", "penalty": 15, "explained_by": "ai",
              "generated": True})
    monkeypatch.setattr(ai, "analyze", lambda source: [invented])

    case = CORPUS / "quality" / "deep_nesting.c"
    with_ai = pipeline.analyze(case, quality=True, ai=True)
    without = pipeline.analyze(case, quality=True)

    def rule_codes(analysis):
        return sorted(d.code for d in analysis.diagnostics
                      if d.origin.value == "rules")

    assert rule_codes(with_ai) == rule_codes(without)
    assert "AI_SECURITY" in {d.code for d in with_ai.diagnostics}
    assert "AI_SECURITY" not in {d.code for d in without.diagnostics}
    assert with_ai.score.dimensions["security"] == 85



def test_diagnostic_flags_match_the_compiler() -> None:

    from nlpcc.backends import clang_runner as runner

    assert "-ferror-limit=0" in runner.CLANG_FLAGS
    assert "-fmax-errors=0" in runner.GCC_FLAGS
    assert not any(f.startswith("-ferror-limit") for f in runner.GCC_FLAGS)
    assert not any(f.startswith("-fmax-errors") for f in runner.CLANG_FLAGS)

    assert runner.CC_CANDIDATES[0] == "clang"
    assert runner.CC_CANDIDATES.index("clang") < runner.CC_CANDIDATES.index("gcc")


def test_a_sanitizer_that_cannot_link_does_not_fail_the_build(monkeypatch) -> None:

    from nlpcc.backends import executor

    monkeypatch.setattr(executor, "SANITIZE_FLAGS",
                        ("-fsanitize=undefined", "-Wl,--nonexistent-option-xyz"))
    result = executor.compile_and_run(CORPUS / "clean.c")

    assert result.compiled is True         
    assert result.sanitized is False        
    assert result.stdout.strip() == "7"


def test_runtime_measurements_describe_the_program_not_the_compiler() -> None:

    from nlpcc.backends import executor

    result = executor.compile_and_run(CORPUS / "clean.c")
    m = result.measurements()

    assert result.stdout.strip() == "7"
    assert 0 < m["peak_memory_kb"] < 20_000     
    assert m["wall_ms"] > 0
    assert m["build_ms"] > 0
    assert m["exit_code"] == 0
