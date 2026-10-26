
from __future__ import annotations

from nlpcc import improve as improve_module
from nlpcc.improve import Improvement, _diff_lines


def test_a_program_that_does_not_compile_is_refused() -> None:
    result = improve_module.improve("int main(void) { int x = 1 return 0; }")
    assert not result.accepted
    assert "compile" in result.reason


def test_without_a_model_it_says_so(monkeypatch) -> None:

    result = improve_module.improve("int main(void) { return 0; }")
    assert not result.accepted
    assert result.reason


def test_diff_rows_are_classified_for_the_ui() -> None:
    diff = ("--- before.c\n+++ after.c\n@@ -1,2 +1,2 @@\n"
            " int main(void) {\n-    int x;\n+    int count = 0;\n")
    kinds = [row["kind"] for row in _diff_lines(diff)]
    assert kinds == ["meta", "meta", "hunk", "same", "remove", "add"]


def test_score_change_is_reported() -> None:
    result = Improvement(original="x", before={"overall": 60}, after={"overall": 85})
    assert result.score_change == 25


def test_a_worse_rewrite_is_not_offered_as_an_improvement() -> None:

    import nlpcc.pipeline as pipeline_module

    result = Improvement(original="x", improved="y",
                         before={"overall": 90, "compiles": True},
                         after={"overall": 70, "compiles": True})
    assert result.score_change < 0
    assert not result.accepted     



CLEAN = """#include <stdio.h>

int main(void) {
    puts("hello");
    return 0;
}
"""


def _reply_with(monkeypatch, text: str) -> None:
    from nlpcc import models

    monkeypatch.setattr(models, "available",
                        lambda prefer=None: models.Provider("openai-compatible", "fake"))
    monkeypatch.setattr(models, "complete_text", lambda *a, **k: text)


def test_before_is_scored_with_the_reports_options(monkeypatch) -> None:
    """The report said 91 while improve said 100 -> 100: it ran without them."""
    import nlpcc.pipeline as pipeline_module

    seen = []
    real = pipeline_module.analyze

    def analyze(path, **kwargs):
        seen.append(kwargs)
        return real(path, **{**kwargs, "run": False, "ai": False})

    monkeypatch.setattr(pipeline_module, "analyze", analyze)
    _reply_with(monkeypatch, CLEAN.replace("hello", "hello, world"))
    improve_module.improve(CLEAN, run=True, ai=True)

    assert len(seen) == 2       # before and after, analyzed the same way
    assert all(k["run"] and k["ai"] for k in seen)


def test_an_unchanged_rewrite_is_not_accepted(monkeypatch) -> None:
    """An echo of the input scores equal, and equal used to mean accepted."""
    _reply_with(monkeypatch, CLEAN.replace("    ", "\t"))
    result = improve_module.improve(CLEAN)
    assert not result.accepted
    assert "unchanged" in result.reason


def test_a_rewrite_that_crashes_at_run_time_is_rejected(monkeypatch) -> None:
    crashes = """#include <stdio.h>

int main(void) {
    volatile int *nowhere = NULL;
    printf("%d\\n", *nowhere);
    return 0;
}
"""
    _reply_with(monkeypatch, crashes)
    result = improve_module.improve(CLEAN, run=True)
    assert not result.accepted
    assert result.after["runtime_errors"]
    assert "run time" in result.reason


def test_an_unchanged_score_says_whether_findings_went(monkeypatch) -> None:
    _reply_with(monkeypatch, CLEAN.replace("hello", "hi"))
    result = improve_module.improve(CLEAN)
    assert result.accepted
    assert "no findings were removed" in result.reason
