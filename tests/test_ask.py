
from __future__ import annotations

from nlpcc import models
from nlpcc.nlp import route

WORKS = """#include <stdio.h>

int main(void) {
    int total = 0;
    for (int i = 1; i <= 5; i++) {
        total += i;
    }
    printf("The sum is %d\\n", total);
    return 0;
}
"""

BROKEN = WORKS.replace("int total = 0;", "int total = 0")


def _model(monkeypatch, reply: str) -> list[str]:
    prompts: list[str] = []

    def complete_text(system, prompt, **kwargs):
        prompts.append(prompt)
        return reply

    monkeypatch.setattr(models, "available",
                        lambda prefer=None: models.Provider("openai-compatible", "fake"))
    monkeypatch.setattr(models, "complete_text", complete_text)
    return prompts


def test_a_question_about_the_code_is_answered_not_generated(monkeypatch) -> None:
    prompts = _model(monkeypatch, "It adds 1 to 5 and prints `The sum is 15`.")
    result = route.run("what does this code do", code=WORKS)

    assert "generated" not in result and "edit" not in result
    assert result["answer"]["text"].startswith("It adds")
    assert "The sum is 15" in prompts[0]
    assert "compiles with no errors" in prompts[0]


def test_a_question_without_code_is_answered(monkeypatch) -> None:
    prompts = _model(monkeypatch, "A pointer holds the address of a variable.")
    result = route.run("what is a pointer", code="")
    assert result["answer"]["text"]
    assert "no program" in prompts[0]


def test_without_a_model_a_question_still_gets_the_analysis() -> None:
    result = route.run("is this correct?", code=BROKEN)
    assert result["answer"]["error"]
    assert result["report"]["summary"]["errors"] >= 1


def test_fix_it_edits_code_that_does_not_compile(monkeypatch) -> None:
    _model(monkeypatch, WORKS)
    result = route.run("fix it", code=BROKEN)
    edit = result["edit"]
    assert edit["accepted"]
    assert "compiles now" in edit["reason"]
    assert "+    int total = 0;" in edit["diff"]


def test_an_edit_that_still_does_not_compile_is_not_offered(monkeypatch) -> None:
    prompts = _model(monkeypatch, BROKEN.replace("total += i;", "total += i"))
    edit = route.run("add comments", code=WORKS)["edit"]
    assert not edit["accepted"]
    assert len(prompts) == 2              
    assert "does not compile" in prompts[1]


def test_an_unchanged_edit_is_not_offered(monkeypatch) -> None:
    _model(monkeypatch, WORKS)
    edit = route.run("add comments", code=WORKS)["edit"]
    assert not edit["accepted"]
    assert "unchanged" in edit["reason"]


def test_fix_it_with_no_code_asks_for_code() -> None:
    result = route.run("fix it", code="")
    assert "Paste your code" in result["error"]


def test_explain_on_working_code_explains_the_code(monkeypatch) -> None:
    _model(monkeypatch, "It sums 1 to 5.")
    result = route.run("explain", code=WORKS)
    assert result["answer"]["text"] == "It sums 1 to 5."
    assert result["intent"]["action"] == "question"    


def test_an_edit_that_fixes_a_crash_says_so(monkeypatch) -> None:
    overrun = WORKS.replace("int total = 0;", "int total = 0;\n    int nums[3] = {1, 2, 3};") \
                   .replace("total += i;", "total += nums[i];")
    fixed = overrun.replace("i <= 5", "i < 3")
    _model(monkeypatch, fixed)
    edit = route.run("fix the error", code=overrun)["edit"]
    assert edit["accepted"]
    assert edit["before"]["runtime_errors"] and not edit["after"]["runtime_errors"]
    assert "no longer fails when run" in edit["reason"]


def test_an_edit_that_crashes_is_not_offered(monkeypatch) -> None:
    crashes = WORKS.replace("int total = 0;", "int total = 0;\n    int *none = 0;") \
                   .replace("return 0;", "return *none;")
    _model(monkeypatch, crashes)
    edit = route.run("add comments", code=WORKS)["edit"]
    assert not edit["accepted"]
    assert "fails at run time" in edit["reason"]


def test_the_diff_does_not_end_in_a_newline_only_change(monkeypatch) -> None:
    _model(monkeypatch, WORKS.replace("int total = 0;", "int total = 0;  // running sum").rstrip())
    edit = route.run("add comments", code=WORKS)["edit"]
    assert "-}" not in edit["diff"]
