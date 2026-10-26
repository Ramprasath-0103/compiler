
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlpcc import pipeline                                     
from nlpcc.diagnostics import Diagnostic, Origin, Severity     
from nlpcc.explain import rules                                 
from nlpcc.frontend import FRONTEND_CODES                       
from nlpcc.frontend.lexer import Lexer                         
from nlpcc.frontend.parser import Parser                        
from nlpcc.frontend.semantic import SemanticAnalyzer            
from nlpcc.frontend.tokens import TokenKind                     

CORPUS = Path(__file__).parent / "corpus"
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "nlpcc" / "frontend"
CASES = sorted(CORPUS.glob("*/*.c"))


def _case_id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def analyze_with_frontend(source: Path) -> list[Diagnostic]:

    text = source.read_text()
    lexer = Lexer(text, str(source))
    tokens = lexer.tokenize()
    parser = Parser(tokens, str(source), text)
    unit = parser.parse()
    analyzer = SemanticAnalyzer(unit, str(source), text)
    return [*lexer.diagnostics, *parser.diagnostics, *analyzer.analyze()]


def frontend_only(diags: list[Diagnostic]) -> list[Diagnostic]:
    return [d for d in diags if d.origin is Origin.FRONTEND]


def lines_of(diags: list[Diagnostic]) -> set[int | None]:
    return {d.position.line if d.position else None for d in diags}

OWN_DETECTIONS: dict[str, tuple[str, int]] = {
    "syntax/missing_semicolon": ("E_MISSING_SEMICOLON", 4),
    "syntax/missing_paren": ("E_MISSING_PAREN", 5),
    "syntax/unbalanced_brace": ("E_MISSING_BRACE", 7),
    "syntax/unterminated_string": ("E_UNTERMINATED_STRING", 4),
    "semantic/undeclared_variable": ("E_UNDECLARED_IDENTIFIER", 5),
    "semantic/out_of_scope": ("E_UNDECLARED_IDENTIFIER", 7),
    "semantic/type_mismatch": ("E_TYPE_MISMATCH", 4),
    "semantic/wrong_arg_count": ("E_WRONG_ARG_COUNT", 8),
    "semantic/uninitialized_read": ("W_UNINITIALIZED_READ", 5),
    "semantic/missing_return": ("W_MISSING_RETURN", 3),
    "subset/struct_outside_subset": ("W_OUTSIDE_SUBSET", 3),
}


@pytest.mark.parametrize("case_id", sorted(OWN_DETECTIONS))
def test_frontend_detects_the_case_itself(case_id: str) -> None:
    code, line = OWN_DETECTIONS[case_id]
    source = CORPUS / f"{case_id}.c"
    found = analyze_with_frontend(source)
    assert any(d.code == code and d.position and d.position.line == line for d in found), (
        f"our front end did not report {code} at line {line}; it reported:\n  "
        + "\n  ".join(f"{d.position} {d.severity.value} {d.code}" for d in found))


@pytest.mark.parametrize("case_id", sorted(OWN_DETECTIONS))
def test_the_finding_reaches_the_report_as_ours(case_id: str) -> None:
    code, line = OWN_DETECTIONS[case_id]
    result = pipeline.analyze(CORPUS / f"{case_id}.c")
    assert result.frontend_available
    ours = [d for d in result.diagnostics
            if d.origin is Origin.FRONTEND and d.code == code]
    assert ours, f"{code} is missing from the report or is not ours"
    assert any(d.position and d.position.line == line for d in ours)
    assert all(d.explanation and d.fix for d in ours), "our finding reached the user unexplained"


def test_declaration_sites_travel_with_the_finding() -> None:
    found = analyze_with_frontend(CORPUS / "semantic" / "wrong_arg_count.c")
    call = next(d for d in found if d.code == "E_WRONG_ARG_COUNT")
    related = call.meta.get("related")
    assert related, "a wrong-argument-count finding must point at the declaration"
    assert related[0]["line"] == 3 and "add" in related[0]["snippet"]

    found = analyze_with_frontend(CORPUS / "semantic" / "uninitialized_read.c")
    read = next(d for d in found if d.code == "W_UNINITIALIZED_READ")
    assert read.meta["related"][0]["line"] == 4


LEGAL_PROGRAMS = ["clean.c", "quality/deep_nesting.c", "quality/poor_naming.c",
                  "runtime/array_overrun.c", "runtime/divide_by_zero.c",
                  "runtime/infinite_loop.c", "runtime/null_dereference.c",
                  "security/unchecked_malloc.c", "security/unsafe_strcpy.c",
                  "semantic/format_mismatch.c"]


@pytest.mark.parametrize("name", LEGAL_PROGRAMS)
def test_frontend_is_silent_on_legal_programs(name: str) -> None:
    found = analyze_with_frontend(CORPUS / name)
    assert not found, (
        "our front end complained about a program that is legal C:\n  "
        + "\n  ".join(str(d) for d in found))


def test_one_mistake_is_one_finding() -> None:
    source = """int main(void) {
    int a = 1;
    int b = 2
    int c = 3;
    int d = a + b;
    int e = c + d;
    if (e > 0) {
        e = e + 1;
    }
    while (e > 10) {
        e = e - 1;
    }
    return e;
}
"""
    lexer = Lexer(source, "<cascade>")
    parser = Parser(lexer.tokenize(), "<cascade>", source)
    parser.parse()
    found = [*lexer.diagnostics, *parser.diagnostics]
    assert [d.code for d in found] == ["E_MISSING_SEMICOLON"], [str(d) for d in found]
    assert found[0].position.line == 3


KNOWN_GAPS: dict[str, dict[int, str]] = {

    "semantic/format_mismatch": {5: "step 6: -Wformat, format string vs arguments"},

    "semantic/missing_return": {2: "step 6: -Wunused-variable"},
    "semantic/out_of_scope": {5: "step 6: -Wunused-variable"},

    "syntax/unterminated_string": {6: "recovery divergence: clang's cascade after the bad line"},
    "subset/struct_outside_subset": {12: "by design: struct is outside the subset, clang covers it"},
}


@pytest.mark.parametrize("source", CASES, ids=_case_id)
def test_gap_against_clang_is_the_declared_one(source: Path) -> None:
    case_id = _case_id(source)
    result = pipeline.analyze(source, run=False)
    clang_lines = lines_of([d for d in result.diagnostics if d.origin is Origin.CLANG])
    our_lines = lines_of(frontend_only(result.diagnostics))
    gap = {line for line in clang_lines - our_lines if line is not None}
    declared = set(KNOWN_GAPS.get(case_id, {}))
    assert gap == declared, (
        f"{case_id}: the gap against clang changed.\n"
        f"  clang reports lines {sorted(l for l in clang_lines if l)}\n"
        f"  we report lines      {sorted(l for l in our_lines if l)}\n"
        f"  newly missed:        {sorted(gap - declared)}\n"
        f"  newly covered:       {sorted(declared - gap)}  <- update KNOWN_GAPS")


@pytest.mark.parametrize("source", CASES, ids=_case_id)
def test_no_hard_error_is_missed(source: Path) -> None:

    case_id = _case_id(source)
    result = pipeline.analyze(source, run=False)
    hard = {d.position.line for d in result.diagnostics
            if d.origin is Origin.CLANG and d.position
            and d.severity in (Severity.ERROR, Severity.FATAL)}
    ours = {line for line in lines_of(frontend_only(result.diagnostics)) if line}
    missed = hard - ours - set(KNOWN_GAPS.get(case_id, {}))
    assert not missed, f"{case_id}: clang reports an error at {sorted(missed)} and we do not"



def _codes_in_source() -> set[str]:

    found: set[str] = set()
    for path in FRONTEND_DIR.glob("*.py"):
        found |= set(re.findall(r'"([EW]_[A-Z0-9_]+)"', path.read_text()))
    return found


def test_every_frontend_code_has_a_rule() -> None:
    explained = {rule.code for rule in rules.RULES if rule.code}
    missing = sorted(set(FRONTEND_CODES) - explained)
    assert not missing, (
        "these codes have no explanation rule in nlpcc/explain/rules.py: "
        + ", ".join(missing))


def test_the_code_registry_matches_the_source() -> None:

    in_source = _codes_in_source()
    registered = set(FRONTEND_CODES)
    assert in_source - registered == set(), (
        "emitted but not in nlpcc.frontend.FRONTEND_CODES: "
        + ", ".join(sorted(in_source - registered)))
    assert registered - in_source == set(), (
        "registered in FRONTEND_CODES but never emitted: "
        + ", ".join(sorted(registered - in_source)))


@pytest.mark.parametrize("source", CASES, ids=_case_id)
def test_our_findings_are_all_explained(source: Path) -> None:
    for diag in frontend_only(pipeline.analyze(source).diagnostics):
        assert diag.meta.get("explained_by") == "rules", (
            f"{diag.code} reached the user as raw text: {diag.message}")



def test_lexer_tracks_line_and_column() -> None:
    tokens = Lexer("int x;\n  x = 42;\n").tokenize()
    positions = [(t.text, t.line, t.column) for t in tokens if t.kind is not TokenKind.EOF]
    assert positions == [("int", 1, 1), ("x", 1, 5), (";", 1, 6),
                         ("x", 2, 3), ("=", 2, 5), ("42", 2, 7), (";", 2, 9)]


def test_lexer_keeps_comments() -> None:

    tokens = Lexer("// explain\nint x; /* and\n more */\n").tokenize()
    comments = [t for t in tokens if t.kind is TokenKind.COMMENT]
    assert [t.text for t in comments] == ["// explain", "/* and\n more */"]
    assert comments[1].line == 2


def test_lexer_recovers_rather_than_raising() -> None:
    lexer = Lexer('int x = `1;\nchar c = \'a;\nchar *s = "open\n')
    tokens = lexer.tokenize()
    codes = [d.code for d in lexer.diagnostics]
    assert codes == ["E_STRAY_CHARACTER", "E_UNTERMINATED_CHAR", "E_UNTERMINATED_STRING"]
    assert tokens[-1].kind is TokenKind.EOF          # it still produced a stream
    assert all(d.position.line for d in lexer.diagnostics)


def test_preprocessor_directives_are_recorded_not_expanded() -> None:
    tokens = Lexer('#include <stdio.h>\n#define N 4\nint x;\n').tokenize()
    directives = [t.text for t in tokens if t.kind is TokenKind.PREPROCESSOR]
    assert directives == ["#include <stdio.h>", "#define N 4"]




def _frontend_codes_for(source: str) -> list[str]:
    lexer = Lexer(source, "<inline>")
    parser = Parser(lexer.tokenize(), "<inline>", source)
    unit = parser.parse()
    analyzer = SemanticAnalyzer(unit, "<inline>", source)
    return [d.code for d in (*lexer.diagnostics, *parser.diagnostics, *analyzer.analyze())]


@pytest.mark.parametrize("source,expected", [
    ("int main(void) { int a = 1; int a = 2; return a; }", ["E_REDECLARATION"]),
    ("int main(void) { break; return 0; }", ["E_BREAK_OUTSIDE_LOOP"]),
    ("int main(void) { void v; return 0; }", ["E_VOID_VARIABLE"]),
    ("int n = 1;\nint main(void) { return n(); }", ["E_NOT_A_FUNCTION"]),
    ("int main(void) { int a = 1; int *p = a; return 0; }", ["E_TYPE_MISMATCH"]),
    ("int main(void) { int x = 1 ? 2 3; return x; }", ["E_MISSING_COLON"]),
    ("int main(void) { int a[2]; a[0 = 1; return 0; }", ["E_MISSING_BRACKET"]),
    ("int main(void) { int x = ; return x; }", ["E_EXPECTED_EXPRESSION"]),
    ("int main(void) { 2x = 1; return 0; }", ["E_MALFORMED_NUMBER"]),
    ("int main(void) { /* never closed\n return 0; }",
     ["E_UNTERMINATED_COMMENT", "E_MISSING_BRACE"]),
    ("int main(void) { int p = 0; int *q = &p; return *q; }", []),
    ("int f(void);\nint f(void) { return 1; }\nint main(void) { return f(); }", []),
    ("int main(void) { int i; for (i = 0; i < 3; i++) { continue; } return i; }", []),
    ("int main(void) { int n; scanf(\"%d\", &n); return n; }", []),
    ("int main(void) { int a = 1, b = 2; return a + b; }", []),
    ("int main(void) { double d = 1; int i = d; char c = 'x'; return i + c; }", []),
], ids=lambda value: value if isinstance(value, list) else value[:40])
def test_checks_that_the_corpus_does_not_reach(source: str, expected: list[str]) -> None:
    assert _frontend_codes_for(source) == expected


@pytest.mark.parametrize("source", [
    "int main(void) { switch (1) { case 1: return 1; } return 0; }",
    "struct point { int x; };\nint main(void) { return 0; }",
    "typedef int myint;\nint main(void) { return 0; }",
    "int main(void) { int i = 0; do { i++; } while (i < 3); return i; }",
    "int main(void) { int grid[2][2]; return grid[0][0]; }",
], ids=["switch", "struct", "typedef", "do-while", "2d-array"])
def test_outside_the_subset_is_one_note_and_no_errors(source: str) -> None:
    codes = _frontend_codes_for(source)
    assert codes.count("W_OUTSIDE_SUBSET") == 1, codes
    assert all(code.startswith("W_") for code in codes), (
        f"a construct outside the subset produced errors, not a note: {codes}")


def test_unparsed_constructs_silence_undeclared_reporting() -> None:
    codes = _frontend_codes_for(
        "struct point { int x; };\n"
        "int main(void) { struct point p; p.x = 1; return p.x + whatever; }")
    assert "E_UNDECLARED_IDENTIFIER" not in codes


def test_unknown_header_silences_undeclared_reporting() -> None:
    codes = _frontend_codes_for('#include "mine.h"\nint main(void) { return helper(); }')
    assert "E_UNDECLARED_IDENTIFIER" not in codes
    codes = _frontend_codes_for('#include <stdio.h>\nint main(void) { return helper(); }')
    assert codes == ["E_UNDECLARED_IDENTIFIER"]


def test_expected_json_exists_for_every_case() -> None:
    missing = [_case_id(c) for c in CASES if not c.with_suffix(".expected.json").exists()]
    assert not missing, missing
    for case in CASES:                       
        json.loads(case.with_suffix(".expected.json").read_text())



def test_valid_c_outside_the_subset_is_a_note_not_an_error(tmp_path) -> None:

    from nlpcc import pipeline
    from nlpcc.diagnostics import Severity

    source = tmp_path / "subset.c"
    source.write_text(
        "#include <stdio.h>\n"
        "int main(void) {\n"
        "    for (size_t i = 0; i < 3; i++) { printf(\"%zu\", i); }\n"
        "    return 0;\n"
        "}\n")

    result = pipeline.analyze(source)
    assert result.subset_limited
    assert not [d for d in result.diagnostics
                if d.severity in (Severity.ERROR, Severity.FATAL)]
    notes = [d for d in result.diagnostics if d.code == "W_OUTSIDE_SUBSET"]
    assert len(notes) == 1
    assert notes[0].meta["demoted_from"]       


def test_real_errors_are_not_demoted(tmp_path) -> None:
    from nlpcc import pipeline
    from nlpcc.diagnostics import Severity

    source = tmp_path / "broken.c"
    source.write_text("int main(void) {\n    int x = 1\n    return 0;\n}\n")

    result = pipeline.analyze(source)
    assert not result.subset_limited
    assert [d for d in result.diagnostics
            if d.severity in (Severity.ERROR, Severity.FATAL)]


def _explained(tmp_path, code: str):
    from nlpcc import pipeline

    path = tmp_path / "x.c"
    path.write_text(code)
    return pipeline.analyze(path).diagnostics


def test_a_missing_semicolon_shows_the_fixed_line(tmp_path) -> None:
    diags = _explained(tmp_path, "int main(void) {\n    int count = 0\n    return count;\n}\n")
    semi = next(d for d in diags if d.code == "E_MISSING_SEMICOLON")
    assert "`int count = 0`" in semi.explanation
    assert semi.fix == "Make it `int count = 0;`."


def test_a_typo_gets_a_did_you_mean(tmp_path) -> None:
    diags = _explained(tmp_path, "int main(void) {\n    int total = 1;\n    return totl;\n}\n")
    unknown = next(d for d in diags if "totl" in d.message)
    assert "Did you mean `total`?" in unknown.explanation
    assert unknown.fix == "Change `totl` to `total`."


def test_a_variable_out_of_scope_says_where_it_lived(tmp_path) -> None:
    code = ("int main(void) {\n    for (int i = 0; i < 3; i++) {\n"
            "        int inner = i;\n    }\n    return inner;\n}\n")
    unknown = next(d for d in _explained(tmp_path, code) if "inner" in d.message
                   and d.severity.value == "error")
    assert "declared on line 3" in unknown.explanation
    assert "already closed" in unknown.explanation


def test_string_is_explained_as_not_c(tmp_path) -> None:
    code = '#include <stdio.h>\nint main(void) {\n    string name = "bob";\n    return 0;\n}\n'
    diag = next(d for d in _explained(tmp_path, code) if "'string'" in d.message)
    assert "no string type" in diag.explanation
    assert "char name[]" in diag.fix


def test_a_missing_include_names_the_header(tmp_path) -> None:
    diags = _explained(tmp_path, 'int main(void) {\n    printf("hi\\n");\n    return 0;\n}\n')
    diag = next(d for d in diags if "printf" in d.message)
    assert diag.fix == "Add `#include <stdio.h>` at the very top of the file."


def test_scanf_without_an_ampersand_names_the_variable(tmp_path) -> None:
    code = ('#include <stdio.h>\nint main(void) {\n    int total = 0;\n'
            '    scanf("%d", total);\n    return total;\n}\n')
    diag = next(d for d in _explained(tmp_path, code) if d.code == "-Wformat")
    assert diag.fix == "Write `&total` instead of `total`."


def test_a_template_with_missing_details_is_not_printed_half_filled() -> None:
    from nlpcc.diagnostics import Diagnostic, Phase, Severity
    from nlpcc.explain.rules import explain

    diag = explain(Diagnostic(phase=Phase.SEMANTIC, severity=Severity.ERROR,
                              message="something undeclared", file="nowhere.c",
                              code="E_UNDECLARED_IDENTIFIER"))
    assert "{" not in diag.explanation.replace("{ }", "")
    assert "{" not in diag.fix.replace("{ }", "")
