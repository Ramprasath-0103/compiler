

from __future__ import annotations

from conftest import CORPUS
from nlpcc.inspection import inspect_source


def test_the_three_phases_are_reported() -> None:
    data = inspect_source(CORPUS / "clean.c")
    assert [p["name"] for p in data["phases"]] == ["lexical", "syntax", "semantic"]
    assert all(p["ok"] for p in data["phases"])


def test_tokens_are_the_lexer_s_own() -> None:
    data = inspect_source(CORPUS / "clean.c")
    tokens = data["phases"][0]["detail"]["tokens"]
    assert tokens[0]["kind"] == "PREPROCESSOR"
    assert {"KEYWORD", "IDENTIFIER", "PUNCTUATOR"} <= {t["kind"] for t in tokens}
    assert all(t["line"] > 0 for t in tokens)


def test_the_tree_is_the_parser_s_own() -> None:
    tree = inspect_source(CORPUS / "clean.c")["phases"][1]["detail"]["tree"]
    assert tree["kind"] == "TranslationUnit"
    kinds = [child["kind"] for child in tree["children"]]
    assert kinds.count("FunctionDef") == 2        


def test_symbols_come_from_the_analyzer_and_exclude_the_standard_library() -> None:
    symbols = inspect_source(CORPUS / "clean.c")["phases"][2]["detail"]["symbols"]
    names = {s["name"] for s in symbols}
    assert {"add", "main"} <= names
    assert "printf" not in names                  
    assert all(s["line"] > 0 for s in symbols)


def test_a_broken_program_still_reports_every_phase() -> None:

    data = inspect_source(CORPUS / "syntax" / "missing_semicolon.c")
    assert len(data["phases"]) == 3
    assert data["phases"][0]["ok"]               
    syntax = data["phases"][1]
    assert not syntax["ok"]
    assert any(d["code"] == "E_MISSING_SEMICOLON" for d in syntax["diagnostics"])


def test_the_view_is_json_able() -> None:
    import json
    json.dumps(inspect_source(CORPUS / "quality" / "deep_nesting.c"))
