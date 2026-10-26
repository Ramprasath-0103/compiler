

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlpcc import pipeline                      
from nlpcc.diagnostics import Diagnostic        
from nlpcc.explain import rules                

CORPUS = Path(__file__).parent / "corpus"
CASES = sorted(CORPUS.glob("*/*.c"))

assert CASES, f"no corpus cases found under {CORPUS}"


def _case_id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def _load(path: Path) -> dict:
    expected = path.with_suffix(".expected.json")
    if not expected.exists():
        pytest.fail(f"{_case_id(path)} has no .expected.json -- every case needs one")
    return json.loads(expected.read_text())


def _matches(diag: Diagnostic, want: dict) -> bool:
    line = diag.position.line if diag.position else None
    if want.get("line") is not None and line != want["line"]:
        return False
    if diag.severity.value != want["severity"]:
        return False
    if want.get("phase") and diag.phase.value != want["phase"]:
        return False
    if want.get("code") and diag.code != want["code"]:
        return False
    return True


@pytest.mark.parametrize("source", CASES, ids=_case_id)
def test_expected_findings_are_reported(source: Path) -> None:
    case = _load(source)
    if case.get("pending"):
        pytest.xfail(case["pending"])

    result = pipeline.analyze(source, run=case.get("run", False),
                              quality=case.get("quality", False))
    got = result.diagnostics

    for want in case["expect"]:
        assert any(_matches(d, want) for d in got), (
            f"{case['title']}: expected {want['severity']} at line {want['line']} "
            f"(code {want.get('code')}), got:\n  "
            + "\n  ".join(f"{d.position} {d.severity.value} {d.code}" for d in got)
        )


@pytest.mark.parametrize("source", CASES, ids=_case_id)
def test_every_finding_is_explained(source: Path) -> None:
    case = _load(source)
    result = pipeline.analyze(source, run=case.get("run", False),
                              quality=case.get("quality", False))

    gaps = rules.unexplained(result.diagnostics)
    assert not gaps, (
        "these findings have no explanation rule -- add one to nlpcc/explain/rules.py:\n  "
        + "\n  ".join(f"{d.code}: {d.message}" for d in gaps)
    )


def test_clean_program_reports_nothing() -> None:
    clean = CORPUS / "clean.c"
    result = pipeline.analyze(clean, run=True, quality=True)
    assert result.ok, [str(d) for d in result.diagnostics]
    assert result.run and result.run.stdout.strip() == "7"


def test_rule_codes_are_unique() -> None:
    codes = [r.code for r in rules.RULES if r.code]
    assert len(codes) == len(set(codes)), "duplicate rule codes shadow each other"
