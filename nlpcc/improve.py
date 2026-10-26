"""Rewrite a program to address its findings, and prove the rewrite is better.

The model proposes; this module checks. A rewrite is accepted only when it
still compiles and its quality score is no worse than the original's -- so
"optimized" is a measurement, not a claim, and a rewrite that makes things
worse is reported as rejected rather than offered as an improvement.

The diff is computed here rather than in the browser so the terminal and the
web interface show the same one.
"""

from __future__ import annotations

import difflib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import generate, models, pipeline
from .diagnostics import Diagnostic, Origin, Phase, Severity
from .optimize import OBJECTIVES

SYSTEM_PROMPT = """You improve small C programs.

You are given a program and a list of problems found in it. Rewrite the
program so those problems are gone.

Rules:
- Keep what the program does identical. Same inputs, same outputs.
- Fix the listed problems. Do not make unrelated changes.
- C99, portable, compiles clean under -Wall -Wextra.
- Check what can fail; no gets, strcpy, sprintf or atoi.
- Name things for what they hold.

Reply with the C source and nothing else: no explanation before or after it,
no markdown fence. Your whole reply must be the contents of a .c file."""


@dataclass
class Improvement:
    original: str
    improved: str = ""
    diff: str = ""
    accepted: bool = False
    reason: str = ""
    provider: str = ""
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    objectives: tuple[str, ...] = OBJECTIVES
    instruction: str = ""         # what the user asked for, for an edit

    @property
    def score_change(self) -> int:
        # No score on a side that did not compile: nothing to compare.
        after, before = self.after.get("overall"), self.before.get("overall")
        if after is None or before is None:
            return 0
        return int(after) - int(before)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "provider": self.provider,
            "original": self.original,
            "improved": self.improved,
            "diff": self.diff,
            "diff_lines": _diff_lines(self.diff),
            "before": self.before,
            "after": self.after,
            "score_change": self.score_change,
            "objectives": list(self.objectives),
            "instruction": self.instruction,
        }


def _diff_lines(diff: str) -> list[dict[str, str]]:
    """The diff as rows the UI can colour without parsing it again."""
    rows = []
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            kind = "meta"
        elif line.startswith("@@"):
            kind = "hunk"
        elif line.startswith("+"):
            kind = "add"
        elif line.startswith("-"):
            kind = "remove"
        else:
            kind = "same"
        rows.append({"kind": kind, "text": line})
    return rows


def _snapshot(analysis) -> dict[str, Any]:
    score = analysis.score
    return {
        "overall": score.overall if score else None,
        "grade": score.grade if score else None,
        "dimensions": dict(score.dimensions) if score else {},
        "findings": len([d for d in analysis.diagnostics
                         if d.severity in (Severity.ERROR, Severity.WARNING)]),
        "compiles": not [d for d in analysis.diagnostics
                         if d.severity in (Severity.ERROR, Severity.FATAL)
                         and d.origin in (Origin.CLANG, Origin.FRONTEND)],
        "runtime_errors": sorted({d.code for d in analysis.diagnostics
                                  if d.phase is Phase.RUNTIME
                                  and d.severity in (Severity.ERROR, Severity.FATAL)}),
    }


def _ending_like(original: str, rewrite: str) -> str:
    """The rewrite with the original's final newline, or lack of one.

    The model's reply is stripped, so without this every diff ended in a
    "-}" / "+}" pair that changed nothing but the newline after it.
    """
    rewrite = rewrite.rstrip("\n")
    return rewrite + "\n" if original.endswith("\n") else rewrite


def _same_program(a: str, b: str) -> bool:
    """True when two sources differ in whitespace at most."""
    return a.split() == b.split()


def _problem_list(analysis) -> str:
    lines = []
    for diag in analysis.diagnostics:
        if diag.severity is Severity.NOTE:
            continue
        where = f"line {diag.position.line}" if diag.position else "the file"
        title = diag.meta.get("title") or diag.message
        lines.append(f"- {where}: {title}")
        if diag.fix:
            lines.append(f"    fix: {diag.fix}")
    return "\n".join(lines) or "- no specific findings; improve it where you can"


def improve(code: str, objectives: tuple[str, ...] = OBJECTIVES,
            run: bool = False, ai: bool = False) -> Improvement:
    """Ask for a better version of ``code`` and verify that it is better.

    ``run`` and ``ai`` must match the report the user is looking at. Both
    sides of the comparison are analyzed the same way, and that way has to be
    the report's: improving with neither on scored a program 100 here while
    its report, with runtime and AI findings counted, said 91 -- and the
    model was never told about the findings that cost those nine points.
    """
    result = Improvement(original=code, objectives=objectives)

    with tempfile.TemporaryDirectory(prefix="nlpcc-improve-") as tmp:
        before_path = Path(tmp) / "before.c"
        before_path.write_text(code)
        before = pipeline.analyze(before_path, run=run, quality=True, ai=ai)
        result.before = _snapshot(before)

        # Checked before looking for a model: there is no point spending a
        # call on a program the compiler has already rejected, and the user
        # needs to hear about the errors rather than about the model.
        if not result.before["compiles"]:
            result.reason = ("the program has to compile before it can be improved; "
                             "fix the errors first")
            return result

        provider = models.available()
        if provider is None:
            result.reason = models.describe()
            return result
        result.provider = str(provider)

        prompt = (
            f"Improve this program, with an eye to {', '.join(objectives)}.\n\n"
            f"The program:\n\n{code}\n\n"
            f"Problems found in it:\n\n{_problem_list(before)}\n"
        )
        reasons: list[str] = []
        reply = models.complete_text(SYSTEM_PROMPT, prompt, provider=provider,
                                     max_tokens=4096, reasons=reasons)
        improved = _ending_like(code, generate._clean(reply))
        if not improved.strip():
            result.reason = ("the model returned no code: "
                             + ("; ".join(reasons) or "no reply"))
            return result
        if _same_program(code, improved):
            # Otherwise an echo of the input scores equal and is "accepted".
            result.improved = improved
            result.reason = "the model returned the program unchanged"
            return result

        after_path = Path(tmp) / "after.c"
        after_path.write_text(improved)
        after = pipeline.analyze(after_path, run=run, quality=True, ai=ai)
        result.improved = improved
        result.after = _snapshot(after)

        result.diff = "".join(difflib.unified_diff(
            code.splitlines(keepends=True),
            improved.splitlines(keepends=True),
            fromfile="before.c", tofile="after.c", n=3))

        if not result.after["compiles"]:
            result.reason = ("the rewrite does not compile, so it is not offered "
                             "as an improvement")
            return result

        new_failures = sorted(set(result.after["runtime_errors"])
                              - set(result.before["runtime_errors"]))
        if new_failures:
            # A crash costs one dimension, which AI findings elsewhere can
            # outweigh -- but a rewrite that crashes where the original did not
            # has changed what the program does, whatever it scores.
            result.reason = ("the rewrite fails at run time where the original "
                             f"did not ({', '.join(new_failures)}), so it is not "
                             "offered as an improvement")
            return result

        before_score = result.before.get("overall") or 0
        after_score = result.after.get("overall") or 0
        if after_score < before_score:
            result.reason = (f"the rewrite scored {after_score} against "
                             f"{before_score}, so it is not an improvement")
            return result

        result.accepted = True
        if after_score == before_score:
            fewer = result.before["findings"] - result.after["findings"]
            if fewer > 0:
                result.reason = (f"the score is unchanged at {after_score}, with "
                                 f"{fewer} finding{'s' if fewer != 1 else ''} fewer")
            else:
                result.reason = (f"the score is unchanged at {after_score} and no "
                                 "findings were removed")
        else:
            result.reason = f"score {before_score} to {after_score}"
        return result


EDIT_PROMPT = """You change a C program exactly the way the user asks.

Rules:
- Do what they asked and nothing else. Leave every other line as it was.
- If they ask you to fix it, fix the problems listed, and only those.
- Keep what the program does, unless the request is to change that.
- C99, compiles clean under -Wall -Wextra.

Reply with the complete C source and nothing else: no explanation before or
after it, no markdown fence. Your whole reply must be the contents of a .c file."""


def edit(code: str, instruction: str) -> Improvement:
    """Change ``code`` as ``instruction`` says, and check the result compiles.

    Unlike improve() this takes code that does not compile -- "fix the error"
    is the commonest thing anyone types -- and it does not insist the score
    goes up: "add comments" or "use a while loop" is the user's call, not a
    quality contest. What it does insist on is that the result builds, that
    something actually changed, and that it does not crash where the original
    did not. Both sides are run: "fix the error" on a program that compiles
    but reads past the end of an array is about the run-time error, and
    without running it the model was never told there was one.
    """
    result = Improvement(original=code, objectives=(), instruction=instruction)

    with tempfile.TemporaryDirectory(prefix="nlpcc-edit-") as tmp:
        before = pipeline.analyze(_write(tmp, "before.c", code), run=True, quality=True)
        result.before = _snapshot(before)

        provider = models.available()
        if provider is None:
            result.reason = models.unreachable("Changing your code")
            return result
        result.provider = str(provider)

        prompt = (f"What to change: {instruction}\n\nThe program:\n\n{code}\n\n"
                  f"Problems found in it:\n\n{_problem_list(before)}\n")
        for attempt in range(2):
            reasons: list[str] = []
            improved = _ending_like(code, generate._clean(models.complete_text(
                EDIT_PROMPT, prompt, provider=provider, max_tokens=4096,
                reasons=reasons)))
            if not improved.strip():
                result.reason = ("the model returned no code: "
                                 + ("; ".join(reasons) or "no reply"))
                return result
            if _same_program(code, improved):
                result.improved = improved
                result.reason = "the model returned the program unchanged"
                return result

            after = pipeline.analyze(_write(tmp, "after.c", improved), run=True,
                                     quality=True)
            result.improved = improved
            result.after = _snapshot(after)
            if result.after["compiles"]:
                break
            # One more try, handed the compiler's own words.
            errors = "\n".join(f"- {line}" for line in _problem_list(after).splitlines()
                               if line.strip())
            prompt = (f"What to change: {instruction}\n\nThe original program:\n\n"
                      f"{code}\n\nYour version does not compile:\n\n{improved}\n\n"
                      f"The compiler said:\n{errors}\n\nReturn the corrected program.")

        result.diff = "".join(difflib.unified_diff(
            code.splitlines(keepends=True), result.improved.splitlines(keepends=True),
            fromfile="before.c", tofile="after.c", n=3))

        if not result.after["compiles"]:
            result.reason = ("the changed program does not compile, so it is not "
                             "offered as a replacement")
            return result

        new_failures = sorted(set(result.after["runtime_errors"])
                              - set(result.before["runtime_errors"]))
        if new_failures:
            result.reason = ("the changed program fails at run time where yours did "
                             f"not ({', '.join(new_failures)}), so it is not offered "
                             "as a replacement")
            return result

        result.accepted = True
        fixed = result.before["findings"] - result.after["findings"]
        parts = ["done, and it compiles"]
        if not result.before["compiles"]:
            parts = ["done -- it compiles now"]
        if result.before["runtime_errors"] and not result.after["runtime_errors"]:
            parts.append("it no longer fails when run")
        if fixed > 0:
            parts.append(f"{fixed} problem{'s' if fixed != 1 else ''} fewer")
        if result.before.get("overall") is not None and result.after.get("overall") is not None \
                and result.after["overall"] != result.before["overall"]:
            parts.append(f"score {result.before['overall']} to {result.after['overall']}")
        result.reason = "; ".join(parts)
        return result


def _write(directory: str, name: str, code: str) -> Path:
    path = Path(directory) / name
    path.write_text(code)
    return path
