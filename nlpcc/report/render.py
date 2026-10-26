"""Render a Report as Markdown.

Written to be read by the person whose code it is: the problems first, in
source order, each with its explanation; the score and its derivation after;
the recommendations last, because they are only worth reading once the errors
are gone.
"""

from __future__ import annotations

from ..diagnostics import Severity
from ..quality.score import WEIGHTS
from .build import Report

_SEVERITY_LABEL = {
    Severity.FATAL: "Fatal", Severity.ERROR: "Error",
    Severity.WARNING: "Warning", Severity.NOTE: "Note",
}


def to_markdown(report: Report) -> str:
    out: list[str] = []
    w = out.append

    w(f"# Code analysis: `{report.source.name}`")
    w("")
    w(f"Generated {report.generated_at}")
    w("")

    status = "compiles" if report.compiles else "does not compile"
    summary = (f"**{len(report.errors)} error(s), {len(report.warnings)} warning(s)** "
               f"— the program {status}.")
    w(summary)
    if report.score:
        w("")
        w(f"**Quality score: {report.score.overall}/100 ({report.score.grade})**")
        if report.score.capped_by_security:
            w("")
            w("> The overall score is capped because a critical security issue was "
              "found. A program with a memory-safety defect does not score well on "
              "the strength of its other qualities.")
        elif report.score.limited_by:
            w("")
            w(f"> The overall score is held down by its weakest dimension "
              f"({report.score.limited_by}).")
    w("")

    if not report.frontend_available:
        w("*Our own front end is not implemented yet; the compiler findings below "
          "come from the system compiler.*")
        w("")

    # -- findings ---------------------------------------------------------
    if report.diagnostics:
        w("## Problems found")
        w("")
        for diag in report.diagnostics:
            where = f"line {diag.position.line}" if diag.position else "whole file"
            title = diag.meta.get("title") or diag.message
            w(f"### {_SEVERITY_LABEL[diag.severity]}: {title}")
            w("")
            w(f"`{report.source.name}` — {where} · {diag.phase.value} phase")
            w("")
            if diag.snippet:
                w("```c")
                w(diag.snippet.strip())
                w("```")
                w("")
            for related in diag.meta.get("related", []):
                w(f"Also line {related['line']}: {related['message']}")
                w("")
            if diag.explanation:
                w(f"**What happened.** {diag.explanation}")
                w("")
                w(f"**Fix.** {diag.fix}")
            else:
                w(f"**The compiler said.** {diag.message}")
            w("")
    else:
        w("## Problems found")
        w("")
        w("None.")
        w("")

    # -- score ------------------------------------------------------------
    if report.score:
        w("## Quality score")
        w("")
        w("| Dimension | Score | Weight | Deductions |")
        w("|---|---:|---:|---|")
        for name, value in report.score.dimensions.items():
            deductions = report.score.for_dimension(name)
            detail = ", ".join(f"{c.code} (−{c.penalty})" for c in deductions) or "—"
            w(f"| {name.title()} | {value} | {WEIGHTS[name]:.0%} | {detail} |")
        w("")
        w(f"Weighted total: **{report.score.overall}/100** ({report.score.grade}). "
          "Each dimension starts at 100; every deduction names the rule that made it.")
        w("")

    # -- metrics ----------------------------------------------------------
    if report.file_metrics and report.file_metrics.functions:
        fm = report.file_metrics
        w("## Measurements")
        w("")
        w(f"{fm.code_lines} lines of code, {fm.comment_lines} of comments "
          f"({fm.comment_density:.0%} density).")
        w("")
        w("| Function | Line | Lines | Params | Complexity | Max nesting |")
        w("|---|---:|---:|---:|---:|---:|")
        for f in fm.functions:
            w(f"| `{f.name}` | {f.line} | {f.length} | {f.parameters} "
              f"| {f.complexity} | {f.max_nesting} |")
        w("")

    # -- recommendations --------------------------------------------------
    w("## Recommendations")
    w("")
    if report.recommendations:
        w(f"Selected for: **{', '.join(report.objectives)}**.")
        w("")
        for rec in report.recommendations:
            where = f" (line {rec.line})" if rec.line else ""
            w(f"### {rec.title}{where}")
            w("")
            w(f"*Serves {', '.join(rec.objectives)} · {rec.impact} impact · "
              f"{rec.effort} effort*")
            w("")
            w(rec.rationale)
            w("")
            w(f"**Do this.** {rec.action}")
            w("")
    else:
        w("Nothing to recommend for the selected objectives "
          f"({', '.join(report.objectives)}).")
        w("")

    if report.program_output:
        w("## Program output")
        w("")
        w("```")
        w(report.program_output.rstrip())
        w("```")
        w("")

    return "\n".join(out)
