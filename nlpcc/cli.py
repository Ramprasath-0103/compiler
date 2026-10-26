"""Terminal interface. `python -m nlpcc analyze program.c`"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import generate as generator
from . import pipeline
from .diagnostics import Severity, to_json
from .explain import llm, rules
from .optimize import OBJECTIVES, recommend
from .quality.score import WEIGHTS
from .report import build as build_report
from .report import to_markdown

_COLOR = {Severity.FATAL: "\033[1;31m", Severity.ERROR: "\033[31m",
          Severity.WARNING: "\033[33m", Severity.NOTE: "\033[36m"}
_RESET = "\033[0m"


# The sanitizer's own report is already a Diagnostic with an explanation above;
# echoing its raw form too is just noise in the demo.
_SANITIZER_NOISE = ("runtime error:", "UndefinedBehaviorSanitizer", "SUMMARY: Undefined",
                    "AddressSanitizer", "SUMMARY: Address")


def _strip_sanitizer_noise(text: str) -> str:
    return "\n".join(line for line in text.splitlines()
                     if not any(marker in line for marker in _SANITIZER_NOISE))


def _render(result: pipeline.Analysis, color: bool, verbose: bool) -> None:
    src = result.source
    print(f"\n  {src}")
    if not result.frontend_available:
        print("  (own front end not implemented yet -- findings below are from the system compiler)")

    if not result.diagnostics:
        print("\n  No problems found.\n")
    for diag in result.diagnostics:
        tint = _COLOR.get(diag.severity, "") if color else ""
        reset = _RESET if color else ""
        where = f"line {diag.position.line}" if diag.position else "file"
        title = diag.meta.get("title") or diag.message
        print(f"\n  {tint}{diag.severity.value.upper()}{reset}  {where}  ({diag.phase.value})")
        print(f"  {title}")
        if diag.snippet:
            print(f"      {diag.snippet.strip()}")
        for related in diag.meta.get("related", []):
            print(f"  also, line {related['line']}: {related['message']}")
            if related.get("snippet"):
                print(f"      {related['snippet'].strip()}")
        if diag.explanation:
            print(f"\n  What happened: {diag.explanation}")
            print(f"  Fix: {diag.fix}")
        else:
            print(f"  Compiler said: {diag.message}")
            print("  (no explanation rule for this yet -- see nlpcc/explain/rules.py)")
        if verbose:
            print(f"  [code={diag.code} origin={diag.origin.value}]")

    if result.run is not None and not result.run.sanitized:
        print("\n  Runtime checks are off: the sanitizer could not be linked on "
              "this machine,\n  so the program ran unchecked. Problems that only "
              "appear at runtime\n  (divide by zero, out-of-bounds writes) will "
              "not be reported.\n  Run 'nlpcc doctor' for details.")

    if result.run is not None:
        print("\n  --- program output ---")
        if result.run.stdout:
            print(result.run.stdout.rstrip())
        stderr = _strip_sanitizer_noise(result.run.stderr)
        if stderr:
            print(stderr.rstrip())
        if result.run.exit_code is not None:
            print(f"  (exit status {result.run.exit_code})")

    gaps = rules.unexplained(result.diagnostics)
    if gaps and not verbose:
        print(f"\n  {len(gaps)} finding(s) have no explanation rule yet.")
    print()


def _render_score(score, color: bool) -> None:
    tint = "\033[1m" if color else ""
    reset = "\033[0m" if color else ""
    print(f"\n  {tint}Quality score: {score.overall}/100 ({score.grade}){reset}\n")
    for name, value in score.dimensions.items():
        bar = "#" * (value // 5) + "." * (20 - value // 5)
        print(f"    {name:<16} {value:>3}  [{bar}]  weight {WEIGHTS[name]:.0%}")
    if score.capped_by_security:
        print("\n  Capped: a critical security issue limits the overall score no "
              "matter\n  how good the other dimensions are.")
    elif score.limited_by:
        print(f"\n  Held down by its weakest dimension ({score.limited_by}).")
    deductions = score.contributions
    if deductions:
        print("\n  Where the points went:")
        for c in sorted(deductions, key=lambda c: -c.penalty):
            where = f"line {c.line}" if c.line else "file"
            print(f"    -{c.penalty:<3} {c.dimension:<16} {c.code:<24} {where}")
    else:
        print("\n  No deductions.")
    print()


def _render_recommendations(recs, objectives) -> None:
    print(f"\n  Recommendations for: {', '.join(objectives)}\n")
    if not recs:
        print("  Nothing to suggest for those objectives.\n")
        return
    for rec in recs:
        where = f"line {rec.line}" if rec.line else "whole file"
        print(f"  [{rec.impact} impact / {rec.effort} effort]  {where}")
        print(f"  {rec.title}")
        print(f"      serves: {', '.join(rec.objectives)}")
        print(f"\n  Why: {rec.rationale}")
        print(f"  Do:  {rec.action}\n")


def _parse_objectives(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return OBJECTIVES
    wanted = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    unknown = [o for o in wanted if o not in OBJECTIVES]
    if unknown:
        raise SystemExit(f"nlpcc: unknown objective(s): {', '.join(unknown)}\n"
                         f"       choose from: {', '.join(OBJECTIVES)}")
    return wanted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="nlpcc",
        description="NLP-driven compiler assistant: compile, explain, analyze.")
    sub = ap.add_subparsers(dest="command", required=True)

    an = sub.add_parser("analyze", help="compile a program and explain its problems")
    an.add_argument("source", type=Path)
    an.add_argument("--run", action="store_true", help="also execute it if it compiles")
    an.add_argument("--stdin", default="", help="input to feed the program when running")
    an.add_argument("--json", action="store_true", help="machine-readable output")
    an.add_argument("-v", "--verbose", action="store_true", help="show codes and origins")
    an.add_argument("--no-color", action="store_true")

    an.add_argument("--explain-with-model", action="store_true",
                    help="ask the model about findings no rule explains")

    sc = sub.add_parser("score", help="rate code quality across the five dimensions")
    sc.add_argument("source", type=Path)
    sc.add_argument("--json", action="store_true")
    sc.add_argument("--ai", action="store_true", help="also run the AI code analysis")
    sc.add_argument("--no-color", action="store_true")

    op = sub.add_parser("optimize", help="suggest improvements for chosen objectives")
    op.add_argument("source", type=Path)
    op.add_argument("--for", dest="objectives", metavar="LIST",
                    help=f"comma-separated: {', '.join(OBJECTIVES)} (default: all)")
    op.add_argument("--json", action="store_true")
    op.add_argument("--ai", action="store_true", help="also run the AI code analysis")

    rp = sub.add_parser("report", help="write the full analysis report")
    rp.add_argument("source", type=Path)
    rp.add_argument("-o", "--output", type=Path, help="write to this file")
    rp.add_argument("--for", dest="objectives", metavar="LIST",
                    help="objectives for the recommendations section")
    rp.add_argument("--json", action="store_true", help="JSON instead of Markdown")
    rp.add_argument("--run", action="store_true", help="also execute the program")
    rp.add_argument("--ai", action="store_true", help="also run the AI code analysis")
    rp.add_argument("--explain-with-model", action="store_true")

    gn = sub.add_parser("generate", help="write C from a natural-language description")
    gn.add_argument("description", nargs="+", help="what the program should do")
    gn.add_argument("-o", "--output", type=Path, help="write the program here")
    gn.add_argument("--json", action="store_true")

    ak = sub.add_parser("ask", help="say what you want in plain English")
    ak.add_argument("request", nargs="+", help='e.g. "why does this not compile"')
    ak.add_argument("-f", "--file", type=Path, help="the program to work on")
    ak.add_argument("--ai", action="store_true", help="use the model where it helps")
    ak.add_argument("--json", action="store_true")

    sub.add_parser("doctor", help="check this machine's toolchain")
    sub.add_parser("models", help="show which model is available")
    sub.add_parser("rules", help="list the explanation rules")

    sv = sub.add_parser("serve", help="run the HTTP API for the web frontend")
    sv.add_argument("--port", type=int, default=8000)

    args = ap.parse_args(argv)

    if args.command == "serve":
        from .server import serve
        serve(args.port)
        return 0

    if args.command == "doctor":
        from .doctor import report as doctor_report
        return doctor_report()

    if args.command == "models":
        from . import models as model_registry
        print(f"\n  {model_registry.describe()}\n")
        return 0

    if args.command == "ask":
        from .nlp import route as nlp_route
        text = " ".join(args.request)
        code = ""
        if args.file:
            if not args.file.exists():
                print(f"nlpcc: no such file: {args.file}", file=sys.stderr)
                return 1
            code = args.file.read_text()

        result = nlp_route.run(text, code=code,
                               source=args.file if args.file else None,
                               use_model=args.ai)
        if args.json:
            print(json.dumps(result, indent=2))
            return 0

        intent = result["intent"]
        print(f"\n  Understood as: {intent['summary']}"
              f"  (confidence {intent['confidence']:.0%})")
        if intent["objectives"]:
            print(f"  Optimizing for: {', '.join(intent['objectives'])}")
        if intent["confidence"] < 0.5:
            print("  Not sure what you meant -- rephrase if that is not it.")

        if "answer" in result:
            answer = result["answer"]
            if answer["text"]:
                print(f"\n{answer['text']}\n")
                return 0
            print(f"\n  {answer['error']}")
            if "report" not in result:
                return 1

        if "edit" in result:
            changed = result["edit"]
            print(f"\n  {'Changed' if changed['accepted'] else 'Not changed'}: "
                  f"{changed['reason']}\n")
            if changed["diff"]:
                print(changed["diff"])
            return 0 if changed["accepted"] else 1

        if "error" in result:
            print(f"\n  {result['error']}\n")
            return 1

        if "generated" in result:
            gen = result["generated"]
            if gen["accepted"]:
                print(f"\n  Wrote and verified it ({gen['attempts']} attempt(s), "
                      f"{gen.get('provider', 'model')}):\n")
                print(gen["code"])
            else:
                print(f"\n  Could not: {gen['failure_reason']}\n", file=sys.stderr)
                return 1
            return 0

        report = result["report"]
        print(f"\n  {report['summary']['errors']} error(s), "
              f"{report['summary']['warnings']} warning(s)")
        for diag in report["diagnostics"]:
            where = f"line {diag['position']['line']}" if diag["position"] else "file"
            title = (diag["meta"] or {}).get("title") or diag["message"]
            print(f"\n  {diag['severity'].upper()}  {where}\n  {title}")
            if diag["explanation"]:
                print(f"\n  What happened: {diag['explanation']}")
                print(f"  Fix: {diag['fix']}")
        if report["score"]:
            print(f"\n  Quality score: {report['score']['overall']}/100 "
                  f"({report['score']['grade']})")
        for rec in report["recommendations"]:
            print(f"\n  Suggestion ({rec['impact']} impact): {rec['title']}")
            print(f"  {rec['action']}")
        print()
        return 0

    if args.command == "rules":
        for rule in rules.RULES:
            print(f"  {rule.code or rule.pattern}\n      {rule.title}")
        print(f"\n  {len(rules.RULES)} rules.")
        return 0

    if args.command == "generate":
        description = " ".join(args.description)
        result = generator.generate(description)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        elif result.accepted:
            print(f"\n  Generated and verified: it compiles cleanly "
                  f"({len(result.attempts)} attempt(s)).\n")
            print(result.code)
            if result.notes:
                print(f"\n  Notes: {result.notes}")
            if args.output:
                args.output.write_text(result.code)
                print(f"\n  Written to {args.output}")
        else:
            print(f"\n  Not accepted: {result.failure_reason}\n", file=sys.stderr)
            if result.code:
                print("  The last attempt, for reference only:\n")
                print(result.code)
        return 0 if result.accepted else 1

    if not args.source.exists():
        print(f"nlpcc: no such file: {args.source}", file=sys.stderr)
        return 1

    if args.command == "score":
        result = pipeline.analyze(args.source, quality=True, ai=args.ai)
        if result.score is None:
            print("nlpcc: cannot score a program that does not compile; "
                  "fix the errors first (run 'nlpcc analyze').", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result.score.to_dict(), indent=2))
        else:
            _render_score(result.score, color=not args.no_color and sys.stdout.isatty())
        return 0

    if args.command == "optimize":
        objectives = _parse_objectives(args.objectives)
        result = pipeline.analyze(args.source, quality=True, ai=args.ai)
        quality_findings = [d for d in result.diagnostics if d.origin.value == "rules"]
        recs = recommend(args.source, objectives, quality_findings=quality_findings,
                         score=result.score)
        if args.json:
            print(json.dumps({"objectives": list(objectives),
                              "recommendations": [r.to_dict() for r in recs]}, indent=2))
        else:
            _render_recommendations(recs, objectives)
        return 0

    if args.command == "report":
        objectives = _parse_objectives(args.objectives)
        result = pipeline.analyze(args.source, run=args.run, quality=True, ai=args.ai)
        if args.explain_with_model:
            llm.explain_unmatched(result.diagnostics, source=args.source)
        quality_findings = [d for d in result.diagnostics if d.origin.value == "rules"]
        recs = recommend(args.source, objectives, quality_findings=quality_findings,
                         score=result.score)
        report = build_report(result, recs, objectives)
        text = report.to_json() if args.json else to_markdown(report)
        if args.output:
            args.output.write_text(text)
            print(f"  Report written to {args.output}")
        else:
            print(text)
        return 0 if report.compiles else 1

    result = pipeline.analyze(args.source, run=args.run, stdin=args.stdin)
    if args.explain_with_model:
        outcome = llm.explain_unmatched(result.diagnostics, source=args.source)
        if not outcome.available and outcome.attempted:
            print(f"  (the model layer is unavailable: {outcome.unavailable_reason})")
    if args.json:
        print(json.dumps({
            "source": str(result.source),
            "ok": result.ok,
            "frontend_available": result.frontend_available,
            "diagnostics": json.loads(to_json(result.diagnostics)),
        }, indent=2))
    else:
        _render(result, color=not args.no_color and sys.stdout.isatty(),
                verbose=args.verbose)
    return 0 if result.ok else 1
