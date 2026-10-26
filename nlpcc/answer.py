"""Answer a question in plain English, about C or about the code in the editor.

Questions used to have nowhere to go. "What does this code do" was routed to
code generation on the strength of the word "code" and came back as a brand
new program with a "Put it in the editor" button, and "what is a pointer" was
answered with a compile report.

The model answers, but from facts this project measured rather than from its
own reading of the code: the compiler's findings, and what the program
actually printed when it was run. "What does it print" is then a quotation,
not a guess -- the difference between precise and plausible.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import models, pipeline
from .diagnostics import Severity

SYSTEM_PROMPT = """You answer questions about C, and about the user's program \
when there is one. The user is learning C.

How to answer:
- Answer exactly what was asked, in the first sentence. Then add only what helps.
- Be precise. Name the lines, variables and values in their code. When the facts
  below say what the program printed, quote that output exactly; never guess it.
- Plain, friendly English. Define any term you have to use.
- Short: under 120 words unless they asked for detail.
- Do not rewrite their whole program, and never tell them to paste or put code
  into an editor. If one line needs changing, show just that line.
- If the facts do not settle it, say what you are unsure of instead of guessing.
- Plain text. Put code in `backticks`; a few lines may go in a ``` block."""

MAX_OUTPUT_SHOWN = 1500       # characters of program output passed to the model


@dataclass
class Answer:
    question: str
    text: str = ""
    provider: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {"question": self.question, "text": self.text,
                "provider": self.provider, "error": self.error}


def _facts(code: str) -> str:
    """What the pipeline found and what the program did, as plain lines."""
    with tempfile.TemporaryDirectory(prefix="nlpcc-answer-") as tmp:
        path = Path(tmp) / "program.c"
        path.write_text(code)
        analysis = pipeline.analyze(path, run=True)

    lines = []
    problems = [d for d in analysis.diagnostics if d.severity is not Severity.NOTE
                and d.phase.value != "runtime"]
    if problems:
        lines.append("The compiler reported:")
        for d in problems:
            where = f"line {d.position.line}" if d.position else "the file"
            lines.append(f"- {where}: {d.severity.value}: {d.message}")
    else:
        lines.append("It compiles with no errors or warnings.")

    run = analysis.run
    if run is None:
        lines.append("It was not run, because it does not compile.")
        return "\n".join(lines)
    lines.append("It was run once with no input typed in.")
    if run.stdout:
        shown = run.stdout[:MAX_OUTPUT_SHOWN]
        lines.append(f"It printed exactly:\n<<<\n{shown}\n>>>")
    else:
        lines.append("It printed nothing.")
    for d in analysis.diagnostics:
        if d.phase.value == "runtime":
            lines.append(f"At run time: {d.message}")
    return "\n".join(lines)


def _tidy(reply: str) -> str:
    reply = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    return reply.strip()


def answer(question: str, code: str = "") -> Answer:
    """A short, precise answer to ``question``, grounded in ``code`` if given."""
    result = Answer(question=question)
    provider = models.available()
    if provider is None:
        result.error = models.unreachable("Answering questions")
        return result
    result.provider = str(provider)

    if code.strip():
        numbered = "\n".join(f"{i:4d}  {line}"
                             for i, line in enumerate(code.splitlines(), 1))
        prompt = (f"Their program, with line numbers:\n\n{numbered}\n\n"
                  f"Facts measured by running the compiler and the program:\n"
                  f"{_facts(code)}\n\nTheir question: {question}")
    else:
        prompt = f"There is no program in the editor.\n\nTheir question: {question}"

    reasons: list[str] = []
    reply = models.complete_text(SYSTEM_PROMPT, prompt, provider=provider,
                                 max_tokens=700, reasons=reasons)
    result.text = _tidy(reply)
    if not result.text:
        result.error = "The model did not answer: " + ("; ".join(reasons) or "no reply")
    return result
