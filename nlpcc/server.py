"""A small HTTP API over the pipeline, for the React frontend.

Thin on purpose: it writes the submitted code to a temporary file, runs the
same pipeline the CLI runs, and returns the same report JSON. No analysis
logic lives here, so the web interface and the terminal can never disagree
about what a program's score is.

Standard library only -- http.server is enough for a local tool and keeps the
project's dependency list honest.

SECURITY: "run" compiles and executes submitted code, so this binds to
127.0.0.1 and nothing else. Do not expose it on a network. The executor's
containment is a timeout and a scratch directory, which is appropriate for a
local demo and not for anything reachable by other people.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import improve as improve_module
from . import inspection, models, pipeline
from .nlp import detect
from .nlp import route as nlp_route
from .optimize import OBJECTIVES, recommend
from .quality.rules import RULES as QUALITY_RULES
from .quality.score import WEIGHTS
from .report import build as build_report
from .report import to_markdown

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_BODY = 256 * 1024          # a source file, not an upload endpoint


@dataclass
class Options:
    code: str
    objectives: tuple[str, ...]
    run: bool
    ai: bool


def _options() -> dict:
    """What the frontend renders its controls from, so they cannot drift."""
    return {
        "objectives": list(OBJECTIVES),
        "dimensions": [
            {"name": name, "weight": weight} for name, weight in WEIGHTS.items()
        ],
        "model": {
            "available": models.available() is not None,
            "describe": models.describe(),
        },
        "rules": [
            {"code": rule.code, "dimension": rule.dimension,
             "title": rule.title, "penalty": rule.penalty,
             "critical": rule.critical}
            for rule in QUALITY_RULES.values()
        ],
    }


def _parse(payload: dict) -> Options | str:
    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        return "no code submitted"

    raw = payload.get("objectives") or list(OBJECTIVES)
    if not isinstance(raw, list):
        return "objectives must be a list"
    objectives = tuple(str(o) for o in raw)
    unknown = [o for o in objectives if o not in OBJECTIVES]
    if unknown:
        return f"unknown objective(s): {', '.join(unknown)}"

    return Options(code=code, objectives=objectives or OBJECTIVES,
                   run=bool(payload.get("run")), ai=bool(payload.get("ai")))


def analyze_source(options: Options) -> dict:
    """Run the pipeline over submitted code and return the report JSON."""
    with tempfile.TemporaryDirectory(prefix="nlpcc-web-") as tmp:
        source = Path(tmp) / "submitted.c"
        source.write_text(options.code)

        analysis = pipeline.analyze(source, run=options.run, quality=True,
                                    ai=options.ai)
        findings = [d for d in analysis.diagnostics if d.origin.value == "rules"]
        recommendations = recommend(source, options.objectives,
                                    quality_findings=findings,
                                    score=analysis.score)
        report = build_report(analysis, recommendations, options.objectives)

        payload = report.to_dict()
        payload["source"] = "submitted.c"       # never leak the temp path
        payload["markdown"] = to_markdown(report)
        for diag in payload["diagnostics"]:
            diag["file"] = "submitted.c"
        return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "nlpcc"

    # -- plumbing ---------------------------------------------------------
    def _send(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:                # noqa: N802 - http.server's naming
        self._send(204, {})

    def log_message(self, fmt: str, *args) -> None:   # noqa: A002
        print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:                    # noqa: N802
        if self.path.rstrip("/") in ("/api/options", "/api/options"):
            self._send(200, _options())
        elif self.path.rstrip("/") in ("", "/api/health"):
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": f"no such endpoint: {self.path}"})

    def do_POST(self) -> None:                   # noqa: N802
        route = self.path.rstrip("/")
        if route not in ("/api/analyze", "/api/ask", "/api/inspect", "/api/improve"):
            self._send(404, {"error": f"no such endpoint: {self.path}"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._send(413, {"error": "that file is too large to analyze here"})
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, {"error": "the request body is not valid JSON"})
            return
        if not isinstance(payload, dict):
            self._send(400, {"error": "the request body must be a JSON object"})
            return

        if route in ("/api/inspect", "/api/improve"):
            code = payload.get("code")
            if not isinstance(code, str) or not code.strip():
                self._send(400, {"error": "no code submitted"})
                return
            try:
                if route == "/api/inspect":
                    with tempfile.TemporaryDirectory(prefix="nlpcc-insp-") as tmp:
                        source = Path(tmp) / "submitted.c"
                        source.write_text(code)
                        self._send(200, inspection.inspect_source(source))
                else:
                    raw = payload.get("objectives") or list(OBJECTIVES)
                    wanted = tuple(o for o in raw if o in OBJECTIVES) or OBJECTIVES
                    self._send(200, improve_module.improve(
                        code, wanted, run=bool(payload.get("run")),
                        ai=bool(payload.get("ai"))).to_dict())
            except Exception as exc:             # noqa: BLE001
                self._send(500, {"error": f"{exc.__class__.__name__}: {exc}"})
            return

        if route == "/api/ask":
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                self._send(400, {"error": "say what you would like done"})
                return
            try:
                self._send(200, nlp_route.run(
                    text, code=str(payload.get("code") or ""),
                    use_model=bool(payload.get("ai"))))
            except Exception as exc:             # noqa: BLE001
                self._send(500, {"error": f"{exc.__class__.__name__}: {exc}"})
            return

        parsed = _parse(payload)
        if isinstance(parsed, str):
            self._send(400, {"error": parsed})
            return

        try:
            self._send(200, analyze_source(parsed))
        except Exception as exc:                 # noqa: BLE001 - report, never 500 blank
            self._send(500, {"error": f"analysis failed: {exc.__class__.__name__}: {exc}"})


def serve(port: int = DEFAULT_PORT) -> None:
    httpd = ThreadingHTTPServer((HOST, port), Handler)
    print(f"\n  nlpcc API on http://{HOST}:{port}")
    print("  endpoints: GET /api/options, POST /api/analyze /ask /inspect /improve")
    print(f"  model: {models.describe()}")
    print("  bound to localhost only -- it compiles and runs submitted code\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped\n")
    finally:
        httpd.server_close()
