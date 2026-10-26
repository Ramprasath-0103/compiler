# NLP-Driven Compiler Assistant with AI-Based Code Analysis

Compiles C, finds what is wrong with it, and explains every finding in plain
English instead of compiler jargon.

**Team:** Nehal Arun (24BCT0151), Ram prasath RP (24BCE0176)

## Status

All nine steps implemented. The pipeline compiles, detects, explains, scores,
advises and reports.

| Step | What                                                | State                                    |
| ---- | --------------------------------------------------- | ---------------------------------------- |
| 1    | Scope contract (`docs/scope.md`)                    | **done** — subset grammar settled        |
| 2    | Repo, CLI, corpus                                   | **done** — 21 cases                      |
| 3    | Pipeline + ground-truth compiler + runtime executor | **done**                                 |
| 4    | Our own front end (lexer → parser → semantic)       | **done** — owns 9 of the corpus cases    |
| 5    | Explanation layer, rules first                      | **done** — 100% corpus coverage          |
| 5b   | Model explanations for unmatched findings           | **done** — cached, optional              |
| 6    | Quality scoring, five dimensions                    | **done** — 14 rules, traceable score     |
| 7    | Optimization recommender                            | **done** — objective-filtered            |
| 8    | Report generation                                   | **done** — JSON source, Markdown render  |
| 9    | Natural-language → C generation                     | **done** — validated by our own pipeline |

Not built, deliberately: React frontend, PostgreSQL, multi-language support.
The CLI is the deliverable. See `docs/scope.md`.

## Running it

```bash
./run.sh
```

Installs anything missing the first time, starts both servers, and opens
http://localhost:5173. Ctrl-C stops both.

```bash
./run.sh setup
./run.sh test
./run.sh stop
```

The terminal interface, if you prefer it:

```bash
./.venv/bin/python -m nlpcc analyze tests/corpus/syntax/missing_semicolon.c
./.venv/bin/python -m nlpcc analyze tests/corpus/runtime/divide_by_zero.c --run
./.venv/bin/python -m nlpcc analyze program.c --json
./.venv/bin/python -m nlpcc rules
```

Needs a C compiler on `PATH`. **clang is preferred** and is what the
diagnostic flags and the output parser are written against; on macOS
`/usr/bin/gcc` is a clang shim, which is fine. A real GNU gcc (what Homebrew
installs as `gcc-15`) is used only if no clang is found, with its own flag
spellings, and on macOS it cannot link Apple's sanitizer runtime.

If it works on one machine and not another:

```bash
./.venv/bin/python -m nlpcc doctor
```

checks the compiler, the diagnostic flags, building and running, the
sanitizer, the model and the frontend, and prints the fix for whatever is
missing.

## Asking in plain English

```bash
./.venv/bin/python -m nlpcc ask "why does this not compile" -f program.c
./.venv/bin/python -m nlpcc ask "make it faster and use less memory" -f program.c
./.venv/bin/python -m nlpcc ask "write me a program that reverses a string"
./.venv/bin/python -m nlpcc ask "what does this code do" -f program.c
./.venv/bin/python -m nlpcc ask "fix the error" -f program.c
```

The request is routed by keyword rules in `nlpcc/nlp/intent.py` — no model,
no API key, instant, and it reports why it decided what it decided. It picks
the stage of the pipeline to run and, for "make it faster", pulls the
objectives out of the sentence.

Questions ("what does this code do", "what is a pointer") get a short answer
from the model, grounded in what the compiler found and what the program
printed when it ran. Requests to change the code ("fix the error", "add
comments", "use a while loop") come back as a diff that has been checked to
compile. Both need a model; without one, a question still gets the analysis.

## The model

```bash
./.venv/bin/python -m nlpcc models
```

The model-backed parts — explaining unmatched diagnostics, AI code analysis,
and code generation — use Google Gemini when a Gemini API key is set. Get a
key from Google AI Studio and put it in `.env` at the project root:

```dotenv
GEMINI_API_KEY=your-key
NLPCC_GEMINI_MODEL=gemini-flash-latest   # optional
```

`GOOGLE_API_KEY` is accepted too. The default model, `gemini-flash-latest`,
is Google's alias for its current Flash model; `NLPCC_GEMINI_MODEL` pins a
specific one. The `.env` file is gitignored and read automatically at
startup; values already exported in the shell take precedence.

Without a Gemini key, the OpenAI-compatible local API is tried next — LM
Studio or another server implementing the OpenAI chat completions API:

```dotenv
NLPCC_LOCAL_URL=https://your-local-llm/v1
NLPCC_LOCAL_API_KEY=your-key   # optional
NLPCC_LOCAL_MODEL=your-model   # optional
```

Anthropic and Ollama remain available as alternate providers. Pin one with
`NLPCC_PROVIDER=gemini`, `openai-compatible`, `anthropic` or `ollama`. **Everything outside
the three model-backed features works with no model at all** — the compiler,
deterministic explanations, five dimensions, score and suggestions do not
depend on an API.

## The web interface

`./run.sh` starts both halves: the Python API on :8000 and the React dev
server on :5173, which proxies `/api` to it. To run them separately:

```bash
./.venv/bin/python -m nlpcc serve
cd frontend && npm run dev
```

Paste C code, pick what to optimize for, and press Analyze — or type what you
want in plain English and press Ask.

Three views, as tabs:

- **Report** — the findings with their explanations, the score with its
  per-dimension breakdown, the suggestions, and what the run cost (build
  time, wall time, CPU time, peak memory). Click any finding to jump to its
  line in the editor.
- **Compiler phases** — the tokens the lexer produced, the parse tree the
  parser built and the symbol table the analyzer filled, taken from the same
  objects the analysis runs on. Click a token, a node or a symbol to jump to
  its line.
- **Improved version** — a rewrite that addresses the findings, shown as a
  diff, accepted only if it still compiles and scores no worse. "Optimized"
  is a measurement here, not a claim. The options are fetched from `/api/options`, so the UI can
  only ever offer objectives the analyzer actually supports.

The server binds to localhost and nothing else: "Run the program" compiles and
executes whatever is submitted, with a timeout and a scratch directory as its
only containment. That is fine on your own machine and nowhere else.

## How it fits together

A diagram of the whole flow is in [docs/workflow.md](docs/workflow.md).

```
source.c
   │
   ├─► nlpcc/frontend/      our lexer, parser, semantic analyzer   (step 4)
   ├─► nlpcc/backends/clang_runner.py   real compiler, ground truth
   └─► nlpcc/backends/executor.py       compile, run, catch crashes
                    │
                    ▼
          Diagnostic  (nlpcc/diagnostics.py)   ← the one shared shape
                    │
                    ▼
          nlpcc/explain/rules.py   plain-English explanation + fix
                    │
                    ▼
              CLI / JSON / report
```

Everything emits `Diagnostic` and nothing else. That is the whole reason a new
analysis source costs nothing downstream: the explanation layer, the scorer and
the report never learn where a finding came from.

Two deliberate decisions:

**Our own front end, with a real compiler beside it.** The front end under
`nlpcc/frontend/` handles the C subset in `docs/scope.md` and is where the
compiler-design content of the project lives — real tokens, a real parse tree,
a real symbol table. clang runs on every file too, as ground truth: anything
clang reports that we miss is a gap in our analyzer, and the corpus makes that
gap visible. Programs outside the subset still work, because clang covers them.

**Rules before the model.** `nlpcc/explain/rules.py` maps a diagnostic code to
an explanation deterministically, offline, with no model call. The LLM layer
will sit _behind_ it, consulted only for findings no rule matched and labelled
as model-generated in the report. Built the other way round, the demo dies
with the network and no two runs produce the same report.

## The corpus is the spec

`tests/corpus/` holds small broken programs, each with an `.expected.json`
saying what the assistant must report. Two assertions per case: the finding is
reported at the right line and severity, and **every** finding has an
explanation. The second is the one that matters — an unexplained finding fails
the suite even when detection is correct, because raw compiler text reaching
the user is the exact problem this project exists to fix.

Three cases `xfail` on purpose (`unchecked_malloc`, `deep_nesting`,
`poor_naming`). They are legal C that no compiler warns about, waiting on the
step-6 rule engine. They will start passing when it lands.

### Adding a case

```bash
$EDITOR tests/corpus/semantic/my_case.c         \
./.venv/bin/python -m nlpcc analyze tests/corpus/semantic/my_case.c -v
```

Write the `.expected.json` by hand from what you _want_ reported — not by
pasting current output, which only freezes today's behaviour:

```json
{
  "title": "what is wrong with this program",
  "run": false,
  "expect": [
    {
      "line": 5,
      "severity": "error",
      "phase": "semantic",
      "code": "clang:...",
      "explained": true
    }
  ]
}
```

`code` and `phase` are optional; `line: null` matches a finding with no line.
If the case needs a step that does not exist yet, add
`"pending": "step N: why"` and it xfails instead of lying.

### Adding an explanation rule

Append a `Rule` to `nlpcc/explain/rules.py`. The code to key on comes from
`analyze -v`. Write for someone in their first semester: say what the compiler
was trying to do, why it could not, and what to type instead. Keep the `fix`
to one actionable sentence.

## Runtime errors

Programs are compiled with `-fsanitize=undefined` before running, so a division
by zero or an out-of-bounds write is reported _with a line number_ rather than
as a bare "segmentation fault". This matters more than it sounds: on Apple
Silicon, integer division by zero does not fault at all — the program keeps
running with a meaningless value, and without the sanitizer there is nothing to
report. See `tests/corpus/runtime/divide_by_zero.c`.

## Security note

`nlpcc/backends/executor.py` runs user-supplied code. Its only containment
today is a wall-clock timeout and a scratch working directory — fine for a
local CLI, not fine for anything networked. Before this is ever exposed beyond
the terminal it needs rlimits, no network, and an unprivileged user.

# Compiler_Project
