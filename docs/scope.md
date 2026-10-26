# Scope contract

Settles what this project builds and, more usefully, what it does not. Every
other document defers to this one. The subset grammar below is settled; the
quality dimensions are the remaining gap.

## Decided

**Our own front end, with clang as ground truth.** We implement lexical,
syntax and semantic analysis for a C subset. clang runs on every file as well:
where it reports something we miss, that is a tracked gap in our analyzer, not
a feature we outsource. Programs outside the subset still get full analysis
from clang, so the tool is useful on day one and the compiler-design work is
genuinely ours.

**Rules before the model.** Explanations come from a reviewable rule table.
The LLM is an enhancement layer for unmatched findings only, and its output is
labelled as model-generated. Non-negotiable: the demo must work offline and
two runs of the same input must produce the same report.

**Quality scores must be explainable.** Every point of the score traces to a
named rule with a published weight. No dimension is scored by asking a model
for a number — "readability: 7/10" with no derivation cannot be defended.

**C only.** One language, done properly.

## The subset grammar — settled

This is the contract the front end implements. `nlpcc/frontend/parser.py` has
one method per rule below, named after the rule, so the parser reads as this
grammar does. Anything not derivable here is outside the subset: the parser
emits one `W_OUTSIDE_SUBSET` note, skips the construct, and clang's findings
carry that file.

Notation: `{ x }` is zero or more, `[ x ]` is optional, `|` is choice,
`'x'` is a literal token, UPPERCASE is a token class from `frontend/tokens.py`.

```
translation_unit      := { external_declaration } EOF
external_declaration  := preprocessor_directive | function_definition | declaration
preprocessor_directive := PREPROCESSOR            (recorded, never expanded)

declaration_specifiers := { storage_class } type_specifier
storage_class         := 'static' | 'extern' | 'const'
type_specifier        := 'void' | 'char' | 'int' | 'float' | 'double'
                       | { 'short' | 'long' | 'signed' | 'unsigned' } [ 'int' | 'char' | 'double' ]
declarator            := { '*' } IDENTIFIER [ '[' [ expression ] ']' ]

function_definition   := declaration_specifiers declarator '(' parameter_list ')'
                         ( compound_statement | ';' )
parameter_list        := 'void' | [ parameter_declaration { ',' parameter_declaration } ]
parameter_declaration := declaration_specifiers declarator

declaration           := declaration_specifiers init_declarator { ',' init_declarator } ';'
init_declarator       := declarator [ '=' initializer ]
initializer           := assignment_expression | '{' [ initializer { ',' initializer } ] '}'

statement             := compound_statement | if_statement | while_statement
                       | for_statement | return_statement | break_statement
                       | continue_statement | declaration | expression_statement
compound_statement    := '{' { statement } '}'
if_statement          := 'if' '(' expression ')' statement [ 'else' statement ]
while_statement       := 'while' '(' expression ')' statement
for_statement         := 'for' '(' ( declaration | [ expression ] ';' )
                         [ expression ] ';' [ expression ] ')' statement
return_statement      := 'return' [ expression ] ';'
break_statement       := 'break' ';'
continue_statement    := 'continue' ';'
expression_statement  := [ expression ] ';'

expression            := assignment_expression
assignment_expression := conditional_expression
                       | unary_expression assignment_operator assignment_expression
assignment_operator   := '=' | '+=' | '-=' | '*=' | '/=' | '%='
                       | '&=' | '|=' | '^=' | '<<=' | '>>='
conditional_expression := logical_or_expression
                         [ '?' expression ':' conditional_expression ]
logical_or_expression  := logical_and_expression  { '||' logical_and_expression }
logical_and_expression := bitwise_or_expression   { '&&' bitwise_or_expression }
bitwise_or_expression  := bitwise_xor_expression  { '|'  bitwise_xor_expression }
bitwise_xor_expression := bitwise_and_expression  { '^'  bitwise_and_expression }
bitwise_and_expression := equality_expression     { '&'  equality_expression }
equality_expression    := relational_expression   { ( '==' | '!=' ) relational_expression }
relational_expression  := shift_expression        { ( '<' | '>' | '<=' | '>=' ) shift_expression }
shift_expression       := additive_expression     { ( '<<' | '>>' ) additive_expression }
additive_expression    := multiplicative_expression { ( '+' | '-' ) multiplicative_expression }
multiplicative_expression := cast_expression      { ( '*' | '/' | '%' ) cast_expression }
cast_expression       := '(' type_specifier { '*' } ')' cast_expression | unary_expression
unary_expression      := ( '+' | '-' | '!' | '~' | '*' | '&' | '++' | '--' ) cast_expression
                       | 'sizeof' '(' type_specifier { '*' } ')'
                       | 'sizeof' unary_expression
                       | postfix_expression
postfix_expression    := primary_expression
                         { '(' argument_list ')' | '[' expression ']' | '++' | '--' }
argument_list         := [ assignment_expression { ',' assignment_expression } ]
primary_expression    := IDENTIFIER | INT_LITERAL | FLOAT_LITERAL | CHAR_LITERAL
                       | STRING_LITERAL { STRING_LITERAL } | '(' expression ')'
```

Three additions to the proposal, each forced by a program already in the
corpus rather than by taste: storage classes (`static int add(...)` in
`clean.c`), the wider arithmetic types `double`/`long`/`short`/`signed`/
`unsigned` (`double temperature` in `semantic/format_mismatch.c` — and
`tokens.py` already lists those keywords), and `sizeof`
(`malloc(1024 * sizeof(int))` in `security/unchecked_malloc.c`). The proposal
is otherwise unchanged.

**Deliberately outside the subset**, each producing one `W_OUTSIDE_SUBSET`
note and no further complaint: `struct`, `union`, `enum`, `typedef`,
`switch`/`case`, `do`/`while`, `goto`, function pointers, varargs (`...`),
multi-dimensional arrays, compound literals, and `#define` expansion.
Pointers are in the subset only as far as declaration (`int *p`),
dereference (`*p`) and address-of (`&x`) — pointer arithmetic beyond array
indexing is not analyzed. See `tests/corpus/subset/` for the case that pins
this behaviour down.

**The semantic analyzer sees no headers.** `#include` is recorded, never
expanded, so names from the standard library have no declaration to find.
`nlpcc/frontend/semantic.py` therefore carries an explicit table of standard
names (`printf`, `malloc`, `NULL`, …); a file that includes a header outside
that table has undeclared-identifier reporting switched off for the whole
file, because a false "undeclared" on correct code costs more than a missed
one. clang still reports it.

## Settled during implementation

### The five quality dimensions

Implemented in `nlpcc/quality/`. Thresholds live together at the top of
`nlpcc/quality/rules.py` so they can be reviewed in one place; weights are in
`nlpcc/quality/score.py`.

| Dimension | Weight | Rules |
|---|---:|---|
| Security | 30% | unsafe function use, unchecked allocation, leaked allocation |
| Readability | 20% | nesting depth, function length, opaque names, comment density, repeated literals |
| Maintainability | 20% | cyclomatic complexity, parameter count, mutable globals |
| Performance | 15% | allocation in a loop, call in a loop condition |
| Practices | 15% | return-count, plus every compiler error and warning |

**How the five combine — the open question, answered.** A weighted mean alone
was wrong in both directions, so two caps sit on top of it:

1. *A critical security finding caps the overall score at 60.* This was the
   case the question named: `tests/corpus/security/unsafe_strcpy.c` scores 100
   on all four other dimensions and still cannot exceed 60, because it has a
   buffer overflow.
2. *The overall score may not sit more than 20 points above its weakest
   dimension.* Measured need, not theory: `quality/deep_nesting.c` scored 93
   "excellent" under a plain weighted mean — five levels of nesting and four
   meaningless parameter names, averaged away by four clean dimensions. Quality
   is a weakest-link property, so the headline now follows the worst dimension.

Compiler errors deduct from the practices dimension, so a file that does not
compile cannot score well on quality.

Every deduction records the rule that made it (`Score.contributions`), which is
what makes a figure defensible when someone asks where it came from.

### Report format

JSON is the single source of truth (`Report.to_dict`); Markdown is rendered
from it. Nothing reads another renderer's text, so a number can only be wrong
in one place. No PDF — a Markdown file prints.

### The model layer and its offline path

Gemini first (`nlpcc/models.py`): when `GEMINI_API_KEY` is set, Google
Gemini answers. Without a key, LM Studio or Ollama on localhost are used and
need no account, and Anthropic is used only when credentials exist. Small
local models are rougher and slower, so the provider is reported with every
answer it produced.

The explanation layer sits strictly *behind* the rule table: only findings no
rule matched are sent.
Answers are cached on disk keyed by the diagnostic's shape with
program-specific names stripped, so repeat runs are free and work offline.
Missing package, missing credentials, no network or a malformed reply each
leave the diagnostic untouched rather than failing the pipeline. Results carry
`explained_by: "llm"` so the report can mark them as generated rather than
authored. `NLPCC_OFFLINE=1` disables the layer outright.

Every model answer that proves correct is a candidate for promotion into
`nlpcc/explain/rules.py` as a permanent rule. That is how the deterministic
layer grows.

### How generated code is validated

As the question proposed: it has to pass through this pipeline first. Step 9
compiles every generated program, and on failure hands the model our own
plain-English diagnostics and asks again (two attempts). Code that still does
not compile is returned marked unaccepted rather than printed as though it
worked.

## Still open

- Persistence: SQLite is chosen but nothing is stored yet. What is kept —
  every analysis, or only reports?
- Should `tests/test_corpus.py` let a case opt into the quality pass with a
  `"quality": true` key? Today the three quality cases are covered by
  `tests/test_quality.py` instead, and stay marked pending in the corpus.
- The executor's containment is a timeout and a scratch directory. Anything
  beyond a local CLI needs rlimits, no network, and an unprivileged user.

## Explicitly not building

PostgreSQL, multi-language support, IDE plugin, anything reachable over a
network. The CLI is the deliverable; the React frontend in `frontend/` is a
local view onto the same pipeline, and its API binds to 127.0.0.1 only
because it compiles and runs submitted code.
