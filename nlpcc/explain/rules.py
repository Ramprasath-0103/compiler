"""The explanation layer, rules first.

Deterministic, offline, and reviewable: a diagnostic code maps to a plain
English explanation and a concrete fix. No model call, so the same input
always produces the same report and the demo works with the network unplugged.

The LLM layer (step 5b, ``nlpcc/explain/llm.py``) sits *behind* this one: it
is consulted only for diagnostics no rule matched, and whatever it returns is
labelled as model-generated in the report. Never the other way round.

Written the way a friend who knows C would say it at your shoulder: first what
happened in *your* code, by name and line, then why C cares, then what to
type. The first version explained every error as a textbook paragraph about C
in general -- correct, and no help to someone looking at their own line 20.

Adding a rule is the main way this project gets better. Each entry needs:
  code         stable id from the Diagnostic (ours, or clang's slug/flag)
  title        a few words, the headline a beginner reads first
  explanation  why it matters, in a sentence or two -- no jargon
  fix          what to actually type instead
and may add:
  says         what happened, filled from this diagnostic: {name} {line}
               {code} and anything ``extract`` or ``kind`` provides
  try_this     the fix for this exact code, filled the same way
A template whose details are missing is skipped, never printed half-filled.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..diagnostics import Diagnostic


@dataclass(frozen=True)
class Rule:
    code: str                 # exact code match, or "" when matching by pattern
    title: str
    explanation: str
    fix: str
    pattern: str = ""         # regex against the raw message, used when code is ""
    says: str = ""
    try_this: str = ""
    extract: str = ""         # regex whose named groups become template values
    kind: str = ""            # extra details worked out in _details()

    def matches(self, diag: Diagnostic) -> bool:
        if self.code and diag.code == self.code:
            return True
        if self.pattern and re.search(self.pattern, diag.message, re.I):
            return True
        return False


# Every diagnostic the project can produce needs an entry here:
# tests/test_corpus.py fails loudly for any corpus diagnostic no rule explains,
# and tests/test_frontend.py fails for any code our own front end can emit and
# has not explained.
#
# -- our own front end ----------------------------------------------------
# These come first on purpose. Matching is first-rule-wins, and an exact code
# match from our own analyzer is always a better explanation than a regex over
# a compiler's message text -- our message is written for the rule that
# explains it. One rule per code in nlpcc.frontend.FRONTEND_CODES; the test
# tests/test_frontend.py::test_every_frontend_code_has_a_rule enforces that.
RULES: tuple[Rule, ...] = (
    # phase 1 -- lexical
    Rule(
        code="E_UNTERMINATED_STRING",
        title="A string is missing its closing quote",
        says="The text that starts on line {line} never gets its closing \".",
        explanation=(
            "Think of quotes as a pair of brackets around text. With the second one "
            "missing, C reads the rest of the line -- the ')' and the ';' too -- as "
            "part of the text. That is why one missing quote can cause several "
            "errors that look unrelated."
        ),
        fix="Add the closing \" at the end of the text, before the ')' or ';'.",
    ),
    Rule(
        code="E_UNTERMINATED_CHAR",
        title="A character is missing its closing quote",
        says="The ' on line {line} is never closed.",
        explanation=(
            "Single quotes hold exactly one character, like 'a' or '\\n'. For "
            "words, C uses double quotes: 'hello' is not text in C, but \"hello\" is."
        ),
        fix="Close it with ', or switch to double quotes if you meant a piece of text.",
    ),
    Rule(
        code="E_UNTERMINATED_COMMENT",
        title="A comment is never closed",
        says="The /* comment that opens on line {line} never reaches a */.",
        explanation=(
            "A /* comment runs until the next */, however far away that is. Without "
            "one, everything below this line counts as comment, so none of it is "
            "being compiled."
        ),
        fix="Add */ where the comment should end.",
    ),
    Rule(
        code="E_STRAY_CHARACTER",
        title="A character C does not understand",
        says="Line {line} has a character C cannot read.",
        explanation=(
            "This usually sneaks in when code is copied from a web page or a word "
            "processor: a curly quote instead of a straight one, or a special space "
            "that looks exactly like a normal one."
        ),
        fix="Delete the character, or retype the line by hand.",
    ),
    Rule(
        code="E_MALFORMED_NUMBER",
        title="A number with letters stuck to it",
        says="Line {line} has a number with letters glued to the end of it.",
        explanation=(
            "In maths '2x' means 2 times x, but C reads '2x' as one broken number. "
            "C needs the * written out."
        ),
        fix="Write the operator, as in 2 * x, or add the missing space.",
    ),

    # phase 2 -- syntax
    Rule(
        code="E_MISSING_SEMICOLON",
        title="Missing semicolon",
        kind="semicolon",
        says="Line {line} ends at `{code}` without a ;.",
        explanation=(
            "The ; is C's full stop. Without it, C reads straight on into the next "
            "line as if it were the same statement, and gets confused there -- so "
            "the error can look like it is on the line below."
        ),
        fix="Put a ; at the end of the statement.",
        try_this="Make it `{fixed}`.",
    ),
    Rule(
        code="E_MISSING_PAREN",
        title="A ( is never closed",
        says="The ( on line {line} never gets its ).",
        explanation=(
            "Brackets come in pairs. C needs the ) to know where the condition "
            "or the list of arguments ends, so it stops at the { or ; it found "
            "instead."
        ),
        fix="Add the ) before the { or ; on that line.",
    ),
    Rule(
        code="E_MISSING_BRACE",
        title="A { is never closed",
        says="The file ends with the { from line {other_line} still open.",
        explanation=(
            "Every { needs a }. The error points at the end of the file because "
            "that is where C ran out of code, but the forgotten } belongs "
            "somewhere after line {other_line}. Indenting each block makes the "
            "gap easy to spot."
        ),
        fix="Add the } that closes the block opened on that line.",
    ),
    Rule(
        code="E_MISSING_BRACKET",
        title="A [ is never closed",
        says="The [ on line {line} never gets its ].",
        explanation=(
            "Array sizes and indexes sit between [ and ]. Without the ], C cannot "
            "tell where the index stops."
        ),
        fix="Add ] after the index or the size.",
    ),
    Rule(
        code="E_MISSING_COLON",
        title="A ? with no :",
        says="The ? on line {line} has no matching :.",
        explanation=(
            "'a ? b : c' means 'if a then b, otherwise c'. The 'otherwise' half is "
            "missing, so there is no answer for when the condition is false."
        ),
        fix="Add : and the value to use when the condition is false.",
    ),
    Rule(
        code="E_UNEXPECTED_TOKEN",
        title="Something is in the wrong place",
        says="C did not expect to see this on line {line}.",
        explanation=(
            "Most of the time the real mistake is a line or two above: a missing "
            "; ) or } there leaves C still reading the earlier statement when it "
            "reaches this one."
        ),
        fix="Check the lines just above for a missing ; ) or } first.",
    ),
    Rule(
        code="E_EXPECTED_EXPRESSION",
        title="A value is missing",
        says="On line {line}, C wanted a value -- a number, a variable or a call -- and found none.",
        explanation=(
            "This usually means an operator with nothing after it, like 'x = ;' or "
            "'a + )'. It can also be a knock-on effect of an earlier mistake, such "
            "as a missing quote."
        ),
        fix="Fill in the missing value, or fix the first error in the file and try again.",
    ),
    Rule(
        code="W_OUTSIDE_SUBSET",
        title="Our checker skipped part of this file",
        says="This file uses C our own checker does not read, such as struct or switch.",
        explanation=(
            "Nothing is wrong with your code because of this. The real compiler "
            "still checked all of it, so any actual errors are still listed. Only "
            "our extra checks skip this part."
        ),
        fix="Nothing to do.",
    ),

    # phase 3 -- semantic
    Rule(
        code="E_UNDECLARED_IDENTIFIER",
        title="Unknown name",
        kind="undeclared",
        says="`{name}` is used on line {line}, but nothing called `{name}` exists there.",
        explanation=(
            "C only knows names you have declared above the place you use them. "
            "The usual reasons are a typo, a missing declaration, or a variable "
            "declared inside other { } braces, which stops existing at their }."
        ),
        fix="Fix the spelling, or declare it above this line, outside any inner { }.",
    ),
    Rule(
        code="E_REDECLARATION",
        title="Same name declared twice",
        says="`{name}` is declared a second time on line {line}.",
        explanation=(
            "Inside one pair of { } a name can only mean one thing. If you only "
            "wanted to change its value, leave out the type: `{name} = ...;`."
        ),
        fix="Remove the type from the second one so it just assigns, or pick a new name.",
    ),
    Rule(
        code="E_TYPE_MISMATCH",
        title="Wrong kind of value",
        says="Line {line}: {message}.",
        explanation=(
            "Each variable holds one kind of thing. \"5\" is text and 5 is a number, "
            "and C will not swap one for the other on its own -- same for an address "
            "and a number."
        ),
        fix="Give it a value of the right type, or change the variable's type.",
    ),
    Rule(
        code="E_WRONG_ARG_COUNT",
        title="Wrong number of arguments",
        extract=r"to '(?P<name>\w+)': it takes (?P<takes>\d+), but (?P<passed>\d+) w",
        says="Line {line} gives `{name}` {passed_said}, but `{name}` takes {takes_said}.",
        explanation=(
            "C has no optional arguments. A function always takes exactly what its "
            "first line lists, in that order."
        ),
        fix="Pass one value for each parameter in the function's definition.",
    ),
    Rule(
        code="E_NOT_A_FUNCTION",
        title="This is not a function",
        says="`{name}` is called like a function on line {line}, but it is a variable.",
        explanation=(
            "Brackets right after a name mean 'call this'. Either a different name "
            "was meant, or the brackets are a slip."
        ),
        fix="Call the function you meant, or remove the ( ) after the variable.",
    ),
    Rule(
        code="E_VOID_VARIABLE",
        title="A variable cannot be void",
        says="`{name}` is declared as void on line {line}.",
        explanation=(
            "void means 'nothing'. It works as a function's return type, but a "
            "variable has to hold something."
        ),
        fix="Use the type of the value it should hold, such as int or double.",
    ),
    Rule(
        code="E_BREAK_OUTSIDE_LOOP",
        title="break with no loop around it",
        says="Line {line} has a break or continue that is not inside a loop.",
        explanation=(
            "break leaves a loop and continue skips to its next round, so outside a "
            "loop they mean nothing. Usually a } a little above closed the loop too "
            "early."
        ),
        fix="Move it inside the loop, or move the loop's } down below it.",
    ),
    Rule(
        code="W_MISSING_RETURN",
        title="A function can end without returning",
        says="`{name}` can reach its last } without a return.",
        explanation=(
            "Its type says it hands back a value. On that path it hands back "
            "nothing, so the caller gets random leftovers -- often a different "
            "number on each run."
        ),
        fix="Make every path end in return with a value, including the very end.",
    ),
    Rule(
        code="W_UNINITIALIZED_READ",
        title="Variable used before it has a value",
        says="`{name}` is read on line {line} before anything is stored in it.",
        explanation=(
            "A new variable in C is not 0. It holds whatever was left in that "
            "memory, so the program can work on your machine and fail on another."
        ),
        fix="Give it a starting value where you declare it.",
        try_this="Give it a starting value where you declare it, like `int {name} = 0;`.",
    ),

    # -- the system compiler, as ground truth ------------------------------
    Rule(
        code="-Wimplicit-function-declaration",
        title="Missing #include",
        kind="include",
        says="`{name}` is used on line {line}, but the file that declares it was never included.",
        explanation=(
            "Functions like printf live in the standard library, and #include "
            "lines at the top tell C about them. Without the right one, C does not "
            "know `{name}` exists."
        ),
        fix="Add the #include for this function at the very top of the file.",
        try_this="Add `#include <{header}>` at the very top of the file.",
    ),
    Rule(
        code="-Wparentheses",
        title="= where == was probably meant",
        pattern=r"assignment as a condition",
        says="The condition on line {line} uses =, which stores a value instead of comparing.",
        explanation=(
            "= puts a value into a variable; == asks 'are these equal?'. With "
            "'if (x = 5)', x becomes 5 and the if is always true."
        ),
        fix="Use == to compare.",
    ),
    Rule(
        code="-Wempty-body",
        title="A stray ; ends the loop or if early",
        says="The ; at the end of line {line} is the whole body of the loop or if.",
        explanation=(
            "A ; straight after 'for (...)', 'while (...)' or 'if (...)' counts as "
            "an empty statement. The block below then runs once, on its own, not "
            "as part of the loop or if."
        ),
        fix="Delete the ; after the closing ).",
    ),
    Rule(
        code="-Warray-bounds",
        title="Index past the end of an array",
        extract=r"index (?P<index>-?\d+) is past the end of the array \(that has type '[^\[']*\[(?P<size>\d+)\]",
        says="Line {line} uses index {index}, but the array has {size} slots, numbered 0 to {last}.",
        explanation=(
            "Counting in C starts at 0, so the last slot is one less than the "
            "size. C will not stop you going past it; you just read or overwrite "
            "whatever memory comes next."
        ),
        fix="Keep indexes below the array's size.",
    ),
    Rule(
        code="-Wdivision-by-zero",
        title="Dividing by zero",
        says="Line {line} divides by zero.",
        explanation=(
            "Dividing a whole number by zero has no answer in C. The program "
            "either crashes or carries on with a garbage value."
        ),
        fix="Divide by a value that cannot be zero, or check it first.",
    ),
    Rule(
        code="clang:expected_semicolon_at_end_of_declaration",
        pattern=r"expected ';'",
        title="Missing semicolon",
        kind="semicolon",
        says="Line {line} is missing a ; at the end.",
        explanation=(
            "The ; is C's full stop. Without it, C runs on into the next line and "
            "gets confused there, so look at the end of this line or the one just "
            "above."
        ),
        fix="Put a ; at the end of the statement.",
        try_this="Make it `{fixed}`.",
    ),
    Rule(
        code="",
        pattern=r"use of undeclared identifier|undeclared \(first use",
        title="Unknown name",
        kind="undeclared",
        says="`{name}` is used on line {line}, but nothing called `{name}` has been declared.",
        explanation=(
            "C only knows names you have declared above the place you use them. "
            "The usual reasons are a typo, a missing declaration, or a variable "
            "declared inside other { } braces."
        ),
        fix="Fix the spelling, or declare it above this line.",
    ),
    Rule(
        code="",
        pattern=r"incompatible type|incompatible pointer|incompatible integer|conversion from",
        title="Wrong kind of value",
        says="Line {line}: {message}.",
        explanation=(
            "The two sides hold different kinds of things -- text and a number, or "
            "an address and a number -- and C will not turn one into the other "
            "for you."
        ),
        fix="Use a value of the variable's type, or change the variable's type.",
    ),
    Rule(
        code="-Wuninitialized",
        title="Variable used before it has a value",
        says="`{name}` is read on line {line} before anything is stored in it.",
        explanation=(
            "A new variable in C is not 0. It holds whatever was left in that "
            "memory, so the result can change from run to run."
        ),
        fix="Give it a starting value where you declare it.",
        try_this="Give it a starting value where you declare it, like `int {name} = 0;`.",
    ),
    Rule(
        code="-Wsometimes-uninitialized",
        title="Variable might not have a value",
        says="`{name}` only gets a value on some paths before line {line} reads it.",
        explanation=(
            "If the if or loop that sets it does not run, it still holds leftover "
            "memory when it is read."
        ),
        fix="Give it a starting value where you declare it.",
    ),
    Rule(
        code="-Wreturn-type",
        title="A function can end without returning",
        says="The function ending on line {line} can reach its } without a return.",
        explanation=(
            "Its type promises a value back. On that path it hands back nothing, "
            "so the caller gets random leftovers."
        ),
        fix="Make every path end in return with a value, including the very end.",
    ),
    Rule(
        code="-Wformat",
        title="printf/scanf format does not match",
        kind="format",
        extract=r"specifies type '(?P<want>[^']+)' but the argument has type '(?P<got>[^']+)'",
        says="On line {line} the format expects {want_said}, but the value given is {got_said}.",
        explanation=(
            "printf and scanf trust the % codes to say what each value is. Get one "
            "wrong and they read the value as the wrong kind of thing, printing "
            "nonsense or crashing."
        ),
        fix="Match each % code to its value: %d for int, %f for double, %s for text, %c for char.",
    ),
    Rule(
        code="R_SEGFAULT",
        title="Crashed: touched memory it should not",
        says="The program crashed with a segmentation fault.",
        explanation=(
            "It tried to use memory that is not its own. The usual reasons: an "
            "index past the end of an array, a pointer that was never set, or "
            "malloc returning NULL without a check."
        ),
        fix="Check array indexes against the length, and pointers against NULL, before using them.",
    ),
    Rule(
        code="R_ARITHMETIC",
        title="Crashed: divided by zero",
        says="The program crashed doing arithmetic, almost always a division by zero.",
        explanation=(
            "Dividing a whole number by zero has no answer, so the processor stops "
            "the program."
        ),
        fix="Check the number you divide by is not zero first.",
    ),
    Rule(
        code="R_TIMEOUT",
        title="Never finished",
        says="Still running after {timeout} seconds, so we stopped it.",
        explanation=(
            "Usually a loop whose condition never becomes false -- the variable it "
            "tests never changes inside the loop. It can also be waiting for "
            "typed input: when we run it, nothing is typed in."
        ),
        fix="Make sure each loop changes the variable its condition checks.",
    ),
    Rule(
        code="S_UNSAFE_INPUT",
        title="This function is unsafe",
        says="`{name}` has no way to know how big the destination is.",
        explanation=(
            "gets() and strcpy() keep copying for as long as the input goes on. "
            "Longer input than the buffer overwrites the memory after it -- a crash "
            "at best, a security hole at worst."
        ),
        fix="Use fgets() with the buffer size, or snprintf() with a limit.",
    ),
    Rule(
        code="R_DIVISION_BY_ZERO",
        title="Divided by zero",
        says="`{code}` on line {line} divided by zero when it ran.",
        explanation=(
            "There is no answer to that. Some machines crash; others, Apple "
            "Silicon included, carry on with a garbage number, which is worse "
            "because nothing visibly fails."
        ),
        fix="Check the number you divide by is not zero first.",
    ),
    Rule(
        code="R_INDEX_OUT_OF_BOUNDS",
        title="Went past the end of an array",
        extract=r"index (?P<index>-?\d+) out of bounds for type '[^\[']*\[(?P<size>\d+)\]",
        says="`{code}` on line {line} used index {index}, but the array has {size} slots, numbered 0 to {last}.",
        explanation=(
            "Counting starts at 0, so the last slot is one less than the size. A "
            "loop written with <= where it needs < is the usual cause."
        ),
        fix="Loop with i < size, not i <= size.",
    ),
    Rule(
        code="R_NULL_DEREFERENCE",
        title="Used a NULL pointer",
        says="`{code}` on line {line} used a pointer that points at nothing (NULL).",
        explanation=(
            "Either the pointer was never given a real address, or it came from "
            "something like malloc() or fopen() that returned NULL because it "
            "failed, and nobody checked."
        ),
        fix="Check the pointer is not NULL before using it.",
    ),
    Rule(
        code="R_SIGNED_OVERFLOW",
        title="Number too big for its type",
        says="The calculation on line {line} went past what an int can hold.",
        explanation=(
            "An int tops out around 2.1 billion. Going past that does not wrap "
            "neatly in C; the result can be anything."
        ),
        fix="Use a bigger type such as long long, or check the values first.",
    ),
    Rule(
        code="R_BAD_SHIFT",
        title="Shifted by too much",
        says="The << or >> on line {line} shifted by more bits than the number has.",
        explanation=(
            "An int has 32 bits, so shifting by 32 or more -- or by a negative "
            "amount -- has no defined result."
        ),
        fix="Keep the shift amount between 0 and 31 for an int.",
    ),
    Rule(
        code="R_MISALIGNED_ACCESS",
        title="Read memory at an odd address",
        says="Line {line} read a value from an address it cannot sit at.",
        explanation=(
            "Usually a pointer cast to the wrong type, such as treating part of a "
            "char buffer as an int."
        ),
        fix="Copy the bytes into a variable of the right type with memcpy() instead of casting.",
    ),
    Rule(
        code="R_UNDEFINED_BEHAVIOR",
        title="Did something C does not allow",
        says="Line {line}: {message}.",
        explanation=(
            "C leaves this case undefined, so the program may seem to work today "
            "and break with another compiler or on another machine."
        ),
        fix="Change that line so it stays within what C defines.",
    ),
    Rule(
        code="R_FORTIFY_ABORT",
        title="Stopped before overflowing a buffer",
        says="A safety check stopped the program just before it wrote past the end of a buffer.",
        explanation=(
            "Something was about to copy more characters than the destination has "
            "room for. The check stopped it before memory was damaged."
        ),
        fix="Make the buffer big enough, or use a size-limited function like snprintf().",
    ),
    Rule(
        code="R_ABORT",
        title="The program stopped itself",
        says="The program called abort().",
        explanation=(
            "Either an assert() failed, or the C library noticed memory had been "
            "damaged earlier -- freeing something twice, or writing past the end "
            "of malloc'd memory."
        ),
        fix="Look for a failed assert, a double free(), or a write past the end of a buffer.",
    ),
    Rule(
        code="R_BUS_ERROR",
        title="Crashed: bad memory address",
        says="The program crashed with a bus error.",
        explanation=(
            "It tried to use an address the hardware cannot use -- usually a "
            "pointer that was never set, or one cast to the wrong type."
        ),
        fix="Check every pointer is set to real memory before using it.",
    ),
    Rule(
        code="R_ILLEGAL_INSTRUCTION",
        title="Crashed: jumped into garbage",
        says="The program crashed trying to run something that is not code.",
        explanation=(
            "This almost always means memory was overwritten earlier, often by "
            "writing past the end of an array on the stack."
        ),
        fix="Check array writes stay inside the array.",
    ),
    Rule(
        code="",
        pattern=r"killed by signal",
        title="The program was stopped by the system",
        says="{message}.",
        explanation="The operating system ended the program before it finished.",
        fix="Run it again; if it repeats, look at what it was doing just before.",
    ),
    Rule(
        code="R_NONZERO_EXIT",
        title="Finished with an error code",
        says="The program ended with exit code {exit_code} instead of 0.",
        explanation=(
            "0 means 'went fine'; anything else means 'something failed'. That is "
            "fine if your program meant it -- for example, giving up because it "
            "got no input, which is what happens when we run it: nothing is typed "
            "in."
        ),
        fix="If that was not on purpose, find which return or exit() ran; return 0 from main on success.",
    ),
    Rule(
        code="nlpcc:no_compiler",
        title="No C compiler found",
        says="There is no C compiler on this machine, so the program could not be run.",
        explanation="Running your code needs clang or gcc installed.",
        fix="Install one (on a Mac: xcode-select --install), then try again.",
    ),
    Rule(
        code="nlpcc:build_failed",
        title="Could not build it to run",
        says="The program could not be built, so it was not run.",
        explanation="Fix the errors above first; it is run once it compiles.",
        fix="Fix the compile errors, then run it again.",
    ),
    Rule(
        code="clang:expected_rparen",
        title="A ( is never closed",
        says="Line {line} is missing a ) to close the ( on line {other_line}.",
        explanation=(
            "Brackets come in pairs. C reached the { or the end of the line "
            "still waiting for the )."
        ),
        fix="Add the ) that closes the condition or argument list.",
    ),
    Rule(
        code="clang:expected_rbrace",
        title="A { is never closed",
        says="The file ends with the { from line {other_line} still open.",
        explanation=(
            "Every { needs a }. The error points at the end because that is where "
            "C ran out of code; the missing } belongs after line {other_line}."
        ),
        fix="Add the } that closes that block.",
    ),
    Rule(
        code="",
        pattern=r"missing terminating|unterminated|invalid-pp-token|invalid pp",
        title="A quote is never closed",
        says="The text on line {line} is missing its closing quote.",
        explanation=(
            "Without the second quote, C reads the rest of the line as text, which "
            "is why one missing quote can cause several odd errors."
        ),
        fix="Add the closing quote at the end of the text.",
    ),
    Rule(
        code="",
        pattern=r"too few arguments|too many arguments",
        title="Wrong number of arguments",
        says="Line {line}: {message}.",
        explanation=(
            "C has no optional arguments. A function always takes exactly what its "
            "first line lists."
        ),
        fix="Pass exactly the values the function's definition lists, in order.",
    ),
    Rule(
        code="",
        pattern=r"redefinition of",
        title="Same name declared twice",
        says="`{name}` is declared again on line {line}.",
        explanation=(
            "Inside one pair of { } a name can only mean one thing. To change its "
            "value, leave out the type: `{name} = ...;`."
        ),
        fix="Remove the type from the second one so it just assigns, or pick a new name.",
    ),
    Rule(
        code="-Wunused-variable",
        title="Variable never used",
        says="`{name}` is set up on line {line} but never used.",
        explanation=(
            "It does no harm, but it is often a sign of a slip: a value you worked "
            "out and then forgot to print or return."
        ),
        fix="Use it, or delete the line.",
    ),
    Rule(
        code="clang:expected_expression",
        title="A value is missing",
        says="On line {line}, C wanted a value and found none.",
        explanation=(
            "Often an operator with nothing after it, or a knock-on effect of an "
            "earlier mistake like a missing quote."
        ),
        fix="Fix the first error in the file, then try again; this one often goes away with it.",
    ),
)


# -- filling in the details ----------------------------------------------------

_QUOTED_NAME = re.compile(r"'([A-Za-z_]\w*)'")
_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")

C_KEYWORDS = frozenset("""auto break case char const continue default do double
else enum extern float for goto if inline int long register restrict return
short signed sizeof static struct switch typedef union unsigned void volatile
while""".split())

# Names people bring from other languages, where "check your spelling" is the
# wrong advice: the spelling is fine, C just does not have it.
NOT_IN_C = {
    "string": ("C has no string type. Text is an array of char.",
               "Write `char {var}[] = ...;` or `const char *{var} = ...;`."),
    "bool": ("bool is not built into C; it comes from <stdbool.h>.",
             "Add `#include <stdbool.h>` at the top."),
    "true": ("true and false come from <stdbool.h>.",
             "Add `#include <stdbool.h>` at the top."),
    "false": ("true and false come from <stdbool.h>.",
              "Add `#include <stdbool.h>` at the top."),
    "NULL": ("NULL comes from a header such as <stdio.h> or <stddef.h>.",
             "Add `#include <stddef.h>` at the top."),
    "cout": ("cout is C++, not C. C prints with printf.",
             "Use `printf(\"...\\n\");` from <stdio.h>."),
    "cin": ("cin is C++, not C. C reads input with scanf or fgets.",
            "Use `scanf(\"%d\", &value);` from <stdio.h>."),
    "endl": ("endl is C++, not C. In C, put \\n inside the text.",
             "Use `printf(\"...\\n\");`."),
    "nullptr": ("nullptr is C++. C uses NULL.", "Use NULL instead."),
}

HEADERS = {
    "stdio.h": "printf puts putchar scanf fgets fputs getchar fprintf sprintf "
               "snprintf fopen fclose fread fwrite perror fflush sscanf",
    "stdlib.h": "malloc calloc realloc free exit atoi atof strtol strtod rand "
                "srand abs qsort bsearch system getenv",
    "string.h": "strlen strcpy strncpy strcat strncat strcmp strncmp strchr "
                "strrchr strstr memset memcpy memmove memcmp strdup",
    "math.h": "sqrt pow fabs sin cos tan floor ceil round log log10 exp fmod hypot",
    "ctype.h": "isdigit isalpha isalnum isspace isupper islower toupper tolower ispunct",
    "time.h": "time clock difftime localtime strftime",
    "stdbool.h": "bool true false",
}
_HEADER_OF = {fn: header for header, fns in HEADERS.items() for fn in fns.split()}

# What a type is called in words, and the % code that prints it.
_TYPE_WORDS = {
    "int": ("a whole number (int)", "%d"),
    "unsigned int": ("an unsigned whole number", "%u"),
    "long": ("a long", "%ld"),
    "long long": ("a long long", "%lld"),
    "unsigned long": ("an unsigned long", "%lu"),
    "size_t": ("a size_t", "%zu"),
    "double": ("a decimal number (double)", "%f"),
    "float": ("a decimal number (float)", "%f"),
    "char": ("a single character (char)", "%c"),
    "char *": ("text (char *)", "%s"),
    "const char *": ("text (char *)", "%s"),
}


def _type_said(name: str) -> str:
    if name in _TYPE_WORDS:
        return _TYPE_WORDS[name][0]
    if name.endswith("*"):
        return f"the address of a {name[:-1].strip()} ({name})"
    return name


def _source_text(diag: Diagnostic) -> str:
    """The file, with string contents blanked so words in them are not names."""
    try:
        text = Path(diag.file).read_text(errors="replace")
    except (OSError, TypeError, ValueError):
        return ""
    return re.sub(r'"(\\.|[^"\\\n])*"', '""', text)


def _depth_at(text: str, line: int) -> int:
    """How many { are open at the start of ``line``."""
    before = "\n".join(text.splitlines()[: line - 1])
    return before.count("{") - before.count("}")


def _details(diag: Diagnostic, rule: Rule) -> dict[str, str]:
    """Everything a template may name, taken from this one diagnostic."""
    values: dict[str, str] = {"message": diag.message.rstrip(".")}
    if diag.position is not None:
        values["line"] = str(diag.position.line)
    code = (diag.snippet or "").strip()
    if code:
        values["code"] = code
    named = _QUOTED_NAME.search(diag.message)
    if named:
        values["name"] = named.group(1)
    for key in ("function", "names", "exit_code"):
        if key in diag.meta:
            values[key] = str(diag.meta[key])
    if isinstance(diag.meta.get("timeout"), (int, float)):
        values["timeout"] = f"{diag.meta['timeout']:g}"
    related = diag.meta.get("related") or []
    if related and related[0].get("line"):
        values["other_line"] = str(related[0]["line"])

    if rule.extract:
        found = re.search(rule.extract, diag.message, re.I)
        if found:
            values.update({k: v for k, v in found.groupdict().items() if v})
    if values.get("size", "").isdigit():
        values["last"] = str(int(values["size"]) - 1)
    for key in ("takes", "passed"):
        if key in values:
            values[f"{key}_said"] = f"{values[key]} value{'' if values[key] == '1' else 's'}"

    if rule.kind == "semicolon" and code and not code.endswith((";", "{", "}")):
        values["fixed"] = code + ";"
    elif rule.kind == "include" and values.get("name") in _HEADER_OF:
        values["header"] = _HEADER_OF[values["name"]]
    elif rule.kind == "format" and "want" in values and "got" in values:
        values["want_said"] = _type_said(values["want"])
        values["got_said"] = _type_said(values["got"])
    return values


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _fill(template: str, values: dict[str, str]) -> str | None:
    """The template with its details in, or None if any detail is missing.

    Not str.format: the text is about C, so it is full of literal braces.
    """
    if not template:
        return None
    if any(f not in values for f in _PLACEHOLDER.findall(template)):
        return None
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def _undeclared(diag: Diagnostic, values: dict[str, str]) -> dict[str, str]:
    """Why this name is unknown, when the file itself says why."""
    name, line = values["name"], values.get("line", "?")
    if name in NOT_IN_C:
        reason, fix = NOT_IN_C[name]
        var = "name"
        after = re.search(r"\bstring\s+([A-Za-z_]\w*)", values.get("code", ""))
        if name == "string" and after:
            var = after.group(1)
        return {"says": f"`{name}` is used on line {line}.", "why": reason,
                "fix": fix.format(var=var)}
    if name in _HEADER_OF:
        header = _HEADER_OF[name]
        return {"says": f"`{name}` is used on line {line}, but <{header}> was never included.",
                "why": ("It comes from the standard library, and the #include lines "
                        "at the top are what tell C about it."),
                "fix": f"Add `#include <{header}>` at the very top of the file."}

    text = _source_text(diag)
    use_line = diag.position.line if diag.position else 0
    declaration = re.compile(
        r"^\s*(?:const\s+|unsigned\s+|static\s+)*(?P<type>[A-Za-z_]\w*)[\s*]+"
        + re.escape(name) + r"\s*(?:=|;|\[|,)")
    for number, source_line in enumerate(text.splitlines()[: max(use_line - 1, 0)], 1):
        found = declaration.match(source_line)
        if not found or found.group("type") in ("return",):
            continue
        kind = found.group("type")
        if kind in NOT_IN_C:
            # string name = ...; fails, so every later use of name fails too.
            return {"says": (f"`{name}` is declared on line {number}, but that line "
                             f"has its own error (`{kind}` is not C), so `{name}` was "
                             "never created."),
                    "why": "Fix that line and this error goes away with it.",
                    "fix": f"Fix line {number} first."}
        if _depth_at(text, use_line) < _depth_at(text, number):
            return {"says": (f"`{name}` is declared on line {number}, inside a {{ }} "
                             f"block that has already closed by line {line}."),
                    "why": ("A variable only exists until the } of the block it was "
                            "declared in."),
                    "fix": (f"Declare `{name}` above that block, then give it its "
                            "value inside.")}
        break

    names = set(_IDENTIFIER.findall(text)) - C_KEYWORDS - {name}
    close = difflib.get_close_matches(name, names, n=1, cutoff=0.75)
    if close:
        return {"suggestion": close[0], "fix": f"Change `{name}` to `{close[0]}`."}
    return {}


def _special_cases(diag: Diagnostic, rule: Rule, values: dict[str, str]) -> dict[str, str]:
    """Better words than the templates for a few common situations.

    Keys: "says" and "why" replace those parts, "fix" the fix, and
    "suggestion" adds a "did you mean".
    """
    if rule.kind == "undeclared" and "name" in values:
        return _undeclared(diag, values)
    if rule.kind == "format" and values.get("want", "").endswith("*") \
            and not values.get("got", "").endswith("*"):
        found = re.search(r'scanf\s*\(\s*"[^"]*"\s*,\s*([A-Za-z_]\w*)\s*[,)]',
                          values.get("code", ""))
        var = found.group(1) if found else "count"
        return {"says": (f"On line {values.get('line', '?')}, scanf is given `{var}`, "
                         f"but it needs `&{var}` -- where to put what it reads."),
                "why": ("Without the &, scanf gets the number stored in the variable "
                        "and treats it as a place in memory to write to."),
                "fix": f"Write `&{var}` instead of `{var}`."}
    if rule.kind == "format" and values.get("got") in _TYPE_WORDS:
        spec = _TYPE_WORDS[values["got"]][1]
        return {"fix": f"Use {spec} for {_type_said(values['got'])}."}
    return {}


def explain(diag: Diagnostic) -> Diagnostic:
    """Attach an explanation to one diagnostic, in place, and return it.

    Idempotent, and it has to be: quality findings arrive from
    nlpcc/quality/rules.py already carrying their own explanation, and the
    pipeline runs this pass over every diagnostic regardless of source.
    Without the early return, no rule here matches a quality code and the
    finding gets stamped "none" -- marking an explained finding unexplained.
    """
    if diag.explanation:
        return diag

    for rule in RULES:
        if rule.matches(diag):
            values = _details(diag, rule)
            special = _special_cases(diag, rule, values)
            says = special.get("says") or _fill(rule.says, values)
            if says and special.get("suggestion"):
                says += f" Did you mean `{special['suggestion']}`?"
            why = (special.get("why") or _fill(rule.explanation, values)
                   or _generic(rule.explanation))
            diag.explanation = f"{says} {why}" if says else why
            diag.fix = (special.get("fix") or _fill(rule.try_this, values)
                        or _generic(rule.fix))
            diag.meta["title"] = rule.title
            diag.meta["explained_by"] = "rules"
            return diag
    diag.meta["explained_by"] = "none"   # candidate for the LLM layer
    return diag


def _generic(text: str) -> str:
    """A template read without its details: placeholders become plain words."""
    words = {"name": "it", "other_line": "the opening one"}
    return _PLACEHOLDER.sub(lambda m: words.get(m.group(1), "that"), text)


def explain_all(diags: Iterable[Diagnostic]) -> list[Diagnostic]:
    return [explain(d) for d in diags]


def unexplained(diags: Iterable[Diagnostic]) -> list[Diagnostic]:
    """Diagnostics no rule covered -- the backlog of rules still to write."""
    return [d for d in diags if d.meta.get("explained_by") in (None, "none")]
