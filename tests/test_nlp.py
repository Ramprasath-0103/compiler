

from __future__ import annotations

import pytest

from nlpcc.nlp import detect
from nlpcc.nlp.intent import (ANALYZE, EDIT, EXPLAIN, GENERATE, OPTIMIZE, QUESTION,
                              RUN, SCORE, tokenize, _stem)


@pytest.mark.parametrize("text,expected", [
    ("why does this not compile?", EXPLAIN),
    ("what does this error mean", EXPLAIN),
    ("help me understand what is wrong", EXPLAIN),
    ("explain the error in plain english", EXPLAIN),
    ("write me a program that sorts an array", GENERATE),
    ("let ai build my code for a calculator", GENERATE),
    ("generate a program that reads a file", GENERATE),

    ("give me correct code", GENERATE),
    ("give code to sum two numbers", GENERATE),
    ("i need code that sorts an array", GENERATE),
    ("show me code for bubble sort", GENERATE),
    ("how good is my code", SCORE),
    ("rate my code", SCORE),
    ("make it faster", OPTIMIZE),
    ("how can i improve this", OPTIMIZE),
    ("make this easier to read", OPTIMIZE),
    ("run it and show the output", RUN),
    ("what does it print", RUN),
    ("what does this code do", QUESTION),
    ("tell me about this code", QUESTION),
    ("what is a pointer", QUESTION),
    ("how do for loops work", QUESTION),
    ("is this correct?", QUESTION),
    ("what does printf do", QUESTION),
    ("fix it", EDIT),
    ("fix the error", EDIT),
    ("can you correct my code", EDIT),
    ("make the code shorter", EDIT),
    ("convert this to use a while loop", EDIT),
    ("add comments", EDIT),
])
def test_requests_route_to_the_right_stage(text: str, expected: str) -> None:
    assert detect(text).action == expected


def test_objectives_are_pulled_out_of_the_sentence() -> None:
    intent = detect("make it faster and use less memory")
    assert intent.action == OPTIMIZE
    assert set(intent.objectives) == {"speed", "memory"}


def test_readability_and_maintainability_are_recognised() -> None:
    assert "readability" in detect("make this cleaner").objectives
    assert "maintainability" in detect("make it more maintainable").objectives


def test_description_drops_the_request_wrapper() -> None:

    intent = detect("please write me a program that reverses a string")
    assert intent.action == GENERATE
    assert "write" not in intent.description
    assert "reverses a string" in intent.description


def test_an_empty_editor_means_generate() -> None:
    intent = detect("a program that adds two numbers", has_code=False)
    assert intent.action == GENERATE


def test_a_question_with_an_empty_editor_is_still_a_question() -> None:
    assert detect("what is a pointer", has_code=False).action == QUESTION


def test_fix_it_with_an_empty_editor_does_not_write_a_program() -> None:
    assert detect("fix it", has_code=False).action == EDIT


def test_nonsense_is_admitted_rather_than_guessed() -> None:
    intent = detect("banana banana banana")
    assert intent.confidence == 0.0
    assert not intent.certain
    assert intent.action == ANALYZE         
    assert intent.because


def test_plain_requests_for_code_are_confident() -> None:
    for text in ("give me correct code",
                 "give code to sum two numbers",
                 "i need code that sorts an array"):
        intent = detect(text)
        assert intent.action == GENERATE, text
        assert intent.certain, f"{text} -> only {intent.confidence:.0%}"


def test_the_task_survives_a_give_me_wrapper() -> None:
    assert "sum two numbers" in detect("give code to sum two numbers").description
    assert "bubble sort" in detect("show me code for bubble sort").description


def test_it_can_say_why() -> None:
    intent = detect("why does this not compile")
    assert intent.because
    assert intent.confidence > 0.5
    assert intent.summary


def test_stemming_groups_word_forms() -> None:
    assert _stem("writing") == "writ"
    assert _stem("generated") == "generat"
    assert _stem("is") == "is"              


def test_tokenizer_drops_punctuation() -> None:
    assert tokenize("Why does this, not compile?!") == \
        ["why", "does", "this", "not", "compile"]


def test_detection_is_instant_and_offline() -> None:
    import time
    start = time.time()
    for _ in range(1000):
        detect("make it faster")
    assert time.time() - start < 1.0
