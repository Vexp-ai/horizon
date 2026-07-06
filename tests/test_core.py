"""Prototype test suite — pure logic, no GPU/network/models.

Run:  .venv/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import pytest

# ---------- common.text ----------

def test_extract_boxed_nested():
    from common.text import extract_boxed

    assert extract_boxed(r"x \boxed{\frac{1}{2}} y") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{a{b}c}") == "a{b}c"
    assert extract_boxed("niente") is None


def test_extract_final_answer_formats():
    from common.text import extract_final_answer

    assert extract_final_answer("blah\n#### 42") == "42"
    assert extract_final_answer(r"<think>x</think> answer is \boxed{17}") == "17"
    assert extract_final_answer("The answer is 12.") == "12"


def test_extract_code_last_block():
    from common.text import extract_code

    t = "```python\nprint(1)\n```\ntesto\n```python\nprint(2)\n```"
    assert extract_code(t) == "print(2)"


# ---------- verify_math ----------

def test_self_consistency_majority():
    from harness.verify_math import self_consistency

    sel = self_consistency([r"\boxed{7}", r"\boxed{7}", r"\boxed{5}"])
    assert sel.answer == "7" and sel.votes == 2


def test_verify_answer_equivalence():
    from harness.verify_math import verify_answer

    assert verify_answer("0.5", "1/2") or verify_answer("1/2", "0.5")
    assert not verify_answer("3", "4")


# ---------- router heuristics ----------

def test_code_heuristic_positive():
    from router.heuristics import detect_code

    assert detect_code("Write a python function to reverse a list.")
    assert detect_code("def foo(x):\n    ...")
    assert detect_code("Your code should pass this test:\nassert f(1)==2")


def test_code_heuristic_negative():
    from router.heuristics import detect_code

    assert not detect_code("What is the boiling point of water at 2 atm?")
    assert not detect_code("Compute 3+4*7 step by step.")


def test_code_heuristic_competitive_programming():
    """v1.2: LCB-style problems MUST go to code (59/100 misrouted in v1.1)."""
    from router.heuristics import detect_code

    # the real case of the measured failure (2026-07-04): LIS with I/O format
    assert detect_code("Given an array of n integers, find the longest increasing "
                       "subsequence. Input: first line n, second line the array. "
                       "Output the length.")
    assert detect_code("You are given a tree with n nodes.\n\nInput\n\nThe first "
                       "line contains one integer n.\n\nOutput\n\nPrint one integer.")
    assert detect_code("Count the beautiful pairs.\n\nExample 1:\n\nInput: nums = "
                       "[2,5,1,4]\nOutput: 5\n\nConstraints:\n\n1 <= nums.length")
    assert detect_code("Return the maximum number of operations you can perform.")


def test_code_heuristic_cp_markers_no_math_false_positives():
    """The v1.2 markers must not capture typical math problems."""
    from router.heuristics import detect_code

    assert not detect_code("Find the sum of all positive integers n such that "
                           "n^2 + 85n + 2017 is a perfect square.")
    assert not detect_code("Let $f(x) = x^2 - 3x$. For what values of x is "
                           "$f(f(x)) = f(x)$? Express your answer as a list.")
    assert not detect_code("A rectangle has perimeter 30 and area 50. What is the "
                           "length of its diagonal, expressed to two decimals?")


# ---------- MC extraction ----------

@pytest.mark.parametrize("text,expected", [
    (r"\boxed{C}", "C"),
    ("The final answer is (B).", "B"),
    ("**D.** perche'...", "D"),
    ("Answer: A", "A"),
    ("nessuna lettera", None),
])
def test_mc_extraction(text, expected):
    from eval.run_benchmarks import extract_mc_letter

    assert extract_mc_letter(text) == expected


# ---------- gen_tests (CodeT) ----------

def test_dual_execution_picks_correct():
    from harness.gen_tests import select_by_dual_execution

    good = "def add(a,b):\n    return a+b"
    bad = "def add(a,b):\n    return a-b"
    tests = ["assert add(1,2)==3", "assert add(0,0)==0", "assert add(2,2)==5"]
    r = select_by_dual_execution([bad, good], tests)
    assert r.best_index == 1


def test_extract_asserts():
    from harness.gen_tests import extract_asserts

    got = extract_asserts("```python\nassert f(1)==2\nx=1\nassert f(2)==4\n```")
    assert got == ["assert f(1)==2", "assert f(2)==4"]


# ---------- prm_select ----------

def test_prm_select_argmax_and_weighted():
    from harness.prm_select import select

    answers = ["5", "7", "7"]
    scores = [[0.9, 0.9], [0.6, 0.4], [0.5, 0.6]]
    i, ans = select(answers, scores, "prm_argmax_mean")
    assert ans == "5"  # the best sample individually (0.9)
    i, ans = select(answers, scores, "prm_weighted_mean")
    assert ans == "7"  # but weighted voting rewards the shared answer (0.5+0.55 > 0.9)


def test_prm_parse_strategy_rejects_unknown():
    from harness.prm_select import parse_strategy

    with pytest.raises(ValueError):
        parse_strategy("prm_magic_max")


# ---------- verify_code I/O (LCB) ----------

def test_io_stdin_mode():
    from harness.verify_code import run_io_test

    prog = "n=input()\nprint(int(n)*2)"
    assert run_io_test(prog, {"input": "21\n", "output": "42"}, "stdin", None)
    assert not run_io_test(prog, {"input": "21\n", "output": "43"}, "stdin", None)


def test_io_call_mode():
    from harness.verify_code import run_io_test

    sol = "class Solution:\n    def addTwo(self, a, b):\n        return a+b"
    assert run_io_test(sol, {"input": "3\n4", "output": "7"}, "call", "addTwo")
    assert not run_io_test(sol, {"input": "3\n4", "output": "8"}, "call", "addTwo")


# ---------- repair loop (8d) ----------

class _FakeClient:
    """Scripted client: each call to generate consumes the next list of texts."""

    def __init__(self, scripted: list[list[str]]):
        self.scripted = list(scripted)
        self.calls: list[dict] = []

    def generate(self, messages, **kw):
        from common.model_client import GenResult

        self.calls.append({"messages": messages, **kw})
        texts = self.scripted.pop(0)
        return [GenResult(text=t, prompt_tokens=10, completion_tokens=20) for t in texts]


_GOOD = "```python\ndef add(a,b):\n    return a+b\n```"
_BAD = "```python\ndef add(a,b):\n    return a-b\n```"
_TESTS = ["assert add(1,2)==3", "assert add(0,5)==5"]


def test_repair_loop_fixes_and_reports_concrete_feedback():
    from harness.repair_loop import assert_tester, repair_solve

    client = _FakeClient([[_BAD], [_BAD], [_GOOD]])  # burst 1 + burst k-1 + repair 1
    out = repair_solve(client, "m", "Write add(a,b).", assert_tester(_TESTS),
                       gen_cfg={"temperature": 0.7, "top_p": 0.95, "max_tokens": 512})
    assert out.report.passed_all and out.iterations == 1
    assert "return a+b" in out.candidates[out.best_index]
    # the repair message contains the CONCRETE failure (test + previous code)
    repair_msg = client.calls[2]["messages"][-1]["content"]
    assert "assert add(1,2)==3" in repair_msg and "return a-b" in repair_msg


def test_repair_loop_early_exit_on_first_pass():
    from harness.repair_loop import assert_tester, repair_solve

    client = _FakeClient([[_GOOD]])
    out = repair_solve(client, "m", "Write add(a,b).", assert_tester(_TESTS),
                       gen_cfg={"max_tokens": 512})
    assert out.report.passed_all and out.iterations == 0
    assert len(client.calls) == 1  # adaptive budget: only 1 generation on the easy case


def test_repair_loop_caps_iterations_and_keeps_best():
    from harness.repair_loop import assert_tester, repair_solve

    client = _FakeClient([[_BAD]] * 6)  # never solves it
    out = repair_solve(client, "m", "Write add(a,b).", assert_tester(_TESTS),
                       gen_cfg={"max_tokens": 512}, initial_k=2, max_iters=4)
    assert not out.report.passed_all and out.iterations == 4
    assert len(client.calls) == 6  # 2 bursts + 4 repairs: budget respected


def test_io_tester_feedback_contains_expected_vs_got():
    from harness.repair_loop import io_tester

    report = io_tester([{"input": "21\n", "output": "42"}], "stdin", None)(
        "n=input()\nprint(int(n)*2+1)")
    assert report.n_passed == 0 and report.n_total == 1
    assert "42" in report.feedback and "43" in report.feedback  # expected vs got


# ---------- decontamination ----------

def test_decontamination_matching():
    from data.decontaminate import exact_hash, is_contaminated, ngrams

    a = "find all primes p such that p squared plus two is prime"
    grams = [ngrams(a, 6)]
    exact = {exact_hash(a)}
    assert is_contaminated(a, exact, grams, 6, 0.6) == "exact"
    assert is_contaminated("compute the derivative of sin x", exact, grams, 6, 0.6) is None
