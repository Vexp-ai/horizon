"""Phase 8d: agentic self-repair loop for the code path (§7.3 extended).

Idea (Self-Debugging/Reflexion, adapted to consumer hardware): instead of blind
best-of-N, generate a few candidates, EXECUTE the frozen tests and, if they fail,
re-iterate giving the model the CONCRETE failure (failing test, expected vs
obtained, traceback). The search becomes sequential and guided.

Three design rules (from the 8d evaluation):
  1. FROZEN TESTS: the tests are fixed BEFORE the loop (public/I-O, or CodeT
     generated only once). The loop may touch only the code, never the tests.
  2. OBJECTIVE FEEDBACK: never prose self-critique; only execution outcomes
     (with small models self-repair only works with concrete feedback).
  3. ADAPTIVE AND HONEST BUDGET: early-exit as soon as a candidate passes everything
     (a single generation on the easy problem); total cap
     initial_k + max_iters generations <= best_of_n of the code path (§15.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from harness import verify_code

REPAIR_PROMPT = (
    "Your previous solution FAILED execution testing "
    "({n_passed}/{n_total} tests passed).\n\n"
    "Previous code:\n```python\n{code}\n```\n\n"
    "Concrete failure:\n{feedback}\n\n"
    "Find the bug, fix it, and return ONE complete corrected ```python``` block."
)


@dataclass
class TestReport:
    n_passed: int
    n_total: int
    feedback: str | None = None   # concrete description of the FIRST failure

    @property
    def passed_all(self) -> bool:
        return self.n_total > 0 and self.n_passed == self.n_total

    @property
    def frac(self) -> float:
        return self.n_passed / self.n_total if self.n_total else 0.0


Tester = Callable[[str], TestReport]


def assert_tester(tests: list[str], timeout: int = 10) -> Tester:
    """Tester for assert-style tests (public HumanEval/MBPP or frozen CodeT)."""

    def run(code: str) -> TestReport:
        passed, fb = 0, None
        for t in tests:
            r = verify_code.execute(code, [t], timeout=timeout)
            if r.passed:
                passed += 1
            elif fb is None:
                err = (r.stderr or r.stdout or "unknown error").strip()[-800:]
                fb = f"Failing test: {t}\nError:\n{err}"
        return TestReport(passed, len(tests), fb)

    return run


def io_tester(tests: list[dict], mode: str, fn_name: str | None,
              timeout: int = 10) -> Tester:
    """Tester for I/O tests (LiveCodeBench): stdin/stdout or call-based."""

    def run(code: str) -> TestReport:
        passed, fb = 0, None
        for t in tests:
            ok, detail = verify_code.run_io_test_detail(code, t, mode, fn_name, timeout)
            if ok:
                passed += 1
            elif fb is None:
                fb = (
                    f"Failing test — input:\n{str(t.get('input', ''))[:400]}\n"
                    f"Expected output:\n{str(t.get('output', ''))[:400]}\n"
                    f"Your program produced:\n{detail[:800]}"
                )
        return TestReport(passed, len(tests), fb)

    return run


@dataclass
class RepairOutcome:
    samples: list[str]        # raw texts (burst + repair), in order
    candidates: list[str]     # extracted code, aligned with samples
    best_index: int
    report: TestReport        # report of the best candidate
    iterations: int           # repair iterations consumed
    total_tokens: int


def repair_solve(client, model: str | None, prompt: str, tester: Tester, *,
                 gen_cfg: dict, initial_k: int = 2, max_iters: int = 4,
                 max_tokens: int | None = None, reasoning_effort: str | None = None,
                 system_prompt: str | None = None) -> RepairOutcome:
    """Generate→execute→repair loop. `tester` encapsulates the ALREADY frozen tests."""
    from common.text import extract_code

    def messages(extra: str | None = None) -> list[dict]:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        # repair feedback goes in the SAME user turn: strict chat templates
        # (Mistral family) reject consecutive same-role messages with HTTP 400,
        # and permissive ones (Qwen/R1) render the merge identically anyway
        content = prompt if not extra else prompt + "\n\n" + extra
        msgs.append({"role": "user", "content": content})
        return msgs

    mt = max_tokens or gen_cfg.get("max_tokens", 8192)
    gen_kw = dict(model=model, temperature=gen_cfg.get("temperature", 0.7),
                  top_p=gen_cfg.get("top_p", 0.95), max_tokens=mt,
                  reasoning_effort=reasoning_effort)
    samples: list[str] = []
    candidates: list[str] = []
    reports: list[TestReport] = []
    tokens = 0

    def add(results) -> bool:
        """Records results and tests them; True if one passes all the tests."""
        nonlocal tokens
        done = False
        for r in results:
            tokens += r.prompt_tokens + r.completion_tokens
            samples.append(r.text)
            code = extract_code(r.text) or r.text
            candidates.append(code)
            reports.append(tester(code))
            done = done or reports[-1].passed_all
        return done

    def outcome(iterations: int) -> RepairOutcome:
        bi = max(range(len(reports)),
                 key=lambda i: (reports[i].passed_all, reports[i].frac))
        return RepairOutcome(samples=samples, candidates=candidates, best_index=bi,
                             report=reports[bi], iterations=iterations,
                             total_tokens=tokens)

    # adaptive initial burst: 1 right away (fast path), the rest only if needed
    if add(client.generate(messages(), n=1, **gen_kw)):
        return outcome(0)
    if reports[0].n_total == 0:
        # no frozen tests: there is no signal to iterate on (the caller
        # should fall back to the classic best-of-N path)
        return outcome(0)
    if initial_k > 1 and add(client.generate(messages(), n=initial_k - 1, **gen_kw)):
        return outcome(0)

    # repair loop: concrete feedback from the current BEST candidate
    for it in range(1, max_iters + 1):
        bi = max(range(len(reports)), key=lambda i: reports[i].frac)
        extra = REPAIR_PROMPT.format(
            n_passed=reports[bi].n_passed, n_total=reports[bi].n_total,
            code=candidates[bi][:6000],
            feedback=reports[bi].feedback or "The code failed the tests.",
        )
        if add(client.generate(messages(extra), n=1, **gen_kw)):
            return outcome(it)
    return outcome(max_iters)
