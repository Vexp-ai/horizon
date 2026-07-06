"""Auto-generated tests for the code verifier (§7.3 "auto-generated tests", CodeT style).

PROBLEM SOLVED: where the benchmark provides no public tests (HumanEval), internal
selection today is just "first one that compiles". Here the model generates K test
suites from the statement ALONE; the candidates are executed against the generated
tests and selection uses "dual execution agreement" (CodeT): the
(candidate, test-set) pair with the widest consensus wins.

HONESTY (§15.3): the generated tests derive only from the statement — no ground
truth. Benchmark scoring stays on the hidden tests.

AUTHORITY RULE: if real public tests exist, they keep priority;
the generated ones are added only as an extra signal.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from harness import verify_code

GEN_TESTS_PROMPT = (
    "Write {k} independent `assert` test cases for the function described below. "
    "Output ONLY a ```python``` block containing the assert statements (no function "
    "implementation, no prose). Base the tests strictly on the specification.\n\n"
    "Specification:\n{spec}"
)

_ASSERT_RE = re.compile(r"^\s*assert\s+.+", re.MULTILINE)


def extract_asserts(text: str) -> list[str]:
    """Extracts the `assert ...` lines from a generated block."""
    from common.text import extract_code

    block = extract_code(text) or text
    return [m.group(0).strip() for m in _ASSERT_RE.finditer(block)][:12]


@dataclass
class DualExecResult:
    best_index: int
    consensus: float          # fraction of generated tests passed by the winner
    n_tests_used: int


def generate_test_suites(client, model: str, spec: str, k_suites: int = 2,
                         asserts_per_suite: int = 6, gen_cfg: dict | None = None) -> list[str]:
    """Asks the model for k_suites independent assert blocks. Returns
    the deduplicated list of asserts."""
    g = gen_cfg or {}
    seen: set[str] = set()
    out: list[str] = []
    results = client.generate(
        [{"role": "user", "content": GEN_TESTS_PROMPT.format(k=asserts_per_suite, spec=spec)}],
        model=model,
        temperature=g.get("temperature", 0.7),
        top_p=g.get("top_p", 0.95),
        max_tokens=g.get("max_tokens_tests", 1024),
        n=k_suites,
    )
    for r in results:
        for a in extract_asserts(r.text):
            if a not in seen:
                seen.add(a)
                out.append(a)
    return out


def select_by_dual_execution(candidates: list[str], gen_tests: list[str],
                             timeout: int = 10) -> DualExecResult:
    """CodeT-style: runs every candidate on EVERY generated assert (individually,
    so one wrong assert does not zero everything out). Candidate score = set of
    passed asserts; candidates that agree on the same set reinforce each other
    (consensus group). The group with the maximum |group| * |passed asserts| wins.
    """
    if not candidates:
        return DualExecResult(0, 0.0, 0)
    if not gen_tests:
        return DualExecResult(0, 0.0, 0)

    passed_sets: list[frozenset[int]] = []
    for c in candidates:
        ok: set[int] = set()
        for i, t in enumerate(gen_tests):
            if verify_code.execute(c, [t], timeout=timeout).passed:
                ok.add(i)
        passed_sets.append(frozenset(ok))

    groups: dict[frozenset[int], list[int]] = defaultdict(list)
    for idx, ps in enumerate(passed_sets):
        groups[ps].append(idx)
    # CodeT score: (number of agreeing candidates) * (number of passed tests)
    best_set, members = max(groups.items(), key=lambda kv: (len(kv[1]) * len(kv[0]), len(kv[0])))
    best_index = members[0]
    return DualExecResult(
        best_index=best_index,
        consensus=len(best_set) / max(len(gen_tests), 1),
        n_tests_used=len(gen_tests),
    )
