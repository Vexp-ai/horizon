"""Code verification (§7.3).

  - INTERNAL selection (§7.3, §15.3): run the candidates against the PUBLIC/
    sample tests (or auto-generated ones) in a SANDBOX; keep the candidate that
    passes the most; retry with the error feedback. Does NOT use the hidden tests.
  - Benchmark SCORING: run against the benchmark's HIDDEN tests.

SECURITY (§2, §7.2, §15): generated code must NEVER run outside a sandbox.
Here the default isolation is a subprocess with resource limits (rlimit) and
a timeout. For serious runs use Docker: set HORIZON_SANDBOX=docker and provide
an image with --docker-image (see run_in_docker).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT = 10


@dataclass
class ExecResult:
    passed: bool
    stdout: str
    stderr: str
    timed_out: bool = False


_RLIMIT_PREAMBLE = textwrap.dedent(
    """
    import resource, sys
    # cap memory (~512MB) and CPU to contain hostile/buggy code
    try:
        resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    except Exception:
        pass
    """
)


def _build_program(solution: str, tests: list[str]) -> str:
    """Concatenates solution + tests into a single executable program."""
    test_block = "\n\n".join(tests)
    return f"{_RLIMIT_PREAMBLE}\n{solution}\n\n# --- tests ---\n{test_block}\n"


def run_local(solution: str, tests: list[str], timeout: int = DEFAULT_TIMEOUT) -> ExecResult:
    """Runs in an isolated subprocess (rlimit + timeout). 'Floor' sandbox."""
    program = _build_program(solution, tests)
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "prog.py"
        script.write_text(program)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True, text=True, timeout=timeout,
                cwd=td, env={"PATH": os.environ.get("PATH", "")},
            )
            return ExecResult(proc.returncode == 0, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired:
            return ExecResult(False, "", "TimeoutExpired", timed_out=True)


def run_in_docker(solution: str, tests: list[str], image: str,
                  timeout: int = DEFAULT_TIMEOUT) -> ExecResult:
    """Runs in an isolated Docker container (network off, ro, no privileges)."""
    program = _build_program(solution, tests)
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "prog.py"
        script.write_text(program)
        cmd = [
            "docker", "run", "--rm", "--network", "none",
            "--memory", "512m", "--cpus", "1", "--pids-limit", "128",
            "-v", f"{td}:/work:ro", "-w", "/work", image,
            "python", "-I", "prog.py",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
            return ExecResult(proc.returncode == 0, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired:
            return ExecResult(False, "", "TimeoutExpired", timed_out=True)


def execute(solution: str, tests: list[str], timeout: int = DEFAULT_TIMEOUT,
            docker_image: str | None = None) -> ExecResult:
    """Sandbox dispatch based on HORIZON_SANDBOX (local|docker)."""
    mode = os.environ.get("HORIZON_SANDBOX", "local")
    if mode == "docker":
        img = docker_image or os.environ.get("HORIZON_DOCKER_IMAGE", "python:3.12-slim")
        return run_in_docker(solution, tests, img, timeout)
    return run_local(solution, tests, timeout)


# --------- internal selection: best-of-N with retry (no hidden tests) ---------

def select_best(
    candidates: list[str], public_tests: list[str], timeout: int = DEFAULT_TIMEOUT,
    docker_image: str | None = None,
) -> tuple[int, ExecResult]:
    """Returns (index of the best candidate, its ExecResult) on the public tests.

    If there are no public tests, returns the first candidate that at least compiles.
    """
    if not public_tests:
        for i, c in enumerate(candidates):
            r = execute(c, ["pass"], timeout, docker_image)
            if r.passed:
                return i, r
        return 0, ExecResult(False, "", "no public tests, nothing compiled")
    best_i, best_r, best_score = 0, None, -1
    for i, c in enumerate(candidates):
        # raw count: the whole block passes/fails (public tests together)
        r = execute(c, public_tests, timeout, docker_image)
        score = 1 if r.passed else 0
        if score > best_score:
            best_i, best_r, best_score = i, r, score
        if r.passed:
            break
    return best_i, best_r  # type: ignore


def retry_feedback(prev_error: str) -> str:
    """Error-feedback message for the retry (§7.3)."""
    err = prev_error.strip()[-1500:]
    return (
        "The previous solution failed the tests with this error:\n"
        f"```\n{err}\n```\n"
        "Fix the code. Return ONLY the complete, corrected ```python``` block."
    )


# --------- benchmark scoring: hidden tests ---------

def score_solution(solution: str | None, hidden_tests: list[str],
                   timeout: int = DEFAULT_TIMEOUT, docker_image: str | None = None) -> bool:
    """Post-selection pass@1 on the benchmark's HIDDEN tests (§7.3)."""
    if not solution:
        return False
    return execute(solution, hidden_tests, timeout, docker_image).passed


# --------- I/O tests (LiveCodeBench): stdin/stdout and call-based ---------

_CALL_HARNESS = textwrap.dedent(
    """
    import json, sys
    __sol_ns = {{}}
    exec(compile(open("solution.py").read(), "solution.py", "exec"), __sol_ns)
    __args = [json.loads(l) for l in open("input.txt").read().strip().split("\\n") if l.strip()]
    __fn_name = {fn_name!r}
    if "Solution" in __sol_ns:
        __fn = getattr(__sol_ns["Solution"](), __fn_name)
    else:
        __fn = __sol_ns[__fn_name]
    __out = __fn(*__args)
    __exp = json.loads(open("expected.txt").read())
    def _norm(x):
        if isinstance(x, tuple): return list(x)
        return x
    assert json.loads(json.dumps(_norm(__out))) == __exp, f"got {{__out!r}}"
    """
)


def run_io_test_detail(solution: str, test: dict, mode: str, fn_name: str | None,
                       timeout: int = DEFAULT_TIMEOUT) -> tuple[bool, str]:
    """Runs ONE LCB I/O test in a sandbox. Returns (ok, outcome detail): the
    detail is the actual output or the error — used for the concrete feedback of
    the repair loop (8d). mode: 'stdin' | 'call'."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "solution.py").write_text(_RLIMIT_PREAMBLE + "\n" + solution)
        try:
            if mode == "stdin":
                proc = subprocess.run(
                    [sys.executable, "-I", "solution.py"],
                    input=test.get("input", ""), capture_output=True, text=True,
                    timeout=timeout, cwd=td, env={"PATH": os.environ.get("PATH", "")},
                )
                if proc.returncode != 0:
                    return False, (proc.stderr or "runtime error").strip()[-800:]
                got = proc.stdout.strip()
                return got == str(test.get("output", "")).strip(), got[:800]
            # call-based
            (tdp / "input.txt").write_text(test.get("input", ""))
            expected = test.get("output", "")
            try:  # the expected output is json; if it isn't, json-ify it
                json.loads(expected)
                (tdp / "expected.txt").write_text(expected)
            except Exception:
                (tdp / "expected.txt").write_text(json.dumps(expected))
            (tdp / "harness.py").write_text(_CALL_HARNESS.format(fn_name=fn_name or "solve"))
            proc = subprocess.run(
                [sys.executable, "-I", "harness.py"],
                capture_output=True, text=True, timeout=timeout, cwd=td,
                env={"PATH": os.environ.get("PATH", "")},
            )
            if proc.returncode == 0:
                return True, ""
            return False, (proc.stderr or proc.stdout or "assertion failed").strip()[-800:]
        except subprocess.TimeoutExpired:
            return False, f"TimeoutExpired ({timeout}s limit)"


def run_io_test(solution: str, test: dict, mode: str, fn_name: str | None,
                timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Runs ONE LCB I/O test in a local sandbox. mode: 'stdin' | 'call'."""
    return run_io_test_detail(solution, test, mode, fn_name, timeout)[0]


def run_io_tests(solution: str | None, tests: list[dict], mode: str,
                 fn_name: str | None, timeout: int = DEFAULT_TIMEOUT,
                 require_all: bool = True) -> tuple[int, int]:
    """Runs a list of I/O tests. Returns (passed, total)."""
    if not solution or not tests:
        return (0, len(tests or []))
    passed = 0
    for t in tests:
        if run_io_test(solution, t, mode, fn_name, timeout):
            passed += 1
        elif require_all:
            # early-exit if 100% is required (scoring): one failure is enough
            return (passed, len(tests))
    return (passed, len(tests))
