"""Benchmark runner (§8, §10) — runs a baseline on one or more benchmarks and
produces quality + hardware/token metrics (§8).

Per-domain scoring (§7.3 — the ground truth enters ONLY here):
  - math:    Math-Verify on pass@1 and maj@k
  - code:    execution of HIDDEN tests in a sandbox (pass@1 post-selection)
  - science: exact-match / multiple-choice

Usage:
  python -m eval.run_benchmarks --baseline 4_full_system --benchmarks math500 humaneval --limit 100
  python -m eval.run_benchmarks --baseline 1_base_naked --benchmarks gsm8k --limit 50

LiveCodeBench (§8): has its own harness; here it is flagged and skipped (integrate
its runner separately and report the number in the master table).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from common import benchmarks as bench
from common.benchmarks import Problem
from common.config import RUNS_DIR
from common.text import extract_final_answer, strip_think
from eval.baselines import API_BASELINES, BASELINES
from harness import verify_code, verify_math
from harness.pipeline import Pipeline, Solution


# ---------------- per-domain scoring ----------------

def score_math(problem: Problem, sol: Solution) -> dict:
    gold = problem.answer or ""
    pass1 = verify_math.verify_answer(extract_final_answer(sol.samples[0]), gold) if sol.samples else False
    maj = verify_math.verify_answer(sol.internal_selection, gold)
    return {"pass@1": bool(pass1), "final": bool(maj if sol.n_samples > 1 else pass1)}


def score_code(problem: Problem, sol: Solution, docker_image: str | None) -> dict:
    from common.text import extract_code

    code = sol.presentation.answer  # internally selected candidate
    first = extract_code(sol.samples[0]) if sol.samples else None

    if problem.meta.get("lcb"):  # LiveCodeBench: private I/O tests (§8)
        mode = problem.meta["exec_mode"]
        fn = problem.meta.get("fn_name")
        priv = problem.meta.get("io_private", [])
        p_sel, tot = verify_code.run_io_tests(code, priv, mode, fn)
        p_first, _ = verify_code.run_io_tests(first, priv, mode, fn)
        return {"pass@1": bool(tot and p_first == tot), "final": bool(tot and p_sel == tot)}

    passed = verify_code.score_solution(code, problem.tests, docker_image=docker_image)
    pass1 = verify_code.score_solution(first, problem.tests, docker_image=docker_image)
    return {"pass@1": bool(pass1), "final": bool(passed)}


# Robust MC extraction: patterns in order of reliability (§8 science).
_MC_PATTERNS = [
    re.compile(r"\\boxed\{\(?([A-D])\)?\}"),                     # \boxed{C} / \boxed{(C)}
    re.compile(r"(?:final answer|answer)\s*(?:is|:)?\s*\(?\*{0,2}([A-D])\*{0,2}\)?\b",
               re.IGNORECASE),                                    # Answer: C / answer is (C)
    re.compile(r"\*\*\(?([A-D])\)?[.:)]?\*\*"),                  # **C** / **(C)**
    re.compile(r"^\s*\(?([A-D])\)?[.:)]?\s*$", re.MULTILINE),    # line containing only the letter
]
_LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_mc_letter(text: str) -> str | None:
    for pat in _MC_PATTERNS:
        m = list(pat.finditer(text))
        if m:
            return m[-1].group(1)
    m = list(_LETTER_RE.finditer(text[-200:]))  # fallback: last letter near the end
    return m[-1].group(1) if m else None


def score_science(problem: Problem, sol: Solution) -> dict:
    gold = (problem.answer or "").strip()
    letters = ["A", "B", "C", "D"]

    def is_correct(sample_text: str) -> bool:
        text = strip_think(sample_text) or sample_text
        pred = extract_mc_letter(text)
        if pred and problem.choices and gold:
            if gold in letters:
                return pred == gold
            if gold in problem.choices:
                return pred == letters[problem.choices.index(gold)]
        # fallback: the text of the correct choice appears near the end
        return bool(gold) and gold.lower() in text[-300:].lower()

    pass1 = is_correct(sol.samples[0]) if sol.samples else False
    # with multiple samples: majority vote on the extracted letters (self-consistency MC)
    if sol.n_samples > 1:
        from collections import Counter

        preds = [extract_mc_letter(strip_think(s) or s) for s in sol.samples]
        dist = Counter(p for p in preds if p)
        if dist:
            maj = dist.most_common(1)[0][0]
            gold_letter = gold if gold in letters else (
                letters[problem.choices.index(gold)]
                if problem.choices and gold in problem.choices else None)
            final = (maj == gold_letter) if gold_letter else pass1
        else:
            final = pass1
    else:
        final = pass1
    return {"pass@1": bool(pass1), "final": bool(final)}


def score(problem: Problem, sol: Solution, docker_image: str | None) -> dict:
    if problem.domain == "math":
        return score_math(problem, sol)
    if problem.domain == "code":
        return score_code(problem, sol, docker_image)
    return score_science(problem, sol)


# ---------------- hardware/token (§8) ----------------

def peak_memory_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB->MB (linux)
    except Exception:
        return 0.0


def gpu_memory_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
    except Exception:
        pass
    return 0.0


# ---------------- loop ----------------

def run(baseline: str, bench_name: str, limit: int | None, docker_image: str | None,
        workers: int = 1, best_of_n: int | None = None) -> dict:
    client, cfg = BASELINES[baseline]()
    if best_of_n:
        cfg.best_of_n = best_of_n
    pipe = Pipeline(client, cfg)
    problems = bench.load(bench_name, limit=limit)
    print(f"[eval] {baseline} on {bench_name}: {len(problems)} problems "
          f"(workers={workers}, N={best_of_n or 'default'})")

    # vLLM does continuous batching: problems in parallel = wall-time ~/workers.
    # The result is identical to the sequential run (same samples per problem); only
    # the per-problem latency reflects GPU contention (documented in §8 notes).
    def solve_one(p):
        try:
            sol = pipe.solve(p)
            return (p, sol, None)
        except Exception as e:  # noqa: BLE001
            return (p, None, str(e)[:200])

    per_problem, agg = [], {"pass@1": 0, "final": 0}
    tokens_sum, lat_sum, n_sum = 0, 0.0, 0
    errors = done = 0
    t0 = time.monotonic()
    if workers <= 1:
        stream = map(solve_one, problems)
    else:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=workers)
        stream = pool.map(solve_one, problems)  # preserves order
    for p, sol, err in stream:
        done += 1
        if err is not None:
            errors += 1
            per_problem.append({"id": p.id, "error": err})
            print(f"[eval]   ERROR on {p.id}: {err[:120]}")
            if errors >= 3 and errors == done:
                print("[eval]   abort: consecutive systemic errors")
                break
            continue
        s = score(p, sol, docker_image)
        for k in agg:
            agg[k] += int(s[k])
        tokens_sum += sol.total_tokens
        lat_sum += sol.latency_s
        n_sum += sol.n_samples
        per_problem.append({"id": p.id, **s, "tokens": sol.total_tokens,
                            "latency_s": round(sol.latency_s, 3), "n": sol.n_samples})
        if done % 10 == 0:
            print(f"[eval]   {done}/{len(problems)} pass@1={agg['pass@1']} final={agg['final']}",
                  flush=True)
    if workers > 1:
        pool.shutdown(wait=False, cancel_futures=True)

    N = max(len(problems) - errors, 1)  # score only on the problems that succeeded
    result = {
        "baseline": baseline,
        "benchmark": bench_name,
        "n_problems": len(problems),
        "n_errors": errors,
        "quality": {
            "pass@1": round(100 * agg["pass@1"] / N, 2),
            "final": round(100 * agg["final"] / N, 2),  # maj@k (math) / post-sel (code)
        },
        "system": {
            "avg_tokens_per_problem": round(tokens_sum / N, 1),
            "avg_samples_per_problem": round(n_sum / N, 2),
            "avg_latency_s": round(lat_sum / N, 3),
            "wall_time_s": round(time.monotonic() - t0, 1),
            "peak_ram_mb": round(peak_memory_mb(), 1),
            "peak_vram_mb": round(gpu_memory_mb(), 1),
        },
        "per_problem": per_problem,
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, choices=list(BASELINES))
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--docker-image", default=None, help="sandbox image for code execution")
    ap.add_argument("--workers", type=int, default=1,
                    help="problems in parallel (vLLM: 8-12; external APIs: 2-4 due to rate limits)")
    ap.add_argument("--best-of-n", type=int, default=None,
                    help="override the N of best-of-N (e.g. Tier A CPU: 8)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.baseline in API_BASELINES:
        print(f"[eval] API baseline: this is the ceiling — see also eval.deepseek_ceiling (§9).")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for b in args.benchmarks:
        res = run(args.baseline, b, args.limit, args.docker_image, workers=args.workers,
                  best_of_n=args.best_of_n)
        all_results.append(res)
        out = args.out or (RUNS_DIR / f"{args.baseline}__{b}.json")
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        q = res.get("quality")
        print(f"[eval] -> {out}" + (f"  quality={q}" if q else ""))

    print("\n[eval] SUMMARY")
    for r in all_results:
        if "quality" in r:
            print(f"  {r['baseline']:20s} {r['benchmark']:14s} "
                  f"pass@1={r['quality']['pass@1']:5.1f}  final={r['quality']['final']:5.1f}")


if __name__ == "__main__":
    main()
