"""Decontamination (§4, §15.1) — MANDATORY before training.

Removes from the training data every problem that appears in (or is too
similar to) the evaluation benchmarks. Without this the numbers are worth
nothing (§4).

Strategy:
  1. Build the set of benchmark prompts (§8).
  2. For each training example, compute:
       - normalized exact hash (exact match), and
       - n-gram overlap (Jaccard on word shingles) above threshold -> contaminated.
  3. Log how many were removed, per reason.

Usage:
  python -m data.decontaminate --in data/processed/math.raw.jsonl \
      --out data/processed/math.clean.jsonl --benchmarks math500 aime2024 gsm8k
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

WORD_RE = re.compile(r"\w+")


def normalize(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def exact_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def ngrams(text: str, n: int = 8) -> set[str]:
    toks = normalize(text).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_benchmark_index(benchmarks: list[str], n: int) -> tuple[set[str], list[set[str]]]:
    """Returns (exact hashes, list of n-gram sets) of the benchmark prompts."""
    from common import benchmarks as bench

    exact: set[str] = set()
    grams: list[set[str]] = []
    for name in benchmarks:
        try:
            problems = bench.load(name)
        except Exception as e:  # LiveCodeBench / benchmarks with their own harness
            print(f"[decontaminate] skip '{name}': {e}")
            continue
        for p in problems:
            exact.add(exact_hash(p.prompt))
            grams.append(ngrams(p.prompt, n))
        print(f"[decontaminate] {name}: {len(problems)} prompts indexed")
    return exact, grams


def is_contaminated(
    prompt: str, exact: set[str], grams: list[set[str]], n: int, thr: float
) -> str | None:
    if exact_hash(prompt) in exact:
        return "exact"
    g = ngrams(prompt, n)
    for bg in grams:
        if jaccard(g, bg) >= thr:
            return f"ngram>={thr}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="raw training jsonl")
    ap.add_argument("--out", dest="out", required=True, help="clean jsonl")
    ap.add_argument("--benchmarks", nargs="+", required=True,
                    help="benchmark names (common/benchmarks.py) to decontaminate against")
    ap.add_argument("--prompt-field", default="problem")
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="n-gram Jaccard threshold to flag contamination")
    args = ap.parse_args()

    exact, grams = build_benchmark_index(args.benchmarks, args.ngram)

    inp, out = Path(args.inp), Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = removed = 0
    reasons: dict[str, int] = {}
    with inp.open() as f, out.open("w") as w:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get(args.prompt_field, "")
            reason = is_contaminated(prompt, exact, grams, args.ngram, args.threshold)
            if reason:
                removed += 1
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[decontaminate] kept={kept} removed={removed} reasons={reasons}")
    # Persistent log required by §4 ("Log how many examples were removed").
    log = out.with_suffix(".decontam.json")
    log.write_text(json.dumps(
        {"kept": kept, "removed": removed, "reasons": reasons,
         "benchmarks": args.benchmarks, "ngram": args.ngram, "threshold": args.threshold},
        indent=2,
    ))
    print(f"[decontaminate] log -> {log}")


if __name__ == "__main__":
    main()
