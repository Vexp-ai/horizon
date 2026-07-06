"""Unified benchmark loader (§8).

Normalizes each benchmark into a list of `Problem` with common fields, so
harness/eval do not know the specific formats. VERIFY the HF IDs (§8): the
configs/splits may change.

`load(name, limit)` returns List[Problem]. Requires `datasets` at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .config import env  # also loads the .env (dotenv) at import time


@dataclass
class Problem:
    id: str
    domain: str  # math | code | science
    prompt: str  # problem text (without system prompt)
    answer: str | None = None  # math/science ground truth (ONLY for scoring)
    tests: list[str] = field(default_factory=list)  # hidden code tests (scoring)
    public_tests: list[str] = field(default_factory=list)  # public tests (internal selection)
    choices: list[str] | None = None  # science MC
    meta: dict[str, Any] = field(default_factory=dict)


# --- per-benchmark adapters: (hf_id, config, split) + mapping function ---

def _gsm8k(row, i):
    ans = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
    return Problem(id=f"gsm8k-{i}", domain="math", prompt=row["question"], answer=ans)


def _math500(row, i):
    return Problem(
        id=f"math500-{i}", domain="math",
        prompt=row["problem"], answer=str(row.get("answer", row.get("solution", ""))),
        meta={"level": row.get("level"), "subject": row.get("subject")},
    )


def _aime(row, i):
    return Problem(
        id=f"aime-{i}", domain="math",
        prompt=row.get("problem") or row.get("question", ""),
        answer=str(row.get("answer", "")),
        meta={"url": str(row.get("url", ""))},  # for the year filter (aime2024)
    )


def _format_mc(question: str, choices: list[str]) -> str:
    """Multiple-choice prompt: the options MUST be in the prompt (bug fix:
    previously the model did not see the A-D alternatives)."""
    letters = ["A", "B", "C", "D"]
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:4]))
    return (f"{question}\n\n{opts}\n\n"
            "Choose the correct option and put ONLY its letter in \\boxed{}.")


def _humaneval(row, i):
    return Problem(
        id=row.get("task_id", f"humaneval-{i}"), domain="code",
        prompt=row["prompt"],
        tests=[row["test"]],
        meta={"entry_point": row.get("entry_point")},
    )


def _mbpp(row, i):
    desc = row["text"] if "text" in row else row.get("prompt", "")
    tests = list(row.get("test_list", []))
    imports = list(row.get("test_imports", []) or [])
    # Standard MBPP protocol: the first assert goes IN the prompt, otherwise
    # the model does not know the expected function name/signature
    # (bug: frontier at ~12%).
    example = tests[0] if tests else ""
    prompt = desc + (f"\n\nYour code should pass this test:\n{example}" if example else "")
    return Problem(
        id=f"mbpp-{row.get('task_id', i)}", domain="code",
        prompt=prompt,
        tests=imports + tests,           # test_imports are needed at runtime
        public_tests=imports + tests[:1],
    )


def _gpqa(row, i):
    import random

    correct = row.get("Correct Answer")
    incorrect = [row.get(f"Incorrect Answer {k}") for k in (1, 2, 3)]
    choices = [c for c in ([correct] + incorrect) if c]
    random.Random(i).shuffle(choices)  # anti-bias: never "the correct one is always A"
    return Problem(
        id=f"gpqa-{i}", domain="science",
        prompt=_format_mc(row.get("Question", ""), choices),
        answer=correct, choices=choices,
    )


def _mmlu(row, i):
    letters = ["A", "B", "C", "D"]
    choices = list(row["choices"])
    return Problem(
        id=f"mmlu-{i}", domain="science",
        prompt=_format_mc(row["question"], choices), choices=choices,
        answer=letters[row["answer"]] if isinstance(row["answer"], int) else str(row["answer"]),
    )


# name -> (hf_id, config, split, mapper). VERIFY (§8).
REGISTRY: dict[str, tuple[str, str | None, str, Callable]] = {
    "gsm8k":      ("openai/gsm8k", "main", "test", _gsm8k),
    "math500":    ("HuggingFaceH4/MATH-500", None, "test", _math500),
    "aime2024":   ("AI-MO/aimo-validation-aime", None, "train", _aime),
    "humaneval":  ("openai/openai_humaneval", None, "test", _humaneval),
    "mbpp":       ("google-research-datasets/mbpp", "sanitized", "test", _mbpp),
    "gpqa":       ("Idavidrein/gpqa", "gpqa_diamond", "train", _gpqa),
    "mmlu_stem":  ("cais/mmlu", "all", "test", _mmlu),
    # LiveCodeBench has its own harness (§8): handled separately in eval/run_benchmarks.py.
}

CORE = ["math500", "livecodebench"]  # verifiable core gate (§0.2)


def load(name: str, limit: int | None = None) -> list[Problem]:
    if name == "livecodebench":
        from common.lcb import load_lcb  # raw jsonl + I/O tests (§8 core)

        return load_lcb(limit=limit)
    if name not in REGISTRY:
        raise KeyError(
            f"Benchmark '{name}' not in the registry. Available: "
            f"{sorted(REGISTRY) + ['livecodebench']}."
        )
    from datasets import load_dataset  # lazy import

    hf_id, config, split, mapper = REGISTRY[name]
    tok = env("HF_TOKEN")  # needed for gated benchmarks (e.g. GPQA)
    ds = (load_dataset(hf_id, config, split=split, token=tok) if config
          else load_dataset(hf_id, split=split, token=tok))
    if name == "mmlu_stem":
        stem = {"physics", "chemistry", "biology", "computer_science", "mathematics", "engineering"}
        ds = ds.filter(lambda r: any(s in str(r.get("subject", "")).lower() for s in stem))
        ds = ds.shuffle(seed=0)  # the dataset is sorted by subject: sample mixed
    problems = [mapper(row, i) for i, row in enumerate(ds)]
    if name == "aime2024":  # the AI-MO dataset covers '22-'24: keep only 2024
        only24 = [p for p in problems if "2024" in p.meta.get("url", "")]
        problems = only24 or problems[-30:]  # fallback: the last 30 (chronological)
    if limit:
        problems = problems[:limit]
    return problems
