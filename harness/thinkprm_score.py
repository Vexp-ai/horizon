"""Phase 8b (iteration 2): scoring with ThinkPRM-1.5B — GENERATIVE verifier.

ThinkPRM (based on R1-Distill-Qwen-1.5B, hence in-distribution with our
traces) generates a critique of the steps and \\boxed{correct}/
\\boxed{incorrect} verdicts. Here we map the verdicts to per-step scores {1.0, 0.0} in
the SAME jsonl format as prm_score.py, so harness/prm_rank.py compares the
strategies WITHOUT modifications.

Usage (pod, free GPU — the main vLLM must be shut down first):
  .venv-eval/bin/python -m vllm.entrypoints.openai.api_server \
      --model launch/ThinkPRM-1.5B --port 8001 --max-model-len 16384 \
      --gpu-memory-utilization 0.40 &
  python -m harness.thinkprm_score --in results/runs/samples_math500_n16.jsonl \
      --out results/runs/thinkprm_math500_n16.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from harness.prm_score import split_steps

# Format from the model card (VERIFIED 2026-07-03): [Math Problem] /
# [Solution] sections + trigger "Let's verify step by step:"; pure completion, NO chat.
VERIFY_PROMPT = """You are given a math problem and a proposed step-by-step solution:

[Math Problem]

{problem}

[Solution]

{solution}

Review and critique each step in the proposed solution to determine whether each step is correct. If the solution is incomplete, only verify the provided steps.

Let's verify step by step:"""

_VERDICT_RE = re.compile(r"\\boxed\{\s*(correct|incorrect)\s*\}", re.IGNORECASE)


def parse_verdicts(text: str) -> list[float]:
    """Per-step verdicts -> score vector {1.0, 0.0} (order of appearance)."""
    return [1.0 if m.group(1).lower() == "correct" else 0.0
            for m in _VERDICT_RE.finditer(text)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="launch/ThinkPRM-1.5B")
    ap.add_argument("--max-verify-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1200)

    def score_one(question: str, sample: str) -> list[float]:
        steps = split_steps(sample)
        sol = "\n\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))[:24000]
        prompt = VERIFY_PROMPT.format(problem=question[:4000], solution=sol)
        resp = client.completions.create(
            model=args.model, prompt=prompt, temperature=0.0,
            max_tokens=args.max_verify_tokens)
        return parse_verdicts(resp.choices[0].text or "")

    rows = [json.loads(l) for l in args.inp.open() if l.strip()]
    rows = [r for r in rows if "error" not in r and r.get("samples")]
    print(f"[thinkprm] {len(rows)} problems from {args.inp.name}")

    # flatten (problem, sample) -> parallel on vLLM (continuous batching)
    tasks = [(ri, si) for ri, r in enumerate(rows) for si in range(len(r["samples"]))]
    scores: dict[tuple[int, int], list[float]] = {}
    n_err = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(score_one, rows[ri]["prompt"], rows[ri]["samples"][si]): (ri, si)
                for ri, si in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            try:
                scores[key] = fut.result()
            except Exception as e:  # noqa: BLE001
                scores[key] = []
                n_err += 1
                if n_err <= 3:
                    print(f"[thinkprm] error on {key}: {str(e)[:120]}")
            if k % 100 == 0:
                print(f"[thinkprm] {k}/{len(tasks)} verifications (err={n_err})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as w:
        for ri, r in enumerate(rows):
            r["prm"] = [scores.get((ri, si), []) for si in range(len(r["samples"]))]
            r.pop("prompt", None)
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[thinkprm] DONE {len(rows)} problems, {n_err} failed verifications -> {args.out}")


if __name__ == "__main__":
    main()
