"""Phase 8b, step 1: generate and SAVE N samples per problem (design:
"generate once, select in many ways").

Output jsonl: {id, prompt, gold, samples: [texts]} — on these we compare
offline self-consistency vs PRM-weighted vs PRM-argmax (harness/prm_rank.py),
isolating the contribution of the selection strategy alone.

Usage (on the pod, with vLLM running):
  python -m eval.gen_samples --benchmark math500 --limit 100 --n 16 \
      --model horizon-math-lora --out results/runs/samples_math500_n16.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import benchmarks as bench
from common.config import base_config
from common.model_client import ModelClient
from harness.pipeline import SYSTEM_PROMPT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--model", default=None, help="adapter/base; default = base tier_a")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bc = base_config()
    model = args.model or bc["base_model"]["tier_a"]
    g = bc["generation"]
    client = ModelClient.vllm(model)
    problems = bench.load(args.benchmark, limit=args.limit)
    print(f"[gen] {args.benchmark}@{len(problems)} n={args.n} model={model}")

    def gen_one(p):
        try:
            rs = client.generate(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": p.prompt}],
                temperature=g["temperature"], top_p=g["top_p"],
                max_tokens=g["max_tokens"], n=args.n,
            )
            return {"id": p.id, "prompt": p.prompt, "gold": p.answer,
                    "samples": [r.text for r in rs]}
        except Exception as e:  # noqa: BLE001
            return {"id": p.id, "error": str(e)[:200]}

    from concurrent.futures import ThreadPoolExecutor

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.out.open("w") as w:
        for row in pool.map(gen_one, problems):
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            done += 1
            err += 1 if "error" in row else 0
            if done % 10 == 0:
                print(f"[gen] {done}/{len(problems)} (err={err})", flush=True)
    print(f"[gen] DONE {done} problems, {err} errors -> {args.out}")


if __name__ == "__main__":
    main()
