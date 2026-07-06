"""Phase 8b, step 3: OFFLINE comparison of the selection strategies on the
SAME samples (from prm_score.py). Isolates the contribution of the selector alone.

Strategies compared (same N):
  - sc            pure self-consistency (majority vote) — current baseline
  - prm_argmax    take the sample with the highest PRM score
  - prm_weighted  majority vote weighted by the PRM scores (per extracted answer)
Per-sample score aggregations: min / prod / last / mean of the steps.

Final scoring vs gold with Math-Verify (the ground truth enters ONLY here).

Usage (local):
  python -m harness.prm_rank --in results/runs/prmscores_math500_n16.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common.text import extract_final_answer
from harness.prm_select import agg, select  # SAME logic as the runtime (zero divergence)
from harness.verify_math import _canon, verify_answer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None, help="use only the first n samples (N curves)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.inp.open() if l.strip()]
    print(f"[rank] {len(rows)} problems from {args.inp.name}")

    strategies = ["sc"] + [f"prm_argmax_{a}" for a in ("min", "prod", "last", "mean")] \
                        + [f"prm_weighted_{a}" for a in ("min", "prod", "last", "mean")]
    hits = {s: 0 for s in strategies}

    for row in rows:
        gold = row.get("gold") or ""
        samples = row["samples"][: args.n] if args.n else row["samples"]
        prm = (row.get("prm") or [[]] * len(samples))[: len(samples)]
        answers = [_canon(extract_final_answer(s)) for s in samples]

        # self-consistency
        votes = defaultdict(int)
        for a in answers:
            if a:
                votes[a] += 1
        sc_ans = max(votes, key=votes.get) if votes else None
        hits["sc"] += int(verify_answer(sc_ans, gold))

        for how in ("min", "prod", "last", "mean"):
            for mode in ("argmax", "weighted"):
                strat = f"prm_{mode}_{how}"
                _, ans = select(answers, prm, strat)  # same function as the runtime
                hits[strat] += int(verify_answer(ans, gold))

    n = len(rows)
    print(f"\n{'strategy':22s} {'acc':>6s}")
    for s in strategies:
        print(f"{s:22s} {100*hits[s]/n:6.1f}")


if __name__ == "__main__":
    main()
