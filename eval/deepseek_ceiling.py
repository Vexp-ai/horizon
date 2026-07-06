"""DeepSeek V4 Flash ceiling via OpenRouter (§9).

Runs THE SAME benchmark prompts through `deepseek/deepseek-v4-flash`
(reasoning effort high), single-shot AND with best-of-N self-consistency (§9),
so both gate questions are answered (4 vs 5 and 4 vs 6, §10).

Reuses run_benchmarks for identical scoring (Math-Verify / hidden tests).

Usage:
  python -m eval.deepseek_ceiling --benchmarks math500 humaneval --limit 100
  python -m eval.deepseek_ceiling --benchmarks math500 --limit 50 --mode single
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import RUNS_DIR
from eval.run_benchmarks import run

# OpenRouter prices verified 2026-07-01 ($/M tokens). CHECK: they may change.
PRICING = {
    "v4flash": (0.09, 0.18),   # deepseek/deepseek-v4-flash
    "r1": (0.70, 2.50),        # deepseek/deepseek-r1 (the father — expensive output!)
}


def estimate_cost(result: dict, ceiling: str = "v4flash") -> float:
    """Rough $ cost estimate from the average token count (§9: order of a few $ up to $20)."""
    p_in, p_out = PRICING.get(ceiling, PRICING["v4flash"])
    sysm = result.get("system", {})
    n = result.get("n_problems", 0)
    avg_tokens = sysm.get("avg_tokens_per_problem", 0)
    total = avg_tokens * n
    # without a precise in/out split, assume 30/70 (reasoning is output-heavy)
    cost = (total * 0.3 / 1e6) * p_in + (total * 0.7 / 1e6) * p_out
    return round(cost, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", choices=["both", "single", "boN"], default="both")
    ap.add_argument("--ceiling", choices=["v4flash", "r1", "both"], default="both",
                    help="which ceiling: V4 Flash (§0.2 gate), R1 (the father), or both")
    ap.add_argument("--docker-image", default=None)
    ap.add_argument("--workers", type=int, default=4,
                    help="problems in parallel (moderate: API rate limits)")
    args = ap.parse_args()

    per_ceiling = {
        "v4flash": {"single": "5_deepseek_single", "boN": "6_deepseek_boN"},
        "r1": {"single": "5r1_single", "boN": "6r1_boN"},
    }
    ceilings = ["v4flash", "r1"] if args.ceiling == "both" else [args.ceiling]
    mode_keys = {"both": ["single", "boN"], "single": ["single"], "boN": ["boN"]}[args.mode]
    modes = [per_ceiling[c][m] for c in ceilings for m in mode_keys]

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    total_cost = 0.0
    for baseline in modes:
        ceiling = "r1" if baseline.startswith(("5r1", "6r1")) else "v4flash"
        for b in args.benchmarks:
            res = run(baseline, b, args.limit, args.docker_image, workers=args.workers)
            out = RUNS_DIR / f"{baseline}__{b}.json"
            out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
            if "quality" in res:
                cost = estimate_cost(res, ceiling)
                total_cost += cost
                print(f"[ceiling] {baseline} {b}: pass@1={res['quality']['pass@1']} "
                      f"final={res['quality']['final']}  ~${cost}")
    print(f"[ceiling] total estimated cost ~${round(total_cost, 2)} (CHECK prices §2)")


if __name__ == "__main__":
    main()
