"""Router v0 eval (§7.1, Phase 3): accuracy + confusion matrix.

Held-out labels: BENCHMARK prompts (the router has never seen them),
domain = the benchmark's domain. Realistic: these are the queries the
router will actually receive during eval.

Runs locally on CPU (embeddings). Output: results/runs/router_v0.json.

Usage:
  python -m eval.router_eval --per-domain 100
"""
from __future__ import annotations

import argparse
import json

from common import benchmarks as bench
from common.config import RUNS_DIR

# benchmark -> gold domain (to label the test queries)
LABELED_SOURCES = {
    "math": ["gsm8k", "math500"],
    "code": ["humaneval", "mbpp"],
    "science": ["gpqa", "mmlu_stem"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-domain", type=int, default=100)
    ap.add_argument("--version", choices=["v0", "v1"], default="v0")
    args = ap.parse_args()

    from router.router_v0 import RouterV0

    labeled: list[tuple[str, str]] = []
    per_source = max(1, args.per_domain // 2)
    for domain, sources in LABELED_SOURCES.items():
        got = 0
        for name in sources:
            try:
                problems = bench.load(name, limit=per_source)
            except Exception as e:
                print(f"[router-eval] skip {name}: {str(e)[:80]}")
                continue
            for p in problems:
                labeled.append((p.prompt, domain))
                got += 1
        print(f"[router-eval] {domain}: {got} labeled queries")

    if args.version == "v1":
        from router.router_v1 import RouterV1

        router = RouterV1()
        # RouterV1 has no evaluate(): reuse V0's (same route() interface)
        res = RouterV0.evaluate(router, labeled)  # type: ignore[arg-type]
    else:
        router = RouterV0()
        res = router.evaluate(labeled)
    print(f"\n[router-eval] accuracy = {res['accuracy']:.3f} on {res['n']} queries")
    print("[router-eval] confusion matrix (rows=gold, columns=pred):")
    doms = list(res["confusion_matrix"].keys())
    print("          " + "  ".join(f"{d:>8s}" for d in doms))
    for g in doms:
        row = res["confusion_matrix"][g]
        print(f"{g:>8s}  " + "  ".join(f"{row[p]:>8d}" for p in doms))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"router_{args.version}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[router-eval] -> {out}")


if __name__ == "__main__":
    main()
