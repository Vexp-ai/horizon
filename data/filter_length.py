"""Length analysis + trace filter (§6 seq length).

Tokenizes each example with the base chat template and:
  - reports the real length distribution (median/p90/p99/max, total tokens);
  - writes a filtered file keeping ONLY traces that fit WHOLE within --max-tokens
    (avoids the truncation that cuts off the final answer — see code analysis).

Why filter instead of truncating: reasoning traces put the solution AT THE
END; truncating them mid-way teaches the model not to conclude.

Usage:
  python -m data.filter_length --specialist code --max-tokens 8192
  python -m data.filter_length --specialist math --analyze-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.chat import build_training_text
from common.config import ROOT, base_config, env

PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specialist", required=True, choices=["math", "code", "science"])
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="keep only traces with <= this many tokens (whole)")
    ap.add_argument("--analyze-only", action="store_true", help="do not write the filtered file")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    base = base_config()["base_model"]["tier_a"]
    tok = AutoTokenizer.from_pretrained(base, token=env("HF_TOKEN"))

    src = PROCESSED / f"{args.specialist}.clean.jsonl"
    out = PROCESSED / f"{args.specialist}.filtered.jsonl"
    lens: list[int] = []
    kept = dropped = 0
    total_tok = capped_tok = 0

    writer = None if args.analyze_only else out.open("w")
    with src.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            text = build_training_text(tok, r["problem"], r["solution"])
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            lens.append(n)
            total_tok += n
            capped_tok += min(n, args.max_tokens)
            if n <= args.max_tokens:
                kept += 1
                if writer:
                    writer.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                dropped += 1
            if (i + 1) % 5000 == 0:
                print(f"[filter] {args.specialist}: {i+1} rows processed...", flush=True)
    if writer:
        writer.close()

    lens.sort()
    n = len(lens)
    p = lambda q: lens[min(int(n * q), n - 1)]
    stats = {
        "specialist": args.specialist, "rows": n,
        "mean": round(sum(lens) / n, 1), "median": p(0.5),
        "p90": p(0.90), "p95": p(0.95), "p99": p(0.99), "max": lens[-1],
        "over_4096": sum(1 for x in lens if x > 4096),
        "over_8192": sum(1 for x in lens if x > 8192),
        "over_16384": sum(1 for x in lens if x > 16384),
        "total_tokens": total_tok,
        "max_tokens_cap": args.max_tokens,
        "kept_intact": kept, "dropped": dropped,
        "tokens_after_filter": sum(x for x in lens if x <= args.max_tokens),
    }
    print(f"\n[filter] {args.specialist}: rows={n} mean={stats['mean']} median={stats['median']} "
          f"p90={stats['p90']} p99={stats['p99']} max={stats['max']}")
    print(f"   >4096={stats['over_4096']}  >8192={stats['over_8192']}  >16384={stats['over_16384']}")
    print(f"   total_tokens={total_tok/1e6:.0f}M | cap@{args.max_tokens}: kept={kept} dropped={dropped} "
          f"tokens_after_filter={stats['tokens_after_filter']/1e6:.0f}M")
    if not args.analyze_only:
        print(f"   written -> {out} ({kept} whole rows)")
    (PROCESSED / f"{args.specialist}.lenstats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
