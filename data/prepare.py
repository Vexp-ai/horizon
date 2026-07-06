"""Data prep for one specialist (§4, §Phase 2): download -> normalize ->
subsample -> (then) decontaminate.

Pipeline:
  1. Downloads the datasets listed in config/lora_<spec>.yaml (§4).
  2. Normalizes into {problem, solution, answer} rows according to `schema`.
  3. Subsamples to `subsample` (~30-50k, §4) with a fixed seed.
  4. Writes data/processed/<spec>.raw.jsonl.
  5. Calls data.decontaminate to produce <spec>.clean.jsonl.

Usage:
  python -m data.prepare --specialist math
  python -m data.prepare --specialist code --skip-decontam   # decontaminate separately
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from common.config import ROOT, specialist_config

PROCESSED = ROOT / "data" / "processed"

# Benchmarks to decontaminate against, per domain (§8, §15.1).
DECONTAM_BENCHMARKS = {
    "math": ["math500", "aime2024", "gsm8k"],
    "code": ["humaneval", "mbpp"],  # LiveCodeBench: own harness, decontaminated separately
    "science": ["gpqa", "mmlu_stem"],
}


_USER_ROLES = {"user", "human"}
_ASSISTANT_ROLES = {"assistant", "gpt"}


def extract_conversation(turns: list) -> tuple[str | None, str | None]:
    """From a list of turns (`messages` role/content format or ShareGPT
    `conversations` from/value) returns (first user turn, last assistant turn).
    """
    user_text = assistant_text = None
    for t in turns:
        if not isinstance(t, dict):
            continue
        role = str(t.get("role") or t.get("from") or "").lower()
        content = t.get("content")
        if content is None:
            content = t.get("value")
        if content is None:
            continue
        if role in _USER_ROLES and user_text is None:
            user_text = str(content)
        elif role in _ASSISTANT_ROLES:
            assistant_text = str(content)  # keep the last one
    return user_text, assistant_text


def normalize_row(row: dict, schema: dict) -> dict | None:
    """Extracts {problem, solution, answer} from a heterogeneous row.

    Handles both datasets with explicit fields (problem/solution/answer, e.g.
    OpenR1-Math-220k) and conversation formats (messages / conversations,
    e.g. Mixture-of-Thoughts, OpenThoughts-114k).
    """
    pf, sf, af = schema.get("problem_field"), schema.get("solution_field"), schema.get("answer_field")
    problem = (row.get(pf) if pf else None) or row.get("problem") or row.get("question") or row.get("prompt")
    solution = (row.get(sf) if sf else None) or row.get("solution") or row.get("response")

    # Fallback to conversation formats.
    if not problem or not solution:
        turns = row.get("messages") or row.get("conversations")
        if isinstance(turns, list):
            u, a = extract_conversation(turns)
            problem = problem or u
            solution = solution or a

    if not problem or not solution:
        return None
    out = {"problem": str(problem), "solution": str(solution)}
    if af:
        out["answer"] = str(row.get(af, ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specialist", required=True, choices=["math", "code", "science"])
    ap.add_argument("--skip-decontam", action="store_true")
    ap.add_argument("--limit-per-dataset", type=int, default=None,
                    help="download cap per dataset (debug/fast)")
    args = ap.parse_args()

    from datasets import load_dataset  # lazy import

    cfg = specialist_config(args.specialist)
    schema = cfg.get("schema", {})
    rng = random.Random(cfg.get("seed", 42))
    PROCESSED.mkdir(parents=True, exist_ok=True)

    target = cfg.get("subsample")
    datasets_cfg = cfg["datasets"]
    # Per-dataset quota with a 30% buffer to compensate for rows dropped during
    # normalization. Subsampling AT THE DATASET LEVEL (arrow memory-mapped) BEFORE
    # materializing in RAM avoids OOM on large datasets (code/science, §RAM 15GB).
    per_ds = int((target / len(datasets_cfg)) * 1.3) + 1 if target else None

    # STREAMING write: normalizes and writes row by row, keeping nothing in
    # RAM (code traces are huge -> OOM if accumulated in a list).
    # The order is already randomized by the dataset-level shuffle; SFTTrainer
    # reshuffles during training anyway, so no global shuffle is needed.
    raw = PROCESSED / f"{args.specialist}.raw.jsonl"
    kept = 0
    with raw.open("w") as w:
        for spec in datasets_cfg:
            if target and kept >= target:
                break
            hf_id, config, split = spec["id"], spec.get("config"), spec.get("split", "train")
            print(f"[prepare] load {hf_id} config={config} split={split}", flush=True)
            ds = load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)
            if args.limit_per_dataset:
                ds = ds.select(range(min(args.limit_per_dataset, len(ds))))
            if per_ds and len(ds) > per_ds:
                ds = ds.shuffle(seed=cfg.get("seed", 42)).select(range(per_ds))
            n_ok = 0
            for row in ds:
                if target and kept >= target:
                    break
                norm = normalize_row(row, schema)
                if norm:
                    w.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    kept += 1
                    n_ok += 1
            print(f"[prepare]   wrote {n_ok} rows from {hf_id} (tot {kept})", flush=True)
    print(f"[prepare] raw total: {kept} -> {raw}")

    if args.skip_decontam:
        print("[prepare] --skip-decontam: remember to run data.decontaminate!")
        return

    clean = PROCESSED / f"{args.specialist}.clean.jsonl"
    benches = DECONTAM_BENCHMARKS[args.specialist]
    cmd = [
        sys.executable, "-m", "data.decontaminate",
        "--in", str(raw), "--out", str(clean),
        "--benchmarks", *benches, "--prompt-field", "problem",
    ]
    print(f"[prepare] decontamination: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"[prepare] ready -> {clean}")


if __name__ == "__main__":
    main()
