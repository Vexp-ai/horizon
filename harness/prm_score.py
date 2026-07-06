"""Phase 8b, step 2: PRM scoring of the saved samples (GPU).

Takes the jsonl from eval/gen_samples.py and adds, for each sample, the PRM's
per-step scores. Output: jsonl with {id, gold, samples, prm: [[p_step,...], ...]}.

Default: Qwen/Qwen2.5-Math-PRM-7B (VERIFY format from the model card: steps
separated by <extra_0>, correctness probability read at the separator tokens).

AUTHORITY RULE (§8b plan v2): the PRM guides the SELECTION; where a hard
verifier exists, final authority stays with the hard verifier.

Usage (pod, free GPU — shut down vLLM first):
  python -m harness.prm_score --in results/runs/samples_math500_n16.jsonl \
      --out results/runs/prmscores_math500_n16.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def split_steps(sample_text: str) -> list[str]:
    """Splits the trace into steps (paragraphs of the <think> + final answer)."""
    import re

    m = re.search(r"<think>(.*?)</think>(.*)", sample_text, re.DOTALL)
    if m:
        think, tail = m.group(1), m.group(2)
    else:
        think, tail = sample_text, ""
    steps = [s.strip() for s in think.split("\n\n") if s.strip()]
    if tail.strip():
        steps.append(tail.strip())
    return steps[:64] or [sample_text[:2000]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-PRM-7B")
    ap.add_argument("--max-steps-tokens", type=int, default=4096)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # workaround transformers>=5: the PRM's custom config (Qwen2RMConfig) does not
    # expose pad_token_id -> we inject it from the tokenizer before loading.
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if not hasattr(cfg, "pad_token_id") or cfg.pad_token_id is None:
        cfg.pad_token_id = tok.pad_token_id or tok.eos_token_id
    model = AutoModel.from_pretrained(
        args.model, config=cfg, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    ).eval()
    sep = "<extra_0>"
    sep_id = tok.encode(sep)[0]

    @torch.no_grad()
    def score_one(question: str, steps: list[str]) -> list[float]:
        resp = sep.join(steps) + sep
        msgs = [
            {"role": "system", "content": "Please reason step by step."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": resp},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False)
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=args.max_steps_tokens).input_ids.to(model.device)
        out = model(input_ids=ids)
        logits = out[0]  # (1, seq, 2) — VERIFY: num_labels=2 from the model card
        probs = torch.softmax(logits, dim=-1)[0, :, 1]
        mask = (ids[0] == sep_id)
        return probs[mask].float().tolist()

    n_rows = n_err = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.inp.open() as f, args.out.open("w") as w:
        for line in f:
            row = json.loads(line)
            if "error" in row or not row.get("samples"):
                continue
            scores = []
            for s in row["samples"]:
                try:
                    scores.append(score_one(row["prompt"], split_steps(s)))
                except Exception as e:  # noqa: BLE001
                    scores.append([])
                    n_err += 1
            row["prm"] = scores
            row.pop("prompt", None)  # reduces file size
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1
            if n_rows % 10 == 0:
                print(f"[prm] {n_rows} problems scored (err={n_err})", flush=True)
    print(f"[prm] DONE {n_rows} problems, {n_err} failed samples -> {args.out}")


if __name__ == "__main__":
    main()
