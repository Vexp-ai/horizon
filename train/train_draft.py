"""Speed sprint, step 2: train a tiny standalone DRAFT model for speculative
decoding (llama.cpp `-md` compatible — no target-internals access needed).

Base: Qwen2.5-0.5B (same tokenizer family as the 7B target — vocabulary
compatibility is required by llama.cpp speculative decoding). Trained as a
plain causal LM on the SAME R1-trace distribution the target was distilled
from, formatted with the TARGET's chat template (common/chat.py), so the
draft predicts the target's continuations, not its own instruct style.

Usage (pod, single 24GB GPU, ~2-3h):
  python -m train.train_draft --data data/processed/draft_corpus.jsonl \
      --out models/horizon-draft-0.5b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DRAFT_BASE = "Qwen/Qwen2.5-0.5B"
TARGET_BASE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # template source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seq", type=int, default=4096,
                    help="drafting needs local coherence, not long context")
    ap.add_argument("--lr", type=float, default=2e-5)
    # batch 1: with the 152k vocab the fp32 logits of a 4x4096 batch alone
    # need ~10 GB — OOM on 24 GB (measured). Same effective batch via accum.
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=32)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from common.chat import build_training_text

    # the TARGET's tokenizer defines the template; the draft shares the vocab
    target_tok = AutoTokenizer.from_pretrained(TARGET_BASE)

    rows = []
    with args.data.open() as f:
        for line in f:
            if args.max_rows and len(rows) >= args.max_rows:
                break
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append({"text": build_training_text(
                target_tok, d["problem"], d["solution"])})
    print(f"[draft] {len(rows)} training rows")
    ds = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_BASE, torch_dtype=torch.bfloat16)
    # llama.cpp `-md` requires draft/target vocab compatibility: train (and
    # ship) the draft with the TARGET's tokenizer and pad the embedding to the
    # target's n_vocab (152064 vs Qwen2.5-0.5B's 151936; tied lm_head follows).
    tok = target_tok
    model.resize_token_embeddings(152064)

    cfg = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=1,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        save_steps=1000,
        save_total_limit=1,   # full-FT checkpoints are ~6 GB each (disk quota)
        max_length=args.seq,
        packing=True,
        dataset_num_proc=8,       # 256-core deadlock gotcha (measured)
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"[draft] DONE -> {args.out}")


if __name__ == "__main__":
    main()
