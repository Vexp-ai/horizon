"""SFT of a LoRA adapter on a shared base (§5, §6).

Two backends, automatic selection:
  - Unsloth (preferred, §2/§6): cheap/single-GPU, long-context. If importable.
  - TRL SFTTrainer + PEFT: portable fallback.

Input: data/processed/<spec>.clean.jsonl (from data.prepare, DECONTAMINATED).
Output: adapters/horizon-<spec>-lora/ (§6).

Usage:
  python -m train.train_lora --specialist math
  python -m train.train_lora --specialist code --tier b   # 1.5B floor base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import ADAPTER_DIR, ROOT, base_config, specialist_config

PROCESSED = ROOT / "data" / "processed"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run first: python -m data.prepare --specialist <spec>"
        )
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_dataset(rows: list[dict], tokenizer, max_seq_length: int):
    """Builds the training text PRESERVING the <think> reasoning (§6).

    NB: do NOT use apply_chat_template on the assistant turn — the R1 template
    strips <think>. See common/chat.build_training_text for the reason why.
    """
    from datasets import Dataset

    from common.chat import build_training_text

    texts = [build_training_text(tokenizer, r["problem"], r["solution"]) for r in rows]
    return Dataset.from_dict({"text": texts})


def train_unsloth(cfg, base_model, rows, out_dir, max_steps=-1):
    from unsloth import FastLanguageModel  # type: ignore
    from trl import SFTConfig, SFTTrainer

    t = cfg["training"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=t["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    lcfg = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        bias=lcfg["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg.get("seed", 42),
    )
    ds = build_dataset(rows, tokenizer, t["max_seq_length"])
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=_sft_args(t, out_dir, max_steps),
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


def train_trl(cfg, base_model, rows, out_dir, max_steps=-1):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    t, lcfg = cfg["training"], cfg["lora"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if t.get("bf16") else torch.float16,
        device_map="auto",
    )
    peft_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        bias=lcfg["bias"],
        task_type="CAUSAL_LM",
    )
    ds = build_dataset(rows, tokenizer, t["max_seq_length"])
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        peft_config=peft_config,
        args=_sft_args(t, out_dir, max_steps),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


def _sft_args(t: dict, out_dir: Path, max_steps: int = -1):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        num_train_epochs=t["num_train_epochs"],
        max_steps=max_steps,          # >0 only for calibration (§Phase 2)
        dataset_num_proc=int(t.get("dataset_num_proc", 8)),  # avoids map deadlock with high num_proc (256-core)
        max_length=t["max_seq_length"],
        packing=t.get("packing", True),
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        optim=t.get("optim", "adamw_8bit"),
        logging_steps=t.get("logging_steps", 10),
        save_steps=t.get("save_steps", 500),
        report_to="none",
        seed=42,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specialist", required=True, choices=["math", "code", "science"])
    ap.add_argument("--tier", choices=["a", "b"], default="a", help="a=7B, b=1.5B floor")
    ap.add_argument("--data", type=Path, default=None, help="override DECONTAMINATED jsonl")
    ap.add_argument("--backend", choices=["auto", "unsloth", "trl"], default="auto")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="CALIBRATION: limits to N steps and estimates the full-epoch time")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="use only the first N rows (the .filtered.jsonl are already shuffled): "
                         "doses the data volume to hit budget/time targets")
    args = ap.parse_args()

    cfg = specialist_config(args.specialist)
    bc = base_config()
    base_model = bc["base_model"]["tier_a" if args.tier == "a" else "tier_b"]

    # Prefer the length-filtered data (whole traces, no truncation) if present.
    if args.data:
        data_path = args.data
    else:
        filt = PROCESSED / f"{args.specialist}.filtered.jsonl"
        data_path = filt if filt.exists() else PROCESSED / f"{args.specialist}.clean.jsonl"
    rows = load_jsonl(data_path)
    if args.max_rows and len(rows) > args.max_rows:
        rows = rows[: args.max_rows]  # the filtered files are already shuffled (§4)
    print(f"[train] {args.specialist}: {len(rows)} examples from {data_path.name} | base={base_model}")

    out_dir = ADAPTER_DIR / cfg["output_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = args.backend
    if backend == "auto":
        try:
            import unsloth  # noqa: F401
            backend = "unsloth"
        except Exception:
            backend = "trl"
    print(f"[train] backend={backend} | max_steps={args.max_steps}")

    import time
    t0 = time.monotonic()
    if backend == "unsloth":
        train_unsloth(cfg, base_model, rows, out_dir, args.max_steps)
    else:
        train_trl(cfg, base_model, rows, out_dir, args.max_steps)
    elapsed = time.monotonic() - t0

    if args.max_steps and args.max_steps > 0:
        # Extrapolate the full-epoch time (§Phase 2 calibration).
        t = cfg["training"]
        eff_batch = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
        # with packing, total steps are not len(rows)/eff_batch; conservative token-based estimate
        per_step = elapsed / args.max_steps
        approx_steps_epoch = max(1, len(rows) // eff_batch)
        est_h = per_step * approx_steps_epoch / 3600
        print(f"\n[calibration] {args.max_steps} steps in {elapsed:.0f}s → {per_step:.1f}s/step")
        print(f"[calibration] ~{approx_steps_epoch} steps/epoch (eff batch={eff_batch}) "
              f"→ FULL EPOCH ~{est_h:.1f}h  (NB: with packing the real step count may differ; "
              f"read 'total optimization steps' in the trainer log for the exact number)")
        print("[calibration] not saving the adapter (partial run).")
        return

    print(f"[train] adapter saved -> {out_dir}")
    print("[train] next: sanity check on the domain dev set (§6, §Phase 2).")


if __name__ == "__main__":
    main()
