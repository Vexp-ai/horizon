"""Builds the training text PRESERVING the <think> reasoning (§6).

WHY THIS EXISTS: the DeepSeek-R1-Distill chat template **removes** the
`<think>...</think>` blocks from past assistant turns (at inference the
reasoning is regenerated, not fed back in). If we built the target with
`apply_chat_template([... , {'role':'assistant', ...}])`, we would train the
LoRA NOT to reason — the opposite of what we need.

Correct construction: user prompt with `add_generation_prompt=True` (which in
R1 already ends with `<｜Assistant｜><think>\n`) + the solution (dropping the
leading `<think>` to avoid duplicating it) + eos. This way the target includes
the whole trace.

Used by train/train_lora.py (training data) and data/filter_length.py (length
measurement) — they MUST use the same function for consistency.
"""
from __future__ import annotations


def build_training_text(tokenizer, problem: str, solution: str) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False, add_generation_prompt=True,
    )
    sol = solution.lstrip()
    if sol.startswith("<think>"):
        # the R1 generation prompt already opens "<think>\n": avoid a double tag
        sol = sol[len("<think>") :].lstrip("\n")
    eos = tokenizer.eos_token or ""
    if eos and sol.rstrip().endswith(eos):
        return prompt + sol
    return prompt + sol + eos
