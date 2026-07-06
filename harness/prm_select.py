"""PRM selection shared between the offline pilot (prm_rank) and the runtime (pipeline).

Strategies: prm_argmax_<agg> | prm_weighted_<agg>, agg in {min, prod, last, mean}.
The same function decides in both contexts: zero pilot/prod divergence.
"""
from __future__ import annotations

import math
from collections import defaultdict


def agg(scores: list[float], how: str) -> float:
    if not scores:
        return 0.0
    if how == "min":
        return min(scores)
    if how == "prod":
        return math.exp(sum(math.log(max(s, 1e-6)) for s in scores))
    if how == "last":
        return scores[-1]
    return sum(scores) / len(scores)  # mean


def parse_strategy(name: str) -> tuple[str, str]:
    """'prm_weighted_min' -> ('weighted', 'min'). Raises on unknown names."""
    parts = name.split("_")
    if len(parts) != 3 or parts[0] != "prm" or parts[1] not in ("argmax", "weighted") \
            or parts[2] not in ("min", "prod", "last", "mean"):
        raise ValueError(f"unknown PRM strategy: {name}")
    return parts[1], parts[2]


def select(answers: list[str | None], prm_steps: list[list[float]],
           strategy: str) -> tuple[int, str | None]:
    """Returns (index of the chosen sample, chosen answer).

    - argmax: sample with the highest aggregated score
    - weighted: majority vote weighted per extracted answer
    """
    mode, how = parse_strategy(strategy)
    w = [agg(p, how) for p in prm_steps]
    if not answers:
        return 0, None
    if mode == "argmax":
        i = max(range(len(answers)), key=lambda k: w[k])
        return i, answers[i]
    votes: dict[str, float] = defaultdict(float)
    for a, wi in zip(answers, w):
        if a:
            votes[a] += wi
    if not votes:
        return 0, None
    best_ans = max(votes, key=votes.get)
    # representative: the sample for that answer with the maximum score
    idxs = [i for i, a in enumerate(answers) if a == best_ans]
    i = max(idxs, key=lambda k: w[k])
    return i, best_ans


class PRMScorer:
    """Runtime scorer (lazy). device: 'cuda' (pod) or 'cpu' (Tier A).

    Default model: Qwen/Qwen2.5-Math-PRM-7B (VERIFY <extra_0> format).
    For the consumer envelope consider ThinkPRM-1.5B (Apache) — same interface.
    """

    def __init__(self, model_name: str, device: str = "cuda", max_tokens: int = 4096):
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self._model = None
        self._tok = None
        self._sep_id = None

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self._model = AutoModel.from_pretrained(
                self.model_name, torch_dtype=dtype, trust_remote_code=True,
                device_map=self.device,
            ).eval()
            self._sep_id = self._tok.encode("<extra_0>")[0]

    def score_steps(self, question: str, steps: list[str]) -> list[float]:
        import torch

        self._ensure()
        resp = "<extra_0>".join(steps) + "<extra_0>"
        msgs = [
            {"role": "system", "content": "Please reason step by step."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": resp},
        ]
        text = self._tok.apply_chat_template(msgs, tokenize=False)
        ids = self._tok(text, return_tensors="pt", truncation=True,
                        max_length=self.max_tokens).input_ids.to(self._model.device)
        with torch.no_grad():
            logits = self._model(input_ids=ids)[0]
        probs = torch.softmax(logits, dim=-1)[0, :, 1]
        mask = (ids[0] == self._sep_id)
        return probs[mask].float().tolist()
