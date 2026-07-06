"""Router v1 (optional) — lightweight classifier on embeddings (§7.1).

Use ONLY if v0 (similarity) makes too many mistakes. Logistic regression on
query embeddings, trained on (query -> domain) pairs labeled by the datasets
themselves (you know which dataset each example comes from).

Usage:
  # labels from data/processed/*.clean.jsonl (domain = file name)
  python -m router.router_v1 train
  python -m router.router_v1 eval
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from common.config import ROOT, base_config

PROCESSED = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "router" / "router_v1.pkl"
DOMAINS = ["math", "code", "science"]


MAX_CHARS = 800  # the domain signal lives in the opening; avoids OOM on long texts


def _embed(texts: list[str]):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(base_config()["router"]["embedding_model"])
    texts = [t[:MAX_CHARS] for t in texts]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True,
                        batch_size=16)


def _collect(max_per_domain: int = 3000) -> tuple[list[str], list[str]]:
    X, y = [], []
    for dom in DOMAINS:
        f = PROCESSED / f"{dom}.clean.jsonl"
        if not f.exists():
            print(f"[router_v1] missing {f} (run data.prepare) — skipping {dom}")
            continue
        # streaming: do NOT read the whole file (code.clean = ~1.8GB -> OOM);
        # the first N lines are enough, and only the opening of each problem.
        n = 0
        with f.open() as fh:
            for line in fh:
                if n >= max_per_domain:
                    break
                if not line.strip():
                    continue
                X.append(json.loads(line)["problem"][:MAX_CHARS])
                y.append(dom)
                n += 1
        print(f"[router_v1] {dom}: {n} examples")
    return X, y


def _augment_from_benchmark_train_splits(per_source: int = 500) -> tuple[list[str], list[str]]:
    """Benchmark-STYLE examples from the TRAIN/VALIDATION splits (DISJOINT from
    the test sets used in eval — zero contamination). Needed to cover the
    distribution shift: the code dataset (competitive programming) does not
    resemble HumanEval/MBPP ("write a function..."), and without this v1
    routed 100% of code to math.
    """
    from datasets import load_dataset

    from common.config import env

    tok = env("HF_TOKEN")
    X, y = [], []
    try:  # GSM8K train (word problems)
        ds = load_dataset("openai/gsm8k", "main", split="train", token=tok)
        for r in ds.select(range(min(per_source, len(ds)))):
            X.append(r["question"][:MAX_CHARS]); y.append("math")
    except Exception as e:
        print(f"[router_v1] augment gsm8k skip: {str(e)[:60]}")
    try:  # MBPP train ("write a function...")
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train", token=tok)
        for r in ds.select(range(min(per_source, len(ds)))):
            X.append((r.get("prompt") or r.get("text", ""))[:MAX_CHARS]); y.append("code")
    except Exception as e:
        print(f"[router_v1] augment mbpp skip: {str(e)[:60]}")
    try:  # MMLU validation STEM (science MC questions)
        ds = load_dataset("cais/mmlu", "all", split="validation", token=tok)
        stem = {"physics", "chemistry", "biology", "computer_science", "mathematics", "engineering"}
        n = 0
        for r in ds:
            if n >= per_source:
                break
            subj = str(r.get("subject", "")).lower()
            if any(s in subj for s in stem) and "math" not in subj:
                X.append(r["question"][:MAX_CHARS]); y.append("science")
                n += 1
    except Exception as e:
        print(f"[router_v1] augment mmlu skip: {str(e)[:60]}")
    print(f"[router_v1] augment from benchmark train splits: {len(X)} examples")
    return X, y


def train(args) -> None:
    from sklearn.linear_model import LogisticRegression

    texts, labels = _collect(args.max_per_domain)
    ax, ay = _augment_from_benchmark_train_splits()
    texts += ax
    labels += ay
    if not texts:
        raise SystemExit("No data: run data.prepare for the specialists first.")
    emb = _embed(texts)
    clf = LogisticRegression(max_iter=1000, C=args.C)
    clf.fit(emb, labels)
    with MODEL_PATH.open("wb") as f:
        pickle.dump({"clf": clf, "model": base_config()["router"]["embedding_model"]}, f)
    print(f"[router_v1] saved -> {MODEL_PATH} (train acc={clf.score(emb, labels):.3f})")


class RouterV1:
    def __init__(self):
        with MODEL_PATH.open("rb") as f:
            blob = pickle.load(f)
        self.clf = blob["clf"]
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(blob["model"], device="cpu")  # GPU = vLLM
        self.mode = base_config()["router"].get("mode", "deep")

    def route(self, query: str):
        from router.heuristics import detect_code
        from router.router_v0 import Route

        # deterministic pre-route for code (see router/heuristics.py)
        if detect_code(query):
            return Route(domain="code", mode=self.mode, scores={"code": 1.0, "heuristic": 1.0})
        e = self._model.encode([query[:MAX_CHARS]], normalize_embeddings=True)
        pred = self.clf.predict(e)[0]
        probs = dict(zip(self.clf.classes_, self.clf.predict_proba(e)[0].tolist()))
        return Route(domain=pred, mode=self.mode, scores=probs)


def evaluate(args) -> None:
    r = RouterV1()
    texts, labels = _collect(args.max_per_domain)
    correct = sum(r.route(t).domain == l for t, l in zip(texts, labels))
    print(f"[router_v1] acc={correct/len(texts):.3f} on {len(texts)} examples")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    t = sub.add_parser("train")
    t.add_argument("--max-per-domain", type=int, default=3000)
    t.add_argument("--C", type=float, default=1.0)
    t.set_defaults(func=train)
    e = sub.add_parser("eval")
    e.add_argument("--max-per-domain", type=int, default=500)
    e.set_defaults(func=evaluate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
