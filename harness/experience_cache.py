"""Phase 8c — Experience cache: the flywheel (§8c plan v2).

Persists (problem, solution, trace) for every success CERTIFIED BY THE
HARD VERIFIER; at inference, cosine-similarity lookup (CPU embeddings)
before generating: hit ⇒ near-instant answer.

ANTI-COLLAPSE RULE (§15.8): ONLY successes certified by the hard verifier
enter (test execution / Math-Verify in sanity), NEVER PRM-only or
self-consistency-only selections. The `certified_by` field documents this.

Storage: jsonl (append-only) + rebuildable .npy embedding matrix.

Usage:
    cache = ExperienceCache()
    hit = cache.lookup("Compute ...", threshold=0.95)
    if hit is None:
        ...generate and verify...
        cache.add(problem_text, solution_text, answer, certified_by="code_exec")
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.config import ROOT, base_config

CACHE_DIR = ROOT / "data" / "experience"
JSONL = CACHE_DIR / "cache.jsonl"
EMB = CACHE_DIR / "embeddings.npy"

_ALLOWED_CERTIFIERS = {"code_exec", "math_verify", "io_tests"}


@dataclass
class CacheHit:
    problem: str
    solution: str
    answer: str | None
    similarity: float
    certified_by: str


class ExperienceCache:
    def __init__(self, embedding_model: str | None = None):
        self.model_name = embedding_model or base_config()["router"]["embedding_model"]
        self._model = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict] = []
        self._emb: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if JSONL.exists():
            self._rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
        if EMB.exists() and self._rows:
            emb = np.load(EMB)
            if len(emb) == len(self._rows):
                self._emb = emb
            else:  # misaligned: rebuild
                self._emb = self._embed([r["problem"] for r in self._rows]) if self._rows else None
                if self._emb is not None:
                    np.save(EMB, self._emb)

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        m = self._ensure_model()
        return np.asarray(m.encode(texts, normalize_embeddings=True, batch_size=16))

    # ---- API ----
    def lookup(self, problem: str, threshold: float = 0.95) -> CacheHit | None:
        if not self._rows or self._emb is None or not len(self._emb):
            return None
        q = self._embed([problem])[0]
        sims = self._emb @ q
        i = int(np.argmax(sims))
        if float(sims[i]) < threshold:
            return None
        r = self._rows[i]
        return CacheHit(problem=r["problem"], solution=r["solution"],
                        answer=r.get("answer"), similarity=float(sims[i]),
                        certified_by=r["certified_by"])

    def add(self, problem: str, solution: str, answer: str | None,
            certified_by: str) -> bool:
        """Returns False (and does NOT save) if the certifier is not 'hard' (§15.8)."""
        if certified_by not in _ALLOWED_CERTIFIERS:
            return False
        row = {"problem": problem, "solution": solution, "answer": answer,
               "certified_by": certified_by}
        with JSONL.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        e = self._embed([problem])
        self._emb = e if self._emb is None or not len(self._emb) else np.vstack([self._emb, e])
        self._rows.append(row)
        np.save(EMB, self._emb)
        return True

    def stats(self) -> dict:
        from collections import Counter

        return {"n": len(self._rows),
                "by_certifier": dict(Counter(r["certified_by"] for r in self._rows))}
