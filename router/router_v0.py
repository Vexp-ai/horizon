"""Router v0 — embedding similarity (§7.1).

Pre-computes the embedding of 3 domain descriptions; for each query,
embedding + cosine -> specialist with the highest score. On STEM it ALWAYS
routes in `deep` mode (long reasoning + best-of-N, §7.1).

Router quality is a *measured variable* (§1 deferred components), not
the thing to perfect now. `evaluate()` produces accuracy + confusion matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

from common.config import base_config

DOMAIN_DESCRIPTIONS = {
    # NB: descriptions tuned on the eval (router_v0 66% -> the fix): GSM8K
    # word problems ended up on science; here we anchor them explicitly to math.
    "math": "mathematics: solve for a number. Arithmetic word problems about money, "
            "ages, rates, quantities to compute; competition math, algebra, equations, "
            "number theory, geometry, calculus, fractions, percentages, proofs. "
            "The answer is a number or formula.",
    "code": "computer programming: write a function or program. Algorithms, data "
            "structures, code generation, debugging, Python, unit tests, "
            "competitive programming with stdin/stdout.",
    "science": "natural sciences knowledge questions: physics laws, chemistry reactions "
               "and molecules, biology, astronomy. Multiple-choice conceptual science "
               "questions about facts, mechanisms and phenomena (not numeric word problems).",
}
DOMAINS = list(DOMAIN_DESCRIPTIONS)


@dataclass
class Route:
    domain: str
    mode: str  # always "deep" in the STEM prototype
    scores: dict[str, float]


class RouterV0:
    def __init__(self, embedding_model: str | None = None, mode: str | None = None):
        bc = base_config()
        self.model_name = embedding_model or bc["router"]["embedding_model"]
        self.mode = mode or bc["router"].get("mode", "deep")
        self._model = None
        self._domain_emb = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # device=cpu: at runtime the GPU is occupied by vLLM (embedding a
            # query on CPU costs ms); on GPU it would OOM (measured, baseline 4).
            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._domain_emb = self._model.encode(
                [DOMAIN_DESCRIPTIONS[d] for d in DOMAINS],
                normalize_embeddings=True,
            )

    def route(self, query: str) -> Route:
        # deterministic pre-embedding heuristic (measured: misrouting code
        # to math cost -16 points on MBPP in the full system)
        from router.heuristics import detect_code

        if detect_code(query):
            return Route(domain="code", mode=self.mode, scores={"code": 1.0, "heuristic": 1.0})
        self._ensure()
        q = self._model.encode([query], normalize_embeddings=True)[0]
        # cosine = dot (normalized embeddings)
        sims = {d: float(q @ self._domain_emb[i]) for i, d in enumerate(DOMAINS)}
        best = max(sims, key=sims.get)
        return Route(domain=best, mode=self.mode, scores=sims)

    def route_batch(self, queries: list[str]) -> list[Route]:
        return [self.route(q) for q in queries]

    def evaluate(self, labeled: list[tuple[str, str]]) -> dict:
        """labeled: [(query, gold_domain)]. Returns accuracy + confusion matrix (§7.1)."""
        cm = {g: {p: 0 for p in DOMAINS} for g in DOMAINS}
        correct = 0
        for query, gold in labeled:
            pred = self.route(query).domain
            cm[gold][pred] += 1
            correct += int(pred == gold)
        acc = correct / len(labeled) if labeled else 0.0
        return {"accuracy": acc, "n": len(labeled), "confusion_matrix": cm}


if __name__ == "__main__":  # quick smoke test (requires sentence-transformers + network)
    import json
    import sys

    r = RouterV0()
    for q in sys.argv[1:] or ["Find all primes p such that p^2+2 is prime."]:
        print(json.dumps(r.route(q).__dict__, indent=2))
