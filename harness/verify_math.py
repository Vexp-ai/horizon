"""Math verification (§7.3).

CRITICAL DISTINCTION (§7.3, §15.3):
  - INTERNAL selection (at inference you do NOT have the answer): self-consistency —
    sample N solutions, extract each one's answer, take the MAJORITY.
    NEVER uses the ground truth.
  - Benchmark SCORING (you have the ground truth): Math-Verify (symbolic
    equivalence) against the correct answer.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from common.text import extract_final_answer


@dataclass
class MathSelection:
    answer: str | None            # chosen answer (majority)
    votes: int                    # votes for the winning answer
    n: int                        # total samples
    distribution: dict[str, int]  # answer -> count (maj@k)
    best_text: str                # text of the first sample with the winning answer


# --------- internal selection: self-consistency (no ground truth) ---------

def _canon(ans: str | None) -> str | None:
    if ans is None:
        return None
    a = ans.strip().strip("$").replace(" ", "")
    a = a.rstrip(".")
    return a or None


def self_consistency(samples: list[str]) -> MathSelection:
    """Majority vote over the answers extracted from N samples (§7.3)."""
    extracted = [extract_final_answer(s) for s in samples]
    canon = [_canon(a) for a in extracted]
    dist = Counter(a for a in canon if a)
    if not dist:
        return MathSelection(None, 0, len(samples), {}, samples[0] if samples else "")
    winner, votes = dist.most_common(1)[0]
    best_text = next(
        (samples[i] for i, a in enumerate(canon) if a == winner), samples[0]
    )
    return MathSelection(winner, votes, len(samples), dict(dist), best_text)


# --------- benchmark scoring: Math-Verify against ground truth ---------

def verify_answer(predicted: str | None, gold: str) -> bool:
    """True if `predicted` is mathematically equivalent to `gold` (§7.3 scoring).

    Prefers Math-Verify (symbolic equivalence); falls back to canonical comparison.
    """
    if predicted is None:
        return False
    try:
        from math_verify import parse, verify  # open-r1 repo

        gold_parsed = parse(gold if "\\boxed" in gold else f"\\boxed{{{gold}}}")
        pred_parsed = parse(
            predicted if "\\boxed" in predicted else f"\\boxed{{{predicted}}}"
        )
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        # Fallback: canonical string/number comparison.
        return _fallback_equal(predicted, gold)


def _fallback_equal(a: str, b: str) -> bool:
    ca, cb = _canon(a), _canon(b)
    if ca is None or cb is None:
        return False
    if ca == cb:
        return True
    try:
        return abs(float(ca) - float(cb)) < 1e-6
    except ValueError:
        return False


def score_samples(samples: list[str], gold: str) -> dict:
    """§7.3 metrics for one problem: pass@1 (first sample) and maj@k."""
    pass1 = verify_answer(extract_final_answer(samples[0]), gold) if samples else False
    sel = self_consistency(samples)
    maj = verify_answer(sel.answer, gold)
    return {
        "pass@1": bool(pass1),
        "maj@k": bool(maj),
        "k": len(samples),
        "selected": sel.answer,
        "votes": sel.votes,
    }
