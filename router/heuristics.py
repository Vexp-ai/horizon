"""Deterministic pre-routing heuristics (§7.1, post-gate iteration).

MOTIVATION (measured, eval 2026-07-02): the embedding router sent 17% of
MBPP and ~50% of HumanEval down the math path -> no test execution -> the 92
of baseline 3 on MBPP collapsed to 76 in the full system.

v1.2 (measured 2026-07-04): COMPETITIVE PROGRAMMING-style problems
(LiveCodeBench) do not contain "write a function"/"def"/"assert" — they are
algorithmic statements with Input/Output/Constraints/Example sections. The
classifier routed 59/100 of them to math (verification disabled -> full system
12.2 vs 38 for the verifier alone). Real example of the failure: "longest
increasing subsequence ... Input: first line n ... Output the length" -> math.

Code prompts have UNAMBIGUOUS syntactic signals that don't require an
embedding to be picked up. If the prompt "speaks code", route it to code
before even consulting the classifier. Conservative: only matches very
strong patterns, everything else goes to the learned router.
"""
from __future__ import annotations

import re

_CODE_PATTERNS = [
    r"```",                          # code fence
    r"\bdef [a-zA-Z_]\w*\s*\(",      # Python function signature
    r"\bassert\b",                   # test/assert in the prompt (HumanEval/MBPP)
    r">>> ",                         # doctest
    r"\bclass [A-Z]\w*\s*[(:]",      # class definition
    r"\breturn\b.*\n",               # function body in the prompt
    r"\bwrite a (python )?(function|program|script)\b",
    r"\bimplement (a|the) (function|method|class|algorithm)\b",
    r"\bstdin\b|\bstdout\b",         # competitive programming I/O
    r"\bunit tests?\b|\btest cases?\b",
    r"\bYour code should pass\b",    # MBPP protocol
    # ---- v1.2: competitive programming markers (LiveCodeBench-style) ----
    r"\b(sample )?(input|output)\s*:",         # "Input:" / "Output:" anywhere (with colon)
    r"^\s*#{0,3}\s*(sample )?(input|output)\s*$",  # I/O header on its own line (no colon)
    r"\boutputs? (the|one|a single)\b",        # "Output the length ..."
    r"^\s*#{0,3}\s*constraints\s*[:\n]",       # "Constraints:" section
    r"^\s*#{0,3}\s*example\s+\d+\s*[:\n]",     # "Example 1:" (LeetCode/LCB)
    r"\bstandard (input|output)\b",            # "read from standard input"
    r"\bfirst line\b.{0,40}\b(contains?|consists?|of (the )?input)\b",
    r"\bclass Solution\b",                     # LeetCode call-based signature
    r"\breturn the\b",                         # "Return the maximum ..." (call-based)
    r"\bprints? (the|a|one|each|all)\b",       # "print the answer"
]
# MULTILINE: the Input/Output/Constraints sections anchor to line start
_CODE_RE = re.compile("|".join(_CODE_PATTERNS), re.IGNORECASE | re.MULTILINE)


def detect_code(prompt: str) -> bool:
    """True if the prompt is unambiguously a code request."""
    return bool(_CODE_RE.search(prompt))
