"""Final-answer extraction and utilities on generated text (§7.3).

Key note (§15.3): answer extraction *from the generation* NEVER uses the
ground truth. Ground truth only enters scoring (verify_math/verify_code).
"""
from __future__ import annotations

import re

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Removes <think>...</think> reasoning blocks, keeping the answer."""
    return THINK_RE.sub("", text).strip()


def extract_boxed(text: str) -> str | None:
    r"""Extracts the content of the last \boxed{...} (MATH/AIME format).

    Handles nested braces by counting opens/closes.
    """
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


_FINAL_HINTS = [
    r"final answer[:\s]*is?\s*",
    r"the answer is\s*",
    r"answer[:\s]+",
    r"####\s*",  # GSM8K format
]


def extract_final_answer(text: str) -> str | None:
    """Best-effort for math: prefers \\boxed{}, then 'answer is' phrases, then
    the last number. Returns None if nothing plausible is found.
    """
    body = strip_think(text) or text
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    for pat in _FINAL_HINTS:
        m = list(re.finditer(pat, body, re.IGNORECASE))
        if m:
            tail = body[m[-1].end() :].strip()
            first_line = tail.splitlines()[0] if tail else ""
            if first_line:
                return first_line.strip().rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", body)
    if nums:
        return nums[-1].replace(",", "")
    return None


_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    """Extracts the last ```python ...``` code block from the generation."""
    body = strip_think(text) or text
    blocks = _CODE_FENCE.findall(body)
    if blocks:
        return blocks[-1].strip()
    return None
