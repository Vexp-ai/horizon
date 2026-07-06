"""Presenter (§1, §7.4) — formats the final answer.

In the prototype the "presentation" is lightweight: it removes the internal
reasoning (<think>) and returns the answer + (for math) the extracted final
answer, (for code) the selected code block. Isolated module so §10 can switch
pieces on/off (here the presenter always stays on).
"""
from __future__ import annotations

from dataclasses import dataclass

from common.text import extract_code, extract_final_answer, strip_think


@dataclass
class Presentation:
    domain: str
    answer: str | None      # final answer (math/science) or code (code)
    display: str            # text that can be shown to the user
    raw: str                # full generated text (with reasoning)


def present(domain: str, selected_text: str) -> Presentation:
    if domain == "code":
        code = extract_code(selected_text) or strip_think(selected_text)
        display = f"```python\n{code}\n```" if code else strip_think(selected_text)
        return Presentation(domain, code, display, selected_text)
    # math / science
    final = extract_final_answer(selected_text)
    body = strip_think(selected_text)
    display = body if body else selected_text
    if final:
        display = f"{display}\n\n**Answer:** {final}"
    return Presentation(domain, final, display, selected_text)
