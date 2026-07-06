"""Openai-compatible generation client (§7.2, §9).

A single abstraction over two backends:
  - local vLLM (Tier A): the `model` field selects the LoRA adapter
    (e.g. "horizon-math-lora") or the base model.
  - OpenRouter (DeepSeek ceiling, §9).

Both speak the OpenAI chat/completions API, so we use `openai`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .config import base_config, env


@dataclass
class GenResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    raw: Any = field(default=None, repr=False)


class ModelClient:
    """Wrapper around an openai-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        default_model: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        from openai import OpenAI  # lazy import: the dep is only needed at runtime

        # Configurable HTTP timeout: on local CPU a best-of-N request can
        # take >10 min (openai default ~600s -> mass 'Connection error',
        # measured: 20/25 errors in the first Tier A run).
        timeout_s = float(env("HORIZON_HTTP_TIMEOUT", "600"))
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY",
                             timeout=timeout_s)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

    # ---- factory ----
    @classmethod
    def vllm(cls, model: str | None = None) -> "ModelClient":
        """Local Tier A backend. `model` = adapter name or base."""
        base_url = env("VLLM_BASE_URL") or base_config()["serving"]["base_url"]
        return cls(base_url=base_url, api_key="EMPTY", default_model=model)

    @classmethod
    def openrouter(cls) -> "ModelClient":
        """DeepSeek ceiling backend (§9)."""
        return cls(
            base_url=env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=env("OPENROUTER_API_KEY", required=True),
            default_model=env("DEEPSEEK_CEILING_MODEL", "deepseek/deepseek-v4-flash"),
            extra_headers={
                "HTTP-Referer": "https://github.com/horizon-stem-prototype",
                "X-Title": "horizon-stem-prototype",
            },
        )

    # ---- generation ----
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        n: int = 1,
        reasoning_effort: str | None = None,
        stop: list[str] | None = None,
        max_retries: int = 4,
    ) -> list[GenResult]:
        """Returns `n` completions. For n>1 it prefers the native `n` param
        (one request) and, if the backend does not support it, falls back to
        n calls.
        """
        model = model or self.default_model
        if model is None:
            raise ValueError("No model specified (neither default nor arg).")

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if stop:
            kwargs["stop"] = stop
        if reasoning_effort:  # DeepSeek/OpenRouter reasoning (§9)
            kwargs["reasoning_effort"] = reasoning_effort
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        # attempt with native `n`
        try:
            return self._call(kwargs, n=n)
        except Exception:
            if n == 1:
                raise
            # fallback: n independent calls
            out: list[GenResult] = []
            for _ in range(n):
                out.extend(self._call(kwargs, n=1, max_retries=max_retries))
            return out

    def _call(self, kwargs: dict, n: int, max_retries: int = 4) -> list[GenResult]:
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                t0 = time.monotonic()
                resp = self.client.chat.completions.create(**kwargs, n=n)
                dt = time.monotonic() - t0
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                results = []
                for choice in resp.choices:
                    results.append(
                        GenResult(
                            text=choice.message.content or "",
                            prompt_tokens=pt,
                            completion_tokens=ct // max(len(resp.choices), 1),
                            latency_s=dt,
                            raw=choice,
                        )
                    )
                return results
            except Exception as e:  # exponential backoff
                last_err = e
                msg = str(e)
                # NON-retryable errors: credits exhausted (402) / auth (401)
                if "402" in msg or "401" in msg or "insufficient" in msg.lower():
                    raise RuntimeError(f"Non-retryable provider error: {msg[:300]}")
                if attempt == max_retries - 1:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"Generation failed after {max_retries} attempts: {last_err}")
