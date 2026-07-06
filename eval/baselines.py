"""The 6 baselines of §10 as configurations of the SAME pipeline (§7.4).

Anti-trap rule (§10, §15.2): the verifier helps everyone. So we isolate the
contributions by turning pieces on/off:

  1. base_naked        Bare base, single-shot                           -> the floor
  2. base_specialist   Base + single specialist (no verification)       -> does specialization help?
  3. base_verifier     Base + verification + best-of-N (no specialists) -> verification alone
  4. full_system       Router + specialists + verification + best-of-N  -> the architecture
  5. deepseek_single   DeepSeek V4 Flash single-shot (API)              -> the ceiling / gate
  6. deepseek_boN      DeepSeek V4 Flash + best-of-N (API)              -> apples-to-apples honesty

Each entry returns (ModelClient, PipelineConfig). See eval/run_benchmarks.py.
"""
from __future__ import annotations

from typing import Callable

from common.model_client import ModelClient
from harness.pipeline import PipelineConfig


def _vllm():
    return ModelClient.vllm()


def _openrouter():
    return ModelClient.openrouter()


def base_naked(_domain=None):
    return _vllm(), PipelineConfig(
        use_router=False, use_specialist=False, use_verifier=False,
    )


def base_specialist(forced_domain: str | None = None):
    # Router off, verification off: isolates the contribution of specialization alone (§10.2).
    return _vllm(), PipelineConfig(
        use_router=False, use_specialist=True, use_verifier=False,
        forced_domain=forced_domain,
    )


def base_verifier(_domain=None):
    # Specialists off, verification+best-of-N on: isolates verification alone (§10.3).
    return _vllm(), PipelineConfig(
        use_router=False, use_specialist=False, use_verifier=True,
    )


def full_system(_domain=None):
    return _vllm(), PipelineConfig(
        use_router=True, use_specialist=True, use_verifier=True,
    )


def full_system_repair(_domain=None):
    # 8d: like full_system but the code path uses the agentic repair loop
    # (generate→execute→repair; adaptive budget <= best-of-N: honest comparison with baseline 4).
    return _vllm(), PipelineConfig(
        use_router=True, use_specialist=True, use_verifier=True, use_repair=True,
    )


def base_verifier_repair(_domain=None):
    # CLEAN 8d (measured 2026-07-04): the v1 router misroutes LCB-style problems
    # to math (b4 final 12.2 vs b3 38) — to isolate repair-vs-boN from
    # misrouting, this baseline replicates b3 (router off) with the repair loop.
    return _vllm(), PipelineConfig(
        use_router=False, use_specialist=False, use_verifier=True, use_repair=True,
    )


def _api_ceiling(model: str, with_verifier: bool):
    # max_tokens 16000: R1 reasons beyond 8192 (measured: 8/10 math500 failures
    # were truncations). Applies to all API ceilings: budget must not be the limit.
    return _openrouter(), PipelineConfig(
        use_router=False, use_specialist=False, use_verifier=with_verifier,
        model_override=model, reasoning_effort="high",  # §9
        max_tokens_override=16000,
    )


def deepseek_single(_domain=None):
    from common.config import env

    return _api_ceiling(env("DEEPSEEK_CEILING_MODEL", "deepseek/deepseek-v4-flash"), False)


def deepseek_boN(_domain=None):
    from common.config import env

    return _api_ceiling(env("DEEPSEEK_CEILING_MODEL", "deepseek/deepseek-v4-flash"), True)


# Additional ceiling (user decision): DeepSeek-R1 = "the father" (teacher of the
# distilled base). Question: does the system on consumer hardware BEAT the father?
# (V4 Flash remains the §0.2 gate; R1 is the identity comparison.)
def r1_single(_domain=None):
    from common.config import env

    return _api_ceiling(env("DEEPSEEK_R1_MODEL", "deepseek/deepseek-r1"), False)


def r1_boN(_domain=None):
    from common.config import env

    return _api_ceiling(env("DEEPSEEK_R1_MODEL", "deepseek/deepseek-r1"), True)


BASELINES: dict[str, Callable] = {
    "1_base_naked": base_naked,
    "2_base_specialist": base_specialist,
    "3_base_verifier": base_verifier,
    "4_full_system": full_system,
    "4b_full_system_repair": full_system_repair,
    "3b_verifier_repair": base_verifier_repair,
    "5_deepseek_single": deepseek_single,
    "6_deepseek_boN": deepseek_boN,
    "5r1_single": r1_single,
    "6r1_boN": r1_boN,
}

# which baselines require OpenRouter network access (§9)
API_BASELINES = {"5_deepseek_single", "6_deepseek_boN", "5r1_single", "6r1_boN"}
