"""Full harness (§7.4): route -> specialist -> generate (code-exec for
code) -> verify/select -> retry/escalation -> present.

Modular by construction: the flags in `PipelineConfig` switch pieces on/off,
so the 6 baselines of §10 are obtained with the SAME class (see eval/baselines.py).

Returns a `Solution` that carries the fields needed for scoring (§7.3): the raw
samples, the internal selection, and the hardware/token metrics for §8.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from common.benchmarks import Problem
from common.config import base_config
from common.model_client import GenResult, ModelClient
from common.text import extract_code, extract_final_answer
from harness import verify_code, verify_math
from harness.presenter import Presentation, present

SYSTEM_PROMPT = (
    "You are a careful STEM expert. Think step by step inside <think>...</think>, "
    "then give the final answer. For math, put the final answer in \\boxed{}. "
    "For code, return a single complete ```python``` block."
)


@dataclass
class PipelineConfig:
    """Switches for baselines (§10)."""
    use_router: bool = True          # if False, use forced_domain
    use_specialist: bool = True      # if False, use the bare base (adapter off)
    use_verifier: bool = True        # if False, single-shot (N=1, no self-consistency/exec)
    best_of_n: Optional[int] = None  # override N; if None use per-domain defaults
    forced_domain: Optional[str] = None  # for Baseline 2 (fixed specialist) or router off
    model_override: Optional[str] = None  # for DeepSeek ceiling: API model id
    max_retries_code: int = 2        # retry with error feedback (§7.3)
    reasoning_effort: Optional[str] = None  # for OpenRouter (§9)
    max_tokens_override: Optional[int] = None  # e.g. R1: 8192 truncates the reasoning (ceiling bug)
    use_repair: Optional[bool] = None  # 8d: repair loop on the code path; None = default from config


@dataclass
class Solution:
    problem_id: str
    domain: str
    samples: list[str]
    presentation: Presentation
    # §8 metrics
    n_samples: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    route_scores: dict = field(default_factory=dict)
    internal_selection: Optional[str] = None


class Pipeline:
    def __init__(self, client: ModelClient, config: PipelineConfig | None = None,
                 router=None):
        self.client = client
        self.cfg = config or PipelineConfig()
        self.bc = base_config()
        self._router = router  # inject RouterV0/V1; lazily loaded if needed

    # ---- routing (§7.1) ----
    def _route(self, problem: Problem):
        if self.cfg.forced_domain:
            return self.cfg.forced_domain, {}
        if not self.cfg.use_router:
            return problem.domain, {}  # known benchmark label (router off)
        if self._router is None:
            version = self.bc["router"].get("version", "v0")
            if version == "v1":
                try:
                    from router.router_v1 import RouterV1

                    self._router = RouterV1()
                except Exception as e:  # missing pkl -> fallback v0
                    print(f"[pipeline] router v1 not available ({e}); falling back to v0")
                    version = "v0"
            if self._router is None:
                from router.router_v0 import RouterV0

                self._router = RouterV0()
        r = self._router.route(problem.prompt)
        return r.domain, r.scores

    def _model_name(self, domain: str) -> str | None:
        if self.cfg.model_override:            # DeepSeek ceiling (§9)
            return self.cfg.model_override
        enabled = self.bc.get("specialist_enabled", {})
        if self.cfg.use_specialist and enabled.get(domain, True):  # LoRA adapter (§7.2, v1.1)
            return self.bc["adapters"].get(domain)
        return self.bc["base_model"]["tier_a"]  # bare base (Baseline 1/3 or specialist off)

    def _n(self, domain: str) -> int:
        if not self.cfg.use_verifier:
            return 1
        if self.cfg.best_of_n:
            return self.cfg.best_of_n
        key = "code_default" if domain == "code" else "math_default"
        return self.bc["best_of_n"][key]

    def _messages(self, prompt: str, extra_user: str | None = None):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
        if extra_user:
            # merged into the same user turn: strict chat templates (Mistral
            # family) reject consecutive same-role messages with HTTP 400
            msgs[-1] = {"role": "user",
                        "content": msgs[-1]["content"] + "\n\n" + extra_user}
        return msgs

    _exp_cache = None  # 8c flywheel (singleton, only if enabled)

    # ---- main loop ----
    def solve(self, problem: Problem) -> Solution:
        t0 = time.monotonic()
        rcfg = self.bc.get("runtime", {})
        if rcfg.get("experience_cache"):
            if Pipeline._exp_cache is None:
                from harness.experience_cache import ExperienceCache

                Pipeline._exp_cache = ExperienceCache()
            hit = Pipeline._exp_cache.lookup(
                problem.prompt, threshold=float(rcfg.get("cache_threshold", 0.95)))
            if hit is not None:
                sol = Solution(
                    problem_id=problem.id, domain=problem.domain, samples=[hit.solution],
                    presentation=present(problem.domain, hit.solution),
                    internal_selection=hit.answer,
                )
                sol.n_samples = 0  # no generation: cache hit
                sol.latency_s = time.monotonic() - t0
                sol.route_scores = {"cache_hit": hit.similarity}
                return sol
        domain, scores = self._route(problem)
        model = self._model_name(domain)
        n = self._n(domain)
        g = self.bc["generation"]
        temperature = g["greedy_temperature"] if n == 1 else g["temperature"]

        # Phase 8d: repair loop on the code path — manages the generation budget
        # ON ITS OWN (adaptive: 1 gen on easy problems, initial_k+max_iters on hard ones).
        use_repair = (self.cfg.use_repair if self.cfg.use_repair is not None
                      else bool(self.bc.get("repair_loop", {}).get("enabled")))
        if domain == "code" and use_repair and self.cfg.use_verifier:
            sol = self._solve_code_repair(problem, domain, model, g)
            if sol is not None:
                sol.n_samples = len(sol.samples)
                sol.latency_s = time.monotonic() - t0
                sol.route_scores = {**scores, **sol.route_scores}
                return sol
            # None = no freezable tests: continue with the classic path

        # Phase 7 (early-stop SC, OFF in the baselines): for math, sample in blocks
        # and stop when the majority is mathematically decided.
        vcfg = self.bc.get("verifier", {})
        if (domain != "code" and n > 4 and self.cfg.use_verifier
                and vcfg.get("early_stop_sc")):
            results = self._sample_early_stop(problem, model, n, g)
        else:
            results = self.client.generate(
                self._messages(problem.prompt),
                model=model,
                temperature=temperature,
                top_p=g["top_p"],
                max_tokens=self.cfg.max_tokens_override or g["max_tokens"],
                n=n,
                reasoning_effort=self.cfg.reasoning_effort,
            )
        samples = [r.text for r in results]
        total_tokens = sum(r.completion_tokens + r.prompt_tokens for r in results)

        if domain == "code":
            sol = self._finish_code(problem, domain, model, samples, g)
        else:
            sol = self._finish_math(problem, domain, samples)

        # code retry may have added samples; recompute
        sol.problem_id = problem.id
        sol.domain = domain
        sol.n_samples = len(sol.samples)
        sol.total_tokens = total_tokens
        sol.latency_s = time.monotonic() - t0
        sol.route_scores = scores
        return sol

    def _finish_code_lcb(self, problem, domain, model, samples, candidates, g) -> Solution:
        mode = problem.meta["exec_mode"]
        fn = problem.meta.get("fn_name")
        pub = problem.meta.get("io_public", [])

        def best_of(cands):
            scores = [verify_code.run_io_tests(c, pub, mode, fn, require_all=False)[0]
                      for c in cands]
            bi = max(range(len(cands)), key=lambda i: scores[i]) if cands else 0
            return bi, (scores[bi] if cands else 0)

        best_i, best_pass = best_of(candidates)
        retries = 0
        while (self.cfg.use_verifier and pub and best_pass < len(pub)
               and retries < self.cfg.max_retries_code):
            feedback = ("Your solution fails some of the provided example tests. "
                        "Re-read the problem, fix the code, and return ONE complete "
                        "```python``` block.")
            more = self.client.generate(
                self._messages(problem.prompt, feedback), model=model,
                temperature=g["temperature"], top_p=g["top_p"],
                max_tokens=self.cfg.max_tokens_override or g["max_tokens"], n=1,
                reasoning_effort=self.cfg.reasoning_effort,
            )
            samples.append(more[0].text)
            candidates.append(extract_code(more[0].text) or more[0].text)
            best_i, best_pass = best_of(candidates)
            retries += 1
        if pub and best_pass == len(pub):
            self._maybe_cache_add(problem, samples[best_i],
                                  candidates[best_i] if candidates else None, "io_tests")
        return Solution(
            problem_id=problem.id, domain=domain, samples=samples,
            presentation=present(domain, samples[best_i]),
            internal_selection=candidates[best_i] if candidates else None,
        )

    def _sample_early_stop(self, problem, model, n_max: int, g) -> list[GenResult]:
        """Sample in blocks of 4; stop when the leader has more votes than the
        number of samples remaining (majority impossible to overturn). Reduces the
        effective N without changing the selection rule (§7.3 v2)."""
        from collections import Counter

        out: list[GenResult] = []
        while len(out) < n_max:
            chunk = min(4, n_max - len(out))
            out += self.client.generate(
                self._messages(problem.prompt), model=model,
                temperature=g["temperature"], top_p=g["top_p"],
                max_tokens=self.cfg.max_tokens_override or g["max_tokens"], n=chunk,
                reasoning_effort=self.cfg.reasoning_effort,
            )
            answers = [extract_final_answer(r.text) for r in out]
            dist = Counter(a for a in answers if a)
            if dist:
                leader_votes = dist.most_common(1)[0][1]
                second = dist.most_common(2)[1][1] if len(dist) > 1 else 0
                remaining = n_max - len(out)
                if leader_votes - second > remaining:
                    break  # no outcome can overturn the majority
        return out

    def _finish_math(self, problem, domain, samples) -> Solution:
        if len(samples) == 1:
            selected = samples[0]
            internal = extract_final_answer(selected)
        else:
            selector = self.bc.get("verifier", {}).get("math_selector", "sc")
            if selector != "sc":
                try:
                    selected, internal = self._select_math_prm(problem, samples, selector)
                except Exception as e:  # PRM not available -> fallback sc
                    print(f"[pipeline] PRM selector failed ({str(e)[:80]}); falling back to sc")
                    selector = "sc"
            if selector == "sc":
                sel = verify_math.self_consistency(samples)  # no ground truth (§15.3)
                selected = sel.best_text
                internal = sel.answer
        return Solution(
            problem_id=problem.id, domain=domain, samples=samples,
            presentation=present(domain, selected), internal_selection=internal,
        )

    def _maybe_cache_add(self, problem, solution_text: str, answer, certifier: str):
        """8c flywheel: save ONLY hard-certified successes (§15.8), if enabled."""
        if not self.bc.get("runtime", {}).get("experience_cache"):
            return
        if Pipeline._exp_cache is None:
            from harness.experience_cache import ExperienceCache

            Pipeline._exp_cache = ExperienceCache()
        Pipeline._exp_cache.add(problem.prompt, solution_text, answer, certified_by=certifier)

    _prm_scorer = None  # class singleton (the PRM is loaded once)

    def _select_math_prm(self, problem, samples: list[str], strategy: str):
        """Runtime selection with the PRM (Phase 8b). The strategy comes from the pilot."""
        from harness import prm_select
        from harness.prm_score import split_steps

        vcfg = self.bc.get("verifier", {})
        if Pipeline._prm_scorer is None:
            Pipeline._prm_scorer = prm_select.PRMScorer(
                vcfg.get("prm_model", "Qwen/Qwen2.5-Math-PRM-7B"),
                device=vcfg.get("prm_device", "cuda"),
            )
        scores = [Pipeline._prm_scorer.score_steps(problem.prompt, split_steps(s))
                  for s in samples]
        answers = [extract_final_answer(s) for s in samples]
        i, ans = prm_select.select(answers, scores, strategy)
        return samples[i], ans

    def _solve_code_repair(self, problem, domain, model, g) -> Optional[Solution]:
        """Phase 8d: generate→execute→repair on FROZEN tests. Returns None if
        there is no test to iterate on (the caller falls back to the classic path)."""
        from harness import repair_loop

        vcfg = self.bc.get("verifier", {})
        certifier = None
        if problem.meta.get("lcb"):
            pub = problem.meta.get("io_public", [])
            if not pub:
                return None
            tester = repair_loop.io_tester(pub, problem.meta["exec_mode"],
                                           problem.meta.get("fn_name"))
            certifier = "io_tests"
        elif problem.public_tests:
            tester = repair_loop.assert_tester(problem.public_tests)
            certifier = "code_exec"
        elif vcfg.get("code_selftests"):
            # CodeT: generate the suites ONCE and FREEZE them (rule 1 of 8d)
            from harness import gen_tests

            try:
                gts = gen_tests.generate_test_suites(
                    self.client, model, problem.prompt,
                    k_suites=int(vcfg.get("selftest_suites", 2)), gen_cfg=g)
            except Exception as e:  # noqa: BLE001
                print(f"[pipeline] selftests for repair failed ({str(e)[:80]})")
                gts = []
            if not gts:
                return None
            tester = repair_loop.assert_tester(gts)
            certifier = None  # selftests are NOT hard certifiers (§15.8)
        else:
            return None

        rl = self.bc.get("repair_loop", {})
        out = repair_loop.repair_solve(
            self.client, model, problem.prompt, tester,
            gen_cfg=g, initial_k=int(rl.get("initial_k", 2)),
            max_iters=int(rl.get("max_iters", 4)),
            max_tokens=self.cfg.max_tokens_override or g["max_tokens"],
            reasoning_effort=self.cfg.reasoning_effort,
            system_prompt=SYSTEM_PROMPT,
        )
        if certifier and out.report.passed_all:
            self._maybe_cache_add(problem, out.samples[out.best_index],
                                  out.candidates[out.best_index], certifier)
        sol = Solution(
            problem_id=problem.id, domain=domain, samples=out.samples,
            presentation=present(domain, out.samples[out.best_index]),
            internal_selection=out.candidates[out.best_index],
        )
        sol.total_tokens = out.total_tokens
        sol.route_scores = {"repair_iterations": out.iterations,
                            "repair_pass_frac": round(out.report.frac, 3)}
        return sol

    def _finish_code(self, problem, domain, model, samples, g) -> Solution:
        candidates = [extract_code(s) or s for s in samples]
        # LiveCodeBench: internal selection on PUBLIC I/O tests (§7.3, never the private ones)
        if problem.meta.get("lcb"):
            return self._finish_code_lcb(problem, domain, model, samples, candidates, g)
        public = problem.public_tests or []
        # §7.3 auto-generated tests (CodeT): when there are NO public tests, generate
        # asserts from the problem statement and select by dual-execution agreement.
        vcfg = self.bc.get("verifier", {})
        if (not public and self.cfg.use_verifier and vcfg.get("code_selftests")
                and len(candidates) > 1):
            from harness import gen_tests

            try:
                gts = gen_tests.generate_test_suites(
                    self.client, model, problem.prompt,
                    k_suites=int(vcfg.get("selftest_suites", 2)), gen_cfg=g,
                )
                if gts:
                    dual = gen_tests.select_by_dual_execution(candidates, gts)
                    if dual.n_tests_used > 0 and dual.consensus > 0:
                        selected_text = samples[dual.best_index]
                        return Solution(
                            problem_id=problem.id, domain=domain, samples=samples,
                            presentation=present(domain, selected_text),
                            internal_selection=candidates[dual.best_index],
                        )
            except Exception as e:  # noqa: BLE001 — fall back to the classic path
                print(f"[pipeline] selftests failed ({str(e)[:80]}); falling back to compile-check")
        best_i, exec_res = verify_code.select_best(candidates, public)
        # retry with error feedback if it fails and public tests exist (§7.3)
        retries = 0
        while (self.cfg.use_verifier and public and exec_res and not exec_res.passed
               and retries < self.cfg.max_retries_code):
            feedback = verify_code.retry_feedback(exec_res.stderr or exec_res.stdout)
            more = self.client.generate(
                self._messages(problem.prompt, feedback),
                model=model, temperature=g["temperature"], top_p=g["top_p"],
                max_tokens=self.cfg.max_tokens_override or g["max_tokens"], n=1,
                reasoning_effort=self.cfg.reasoning_effort,
            )
            samples.append(more[0].text)
            candidates.append(extract_code(more[0].text) or more[0].text)
            best_i, exec_res = verify_code.select_best(candidates, public)
            retries += 1
        selected_text = samples[best_i] if best_i < len(samples) else samples[0]
        if public and exec_res and exec_res.passed:
            self._maybe_cache_add(problem, selected_text,
                                  candidates[best_i] if candidates else None, "code_exec")
        return Solution(
            problem_id=problem.id, domain=domain, samples=samples,
            presentation=present(domain, selected_text),
            internal_selection=candidates[best_i] if candidates else None,
        )
