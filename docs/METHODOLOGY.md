# Methodology

This document is the public reference for the `§N` markers found throughout
the codebase. They index the sections of the internal working plan; everything
needed to understand, audit or reproduce the experiments is summarized here
under the same numbering.

## §0 — Goal and gate

Question under test: can a small open model plus a verification-first layer,
on a 16 GB consumer machine, reach the quality of far larger models on
verifiable STEM/coding tasks? The **gate** is evaluated on a *verifiable core*
(MATH-500 + LiveCodeBench) against explicit thresholds (§12) — and reported
whichever way it lands.

## §3 / §5 — Hardware tiers and models

- **GPU tier**: single 24 GB GPU (RTX 4090 class) — used for training and for
  fast, exact (bf16) evaluation of the same configs that ship to consumers.
- **Consumer tier**: 16 GB machine, CPU or small GPU, llama.cpp, 4-bit
  (Q4_K_M). Coexistence budget: the system targets ≤8-10 GB so the machine
  remains usable for other work.
- Base model (reference runs): `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
  (MIT). The layer is base-agnostic; transfer runs are pre-registered in
  [`experiments/A2_preregistration.md`](../experiments/A2_preregistration.md).

## §4 — Data and decontamination

Specialist training data derives from open datasets (Mixture-of-Thoughts
math/code/science). Before any training: **exact-hash + n-gram (n=6, Jaccard
0.6) decontamination** against every evaluation benchmark used anywhere in
this repo (`data/decontaminate.py`); removed counts are logged. Traces longer
than the training context are dropped whole, never truncated
(`data/filter_length.py`).

## §6 — Specialist training

QLoRA per domain on the shared base (r=64, α=128, dropout 0.05, all linear
targets, seq 16384, packing, bf16, cosine LR 1e-4, 1 epoch; see
`config/lora_*.yaml`). Note: measured post-hoc, specialists contributed ~0
net on the evaluated benchmarks — published as a finding (§15); the layer's
value does not depend on them.

## §7 — The layer (router, harness, verifiers)

- **Router** (§7.1): deterministic code heuristics first (v1.2 includes
  competitive-programming markers; measured: 0/100 misrouted on
  LiveCodeBench, vs 59/100 for the v1.1 classifier alone), then an embedding
  classifier (bge-m3 + logistic head), CPU-only.
- **Serving** (§7.2): vLLM multi-LoRA (GPU tier) or llama.cpp (consumer
  tier); adapters hot-swap per request.
- **Verification & selection** (§7.3): *hard verifiers hold final authority.*
  Math: self-consistency (majority vote) over N samples, answers checked with
  Math-Verify. Code: candidates executed in a sandbox against **public tests
  only** (hidden tests are reserved for scoring); where no public tests
  exist, CodeT-style self-generated asserts with dual-execution agreement.
  **Agentic repair loop** (code): small initial burst → run frozen tests →
  feed the concrete failure back → retry; budget capped at the best-of-N
  equivalent. Tests are frozen before the loop and never edited by it.
- **Presenter** (§7.4): the selected answer ships with its evidence (tests
  run, pass/fail, logs).

## §8 — Benchmarks and scoring

MATH-500, GSM8K (Math-Verify equivalence), HumanEval, MBPP (hidden-test
execution in sandbox), LiveCodeBench (own I/O harness — stdin and call-based;
**post-cutoff window: problems dated ≥ 2024-08-01**), AIME 2024, MMLU-STEM
(letter-extraction with majority vote). 100 problems per benchmark unless
stated. Metrics: `pass@1` (first sample) and `final` (post-selection).
Statistical note: n=100 → ±3-4 points per cell.

## §9 — Ceilings

Same problems, same scoring, via API: DeepSeek R1 671B (the base's teacher)
and DeepSeek V4 Flash (frontier reference), with a 16k output budget
(measured: 8k truncates long reasoning and understates the ceiling — we
report ceilings at their best).

## §10 — Baseline isolation (anti-fooling protocol)

The verifier helps *any* model, so every contribution is isolated:
bare base (1) · +specialist only (2) · +verifier only (3) · +repair (3b) ·
full system (4, 4b) · ceilings plain (5, 5r1) · **ceilings with our own
verifier (6, 6r1)** — the architecture must beat its pieces and survive an
apples-to-apples ceiling, not just beat the bare base.

## §12 — Gate criteria

Core: full system ≥ ceiling single-shot on the verifiable core; ≥ +5 vs bare
base and vs specialist-only; positive vs verifier-only; hardware within the
consumer envelope. Verdict published with the tables
(`report/gate_decision.md`).

## §15 — Honesty rules

Decontaminated training data (§4); post-cutoff LiveCodeBench; internal
selection on public tests only; generated code runs only in a sandbox
(subprocess rlimit+timeout; Docker hardened mode available); negative results
are published (two PRM selectors rejected; specialists ~neutral); every
number in the tables has its raw per-problem JSON in `results/runs/`.
