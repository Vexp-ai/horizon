# Gate decision (§12)

> Draft auto-generated from the runs. Complete with qualitative analysis (§Phase 6).

## Verifiable core: 4 vs 5 (parity with the single-shot ceiling)

- **math500**: system=92.0 vs DeepSeek=98.0 → 94% ❌ <100%
- **livecodebench**: system=39.0 vs DeepSeek=69.0 → 57% ❌ <100%

## Architecture merit: 4 vs 1/2/3

- **aime2024** ⚠️: vs base(1)=+10.0, vs specialist(2)=n/a, vs verifier(3)=n/a  (threshold ≥ +5 over 1 and 2)
- **gsm8k** ⚠️: vs base(1)=+10.0, vs specialist(2)=+4.0, vs verifier(3)=+5.0  (threshold ≥ +5 over 1 and 2)
- **humaneval** ✅: vs base(1)=+20.0, vs specialist(2)=+25.0, vs verifier(3)=+1.0  (threshold ≥ +5 over 1 and 2)
- **livecodebench** ⚠️: vs base(1)=+14.0, vs specialist(2)=n/a, vs verifier(3)=+1.0  (threshold ≥ +5 over 1 and 2)
- **math500** ✅: vs base(1)=+7.0, vs specialist(2)=+5.0, vs verifier(3)=+-1.0  (threshold ≥ +5 over 1 and 2)
- **mbpp** ✅: vs base(1)=+20.0, vs specialist(2)=+21.0, vs verifier(3)=+0.0  (threshold ≥ +5 over 1 and 2)
- **mmlu_stem** ⚠️: vs base(1)=+0.7, vs specialist(2)=n/a, vs verifier(3)=n/a  (threshold ≥ +5 over 1 and 2)

## Honesty: 4 vs 6 (DeepSeek with verification as well)

- **gsm8k**: system=95.0 vs DeepSeek+boN=96.0 (Δ=-1.0)
- **humaneval**: system=99.0 vs DeepSeek+boN=97.0 (Δ=+2.0)
- **math500**: system=92.0 vs DeepSeek+boN=98.0 (Δ=-6.0)
- **mbpp**: system=92.0 vs DeepSeek+boN=96.0 (Δ=-4.0)

## The father: 4 vs R1 (does the system on consumer hardware beat the teacher?)

- **aime2024**: system=53.3 vs R1=66.7 (Δ=-13.3) ❌ below
- **gsm8k**: system=95.0 vs R1=96.0 (Δ=-1.0) ❌ below
- **humaneval**: system=99.0 vs R1=94.0 (Δ=+5.0) ✅ BEATS
- **livecodebench**: system=39.0 vs R1=46.0 (Δ=-7.0) ❌ below
- **math500**: system=92.0 vs R1=97.0 (Δ=-5.0) ❌ below
- **mbpp**: system=92.0 vs R1=93.0 (Δ=-1.0) ❌ below

## Verdict (auto, §12)

- Core NOT at parity on all benchmarks → **Iterate / Rethink**: analyze the bottleneck (router? specialists? N? quant?).

*(The final decision requires the hardware table §Phase 5 and the human judgment §Phase 6.)*
